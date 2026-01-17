"""
Temporal Fusion Transformer for Electricity Price Forecasting

This script trains a TFT model using PyTorch Forecasting for weekly electricity price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import argparse
import os

warnings.filterwarnings('ignore')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def prepare_tft_data(csv_path, sheet_name, date_column='ini_date'):
    """
    Prepare data for Temporal Fusion Transformer.

    Args:
        csv_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        date_column: Name of the date column

    Returns:
        DataFrame ready for TFT training
    """
    print(f"Loading data from {csv_path}, sheet: {sheet_name}")
    df = pd.read_excel(csv_path, sheet_name=sheet_name)

    # Convert date to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)

    # Create time index
    df['time_idx'] = range(len(df))

    # Create group identifier
    if 'submarket' in df.columns:
        df['group'] = df['submarket']
    else:
        df['group'] = sheet_name

    # Add temporal features
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter
    df['year'] = df[date_column].dt.year
    df['week_of_year'] = df[date_column].dt.isocalendar().week

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(f"Data prepared: {len(df)} samples from {df[date_column].min()} to {df[date_column].max()}")

    return df


def create_datasets(df, max_encoder_length=52, max_prediction_length=1,
                   training_cutoff_ratio=0.7, batch_size=64):
    """
    Create TimeSeriesDataSet for training and validation.
    """
    target = 'pld'

    # Define features
    time_varying_known_categoricals = ['month', 'quarter', 'week_of_year']
    time_varying_unknown_reals = [
        'load_energy', 'max_demand', 'ena', 'hidro_gen',
        'thermo_gen', 'stored_energy', 'exports', 'imports'
    ]
    static_categoricals = ['group']

    # Set training cutoff
    training_cutoff = int(len(df) * training_cutoff_ratio)
    print(f"\nSplitting data: {training_cutoff} training, {len(df) - training_cutoff} validation")

    # Create training dataset
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target=target,
        group_ids=['group'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=[],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals + [target],
        target_normalizer=GroupNormalizer(
            groups=['group'], transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False
    )

    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    print(f"Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    return training, validation, train_dataloader, val_dataloader


def train_model(train_dataloader, val_dataloader, training_dataset, max_epochs=100,
                learning_rate=0.03, hidden_size=64, attention_head_size=4):
    """
    Configure and train the TFT model.
    """
    print("\n" + "="*60)
    print("CONFIGURING TFT MODEL")
    print("="*60)

    # Configure callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
        enable_model_summary=True,
    )

    # Configure TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer="ranger",
    )

    print(f"Model size: {tft.size()/1e3:.1f}k parameters")

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("\nTraining completed!")

    return trainer, tft


def evaluate_model(trainer, val_dataloader, sheet_name):
    """
    Load best model and evaluate on validation set.
    """
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Make predictions
    print("Generating predictions...")
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader, mode="prediction", return_x=False)

    # Convert to numpy
    actuals_np = actuals.cpu().numpy().flatten()
    predictions_np = predictions.cpu().numpy().flatten()

    # Remove NaN values
    mask = ~(np.isnan(actuals_np) | np.isnan(predictions_np))
    actuals_clean = actuals_np[mask]
    predictions_clean = predictions_np[mask]

    # Calculate metrics
    rmse = sqrt(mean_squared_error(actuals_clean, predictions_clean))
    mae = np.mean(np.abs(actuals_clean - predictions_clean))
    mape = np.mean(np.abs((actuals_clean - predictions_clean) / actuals_clean)) * 100

    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS ({sheet_name.upper()})")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*60}\n")

    return best_tft, actuals_clean, predictions_clean, rmse, mae, mape


def plot_results(actuals, predictions, sheet_name):
    """
    Create visualization plots.
    """
    print("Creating visualizations...")

    # 1. Predictions vs Actuals
    plt.figure(figsize=(16, 6))
    plt.plot(actuals, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions, label='TFT Prediction', alpha=0.7, linewidth=2)
    plt.title(f'TFT Predictions vs Actuals - {sheet_name.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Price (PLD)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tft_predictions_{sheet_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: tft_predictions_{sheet_name}.png")

    # 2. Residuals
    residuals = actuals - predictions
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Residual')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'tft_residuals_{sheet_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: tft_residuals_{sheet_name}.png")

    # 3. Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5, s=20)
    plt.plot([actuals.min(), actuals.max()],
             [actuals.min(), actuals.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title(f'TFT: Predicted vs Actual - {sheet_name.upper()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tft_scatter_{sheet_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: tft_scatter_{sheet_name}.png")


def save_results(actuals, predictions, model, sheet_name):
    """
    Save predictions and model to files.
    """
    print("\nSaving results...")

    # Save predictions
    residuals = actuals - predictions
    results_df = pd.DataFrame({
        'actual': actuals,
        'tft_prediction': predictions,
        'residual': residuals
    })
    results_df.to_csv(f'tft_results_{sheet_name}.csv', index=False)
    print(f"  ✓ Saved: tft_results_{sheet_name}.csv")

    # Save model
    torch.save(model.state_dict(), f'tft_model_{sheet_name}.pt')
    print(f"  ✓ Saved: tft_model_{sheet_name}.pt")


def compare_with_lstm(sheet_name, tft_rmse):
    """
    Compare TFT results with LSTM baseline if available.
    """
    try:
        benchmark_df = pd.read_excel('benchmark_consolidated.xlsx', sheet_name=sheet_name)

        lstm_rmse = sqrt(mean_squared_error(benchmark_df['actuals'], benchmark_df['predictions']))
        decomp_rmse = sqrt(mean_squared_error(benchmark_df['actuals'], benchmark_df['decomp_predictions_as_of']))

        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON ({sheet_name.upper()})")
        print(f"{'='*60}")
        print(f"TFT RMSE:          {tft_rmse:.3f}")
        print(f"LSTM RMSE:         {lstm_rmse:.3f}")
        print(f"DECOMP RMSE:       {decomp_rmse:.3f}")
        print(f"\nTFT Improvement over LSTM:  {((lstm_rmse - tft_rmse) / lstm_rmse * 100):+.2f}%")
        print(f"TFT Improvement over DECOMP: {((decomp_rmse - tft_rmse) / decomp_rmse * 100):+.2f}%")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nNote: Could not load benchmark data for comparison: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train TFT model for electricity price forecasting')
    parser.add_argument('--data', type=str, default='FINAL_INPUTS_v2.xls',
                        help='Path to input data file')
    parser.add_argument('--sheet', type=str, default='southeast',
                        choices=['southeast', 'northeast', 'north', 'south'],
                        help='Sheet name (submarket)')
    parser.add_argument('--encoder-length', type=int, default=52,
                        help='Encoder length (lookback window in weeks)')
    parser.add_argument('--prediction-length', type=int, default=1,
                        help='Prediction length (forecast horizon in weeks)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--learning-rate', type=float, default=0.03,
                        help='Learning rate')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TEMPORAL FUSION TRANSFORMER TRAINING")
    print("="*60)
    print(f"Data file: {args.data}")
    print(f"Submarket: {args.sheet}")
    print(f"Encoder length: {args.encoder_length} weeks")
    print(f"Prediction length: {args.prediction_length} week(s)")
    print(f"Max epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60 + "\n")

    # Step 1: Prepare data
    df = prepare_tft_data(args.data, args.sheet)

    # Step 2: Create datasets
    training, validation, train_dataloader, val_dataloader = create_datasets(
        df,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
        batch_size=args.batch_size
    )

    # Step 3: Train model
    trainer, tft = train_model(
        train_dataloader,
        val_dataloader,
        training,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size
    )

    # Step 4: Evaluate
    best_tft, actuals, predictions, rmse, mae, mape = evaluate_model(
        trainer,
        val_dataloader,
        args.sheet
    )

    # Step 5: Visualize
    plot_results(actuals, predictions, args.sheet)

    # Step 6: Save results
    save_results(actuals, predictions, best_tft, args.sheet)

    # Step 7: Compare with baseline
    compare_with_lstm(args.sheet, rmse)

    print("\n" + "="*60)
    print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
