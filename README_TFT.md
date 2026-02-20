# Temporal Fusion Transformer for Electricity Price Forecasting

This repository contains a complete implementation of Temporal Fusion Transformers (TFT) using PyTorch Forecasting for weekly electricity price prediction in Brazilian submarkets.

## Overview

The TFT model is a state-of-the-art deep learning architecture designed specifically for time series forecasting. It combines:
- **Multi-head attention mechanisms** for learning temporal relationships
- **Variable selection networks** for identifying important features
- **Gating mechanisms** for controlling information flow
- **Quantile forecasting** for uncertainty estimation

## Features

- ✅ Complete data preprocessing pipeline
- ✅ TFT model with configurable hyperparameters
- ✅ Automatic feature engineering (temporal features)
- ✅ Training with early stopping and learning rate scheduling
- ✅ Comprehensive evaluation metrics (RMSE, MAE, MAPE)
- ✅ Multiple visualization types
- ✅ Feature importance analysis
- ✅ Comparison with LSTM baseline
- ✅ Uncertainty quantification

## Installation

### Option 1: Using pip

```bash
pip install -r requirements.txt
```

### Option 2: Manual installation

```bash
pip install torch>=2.0.0
pip install pytorch-forecasting>=1.0.0
pip install pytorch-lightning>=2.0.0
pip install pandas numpy matplotlib scikit-learn openpyxl xlrd
```

## Usage

### Option 1: Jupyter Notebook (Recommended for exploration)

Open and run `tft_implementation.ipynb` in Jupyter:

```bash
jupyter notebook tft_implementation.ipynb
```

The notebook provides:
- Step-by-step explanations
- Interactive visualizations
- Feature importance analysis
- Hyperparameter tuning guidance

### Option 2: Python Script (Recommended for production)

Run the training script from the command line:

```bash
# Train on Southeast submarket (default)
python tft_train.py

# Train on different submarket
python tft_train.py --sheet northeast

# Customize hyperparameters
python tft_train.py --sheet southeast --encoder-length 52 --epochs 150 --hidden-size 128

# See all options
python tft_train.py --help
```

#### Command-line Arguments

- `--data`: Path to input Excel file (default: `FINAL_INPUTS_v2.xls`)
- `--sheet`: Submarket to train on (`southeast`, `northeast`, `north`, `south`)
- `--encoder-length`: Lookback window in weeks (default: 52)
- `--prediction-length`: Forecast horizon in weeks (default: 1)
- `--epochs`: Maximum training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--hidden-size`: Hidden layer size (default: 64)
- `--learning-rate`: Learning rate (default: 0.03)

## Data Format

The expected data format is an Excel file with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `ini_date` | Week start date | datetime |
| `pld` | Price (target variable) | float |
| `load_energy` | Energy load | float |
| `max_demand` | Maximum demand | float |
| `ena` | Natural energy affluent | float |
| `hidro_gen` | Hydroelectric generation | float |
| `thermo_gen` | Thermoelectric generation | float |
| `stored_energy` | Stored energy in reservoirs | float |
| `exports` | Energy exports | float |
| `imports` | Energy imports | float |

## Model Architecture

The TFT model consists of:

1. **Input Layer**: Processes time-varying and static features
2. **Variable Selection Network**: Identifies relevant features
3. **LSTM Encoder-Decoder**: Captures temporal dependencies
4. **Multi-Head Attention**: Learns relationships between time steps
5. **Gating Layers**: Controls information flow
6. **Quantile Outputs**: Produces probabilistic forecasts

## Output Files

After training, the following files are generated:

### Predictions and Metrics
- `tft_results_{submarket}.csv`: Actual vs predicted values
- Model performance metrics printed to console

### Visualizations
- `tft_predictions_{submarket}.png`: Time series plot of predictions vs actuals
- `tft_residuals_{submarket}.png`: Residual analysis (time series + distribution)
- `tft_scatter_{submarket}.png`: Scatter plot of predicted vs actual
- `tft_interpretation_{submarket}.png`: Feature importance plot
- `tft_uncertainty_{submarket}_sample_*.png`: Uncertainty quantification plots

### Model Checkpoints
- `tft_model_{submarket}.pt`: Trained model weights
- `lightning_logs/`: TensorBoard logs for training monitoring

## Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

Then open http://localhost:6006 in your browser.

## Example Results

Typical performance on Southeast submarket:
- **RMSE**: ~80-100 (varies by period)
- **MAE**: ~60-80
- **MAPE**: ~15-25%

The TFT model generally shows improvements over traditional LSTM approaches, especially in:
- Long-term dependencies
- Handling multiple input features
- Uncertainty estimation

## Key Features Explained

### 1. Data Preparation
The pipeline automatically:
- Converts dates to datetime format
- Creates temporal features (month, quarter, week)
- Handles missing values
- Normalizes features appropriately

### 2. Feature Types
- **Time-varying known**: Features known in advance (e.g., calendar features)
- **Time-varying unknown**: Features only known up to present (e.g., load, generation)
- **Static**: Features constant over time (e.g., submarket identifier)

### 3. Training Strategy
- 70/30 train/validation split
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Gradient clipping for stability

### 4. Uncertainty Quantification
TFT provides prediction intervals through quantile regression, helping assess forecast reliability.

## Hyperparameter Tuning

Key hyperparameters to experiment with:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `hidden_size` | Size of hidden layers | 32-256 |
| `attention_head_size` | Number of attention heads | 1-8 |
| `encoder_length` | Lookback window | 12-104 weeks |
| `dropout` | Dropout rate | 0.1-0.3 |
| `learning_rate` | Initial learning rate | 0.001-0.1 |

You can use the built-in hyperparameter optimization (see notebook cell 9 alternative).

## Comparison with LSTM

The TFT implementation includes automatic comparison with LSTM baseline results if `benchmark_consolidated.xlsx` is available. Expected improvements:

- Better handling of long sequences
- Automatic feature selection
- Interpretable attention weights
- Probabilistic forecasting capabilities

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (e.g., to 32 or 16)
- Reduce `hidden_size` (e.g., to 32)
- Reduce `encoder_length` (e.g., to 26)

### Training Too Slow
- Use GPU if available (automatic)
- Reduce `max_encoder_length`
- Use smaller `hidden_size`

### Poor Performance
- Increase `encoder_length` to capture more history
- Increase `hidden_size` for more capacity
- Train for more `epochs`
- Adjust learning rate

### NaN in Predictions
- Check for missing values in data
- Reduce learning rate
- Increase gradient clipping

## Advanced Usage

### Multi-Step Forecasting

To forecast multiple weeks ahead, change `max_prediction_length`:

```python
MAX_PREDICTION_LENGTH = 4  # Forecast 4 weeks ahead
```

### Multiple Submarkets

Train on all submarkets in a loop:

```bash
for submarket in southeast northeast north south; do
    python tft_train.py --sheet $submarket
done
```

### Custom Features

Add additional features by modifying the feature lists in the code:

```python
time_varying_unknown_reals = [
    'load_energy', 'max_demand', 'ena',
    'your_custom_feature'  # Add here
]
```

## References

1. Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." International Journal of Forecasting.
2. PyTorch Forecasting Documentation: https://pytorch-forecasting.readthedocs.io/

## License

This implementation is for research and educational purposes.

## Contact

For questions or issues, please open an issue in the repository.
