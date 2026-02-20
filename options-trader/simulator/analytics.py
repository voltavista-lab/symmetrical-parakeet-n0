"""
Analytics, statistics, and visualizations for simulation results.

Produces summary statistics, trade logs, and charts for backtests.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Summary statistics ───────────────────────────────────────────────────────

def compute_stats(
    trades: list[dict],
    equity_curve: pd.Series,
    portfolio_size: float = 50_000.0,
) -> dict:
    """
    Compute summary statistics for a simulation run.

    Args:
        trades:         List of trade dicts (from TradeRecord.to_dict())
        equity_curve:   Daily cumulative P&L series (in dollars, 1 contract basis)
        portfolio_size: Starting capital for return calculations

    Returns:
        Dict of statistics.
    """
    if not trades:
        return _empty_stats()

    df = pd.DataFrame(trades)
    total_trades = len(df)
    if total_trades == 0:
        return _empty_stats()

    # Win/loss classification
    df["win"] = df["pnl"] > 0

    win_rate = df["win"].mean() * 100
    avg_win = df[df["win"]]["pnl"].mean() * 100 if df["win"].any() else 0.0
    avg_loss = df[~df["win"]]["pnl"].mean() * 100 if (~df["win"]).any() else 0.0

    total_pnl_dollars = equity_curve.iloc[-1] if not equity_curve.empty else 0.0
    annualized_return = _annualized_return(equity_curve, portfolio_size)
    max_drawdown = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio(equity_curve, portfolio_size)

    # Days in trade
    df["days"] = (pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])).dt.days
    avg_days = df["days"].mean()

    # Exit reasons
    exit_counts = df["exit_reason"].value_counts().to_dict()

    # Rolls
    num_rolls = int((df["exit_reason"] == "roll_triggered").sum())

    return {
        "num_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 1),
        "avg_loss_pct": round(avg_loss, 1),
        "total_pnl": round(total_pnl_dollars, 2),
        "annualized_return": round(annualized_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "avg_days_in_trade": round(avg_days, 1),
        "sharpe_ratio": round(sharpe, 3),
        "num_rolls": num_rolls,
        "exit_reasons": exit_counts,
    }


def _empty_stats() -> dict:
    return {
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "total_pnl": 0.0,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "avg_days_in_trade": 0.0,
        "sharpe_ratio": 0.0,
        "num_rolls": 0,
        "exit_reasons": {},
    }


def _annualized_return(equity_curve: pd.Series, portfolio_size: float) -> float:
    """Compute annualized return from equity curve."""
    if equity_curve.empty or portfolio_size <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / portfolio_size
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1 / 365.25)
    try:
        return ((1 + total_return) ** (1 / years) - 1) * 100
    except Exception:
        return 0.0


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from peak (in dollars)."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def _sharpe_ratio(equity_curve: pd.Series, portfolio_size: float, rf_annual: float = 0.045) -> float:
    """Approximate Sharpe ratio from daily P&L changes."""
    if equity_curve.empty or len(equity_curve) < 5:
        return 0.0
    daily_returns = equity_curve.diff().dropna() / portfolio_size
    if daily_returns.std() == 0:
        return 0.0
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    return float(excess.mean() / excess.std() * math.sqrt(252))


# ─── Display helpers ──────────────────────────────────────────────────────────

def format_trade_log(trades: list[dict]) -> pd.DataFrame:
    """Format trade list as a display-ready DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    display_cols = {
        "entry_date": "Entry",
        "exit_date": "Exit",
        "strike": "Strike",
        "dte_at_entry": "DTE",
        "credit_received": "Credit",
        "exit_price": "Exit Price",
        "pnl": "P&L/shr",
        "pnl_pct": "P&L %",
        "exit_reason": "Reason",
        "roll_number": "Roll #",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    out = df[list(available.keys())].rename(columns=available)

    if "Strike" in out.columns:
        out["Strike"] = out["Strike"].map("${:.2f}".format)
    if "Credit" in out.columns:
        out["Credit"] = out["Credit"].map("${:.2f}".format)
    if "Exit Price" in out.columns:
        out["Exit Price"] = out["Exit Price"].map("${:.2f}".format)
    if "P&L/shr" in out.columns:
        out["P&L/shr"] = out["P&L/shr"].map("${:.2f}".format)
    if "P&L %" in out.columns:
        out["P&L %"] = out["P&L %"].map("{:.1f}%".format)

    return out


def format_stats_table(stats: dict, ticker: str, strategy: str) -> pd.DataFrame:
    """Format summary stats as a two-column DataFrame for display."""
    rows = [
        ("Ticker", ticker),
        ("Strategy", strategy),
        ("Total Trades", str(stats.get("num_trades", 0))),
        ("Win Rate", f"{stats.get('win_rate', 0):.1f}%"),
        ("Avg Win", f"{stats.get('avg_win_pct', 0):.1f}% of credit"),
        ("Avg Loss", f"{stats.get('avg_loss_pct', 0):.1f}% of credit"),
        ("Total P&L", f"${stats.get('total_pnl', 0):,.2f}"),
        ("Annualized Return", f"{stats.get('annualized_return', 0):.2f}%"),
        ("Max Drawdown", f"${stats.get('max_drawdown', 0):,.2f}"),
        ("Avg Days in Trade", f"{stats.get('avg_days_in_trade', 0):.1f}"),
        ("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.3f}"),
        ("Number of Rolls", str(stats.get("num_rolls", 0))),
    ]
    exit_reasons = stats.get("exit_reasons", {})
    for reason, count in exit_reasons.items():
        rows.append((f"Exit: {reason}", str(count)))

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_equity_curve(
    equity_curve: pd.Series,
    ticker: str,
    strategy: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot the cumulative P&L equity curve."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color="#2196F3")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.fill_between(
            equity_curve.index,
            equity_curve.values,
            0,
            where=(equity_curve.values >= 0),
            alpha=0.2,
            color="#4CAF50",
        )
        ax.fill_between(
            equity_curve.index,
            equity_curve.values,
            0,
            where=(equity_curve.values < 0),
            alpha=0.2,
            color="#F44336",
        )
        ax.set_title(f"{ticker} — {strategy.replace('_', ' ').title()} Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Equity curve saved to %s", save_path)
        if show:
            plt.show()
        plt.close()
    except Exception as exc:
        logger.warning("Could not plot equity curve: %s", exc)


def plot_pnl_distribution(
    trades: list[dict],
    ticker: str,
    strategy: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot histogram of P&L per trade."""
    try:
        import matplotlib.pyplot as plt

        if not trades:
            return

        df = pd.DataFrame(trades)
        pnl_vals = df["pnl_pct"].values

        fig, ax = plt.subplots(figsize=(10, 4))
        wins = [v for v in pnl_vals if v >= 0]
        losses = [v for v in pnl_vals if v < 0]

        bins = np.linspace(min(pnl_vals) * 1.1, max(pnl_vals) * 1.1, 30)
        ax.hist(wins, bins=bins, color="#4CAF50", alpha=0.7, label=f"Wins ({len(wins)})")
        ax.hist(losses, bins=bins, color="#F44336", alpha=0.7, label=f"Losses ({len(losses)})")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{ticker} — {strategy.replace('_', ' ').title()} P&L Distribution")
        ax.set_xlabel("P&L (% of credit received)")
        ax.set_ylabel("Number of Trades")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("P&L distribution saved to %s", save_path)
        if show:
            plt.show()
        plt.close()
    except Exception as exc:
        logger.warning("Could not plot P&L distribution: %s", exc)


def compare_parameter_sets(results: list[dict]) -> pd.DataFrame:
    """
    Display a comparison table of multiple simulation runs.

    Args:
        results: List of dicts, each containing 'label', 'stats', and optionally
                 'profit_target_pct', 'stop_loss_pct', 'roll_trigger_pct'

    Returns:
        DataFrame for tabular display.
    """
    rows = []
    for r in results:
        stats = r.get("stats", {})
        rows.append(
            {
                "Label": r.get("label", ""),
                "Profit Tgt %": r.get("profit_target_pct", "-"),
                "Stop Loss %": r.get("stop_loss_pct", "-"),
                "Roll Trig %": r.get("roll_trigger_pct", "-"),
                "Trades": stats.get("num_trades", 0),
                "Win Rate": f"{stats.get('win_rate', 0):.1f}%",
                "Total P&L": f"${stats.get('total_pnl', 0):,.0f}",
                "Ann. Return": f"{stats.get('annualized_return', 0):.2f}%",
                "Drawdown": f"${stats.get('max_drawdown', 0):,.0f}",
                "Sharpe": f"{stats.get('sharpe_ratio', 0):.3f}",
                "Rolls": stats.get("num_rolls", 0),
            }
        )
    return pd.DataFrame(rows)
