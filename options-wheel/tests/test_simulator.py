"""Tests for simulator modules."""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.backtest import _estimate_premium, _compute_hv, simulate_wheel
from simulator.report import generate_report


# --- Tests for helper functions ---

def test_estimate_premium_put():
    premium = _estimate_premium(100, 95, 45, 0.25, "put")
    assert premium > 0
    assert premium < 10  # OTM put shouldn't be too expensive


def test_estimate_premium_call():
    premium = _estimate_premium(100, 105, 45, 0.25, "call")
    assert premium > 0
    assert premium < 10


def test_estimate_premium_atm():
    put = _estimate_premium(100, 100, 45, 0.30, "put")
    call = _estimate_premium(100, 100, 45, 0.30, "call")
    # ATM options should have meaningful premium
    assert put > 1
    assert call > 1


def test_estimate_premium_zero_dte():
    assert _estimate_premium(100, 95, 0, 0.25, "put") == 0.0


def test_estimate_premium_zero_iv():
    assert _estimate_premium(100, 95, 45, 0, "put") == 0.0


def test_compute_hv_normal():
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 60))))
    hv = _compute_hv(prices)
    assert 0.05 <= hv <= 1.0  # reasonable annualized vol


def test_compute_hv_short_series():
    prices = pd.Series([100, 101, 102])
    hv = _compute_hv(prices)
    assert hv == 0.25  # fallback


# --- Tests for simulate_wheel ---

@patch("simulator.backtest.yf.Ticker")
def test_simulate_wheel_basic(mock_ticker_cls):
    """Test simulation with synthetic trending price data."""
    mock_ticker = MagicMock()
    np.random.seed(123)
    n_days = 250
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    # Slightly upward trending prices
    returns = np.random.normal(0.0003, 0.015, n_days)
    prices = 50 * np.exp(np.cumsum(returns))
    hist_df = pd.DataFrame({"Close": prices}, index=dates)
    mock_ticker.history.return_value = hist_df
    mock_ticker_cls.return_value = mock_ticker

    params = {
        "target_dte": 45,
        "dte_range": (30, 60),
        "profit_target_pct": 50,
        "max_loss_close_pct": 40,
        "roll_trigger_pct": 60,
        "min_iv_rank": 30,
        "min_dividend_yield": 1.0,
        "max_stock_price": 200,
        "min_open_interest": 100,
        "min_volume": 50,
        "delta_target_put": -0.30,
        "delta_range_put": (-0.20, -0.40),
        "delta_target_call": 0.30,
        "delta_range_call": (0.20, 0.40),
    }

    results = simulate_wheel("TEST", "2023-01-01", "2023-12-31", params)
    assert results["ticker"] == "TEST"
    assert results["total_trades"] > 0
    assert "wins" in results
    assert "losses" in results
    assert "total_pnl" in results
    assert "monthly_pnl" in results
    assert results["win_rate"] >= 0


@patch("simulator.backtest.yf.Ticker")
def test_simulate_wheel_empty_data(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    params = {
        "target_dte": 45, "profit_target_pct": 50, "max_loss_close_pct": 40,
        "roll_trigger_pct": 60, "delta_target_put": -0.30, "delta_target_call": 0.30,
    }
    results = simulate_wheel("BAD", "2023-01-01", "2023-12-31", params)
    assert results["total_trades"] == 0


# --- Tests for report generation ---

def test_generate_report_with_trades():
    results = {
        "ticker": "TEST", "start_date": "2023-01-01", "end_date": "2023-12-31",
        "trades": [
            {"close_reason": "profit_target", "pnl": 50},
            {"close_reason": "stop_loss", "pnl": -30},
        ],
        "total_trades": 2, "wins": 1, "losses": 1,
        "win_rate": 50.0, "total_premium_collected": 500.0,
        "total_pnl": 20.0, "avg_pnl": 10.0, "max_drawdown": 30.0,
        "assignments": 0, "buy_hold_return": 15.0,
        "strategy_return": 12.0, "sharpe_ratio": 0.8,
        "monthly_pnl": {"2023-06": 50.0, "2023-09": -30.0},
    }
    report = generate_report(results)
    assert "TEST" in report
    assert "Win rate" in report
    assert "50.0%" in report
    assert "MONTHLY P&L" in report
    assert "profit_target" in report


def test_generate_report_no_trades():
    results = {
        "ticker": "EMPTY", "start_date": "2023-01-01", "end_date": "2023-06-01",
        "trades": [], "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_premium_collected": 0, "total_pnl": 0,
        "avg_pnl": 0, "max_drawdown": 0, "assignments": 0,
        "buy_hold_return": 0, "strategy_return": 0, "sharpe_ratio": 0,
        "monthly_pnl": {},
    }
    report = generate_report(results)
    assert "EMPTY" in report
    assert "No trades" in report
