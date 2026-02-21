"""Tests for strategy modules."""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy.short_put import score_short_put, select_best_put, capital_required_put
from strategy.covered_call import score_covered_call, select_best_call, capital_required_call
from strategy.position_mgmt import (
    check_profit_target,
    check_stop_loss,
    check_roll_trigger,
    calculate_roll,
    position_summary,
)


# --- Short Put Tests ---

def test_score_short_put_basic():
    option = {
        "mid_price": 2.0, "strike": 55.0, "dte": 45, "delta": -0.30,
        "implied_vol": 0.30, "iv_rank": 60, "annual_premium_yield": 25.0,
        "openInterest": 500, "volume": 100,
    }
    stock = {"price": 60.0, "dividend_yield": 3.0}
    score, reason = score_short_put(option, stock)
    assert 0 < score <= 100
    assert "APY=" in reason
    assert "delta=" in reason


def test_score_short_put_zero_values():
    option = {
        "mid_price": 0, "strike": 0, "dte": 0, "delta": 0,
        "implied_vol": 0, "iv_rank": 0, "annual_premium_yield": 0,
        "openInterest": 0, "volume": 0,
    }
    stock = {"price": 0, "dividend_yield": 0}
    score, _ = score_short_put(option, stock)
    assert score >= 0


def test_select_best_put():
    df = pd.DataFrame([
        {"option_type": "put", "ticker": "A", "composite_score": 70},
        {"option_type": "put", "ticker": "B", "composite_score": 85},
        {"option_type": "call", "ticker": "C", "composite_score": 90},
    ])
    result = select_best_put(df)
    assert result is not None
    assert result["ticker"] == "B"


def test_select_best_put_empty():
    result = select_best_put(pd.DataFrame())
    assert result is None


def test_select_best_put_no_puts():
    df = pd.DataFrame([
        {"option_type": "call", "ticker": "C", "composite_score": 90},
    ])
    result = select_best_put(df)
    assert result is None


def test_capital_required_put():
    assert capital_required_put(55.0) == 5500.0


# --- Covered Call Tests ---

def test_score_covered_call_basic():
    option = {
        "mid_price": 3.0, "strike": 65.0, "dte": 45, "delta": 0.30,
        "implied_vol": 0.28, "iv_rank": 55, "annual_premium_yield": 30.0,
        "openInterest": 300, "volume": 80,
    }
    stock = {"price": 60.0, "dividend_yield": 2.5}
    score, reason = score_covered_call(option, stock)
    assert 0 < score <= 100
    assert "upside=" in reason


def test_score_covered_call_itm():
    """ITM call (strike < price) should score lower on upside."""
    option = {
        "mid_price": 5.0, "strike": 55.0, "dte": 45, "delta": 0.60,
        "implied_vol": 0.30, "iv_rank": 50, "annual_premium_yield": 35.0,
        "openInterest": 200, "volume": 60,
    }
    stock = {"price": 60.0, "dividend_yield": 2.0}
    score, _ = score_covered_call(option, stock)
    assert score >= 0


def test_select_best_call():
    df = pd.DataFrame([
        {"option_type": "call", "ticker": "A", "composite_score": 60},
        {"option_type": "call", "ticker": "B", "composite_score": 80},
        {"option_type": "put", "ticker": "C", "composite_score": 95},
    ])
    result = select_best_call(df)
    assert result is not None
    assert result["ticker"] == "B"


def test_select_best_call_empty():
    result = select_best_call(pd.DataFrame())
    assert result is None


def test_capital_required_call():
    assert capital_required_call(60.0) == 6000.0


# --- Position Management Tests ---

def test_check_profit_target_met():
    assert check_profit_target(2.0, 0.90, 50) is True  # 55% profit


def test_check_profit_target_not_met():
    assert check_profit_target(2.0, 1.50, 50) is False  # 25% profit


def test_check_profit_target_zero_entry():
    assert check_profit_target(0, 1.0, 50) is False


def test_check_stop_loss_triggered():
    assert check_stop_loss(2.0, 2.90, 40) is True  # 45% loss


def test_check_stop_loss_not_triggered():
    assert check_stop_loss(2.0, 2.50, 40) is False  # 25% loss


def test_check_roll_trigger_yes():
    assert check_roll_trigger(2.0, 3.30, 60) is True  # 65% loss


def test_check_roll_trigger_no():
    assert check_roll_trigger(2.0, 2.80, 60) is False  # 40% loss


def test_calculate_roll():
    current = {"mid_price": 3.0, "strike": 55.0, "dte": 15, "implied_vol": 0.30}
    result = calculate_roll(current, 45)
    assert "close_cost" in result
    assert "new_premium" in result
    assert "net_credit" in result
    assert result["new_premium"] > result["close_cost"]  # longer DTE => more premium


def test_calculate_roll_zero_dte():
    current = {"mid_price": 3.0, "strike": 55.0, "dte": 0, "implied_vol": 0.30}
    result = calculate_roll(current, 45)
    assert result["new_premium"] == 3.0  # falls back to current mid


def test_position_summary_basic():
    positions = [
        {"ticker": "T", "option_type": "put", "entry_price": 1.50, "current_price": 0.70, "quantity": 1},
        {"ticker": "KO", "option_type": "put", "entry_price": 2.00, "current_price": 2.80, "quantity": 1},
    ]
    summary = position_summary(positions)
    assert summary["total_positions"] == 2
    assert summary["win_count"] == 1
    assert summary["loss_count"] == 1
    assert summary["total_premium_collected"] == (150 + 200)
    # T PnL: (1.50 - 0.70) * 100 = 80, KO PnL: (2.00 - 2.80) * 100 = -80
    assert summary["total_pnl"] == 0


def test_position_summary_empty():
    summary = position_summary([])
    assert summary["total_positions"] == 0
    assert summary["total_pnl"] == 0


def test_position_summary_all_winners():
    positions = [
        {"ticker": "A", "option_type": "put", "entry_price": 2.0, "current_price": 0.5},
        {"ticker": "B", "option_type": "call", "entry_price": 3.0, "current_price": 1.0},
    ]
    summary = position_summary(positions)
    assert summary["win_count"] == 2
    assert summary["loss_count"] == 0
    assert summary["total_pnl"] > 0
