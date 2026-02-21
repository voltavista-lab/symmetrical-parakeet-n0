"""Tests for config module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import WATCHLIST, STRATEGY


def test_watchlist_not_empty():
    assert len(WATCHLIST) > 0


def test_watchlist_contains_strings():
    for ticker in WATCHLIST:
        assert isinstance(ticker, str)
        assert len(ticker) >= 1


def test_watchlist_no_duplicates():
    assert len(WATCHLIST) == len(set(WATCHLIST))


def test_strategy_has_required_keys():
    required = [
        "target_dte", "dte_range", "profit_target_pct", "max_loss_close_pct",
        "roll_trigger_pct", "min_iv_rank", "min_dividend_yield",
        "max_stock_price", "min_open_interest", "min_volume",
        "delta_target_put", "delta_range_put",
        "delta_target_call", "delta_range_call",
    ]
    for key in required:
        assert key in STRATEGY, f"Missing key: {key}"


def test_strategy_dte_range_valid():
    low, high = STRATEGY["dte_range"]
    assert low < high
    assert low > 0
    assert STRATEGY["target_dte"] >= low
    assert STRATEGY["target_dte"] <= high


def test_strategy_delta_ranges_valid():
    put_low, put_high = STRATEGY["delta_range_put"]
    assert put_low < 0 and put_high < 0
    # put_low is closer to 0 (less negative), put_high is more negative
    assert put_low > put_high

    call_low, call_high = STRATEGY["delta_range_call"]
    assert call_low > 0 and call_high > 0
    assert call_low < call_high


def test_strategy_percentages_in_range():
    assert 0 < STRATEGY["profit_target_pct"] <= 100
    assert 0 < STRATEGY["max_loss_close_pct"] <= 100
    assert 0 < STRATEGY["roll_trigger_pct"] <= 100
    assert 0 <= STRATEGY["min_iv_rank"] <= 100


def test_strategy_numeric_values_positive():
    assert STRATEGY["target_dte"] > 0
    assert STRATEGY["min_dividend_yield"] >= 0
    assert STRATEGY["max_stock_price"] > 0
    assert STRATEGY["min_open_interest"] >= 0
    assert STRATEGY["min_volume"] >= 0
