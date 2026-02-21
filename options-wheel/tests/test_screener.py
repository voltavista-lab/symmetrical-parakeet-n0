"""Tests for screener modules (stock_filter and options_scanner)."""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screener.stock_filter import get_stock_fundamentals, filter_watchlist
from screener.options_scanner import (
    _estimate_delta,
    get_options_chain,
    calculate_iv_rank,
    _compute_composite_score,
    scan_for_candidates,
)


# --- Mock yfinance data ---

MOCK_AAPL_INFO = {
    "currentPrice": 175.0,
    "dividendYield": 0.005,  # 0.5%
    "marketCap": 2_800_000_000_000,
    "sector": "Technology",
    "shortName": "Apple Inc.",
    "beta": 1.2,
    "fiftyTwoWeekHigh": 200.0,
    "fiftyTwoWeekLow": 140.0,
}

MOCK_T_INFO = {
    "currentPrice": 18.0,
    "dividendYield": 0.065,  # 6.5%
    "marketCap": 130_000_000_000,
    "sector": "Communication Services",
    "shortName": "AT&T Inc.",
    "beta": 0.7,
    "fiftyTwoWeekHigh": 22.0,
    "fiftyTwoWeekLow": 14.0,
}

MOCK_KO_INFO = {
    "currentPrice": 60.0,
    "dividendYield": 0.03,  # 3.0%
    "marketCap": 260_000_000_000,
    "sector": "Consumer Defensive",
    "shortName": "The Coca-Cola Company",
    "beta": 0.6,
    "fiftyTwoWeekHigh": 65.0,
    "fiftyTwoWeekLow": 52.0,
}

MOCK_EXPENSIVE_INFO = {
    "currentPrice": 450.0,
    "dividendYield": 0.02,
    "marketCap": 500_000_000_000,
    "sector": "Technology",
    "shortName": "Expensive Corp",
    "beta": 1.5,
    "fiftyTwoWeekHigh": 500.0,
    "fiftyTwoWeekLow": 350.0,
}


def _make_mock_ticker(info_dict):
    mock = MagicMock()
    mock.info = info_dict
    return mock


# --- Tests for get_stock_fundamentals ---


@patch("screener.stock_filter.yf.Ticker")
def test_get_fundamentals_success(mock_ticker_cls):
    mock_ticker_cls.return_value = _make_mock_ticker(MOCK_T_INFO)
    result = get_stock_fundamentals("T")
    assert result is not None
    assert result["ticker"] == "T"
    assert result["price"] == 18.0
    assert abs(result["dividend_yield"] - 6.5) < 0.01
    assert result["sector"] == "Communication Services"


@patch("screener.stock_filter.yf.Ticker")
def test_get_fundamentals_no_price(mock_ticker_cls):
    mock_ticker_cls.return_value = _make_mock_ticker({"sector": "Tech"})
    result = get_stock_fundamentals("BAD")
    assert result is None


@patch("screener.stock_filter.yf.Ticker")
def test_get_fundamentals_exception(mock_ticker_cls):
    mock_ticker_cls.side_effect = Exception("API error")
    result = get_stock_fundamentals("ERR")
    assert result is None


@patch("screener.stock_filter.yf.Ticker")
def test_get_fundamentals_no_dividend(mock_ticker_cls):
    info = {**MOCK_AAPL_INFO, "dividendYield": None}
    mock_ticker_cls.return_value = _make_mock_ticker(info)
    result = get_stock_fundamentals("AAPL")
    assert result is not None
    assert result["dividend_yield"] == 0.0


# --- Tests for filter_watchlist ---


@patch("screener.stock_filter.get_stock_fundamentals")
def test_filter_watchlist_basic(mock_get):
    def side_effect(ticker):
        data = {"AAPL": MOCK_AAPL_INFO, "T": MOCK_T_INFO, "KO": MOCK_KO_INFO}
        info = data.get(ticker)
        if info is None:
            return None
        price = info["currentPrice"]
        div_yield = (info.get("dividendYield") or 0) * 100
        return {
            "ticker": ticker,
            "price": price,
            "dividend_yield": div_yield,
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "name": info.get("shortName", ticker),
            "beta": info.get("beta", 1.0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", price),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", price),
        }

    mock_get.side_effect = side_effect
    results = filter_watchlist(["AAPL", "T", "KO"], min_div_yield=1.0, max_price=200, delay=0)

    tickers = [r["ticker"] for r in results]
    # AAPL has 0.5% yield -> excluded; T at 6.5% and KO at 3.0% -> included
    assert "AAPL" not in tickers
    assert "T" in tickers
    assert "KO" in tickers


@patch("screener.stock_filter.get_stock_fundamentals")
def test_filter_watchlist_price_filter(mock_get):
    def side_effect(ticker):
        info = MOCK_EXPENSIVE_INFO
        price = info["currentPrice"]
        div_yield = (info.get("dividendYield") or 0) * 100
        return {
            "ticker": ticker,
            "price": price,
            "dividend_yield": div_yield,
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "name": info.get("shortName", ticker),
            "beta": info.get("beta", 1.0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", price),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", price),
        }

    mock_get.side_effect = side_effect
    results = filter_watchlist(["EXP"], min_div_yield=1.0, max_price=200, delay=0)
    assert len(results) == 0


@patch("screener.stock_filter.get_stock_fundamentals")
def test_filter_watchlist_empty_input(mock_get):
    results = filter_watchlist([], delay=0)
    assert results == []
    mock_get.assert_not_called()


# --- Tests for options_scanner ---


def test_estimate_delta_call_atm():
    """ATM call should have delta ~0.5."""
    delta = _estimate_delta("call", 100, 100, 45, 0.30)
    assert 0.45 < delta < 0.65


def test_estimate_delta_put_atm():
    """ATM put should have delta ~-0.5."""
    delta = _estimate_delta("put", 100, 100, 45, 0.30)
    assert -0.65 < delta < -0.45


def test_estimate_delta_otm_put():
    """OTM put (strike < price) should have delta closer to 0."""
    delta = _estimate_delta("put", 100, 85, 45, 0.25)
    assert -0.40 < delta < 0


def test_estimate_delta_zero_dte():
    delta = _estimate_delta("call", 100, 100, 0, 0.3)
    assert delta == 0.0


def test_estimate_delta_zero_iv():
    delta = _estimate_delta("call", 100, 100, 45, 0)
    assert delta == 0.0


@patch("screener.options_scanner.yf.Ticker")
def test_get_options_chain_empty_expirations(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.options = []
    mock_ticker_cls.return_value = mock_ticker
    df = get_options_chain("TEST")
    assert df.empty


@patch("screener.options_scanner.yf.Ticker")
def test_calculate_iv_rank_normal(mock_ticker_cls):
    """Test IV rank with synthetic historical data."""
    mock_ticker = MagicMock()
    # Create synthetic price series with known volatility pattern
    np.random.seed(42)
    dates = pd.date_range(end="2024-06-01", periods=300, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
    hist_df = pd.DataFrame({"Close": prices}, index=dates)
    mock_ticker.history.return_value = hist_df
    mock_ticker_cls.return_value = mock_ticker

    rank = calculate_iv_rank("TEST")
    assert 0 <= rank <= 100


@patch("screener.options_scanner.yf.Ticker")
def test_calculate_iv_rank_insufficient_data(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker
    rank = calculate_iv_rank("TEST")
    assert rank == 50.0


def test_compute_composite_score():
    row = pd.Series({
        "iv_rank": 60,
        "annual_premium_yield": 25,
        "dividend_yield": 3.0,
        "openInterest": 500,
        "volume": 100,
    })
    score = _compute_composite_score(row)
    assert 0 < score <= 100


def test_compute_composite_score_zeros():
    row = pd.Series({
        "iv_rank": 0,
        "annual_premium_yield": 0,
        "dividend_yield": 0,
        "openInterest": 0,
        "volume": 0,
    })
    score = _compute_composite_score(row)
    assert score == 0.0


@patch("screener.options_scanner.calculate_iv_rank")
@patch("screener.options_scanner.get_options_chain")
def test_scan_for_candidates_filters(mock_chain, mock_iv):
    """Test that scan applies OI, volume, and delta filters."""
    mock_iv.return_value = 50.0
    mock_chain.return_value = pd.DataFrame([
        {"type": "put", "strike": 55.0, "expiry": "2024-07-19", "dte": 45,
         "bid": 1.5, "ask": 1.8, "mid_price": 1.65, "impliedVolatility": 0.30,
         "openInterest": 200, "volume": 100},
        {"type": "put", "strike": 55.0, "expiry": "2024-07-19", "dte": 45,
         "bid": 0.5, "ask": 0.6, "mid_price": 0.55, "impliedVolatility": 0.30,
         "openInterest": 5, "volume": 2},  # Low liquidity - should be filtered
    ])

    stocks = [{"ticker": "KO", "price": 60.0, "dividend_yield": 3.0}]
    params = {
        "dte_range": (30, 60),
        "min_open_interest": 100,
        "min_volume": 50,
        "delta_range_put": (-0.20, -0.40),
        "delta_range_call": (0.20, 0.40),
        "min_iv_rank": 30,
    }
    result = scan_for_candidates(stocks, params, delay=0)
    # Second row should be filtered out due to low OI/volume
    assert len(result) <= 1


@patch("screener.options_scanner.calculate_iv_rank")
@patch("screener.options_scanner.get_options_chain")
def test_scan_for_candidates_empty_chain(mock_chain, mock_iv):
    mock_iv.return_value = 50.0
    mock_chain.return_value = pd.DataFrame()
    stocks = [{"ticker": "KO", "price": 60.0, "dividend_yield": 3.0}]
    params = {
        "dte_range": (30, 60),
        "min_open_interest": 100,
        "min_volume": 50,
        "delta_range_put": (-0.20, -0.40),
        "delta_range_call": (0.20, 0.40),
        "min_iv_rank": 30,
    }
    result = scan_for_candidates(stocks, params, delay=0)
    assert result.empty
