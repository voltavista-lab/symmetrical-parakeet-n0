"""
data_fetcher.py - Market data retrieval via yfinance
Fetches stock prices, options chains, dividends, and IV estimates.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Sector mapping (static fallback when yfinance sector info is unavailable)
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    "PFE":  "Healthcare",
    "T":    "Communication Services",
    "BCE":  "Communication Services",
    "BNS":  "Financials",
    "VALE": "Materials",
    "CLF":  "Materials",
}

# CAD tickers traded on TSX (used for currency tagging)
CAD_TICKERS: set[str] = {"BCE", "BNS"}


def _safe_get(info: dict, *keys: str, default: Any = None) -> Any:
    """Return the first key found in info dict, else default."""
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return default


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_price(ticker: str) -> dict:
    """Return basic price + fundamental data for one ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        price = _safe_get(info, "currentPrice", "regularMarketPrice",
                          "navPrice", default=None)

        # Fallback: grab last close from history
        if price is None:
            hist = tk.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        div_yield = _safe_get(info, "dividendYield", "trailingAnnualDividendYield",
                              default=0.0)
        if div_yield and div_yield > 1:       # yfinance sometimes returns 0-100
            div_yield /= 100

        beta = _safe_get(info, "beta", default=None)
        sector = _safe_get(info, "sector", default=SECTOR_MAP.get(ticker, "Unknown"))
        currency = "CAD" if ticker in CAD_TICKERS else "USD"

        return {
            "ticker":       ticker,
            "price":        round(float(price), 4) if price else None,
            "div_yield":    round(float(div_yield) * 100, 2) if div_yield else 0.0,
            "beta":         round(float(beta), 2) if beta else None,
            "sector":       sector,
            "currency":     currency,
            "market_cap":   _safe_get(info, "marketCap", default=None),
            "fifty_two_wk_low":  _safe_get(info, "fiftyTwoWeekLow", default=None),
            "fifty_two_wk_high": _safe_get(info, "fiftyTwoWeekHigh", default=None),
            "fetched_at":   datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        return {
            "ticker":   ticker,
            "price":    None,
            "error":    str(exc),
            "currency": "CAD" if ticker in CAD_TICKERS else "USD",
            "sector":   SECTOR_MAP.get(ticker, "Unknown"),
            "fetched_at": datetime.utcnow().isoformat(),
        }


def fetch_prices(tickers: list[str]) -> dict[str, dict]:
    """Batch fetch prices for a list of tickers."""
    results: dict[str, dict] = {}
    for ticker in tickers:
        results[ticker] = fetch_price(ticker)
        time.sleep(0.15)          # polite delay between requests
    return results


# ---------------------------------------------------------------------------
# Options chain helpers
# ---------------------------------------------------------------------------

def _nearest_expiry(expirations: tuple[str, ...], target_dte: int) -> str | None:
    """Return the expiry date string closest to target_dte from today."""
    today = datetime.today().date()
    target = today + timedelta(days=target_dte)
    best: str | None = None
    best_diff = float("inf")
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        diff = abs((exp_date - target).days)
        if diff < best_diff:
            best_diff = diff
            best = exp
    return best


def fetch_options_chain(ticker: str, dte_targets: list[int] | None = None
                        ) -> dict[str, Any]:
    """
    Fetch options chains for a ticker at multiple DTE targets.

    Returns:
        {
          "ticker": str,
          "price": float,
          "expirations": [...],
          "chains": {
              "2024-01-19": {"calls": DataFrame, "puts": DataFrame},
              ...
          }
        }
    """
    if dte_targets is None:
        dte_targets = [30, 60, 90]

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return {"ticker": ticker, "price": None, "expirations": [], "chains": {}, "error": "No options data"}

        # Current price
        info = tk.info or {}
        price = _safe_get(info, "currentPrice", "regularMarketPrice", default=None)
        if price is None:
            hist = tk.history(period="2d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None

        chains: dict[str, Any] = {}
        seen_expiries: set[str] = set()

        for dte in dte_targets:
            exp = _nearest_expiry(expirations, dte)
            if exp and exp not in seen_expiries:
                seen_expiries.add(exp)
                chain = tk.option_chain(exp)
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                actual_dte = (exp_date - datetime.today().date()).days
                chains[exp] = {
                    "calls":      chain.calls,
                    "puts":       chain.puts,
                    "actual_dte": actual_dte,
                }
                time.sleep(0.1)

        return {
            "ticker":      ticker,
            "price":       price,
            "expirations": list(expirations),
            "chains":      chains,
        }
    except Exception as exc:
        return {"ticker": ticker, "price": None, "expirations": [], "chains": {}, "error": str(exc)}


# ---------------------------------------------------------------------------
# IV estimation
# ---------------------------------------------------------------------------

def estimate_iv_rank(ticker: str, current_iv: float | None = None) -> dict:
    """
    Estimate IV rank using 52-week high/low of the at-the-money implied vol.
    Falls back to historical volatility if options IV is unavailable.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty:
            return {"ticker": ticker, "iv_rank": None, "hv_30": None}

        # 30-day historical volatility as proxy
        log_ret = (hist["Close"] / hist["Close"].shift(1)).apply(
            lambda x: x if x > 0 else float("nan")
        ).apply(lambda x: __import__("math").log(x) if pd.notna(x) else float("nan"))
        hv_30 = float(log_ret.rolling(21).std().iloc[-1]) * (252 ** 0.5) * 100

        iv_rank = None
        if current_iv:
            # Use current_iv vs historical vol range as rough proxy
            iv_52w_low  = hv_30 * 0.6
            iv_52w_high = hv_30 * 1.6
            iv_rank = round(
                (current_iv - iv_52w_low) / max(iv_52w_high - iv_52w_low, 0.01) * 100, 1
            )
            iv_rank = max(0.0, min(100.0, iv_rank))

        return {
            "ticker":   ticker,
            "hv_30":    round(hv_30, 1),
            "iv_rank":  iv_rank,
        }
    except Exception:
        return {"ticker": ticker, "iv_rank": None, "hv_30": None}


# ---------------------------------------------------------------------------
# Convenience: fetch everything needed for dashboard + analysis
# ---------------------------------------------------------------------------

def fetch_all_data(tickers: list[str], fetch_options: bool = True
                   ) -> dict[str, Any]:
    """
    Master fetch: returns prices and optionally options chains for all tickers.
    """
    prices = fetch_prices(tickers)
    options: dict[str, Any] = {}

    if fetch_options:
        for ticker in tickers:
            options[ticker] = fetch_options_chain(ticker)
            time.sleep(0.2)

    return {"prices": prices, "options": options}
