"""Options scanner - finds options matching DTE/IV/delta criteria."""

import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def _estimate_delta(option_type: str, stock_price: float, strike: float,
                    dte: int, iv: float, risk_free: float = 0.05) -> float:
    """Estimate option delta using Black-Scholes."""
    if dte <= 0 or iv <= 0 or stock_price <= 0 or strike <= 0:
        return 0.0
    t = dte / 365.0
    d1 = (np.log(stock_price / strike) + (risk_free + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def get_options_chain(ticker: str, dte_range: tuple[int, int] = (30, 60)) -> pd.DataFrame:
    """Fetch options chain for a ticker, filtering by DTE range.

    Returns DataFrame with columns: type, strike, expiry, dte, bid, ask,
    mid_price, impliedVolatility, openInterest, volume.
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return pd.DataFrame()

    today = datetime.date.today()
    min_dte, max_dte = dte_range
    rows = []

    for exp_str in expirations:
        exp_date = datetime.date.fromisoformat(exp_str)
        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = stock.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in [("put", chain.puts), ("call", chain.calls)]:
            for _, row in df.iterrows():
                rows.append({
                    "type": opt_type,
                    "strike": row.get("strike", 0),
                    "expiry": exp_str,
                    "dte": dte,
                    "bid": row.get("bid", 0),
                    "ask": row.get("ask", 0),
                    "mid_price": (row.get("bid", 0) + row.get("ask", 0)) / 2,
                    "impliedVolatility": row.get("impliedVolatility", 0),
                    "openInterest": row.get("openInterest", 0),
                    "volume": row.get("volume", 0) or 0,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def calculate_iv_rank(ticker: str, lookback_days: int = 252) -> float:
    """Calculate IV rank (0-100) based on historical volatility range.

    Uses realized volatility of daily returns as a proxy when historical
    IV data is not available directly.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"{lookback_days + 30}d")
    if hist.empty or len(hist) < 30:
        return 50.0  # default if insufficient data

    returns = hist["Close"].pct_change().dropna()
    window = 21  # ~1 month rolling window
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    if rolling_vol.empty:
        return 50.0

    current_vol = rolling_vol.iloc[-1]
    low_vol = rolling_vol.min()
    high_vol = rolling_vol.max()

    if high_vol == low_vol:
        return 50.0

    iv_rank = (current_vol - low_vol) / (high_vol - low_vol) * 100
    return float(np.clip(iv_rank, 0, 100))


def _compute_composite_score(row: pd.Series) -> float:
    """Compute composite score from iv_rank, premium yield, dividend, liquidity."""
    iv_score = row.get("iv_rank", 0) / 100 * 30
    premium_score = min(row.get("annual_premium_yield", 0) / 50, 1.0) * 30
    div_score = min(row.get("dividend_yield", 0) / 5, 1.0) * 20
    oi = row.get("openInterest", 0) or 0
    vol = row.get("volume", 0) or 0
    liquidity_score = min((oi + vol * 10) / 5000, 1.0) * 20
    return iv_score + premium_score + div_score + liquidity_score


def scan_for_candidates(
    filtered_stocks: list[dict],
    strategy_params: dict,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Scan filtered stocks for option candidates matching strategy criteria.

    Args:
        filtered_stocks: List of dicts from filter_watchlist.
        strategy_params: Dict with strategy parameters from config.
        delay: Seconds between API calls.

    Returns:
        DataFrame with scored candidates.
    """
    dte_range = strategy_params["dte_range"]
    min_oi = strategy_params["min_open_interest"]
    min_vol = strategy_params["min_volume"]
    delta_range_put = strategy_params["delta_range_put"]
    delta_range_call = strategy_params["delta_range_call"]
    min_iv_rank = strategy_params["min_iv_rank"]

    all_candidates = []

    for stock_info in filtered_stocks:
        ticker = stock_info["ticker"]
        price = stock_info["price"]

        try:
            chain_df = get_options_chain(ticker, dte_range)
        except Exception:
            continue

        if chain_df.empty:
            time.sleep(delay)
            continue

        try:
            iv_rank = calculate_iv_rank(ticker)
        except Exception:
            iv_rank = 50.0

        if iv_rank < min_iv_rank:
            time.sleep(delay)
            continue

        for _, row in chain_df.iterrows():
            oi = row.get("openInterest", 0) or 0
            vol = row.get("volume", 0) or 0
            if oi < min_oi or vol < min_vol:
                continue

            iv = row.get("impliedVolatility", 0) or 0.01
            dte = row["dte"]
            strike = row["strike"]
            opt_type = row["type"]

            delta = _estimate_delta(opt_type, price, strike, dte, iv)

            if opt_type == "put":
                lo, hi = delta_range_put
                if not (hi <= delta <= lo):  # delta is negative
                    continue
            else:
                lo, hi = delta_range_call
                if not (lo <= delta <= hi):
                    continue

            mid = row["mid_price"]
            annual_yield = (mid / strike * 365 / dte * 100) if dte > 0 and strike > 0 else 0

            candidate = {
                "ticker": ticker,
                "option_type": opt_type,
                "strike": strike,
                "expiry": row["expiry"],
                "dte": dte,
                "bid": row["bid"],
                "ask": row["ask"],
                "mid_price": mid,
                "implied_vol": iv,
                "delta": round(delta, 3),
                "openInterest": oi,
                "volume": vol,
                "annual_premium_yield": round(annual_yield, 2),
                "dividend_yield": stock_info["dividend_yield"],
                "iv_rank": round(iv_rank, 1),
            }
            candidate["composite_score"] = round(
                _compute_composite_score(pd.Series(candidate)), 2
            )
            all_candidates.append(candidate)

        time.sleep(delay)

    if not all_candidates:
        return pd.DataFrame()

    df = pd.DataFrame(all_candidates)
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
