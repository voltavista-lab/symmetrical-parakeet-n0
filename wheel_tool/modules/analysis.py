"""
analysis.py - Core analytics: moneyness, roll scenarios, CC screening,
              wheel tracker, and risk summary.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .data_fetcher import estimate_iv_rank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    "PFE":  "Healthcare",
    "T":    "Communication Services",
    "BCE":  "Communication Services",
    "BNS":  "Financials",
    "VALE": "Materials",
    "CLF":  "Materials",
}

CAD_TICKERS: set[str] = {"BCE", "BNS"}

# Thresholds
CRITICAL_PNL_PCT    = -50.0   # flag as critical if loss > 50%
HIGH_YIELD_THRESHOLD = 10.0   # annualised yield % for "high priority" flag
SECTOR_CONC_LIMIT    = 40.0   # % of total value triggering sector warning


# ---------------------------------------------------------------------------
# Black-Scholes helpers (approximation for delta / theoretical price)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc approximation."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bs_put_price(S: float, K: float, T: float, r: float,
                 sigma: float) -> float:
    """Black-Scholes put price. T = years to expiry."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float,
                  sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_delta(S: float, K: float, T: float, r: float,
                 sigma: float) -> float:
    """Put delta (negative). Returns 0 on bad inputs."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) - 1.0


def bs_call_delta(S: float, K: float, T: float, r: float,
                  sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def _extract_iv(chain_row: pd.Series | None) -> float:
    """Pull IV from an option chain row; default to 0.30 if missing."""
    if chain_row is None:
        return 0.30
    iv = chain_row.get("impliedVolatility", None)
    if iv is None or (isinstance(iv, float) and math.isnan(iv)):
        return 0.30
    return max(float(iv), 0.01)


def _find_closest_strike(df: pd.DataFrame, strike: float) -> pd.Series | None:
    """Return the row in an options DataFrame whose strike is closest to target."""
    if df is None or df.empty:
        return None
    idx = (df["strike"] - strike).abs().idxmin()
    return df.loc[idx]


def _find_strike_row(df: pd.DataFrame, strike: float) -> pd.Series | None:
    """Exact match first, then nearest."""
    if df is None or df.empty:
        return None
    exact = df[df["strike"] == strike]
    if not exact.empty:
        return exact.iloc[0]
    return _find_closest_strike(df, strike)


# ---------------------------------------------------------------------------
# Moneyness & position enrichment
# ---------------------------------------------------------------------------

def enrich_positions(positions: list[dict], prices: dict[str, dict],
                     options_data: dict[str, Any]) -> list[dict]:
    """
    Add live price data, moneyness, intrinsic value, and distance-to-strike
    to each position dict. Returns enriched copy.
    """
    enriched = []
    for pos in positions:
        p = dict(pos)
        ticker = p["ticker"]
        price_info = prices.get(ticker, {})
        spot = price_info.get("price")

        p["spot_price"] = spot
        p["currency"]   = price_info.get("currency", p.get("currency", "USD"))

        if p["type"] == "short_put" and p.get("strike") and spot:
            strike = p["strike"]
            contracts = abs(p["contracts"])

            # Moneyness
            moneyness_pct = (spot - strike) / strike * 100
            p["moneyness_pct"]    = round(moneyness_pct, 2)
            p["itm"]              = spot < strike
            p["buffer_pct"]       = round(moneyness_pct, 2)   # positive = OTM

            # Intrinsic value (per share)
            intrinsic = max(strike - spot, 0.0)
            p["intrinsic_value"]  = round(intrinsic, 4)
            p["intrinsic_total"]  = round(intrinsic * contracts * 100, 2)

            # Max loss if assigned
            p["max_loss_if_assigned"] = round(strike * contracts * 100, 2)

        elif p["type"] == "covered_call" and spot:
            p["current_stock_value"] = round(abs(p["contracts"]) * 100 * spot, 2)

        # Critical flag
        p["is_critical"] = (
            p.get("pnl_pct", 0) < CRITICAL_PNL_PCT
            or (p["type"] == "short_put" and p.get("itm", False))
        )

        enriched.append(p)
    return enriched


# ---------------------------------------------------------------------------
# Roll analysis
# ---------------------------------------------------------------------------

def _get_put_mid(puts_df: pd.DataFrame | None, strike: float) -> float | None:
    """Return mid-price (bid+ask)/2 for a given strike in a puts DataFrame."""
    if puts_df is None or puts_df.empty:
        return None
    row = _find_strike_row(puts_df, strike)
    if row is None:
        return None
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    if bid == 0 and ask == 0:
        last = row.get("lastPrice", 0) or 0
        return float(last) if last else None
    return round((float(bid) + float(ask)) / 2, 4)


def analyse_rolls(position: dict, options_data: dict[str, Any],
                  spot: float | None) -> list[dict]:
    """
    For a short-put position, generate roll scenarios:
      - Roll out 30 / 60 / 90 days at same strike
      - Roll out 30 / 60 days at strike -0.50 and -1.00

    Returns list of scenario dicts sorted by net_credit descending.
    """
    if position["type"] != "short_put":
        return []

    ticker  = position["ticker"]
    strike  = position.get("strike")
    n_contr = abs(position.get("contracts", 1))

    if not strike or not spot:
        return []

    odata   = options_data.get(ticker, {})
    chains  = odata.get("chains", {})
    if not chains:
        return [{"error": "No options data available – cannot compute rolls"}]

    # Current position value (what we'd buy-to-close for)
    current_value = position.get("value", 0)      # negative = we receive cash
    # Cost to close = abs(current_value) per contract block, already total
    # position["value"] is negative for short puts (liability)
    close_cost_total = abs(current_value)          # we *pay* this to close

    scenarios = []
    strike_offsets = [0.0, -0.50, -1.00]

    for exp_date, chain_info in chains.items():
        actual_dte = chain_info.get("actual_dte", 0)
        puts_df    = chain_info.get("puts")

        for offset in strike_offsets:
            new_strike = round(strike + offset, 2)
            if new_strike <= 0:
                continue

            mid = _get_put_mid(puts_df, new_strike)
            if mid is None or mid == 0:
                # Fallback: approximate via HV-based BS
                hv_data = estimate_iv_rank(ticker)
                hv = (hv_data.get("hv_30") or 30) / 100
                T  = actual_dte / 365
                mid = bs_put_price(spot, new_strike, T, 0.045, hv)
                mid = round(mid, 4)

            # Net credit = premium received on new - cost to close existing
            gross_new_credit = mid * n_contr * 100
            net_credit       = round(gross_new_credit - close_cost_total, 2)

            scenarios.append({
                "expiry":         exp_date,
                "dte":            actual_dte,
                "new_strike":     new_strike,
                "strike_change":  offset,
                "new_mid":        round(mid, 4),
                "gross_credit":   round(gross_new_credit, 2),
                "close_cost":     round(close_cost_total, 2),
                "net_credit":     net_credit,
                "is_net_credit":  net_credit >= 0,
            })

    scenarios.sort(key=lambda x: x["net_credit"], reverse=True)
    return scenarios


# ---------------------------------------------------------------------------
# Covered call screener
# ---------------------------------------------------------------------------

def _best_call_strike(calls_df: pd.DataFrame, spot: float, dte: int,
                      target_delta_max: float = 0.40) -> dict | None:
    """
    Find the best OTM call strike:
    - Prioritise strikes with delta ≤ target_delta_max (~0.30–0.40)
    - Return the one with highest premium-to-strike ratio
    """
    if calls_df is None or calls_df.empty or spot is None:
        return None

    otm_calls = calls_df[calls_df["strike"] > spot].copy()
    if otm_calls.empty:
        return None

    # Use last price if mid not readily available
    otm_calls = otm_calls.copy()
    otm_calls["mid"] = (
        (otm_calls.get("bid", pd.Series(dtype=float)) + otm_calls.get("ask", pd.Series(dtype=float))) / 2
    ).fillna(otm_calls.get("lastPrice", pd.Series(dtype=float)))

    # Filter out zero-premium rows
    otm_calls = otm_calls[otm_calls["mid"] > 0.01]
    if otm_calls.empty:
        return None

    # Use BS delta if chain delta unavailable
    r     = 0.045
    T     = dte / 365
    rows  = []
    for _, row in otm_calls.iterrows():
        iv    = float(row.get("impliedVolatility") or 0.30)
        iv    = max(iv, 0.01)
        delta = bs_call_delta(spot, float(row["strike"]), T, r, iv)
        mid   = float(row["mid"])
        rows.append({
            "strike":    float(row["strike"]),
            "mid":       mid,
            "iv":        round(iv * 100, 1),
            "delta":     round(delta, 3),
            "bid":       float(row.get("bid") or 0),
            "ask":       float(row.get("ask") or 0),
        })

    # Keep delta ≤ target
    filtered = [r for r in rows if r["delta"] <= target_delta_max]
    if not filtered:
        filtered = rows      # fall back to all OTM

    # Best = highest annualised yield
    for r in filtered:
        ann_yield = (r["mid"] / r["strike"]) * (365 / max(dte, 1)) * 100
        r["ann_yield_pct"] = round(ann_yield, 2)

    filtered.sort(key=lambda x: x["ann_yield_pct"], reverse=True)
    return filtered[0] if filtered else None


def screen_covered_calls(tickers: list[str], prices: dict[str, dict],
                         options_data: dict[str, Any]) -> list[dict]:
    """
    Screen a list of tickers for covered call opportunities at ~30 DTE.
    Returns list of opportunity dicts sorted by annualised yield.
    """
    results = []

    for ticker in tickers:
        price_info = prices.get(ticker, {})
        spot       = price_info.get("price")
        div_yield  = price_info.get("div_yield", 0.0)

        odata      = options_data.get(ticker, {})
        chains     = odata.get("chains", {})
        if not chains or not spot:
            results.append({
                "ticker":    ticker,
                "spot":      spot,
                "div_yield": div_yield,
                "error":     "No data",
            })
            continue

        # Use nearest to 30-DTE chain
        target_exp  = None
        target_dte  = 9999
        target_chain = None
        for exp, cinfo in chains.items():
            dte = cinfo.get("actual_dte", 0)
            if 14 <= dte <= 50 and abs(dte - 30) < abs(target_dte - 30):
                target_exp   = exp
                target_dte   = dte
                target_chain = cinfo

        if target_chain is None:
            # take the first available
            exp, target_chain = next(iter(chains.items()))
            target_dte = target_chain.get("actual_dte", 30)

        calls_df = target_chain.get("calls")
        best     = _best_call_strike(calls_df, spot, target_dte)

        if best is None:
            results.append({
                "ticker":    ticker,
                "spot":      spot,
                "div_yield": div_yield,
                "dte":       target_dte,
                "error":     "No suitable call found",
            })
            continue

        # IV rank
        iv_rank_info = estimate_iv_rank(ticker, best["iv"])
        hv_30        = iv_rank_info.get("hv_30")

        total_ann_yield = round(best["ann_yield_pct"] + (div_yield or 0.0), 2)

        results.append({
            "ticker":         ticker,
            "spot":           round(spot, 2),
            "div_yield":      div_yield,
            "dte":            target_dte,
            "expiry":         target_exp,
            "strike":         best["strike"],
            "premium":        best["mid"],
            "delta":          best["delta"],
            "iv":             best["iv"],
            "iv_rank":        iv_rank_info.get("iv_rank"),
            "hv_30":          hv_30,
            "ann_yield_pct":  best["ann_yield_pct"],
            "total_yield_pct": total_ann_yield,
            "high_priority":  total_ann_yield >= HIGH_YIELD_THRESHOLD,
            "currency":       price_info.get("currency", "USD"),
            "sector":         price_info.get("sector", SECTOR_MAP.get(ticker, "Unknown")),
        })

    results.sort(key=lambda x: x.get("total_yield_pct", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Wheel tracker / effective cost basis
# ---------------------------------------------------------------------------

def compute_wheel_state(positions: list[dict], premium_history: list[dict],
                        prices: dict[str, dict]) -> list[dict]:
    """
    For each ticker with active positions, compute:
    - Total premiums collected (history)
    - Effective cost basis = original cost - premiums collected
    - Breakeven price
    - If assigned put → suggest best CC strike
    """
    # Aggregate premiums by ticker
    premiums_by_ticker: dict[str, float] = {}
    for entry in premium_history:
        t = entry.get("ticker", "")
        premiums_by_ticker[t] = premiums_by_ticker.get(t, 0.0) + entry.get("premium", 0.0)

    # Build per-ticker state
    tickers_seen: dict[str, dict] = {}
    for pos in positions:
        ticker = pos["ticker"]
        if ticker not in tickers_seen:
            tickers_seen[ticker] = {
                "ticker":             ticker,
                "has_shares":         False,
                "short_puts":         [],
                "covered_calls":      [],
                "total_premiums_collected": premiums_by_ticker.get(ticker, 0.0),
            }
        state = tickers_seen[ticker]
        if pos["type"] == "covered_call":
            state["has_shares"] = True
            state["covered_calls"].append(pos)
        elif pos["type"] == "short_put":
            state["short_puts"].append(pos)

    results = []
    for ticker, state in tickers_seen.items():
        price_info = prices.get(ticker, {})
        spot       = price_info.get("price")

        # Effective cost basis estimation for covered call positions
        cost_basis_per_share = None
        breakeven = None
        if state["covered_calls"]:
            cc = state["covered_calls"][0]
            # Value divided by (contracts * 100) ≈ cost basis if pnl% is known
            # effective_basis = current_price / (1 + pnl_pct/100)
            pnl_pct = cc.get("pnl_pct", 0) / 100
            if spot and pnl_pct != -1:
                cost_basis_per_share = round(spot / (1 + pnl_pct), 4)

        premium_per_share = (
            state["total_premiums_collected"] /
            (sum(abs(p["contracts"]) for p in state["covered_calls"]) * 100 or 1)
        )
        if cost_basis_per_share:
            breakeven = round(cost_basis_per_share - premium_per_share, 4)

        # Suggest a CC if recently assigned (has shares, no CC currently)
        suggest_cc = state["has_shares"] and not state["covered_calls"]

        state.update({
            "spot":                  spot,
            "cost_basis_per_share":  cost_basis_per_share,
            "premium_per_share":     round(premium_per_share, 4),
            "breakeven":             breakeven,
            "suggest_cc":            suggest_cc,
        })
        results.append(state)

    return results


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------

def compute_risk_summary(enriched_positions: list[dict],
                         account: dict,
                         prices: dict[str, dict]) -> dict:
    """
    Compute portfolio-wide risk metrics:
    - Total capital at risk from short puts (max loss if all assigned)
    - Sector concentration
    - CAD vs USD exposure
    - Total P&L summary
    """
    usd_cad = 1.36     # approximate; can be replaced with live rate

    total_pnl       = 0.0
    total_value     = 0.0
    capital_at_risk = 0.0

    sector_exposure: dict[str, float] = {}
    usd_exposure = 0.0
    cad_exposure = 0.0

    for pos in enriched_positions:
        pnl   = pos.get("pnl", 0.0) or 0.0
        value = abs(pos.get("value", 0.0) or 0.0)
        total_pnl   += pnl
        total_value += value

        ticker   = pos["ticker"]
        currency = pos.get("currency", "USD")
        sector   = SECTOR_MAP.get(ticker, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + value

        if currency == "CAD":
            cad_exposure += value
        else:
            usd_exposure += value

        if pos["type"] == "short_put":
            capital_at_risk += pos.get("max_loss_if_assigned", 0.0)

    # Sector concentration as % of total value
    sector_pct: dict[str, float] = {}
    for sec, val in sector_exposure.items():
        sector_pct[sec] = round(val / max(total_value, 1) * 100, 1)

    concentrated_sectors = [s for s, pct in sector_pct.items() if pct >= SECTOR_CONC_LIMIT]

    # CAD/USD breakdown
    total_portfolio_cad = cad_exposure + usd_exposure * usd_cad
    cad_pct = round(cad_exposure / max(total_portfolio_cad, 1) * 100, 1)
    usd_pct = round(100 - cad_pct, 1)

    # Count criticals
    critical_positions = [p for p in enriched_positions if p.get("is_critical")]

    return {
        "total_pnl":              round(total_pnl, 2),
        "total_value":            round(total_value, 2),
        "capital_at_risk_usd":    round(capital_at_risk, 2),
        "capital_at_risk_cad":    round(capital_at_risk * usd_cad, 2),
        "account_value_cad":      account.get("value_cad", 0),
        "buying_power_cad":       account.get("buying_power_cad", 0),
        "cash_cad":               account.get("cash_cad", 0),
        "risk_pct_of_account":    round(capital_at_risk * usd_cad / max(account.get("value_cad", 1), 1) * 100, 1),
        "sector_exposure_pct":    sector_pct,
        "concentrated_sectors":   concentrated_sectors,
        "cad_exposure_pct":       cad_pct,
        "usd_exposure_pct":       usd_pct,
        "critical_count":         len(critical_positions),
        "critical_positions":     [f"{p['ticker']} {p['type']}" for p in critical_positions],
        "usd_cad_rate_used":      usd_cad,
    }
