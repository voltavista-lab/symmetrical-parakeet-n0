"""Options chain analysis: IV Rank, delta selection, scoring."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.adapters import OptionsDataAdapter, StockQuote
from data.cache import get_cache

logger = logging.getLogger(__name__)


# ─── Black-Scholes helpers ────────────────────────────────────────────────────

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
) -> float:
    """
    Black-Scholes option price.

    Args:
        S:           Spot price
        K:           Strike price
        T:           Time to expiration in years
        r:           Risk-free rate (decimal)
        sigma:       Implied volatility (annualised, decimal)
        option_type: "put" or "call"

    Returns:
        Theoretical option price.
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, (K - S) if option_type == "put" else (S - K))

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(0.0, price)


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return -1.0 if option_type == "put" else 1.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    return norm.cdf(d1)


def find_strike_by_delta(
    S: float,
    T: float,
    sigma: float,
    target_delta: float,
    option_type: str = "put",
    r: float = 0.045,
) -> float:
    """
    Find the strike price closest to a target delta using an analytical formula.

    ``target_delta`` is the absolute delta magnitude (e.g. 0.30 means 30-delta).
    For puts: |delta| = 1 - N(d1) → N(d1) = 1 - target_delta → d1 = N^-1(1 - target_delta).
    For calls: delta = N(d1) → d1 = N^-1(target_delta).
    """
    if option_type == "put":
        d1 = norm.ppf(1.0 - target_delta)
    else:
        d1 = norm.ppf(target_delta)

    # K = S * exp(-(d1 * sigma * sqrt(T)) + (r + sigma²/2) * T)
    exponent = -(d1 * sigma * math.sqrt(T)) + (r + 0.5 * sigma ** 2) * T
    return S * math.exp(exponent)


def probability_of_profit(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str = "put",
    credit: float = 0.0,
    r: float = 0.045,
) -> float:
    """
    Probability of profit for a short option position at expiration.

    For a short put with credit:   POP = P(S_T > K - credit)
    For a short call with credit:  POP = P(S_T < K + credit)

    Uses log-normal distribution of underlying price.
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    if option_type == "put":
        breakeven = K - credit
        # P(S_T > breakeven) = N(d2) where we compute with breakeven as K
        d2 = (math.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d2))
    else:
        breakeven = K + credit
        d2 = (math.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(1.0 - norm.cdf(d2))


# ─── IV Rank calculation ──────────────────────────────────────────────────────

def calculate_ivr(iv_series: pd.Series, current_iv: float) -> float:
    """
    Calculate IV Rank (IVR) as the percentile of current IV vs. 52-week range.

    IVR = (current_IV - 52w_low) / (52w_high - 52w_low) × 100

    Returns a value in [0, 100].
    """
    if iv_series.empty or len(iv_series) < 5:
        return 0.0
    # Use last 252 trading days (~1 year)
    series = iv_series.tail(252).dropna()
    if series.empty:
        return 0.0
    iv_low = float(series.min())
    iv_high = float(series.max())
    if iv_high <= iv_low:
        return 50.0
    ivr = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
    return round(max(0.0, min(100.0, ivr)), 1)


# ─── Main chain analyser ──────────────────────────────────────────────────────

class OptionsChainAnalyser:
    """Analyses an options chain to find optimal premium-selling candidates."""

    def __init__(self, config: dict, adapter: OptionsDataAdapter) -> None:
        self._cfg = config
        self._adapter = adapter
        self._opts_cfg = config.get("options", {})
        self._screener_cfg = config.get("screener", {})
        self._cache = get_cache(config.get("data", {}).get("cache_dir", ".cache"))
        self._rf_rate = 0.045  # approximate risk-free rate

    def analyse(self, quote: StockQuote) -> Optional[dict]:
        """
        Run full options chain analysis for a stock.

        Returns a dict with IVR, best put candidate, IV/HV spread, etc.
        Returns None if no suitable options data found.
        """
        ticker = quote.ticker
        min_dte = self._cfg.get("options", {}).get("dte_window_min", 30)
        max_dte = self._cfg.get("options", {}).get("dte_window_max", 60)
        chain_ttl = self._cfg.get("data", {}).get("options_chain_ttl_minutes", 60)

        # Cache key for options chain
        cache_key = f"chain_{ticker}_{min_dte}_{max_dte}"
        chain = self._cache.get(cache_key, ttl_minutes=chain_ttl)
        if chain is None:
            chain = self._adapter.get_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)
            if not chain.empty:
                self._cache.set(cache_key, chain)

        # IV history for IVR calculation
        iv_cache_key = f"iv_hist_{ticker}"
        iv_ttl = self._cfg.get("data", {}).get("iv_history_ttl_hours", 24) * 60
        iv_history = self._cache.get(iv_cache_key, ttl_minutes=iv_ttl)
        if iv_history is None:
            iv_history = self._adapter.get_iv_history(ticker)
            if not iv_history.empty:
                self._cache.set(iv_cache_key, iv_history)

        # Determine current IV from chain or HV
        current_iv = quote.historical_volatility_30d / 100.0  # convert pct to decimal
        if not chain.empty and "implied_volatility" in chain.columns:
            atm_iv = self._atm_iv(chain, quote.price)
            if atm_iv > 0:
                current_iv = atm_iv

        ivr = calculate_ivr(iv_history, current_iv * 100)  # pass as pct

        # IV vs HV spread (both in percent)
        iv_pct = current_iv * 100
        hv_pct = quote.historical_volatility_30d
        iv_hv_spread = iv_pct - hv_pct

        # Find best put candidate
        best_put = self._find_best_put(chain, quote, current_iv)
        if best_put is None:
            # Synthesize using BS if chain empty
            best_put = self._synthesize_put(quote, current_iv)

        if best_put is None:
            return None

        return {
            "ticker": ticker,
            "price": quote.price,
            "sector": quote.sector,
            "dividend_yield": quote.dividend_yield,
            "hv30": hv_pct,
            "current_iv": iv_pct,
            "ivr": ivr,
            "iv_hv_spread": iv_hv_spread,
            "avg_volume": quote.avg_volume,
            "ex_dividend_date": quote.ex_dividend_date,
            **best_put,
        }

    def _atm_iv(self, chain: pd.DataFrame, spot: float) -> float:
        """Return average IV of near-ATM options (within 2% of spot)."""
        if chain.empty:
            return 0.0
        atm_mask = (chain["strike"] - spot).abs() / spot < 0.02
        atm = chain[atm_mask]
        if atm.empty:
            # Widest ATM: pick 3 closest strikes
            chain = chain.copy()
            chain["_dist"] = (chain["strike"] - spot).abs()
            atm = chain.nsmallest(6, "_dist")
        iv_vals = atm["implied_volatility"].replace(0, np.nan).dropna()
        if iv_vals.empty:
            return 0.0
        return float(iv_vals.mean())

    def _find_best_put(
        self, chain: pd.DataFrame, quote: StockQuote, current_iv: float
    ) -> Optional[dict]:
        """Select the best short-put candidate from the live chain."""
        if chain.empty:
            return None

        puts = chain[chain["option_type"] == "put"].copy()
        if puts.empty:
            return None

        target_delta = self._opts_cfg.get("short_put_delta", 0.30)
        spot = quote.price

        # Compute delta from BS if not in chain
        if puts["delta"].isna().all():
            puts = puts.copy()
            puts["delta"] = puts.apply(
                lambda r: abs(
                    bs_delta(
                        S=spot,
                        K=r["strike"],
                        T=r["dte"] / 365.0,
                        r=self._rf_rate,
                        sigma=r["implied_volatility"] if r["implied_volatility"] > 0 else current_iv,
                        option_type="put",
                    )
                ),
                axis=1,
            )
        else:
            puts["delta"] = puts["delta"].abs()

        # Filter to valid delta range (0.10 to 0.50 for robustness)
        valid = puts[(puts["delta"] >= 0.10) & (puts["delta"] <= 0.50)]
        if valid.empty:
            valid = puts

        # Pick contract closest to target delta
        valid = valid.copy()
        valid["_delta_diff"] = (valid["delta"] - target_delta).abs()
        best_row = valid.nsmallest(1, "_delta_diff").iloc[0]

        # Use mid-price as premium
        premium = (best_row["bid"] + best_row["ask"]) / 2.0
        if premium <= 0:
            premium = best_row["last"] if best_row["last"] > 0 else 0.0

        if premium <= 0:
            # Synthesize with BS
            iv = best_row["implied_volatility"] if best_row["implied_volatility"] > 0 else current_iv
            premium = bs_price(
                S=spot,
                K=best_row["strike"],
                T=best_row["dte"] / 365.0,
                r=self._rf_rate,
                sigma=iv,
                option_type="put",
            )

        premium_pct = (premium / best_row["strike"]) * 100 if best_row["strike"] > 0 else 0.0

        iv = best_row.get("implied_volatility", current_iv)
        if not iv or iv == 0:
            iv = current_iv

        pop = probability_of_profit(
            S=spot,
            K=float(best_row["strike"]),
            T=float(best_row["dte"]) / 365.0,
            sigma=float(iv),
            option_type="put",
            credit=premium,
            r=self._rf_rate,
        )

        return {
            "best_put_strike": float(best_row["strike"]),
            "best_put_exp": best_row["expiration"],
            "best_put_dte": int(best_row["dte"]),
            "best_put_premium": round(premium, 2),
            "best_put_premium_pct": round(premium_pct, 2),
            "best_put_delta": round(float(best_row["delta"]), 3),
            "best_put_iv": round(float(iv) * 100, 1) if iv < 5 else round(float(iv), 1),
            "best_put_pop": round(pop * 100, 1),
            "best_put_oi": int(best_row.get("open_interest", 0) or 0),
        }

    def _synthesize_put(self, quote: StockQuote, current_iv: float) -> Optional[dict]:
        """Synthesize put pricing using Black-Scholes when chain is unavailable."""
        if current_iv <= 0 or quote.price <= 0:
            return None

        target_delta = self._opts_cfg.get("short_put_delta", 0.30)
        target_dte = self._cfg.get("options", {}).get("target_dte", 45)
        spot = quote.price

        # Find the strike for target delta
        strike = find_strike_by_delta(
            S=spot,
            T=target_dte / 365.0,
            sigma=current_iv,
            target_delta=target_delta,
            option_type="put",
            r=self._rf_rate,
        )
        # Round to nearest $1 or $2.50
        increment = 2.5 if spot > 50 else 1.0
        strike = round(strike / increment) * increment

        premium = bs_price(
            S=spot,
            K=strike,
            T=target_dte / 365.0,
            r=self._rf_rate,
            sigma=current_iv,
            option_type="put",
        )
        premium_pct = (premium / strike) * 100

        delta = abs(
            bs_delta(
                S=spot, K=strike, T=target_dte / 365.0, r=self._rf_rate, sigma=current_iv, option_type="put"
            )
        )
        pop = probability_of_profit(
            S=spot,
            K=strike,
            T=target_dte / 365.0,
            sigma=current_iv,
            option_type="put",
            credit=premium,
            r=self._rf_rate,
        )

        today = date.today()
        expiration = today + timedelta(days=target_dte)

        return {
            "best_put_strike": round(strike, 2),
            "best_put_exp": expiration,
            "best_put_dte": target_dte,
            "best_put_premium": round(premium, 2),
            "best_put_premium_pct": round(premium_pct, 2),
            "best_put_delta": round(delta, 3),
            "best_put_iv": round(current_iv * 100, 1),
            "best_put_pop": round(pop * 100, 1),
            "best_put_oi": 0,  # synthesized — no real OI
        }
