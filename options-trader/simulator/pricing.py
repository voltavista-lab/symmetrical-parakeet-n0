"""
Option pricing models for backtesting.

Uses Black-Scholes as the primary model since historical options chains
are typically unavailable for free. Provides a close-enough approximation
for strategy simulation and parameter optimization.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
from scipy.stats import norm


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
) -> float:
    """
    Black-Scholes European option price.

    Args:
        S:           Underlying spot price
        K:           Strike price
        T:           Time to expiration in years
        r:           Annual risk-free interest rate (decimal)
        sigma:       Annual implied volatility (decimal)
        option_type: "put" or "call"

    Returns:
        Theoretical option price per share.
    """
    if T <= 0:
        intrinsic = max(0.0, K - S) if option_type == "put" else max(0.0, S - K)
        return intrinsic
    if sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    return max(0.0, price)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return (-1.0 if S < K else 0.0) if option_type == "put" else (1.0 if S > K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return (norm.cdf(d1) - 1.0) if option_type == "put" else norm.cdf(d1)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> float:
    """Black-Scholes theta (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "put":
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )
    else:
        theta = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
    return theta / 365.0  # per calendar day


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (per 1% move in IV)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T) / 100.0


def estimate_iv_from_hv(
    hv_series: "pd.Series",  # type: ignore[name-defined]
    current_date: date,
    iv_premium_ratio: float = 1.15,
) -> float:
    """
    Estimate implied volatility from historical volatility.

    Options markets typically price in an IV premium over realised HV.
    The ratio of IV/HV historically averages around 1.1–1.2.

    Args:
        hv_series:          Series of historical daily HV values (decimal)
        current_date:       Date to look up
        iv_premium_ratio:   Multiplier applied to HV to estimate IV

    Returns:
        Estimated IV as a decimal (e.g., 0.25 for 25%).
    """
    import pandas as pd

    ts = pd.Timestamp(current_date)
    if hv_series.empty:
        return 0.25  # default fallback

    # Get closest available date
    try:
        idx = hv_series.index.get_indexer([ts], method="nearest")
        hv = float(hv_series.iloc[idx[0]])
    except (KeyError, IndexError):
        hv = float(hv_series.iloc[-1])

    if math.isnan(hv) or hv <= 0:
        return 0.25

    return hv * iv_premium_ratio


def find_strike_by_delta(
    S: float,
    T: float,
    sigma: float,
    target_delta: float,
    option_type: str = "put",
    r: float = 0.045,
) -> float:
    """
    Find theoretical strike for a target delta using BS inversion.

    ``target_delta`` is the absolute delta magnitude (e.g. 0.30 means 30-delta).
    For puts the actual delta is negative; for calls it is positive.
    """
    from scipy.stats import norm

    if option_type == "put":
        # Put delta = N(d1) - 1; for |delta|=0.30 → N(d1)=0.70 → d1=N^-1(0.70)
        d1 = norm.ppf(1.0 - target_delta)
    else:
        # Call delta = N(d1); for delta=0.30 → d1=N^-1(0.30)
        d1 = norm.ppf(target_delta)

    exponent = -(d1 * sigma * math.sqrt(T)) + (r + 0.5 * sigma ** 2) * T
    return S * math.exp(exponent)


def price_option_on_date(
    S: float,
    K: float,
    expiration: date,
    pricing_date: date,
    sigma: float,
    option_type: str = "put",
    r: float = 0.045,
) -> float:
    """
    Price an option on a specific date using Black-Scholes.

    Args:
        S:            Underlying spot on pricing_date
        K:            Strike price
        expiration:   Option expiration date
        pricing_date: Date to price the option
        sigma:        IV estimate for pricing_date (decimal)
        option_type:  "put" or "call"
        r:            Risk-free rate (decimal)

    Returns:
        Theoretical mid-price for the option.
    """
    T = max(0.0, (expiration - pricing_date).days / 365.0)
    return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
