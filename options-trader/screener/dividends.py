"""Dividend data and ex-date tracking for options strategy decisions."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from data.adapters import StockQuote

logger = logging.getLogger(__name__)


def days_to_ex_dividend(quote: StockQuote) -> Optional[int]:
    """
    Return days until the next ex-dividend date, or None if unknown.

    Negative values mean the ex-date was in the past (use with caution).
    """
    if quote.ex_dividend_date is None:
        return None
    return (quote.ex_dividend_date - date.today()).days


def covered_call_ex_div_risk(
    quote: StockQuote,
    expiration: date,
) -> str:
    """
    Assess ex-dividend risk for a covered call position.

    When holding a covered call, early assignment risk spikes immediately
    before the ex-dividend date because call buyers may exercise to capture
    the dividend. This function returns a risk label.

    Returns:
        "HIGH"   — ex-div falls within the option's life and is within 14 days
        "MEDIUM" — ex-div falls within the option's life but > 14 days away
        "LOW"    — ex-div is outside the option's expiration window
        "UNKNOWN" — ex-div date not available
    """
    if quote.ex_dividend_date is None:
        return "UNKNOWN"

    today = date.today()
    days_to_exp = (expiration - today).days
    days_to_exdiv = (quote.ex_dividend_date - today).days

    if days_to_exdiv < 0 or days_to_exdiv > days_to_exp:
        return "LOW"

    if days_to_exdiv <= 14:
        return "HIGH"

    return "MEDIUM"


def short_put_dividend_check(
    quote: StockQuote,
    expiration: date,
) -> dict:
    """
    For short puts, check if assignment near ex-div is a concern.

    If a short put is assigned just before ex-div, the trader receives
    shares that will go ex-dividend shortly after — which may be desirable
    (collect dividend) or problematic (capital tied up). This returns info
    for the trade log display.

    Returns a dict with:
        ex_div_in_window: bool
        days_to_ex_div: int | None
        note: str
    """
    if quote.ex_dividend_date is None:
        return {
            "ex_div_in_window": False,
            "days_to_ex_div": None,
            "note": "Ex-div date unknown",
        }

    today = date.today()
    days_to_exp = (expiration - today).days
    days_to_exdiv = (quote.ex_dividend_date - today).days
    in_window = 0 <= days_to_exdiv <= days_to_exp

    note = ""
    if in_window:
        note = (
            f"Ex-div in {days_to_exdiv}d (before expiry in {days_to_exp}d). "
            "Assignment would capture dividend but requires capital."
        )
    else:
        note = f"Ex-div in {days_to_exdiv}d — outside option window."

    return {
        "ex_div_in_window": in_window,
        "days_to_ex_div": days_to_exdiv,
        "note": note,
    }


def annualised_dividend_yield(quote: StockQuote) -> float:
    """Return the annualised dividend yield as a percentage."""
    return quote.dividend_yield


def combined_premium_dividend_yield(
    premium: float,
    strike: float,
    dividend_yield_pct: float,
    dte: int,
) -> float:
    """
    For covered calls: annualised combined yield of premium + dividend.

    Args:
        premium:            Option premium received (per share)
        strike:             Call strike price
        dividend_yield_pct: Annual dividend yield in percent
        dte:                Days to expiration

    Returns:
        Combined annualised return as a percentage of the strike price.
    """
    if strike <= 0 or dte <= 0:
        return 0.0
    premium_annualised = (premium / strike) * (365.0 / dte) * 100
    return premium_annualised + dividend_yield_pct
