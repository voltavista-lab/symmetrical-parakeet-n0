"""Composite scoring and ranking of screener candidates."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _normalise(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    """Min-max normalise a series to [low, high] range."""
    vmin, vmax = series.min(), series.max()
    if vmax == vmin:
        return pd.Series([0.5] * len(series), index=series.index)
    return low + (series - vmin) / (vmax - vmin) * (high - low)


class CandidateScorer:
    """
    Scores and ranks screener candidates using configurable weights.

    Dimensions scored:
    - IVR:           Higher IVR → better time to sell premium
    - premium_pct:   Higher premium relative to strike → better return
    - dividend_yield: Higher dividend → more total return
    - pop:           Higher POP → higher probability of success
    - liquidity:     Higher volume/OI → easier fills and tighter spreads
    """

    DEFAULT_WEIGHTS = {
        "ivr": 0.30,
        "premium_pct": 0.25,
        "dividend_yield": 0.15,
        "pop": 0.20,
        "liquidity": 0.10,
    }

    def __init__(self, config: dict) -> None:
        score_cfg = config.get("screener", {}).get("score_weights", {})
        self._weights = {k: score_cfg.get(k, v) for k, v in self.DEFAULT_WEIGHTS.items()}
        # Normalise weights to sum to 1
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def score_candidates(self, candidates: list[dict]) -> pd.DataFrame:
        """
        Score and rank a list of candidate dicts.

        Each dict should contain the keys from OptionsChainAnalyser.analyse().

        Returns a sorted DataFrame with an added 'composite_score' column.
        """
        if not candidates:
            return pd.DataFrame()

        df = pd.DataFrame(candidates)

        # ── Build feature columns ──────────────────────────────────────────────
        # IVR score (0-100 → normalised)
        df["_ivr_norm"] = _normalise(df["ivr"].clip(0, 100))

        # Premium % of strike (higher = better, cap at 5%)
        df["_premium_norm"] = _normalise(df["best_put_premium_pct"].clip(0, 5))

        # Dividend yield (cap at 10%)
        df["_div_norm"] = _normalise(df["dividend_yield"].clip(0, 10))

        # POP (already 0-100)
        df["_pop_norm"] = _normalise(df["best_put_pop"].clip(50, 95))

        # Liquidity: log of avg_volume
        import numpy as np
        df["_liq_norm"] = _normalise(np.log1p(df["avg_volume"]))

        # ── Composite score ────────────────────────────────────────────────────
        df["composite_score"] = (
            self._weights["ivr"] * df["_ivr_norm"]
            + self._weights["premium_pct"] * df["_premium_norm"]
            + self._weights["dividend_yield"] * df["_div_norm"]
            + self._weights["pop"] * df["_pop_norm"]
            + self._weights["liquidity"] * df["_liq_norm"]
        ).round(4)

        # Drop internal columns
        internal = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=internal)

        # Sort descending
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df.index += 1  # 1-based rank
        df.index.name = "rank"

        return df


def display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename columns for terminal/CSV display.

    Returns a clean DataFrame with human-readable column names.
    """
    col_map = {
        "ticker": "Ticker",
        "price": "Price",
        "sector": "Sector",
        "dividend_yield": "Div Yield %",
        "ivr": "IVR",
        "current_iv": "IV %",
        "hv30": "HV30 %",
        "iv_hv_spread": "IV-HV",
        "best_put_strike": "Put Strike",
        "best_put_dte": "DTE",
        "best_put_premium": "Premium",
        "best_put_premium_pct": "Prem %",
        "best_put_delta": "Delta",
        "best_put_pop": "POP %",
        "ex_dividend_date": "Ex-Div",
        "avg_volume": "Avg Vol",
        "composite_score": "Score",
    }

    available = {k: v for k, v in col_map.items() if k in df.columns}
    display = df[list(available.keys())].rename(columns=available)

    # Format numeric columns
    if "Price" in display.columns:
        display["Price"] = display["Price"].map("${:.2f}".format)
    if "Put Strike" in display.columns:
        display["Put Strike"] = display["Put Strike"].map("${:.2f}".format)
    if "Premium" in display.columns:
        display["Premium"] = display["Premium"].map("${:.2f}".format)
    if "Avg Vol" in display.columns:
        display["Avg Vol"] = display["Avg Vol"].map("{:,.0f}".format)
    if "Score" in display.columns:
        display["Score"] = display["Score"].map("{:.3f}".format)

    return display
