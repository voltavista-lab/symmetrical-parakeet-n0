"""Tests for screener components: scoring, dividends, universe filtering."""

from __future__ import annotations

import sys
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screener.scoring import CandidateScorer, display_columns
from screener.dividends import (
    covered_call_ex_div_risk,
    days_to_ex_dividend,
    combined_premium_dividend_yield,
)
from data.adapters.base import StockQuote


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_quote(
    ticker: str = "TEST",
    price: float = 100.0,
    dividend_yield: float = 3.0,
    ex_dividend_date: date | None = None,
    avg_volume: int = 1_000_000,
) -> StockQuote:
    return StockQuote(
        ticker=ticker,
        price=price,
        volume=avg_volume,
        avg_volume=avg_volume,
        market_cap=10_000_000_000,
        sector="Technology",
        industry="Software",
        dividend_yield=dividend_yield,
        dividend_rate=dividend_yield * price / 100,
        ex_dividend_date=ex_dividend_date,
        next_earnings_date=None,
        fifty_two_week_high=price * 1.2,
        fifty_two_week_low=price * 0.8,
        historical_volatility_30d=25.0,
    )


def make_candidate(
    ticker: str = "TEST",
    price: float = 100.0,
    ivr: float = 50.0,
    dividend_yield: float = 3.0,
    best_put_premium_pct: float = 1.5,
    best_put_pop: float = 70.0,
    avg_volume: int = 1_000_000,
) -> dict:
    return {
        "ticker": ticker,
        "price": price,
        "sector": "Technology",
        "dividend_yield": dividend_yield,
        "hv30": 22.0,
        "current_iv": 27.0,
        "ivr": ivr,
        "iv_hv_spread": 5.0,
        "avg_volume": avg_volume,
        "ex_dividend_date": None,
        "best_put_strike": price * 0.92,
        "best_put_exp": date.today(),
        "best_put_dte": 45,
        "best_put_premium": price * best_put_premium_pct / 100,
        "best_put_premium_pct": best_put_premium_pct,
        "best_put_delta": 0.28,
        "best_put_iv": 27.0,
        "best_put_pop": best_put_pop,
        "best_put_oi": 1200,
        "composite_score": 0.0,
    }


# ─── CandidateScorer tests ────────────────────────────────────────────────────

class TestCandidateScorer:
    DEFAULT_CONFIG = {
        "screener": {
            "score_weights": {
                "ivr": 0.30,
                "premium_pct": 0.25,
                "dividend_yield": 0.15,
                "pop": 0.20,
                "liquidity": 0.10,
            }
        }
    }

    def test_score_single_candidate_returns_dataframe(self):
        scorer = CandidateScorer(self.DEFAULT_CONFIG)
        result = scorer.score_candidates([make_candidate()])
        assert not result.empty
        assert "composite_score" in result.columns

    def test_score_between_0_and_1(self):
        scorer = CandidateScorer(self.DEFAULT_CONFIG)
        candidates = [make_candidate(ticker=f"T{i}", ivr=float(i * 10)) for i in range(1, 6)]
        result = scorer.score_candidates(candidates)
        assert result["composite_score"].between(0, 1).all()

    def test_higher_ivr_scores_higher_all_else_equal(self):
        scorer = CandidateScorer(self.DEFAULT_CONFIG)
        low_ivr = make_candidate(ticker="A", ivr=30.0)
        high_ivr = make_candidate(ticker="B", ivr=70.0)
        result = scorer.score_candidates([low_ivr, high_ivr])
        scores = result.set_index("ticker")["composite_score"]
        assert scores["B"] > scores["A"]

    def test_empty_input_returns_empty(self):
        scorer = CandidateScorer(self.DEFAULT_CONFIG)
        result = scorer.score_candidates([])
        assert result.empty

    def test_result_sorted_descending(self):
        scorer = CandidateScorer(self.DEFAULT_CONFIG)
        candidates = [make_candidate(ticker=f"T{i}", ivr=float(i * 5 + 20)) for i in range(5)]
        result = scorer.score_candidates(candidates)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_weight_normalisation(self):
        """Weights don't need to sum to 1 — scorer normalises them."""
        config = {
            "screener": {
                "score_weights": {
                    "ivr": 3.0,
                    "premium_pct": 2.5,
                    "dividend_yield": 1.5,
                    "pop": 2.0,
                    "liquidity": 1.0,
                }
            }
        }
        scorer = CandidateScorer(config)
        candidates = [make_candidate(ticker="A"), make_candidate(ticker="B", ivr=80.0)]
        result = scorer.score_candidates(candidates)
        assert not result.empty


# ─── Dividends tests ──────────────────────────────────────────────────────────

class TestDividends:
    def test_days_to_ex_dividend_future(self):
        future = date.today().replace(year=date.today().year + 1)
        quote = make_quote(ex_dividend_date=future)
        days = days_to_ex_dividend(quote)
        assert days > 0

    def test_days_to_ex_dividend_none_when_no_date(self):
        quote = make_quote(ex_dividend_date=None)
        assert days_to_ex_dividend(quote) is None

    def test_covered_call_ex_div_risk_low_outside_window(self):
        far_future = date.today().replace(year=date.today().year + 2)
        quote = make_quote(ex_dividend_date=far_future)
        exp = date.today().replace(month=date.today().month + 1 if date.today().month < 12 else 1,
                                    year=date.today().year if date.today().month < 12 else date.today().year + 1)
        risk = covered_call_ex_div_risk(quote, expiration=exp)
        assert risk == "LOW"

    def test_covered_call_ex_div_risk_unknown_when_no_date(self):
        quote = make_quote(ex_dividend_date=None)
        risk = covered_call_ex_div_risk(quote, expiration=date.today())
        assert risk == "UNKNOWN"

    def test_combined_yield_calculation(self):
        result = combined_premium_dividend_yield(
            premium=1.50,
            strike=100.0,
            dividend_yield_pct=3.0,
            dte=45,
        )
        # premium annualised: (1.50/100) * (365/45) * 100 = 12.17%
        expected = (1.50 / 100.0) * (365.0 / 45) * 100 + 3.0
        assert abs(result - expected) < 0.01

    def test_combined_yield_zero_for_invalid_inputs(self):
        assert combined_premium_dividend_yield(1.0, 0.0, 3.0, 45) == 0.0
        assert combined_premium_dividend_yield(1.0, 100.0, 3.0, 0) == 0.0


# ─── display_columns tests ────────────────────────────────────────────────────

class TestDisplayColumns:
    def test_returns_dataframe(self):
        scorer = CandidateScorer({"screener": {"score_weights": {}}})
        ranked = scorer.score_candidates([make_candidate()])
        display = display_columns(ranked)
        assert isinstance(display, pd.DataFrame)

    def test_has_human_readable_columns(self):
        scorer = CandidateScorer({"screener": {"score_weights": {}}})
        ranked = scorer.score_candidates([make_candidate()])
        display = display_columns(ranked)
        assert "Ticker" in display.columns
        assert "Score" in display.columns
