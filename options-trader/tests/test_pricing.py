"""Tests for Black-Scholes pricing and option math utilities."""

from __future__ import annotations

import math
from datetime import date

import pytest

# Adjust path so tests can import the package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.pricing import (
    bs_delta,
    bs_price,
    bs_theta,
    bs_vega,
    estimate_iv_from_hv,
    find_strike_by_delta,
    price_option_on_date,
)
from screener.options_chain import (
    bs_delta as screener_bs_delta,
    bs_price as screener_bs_price,
    calculate_ivr,
    find_strike_by_delta as screener_find_strike_by_delta,
    probability_of_profit,
)


# ─── Black-Scholes correctness ────────────────────────────────────────────────

class TestBSPrice:
    """Test Black-Scholes option pricing against known values."""

    def test_atm_put_positive(self):
        """ATM put should have positive price."""
        price = bs_price(S=100, K=100, T=0.25, r=0.045, sigma=0.20, option_type="put")
        assert price > 0

    def test_atm_call_positive(self):
        price = bs_price(S=100, K=100, T=0.25, r=0.045, sigma=0.20, option_type="call")
        assert price > 0

    def test_deep_itm_put(self):
        """Deep ITM put should be close to intrinsic value."""
        price = bs_price(S=80, K=100, T=0.001, r=0.0, sigma=0.20, option_type="put")
        intrinsic = 100 - 80
        assert abs(price - intrinsic) < 1.0

    def test_deep_otm_put_near_zero(self):
        """Deep OTM put should be near zero."""
        price = bs_price(S=200, K=100, T=0.25, r=0.045, sigma=0.20, option_type="put")
        assert price < 0.01

    def test_expired_option_intrinsic(self):
        """Expired option should return intrinsic value."""
        put_price = bs_price(S=90, K=100, T=0.0, r=0.045, sigma=0.20, option_type="put")
        assert abs(put_price - 10.0) < 1e-6

        call_price = bs_price(S=110, K=100, T=0.0, r=0.045, sigma=0.20, option_type="call")
        assert abs(call_price - 10.0) < 1e-6

    def test_zero_sigma_returns_zero(self):
        """Zero volatility → zero extrinsic value."""
        price = bs_price(S=100, K=110, T=0.25, r=0.045, sigma=0.0, option_type="put")
        assert price == 0.0

    def test_put_call_parity(self):
        """P - C = K*e^(-rT) - S (put-call parity)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.045, 0.20
        put = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")
        call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
        parity_rhs = K * math.exp(-r * T) - S
        assert abs((put - call) - parity_rhs) < 0.01

    def test_known_approximate_value(self):
        """
        ATM put, S=100, K=100, T=1yr, r=5%, sigma=20%.
        BS: d1=0.35, d2=0.15; put ≈ K*e^(-rT)*N(-d2) - S*N(-d1) ≈ 5.57.
        """
        price = bs_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert 5.0 < price < 7.0


class TestBSDelta:
    def test_atm_put_delta_near_minus_half(self):
        # With r=0.045, ATM put delta is shifted slightly from -0.5 due to carry
        delta = bs_delta(S=100, K=100, T=0.25, r=0.045, sigma=0.20, option_type="put")
        assert -0.60 < delta < -0.40

    def test_atm_call_delta_near_half(self):
        # Symmetric: call delta = 1 + put delta (not exactly 0.5 with non-zero r)
        delta = bs_delta(S=100, K=100, T=0.25, r=0.045, sigma=0.20, option_type="call")
        assert 0.40 < delta < 0.60

    def test_deep_itm_put_delta_near_minus_one(self):
        delta = bs_delta(S=60, K=100, T=0.25, r=0.045, sigma=0.20, option_type="put")
        assert delta < -0.90

    def test_deep_otm_put_delta_near_zero(self):
        delta = bs_delta(S=140, K=100, T=0.25, r=0.045, sigma=0.20, option_type="put")
        assert -0.10 < delta < 0.0


class TestFindStrikeByDelta:
    def test_30_delta_put_below_spot(self):
        """30-delta put strike should be below current spot."""
        strike = find_strike_by_delta(S=100, T=0.25, sigma=0.20, target_delta=0.30, option_type="put")
        assert strike < 100

    def test_30_delta_call_above_spot(self):
        """30-delta call strike should be above current spot."""
        strike = find_strike_by_delta(S=100, T=0.25, sigma=0.20, target_delta=0.30, option_type="call")
        assert strike > 100

    def test_round_trip_delta(self):
        """Found strike should produce delta close to target."""
        target = 0.30
        strike = find_strike_by_delta(S=100, T=0.25, sigma=0.20, target_delta=target, option_type="put")
        actual_delta = abs(bs_delta(S=100, K=strike, T=0.25, r=0.045, sigma=0.20, option_type="put"))
        assert abs(actual_delta - target) < 0.05  # within 5 delta points


class TestProbabilityOfProfit:
    def test_pop_increases_with_otm_distance(self):
        """Further OTM strikes should have higher POP."""
        pop_near = probability_of_profit(S=100, K=95, T=0.25, sigma=0.20, option_type="put", credit=1.0)
        pop_far = probability_of_profit(S=100, K=80, T=0.25, sigma=0.20, option_type="put", credit=1.0)
        assert pop_far > pop_near

    def test_pop_between_0_and_1(self):
        pop = probability_of_profit(S=100, K=90, T=0.25, sigma=0.20, option_type="put", credit=1.0)
        assert 0.0 <= pop <= 1.0


class TestCalculateIVR:
    def test_ivr_at_52w_high_is_100(self):
        import pandas as pd
        series = pd.Series([20.0, 25.0, 30.0, 35.0, 40.0])
        ivr = calculate_ivr(series, current_iv=40.0)
        assert ivr == 100.0

    def test_ivr_at_52w_low_is_0(self):
        import pandas as pd
        series = pd.Series([20.0, 25.0, 30.0, 35.0, 40.0])
        ivr = calculate_ivr(series, current_iv=20.0)
        assert ivr == 0.0

    def test_ivr_at_midpoint_is_50(self):
        import pandas as pd
        # Series must have at least 5 elements to pass the guard check
        series = pd.Series([20.0, 22.0, 28.0, 35.0, 40.0])
        ivr = calculate_ivr(series, current_iv=30.0)
        assert ivr == 50.0

    def test_ivr_empty_series_returns_zero(self):
        import pandas as pd
        ivr = calculate_ivr(pd.Series([], dtype=float), current_iv=25.0)
        assert ivr == 0.0


class TestPriceOptionOnDate:
    def test_basic_pricing(self):
        price = price_option_on_date(
            S=100,
            K=95,
            expiration=date(2025, 6, 1),
            pricing_date=date(2025, 1, 1),
            sigma=0.25,
            option_type="put",
        )
        assert price > 0

    def test_pricing_at_expiration(self):
        """At expiration, OTM put should be near zero."""
        price = price_option_on_date(
            S=100,
            K=90,
            expiration=date(2025, 6, 1),
            pricing_date=date(2025, 6, 1),
            sigma=0.25,
            option_type="put",
        )
        assert price < 0.01
