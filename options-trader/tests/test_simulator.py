"""Tests for simulation engine, trade management, and analytics."""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.management import (
    ExitReason,
    ManagementConfig,
    Position,
    TradeManager,
)
from simulator.analytics import (
    compute_stats,
    format_trade_log,
    format_stats_table,
    compare_parameter_sets,
)
from simulator.engine import TradeRecord


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_position(
    ticker: str = "TEST",
    strategy: str = "short_put",
    strike: float = 95.0,
    entry_date: date | None = None,
    expiration_days: int = 45,
    credit_received: float = 2.00,
    current_price: float = 1.00,
    underlying_price: float = 100.0,
    buying_power_reduction: float = 9_300.0,
    roll_count: int = 0,
) -> Position:
    today = entry_date or date.today()
    expiration = today + timedelta(days=expiration_days)
    return Position(
        ticker=ticker,
        strategy=strategy,
        option_type="put",
        strike=strike,
        expiration=expiration,
        entry_date=today,
        credit_received=credit_received,
        buying_power_reduction=buying_power_reduction,
        current_price=current_price,
        underlying_price=underlying_price,
        roll_count=roll_count,
    )


def make_trade_record(
    ticker: str = "TEST",
    strategy: str = "short_put",
    pnl: float = 1.00,
    pnl_pct: float = 50.0,
    credit_received: float = 2.00,
    exit_reason: str = "profit_target",
    entry_date: date | None = None,
    exit_date: date | None = None,
    roll_number: int = 0,
) -> dict:
    today = entry_date or date.today() - timedelta(days=30)
    ex = exit_date or date.today()
    return {
        "ticker": ticker,
        "strategy": strategy,
        "entry_date": today,
        "exit_date": ex,
        "strike": 95.0,
        "expiration": ex + timedelta(days=10),
        "dte_at_entry": 45,
        "credit_received": credit_received,
        "exit_price": credit_received - pnl,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "roll_number": roll_number,
        "underlying_entry": 100.0,
        "underlying_exit": 103.0,
        "cumulative_credit": credit_received,
    }


DEFAULT_MGMT = ManagementConfig(
    profit_target_pct=50.0,
    roll_trigger_pct=60.0,
    stop_loss_pct=40.0,
    roll_max_times=2,
    max_dte_in_trade=90,
)


# ─── ManagementConfig tests ───────────────────────────────────────────────────

class TestManagementConfig:
    def test_from_config_defaults(self):
        cfg = ManagementConfig.from_config({})
        assert cfg.profit_target_pct == 50.0
        assert cfg.roll_trigger_pct == 60.0
        assert cfg.stop_loss_pct == 40.0

    def test_from_config_override(self):
        cfg = ManagementConfig.from_config(
            {"strategy": {"management": {"profit_target_pct": 55.0, "stop_loss_pct": 35.0}}}
        )
        assert cfg.profit_target_pct == 55.0
        assert cfg.stop_loss_pct == 35.0


# ─── Position tests ───────────────────────────────────────────────────────────

class TestPosition:
    def test_unrealized_pnl_profit(self):
        """Position should show profit when current mark < credit received."""
        pos = make_position(credit_received=2.00, current_price=1.00)
        assert pos.unrealized_pnl == pytest.approx(1.00)

    def test_unrealized_pnl_loss(self):
        pos = make_position(credit_received=2.00, current_price=3.00)
        assert pos.unrealized_pnl == pytest.approx(-1.00)

    def test_pnl_pct_at_50_profit(self):
        pos = make_position(credit_received=2.00, current_price=1.00)
        assert pos.pnl_pct_of_credit == pytest.approx(50.0)

    def test_loss_pct_of_bpr(self):
        """Loss of $1/share on $9300 BPR should compute correctly."""
        pos = make_position(credit_received=2.00, current_price=3.00, buying_power_reduction=9_300.0)
        # Unrealized loss = $1.00/share → $100/contract. But BPR is in dollars.
        # loss_pct_of_bpr = (-unrealized_pnl / bpr) * 100 = (1/9300)*100 ≈ 1.08%
        assert pos.loss_pct_of_bpr > 0

    def test_loss_pct_of_bpr_zero_no_loss(self):
        pos = make_position(credit_received=2.00, current_price=1.00)
        assert pos.loss_pct_of_bpr == 0.0


# ─── TradeManager tests ───────────────────────────────────────────────────────

class TestTradeManager:
    def setup_method(self):
        self.manager = TradeManager(DEFAULT_MGMT)

    def test_profit_target_hit(self):
        """Position at 50% profit should trigger profit_target."""
        pos = make_position(credit_received=2.00, current_price=1.00)  # 50% profit
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.PROFIT_TARGET

    def test_no_action_when_holding(self):
        """Position at 30% profit, no rules triggered."""
        pos = make_position(credit_received=2.00, current_price=1.50)  # 25% profit
        reason = self.manager.check_management(pos)
        assert reason is None

    def test_roll_trigger_hit(self):
        """Position where cost to close is 60% above credit → roll."""
        # Sold for $2.00, now costs $3.20 to close (60% above $2.00)
        pos = make_position(credit_received=2.00, current_price=3.20)
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.ROLL_TRIGGERED

    def test_max_rolls_reached_triggers_close(self):
        """After max rolls hit, roll_triggered becomes max_rolls_reached."""
        pos = make_position(credit_received=2.00, current_price=3.20, roll_count=2)
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.MAX_ROLLS_REACHED

    def test_stop_loss_hit_before_roll(self):
        """
        Stop loss has priority over roll trigger.

        BPR = $9300 (per contract = 100 shares), stop_loss_pct = 40%.
        Loss threshold = 40% × $9300 = $3720 per contract = $37.20/share.
        With credit = $2.00/share, position must cost $39.20 to close.

        loss_pct_of_bpr = (loss_per_share × 100 / bpr) × 100
                        = (37.20 × 100 / 9300) × 100 = 40% → STOP_LOSS fires.

        Roll trigger also fires (current >> 1.6×credit), but stop loss has
        higher priority (checked first in check_management).
        """
        bpr = 9_300.0
        credit = 2.00
        # 40% of BPR = $3720/contract = $37.20/share loss
        per_share_loss = DEFAULT_MGMT.stop_loss_pct / 100.0 * bpr / 100
        current_price_for_stop = credit + per_share_loss  # = 2.00 + 37.20 = 39.20
        pos = make_position(
            credit_received=credit,
            current_price=current_price_for_stop,
            buying_power_reduction=bpr,
        )
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.STOP_LOSS

    def test_expiration_otm(self):
        """Expired OTM put should return EXPIRATION."""
        pos = make_position(credit_received=2.00, current_price=0.01, expiration_days=0)
        pos.underlying_price = 100.0  # above strike 95
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.EXPIRATION

    def test_expiration_itm_assignment(self):
        """Expired ITM put should trigger ASSIGNMENT."""
        pos = make_position(credit_received=2.00, current_price=5.00, expiration_days=0)
        pos.underlying_price = 89.0  # below strike 95
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.ASSIGNMENT

    def test_max_dte_exceeded(self):
        """Position held too long should be force-closed (expiration must be in future)."""
        # entry 91 days ago, expiration 150 days from entry (so still 59 days away)
        old_entry = date.today() - timedelta(days=91)
        pos = make_position(
            credit_received=2.00,
            current_price=1.50,
            entry_date=old_entry,
            expiration_days=150,  # expiration = entry + 150 = today + 59 → not expired
        )
        reason = self.manager.check_management(pos)
        assert reason == ExitReason.MAX_DTE_EXCEEDED

    def test_calculate_bpr_short_put(self):
        bpr = self.manager.calculate_bpr(
            strike=95.0, premium=2.0, strategy="short_put", underlying_price=100.0
        )
        # Cash-secured: (95 - 2) * 100 = 9300
        assert bpr == pytest.approx(9_300.0)

    def test_calculate_bpr_covered_call(self):
        bpr = self.manager.calculate_bpr(
            strike=105.0, premium=2.0, strategy="covered_call", underlying_price=100.0
        )
        # Stock cost: 100 * 100 = 10,000
        assert bpr == pytest.approx(10_000.0)


# ─── Analytics tests ──────────────────────────────────────────────────────────

class TestComputeStats:
    def _make_equity(self, values: list[float]) -> pd.Series:
        dates = pd.date_range("2023-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=dates)

    def test_empty_trades_returns_zeros(self):
        stats = compute_stats([], pd.Series(dtype=float))
        assert stats["num_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_win_rate_calculation(self):
        trades = [
            make_trade_record(pnl=1.0),   # win
            make_trade_record(pnl=-0.5),  # loss
            make_trade_record(pnl=0.5),   # win
            make_trade_record(pnl=0.1),   # win
        ]
        equity = self._make_equity([100, 50, 150, 200])
        stats = compute_stats(trades, equity)
        assert stats["win_rate"] == pytest.approx(75.0)
        assert stats["num_trades"] == 4

    def test_total_pnl_uses_equity_endpoint(self):
        trades = [make_trade_record(pnl=1.0)]
        equity = self._make_equity([0, 50, 100, 200])
        stats = compute_stats(trades, equity)
        assert stats["total_pnl"] == pytest.approx(200.0)

    def test_max_drawdown_is_negative(self):
        trades = [make_trade_record()]
        equity = self._make_equity([0, 200, 100, 300, 50, 400])
        stats = compute_stats(trades, equity)
        assert stats["max_drawdown"] < 0


class TestFormatTradeLog:
    def test_returns_dataframe(self):
        trades = [make_trade_record()]
        df = format_trade_log(trades)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_empty_returns_empty(self):
        df = format_trade_log([])
        assert df.empty


class TestFormatStatsTable:
    def test_returns_dataframe_with_two_columns(self):
        stats = {
            "win_rate": 67.0, "num_trades": 10, "total_pnl": 1500.0,
            "annualized_return": 12.5, "max_drawdown": -300.0,
            "avg_days_in_trade": 28.0, "sharpe_ratio": 1.2, "num_rolls": 2,
            "avg_win_pct": 50.0, "avg_loss_pct": -80.0, "exit_reasons": {},
        }
        df = format_stats_table(stats, "AAPL", "short_put")
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) > 5


class TestCompareParameterSets:
    def test_returns_dataframe(self):
        results = [
            {"label": "PT=50%", "profit_target_pct": 50, "stop_loss_pct": 40, "roll_trigger_pct": 60,
             "stats": {"win_rate": 65, "total_pnl": 1000, "annualized_return": 10, "max_drawdown": -200, "sharpe_ratio": 1.1, "num_trades": 8, "num_rolls": 2}},
            {"label": "PT=55%", "profit_target_pct": 55, "stop_loss_pct": 40, "roll_trigger_pct": 60,
             "stats": {"win_rate": 70, "total_pnl": 1200, "annualized_return": 12, "max_drawdown": -150, "sharpe_ratio": 1.3, "num_trades": 7, "num_rolls": 1}},
        ]
        df = compare_parameter_sets(results)
        assert not df.empty
        assert "Win Rate" in df.columns
        assert len(df) == 2
