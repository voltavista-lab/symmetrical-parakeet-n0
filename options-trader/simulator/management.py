"""
Trade management logic implementing Tastytrade mechanical rules.

The three management rules implemented here:
1. Profit target  — close when position reaches X% of max profit
2. Roll trigger   — roll when position is down Y% vs. initial credit
3. Hard stop loss — force-close if unrealized loss exceeds Z% of BPR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    ROLL_TRIGGERED = "roll_triggered"
    MAX_DTE_EXCEEDED = "max_dte_exceeded"
    EXPIRATION = "expiration"
    ASSIGNMENT = "assignment"
    MAX_ROLLS_REACHED = "max_rolls_reached"


@dataclass
class Position:
    """Represents an open short option position."""

    ticker: str
    strategy: str              # "short_put" | "covered_call"
    option_type: str           # "put" | "call"
    strike: float
    expiration: date
    entry_date: date
    credit_received: float     # Per-share premium collected (positive = credit)
    buying_power_reduction: float  # Capital required / reserved
    current_price: float = 0.0    # Current mark (cost to close)
    underlying_price: float = 0.0
    roll_count: int = 0
    cumulative_credit: float = 0.0  # Total credit across rolls
    notes: str = ""

    @property
    def dte(self) -> int:
        return max(0, (self.expiration - date.today()).days)

    @property
    def days_in_trade(self) -> int:
        return (date.today() - self.entry_date).days

    @property
    def unrealized_pnl(self) -> float:
        """P&L = credit received - current cost to close. Positive = profit."""
        return self.credit_received - self.current_price

    @property
    def pnl_pct_of_credit(self) -> float:
        """How much of the original credit has been captured (positive = profit)."""
        if self.credit_received == 0:
            return 0.0
        return (self.unrealized_pnl / self.credit_received) * 100

    @property
    def loss_pct_of_bpr(self) -> float:
        """
        Unrealized loss as % of buying power reduction (positive = loss).

        Prices are per-share; BPR is per-contract (×100 shares).
        Multiply per-share loss by 100 to convert to per-contract dollars.
        """
        if self.buying_power_reduction == 0:
            return 0.0
        loss_per_contract = max(0.0, -self.unrealized_pnl) * 100
        return (loss_per_contract / self.buying_power_reduction) * 100


@dataclass
class ManagementConfig:
    """Management rule parameters loaded from config.yaml."""

    profit_target_pct: float = 50.0    # Close at 50% of max profit
    roll_trigger_pct: float = 60.0     # Roll when position is up 60% vs credit
    stop_loss_pct: float = 40.0        # Hard stop at 40% of BPR
    roll_same_strike: bool = True
    roll_max_times: int = 2
    max_dte_in_trade: int = 90

    @classmethod
    def from_config(cls, config: dict) -> "ManagementConfig":
        mgmt = config.get("strategy", {}).get("management", {})
        return cls(
            profit_target_pct=float(mgmt.get("profit_target_pct", 50)),
            roll_trigger_pct=float(mgmt.get("roll_trigger_pct", 60)),
            stop_loss_pct=float(mgmt.get("stop_loss_pct", 40)),
            roll_same_strike=bool(mgmt.get("roll_same_strike", True)),
            roll_max_times=int(mgmt.get("roll_max_times", 2)),
            max_dte_in_trade=int(mgmt.get("max_dte_in_trade", 90)),
        )


class TradeManager:
    """
    Evaluates open positions against management rules daily.

    Tastytrade mechanical rules:
    - Profit target: If sold for $2.00, close when buyback is $1.00 (50% captured)
    - Roll trigger:  If sold for $2.00 and it now costs $3.20 to close → roll (up 60%)
    - Stop loss:     If BPR was $5,000 and unrealized loss hits $2,000 → close (40% of BPR)
    - Max DTE:       Close regardless if DTE exceeds max_dte_in_trade days total
    """

    def __init__(self, mgmt_config: ManagementConfig) -> None:
        self._cfg = mgmt_config

    def check_management(self, position: Position) -> Optional[ExitReason]:
        """
        Evaluate a position against all management rules.

        Returns the ExitReason if the position should be closed/rolled,
        or None if it should be held.

        Rule priority:
        1. Expiration (DTE <= 0)
        2. Max DTE in trade (force-close)
        3. Hard stop loss (immediate exit)
        4. Roll trigger (roll before stop)
        5. Profit target (winner management)
        """
        # 1. Expiration — check ITM (assignment) before returning generic expiration
        if position.dte <= 0:
            if position.option_type == "put" and position.underlying_price < position.strike:
                return ExitReason.ASSIGNMENT
            if position.option_type == "call" and position.underlying_price > position.strike:
                return ExitReason.ASSIGNMENT
            return ExitReason.EXPIRATION

        # 2. Max DTE in trade (total trade age, not contract DTE)
        if position.days_in_trade >= self._cfg.max_dte_in_trade:
            logger.debug(
                "%s max DTE in trade reached (%dd)", position.ticker, position.days_in_trade
            )
            return ExitReason.MAX_DTE_EXCEEDED

        # 3. Hard stop loss — close immediately, no roll
        if position.loss_pct_of_bpr >= self._cfg.stop_loss_pct:
            logger.debug(
                "%s stop loss: %.1f%% of BPR lost (threshold %.1f%%)",
                position.ticker,
                position.loss_pct_of_bpr,
                self._cfg.stop_loss_pct,
            )
            return ExitReason.STOP_LOSS

        # 4. Roll trigger — position is down relative to credit received
        # "down 60%" means current_price = 1.60 × credit_received
        if position.credit_received > 0:
            loss_vs_credit_pct = (
                (position.current_price - position.credit_received)
                / position.credit_received
                * 100
            )
            if loss_vs_credit_pct >= self._cfg.roll_trigger_pct:
                if position.roll_count < self._cfg.roll_max_times:
                    logger.debug(
                        "%s roll trigger: cost to close is %.1f%% above entry credit",
                        position.ticker,
                        loss_vs_credit_pct,
                    )
                    return ExitReason.ROLL_TRIGGERED
                else:
                    logger.debug(
                        "%s max rolls reached (%d), accepting loss",
                        position.ticker,
                        position.roll_count,
                    )
                    return ExitReason.MAX_ROLLS_REACHED

        # 5. Profit target — captured X% of maximum profit
        if position.pnl_pct_of_credit >= self._cfg.profit_target_pct:
            logger.debug(
                "%s profit target hit: %.1f%% of credit captured",
                position.ticker,
                position.pnl_pct_of_credit,
            )
            return ExitReason.PROFIT_TARGET

        return None  # Hold the position

    def calculate_bpr(
        self,
        strike: float,
        premium: float,
        strategy: str,
        underlying_price: float,
    ) -> float:
        """
        Estimate buying power reduction for position sizing.

        Short put (cash-secured): BPR = (strike - premium) × 100
        Short put (portfolio margin): BPR ≈ 20% × strike × 100
        Covered call: BPR = underlying_price × 100 (stock cost)

        We use cash-secured for conservative sizing.
        """
        if strategy == "covered_call":
            return max(0.0, underlying_price * 100)
        # cash-secured short put
        return max(0.0, (strike - premium) * 100)
