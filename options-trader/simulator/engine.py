"""
Simulation engine: historical backtesting of premium-selling strategies.

Simulates Tastytrade mechanical short-put and covered-call strategies
using Black-Scholes pricing when live historical options chains are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data.adapters import OptionsDataAdapter
from data.cache import get_cache
from simulator.management import (
    ExitReason,
    ManagementConfig,
    Position,
    TradeManager,
)
from simulator.pricing import (
    estimate_iv_from_hv,
    find_strike_by_delta,
    price_option_on_date,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A completed trade (possibly rolled multiple times)."""

    ticker: str
    strategy: str
    entry_date: date
    exit_date: date
    strike: float
    expiration: date
    dte_at_entry: int
    credit_received: float
    exit_price: float
    pnl: float               # Per share (positive = profit)
    pnl_pct: float           # % of credit received
    exit_reason: str
    roll_number: int = 0
    underlying_entry: float = 0.0
    underlying_exit: float = 0.0
    cumulative_credit: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "strike": self.strike,
            "expiration": self.expiration,
            "dte_at_entry": self.dte_at_entry,
            "credit_received": self.credit_received,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "roll_number": self.roll_number,
            "underlying_entry": self.underlying_entry,
            "underlying_exit": self.underlying_exit,
            "cumulative_credit": self.cumulative_credit,
            "notes": self.notes,
        }


class SimulationEngine:
    """
    Backtests premium-selling strategies over a date range.

    Architecture:
    - Fetches historical prices for the underlying
    - Computes historical volatility as IV proxy
    - Simulates entry/management/exit for each trade mechanically
    - Returns a list of TradeRecord and daily equity curve
    """

    RF_RATE = 0.045  # Approximate risk-free rate used throughout

    def __init__(
        self,
        config: dict,
        adapter: OptionsDataAdapter,
    ) -> None:
        self._cfg = config
        self._adapter = adapter
        self._mgmt_cfg = ManagementConfig.from_config(config)
        self._trade_manager = TradeManager(self._mgmt_cfg)
        self._cache = get_cache(config.get("data", {}).get("cache_dir", ".cache"))

        strategy_cfg = config.get("strategy", {})
        entry_cfg = strategy_cfg.get("entry", {})
        self._target_delta = float(entry_cfg.get("delta", 0.30))
        self._target_dte = int(entry_cfg.get("target_dte", 45))
        self._dte_min = int(entry_cfg.get("dte_window", [30, 60])[0])
        self._dte_max = int(entry_cfg.get("dte_window", [30, 60])[1])
        self._min_ivr = float(entry_cfg.get("min_ivr", 30))

        sizing_cfg = strategy_cfg.get("sizing", {})
        self._portfolio_size = float(sizing_cfg.get("portfolio_size", 50_000))
        self._max_single_pct = float(sizing_cfg.get("max_single_position_pct", 5)) / 100

    # ─── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        ticker: str,
        start: date,
        end: date,
        strategy: str = "short_put",
        mgmt_override: Optional[dict] = None,
    ) -> tuple[list[TradeRecord], pd.Series]:
        """
        Run simulation for a single ticker over the date range.

        Args:
            ticker:          Stock symbol
            start:           Simulation start date
            end:             Simulation end date
            strategy:        "short_put" or "covered_call"
            mgmt_override:   Override management parameters for optimization

        Returns:
            (trades, equity_curve) where equity_curve is a daily P&L Series
        """
        if mgmt_override:
            mgmt_cfg = ManagementConfig(
                profit_target_pct=float(mgmt_override.get("profit_target_pct", self._mgmt_cfg.profit_target_pct)),
                roll_trigger_pct=float(mgmt_override.get("roll_trigger_pct", self._mgmt_cfg.roll_trigger_pct)),
                stop_loss_pct=float(mgmt_override.get("stop_loss_pct", self._mgmt_cfg.stop_loss_pct)),
                roll_same_strike=self._mgmt_cfg.roll_same_strike,
                roll_max_times=self._mgmt_cfg.roll_max_times,
                max_dte_in_trade=self._mgmt_cfg.max_dte_in_trade,
            )
            trade_manager = TradeManager(mgmt_cfg)
        else:
            trade_manager = self._trade_manager

        # Fetch historical data (with extra lookback for HV calculation)
        hx_start = start - timedelta(days=90)
        prices = self._get_price_history(ticker, hx_start, end)
        if prices.empty:
            logger.error("No price history for %s", ticker)
            return [], pd.Series(dtype=float)

        # Calculate rolling 30-day historical volatility (as IV proxy)
        iv_series = self._compute_iv_proxy(prices)

        trades: list[TradeRecord] = []
        equity_curve_data: dict[date, float] = {}
        current_position: Optional[Position] = None
        cumulative_pnl = 0.0

        # Filter prices to simulation range
        sim_dates = [d for d in prices.index if start <= d <= end]

        for sim_date in sim_dates:
            spot = float(prices.loc[sim_date, "Close"])
            dividend = float(prices.loc[sim_date, "Dividends"]) if "Dividends" in prices.columns else 0.0

            # ── Mark current position ─────────────────────────────────────────
            if current_position is not None:
                iv_today = estimate_iv_from_hv(iv_series, sim_date)
                mark = price_option_on_date(
                    S=spot,
                    K=current_position.strike,
                    expiration=current_position.expiration,
                    pricing_date=sim_date,
                    sigma=iv_today,
                    option_type=current_position.option_type,
                    r=self.RF_RATE,
                )
                current_position.current_price = mark
                current_position.underlying_price = spot

                # ── Check management rules ────────────────────────────────────
                exit_reason = trade_manager.check_management(current_position)

                if exit_reason is not None:
                    # Close or roll
                    trade = self._close_position(
                        position=current_position,
                        exit_date=sim_date,
                        exit_price=mark,
                        exit_reason=exit_reason,
                        underlying_exit=spot,
                    )
                    cumulative_pnl += trade.pnl * 100  # per-share → per-contract

                    if exit_reason == ExitReason.ROLL_TRIGGERED:
                        # Open new position at same strike, next cycle
                        current_position = self._open_position(
                            ticker=ticker,
                            strategy=strategy,
                            entry_date=sim_date,
                            spot=spot,
                            iv_series=iv_series,
                            prices=prices,
                            roll_count=current_position.roll_count + 1,
                            cumulative_credit=current_position.cumulative_credit + current_position.credit_received - mark,
                            force_strike=current_position.strike if self._mgmt_cfg.roll_same_strike else None,
                        )
                        if current_position is not None:
                            trade.notes = f"Rolled to new position; roll #{current_position.roll_count}"
                    else:
                        current_position = None

                    trades.append(trade)

            # ── Try to open a new position ────────────────────────────────────
            if current_position is None:
                iv_today = estimate_iv_from_hv(iv_series, sim_date)
                ivr = self._estimate_ivr(iv_series, sim_date)

                if ivr >= self._min_ivr:
                    current_position = self._open_position(
                        ticker=ticker,
                        strategy=strategy,
                        entry_date=sim_date,
                        spot=spot,
                        iv_series=iv_series,
                        prices=prices,
                    )
                else:
                    logger.debug(
                        "%s %s: IVR %.1f < %.1f threshold, no entry",
                        ticker, sim_date, ivr, self._min_ivr,
                    )

            # Daily mark-to-market for equity curve
            daily_position_pnl = (
                current_position.unrealized_pnl * 100 if current_position else 0.0
            )
            equity_curve_data[sim_date] = cumulative_pnl + daily_position_pnl

        # Close any remaining open position at end of simulation
        if current_position is not None:
            final_date = sim_dates[-1] if sim_dates else end
            final_spot = float(prices.loc[final_date, "Close"]) if final_date in prices.index else 0.0
            final_iv = estimate_iv_from_hv(iv_series, final_date)
            final_mark = price_option_on_date(
                S=final_spot,
                K=current_position.strike,
                expiration=current_position.expiration,
                pricing_date=final_date,
                sigma=final_iv,
                option_type=current_position.option_type,
                r=self.RF_RATE,
            )
            trade = self._close_position(
                position=current_position,
                exit_date=final_date,
                exit_price=final_mark,
                exit_reason=ExitReason.EXPIRATION,
                underlying_exit=final_spot,
            )
            cumulative_pnl += trade.pnl * 100
            trades.append(trade)

        equity_curve = pd.Series(equity_curve_data)
        equity_curve.index = pd.to_datetime(equity_curve.index)
        return trades, equity_curve

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _get_price_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch price history with caching."""
        cache_key = f"prices_{ticker}_{start}_{end}"
        cached = self._cache.get(cache_key, ttl_minutes=24 * 60)
        if cached is not None:
            return cached
        prices = self._adapter.get_historical_prices(ticker, start, end)
        if not prices.empty:
            self._cache.set(cache_key, prices)
        return prices

    def _compute_iv_proxy(self, prices: pd.DataFrame) -> pd.Series:
        """Compute rolling 30-day HV as IV proxy (annualised decimal)."""
        closes = prices["Close"]
        log_ret = np.log(closes / closes.shift(1)).dropna()
        hv = log_ret.rolling(window=21).std() * np.sqrt(252)
        hv = hv.dropna()
        # Apply IV premium multiplier (IV typically > HV by ~15%)
        iv_proxy = hv * 1.15
        iv_proxy.index = pd.to_datetime(iv_proxy.index)
        return iv_proxy

    def _estimate_ivr(self, iv_series: pd.Series, current_date: date) -> float:
        """Estimate IV Rank for entry filter."""
        from screener.options_chain import calculate_ivr
        ts = pd.Timestamp(current_date)
        if iv_series.empty:
            return 50.0  # assume mid-range if no data
        try:
            idx = iv_series.index.get_indexer([ts], method="nearest")
            current_iv = float(iv_series.iloc[idx[0]]) * 100  # as percentage
        except Exception:
            current_iv = float(iv_series.iloc[-1]) * 100

        return calculate_ivr(iv_series * 100, current_iv)  # series also as pct

    def _open_position(
        self,
        ticker: str,
        strategy: str,
        entry_date: date,
        spot: float,
        iv_series: pd.Series,
        prices: pd.DataFrame,
        roll_count: int = 0,
        cumulative_credit: float = 0.0,
        force_strike: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Open a new short option position.

        Finds the expiration closest to target_dte, selects strike by delta,
        and prices using Black-Scholes.
        """
        iv = estimate_iv_from_hv(iv_series, entry_date)
        if iv <= 0:
            return None

        # Find target expiration (closest to target_dte in [dte_min, dte_max])
        target_exp = entry_date + timedelta(days=self._target_dte)
        # In backtesting we use the exact target DTE (can't query real chains)
        expiration = target_exp

        dte = (expiration - entry_date).days
        T = dte / 365.0

        option_type = "put" if strategy == "short_put" else "call"

        if force_strike is not None:
            strike = force_strike
        else:
            strike = find_strike_by_delta(
                S=spot,
                T=T,
                sigma=iv,
                target_delta=self._target_delta,
                option_type=option_type,
                r=self.RF_RATE,
            )
            # Round to nearest $1
            strike = round(strike)

        premium = price_option_on_date(
            S=spot,
            K=strike,
            expiration=expiration,
            pricing_date=entry_date,
            sigma=iv,
            option_type=option_type,
            r=self.RF_RATE,
        )

        if premium <= 0:
            return None

        bpr = max(0.0, (strike - premium) * 100)  # cash-secured

        # Position size check
        max_capital = self._portfolio_size * self._max_single_pct
        if bpr > max_capital:
            logger.debug(
                "%s position too large: BPR $%.0f > max $%.0f", ticker, bpr, max_capital
            )
            return None

        return Position(
            ticker=ticker,
            strategy=strategy,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            entry_date=entry_date,
            credit_received=premium,
            buying_power_reduction=bpr,
            current_price=premium,
            underlying_price=spot,
            roll_count=roll_count,
            cumulative_credit=cumulative_credit,
        )

    def _close_position(
        self,
        position: Position,
        exit_date: date,
        exit_price: float,
        exit_reason: ExitReason,
        underlying_exit: float,
    ) -> TradeRecord:
        """Close a position and create a TradeRecord."""
        pnl = position.credit_received - exit_price  # positive = profit
        pnl_pct = (pnl / position.credit_received * 100) if position.credit_received > 0 else 0.0

        return TradeRecord(
            ticker=position.ticker,
            strategy=position.strategy,
            entry_date=position.entry_date,
            exit_date=exit_date,
            strike=position.strike,
            expiration=position.expiration,
            dte_at_entry=(position.expiration - position.entry_date).days,
            credit_received=position.credit_received,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason.value,
            roll_number=position.roll_count,
            underlying_entry=position.underlying_price,
            underlying_exit=underlying_exit,
            cumulative_credit=position.cumulative_credit + position.credit_received,
        )
