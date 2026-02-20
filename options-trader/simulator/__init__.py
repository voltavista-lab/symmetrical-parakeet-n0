"""Simulator package: backtesting engine, pricing, trade management, and analytics."""

from .analytics import compute_stats, format_trade_log, format_stats_table
from .engine import SimulationEngine, TradeRecord
from .management import ExitReason, ManagementConfig, Position, TradeManager
from .pricing import bs_price, bs_delta, find_strike_by_delta, price_option_on_date

__all__ = [
    "SimulationEngine",
    "TradeRecord",
    "TradeManager",
    "ManagementConfig",
    "Position",
    "ExitReason",
    "bs_price",
    "bs_delta",
    "find_strike_by_delta",
    "price_option_on_date",
    "compute_stats",
    "format_trade_log",
    "format_stats_table",
]
