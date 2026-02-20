"""Data adapter package — provider-agnostic options data access."""

from .base import OptionContract, OptionsDataAdapter, StockQuote
from .polygon_adapter import PolygonAdapter
from .tradier_adapter import TradierAdapter
from .yfinance_adapter import YFinanceAdapter

__all__ = [
    "OptionsDataAdapter",
    "OptionContract",
    "StockQuote",
    "YFinanceAdapter",
    "PolygonAdapter",
    "TradierAdapter",
]


def get_adapter(config: dict) -> OptionsDataAdapter:
    """
    Factory: return the configured options data adapter.

    Falls back to yfinance if no API key is provided for a premium provider.
    """
    provider = config.get("data", {}).get("options_provider", "yfinance").lower()
    data_cfg = config.get("data", {})

    if provider == "polygon":
        key = data_cfg.get("polygon_api_key", "")
        if key:
            return PolygonAdapter(api_key=key)

    elif provider == "tradier":
        key = data_cfg.get("tradier_api_key", "")
        if key:
            return TradierAdapter(
                api_key=key,
                account_id=data_cfg.get("tradier_account_id", ""),
                sandbox=data_cfg.get("tradier_sandbox", True),
            )

    # Default / fallback
    return YFinanceAdapter()
