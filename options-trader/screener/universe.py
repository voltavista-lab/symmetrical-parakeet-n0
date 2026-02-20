"""Stock universe filtering — first pass to find dividend-paying, liquid stocks."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.adapters import OptionsDataAdapter, StockQuote
from data.cache import get_cache

logger = logging.getLogger(__name__)

# S&P 500 commonly used dividend payers (subset for when full list unavailable)
_SP500_DIVIDEND_SAMPLE = [
    "AAPL", "MSFT", "JNJ", "KO", "PEP", "PG", "MCD", "T", "VZ", "IBM",
    "CVX", "XOM", "JPM", "BAC", "WFC", "HD", "WMT", "TGT", "ABBV", "MRK",
    "PFE", "O", "MAIN", "INTC", "CSCO", "MMM", "GE", "CAT", "MO", "PM",
    "SO", "D", "DUK", "NEE", "ETR", "AEP", "ED", "FE", "PNW", "PCG",
    "WBA", "CVS", "MDT", "ABT", "BMY", "LLY", "AMGN", "GILD", "BIIB",
    "C", "GS", "MS", "BLK", "AXP", "V", "MA", "USB", "PNC", "TFC",
]


def fetch_sp500_tickers() -> list[str]:
    """Attempt to fetch S&P 500 tickers from Wikipedia; fall back to sample list."""
    cache = get_cache()
    cache_key = "sp500_tickers"
    cached = cache.get(cache_key, ttl_minutes=24 * 60)
    if cached:
        return cached

    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        cache.set(cache_key, tickers)
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Could not fetch S&P 500 list (%s); using sample list", exc)
        return _SP500_DIVIDEND_SAMPLE


class UniverseFilter:
    """
    Filters a stock universe down to dividend-paying, liquid candidates
    suitable for premium-selling strategies.
    """

    def __init__(self, config: dict, adapter: OptionsDataAdapter) -> None:
        self._cfg = config
        self._adapter = adapter
        self._screener_cfg = config.get("screener", {})
        self._universe_cfg = config.get("universe", {})
        self._cache = get_cache(config.get("data", {}).get("cache_dir", ".cache"))

    def get_universe(self) -> list[str]:
        """Return list of tickers to screen based on config."""
        mode = self._universe_cfg.get("mode", "watchlist")
        if mode == "sp500":
            return fetch_sp500_tickers()
        elif mode == "russell1000":
            # Russell 1000 fetch is complex; fall back to S&P 500
            logger.info("Russell 1000 not directly available; using S&P 500")
            return fetch_sp500_tickers()
        else:
            # watchlist
            return self._universe_cfg.get("watchlist", _SP500_DIVIDEND_SAMPLE)

    def filter_stocks(
        self,
        tickers: Optional[list[str]] = None,
        min_yield: Optional[float] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volume: Optional[float] = None,
    ) -> list[StockQuote]:
        """
        Apply first-pass filters on the universe.

        Returns a list of StockQuote objects passing all filters.
        """
        if tickers is None:
            tickers = self.get_universe()

        min_yield = min_yield if min_yield is not None else self._screener_cfg.get("min_dividend_yield", 2.0)
        min_price = min_price if min_price is not None else self._screener_cfg.get("min_price", 15.0)
        max_price = max_price if max_price is not None else self._screener_cfg.get("max_price", 200.0)
        min_volume = min_volume if min_volume is not None else self._screener_cfg.get("min_avg_daily_volume", 500_000)

        qualified: list[StockQuote] = []
        ttl_funds = self._cfg.get("data", {}).get("fundamentals_ttl_hours", 24) * 60

        logger.info("Filtering %d tickers from universe...", len(tickers))

        for ticker in tickers:
            cache_key = f"quote_{ticker}"
            quote = self._cache.get(cache_key, ttl_minutes=ttl_funds)
            if quote is None:
                quote = self._adapter.get_stock_quote(ticker)
                if quote is not None:
                    self._cache.set(cache_key, quote)

            if quote is None:
                logger.debug("No data for %s, skipping", ticker)
                continue

            # Price range filter
            if not (min_price <= quote.price <= max_price):
                logger.debug(
                    "%s price $%.2f outside range [%.0f, %.0f]",
                    ticker, quote.price, min_price, max_price,
                )
                continue

            # Volume filter
            if quote.avg_volume < min_volume:
                logger.debug(
                    "%s avg volume %d < %d", ticker, quote.avg_volume, int(min_volume)
                )
                continue

            # Dividend yield filter
            if quote.dividend_yield < min_yield:
                logger.debug(
                    "%s dividend yield %.2f%% < %.1f%%",
                    ticker, quote.dividend_yield, min_yield,
                )
                continue

            logger.debug(
                "%s PASS: price=%.2f, yield=%.2f%%, vol=%d",
                ticker, quote.price, quote.dividend_yield, quote.avg_volume,
            )
            qualified.append(quote)

        logger.info("%d / %d stocks passed universe filters", len(qualified), len(tickers))
        return qualified
