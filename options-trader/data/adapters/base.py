"""Base adapter interface for options data providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd


@dataclass
class OptionContract:
    """Represents a single option contract."""

    ticker: str
    expiration: date
    strike: float
    option_type: str          # "call" or "put"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    in_the_money: bool = False

    @property
    def mid(self) -> float:
        """Mid-price of bid/ask spread."""
        return (self.bid + self.ask) / 2.0

    @property
    def dte(self) -> int:
        """Days to expiration from today."""
        today = date.today()
        return (self.expiration - today).days


@dataclass
class StockQuote:
    """Current quote for an underlying stock."""

    ticker: str
    price: float
    volume: int
    avg_volume: int
    market_cap: float
    sector: str
    industry: str
    dividend_yield: float
    dividend_rate: float
    ex_dividend_date: Optional[date]
    next_earnings_date: Optional[date]
    fifty_two_week_high: float
    fifty_two_week_low: float
    historical_volatility_30d: float = 0.0
    iv_rank: float = 0.0          # Populated by IV calculation


class OptionsDataAdapter(abc.ABC):
    """Abstract base class for all options data providers."""

    @abc.abstractmethod
    def get_stock_quote(self, ticker: str) -> Optional[StockQuote]:
        """Fetch current stock quote and fundamentals."""
        ...

    @abc.abstractmethod
    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 30,
        max_dte: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch options chain for a ticker filtered to DTE window.

        Returns a DataFrame with columns matching OptionContract fields.
        """
        ...

    @abc.abstractmethod
    def get_historical_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price history.

        Returns DataFrame indexed by date with columns:
        Open, High, Low, Close, Volume, Dividends, Stock Splits
        """
        ...

    @abc.abstractmethod
    def get_iv_history(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        Fetch historical implied volatility series.

        Returns a Series indexed by date with IV values (annualized, decimal).
        Falls back to historical volatility proxy if provider doesn't support it.
        """
        ...

    def get_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__
