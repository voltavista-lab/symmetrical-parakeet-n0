"""Polygon.io adapter for options data (requires API key)."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

from .base import OptionsDataAdapter, StockQuote
from .yfinance_adapter import YFinanceAdapter

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


class PolygonAdapter(OptionsDataAdapter):
    """
    Options data via Polygon.io REST API.

    Falls back to yfinance for fundamentals not available in Polygon free tier.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = requests.Session()
        self._session.params = {"apiKey": api_key}  # type: ignore[assignment]
        self._yf = YFinanceAdapter()  # fallback for stock fundamentals

    # ─── HTTP helpers ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None, retries: int = 3) -> Any:
        url = f"{POLYGON_BASE}{path}"
        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Polygon rate limit, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                logger.warning("Polygon request failed (%s), retrying...", exc)
                time.sleep(2 ** attempt)
        return {}

    # ─── Interface implementation ─────────────────────────────────────────────

    def get_stock_quote(self, ticker: str) -> Optional[StockQuote]:
        """Use yfinance for fundamentals; Polygon for current price snapshot."""
        # Get fundamentals from yfinance (more complete for free usage)
        quote = self._yf.get_stock_quote(ticker)
        if quote is None:
            return None

        # Override price with Polygon snapshot if possible
        try:
            data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
            snapshot = data.get("ticker", {})
            day = snapshot.get("day", {})
            price = day.get("c") or snapshot.get("lastTrade", {}).get("p")
            if price:
                quote.price = float(price)
                quote.volume = int(day.get("v", quote.volume) or quote.volume)
        except Exception as exc:
            logger.warning("Polygon snapshot failed for %s: %s", ticker, exc)

        return quote

    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 30,
        max_dte: int = 60,
    ) -> pd.DataFrame:
        """Fetch options chain from Polygon options contracts endpoint."""
        today = date.today()
        exp_from = (today + timedelta(days=min_dte)).isoformat()
        exp_to = (today + timedelta(days=max_dte)).isoformat()

        rows: list[dict] = []
        url = "/v3/snapshot/options/" + ticker
        params: dict = {
            "expiration_date.gte": exp_from,
            "expiration_date.lte": exp_to,
            "limit": 250,
        }

        try:
            while True:
                data = self._get(url, params=params)
                results = data.get("results", [])
                for r in results:
                    details = r.get("details", {})
                    greeks = r.get("greeks", {})
                    day = r.get("day", {})
                    exp_str = details.get("expiration_date", "")
                    if not exp_str:
                        continue
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days

                    rows.append(
                        {
                            "ticker": ticker,
                            "expiration": exp_date,
                            "dte": dte,
                            "strike": float(details.get("strike_price", 0)),
                            "option_type": details.get("contract_type", "").lower(),
                            "bid": float(r.get("bid", 0) or 0),
                            "ask": float(r.get("ask", 0) or 0),
                            "last": float(day.get("close", 0) or r.get("last_quote", {}).get("ask", 0) or 0),
                            "volume": int(day.get("volume", 0) or 0),
                            "open_interest": int(r.get("open_interest", 0) or 0),
                            "implied_volatility": float(r.get("implied_volatility", 0) or 0),
                            "delta": float(greeks.get("delta", float("nan")) or float("nan")),
                            "gamma": float(greeks.get("gamma", float("nan")) or float("nan")),
                            "theta": float(greeks.get("theta", float("nan")) or float("nan")),
                            "vega": float(greeks.get("vega", float("nan")) or float("nan")),
                            "in_the_money": bool(r.get("in_the_money", False)),
                        }
                    )

                next_url = data.get("next_url")
                if not next_url:
                    break
                # Polygon pagination: next_url is full URL; extract path+params
                params = {"cursor": next_url.split("cursor=")[-1]}
                url = "/v3/snapshot/options/" + ticker

        except Exception as exc:
            logger.error("Polygon options chain error for %s: %s", ticker, exc)
            logger.info("Falling back to yfinance for %s", ticker)
            return self._yf.get_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)

        if not rows:
            logger.warning("No Polygon options data for %s; using yfinance fallback", ticker)
            return self._yf.get_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)

        return pd.DataFrame(rows)

    def get_historical_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV history from Polygon aggregates endpoint."""
        try:
            path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
            data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 5000})
            results = data.get("results", [])
            if not results:
                return self._yf.get_historical_prices(ticker, start, end)

            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
            df = df.set_index("date").rename(
                columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
            )
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            return df[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
        except Exception as exc:
            logger.warning("Polygon history failed for %s: %s; using yfinance", ticker, exc)
            return self._yf.get_historical_prices(ticker, start, end)

    def get_iv_history(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """Delegate IV history to yfinance (Polygon historical IV needs Options tier)."""
        return self._yf.get_iv_history(ticker, lookback_days=lookback_days)
