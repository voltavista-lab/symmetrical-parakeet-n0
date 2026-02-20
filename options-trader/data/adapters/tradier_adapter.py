"""Tradier adapter for options data (requires API key)."""

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

TRADIER_LIVE = "https://api.tradier.com/v1"
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"


class TradierAdapter(OptionsDataAdapter):
    """
    Options data via Tradier brokerage API.

    Tradier provides real options chains with Greeks on all account tiers.
    Sandbox mode is available for testing without live data.
    """

    def __init__(self, api_key: str, account_id: str = "", sandbox: bool = False) -> None:
        self._api_key = api_key
        self._account_id = account_id
        self._base = TRADIER_SANDBOX if sandbox else TRADIER_LIVE
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        self._yf = YFinanceAdapter()

    # ─── HTTP helper ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None, retries: int = 3) -> Any:
        url = f"{self._base}{path}"
        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Tradier rate limit, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                logger.warning("Tradier request failed (%s), retrying...", exc)
                time.sleep(2 ** attempt)
        return {}

    # ─── Interface implementation ─────────────────────────────────────────────

    def get_stock_quote(self, ticker: str) -> Optional[StockQuote]:
        """Use yfinance for fundamentals + Tradier for live quote."""
        quote = self._yf.get_stock_quote(ticker)
        if quote is None:
            return None

        try:
            data = self._get("/markets/quotes", {"symbols": ticker, "greeks": "false"})
            quotes = data.get("quotes", {}).get("quote", {})
            if isinstance(quotes, list):
                quotes = next((q for q in quotes if q.get("symbol") == ticker), {})
            if quotes:
                price = quotes.get("last") or quotes.get("bid") or quote.price
                quote.price = float(price or quote.price)
                quote.volume = int(quotes.get("volume", quote.volume) or quote.volume)
        except Exception as exc:
            logger.warning("Tradier quote failed for %s: %s", ticker, exc)

        return quote

    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 30,
        max_dte: int = 60,
    ) -> pd.DataFrame:
        """Fetch options chain with Greeks from Tradier."""
        today = date.today()

        # Get expirations
        try:
            exp_data = self._get("/markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
            expirations_raw = exp_data.get("expirations", {}).get("date", [])
            if isinstance(expirations_raw, str):
                expirations_raw = [expirations_raw]
        except Exception as exc:
            logger.warning("Tradier expirations failed for %s: %s; using yfinance", ticker, exc)
            return self._yf.get_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)

        # Filter to DTE window
        target_exps: list[str] = []
        for exp_str in expirations_raw:
            try:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp - today).days
                if min_dte <= dte <= max_dte:
                    target_exps.append(exp_str)
            except ValueError:
                continue

        if not target_exps and expirations_raw:
            # Take closest to 45 DTE
            target_date = today + timedelta(days=45)
            all_exps = []
            for e in expirations_raw:
                try:
                    all_exps.append((datetime.strptime(e, "%Y-%m-%d").date(), e))
                except ValueError:
                    continue
            if all_exps:
                closest = min(all_exps, key=lambda x: abs((x[0] - target_date).days))
                target_exps = [closest[1]]

        rows: list[dict] = []
        for exp_str in target_exps:
            try:
                chain_data = self._get(
                    "/markets/options/chains",
                    {"symbol": ticker, "expiration": exp_str, "greeks": "true"},
                )
                options = chain_data.get("options", {}).get("option", [])
                if isinstance(options, dict):
                    options = [options]

                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days

                for opt in options:
                    greeks = opt.get("greeks") or {}
                    rows.append(
                        {
                            "ticker": ticker,
                            "expiration": exp_date,
                            "dte": dte,
                            "strike": float(opt.get("strike", 0)),
                            "option_type": opt.get("option_type", "").lower(),
                            "bid": float(opt.get("bid", 0) or 0),
                            "ask": float(opt.get("ask", 0) or 0),
                            "last": float(opt.get("last", 0) or 0),
                            "volume": int(opt.get("volume", 0) or 0),
                            "open_interest": int(opt.get("open_interest", 0) or 0),
                            "implied_volatility": float(opt.get("smv_vol", 0) or 0),
                            "delta": float(greeks.get("delta", float("nan")) or float("nan")),
                            "gamma": float(greeks.get("gamma", float("nan")) or float("nan")),
                            "theta": float(greeks.get("theta", float("nan")) or float("nan")),
                            "vega": float(greeks.get("vega", float("nan")) or float("nan")),
                            "in_the_money": opt.get("in_the_money") == "true",
                        }
                    )
            except Exception as exc:
                logger.warning("Tradier chain error %s %s: %s", ticker, exp_str, exc)

        if not rows:
            logger.warning("No Tradier options data for %s; using yfinance fallback", ticker)
            return self._yf.get_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)

        return pd.DataFrame(rows)

    def get_historical_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV history from Tradier (falls back to yfinance)."""
        try:
            data = self._get(
                "/markets/history",
                {
                    "symbol": ticker,
                    "interval": "daily",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
            history = data.get("history", {}).get("day", [])
            if not history:
                return self._yf.get_historical_prices(ticker, start, end)
            if isinstance(history, dict):
                history = [history]

            df = pd.DataFrame(history)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.set_index("date").rename(
                columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            )
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            return df[["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
        except Exception as exc:
            logger.warning("Tradier history failed for %s: %s; using yfinance", ticker, exc)
            return self._yf.get_historical_prices(ticker, start, end)

    def get_iv_history(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """Delegate to yfinance (Tradier historical IV not in basic tier)."""
        return self._yf.get_iv_history(ticker, lookback_days=lookback_days)
