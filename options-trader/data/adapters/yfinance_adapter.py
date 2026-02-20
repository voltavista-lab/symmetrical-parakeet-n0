"""yfinance adapter — free, no API key required. Used as default and fallback."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .base import OptionContract, OptionsDataAdapter, StockQuote

logger = logging.getLogger(__name__)


class YFinanceAdapter(OptionsDataAdapter):
    """Options data via yfinance (free, no API key required)."""

    # Sector string normalisation map
    _SECTOR_MAP: dict[str, str] = {
        "Technology": "Technology",
        "Financial Services": "Financials",
        "Financials": "Financials",
        "Healthcare": "Healthcare",
        "Consumer Defensive": "Consumer Staples",
        "Consumer Cyclical": "Consumer Discretionary",
        "Communication Services": "Communication Services",
        "Energy": "Energy",
        "Utilities": "Utilities",
        "Real Estate": "Real Estate",
        "Basic Materials": "Materials",
        "Industrials": "Industrials",
    }

    def get_stock_quote(self, ticker: str) -> Optional[StockQuote]:
        """Fetch current stock quote and fundamentals from yfinance."""
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            price = float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("navPrice")
                or 0.0
            )
            if price == 0.0:
                # Fallback: last close from recent history
                hist = tk.history(period="2d")
                if hist.empty:
                    logger.warning("No price data for %s", ticker)
                    return None
                price = float(hist["Close"].iloc[-1])

            # Ex-dividend date
            ex_div_date: Optional[date] = None
            raw_exdiv = info.get("exDividendDate")
            if raw_exdiv:
                try:
                    ex_div_date = datetime.fromtimestamp(raw_exdiv).date()
                except (OSError, OverflowError, ValueError):
                    try:
                        ex_div_date = pd.to_datetime(raw_exdiv).date()
                    except Exception:
                        pass

            # Next earnings date
            earnings_date: Optional[date] = None
            try:
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    if "Earnings Date" in cal.index:
                        ed_val = cal.loc["Earnings Date"]
                        if isinstance(ed_val, pd.Series):
                            ed_val = ed_val.iloc[0]
                        earnings_date = pd.to_datetime(ed_val).date()
            except Exception:
                pass

            raw_sector = info.get("sector", "Unknown")
            sector = self._SECTOR_MAP.get(raw_sector, raw_sector or "Unknown")

            hv30 = self._calculate_hv(ticker, window=30)

            return StockQuote(
                ticker=ticker,
                price=price,
                volume=int(info.get("volume", 0) or 0),
                avg_volume=int(info.get("averageVolume", 0) or 0),
                market_cap=float(info.get("marketCap", 0) or 0),
                sector=sector,
                industry=info.get("industry", "Unknown") or "Unknown",
                dividend_yield=float(info.get("dividendYield", 0.0) or 0.0) * 100,  # pct
                dividend_rate=float(info.get("dividendRate", 0.0) or 0.0),
                ex_dividend_date=ex_div_date,
                next_earnings_date=earnings_date,
                fifty_two_week_high=float(info.get("fiftyTwoWeekHigh", price) or price),
                fifty_two_week_low=float(info.get("fiftyTwoWeekLow", price) or price),
                historical_volatility_30d=hv30,
            )
        except Exception as exc:
            logger.error("Error fetching quote for %s: %s", ticker, exc)
            return None

    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 30,
        max_dte: int = 60,
    ) -> pd.DataFrame:
        """Fetch options chain filtered to the target DTE window."""
        try:
            tk = yf.Ticker(ticker)
            today = date.today()

            # Filter expirations within our DTE window
            target_expirations: list[str] = []
            for exp_str in tk.options:
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp - today).days
                if min_dte <= dte <= max_dte:
                    target_expirations.append(exp_str)

            if not target_expirations:
                logger.warning(
                    "No expirations in %d–%d DTE window for %s", min_dte, max_dte, ticker
                )
                # Broaden search and take the closest to target_dte (45)
                all_exps = [
                    (datetime.strptime(e, "%Y-%m-%d").date(), e)
                    for e in tk.options
                ]
                if not all_exps:
                    return pd.DataFrame()
                target_date = today + timedelta(days=45)
                closest = min(all_exps, key=lambda x: abs((x[0] - target_date).days))
                target_expirations = [closest[1]]

            rows: list[dict] = []
            for exp_str in target_expirations:
                try:
                    chain = tk.option_chain(exp_str)
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days

                    for opt_type, df in [("put", chain.puts), ("call", chain.calls)]:
                        for _, row in df.iterrows():
                            iv = float(row.get("impliedVolatility", 0.0) or 0.0)
                            rows.append(
                                {
                                    "ticker": ticker,
                                    "expiration": exp_date,
                                    "dte": dte,
                                    "strike": float(row["strike"]),
                                    "option_type": opt_type,
                                    "bid": float(row.get("bid", 0.0) or 0.0),
                                    "ask": float(row.get("ask", 0.0) or 0.0),
                                    "last": float(row.get("lastPrice", 0.0) or 0.0),
                                    "volume": int(row.get("volume", 0) or 0),
                                    "open_interest": int(row.get("openInterest", 0) or 0),
                                    "implied_volatility": iv,
                                    "delta": float(row.get("delta", np.nan) or np.nan),
                                    "gamma": float(row.get("gamma", np.nan) or np.nan),
                                    "theta": float(row.get("theta", np.nan) or np.nan),
                                    "vega": float(row.get("vega", np.nan) or np.nan),
                                    "in_the_money": bool(row.get("inTheMoney", False)),
                                }
                            )
                except Exception as exc:
                    logger.warning("Error fetching chain %s %s: %s", ticker, exp_str, exc)

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame(rows)

        except Exception as exc:
            logger.error("Error fetching options chain for %s: %s", ticker, exc)
            return pd.DataFrame()

    def get_historical_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV + dividends history."""
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
            )
            hist.index = pd.to_datetime(hist.index).date
            return hist
        except Exception as exc:
            logger.error("Error fetching history for %s: %s", ticker, exc)
            return pd.DataFrame()

    def get_iv_history(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        Proxy for IV history using at-the-money options IV over time.

        yfinance doesn't provide historical IV directly, so we compute
        historical volatility as a proxy for IVR calculations.
        """
        try:
            end = date.today()
            start = end - timedelta(days=lookback_days + 60)
            hist = self.get_historical_prices(ticker, start, end)
            if hist.empty:
                return pd.Series(dtype=float)

            # Compute rolling 30-day realised vol as IV proxy
            closes = hist["Close"]
            log_returns = np.log(closes / closes.shift(1)).dropna()
            rolling_vol = log_returns.rolling(window=21).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()

            # Convert index to datetime for consistent handling
            rolling_vol.index = pd.to_datetime(rolling_vol.index)
            return rolling_vol

        except Exception as exc:
            logger.error("Error computing IV history for %s: %s", ticker, exc)
            return pd.Series(dtype=float)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _calculate_hv(self, ticker: str, window: int = 30) -> float:
        """Calculate annualised historical volatility over `window` trading days."""
        try:
            end = date.today()
            start = end - timedelta(days=window * 2 + 10)
            hist = self.get_historical_prices(ticker, start, end)
            if hist.empty or len(hist) < window:
                return 0.0
            closes = hist["Close"].tail(window + 1)
            log_returns = np.log(closes / closes.shift(1)).dropna()
            hv = float(log_returns.std() * np.sqrt(252) * 100)  # as percentage
            return round(hv, 2)
        except Exception:
            return 0.0
