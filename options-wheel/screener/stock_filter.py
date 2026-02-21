"""Stock screener - filters watchlist for wheel-worthy candidates."""

import time
import yfinance as yf


def get_stock_fundamentals(ticker: str) -> dict | None:
    """Fetch fundamental data for a single ticker.

    Returns dict with price, div_yield, market_cap, sector, name
    or None if data cannot be retrieved.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "currentPrice" not in info and "regularMarketPrice" not in info:
            return None

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return None

        div_yield = info.get("dividendYield")
        div_yield_pct = (div_yield * 100) if div_yield else 0.0

        return {
            "ticker": ticker,
            "price": price,
            "dividend_yield": div_yield_pct,
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "name": info.get("shortName", ticker),
            "beta": info.get("beta", 1.0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", price),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", price),
        }
    except Exception:
        return None


def filter_watchlist(
    watchlist: list[str],
    min_div_yield: float = 1.0,
    max_price: float = 200.0,
    delay: float = 0.5,
) -> list[dict]:
    """Filter watchlist to stocks meeting wheel criteria.

    Args:
        watchlist: List of ticker symbols.
        min_div_yield: Minimum dividend yield percentage.
        max_price: Maximum stock price.
        delay: Seconds to wait between API calls (rate limiting).

    Returns:
        List of dicts with fundamentals for qualifying stocks.
    """
    results = []
    for ticker in watchlist:
        fundamentals = get_stock_fundamentals(ticker)
        if fundamentals is None:
            continue
        if fundamentals["price"] > max_price:
            continue
        if fundamentals["dividend_yield"] < min_div_yield:
            continue
        results.append(fundamentals)
        time.sleep(delay)
    return results
