"""Historical simulation engine for the options wheel strategy.

LIMITATIONS:
- Uses historical stock prices, NOT real historical options data.
- Premiums are estimated from historical volatility and Black-Scholes approximation.
- Delta is approximated; real delta would shift intraday.
- Assignment is simplified: assigned if price <= strike at expiry (puts) or >= strike (calls).
- This is an APPROXIMATION for parameter validation, not exact P&L recreation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def _estimate_premium(stock_price: float, strike: float, dte: int,
                      hv: float, option_type: str, risk_free: float = 0.05) -> float:
    """Estimate option premium using Black-Scholes."""
    if dte <= 0 or hv <= 0 or stock_price <= 0 or strike <= 0:
        return 0.0
    t = dte / 365.0
    d1 = (np.log(stock_price / strike) + (risk_free + 0.5 * hv**2) * t) / (hv * np.sqrt(t))
    d2 = d1 - hv * np.sqrt(t)

    if option_type == "put":
        price = strike * np.exp(-risk_free * t) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    else:
        price = stock_price * norm.cdf(d1) - strike * np.exp(-risk_free * t) * norm.cdf(d2)
    return max(float(price), 0.01)


def _compute_hv(prices: pd.Series, window: int = 21) -> float:
    """Compute annualized historical volatility from a price series."""
    if len(prices) < window + 1:
        return 0.25  # default fallback
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        return 0.25
    hv = returns.iloc[-window:].std() * np.sqrt(252)
    return max(float(hv), 0.05)


def simulate_wheel(ticker: str, start_date: str, end_date: str,
                   strategy_params: dict) -> dict:
    """Simulate the wheel strategy on historical data.

    Steps:
    1. Start in "sell put" mode.
    2. Sell a put at ~target delta OTM.
    3. Each day, check profit target / stop loss / assignment.
    4. If assigned, switch to "covered call" mode.
    5. If called away, switch back to "sell put" mode.

    Returns dict with detailed results.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    if hist.empty or len(hist) < 30:
        return _empty_results(ticker, start_date, end_date)

    prices = hist["Close"]
    dates = hist.index

    target_dte = strategy_params["target_dte"]
    profit_target_pct = strategy_params["profit_target_pct"]
    max_loss_pct = strategy_params["max_loss_close_pct"]
    delta_target_put = abs(strategy_params["delta_target_put"])
    delta_target_call = strategy_params["delta_target_call"]

    # State tracking
    mode = "sell_put"  # or "covered_call"
    trades = []
    current_trade = None
    total_premium = 0.0
    assignments = 0
    rolls = 0
    cost_basis = 0.0  # for covered call mode

    i = 0
    while i < len(prices):
        price = float(prices.iloc[i])
        date = dates[i]

        if current_trade is None:
            # Open new position
            hv = _compute_hv(prices.iloc[max(0, i - 60):i + 1])
            if mode == "sell_put":
                strike = round(price * (1 - delta_target_put), 2)
                premium = _estimate_premium(price, strike, target_dte, hv, "put")
                current_trade = {
                    "type": "put", "open_date": date, "strike": strike,
                    "entry_premium": premium, "dte_remaining": target_dte,
                    "stock_price_at_open": price,
                }
            else:  # covered_call
                strike = round(price * (1 + delta_target_call), 2)
                premium = _estimate_premium(price, strike, target_dte, hv, "call")
                current_trade = {
                    "type": "call", "open_date": date, "strike": strike,
                    "entry_premium": premium, "dte_remaining": target_dte,
                    "stock_price_at_open": price,
                }
            total_premium += premium
            i += 1
            continue

        # Manage existing position
        current_trade["dte_remaining"] -= 1
        dte_left = current_trade["dte_remaining"]
        entry_premium = current_trade["entry_premium"]
        strike = current_trade["strike"]

        # Estimate current option value (simplified decay)
        hv = _compute_hv(prices.iloc[max(0, i - 60):i + 1])
        if dte_left > 0:
            current_value = _estimate_premium(
                price, strike, dte_left, hv, current_trade["type"]
            )
        else:
            # At expiration
            if current_trade["type"] == "put":
                current_value = max(strike - price, 0)
            else:
                current_value = max(price - strike, 0)

        profit_pct = (entry_premium - current_value) / entry_premium * 100 if entry_premium > 0 else 0
        loss_pct = (current_value - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

        closed = False
        close_reason = ""

        # Check profit target
        if profit_pct >= profit_target_pct:
            closed = True
            close_reason = "profit_target"

        # Check stop loss
        elif loss_pct >= max_loss_pct:
            closed = True
            close_reason = "stop_loss"

        # Check expiration
        elif dte_left <= 0:
            if current_trade["type"] == "put" and price <= strike:
                closed = True
                close_reason = "assigned"
                assignments += 1
                cost_basis = strike - entry_premium
                mode = "covered_call"
            elif current_trade["type"] == "call" and price >= strike:
                closed = True
                close_reason = "called_away"
                mode = "sell_put"
            else:
                closed = True
                close_reason = "expired_otm"

        if closed:
            pnl = (entry_premium - current_value) * 100
            current_trade.update({
                "close_date": date,
                "close_reason": close_reason,
                "close_premium": current_value,
                "pnl": round(pnl, 2),
                "stock_price_at_close": price,
            })
            trades.append(current_trade)
            current_trade = None

        i += 1

    # Close any open trade at end
    if current_trade is not None:
        price = float(prices.iloc[-1])
        hv = _compute_hv(prices.iloc[-60:])
        dte_left = max(current_trade["dte_remaining"], 1)
        current_value = _estimate_premium(
            price, current_trade["strike"], dte_left, hv, current_trade["type"]
        )
        pnl = (current_trade["entry_premium"] - current_value) * 100
        current_trade.update({
            "close_date": dates[-1],
            "close_reason": "end_of_period",
            "close_premium": current_value,
            "pnl": round(pnl, 2),
            "stock_price_at_close": price,
        })
        trades.append(current_trade)

    return _compile_results(ticker, start_date, end_date, trades,
                            total_premium, assignments, prices)


def _empty_results(ticker: str, start: str, end: str) -> dict:
    return {
        "ticker": ticker, "start_date": start, "end_date": end,
        "trades": [], "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_premium_collected": 0, "total_pnl": 0,
        "avg_pnl": 0, "max_drawdown": 0, "assignments": 0,
        "buy_hold_return": 0, "strategy_return": 0,
        "sharpe_ratio": 0, "monthly_pnl": {},
    }


def _compile_results(ticker: str, start: str, end: str, trades: list,
                     total_premium: float, assignments: int,
                     prices: pd.Series) -> dict:
    if not trades:
        return _empty_results(ticker, start, end)

    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p >= 0)
    losses = sum(1 for p in pnls if p < 0)
    total_pnl = sum(pnls)

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Buy and hold return
    start_price = float(prices.iloc[0])
    end_price = float(prices.iloc[-1])
    buy_hold_pct = (end_price - start_price) / start_price * 100

    # Strategy return (premium-based, approximate)
    initial_capital = start_price * 100  # 1 contract worth
    strategy_pct = total_pnl / initial_capital * 100 if initial_capital > 0 else 0

    # Sharpe-like ratio
    if len(pnls) > 1:
        pnl_std = np.std(pnls)
        sharpe = (np.mean(pnls) / pnl_std * np.sqrt(12)) if pnl_std > 0 else 0
    else:
        sharpe = 0

    # Monthly PnL
    monthly = {}
    for t in trades:
        close = t.get("close_date")
        if hasattr(close, "strftime"):
            key = close.strftime("%Y-%m")
        else:
            key = str(close)[:7]
        monthly[key] = monthly.get(key, 0) + t["pnl"]

    return {
        "ticker": ticker, "start_date": start, "end_date": end,
        "trades": trades, "total_trades": len(trades),
        "wins": wins, "losses": losses,
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "total_premium_collected": round(total_premium * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        "max_drawdown": round(max_dd, 2),
        "assignments": assignments,
        "buy_hold_return": round(buy_hold_pct, 2),
        "strategy_return": round(strategy_pct, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "monthly_pnl": {k: round(v, 2) for k, v in monthly.items()},
    }
