"""Covered call strategy logic and scoring."""

import pandas as pd


def score_covered_call(option_row: dict | pd.Series, stock_info: dict) -> tuple[float, str]:
    """Score a covered call candidate.

    Args:
        option_row: Dict/Series with option data.
        stock_info: Dict with stock fundamentals (price, dividend_yield).

    Returns:
        Tuple of (score 0-100, reasoning string).
    """
    reasons = []

    # Premium yield score (0-30)
    apy = option_row.get("annual_premium_yield", 0)
    premium_score = min(apy / 40, 1.0) * 30
    reasons.append(f"APY={apy:.1f}%")

    # Delta score - prefer closer to 0.30 target (0-20)
    delta = option_row.get("delta", 0)
    delta_diff = abs(delta - 0.30)
    delta_score = max(0, (1 - delta_diff / 0.15)) * 20
    reasons.append(f"delta={delta:.3f}")

    # IV rank score (0-20)
    iv_rank = option_row.get("iv_rank", 0)
    iv_score = (iv_rank / 100) * 20
    reasons.append(f"IVR={iv_rank:.0f}")

    # Upside to strike (0-15): prefer strikes above current price
    price = stock_info.get("price", 0)
    strike = option_row.get("strike", 0)
    if price > 0 and strike > 0:
        upside_pct = (strike - price) / price * 100
        upside_score = min(max(upside_pct, 0) / 10, 1.0) * 15
        reasons.append(f"upside={upside_pct:.1f}%")
    else:
        upside_score = 0

    # Liquidity score (0-15)
    oi = option_row.get("openInterest", 0) or 0
    vol = option_row.get("volume", 0) or 0
    liq_score = min((oi + vol * 5) / 3000, 1.0) * 15
    reasons.append(f"OI={oi}")

    total = premium_score + delta_score + iv_score + upside_score + liq_score
    return round(total, 2), " | ".join(reasons)


def select_best_call(candidates_df: pd.DataFrame) -> pd.Series | None:
    """Select the best covered call from a scored DataFrame.

    Returns:
        Best option row as Series, or None if empty.
    """
    if candidates_df.empty:
        return None

    calls = candidates_df[candidates_df["option_type"] == "call"].copy()
    if calls.empty:
        return None

    score_col = "call_score" if "call_score" in calls.columns else "composite_score"
    if score_col not in calls.columns:
        return None

    best_idx = calls[score_col].idxmax()
    return calls.loc[best_idx]


def capital_required_call(stock_price: float) -> float:
    """Covered call capital requirement = stock_price * 100."""
    return stock_price * 100
