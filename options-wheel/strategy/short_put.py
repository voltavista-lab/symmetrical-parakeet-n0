"""Short put strategy logic and scoring."""

import pandas as pd


def score_short_put(option_row: dict | pd.Series, stock_info: dict) -> tuple[float, str]:
    """Score a short put candidate.

    Args:
        option_row: Dict/Series with option data (mid_price, strike, dte, delta,
                    implied_vol, iv_rank, annual_premium_yield, openInterest, volume).
        stock_info: Dict with stock fundamentals (price, dividend_yield).

    Returns:
        Tuple of (score 0-100, reasoning string).
    """
    reasons = []

    # Premium yield score (0-30)
    apy = option_row.get("annual_premium_yield", 0)
    premium_score = min(apy / 40, 1.0) * 30
    reasons.append(f"APY={apy:.1f}%")

    # Delta score - prefer closer to -0.30 target (0-20)
    delta = abs(option_row.get("delta", 0))
    delta_diff = abs(delta - 0.30)
    delta_score = max(0, (1 - delta_diff / 0.15)) * 20
    reasons.append(f"delta={option_row.get('delta', 0):.3f}")

    # IV rank score (0-20)
    iv_rank = option_row.get("iv_rank", 0)
    iv_score = (iv_rank / 100) * 20
    reasons.append(f"IVR={iv_rank:.0f}")

    # Dividend yield score (0-15)
    div_yield = stock_info.get("dividend_yield", 0)
    div_score = min(div_yield / 5, 1.0) * 15
    reasons.append(f"div={div_yield:.1f}%")

    # Liquidity score (0-15)
    oi = option_row.get("openInterest", 0) or 0
    vol = option_row.get("volume", 0) or 0
    liq_score = min((oi + vol * 5) / 3000, 1.0) * 15
    reasons.append(f"OI={oi}")

    total = premium_score + delta_score + iv_score + div_score + liq_score
    return round(total, 2), " | ".join(reasons)


def select_best_put(candidates_df: pd.DataFrame) -> pd.Series | None:
    """Select the best short put from a scored DataFrame.

    Expects candidates_df to have a 'put_score' column already computed,
    or computes one from composite_score if available.

    Returns:
        Best option row as Series, or None if empty.
    """
    if candidates_df.empty:
        return None

    puts = candidates_df[candidates_df["option_type"] == "put"].copy()
    if puts.empty:
        return None

    score_col = "put_score" if "put_score" in puts.columns else "composite_score"
    if score_col not in puts.columns:
        return None

    best_idx = puts[score_col].idxmax()
    return puts.loc[best_idx]


def capital_required_put(strike: float) -> float:
    """Cash-secured put capital requirement = strike * 100."""
    return strike * 100
