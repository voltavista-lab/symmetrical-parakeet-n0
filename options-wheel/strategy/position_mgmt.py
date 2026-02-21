"""Position management - profit targets, stop losses, rolls."""


def check_profit_target(entry_price: float, current_price: float, target_pct: float) -> bool:
    """Check if option position has hit profit target.

    For short options, profit = entry_price - current_price.
    target_pct is the percentage of max profit to capture (e.g. 50).

    Returns True if profit target is met.
    """
    if entry_price <= 0:
        return False
    profit_pct = (entry_price - current_price) / entry_price * 100
    return profit_pct >= target_pct


def check_stop_loss(entry_price: float, current_price: float, max_loss_pct: float) -> bool:
    """Check if option position has hit stop loss.

    Loss occurs when current_price > entry_price for short options.
    max_loss_pct is the maximum loss as percentage of entry premium.

    Returns True if stop loss is triggered.
    """
    if entry_price <= 0:
        return False
    loss_pct = (current_price - entry_price) / entry_price * 100
    return loss_pct >= max_loss_pct


def check_roll_trigger(entry_price: float, current_price: float, roll_pct: float) -> bool:
    """Check if position should be rolled.

    Roll when loss exceeds roll_pct of entry premium.

    Returns True if roll is triggered.
    """
    if entry_price <= 0:
        return False
    loss_pct = (current_price - entry_price) / entry_price * 100
    return loss_pct >= roll_pct


def calculate_roll(current_option: dict, new_expiry_dte: int) -> dict:
    """Estimate roll credit/debit.

    Args:
        current_option: Dict with mid_price, strike, dte, implied_vol.
        new_expiry_dte: DTE for the new option to roll into.

    Returns:
        Dict with estimated roll details.
    """
    current_mid = current_option.get("mid_price", 0)
    current_dte = current_option.get("dte", 0)
    iv = current_option.get("implied_vol", 0.3)
    strike = current_option.get("strike", 0)

    # Approximate new premium using time decay scaling
    if current_dte > 0 and new_expiry_dte > 0:
        time_ratio = (new_expiry_dte / current_dte) ** 0.5
        estimated_new_premium = current_mid * time_ratio
    else:
        estimated_new_premium = current_mid

    # Net credit = new premium - cost to close current
    net_credit = estimated_new_premium - current_mid

    return {
        "close_cost": current_mid,
        "new_premium": round(estimated_new_premium, 2),
        "net_credit": round(net_credit, 2),
        "new_dte": new_expiry_dte,
        "same_strike": strike,
    }


def position_summary(positions: list[dict]) -> dict:
    """Compute portfolio-level stats from a list of positions.

    Each position dict should have: ticker, option_type, entry_price,
    current_price, strike, quantity (default 1).

    Returns:
        Dict with total_positions, total_premium_collected, total_pnl,
        win_count, loss_count, avg_pnl.
    """
    if not positions:
        return {
            "total_positions": 0,
            "total_premium_collected": 0,
            "total_pnl": 0,
            "win_count": 0,
            "loss_count": 0,
            "avg_pnl": 0,
        }

    total_premium = 0
    total_pnl = 0
    wins = 0
    losses = 0

    for pos in positions:
        qty = pos.get("quantity", 1)
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", 0)
        pnl = (entry - current) * 100 * qty  # short option PnL per contract
        total_premium += entry * 100 * qty
        total_pnl += pnl
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

    n = len(positions)
    return {
        "total_positions": n,
        "total_premium_collected": round(total_premium, 2),
        "total_pnl": round(total_pnl, 2),
        "win_count": wins,
        "loss_count": losses,
        "avg_pnl": round(total_pnl / n, 2) if n else 0,
    }
