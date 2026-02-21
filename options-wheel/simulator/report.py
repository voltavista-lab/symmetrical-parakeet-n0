"""Report generation for backtest results."""


def generate_report(results: dict) -> str:
    """Generate a formatted text report from backtest results.

    Args:
        results: Dict from simulate_wheel().

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  WHEEL STRATEGY BACKTEST REPORT")
    lines.append(f"  {results['ticker']}  |  {results['start_date']} to {results['end_date']}")
    lines.append("=" * 60)

    if results["total_trades"] == 0:
        lines.append("  No trades executed. Insufficient data or no signals.")
        return "\n".join(lines)

    lines.append("")
    lines.append("  PERFORMANCE SUMMARY")
    lines.append("  " + "-" * 40)
    lines.append(f"  Total trades:          {results['total_trades']}")
    lines.append(f"  Wins / Losses:         {results['wins']} / {results['losses']}")
    lines.append(f"  Win rate:              {results['win_rate']}%")
    lines.append(f"  Total premium:         ${results['total_premium_collected']:,.2f}")
    lines.append(f"  Total P&L:             ${results['total_pnl']:,.2f}")
    lines.append(f"  Avg P&L per trade:     ${results['avg_pnl']:,.2f}")
    lines.append(f"  Max drawdown:          ${results['max_drawdown']:,.2f}")
    lines.append(f"  Assignments:           {results['assignments']}")
    lines.append("")
    lines.append("  RETURNS COMPARISON")
    lines.append("  " + "-" * 40)
    lines.append(f"  Strategy return:       {results['strategy_return']:.2f}%")
    lines.append(f"  Buy & hold return:     {results['buy_hold_return']:.2f}%")
    lines.append(f"  Sharpe-like ratio:     {results['sharpe_ratio']:.2f}")

    monthly = results.get("monthly_pnl", {})
    if monthly:
        lines.append("")
        lines.append("  MONTHLY P&L")
        lines.append("  " + "-" * 40)

        sorted_months = sorted(monthly.keys())
        best_month = max(monthly, key=monthly.get)
        worst_month = min(monthly, key=monthly.get)

        for month in sorted_months:
            pnl = monthly[month]
            bar = "+" * max(int(pnl / 20), 0) if pnl >= 0 else "-" * max(int(abs(pnl) / 20), 0)
            lines.append(f"  {month}:  ${pnl:>8,.2f}  {bar}")

        lines.append("")
        lines.append(f"  Best month:   {best_month} (${monthly[best_month]:,.2f})")
        lines.append(f"  Worst month:  {worst_month} (${monthly[worst_month]:,.2f})")

    # Trade details summary
    trades = results.get("trades", [])
    if trades:
        lines.append("")
        lines.append("  TRADE CLOSE REASONS")
        lines.append("  " + "-" * 40)
        reasons = {}
        for t in trades:
            r = t.get("close_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason:<20s}  {count}")

    lines.append("")
    lines.append("  NOTE: This backtest uses estimated premiums from")
    lines.append("  historical volatility. Real results will differ.")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)
