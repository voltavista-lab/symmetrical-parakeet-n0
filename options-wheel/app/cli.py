"""CLI interface for Options Wheel Strategy Finder."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import WATCHLIST, STRATEGY
from screener.stock_filter import filter_watchlist
from screener.options_scanner import scan_for_candidates
from strategy.short_put import score_short_put, select_best_put
from strategy.covered_call import score_covered_call, select_best_call
from strategy.position_mgmt import position_summary

try:
    from rich.console import Console
    from rich.table import Table
    USE_RICH = True
except ImportError:
    USE_RICH = False


POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "positions.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _save_df_csv(df, filename: str) -> str:
    """Save DataFrame to CSV in the output directory. Returns the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def _print_table_plain(headers: list[str], rows: list[list]) -> None:
    """Fallback plain-text table printer."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def _print_rich_table(title: str, headers: list[str], rows: list[list]) -> None:
    """Print a table using Rich with wide output for Colab compatibility."""
    console = Console(width=200)
    table = Table(title=title, show_lines=True, expand=False)
    for h in headers:
        table.add_column(h, no_wrap=True)
    for row in rows:
        table.add_row(*[str(v) for v in row])
    console.print(table)


def _display_candidates(title: str, df, top_n: int = 10, csv_filename: str = "") -> None:
    """Display top N candidates as a table, optionally saving full results to CSV."""
    if df.empty:
        print(f"\n{title}: No candidates found.")
        return

    # Save full results to CSV if requested
    if csv_filename:
        _save_df_csv(df, csv_filename)

    cols = ["ticker", "option_type", "strike", "expiry", "dte", "mid_price",
            "delta", "implied_vol", "iv_rank", "annual_premium_yield", "composite_score"]
    display_cols = [c for c in cols if c in df.columns]
    top = df.head(top_n)

    headers = display_cols
    rows = []
    for _, row in top.iterrows():
        rows.append([
            round(row[c], 3) if isinstance(row[c], float) else row[c]
            for c in display_cols
        ])

    if USE_RICH:
        _print_rich_table(title, headers, rows)
    else:
        print(f"\n{'=' * 100}")
        print(f"  {title}")
        print(f"{'=' * 100}")
        _print_table_plain(headers, rows)
    print()


def cmd_scan(args) -> None:
    """Run full pipeline: filter stocks -> scan options -> rank candidates."""
    watchlist = WATCHLIST[:args.limit] if args.limit else WATCHLIST
    print(f"Scanning {len(watchlist)} stocks...")

    filtered = filter_watchlist(
        watchlist,
        min_div_yield=STRATEGY["min_dividend_yield"],
        max_price=STRATEGY["max_stock_price"],
        delay=args.delay,
    )
    print(f"Found {len(filtered)} stocks passing fundamental filters.")

    if not filtered:
        print("No stocks passed filters. Try adjusting criteria in config.py.")
        return

    candidates = scan_for_candidates(filtered, STRATEGY, delay=args.delay)
    if candidates.empty:
        print("No option candidates found matching criteria.")
        return

    puts = candidates[candidates["option_type"] == "put"]
    calls = candidates[candidates["option_type"] == "call"]

    save_csv = getattr(args, "output", "") == "csv"
    _display_candidates("Top Short Put Candidates", puts, top_n=10,
                        csv_filename="scan_puts.csv" if save_csv else "")
    _display_candidates("Top Covered Call Candidates", calls, top_n=10,
                        csv_filename="scan_calls.csv" if save_csv else "")
    if save_csv:
        _save_df_csv(candidates, "scan_all.csv")
        print("\nAll results saved to output/ directory.")


def cmd_analyze(args) -> None:
    """Deep dive analysis on a single ticker."""
    ticker = args.ticker.upper()
    print(f"Analyzing {ticker}...")

    from screener.stock_filter import get_stock_fundamentals
    from screener.options_scanner import get_options_chain, calculate_iv_rank

    fundamentals = get_stock_fundamentals(ticker)
    if fundamentals is None:
        print(f"Could not fetch data for {ticker}.")
        return

    print(f"\n--- {fundamentals['name']} ({ticker}) ---")
    print(f"  Price: ${fundamentals['price']:.2f}")
    print(f"  Div Yield: {fundamentals['dividend_yield']:.2f}%")
    print(f"  Sector: {fundamentals['sector']}")
    print(f"  Market Cap: ${fundamentals['market_cap']:,.0f}")
    print(f"  Beta: {fundamentals.get('beta', 'N/A')}")

    iv_rank = calculate_iv_rank(ticker)
    print(f"  IV Rank: {iv_rank:.1f}")

    chain = get_options_chain(ticker, STRATEGY["dte_range"])
    if chain.empty:
        print(f"\n  No options found in DTE range {STRATEGY['dte_range']}.")
        return

    puts = chain[chain["type"] == "put"].sort_values("strike", ascending=False)
    calls = chain[chain["type"] == "call"].sort_values("strike")

    save_csv = getattr(args, "output", "") == "csv"
    if save_csv:
        if not puts.empty:
            _save_df_csv(puts, f"analyze_{ticker}_puts.csv")
        if not calls.empty:
            _save_df_csv(calls, f"analyze_{ticker}_calls.csv")
        print(f"\nAll {ticker} options saved to output/ directory.")

    headers = ["strike", "expiry", "dte", "bid", "ask", "mid_price", "impliedVolatility", "openInterest", "volume"]

    def _to_rows(df):
        rows = []
        for _, row in df.head(15).iterrows():
            rows.append([
                round(row[c], 3) if isinstance(row[c], float) else row[c]
                for c in headers
            ])
        return rows

    if not puts.empty:
        if USE_RICH:
            _print_rich_table(f"{ticker} Puts", headers, _to_rows(puts))
        else:
            print(f"\n  {ticker} Puts:")
            _print_table_plain(headers, _to_rows(puts))

    if not calls.empty:
        if USE_RICH:
            _print_rich_table(f"{ticker} Calls", headers, _to_rows(calls))
        else:
            print(f"\n  {ticker} Calls:")
            _print_table_plain(headers, _to_rows(calls))


def cmd_portfolio(args) -> None:
    """Show current positions and management signals."""
    if not os.path.exists(POSITIONS_FILE):
        print(f"No positions file found at {POSITIONS_FILE}.")
        print("Create positions.json with your positions to track them.")
        return

    with open(POSITIONS_FILE) as f:
        positions = json.load(f)

    if not positions:
        print("No positions in portfolio.")
        return

    summary = position_summary(positions)
    print("\n--- Portfolio Summary ---")
    print(f"  Total positions: {summary['total_positions']}")
    print(f"  Premium collected: ${summary['total_premium_collected']:,.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"  Wins: {summary['win_count']}  Losses: {summary['loss_count']}")
    print(f"  Avg P&L: ${summary['avg_pnl']:,.2f}")

    print("\n--- Position Details ---")
    profit_target = STRATEGY["profit_target_pct"]
    stop_loss = STRATEGY["max_loss_close_pct"]
    roll_trigger = STRATEGY["roll_trigger_pct"]

    from strategy.position_mgmt import check_profit_target, check_stop_loss, check_roll_trigger

    for pos in positions:
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", 0)
        ticker = pos.get("ticker", "?")
        opt_type = pos.get("option_type", "?")
        signals = []
        if check_profit_target(entry, current, profit_target):
            signals.append("CLOSE (profit target)")
        if check_stop_loss(entry, current, stop_loss):
            signals.append("CLOSE (stop loss)")
        if check_roll_trigger(entry, current, roll_trigger):
            signals.append("ROLL")

        signal_str = ", ".join(signals) if signals else "HOLD"
        pnl = (entry - current) * 100 * pos.get("quantity", 1)
        print(f"  {ticker} {opt_type} | entry=${entry:.2f} current=${current:.2f} | "
              f"P&L=${pnl:,.0f} | Signal: {signal_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Options Wheel Strategy Finder")
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan watchlist for wheel candidates")
    scan_parser.add_argument("--limit", type=int, default=0, help="Limit watchlist to N stocks")
    scan_parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    scan_parser.add_argument("--output", choices=["csv"], default="", help="Save results to CSV files in output/")
    scan_parser.set_defaults(func=cmd_scan)

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Deep dive on a single ticker")
    analyze_parser.add_argument("ticker", help="Ticker symbol")
    analyze_parser.add_argument("--output", choices=["csv"], default="", help="Save results to CSV files in output/")
    analyze_parser.set_defaults(func=cmd_analyze)

    # portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Show portfolio positions")
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # backtest command (placeholder, implemented in Phase 5)
    backtest_parser = subparsers.add_parser("backtest", help="Backtest wheel strategy")
    backtest_parser.add_argument("ticker", help="Ticker symbol")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.set_defaults(func=lambda args: _cmd_backtest(args))

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


def _cmd_backtest(args) -> None:
    """Run backtest (imported here to avoid circular imports at module load)."""
    from simulator.backtest import simulate_wheel
    from simulator.report import generate_report

    print(f"Backtesting {args.ticker} from {args.start} to {args.end}...")
    results = simulate_wheel(args.ticker, args.start, args.end, STRATEGY)
    report = generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
