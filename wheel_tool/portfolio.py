#!/usr/bin/env python3
"""
portfolio.py - Wheel Strategy Portfolio Manager
Entry point: python portfolio.py

Menu:
  [1] Dashboard       Live positions with colour-coded P&L
  [2] Roll Analysis   Roll scenarios for underwater short puts
  [3] CC Screener     Covered call opportunities across watchlist
  [4] Wheel Tracker   Cost-basis / breakeven tracker per ticker
  [5] Risk Summary    Capital-at-risk, sector concentration, currency split
  [6] Add/Edit Pos    Persist a new or updated position
  [7] Export Report   Write portfolio_report.txt
  [0] Quit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

# ── module imports ────────────────────────────────────────────────────────────
from modules import data_fetcher as df
from modules import analysis as an
from modules import display as ui

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
POSITIONS_FILE = BASE_DIR / "positions.json"
REPORT_FILE    = BASE_DIR / "portfolio_report.txt"

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Data persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    """Load positions.json; exit cleanly if file is missing/corrupt."""
    if not POSITIONS_FILE.exists():
        console.print(f"[red]positions.json not found at {POSITIONS_FILE}[/red]")
        sys.exit(1)
    try:
        with open(POSITIONS_FILE) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        console.print(f"[red]positions.json is malformed: {exc}[/red]")
        sys.exit(1)


def save_data(data: dict) -> None:
    with open(POSITIONS_FILE, "w") as fh:
        json.dump(data, fh, indent=2)
    console.print(f"  [dim]Saved → {POSITIONS_FILE}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Fetch helpers  (cached per session to avoid hammering yfinance)
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_CACHE: dict = {}


def _get_all_tickers(data: dict) -> list[str]:
    """Unique tickers from positions + watchlist."""
    pos_tickers = [p["ticker"] for p in data["positions"]]
    watch       = data.get("screener_watchlist", [])
    return sorted(set(pos_tickers + watch))


def fetch_market_data(data: dict, force: bool = False,
                      fetch_options: bool = True) -> tuple[dict, dict]:
    """
    Return (prices_dict, options_dict).
    Results are cached for the session; pass force=True to refresh.
    """
    global _SESSION_CACHE

    if not force and _SESSION_CACHE.get("prices"):
        return _SESSION_CACHE["prices"], _SESSION_CACHE.get("options", {})

    tickers = _get_all_tickers(data)
    console.print(f"  [dim]Fetching live data for: {', '.join(tickers)} …[/dim]")

    prices  = df.fetch_prices(tickers)

    options: dict = {}
    if fetch_options:
        for ticker in tickers:
            console.print(f"  [dim]  options chain → {ticker}[/dim]", end="\r")
            options[ticker] = df.fetch_options_chain(ticker)
        console.print(" " * 50, end="\r")   # clear the progress line

    _SESSION_CACHE = {"prices": prices, "options": options}
    return prices, options


# ─────────────────────────────────────────────────────────────────────────────
# Menu handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_dashboard(data: dict) -> None:
    console.print("\n  [dim]Loading market data…[/dim]")
    prices, options = fetch_market_data(data, fetch_options=False)
    enriched = an.enrich_positions(data["positions"], prices, options)
    ui.print_dashboard(enriched, data["account"])


def handle_roll_analysis(data: dict) -> None:
    console.print("\n  [dim]Loading options chains for roll analysis…[/dim]")
    prices, options = fetch_market_data(data, fetch_options=True)
    enriched = an.enrich_positions(data["positions"], prices, options)

    short_puts = [p for p in enriched if p["type"] == "short_put"]
    if not short_puts:
        console.print("  [green]No short put positions to analyse.[/green]")
        return

    console.print(f"\n  Found [bold]{len(short_puts)}[/bold] short put position(s):\n")
    for i, pos in enumerate(short_puts):
        itm_flag = " [bold red](ITM)[/bold red]" if pos.get("itm") else ""
        console.print(
            f"  [{i+1}] {pos['ticker']:5s}  ${pos.get('strike', '?'):.2f} Put"
            f"  x{abs(pos['contracts'])}{itm_flag}"
            f"  P&L: {pos.get('pnl_pct', 0):+.1f}%"
        )

    choice_raw = Prompt.ask(
        "\n  Select position number (or 'a' for all, 0 to cancel)",
        default="a"
    ).strip().lower()

    if choice_raw == "0":
        return

    if choice_raw == "a":
        targets = short_puts
    else:
        try:
            idx = int(choice_raw) - 1
            targets = [short_puts[idx]]
        except (ValueError, IndexError):
            console.print("  [red]Invalid selection.[/red]")
            return

    for pos in targets:
        scenarios = an.analyse_rolls(pos, options, pos.get("spot_price"))
        ui.print_roll_analysis(pos, scenarios)
        console.print()


def handle_cc_screener(data: dict) -> None:
    console.print("\n  [dim]Loading options chains for CC screening…[/dim]")
    prices, options = fetch_market_data(data, fetch_options=True)
    watchlist = data.get("screener_watchlist", [])
    results   = an.screen_covered_calls(watchlist, prices, options)
    ui.print_cc_screener(results)


def handle_wheel_tracker(data: dict) -> None:
    console.print("\n  [dim]Loading price data…[/dim]")
    prices, _ = fetch_market_data(data, fetch_options=False)
    wheel_states = an.compute_wheel_state(
        data["positions"],
        data.get("premium_history", []),
        prices,
    )
    ui.print_wheel_tracker(wheel_states)


def handle_risk_summary(data: dict) -> None:
    console.print("\n  [dim]Loading price data…[/dim]")
    prices, options = fetch_market_data(data, fetch_options=False)
    enriched = an.enrich_positions(data["positions"], prices, options)
    risk = an.compute_risk_summary(enriched, data["account"], prices)
    ui.print_risk_summary(risk)


def handle_add_position(data: dict) -> None:
    new_pos = ui.prompt_add_position()
    if new_pos is None:
        return

    # Replace existing ID or append
    existing_ids = [p["id"] for p in data["positions"]]
    if new_pos["id"] in existing_ids:
        idx = existing_ids.index(new_pos["id"])
        data["positions"][idx] = new_pos
        console.print(f"  [cyan]Updated position {new_pos['id']}[/cyan]")
    else:
        data["positions"].append(new_pos)
        console.print(f"  [green]Added position {new_pos['id']}[/green]")

    # Optionally record premium
    prem_raw = console.input(
        "  Record premium collected for this position? (amount in $, or Enter to skip): "
    ).strip()
    if prem_raw:
        try:
            prem = float(prem_raw)
            data.setdefault("premium_history", []).append({
                "position_id": new_pos["id"],
                "ticker":      new_pos["ticker"],
                "premium":     prem,
                "type":        new_pos["type"],
            })
        except ValueError:
            pass

    save_data(data)

    # Invalidate cache so next fetch reflects new positions
    global _SESSION_CACHE
    _SESSION_CACHE = {}


def handle_export(data: dict) -> None:
    console.print("\n  [dim]Generating report…[/dim]")
    prices, options = fetch_market_data(data, fetch_options=True)
    enriched    = an.enrich_positions(data["positions"], prices, options)
    risk        = an.compute_risk_summary(enriched, data["account"], prices)
    watchlist   = data.get("screener_watchlist", [])
    cc_results  = an.screen_covered_calls(watchlist, prices, options)

    path = ui.export_report(enriched, risk, cc_results, str(REPORT_FILE))
    console.print(f"  [green]Report written → {path}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# Refresh / account update helpers
# ─────────────────────────────────────────────────────────────────────────────

def handle_refresh(data: dict) -> None:
    """Force re-fetch of market data."""
    global _SESSION_CACHE
    _SESSION_CACHE = {}
    console.print("  [dim]Cache cleared. Data will refresh on next action.[/dim]")


def handle_update_account(data: dict) -> None:
    """Quick update of account values."""
    console.print("\n  [bold]Update Account Values[/bold]  (Enter to keep current)")
    acc = data["account"]

    def _upd(key: str, label: str) -> None:
        cur = acc.get(key, 0)
        raw = console.input(f"  {label} [{cur}]: ").strip()
        if raw:
            try:
                acc[key] = float(raw)
            except ValueError:
                pass

    _upd("value_cad",       "Account value (CAD)")
    _upd("buying_power_cad","Buying power (CAD)")
    _upd("cash_cad",        "Cash available (CAD)")
    _upd("cash_usd",        "Cash available (USD)")
    save_data(data)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

MENU_HANDLERS = {
    "1": ("Dashboard",         handle_dashboard),
    "2": ("Roll Analysis",     handle_roll_analysis),
    "3": ("CC Screener",       handle_cc_screener),
    "4": ("Wheel Tracker",     handle_wheel_tracker),
    "5": ("Risk Summary",      handle_risk_summary),
    "6": ("Add/Edit Position", handle_add_position),
    "7": ("Export Report",     handle_export),
    "r": ("Refresh Data",      handle_refresh),
    "u": ("Update Account",    handle_update_account),
}


def main() -> None:
    data = load_data()
    ui.print_header(data["account"])

    while True:
        ui.print_menu()
        # Extended options shown in dim
        console.print("  [dim][r] Refresh data   [u] Update account values[/dim]\n")
        choice = Prompt.ask("  Choose", default="1").strip().lower()

        if choice == "0":
            console.print("\n  [cyan]Goodbye.[/cyan]\n")
            break

        if choice in MENU_HANDLERS:
            name, handler = MENU_HANDLERS[choice]
            console.print(f"\n  → [bold cyan]{name}[/bold cyan]")
            try:
                handler(data)
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"\n  [red]Error: {exc}[/red]")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
        else:
            console.print("  [yellow]Unknown option. Try 0–7, r, or u.[/yellow]")


if __name__ == "__main__":
    main()
