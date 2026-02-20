"""
display.py - Rich terminal UI: tables, panels, colour-coded output.
All rendering lives here; no business logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule

console = Console()

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _pnl_color(pnl_pct: float | None) -> str:
    if pnl_pct is None:
        return "white"
    if pnl_pct > 0:
        return "bright_green"
    if pnl_pct >= -10:
        return "yellow"
    return "bright_red"


def _pnl_str(pnl: float | None, pnl_pct: float | None) -> Text:
    pnl_text = f"${pnl:+.2f} ({pnl_pct:+.1f}%)" if pnl is not None and pnl_pct is not None else "N/A"
    return Text(pnl_text, style=_pnl_color(pnl_pct))


def _fmt_price(val: float | None, currency: str = "USD", prefix: str = "") -> str:
    if val is None:
        return "—"
    sym = "C$" if currency == "CAD" else "$"
    return f"{prefix}{sym}{val:,.2f}"


def _flag(condition: bool, label: str, color: str = "red") -> Text:
    return Text(f" [{label}]", style=f"bold {color}") if condition else Text("")


# ---------------------------------------------------------------------------
# Header / title
# ---------------------------------------------------------------------------

def print_header(account: dict) -> None:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = Text("⚙  Wheel Strategy Portfolio Manager", style="bold cyan")
    subtitle = (
        f"  Account: C${account.get('value_cad', 0):,.2f}  │"
        f"  Buying Power: C${account.get('buying_power_cad', 0):,.2f}  │"
        f"  Cash (CAD): C${account.get('cash_cad', 0):,.2f}  │"
        f"  Cash (USD): ${account.get('cash_usd', 0):,.2f}  │"
        f"  {now}"
    )
    console.print(Panel(title, subtitle=subtitle, style="bold blue", expand=True))


def print_menu() -> None:
    console.print()
    console.print(Rule("[bold cyan]Main Menu[/bold cyan]"))
    options = [
        "[1] Dashboard",
        "[2] Roll Analysis",
        "[3] CC Screener",
        "[4] Wheel Tracker",
        "[5] Risk Summary",
        "[6] Add / Edit Position",
        "[7] Export Report",
        "[0] Quit",
    ]
    for opt in options:
        console.print(f"  {opt}", style="cyan")
    console.print()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def print_dashboard(enriched_positions: list[dict], account: dict) -> None:
    console.print()
    console.print(Rule("[bold yellow]Portfolio Dashboard[/bold yellow]"))

    # ── Positions table ──────────────────────────────────────────────────────
    tbl = Table(
        title="Current Positions",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    tbl.add_column("ID",         style="dim",        width=8)
    tbl.add_column("Type",       style="cyan",        width=14)
    tbl.add_column("Ticker",     style="bold white",  width=7)
    tbl.add_column("Contracts",  justify="right",     width=10)
    tbl.add_column("Strike",     justify="right",     width=8)
    tbl.add_column("Spot",       justify="right",     width=9)
    tbl.add_column("Buffer %",   justify="right",     width=9)
    tbl.add_column("Value",      justify="right",     width=12)
    tbl.add_column("P&L",        justify="right",     width=20)
    tbl.add_column("Flags",      width=14)

    total_pnl   = 0.0
    total_value = 0.0

    for pos in enriched_positions:
        pnl_pct  = pos.get("pnl_pct", 0.0)
        pnl      = pos.get("pnl", 0.0)
        value    = pos.get("value", 0.0)
        currency = pos.get("currency", "USD")
        spot     = pos.get("spot_price")

        total_pnl   += pnl or 0
        total_value += abs(value or 0)

        # Buffer col (only for short puts)
        buffer_str = ""
        if pos["type"] == "short_put" and pos.get("buffer_pct") is not None:
            bp = pos["buffer_pct"]
            color = "green" if bp > 0 else "red"
            buffer_str = f"[{color}]{bp:+.1f}%[/{color}]"

        # Flags
        flags = Text()
        if pos.get("is_critical"):
            flags.append("CRITICAL ", style="bold red")
        if pos.get("itm"):
            flags.append("ITM ", style="bold red")
        if pnl_pct is not None and pnl_pct < -50:
            flags.append(">50% loss", style="red")

        pnl_text = _pnl_str(pnl, pnl_pct)

        strike_str = f"${pos['strike']:.2f}" if pos.get("strike") else "—"
        spot_str   = _fmt_price(spot, currency)

        tbl.add_row(
            pos.get("id", "—"),
            pos["type"].replace("_", " "),
            pos["ticker"],
            str(pos.get("contracts", "—")),
            strike_str,
            spot_str,
            buffer_str,
            _fmt_price(abs(value), currency),
            pnl_text,
            flags,
        )

    console.print(tbl)

    # ── Summary bar ──────────────────────────────────────────────────────────
    pnl_color = "bright_green" if total_pnl >= 0 else "bright_red"
    summary = (
        f"  Total Value: [bold white]${total_value:,.2f}[/bold white]"
        f"  │  Total P&L: [{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]"
        f"  │  Positions: [bold]{len(enriched_positions)}[/bold]"
    )
    console.print(Panel(summary, style="dim", expand=True))


# ---------------------------------------------------------------------------
# Roll analysis
# ---------------------------------------------------------------------------

def print_roll_analysis(position: dict, scenarios: list[dict]) -> None:
    ticker   = position["ticker"]
    strike   = position.get("strike", "?")
    n        = abs(position.get("contracts", 1))
    spot     = position.get("spot_price", "?")
    pnl_pct  = position.get("pnl_pct", 0)

    console.print()
    header = (
        f"Roll Analysis  │  [bold]{ticker}[/bold]  ${strike} Put  "
        f"x{n}  │  Spot: [cyan]${spot}[/cyan]  "
        f"│  P&L: [{_pnl_color(pnl_pct)}]{pnl_pct:+.1f}%[/{_pnl_color(pnl_pct)}]"
    )
    console.print(Rule(header))

    if not scenarios or (len(scenarios) == 1 and "error" in scenarios[0]):
        msg = scenarios[0].get("error", "No roll scenarios available") if scenarios else "No data"
        console.print(f"  [yellow]{msg}[/yellow]")
        return

    tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", expand=True)
    tbl.add_column("Expiry",       width=12)
    tbl.add_column("DTE",          justify="right", width=6)
    tbl.add_column("New Strike",   justify="right", width=11)
    tbl.add_column("Δ Strike",     justify="right", width=9)
    tbl.add_column("New Mid",      justify="right", width=10)
    tbl.add_column("Gross Credit", justify="right", width=13)
    tbl.add_column("Close Cost",   justify="right", width=12)
    tbl.add_column("Net Credit",   justify="right", width=13)
    tbl.add_column("Verdict",      width=14)

    # Show top 12 scenarios
    for s in scenarios[:12]:
        net  = s["net_credit"]
        nc   = f"[bright_green]${net:+.2f}[/bright_green]" if net >= 0 else f"[red]${net:+.2f}[/red]"
        verdict = Text("✓ CREDIT",  style="bright_green bold") if s["is_net_credit"] else Text("✗ DEBIT", style="red")
        strike_delta = f"{s['strike_change']:+.2f}" if s["strike_change"] != 0 else "—"
        tbl.add_row(
            s["expiry"],
            str(s["dte"]),
            f"${s['new_strike']:.2f}",
            strike_delta,
            f"${s['new_mid']:.4f}",
            f"${s['gross_credit']:.2f}",
            f"${s['close_cost']:.2f}",
            nc,
            verdict,
        )

    console.print(tbl)

    # Highlight best net-credit scenario
    best = next((s for s in scenarios if s["is_net_credit"]), None)
    if best:
        console.print(
            f"  [bold green]Best roll:[/bold green] "
            f"{best['expiry']} @ ${best['new_strike']:.2f}  →  "
            f"net credit [bright_green]${best['net_credit']:.2f}[/bright_green]"
        )
    else:
        console.print("  [yellow]No net-credit roll found – all scenarios result in a debit.[/yellow]")


# ---------------------------------------------------------------------------
# CC Screener
# ---------------------------------------------------------------------------

def print_cc_screener(results: list[dict]) -> None:
    console.print()
    console.print(Rule("[bold yellow]Covered Call Screener  (~30 DTE)[/bold yellow]"))

    tbl = Table(box=box.ROUNDED, header_style="bold magenta", expand=True)
    tbl.add_column("Ticker",    style="bold white", width=7)
    tbl.add_column("Curr",      width=5)
    tbl.add_column("Spot",      justify="right", width=9)
    tbl.add_column("DTE",       justify="right", width=6)
    tbl.add_column("Strike",    justify="right", width=9)
    tbl.add_column("Premium",   justify="right", width=10)
    tbl.add_column("Delta",     justify="right", width=8)
    tbl.add_column("IV %",      justify="right", width=7)
    tbl.add_column("IV Rank",   justify="right", width=9)
    tbl.add_column("HV 30d",    justify="right", width=8)
    tbl.add_column("Div Yield", justify="right", width=10)
    tbl.add_column("Ann Yield", justify="right", width=11)
    tbl.add_column("Priority",  width=12)

    for r in results:
        if "error" in r:
            tbl.add_row(
                r["ticker"], r.get("currency", "USD"),
                _fmt_price(r.get("spot"), r.get("currency", "USD")),
                "—", "—", "—", "—", "—", "—", "—", "—",
                f"[dim]{r['error']}[/dim]",
                "",
            )
            continue

        ann  = r.get("total_yield_pct", 0)
        prio = (
            Text("★ HIGH PRI", style="bold bright_green")
            if r.get("high_priority") else
            Text("", style="dim")
        )
        tbl.add_row(
            r["ticker"],
            r.get("currency", "USD"),
            _fmt_price(r["spot"], r.get("currency", "USD")),
            str(r.get("dte", "—")),
            f"${r['strike']:.2f}",
            f"${r['premium']:.4f}",
            f"{r['delta']:.3f}",
            f"{r.get('iv', 0):.1f}%",
            f"{r['iv_rank']:.0f}" if r.get("iv_rank") else "—",
            f"{r['hv_30']:.1f}%" if r.get("hv_30") else "—",
            f"{r['div_yield']:.2f}%",
            f"[bold]{ann:.1f}%[/bold]",
            prio,
        )

    console.print(tbl)
    console.print("  [dim]Ann Yield = (premium / strike) × (365 / DTE) × 100  +  dividend yield[/dim]")


# ---------------------------------------------------------------------------
# Wheel tracker
# ---------------------------------------------------------------------------

def print_wheel_tracker(wheel_states: list[dict]) -> None:
    console.print()
    console.print(Rule("[bold yellow]Wheel Strategy Tracker[/bold yellow]"))

    tbl = Table(box=box.ROUNDED, header_style="bold magenta", expand=True)
    tbl.add_column("Ticker",        style="bold white", width=7)
    tbl.add_column("Spot",          justify="right", width=10)
    tbl.add_column("Has Shares",    justify="center", width=11)
    tbl.add_column("Short Puts",    justify="center", width=11)
    tbl.add_column("Covered Calls", justify="center", width=13)
    tbl.add_column("Cost Basis",    justify="right", width=12)
    tbl.add_column("Prem/Share",    justify="right", width=12)
    tbl.add_column("Breakeven",     justify="right", width=12)
    tbl.add_column("Total Prem",    justify="right", width=12)
    tbl.add_column("Action",        width=22)

    for ws in wheel_states:
        spot  = ws.get("spot")
        cb    = ws.get("cost_basis_per_share")
        be    = ws.get("breakeven")

        # Breakeven colour
        be_text = Text("—")
        if be and spot:
            be_pct  = (spot - be) / be * 100
            be_color = "bright_green" if spot > be else "red"
            be_text = Text(f"${be:.2f} ({be_pct:+.1f}%)", style=be_color)

        action = Text("")
        if ws.get("suggest_cc"):
            action = Text("→ Sell CC on shares", style="bold cyan")
        elif ws["short_puts"] and not ws["has_shares"]:
            action = Text("Monitoring puts", style="yellow")
        elif ws["covered_calls"]:
            action = Text("Wheel active", style="green")

        tbl.add_row(
            ws["ticker"],
            _fmt_price(spot, "USD"),
            "✓" if ws["has_shares"] else "✗",
            str(len(ws["short_puts"])),
            str(len(ws["covered_calls"])),
            _fmt_price(cb, "USD"),
            _fmt_price(ws.get("premium_per_share"), "USD"),
            be_text,
            _fmt_price(ws["total_premiums_collected"], "USD"),
            action,
        )

    console.print(tbl)
    console.print(
        "  [dim]Cost Basis is estimated from current P&L %. "
        "Update via 'Add/Edit Position' for exact values.[/dim]"
    )


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------

def print_risk_summary(risk: dict) -> None:
    console.print()
    console.print(Rule("[bold red]Risk Summary[/bold red]"))

    # Key metrics panel
    risk_pct  = risk.get("risk_pct_of_account", 0)
    risk_color = "red" if risk_pct > 60 else ("yellow" if risk_pct > 30 else "green")
    metrics = (
        f"  Capital at Risk (puts): [bold]{risk_color}]"
        f"${risk['capital_at_risk_usd']:,.2f} USD"
        f" / C${risk['capital_at_risk_cad']:,.2f}[/{risk_color}]"
        f"  ({risk_pct:.1f}% of account)\n"
        f"  Total P&L:  [{_pnl_color(risk['total_pnl'])}]${risk['total_pnl']:+,.2f}[/{_pnl_color(risk['total_pnl'])}]\n"
        f"  Account:    C${risk['account_value_cad']:,.2f}  │  "
        f"Buying Power: C${risk['buying_power_cad']:,.2f}  │  "
        f"Cash: C${risk['cash_cad']:,.2f}"
    )
    console.print(Panel(metrics, title="Capital Overview", style="bold", expand=True))

    # Sector concentration table
    sec_tbl = Table(title="Sector Concentration", box=box.SIMPLE_HEAD,
                    header_style="bold magenta")
    sec_tbl.add_column("Sector",     style="white", width=25)
    sec_tbl.add_column("Exposure %", justify="right", width=12)
    sec_tbl.add_column("Warning",    width=12)

    for sec, pct in sorted(risk["sector_exposure_pct"].items(),
                           key=lambda x: x[1], reverse=True):
        warn = Text("⚠ CONCENTRATED", style="bold red") if sec in risk["concentrated_sectors"] else Text("")
        color = "red" if sec in risk["concentrated_sectors"] else "white"
        sec_tbl.add_row(sec, Text(f"{pct:.1f}%", style=color), warn)

    # Currency exposure table
    cur_tbl = Table(title="Currency Exposure", box=box.SIMPLE_HEAD,
                    header_style="bold magenta")
    cur_tbl.add_column("Currency", style="white", width=10)
    cur_tbl.add_column("% of Total", justify="right", width=12)

    cur_tbl.add_row("USD", f"{risk['usd_exposure_pct']:.1f}%")
    cur_tbl.add_row("CAD", f"{risk['cad_exposure_pct']:.1f}%")
    cur_tbl.add_row("[dim]Rate used[/dim]", f"[dim]{risk['usd_cad_rate_used']} USD/CAD[/dim]")

    console.print(Columns([sec_tbl, cur_tbl]))

    # Critical positions
    if risk.get("critical_count", 0) > 0:
        crit_panel_text = "  " + "\n  ".join(risk["critical_positions"])
        console.print(Panel(
            crit_panel_text,
            title=f"[bold red]Critical Positions ({risk['critical_count']})[/bold red]",
            style="red",
            expand=True,
        ))
    else:
        console.print("  [green]No critical positions flagged.[/green]")


# ---------------------------------------------------------------------------
# Position add/edit prompt
# ---------------------------------------------------------------------------

def prompt_add_position() -> dict | None:
    """Interactive prompt to gather a new position. Returns dict or None."""
    console.print()
    console.print(Rule("[bold cyan]Add / Edit Position[/bold cyan]"))
    console.print("  [dim](Press Enter to cancel at any time)[/dim]")

    def ask(prompt: str, required: bool = True) -> str | None:
        val = console.input(f"  {prompt}: ").strip()
        if not val:
            if required:
                console.print("  [yellow]Cancelled.[/yellow]")
                return None
            return None
        return val

    pos_type = ask("Position type [covered_call / short_put]")
    if not pos_type or pos_type not in ("covered_call", "short_put"):
        console.print("  [red]Invalid type.[/red]")
        return None

    ticker = ask("Ticker (e.g. PFE)")
    if not ticker:
        return None
    ticker = ticker.upper()

    contracts_raw = ask("Contracts (negative for short, e.g. -2)")
    if not contracts_raw:
        return None
    try:
        contracts = int(contracts_raw)
    except ValueError:
        console.print("  [red]Invalid number.[/red]")
        return None

    strike = None
    if pos_type == "short_put":
        strike_raw = ask("Strike price (e.g. 10.50)")
        if not strike_raw:
            return None
        try:
            strike = float(strike_raw)
        except ValueError:
            console.print("  [red]Invalid strike.[/red]")
            return None

    expiry = ask("Expiry date (YYYY-MM-DD, optional)", required=False)
    pnl_raw = ask("Current P&L $ (optional, e.g. -72.00)", required=False)
    pnl = float(pnl_raw) if pnl_raw else None
    pnl_pct_raw = ask("Current P&L % (optional, e.g. -67.92)", required=False)
    pnl_pct = float(pnl_pct_raw) if pnl_pct_raw else None
    value_raw = ask("Current value $ (optional, negative = liability)", required=False)
    value = float(value_raw) if value_raw else None
    notes = ask("Notes (optional)", required=False) or ""

    import uuid
    new_id = "pos_" + uuid.uuid4().hex[:6]

    return {
        "id":       new_id,
        "type":     pos_type,
        "ticker":   ticker,
        "contracts": contracts,
        "strike":   strike,
        "expiry":   expiry,
        "value":    value,
        "pnl":      pnl,
        "pnl_pct":  pnl_pct,
        "currency": "CAD" if ticker in ("BCE", "BNS") else "USD",
        "notes":    notes,
        "premium_collected":   None,
        "cost_basis_per_share": None,
    }


# ---------------------------------------------------------------------------
# Export report
# ---------------------------------------------------------------------------

def export_report(enriched_positions: list[dict], risk: dict,
                  cc_results: list[dict], filename: str = "portfolio_report.txt") -> str:
    """Write a plain-text report to filename. Returns the path."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 72,
        "  WHEEL STRATEGY PORTFOLIO REPORT",
        f"  Generated: {now}",
        "=" * 72,
        "",
        "─── POSITIONS ─────────────────────────────────────────────────────",
    ]

    for pos in enriched_positions:
        spot_str   = f"${pos['spot_price']:.2f}" if pos.get("spot_price") else "N/A"
        strike_str = f"${pos['strike']:.2f}" if pos.get("strike") else "—"
        buf_str    = f"{pos.get('buffer_pct', 0):+.1f}%" if pos.get("buffer_pct") is not None else "—"
        flags      = " [CRITICAL]" if pos.get("is_critical") else ""
        lines.append(
            f"  {pos.get('id','?'):8s}  {pos['type']:14s}  {pos['ticker']:5s}"
            f"  strike={strike_str:8s}  spot={spot_str:8s}  buffer={buf_str:7s}"
            f"  P&L={pos.get('pnl', 0):+.2f} ({pos.get('pnl_pct', 0):+.1f}%){flags}"
        )

    lines += [
        "",
        "─── RISK SUMMARY ───────────────────────────────────────────────────",
        f"  Total P&L:          ${risk['total_pnl']:+,.2f}",
        f"  Capital at Risk:    ${risk['capital_at_risk_usd']:,.2f} USD  /  "
        f"C${risk['capital_at_risk_cad']:,.2f}",
        f"  Risk % of Account:  {risk['risk_pct_of_account']:.1f}%",
        f"  Critical Positions: {risk['critical_count']}",
        "",
        "  Sector Exposure:",
    ]
    for sec, pct in sorted(risk["sector_exposure_pct"].items(),
                           key=lambda x: x[1], reverse=True):
        warn = " ⚠ CONCENTRATED" if sec in risk.get("concentrated_sectors", []) else ""
        lines.append(f"    {sec:30s} {pct:.1f}%{warn}")

    lines += [
        "",
        "─── CC SCREENER (top opportunities) ─────────────────────────────────",
    ]
    for r in cc_results[:6]:
        if "error" not in r:
            lines.append(
                f"  {r['ticker']:5s}  spot=${r['spot']:.2f}  strike=${r['strike']:.2f}"
                f"  DTE={r['dte']}  prem=${r['premium']:.4f}"
                f"  ann_yield={r.get('total_yield_pct', 0):.1f}%"
                + ("  ★ HIGH PRI" if r.get("high_priority") else "")
            )

    lines += ["", "=" * 72]

    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return filename
