"""
Options Trading Finder & Simulator — CLI entry point.

Commands:
    screen          Run the stock & options screener
    simulate        Backtest a single ticker with the Tastytrade mechanical strategy
    optimize        Grid search over management parameters for a ticker
    full-pipeline   Screen then simulate the top N candidates
    list-runs       List saved simulation runs from the database
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

# Use a wide fixed width when stdout is not a real TTY (e.g. Colab, pipes).
# Rich defaults to 80 chars in non-TTY mode which causes aggressive truncation.
_IS_TTY = sys.stdout.isatty()
console = Console(width=None if _IS_TTY else 220)


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return config.yaml as a dict."""
    p = Path(config_path)
    if not p.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(config: dict) -> None:
    output_cfg = config.get("output", {})
    level_name = output_cfg.get("log_level", "INFO")
    log_file = output_cfg.get("log_file", "options_trader.log")
    level = getattr(logging, level_name.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# ─── Rich table helpers ───────────────────────────────────────────────────────

def dataframe_to_rich_table(df, title: str = "", max_rows: int = 50) -> Table:
    """Convert a pandas DataFrame to a Rich table."""
    import pandas as pd

    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    if not df.empty:
        # Add rank/index column if it has a name
        if df.index.name:
            table.add_column(df.index.name, style="bold yellow", justify="right")
        for col in df.columns:
            table.add_column(str(col), justify="right")

        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= max_rows:
                table.add_row(*["..." for _ in range(len(df.columns) + (1 if df.index.name else 0))])
                break
            values = []
            if df.index.name:
                values.append(str(idx))
            for v in row:
                values.append("" if (hasattr(v, "__class__") and v.__class__.__name__ == "float" and str(v) == "nan") else str(v))
            table.add_row(*values)
    return table


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--config",
    "-c",
    default="config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Options Trading Finder & Simulator — Tastytrade premium-selling approach."""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj["config"] = cfg
    setup_logging(cfg)

    # Ensure output directories exist
    Path(cfg.get("output", {}).get("csv_dir", "exports")).mkdir(parents=True, exist_ok=True)
    Path(cfg.get("data", {}).get("cache_dir", ".cache")).mkdir(parents=True, exist_ok=True)
    db_dir = Path(cfg.get("output", {}).get("db_path", "data/options_trader.db")).parent
    db_dir.mkdir(parents=True, exist_ok=True)


# ── screen ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--min-yield", "-y", type=float, default=None, help="Minimum dividend yield %")
@click.option("--min-ivr", type=float, default=None, help="Minimum IV Rank")
@click.option("--min-price", type=float, default=None, help="Minimum stock price")
@click.option("--max-price", type=float, default=None, help="Maximum stock price")
@click.option("--min-volume", type=float, default=None, help="Minimum average daily volume")
@click.option("--top-n", "-n", type=int, default=None, help="Number of top candidates to show")
@click.option("--export/--no-export", default=None, help="Export results to CSV")
@click.option("--tickers", "-t", default=None, help="Comma-separated ticker list (overrides universe)")
@click.pass_context
def screen(
    ctx: click.Context,
    min_yield: Optional[float],
    min_ivr: Optional[float],
    min_price: Optional[float],
    max_price: Optional[float],
    min_volume: Optional[float],
    top_n: Optional[int],
    export: Optional[bool],
    tickers: Optional[str],
) -> None:
    """Run the stock & options screener to find premium-selling candidates."""
    from data.adapters import get_adapter
    from data.db import Database
    from screener.universe import UniverseFilter
    from screener.options_chain import OptionsChainAnalyser
    from screener.scoring import CandidateScorer, display_columns

    cfg = ctx.obj["config"]

    # Resolve parameters (CLI overrides config)
    if min_yield is not None:
        cfg.setdefault("screener", {})["min_dividend_yield"] = min_yield
    if min_ivr is not None:
        cfg.setdefault("screener", {})["min_ivr"] = min_ivr
    if min_price is not None:
        cfg.setdefault("screener", {})["min_price"] = min_price
    if max_price is not None:
        cfg.setdefault("screener", {})["max_price"] = max_price
    if min_volume is not None:
        cfg.setdefault("screener", {})["min_avg_daily_volume"] = min_volume

    n = top_n or cfg.get("screener", {}).get("top_n", 20)
    do_export = export if export is not None else cfg.get("output", {}).get("export_csv", True)

    adapter = get_adapter(cfg)
    console.print(f"\n[bold green]Options Screener[/bold green] — using [cyan]{adapter.get_name()}[/cyan] adapter\n")

    # Universe
    ufilter = UniverseFilter(cfg, adapter)
    ticker_list: Optional[list[str]] = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]

    with console.status("[bold]Filtering stock universe...[/bold]"):
        qualified = ufilter.filter_stocks(
            tickers=ticker_list,
            min_yield=min_yield,
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume,
        )

    if not qualified:
        console.print("[yellow]No stocks passed the universe filters. Try relaxing your criteria.[/yellow]")
        return

    console.print(f"[green]{len(qualified)}[/green] stocks passed universe filters. Analysing options chains...\n")

    # Options analysis
    analyser = OptionsChainAnalyser(cfg, adapter)
    candidates: list[dict] = []

    with console.status("[bold]Fetching options chains...[/bold]") as status:
        for quote in qualified:
            status.update(f"[bold]Analysing {quote.ticker}...[/bold]")
            result = analyser.analyse(quote)
            if result is None:
                continue
            # Apply IVR filter
            ivr_thresh = cfg.get("screener", {}).get("min_ivr", 30)
            if result.get("ivr", 0) < ivr_thresh:
                logging.getLogger(__name__).debug(
                    "%s IVR %.1f < %.1f, skipping", quote.ticker, result.get("ivr", 0), ivr_thresh
                )
                continue
            candidates.append(result)

    if not candidates:
        console.print("[yellow]No candidates after options analysis. Consider lowering --min-ivr.[/yellow]")
        return

    # Score and rank
    scorer = CandidateScorer(cfg)
    ranked = scorer.score_candidates(candidates)
    top = ranked.head(n)

    # Display
    display = display_columns(top)
    table = dataframe_to_rich_table(display, title=f"Top {n} Premium-Selling Candidates", max_rows=n)
    console.print(table)

    # Summary
    console.print(f"\n[bold]Found {len(ranked)} candidates[/bold] (showing top {min(n, len(ranked))})")
    console.print(
        f"Filters: yield≥{cfg['screener'].get('min_dividend_yield', 2)}%, "
        f"IVR≥{cfg['screener'].get('min_ivr', 30)}, "
        f"price ${cfg['screener'].get('min_price', 15)}–${cfg['screener'].get('max_price', 200)}"
    )

    # Save to DB
    db = Database(cfg.get("output", {}).get("db_path", "data/options_trader.db"))
    run_id = db.save_screener_run(cfg)
    candidate_dicts = top.reset_index().to_dict("records")
    db.save_screener_candidates(run_id, candidate_dicts)

    # Export CSV
    if do_export:
        csv_dir = cfg.get("output", {}).get("csv_dir", "exports")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{csv_dir}/screener_{ts}.csv"
        ranked.to_csv(csv_path)
        console.print(f"[dim]Exported to {csv_path}[/dim]")


# ── simulate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", "-t", required=True, help="Stock symbol to simulate")
@click.option(
    "--start",
    "-s",
    default="2022-01-01",
    show_default=True,
    help="Simulation start date (YYYY-MM-DD)",
)
@click.option(
    "--end",
    "-e",
    default=None,
    help="Simulation end date (YYYY-MM-DD); defaults to today",
)
@click.option(
    "--strategy",
    "-S",
    default="short_put",
    type=click.Choice(["short_put", "covered_call"]),
    show_default=True,
    help="Options strategy to simulate",
)
@click.option("--optimize", is_flag=True, default=False, help="Compare multiple parameter sets")
@click.option("--plot/--no-plot", default=False, help="Show equity curve chart")
@click.option("--export/--no-export", default=None, help="Export trade log to CSV")
@click.pass_context
def simulate(
    ctx: click.Context,
    ticker: str,
    start: str,
    end: Optional[str],
    strategy: str,
    optimize: bool,
    plot: bool,
    export: Optional[bool],
) -> None:
    """Backtest a short-put or covered-call strategy using Tastytrade mechanical rules."""
    from data.adapters import get_adapter
    from data.db import Database
    from simulator.engine import SimulationEngine
    from simulator.analytics import (
        compute_stats,
        format_trade_log,
        format_stats_table,
        plot_equity_curve,
        plot_pnl_distribution,
        compare_parameter_sets,
    )

    cfg = ctx.obj["config"]
    adapter = get_adapter(cfg)
    ticker = ticker.upper()

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()

    portfolio_size = float(
        cfg.get("strategy", {}).get("sizing", {}).get("portfolio_size", 50_000)
    )

    console.print(
        f"\n[bold green]Simulation[/bold green] — [cyan]{ticker}[/cyan] "
        f"[{strategy.replace('_', ' ')}] {start_date} → {end_date}\n"
    )

    engine = SimulationEngine(cfg, adapter)

    if optimize:
        # Run multiple parameter sets for comparison
        mgmt_cfg = cfg.get("strategy", {}).get("management", {})
        profit_range = mgmt_cfg.get("profit_target_range", [50, 55])
        param_sets = [
            {"profit_target_pct": pt, "stop_loss_pct": 40, "roll_trigger_pct": 60, "label": f"PT={pt}%"}
            for pt in profit_range
        ]

        comparison_results = []
        for params in param_sets:
            label = params.pop("label")
            with console.status(f"[bold]Running {label}...[/bold]"):
                trades, eq = engine.run(ticker, start_date, end_date, strategy, mgmt_override=params)
                trade_dicts = [t.to_dict() for t in trades]
                stats = compute_stats(trade_dicts, eq, portfolio_size)
            comparison_results.append({"label": label, "stats": stats, **params})

        comp_df = compare_parameter_sets(comparison_results)
        table = dataframe_to_rich_table(comp_df, title="Parameter Comparison", max_rows=20)
        console.print(table)

    else:
        # Single run
        with console.status(f"[bold]Running simulation for {ticker}...[/bold]"):
            trades, equity_curve = engine.run(ticker, start_date, end_date, strategy)

        if not trades:
            console.print("[yellow]No trades were executed. Check IVR threshold or date range.[/yellow]")
            return

        trade_dicts = [t.to_dict() for t in trades]
        stats = compute_stats(trade_dicts, equity_curve, portfolio_size)

        # Stats table
        stats_df = format_stats_table(stats, ticker, strategy)
        stats_table = dataframe_to_rich_table(stats_df, title="Simulation Summary")
        console.print(stats_table)

        # Trade log
        log_df = format_trade_log(trade_dicts)
        log_table = dataframe_to_rich_table(log_df, title="Trade Log", max_rows=30)
        console.print(log_table)

        # Save to DB
        db = Database(cfg.get("output", {}).get("db_path", "data/options_trader.db"))
        run_id = db.save_simulation_run(ticker, strategy, start_date, end_date, cfg, stats)
        db.save_trade_log(run_id, trade_dicts)
        console.print(f"[dim]Saved to database (run_id={run_id})[/dim]")

        # Export CSV
        do_export = export if export is not None else cfg.get("output", {}).get("export_csv", True)
        if do_export:
            csv_dir = cfg.get("output", {}).get("csv_dir", "exports")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"{csv_dir}/sim_{ticker}_{strategy}_{ts}.csv"
            log_df.to_csv(csv_path, index=False)
            console.print(f"[dim]Trade log exported to {csv_path}[/dim]")

        # Plots
        if plot:
            plot_equity_curve(equity_curve, ticker, strategy, show=True)
            plot_pnl_distribution(trade_dicts, ticker, strategy, show=True)


# ── optimize ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", "-t", required=True, help="Stock symbol")
@click.option("--start", "-s", default="2022-01-01", show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", default=None, help="End date (YYYY-MM-DD); defaults to today")
@click.option(
    "--strategy",
    "-S",
    default="short_put",
    type=click.Choice(["short_put", "covered_call"]),
    show_default=True,
)
@click.option("--plot/--no-plot", default=False, help="Show heatmap")
@click.option("--export/--no-export", default=True, help="Export results to CSV")
@click.pass_context
def optimize(
    ctx: click.Context,
    ticker: str,
    start: str,
    end: Optional[str],
    strategy: str,
    plot: bool,
    export: bool,
) -> None:
    """Grid search over management parameters to find the optimal configuration."""
    from data.adapters import get_adapter
    from optimizer.grid_search import GridSearchOptimizer

    cfg = ctx.obj["config"]
    adapter = get_adapter(cfg)
    ticker = ticker.upper()

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()

    console.print(
        f"\n[bold green]Optimizer[/bold green] — [cyan]{ticker}[/cyan] "
        f"[{strategy}] {start_date} → {end_date}\n"
    )

    optimizer_instance = GridSearchOptimizer(cfg, adapter)

    with console.status("[bold]Running grid search...[/bold]"):
        results = optimizer_instance.run(
            ticker=ticker,
            start=start_date,
            end=end_date,
            strategy=strategy,
            verbose=False,
        )

    if not results:
        console.print("[yellow]No results produced. Check ticker and date range.[/yellow]")
        return

    df = GridSearchOptimizer.to_dataframe(results)
    table = dataframe_to_rich_table(df.head(20), title="Optimizer Results (Top 20 by Sharpe)")
    console.print(table)

    best_sharpe = GridSearchOptimizer.best_by_sharpe(results)
    best_return = GridSearchOptimizer.best_by_return(results)

    if best_sharpe:
        console.print(
            f"\n[bold]Best by Sharpe:[/bold] "
            f"profit_target={best_sharpe.profit_target_pct:.0f}% "
            f"stop_loss={best_sharpe.stop_loss_pct:.0f}% "
            f"roll_trigger={best_sharpe.roll_trigger_pct:.0f}% "
            f"→ Sharpe={best_sharpe.sharpe_ratio:.3f}"
        )
    if best_return:
        console.print(
            f"[bold]Best by Return:[/bold] "
            f"profit_target={best_return.profit_target_pct:.0f}% "
            f"stop_loss={best_return.stop_loss_pct:.0f}% "
            f"roll_trigger={best_return.roll_trigger_pct:.0f}% "
            f"→ Ann. Return={best_return.annualized_return:.2f}%"
        )

    if export:
        csv_dir = cfg.get("output", {}).get("csv_dir", "exports")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{csv_dir}/optimizer_{ticker}_{ts}.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"[dim]Results exported to {csv_path}[/dim]")

    if plot:
        GridSearchOptimizer.plot_heatmap(results, metric="sharpe_ratio", show=True)


# ── full-pipeline ─────────────────────────────────────────────────────────────

@cli.command("full-pipeline")
@click.option("--top", "-n", default=5, show_default=True, help="Number of top candidates to simulate")
@click.option("--start", "-s", default="2022-01-01", show_default=True, help="Simulation start date")
@click.option("--end", "-e", default=None, help="Simulation end date; defaults to today")
@click.option(
    "--strategy",
    "-S",
    default="short_put",
    type=click.Choice(["short_put", "covered_call"]),
    show_default=True,
)
@click.pass_context
def full_pipeline(
    ctx: click.Context,
    top: int,
    start: str,
    end: Optional[str],
    strategy: str,
) -> None:
    """Screen the universe then simulate the top N candidates."""
    from data.adapters import get_adapter
    from screener.universe import UniverseFilter
    from screener.options_chain import OptionsChainAnalyser
    from screener.scoring import CandidateScorer, display_columns
    from simulator.engine import SimulationEngine
    from simulator.analytics import compute_stats, format_stats_table

    cfg = ctx.obj["config"]
    adapter = get_adapter(cfg)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()

    console.print("\n[bold green]Full Pipeline[/bold green] — screen → simulate\n")

    # ── Step 1: Screen ──────────────────────────────────────────────────────
    console.rule("[bold]Step 1: Screening[/bold]")
    ufilter = UniverseFilter(cfg, adapter)
    analyser = OptionsChainAnalyser(cfg, adapter)
    scorer = CandidateScorer(cfg)

    with console.status("[bold]Filtering universe...[/bold]"):
        qualified = ufilter.filter_stocks()

    console.print(f"[green]{len(qualified)}[/green] stocks passed universe filters.\n")

    candidates: list[dict] = []
    with console.status("[bold]Analysing options chains...[/bold]") as status:
        for quote in qualified:
            status.update(f"Analysing [cyan]{quote.ticker}[/cyan]...")
            result = analyser.analyse(quote)
            if result:
                candidates.append(result)

    if not candidates:
        console.print("[yellow]No candidates found. Exiting.[/yellow]")
        return

    ranked = scorer.score_candidates(candidates)
    top_candidates = ranked.head(top)
    tickers_to_simulate = top_candidates["ticker"].tolist()

    display = display_columns(top_candidates)
    table = dataframe_to_rich_table(display, title=f"Top {top} Screener Candidates")
    console.print(table)

    # ── Step 2: Simulate each ──────────────────────────────────────────────
    console.rule("[bold]Step 2: Simulating[/bold]")
    engine = SimulationEngine(cfg, adapter)
    portfolio_size = float(cfg.get("strategy", {}).get("sizing", {}).get("portfolio_size", 50_000))

    summary_rows = []
    for ticker in tickers_to_simulate:
        with console.status(f"[bold]Simulating {ticker}...[/bold]"):
            trades, equity_curve = engine.run(ticker, start_date, end_date, strategy)
        trade_dicts = [t.to_dict() for t in trades]
        stats = compute_stats(trade_dicts, equity_curve, portfolio_size)
        summary_rows.append(
            {
                "Ticker": ticker,
                "Trades": stats["num_trades"],
                "Win %": f"{stats['win_rate']:.1f}",
                "Total P&L": f"${stats['total_pnl']:,.0f}",
                "Ann. Return": f"{stats['annualized_return']:.2f}%",
                "Max DD": f"${stats['max_drawdown']:,.0f}",
                "Sharpe": f"{stats['sharpe_ratio']:.3f}",
                "Rolls": stats["num_rolls"],
            }
        )

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_table = dataframe_to_rich_table(summary_df, title="Pipeline Summary")
    console.print(summary_table)


# ── list-runs ─────────────────────────────────────────────────────────────────

@cli.command("list-runs")
@click.pass_context
def list_runs(ctx: click.Context) -> None:
    """List saved simulation runs from the database."""
    from data.db import Database

    cfg = ctx.obj["config"]
    db = Database(cfg.get("output", {}).get("db_path", "data/options_trader.db"))
    runs = db.list_simulation_runs()

    if runs.empty:
        console.print("[yellow]No simulation runs found in database.[/yellow]")
        return

    cols = ["id", "ticker", "strategy", "start_date", "end_date", "win_rate", "total_pnl", "annualized_return", "num_trades", "run_at"]
    available = [c for c in cols if c in runs.columns]
    table = dataframe_to_rich_table(runs[available].head(30), title="Saved Simulation Runs")
    console.print(table)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
