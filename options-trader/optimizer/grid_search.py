"""
Parameter optimizer: grid search over management parameters.

Tests combinations of profit target %, stop loss %, and roll trigger %
to find the configuration that maximises risk-adjusted return (Sharpe ratio)
or total P&L for a given ticker and date range.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from data.adapters import OptionsDataAdapter
from simulator.analytics import compute_stats
from simulator.engine import SimulationEngine

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """A single parameter combination result."""

    profit_target_pct: float
    stop_loss_pct: float
    roll_trigger_pct: float
    num_trades: int
    win_rate: float
    total_pnl: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    num_rolls: int

    def to_dict(self) -> dict:
        return {
            "profit_target_pct": self.profit_target_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "roll_trigger_pct": self.roll_trigger_pct,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "num_rolls": self.num_rolls,
        }


class GridSearchOptimizer:
    """
    Exhaustively tests parameter combinations for a given ticker and date range.

    Usage:
        optimizer = GridSearchOptimizer(config, adapter)
        results = optimizer.run("AAPL", date(2022,1,1), date(2024,1,1))
        best = optimizer.best_by_sharpe(results)
    """

    def __init__(
        self,
        config: dict,
        adapter: OptionsDataAdapter,
    ) -> None:
        self._config = config
        self._adapter = adapter
        optimizer_cfg = config.get("optimizer", {})

        self._profit_targets = optimizer_cfg.get("profit_target_range", [40, 50, 55, 60])
        self._stop_losses = optimizer_cfg.get("stop_loss_range", [30, 40, 50])
        self._roll_triggers = optimizer_cfg.get("roll_trigger_range", [50, 60, 75])

    def run(
        self,
        ticker: str,
        start: date,
        end: date,
        strategy: str = "short_put",
        profit_targets: Optional[list[float]] = None,
        stop_losses: Optional[list[float]] = None,
        roll_triggers: Optional[list[float]] = None,
        verbose: bool = True,
    ) -> list[GridSearchResult]:
        """
        Run grid search over parameter combinations.

        Args:
            ticker:         Stock symbol
            start:          Simulation start date
            end:            Simulation end date
            strategy:       "short_put" or "covered_call"
            profit_targets: Override profit target % values to test
            stop_losses:    Override stop loss % values to test
            roll_triggers:  Override roll trigger % values to test
            verbose:        Log progress

        Returns:
            List of GridSearchResult sorted by Sharpe ratio descending.
        """
        pts = profit_targets or self._profit_targets
        sls = stop_losses or self._stop_losses
        rts = roll_triggers or self._roll_triggers

        combos = list(itertools.product(pts, sls, rts))
        logger.info(
            "Grid search: %d combinations for %s (%s to %s)",
            len(combos), ticker, start, end,
        )

        engine = SimulationEngine(config=self._config, adapter=self._adapter)
        portfolio_size = float(
            self._config.get("strategy", {}).get("sizing", {}).get("portfolio_size", 50_000)
        )

        results: list[GridSearchResult] = []
        for i, (pt, sl, rt) in enumerate(combos):
            if verbose:
                logger.info(
                    "[%d/%d] Testing: profit_target=%g%% stop_loss=%g%% roll_trigger=%g%%",
                    i + 1, len(combos), pt, sl, rt,
                )

            mgmt_override = {
                "profit_target_pct": pt,
                "stop_loss_pct": sl,
                "roll_trigger_pct": rt,
            }

            try:
                trades, equity_curve = engine.run(
                    ticker=ticker,
                    start=start,
                    end=end,
                    strategy=strategy,
                    mgmt_override=mgmt_override,
                )
                trade_dicts = [t.to_dict() for t in trades]
                stats = compute_stats(trade_dicts, equity_curve, portfolio_size)

                results.append(
                    GridSearchResult(
                        profit_target_pct=pt,
                        stop_loss_pct=sl,
                        roll_trigger_pct=rt,
                        num_trades=stats["num_trades"],
                        win_rate=stats["win_rate"],
                        total_pnl=stats["total_pnl"],
                        annualized_return=stats["annualized_return"],
                        max_drawdown=stats["max_drawdown"],
                        sharpe_ratio=stats["sharpe_ratio"],
                        num_rolls=stats["num_rolls"],
                    )
                )
            except Exception as exc:
                logger.warning("Error in grid search combo (%g,%g,%g): %s", pt, sl, rt, exc)

        # Sort by Sharpe ratio descending
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    # ─── Result analysis helpers ──────────────────────────────────────────────

    @staticmethod
    def best_by_sharpe(results: list[GridSearchResult]) -> Optional[GridSearchResult]:
        """Return the parameter set with the highest Sharpe ratio."""
        if not results:
            return None
        return max(results, key=lambda r: r.sharpe_ratio)

    @staticmethod
    def best_by_return(results: list[GridSearchResult]) -> Optional[GridSearchResult]:
        """Return the parameter set with the highest annualized return."""
        if not results:
            return None
        return max(results, key=lambda r: r.annualized_return)

    @staticmethod
    def to_dataframe(results: list[GridSearchResult]) -> pd.DataFrame:
        """Convert results to a DataFrame for display or export."""
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in results])

    @staticmethod
    def plot_heatmap(
        results: list[GridSearchResult],
        metric: str = "sharpe_ratio",
        fixed_roll_trigger: Optional[float] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a heatmap of profit_target vs stop_loss for a fixed roll trigger.

        Args:
            results:             Grid search results
            metric:              Column to plot ('sharpe_ratio', 'total_pnl', 'win_rate', etc.)
            fixed_roll_trigger:  Fix roll trigger value to use for the 2D slice.
                                 If None, uses the most common value.
            save_path:           Path to save the figure
            show:                Whether to display interactively
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path

            df = GridSearchOptimizer.to_dataframe(results)
            if df.empty:
                return

            if fixed_roll_trigger is None:
                fixed_roll_trigger = df["roll_trigger_pct"].mode().iloc[0]

            slice_df = df[df["roll_trigger_pct"] == fixed_roll_trigger]
            if slice_df.empty:
                logger.warning("No data for roll_trigger=%.0f%%", fixed_roll_trigger)
                return

            pivot = slice_df.pivot_table(
                index="profit_target_pct",
                columns="stop_loss_pct",
                values=metric,
                aggfunc="mean",
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
            plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels([f"{v:.0f}%" for v in pivot.columns])
            ax.set_yticklabels([f"{v:.0f}%" for v in pivot.index])
            ax.set_xlabel("Stop Loss %")
            ax.set_ylabel("Profit Target %")
            ax.set_title(
                f"Optimizer: {metric.replace('_', ' ').title()}\n"
                f"(Roll Trigger = {fixed_roll_trigger:.0f}%)"
            )

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=9, color="black",
                        )

            plt.tight_layout()
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150)
            if show:
                plt.show()
            plt.close()

        except Exception as exc:
            logger.warning("Could not plot heatmap: %s", exc)
