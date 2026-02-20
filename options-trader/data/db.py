"""SQLite persistence layer for screener results and simulation runs."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite wrapper for storing screener results and simulation data.

    Tables:
        screener_results  — snapshots of screener runs
        screener_candidates — individual stock candidates per run
        simulation_runs   — metadata for each backtest run
        trade_log         — individual trade records within a simulation
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS screener_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at      TEXT NOT NULL,
        config_json TEXT,
        notes       TEXT
    );

    CREATE TABLE IF NOT EXISTS screener_candidates (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          INTEGER REFERENCES screener_results(id),
        ticker          TEXT NOT NULL,
        price           REAL,
        sector          TEXT,
        dividend_yield  REAL,
        ivr             REAL,
        hv30            REAL,
        iv_hv_spread    REAL,
        avg_volume      INTEGER,
        best_put_strike REAL,
        best_put_exp    TEXT,
        best_put_dte    INTEGER,
        best_put_premium REAL,
        best_put_premium_pct REAL,
        best_put_delta  REAL,
        best_put_iv     REAL,
        best_put_pop    REAL,
        ex_dividend_date TEXT,
        composite_score REAL,
        data_json       TEXT
    );

    CREATE TABLE IF NOT EXISTS simulation_runs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at          TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        strategy        TEXT NOT NULL,
        start_date      TEXT NOT NULL,
        end_date        TEXT NOT NULL,
        config_json     TEXT,
        win_rate        REAL,
        total_pnl       REAL,
        annualized_return REAL,
        max_drawdown    REAL,
        avg_days_in_trade REAL,
        sharpe_ratio    REAL,
        num_trades      INTEGER,
        num_rolls       INTEGER,
        notes           TEXT
    );

    CREATE TABLE IF NOT EXISTS trade_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          INTEGER REFERENCES simulation_runs(id),
        entry_date      TEXT NOT NULL,
        exit_date       TEXT,
        ticker          TEXT NOT NULL,
        strategy        TEXT NOT NULL,
        strike          REAL NOT NULL,
        expiration      TEXT NOT NULL,
        dte_at_entry    INTEGER,
        credit_received REAL,
        exit_price      REAL,
        pnl             REAL,
        pnl_pct         REAL,
        exit_reason     TEXT,
        roll_number     INTEGER DEFAULT 0,
        data_json       TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_candidates_run ON screener_candidates(run_id);
    CREATE INDEX IF NOT EXISTS idx_trade_log_run  ON trade_log(run_id);
    CREATE INDEX IF NOT EXISTS idx_trade_log_ticker ON trade_log(ticker);
    """

    def __init__(self, db_path: str = "data/options_trader.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.DDL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─── Screener ─────────────────────────────────────────────────────────────

    def save_screener_run(self, config: dict, notes: str = "") -> int:
        """Insert a screener run record and return its ID."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO screener_results (run_at, config_json, notes) VALUES (?,?,?)",
                (datetime.utcnow().isoformat(), json.dumps(config), notes),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def save_screener_candidates(self, run_id: int, candidates: list[dict]) -> None:
        """Bulk insert candidate rows for a screener run."""
        rows = []
        for c in candidates:
            rows.append(
                (
                    run_id,
                    c.get("ticker"),
                    c.get("price"),
                    c.get("sector"),
                    c.get("dividend_yield"),
                    c.get("ivr"),
                    c.get("hv30"),
                    c.get("iv_hv_spread"),
                    c.get("avg_volume"),
                    c.get("best_put_strike"),
                    str(c.get("best_put_exp", "")),
                    c.get("best_put_dte"),
                    c.get("best_put_premium"),
                    c.get("best_put_premium_pct"),
                    c.get("best_put_delta"),
                    c.get("best_put_iv"),
                    c.get("best_put_pop"),
                    str(c.get("ex_dividend_date", "")),
                    c.get("composite_score"),
                    json.dumps({k: v for k, v in c.items() if not isinstance(v, (pd.DataFrame,))}),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO screener_candidates
                   (run_id, ticker, price, sector, dividend_yield, ivr, hv30,
                    iv_hv_spread, avg_volume, best_put_strike, best_put_exp,
                    best_put_dte, best_put_premium, best_put_premium_pct,
                    best_put_delta, best_put_iv, best_put_pop, ex_dividend_date,
                    composite_score, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def get_latest_screener_candidates(self) -> pd.DataFrame:
        """Return candidates from the most recent screener run."""
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT c.* FROM screener_candidates c
                   JOIN screener_results r ON c.run_id = r.id
                   ORDER BY r.run_at DESC, c.composite_score DESC"""
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ─── Simulation ───────────────────────────────────────────────────────────

    def save_simulation_run(
        self,
        ticker: str,
        strategy: str,
        start_date: date,
        end_date: date,
        config: dict,
        stats: dict,
        notes: str = "",
    ) -> int:
        """Insert a simulation run and return its ID."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO simulation_runs
                   (run_at, ticker, strategy, start_date, end_date, config_json,
                    win_rate, total_pnl, annualized_return, max_drawdown,
                    avg_days_in_trade, sharpe_ratio, num_trades, num_rolls, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.utcnow().isoformat(),
                    ticker,
                    strategy,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    json.dumps(config),
                    stats.get("win_rate"),
                    stats.get("total_pnl"),
                    stats.get("annualized_return"),
                    stats.get("max_drawdown"),
                    stats.get("avg_days_in_trade"),
                    stats.get("sharpe_ratio"),
                    stats.get("num_trades"),
                    stats.get("num_rolls"),
                    notes,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def save_trade_log(self, run_id: int, trades: list[dict]) -> None:
        """Bulk insert trade log entries for a simulation run."""
        rows = []
        for t in trades:
            rows.append(
                (
                    run_id,
                    str(t.get("entry_date", "")),
                    str(t.get("exit_date", "")),
                    t.get("ticker"),
                    t.get("strategy"),
                    t.get("strike"),
                    str(t.get("expiration", "")),
                    t.get("dte_at_entry"),
                    t.get("credit_received"),
                    t.get("exit_price"),
                    t.get("pnl"),
                    t.get("pnl_pct"),
                    t.get("exit_reason"),
                    t.get("roll_number", 0),
                    json.dumps({k: str(v) for k, v in t.items()}),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO trade_log
                   (run_id, entry_date, exit_date, ticker, strategy, strike,
                    expiration, dte_at_entry, credit_received, exit_price,
                    pnl, pnl_pct, exit_reason, roll_number, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def get_trade_log(self, run_id: Optional[int] = None, ticker: Optional[str] = None) -> pd.DataFrame:
        """Fetch trade log, optionally filtered by run or ticker."""
        where_clauses, params = [], []
        if run_id is not None:
            where_clauses.append("run_id = ?")
            params.append(run_id)
        if ticker:
            where_clauses.append("ticker = ?")
            params.append(ticker)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as conn:
            cur = conn.execute(f"SELECT * FROM trade_log {where} ORDER BY entry_date", params)
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def list_simulation_runs(self) -> pd.DataFrame:
        """List all simulation runs summary."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM simulation_runs ORDER BY run_at DESC")
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
