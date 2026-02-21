# Options Wheel Strategy Finder - Progress

## Phase 1: Config & Stock Screener - COMPLETE
- Created `config.py` with watchlist and strategy params
- Created `screener/stock_filter.py` with `get_stock_fundamentals()` and `filter_watchlist()`
- Created `tests/test_config.py` (8 tests) and `tests/test_screener.py` (7 stock filter tests)
- All tests pass

## Phase 2: Options Scanner - COMPLETE
- Created `screener/options_scanner.py` with `get_options_chain()`, `calculate_iv_rank()`, `scan_for_candidates()`
- Black-Scholes delta estimation via `_estimate_delta()`
- Composite scoring with IV rank, premium yield, dividend yield, liquidity
- Added 12 scanner tests to `tests/test_screener.py`
- All tests pass

## Phase 3: Strategy Logic & Position Management - COMPLETE
- Created `strategy/short_put.py` with scoring and selection
- Created `strategy/covered_call.py` with scoring and selection
- Created `strategy/position_mgmt.py` with profit target, stop loss, roll, position summary
- Created `tests/test_strategy.py` (23 tests)
- All tests pass

## Phase 4: CLI App - COMPLETE
- Created `app/cli.py` with `scan`, `analyze`, `portfolio`, `backtest` commands
- Rich table output with plain-text fallback
- Created `positions.json` for portfolio tracking
- `python app/cli.py portfolio` works correctly

## Phase 5: Backtest Simulator - COMPLETE
- Created `simulator/backtest.py` with `simulate_wheel()` engine
- Black-Scholes premium estimation, HV computation, wheel state machine
- Created `simulator/report.py` with formatted text reports
- Created `tests/test_simulator.py` (11 tests)
- All 61 tests pass

## Phase 6: Streamlit Dashboard - NOT STARTED (optional)

## Status: Phases 1-5 COMPLETE, 61/61 tests passing
