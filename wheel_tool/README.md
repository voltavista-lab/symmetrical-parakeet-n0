# Wheel Strategy Portfolio Manager

A terminal-based Python tool for monitoring, managing, and optimising an
income-focused **wheel strategy** (covered calls + cash-secured puts).

## Features

| Menu | Description |
|------|-------------|
| **[1] Dashboard** | Colour-coded position table — green (profit), yellow (breakeven), red (loss). Flags critical positions (>50% loss or ITM puts). |
| **[2] Roll Analysis** | For each underwater short put, shows roll scenarios out 30/60/90 days at same or reduced strike. Highlights best net-credit roll. |
| **[3] CC Screener** | Scans watchlist tickers for covered call opportunities at ~30 DTE. Shows premium, delta, IV rank, and annualised yield. Flags >10% yield as high-priority. |
| **[4] Wheel Tracker** | Tracks effective cost basis and breakeven per ticker after accounting for all premiums collected. Suggests next CC when shares are assigned. |
| **[5] Risk Summary** | Capital at risk from all short puts, sector concentration warnings (>40%), and CAD/USD exposure breakdown. |
| **[6] Add/Edit Position** | Interactive prompt to add or update a position; optionally records premiums collected to history. |
| **[7] Export Report** | Writes `portfolio_report.txt` with a full plain-text summary. |

---

## Setup

### Requirements
- Python 3.11+
- Internet access for live data (yfinance)

### Install

```bash
cd wheel_tool
pip install -r requirements.txt
```

### Run

```bash
python portfolio.py
```

Pass `--debug` for full tracebacks on errors:

```bash
python portfolio.py --debug
```

---

## File Structure

```
wheel_tool/
├── portfolio.py          # Entry point + menu loop
├── positions.json        # Persistent position storage (edit directly or via menu)
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── data_fetcher.py   # yfinance price + options chain fetching
    ├── analysis.py       # Roll engine, CC screener, wheel tracker, risk calcs
    └── display.py        # Rich terminal UI (tables, panels, colour coding)
```

---

## Managing Positions

### Editing `positions.json` directly

The file contains two main sections:

```jsonc
{
  "account": {
    "value_cad": 18788.00,        // Total account value in CAD
    "buying_power_cad": 20322.00,
    "cash_cad": 9310.56,
    "cash_usd": 0.00
  },
  "positions": [
    {
      "id": "pos_001",            // Unique ID (any string)
      "type": "covered_call",     // "covered_call" or "short_put"
      "ticker": "PFE",
      "contracts": 1,             // Positive = long/CC, Negative = short put
      "strike": null,             // Required for short_put
      "expiry": "2025-03-21",     // Optional: YYYY-MM-DD
      "premium_collected": 0.45,  // Optional: $/share received when opened
      "cost_basis_per_share": 26.50, // Optional: your actual cost basis
      "value": 2656.50,           // Current position value
      "pnl": 155.81,              // Current P&L in $
      "pnl_pct": 6.23,            // Current P&L in %
      "currency": "USD",
      "notes": ""
    }
  ],
  "premium_history": [
    // Each entry represents a premium received on a wheel trade
    {
      "position_id": "pos_001",
      "ticker": "PFE",
      "premium": 45.00,           // Total $ received (not per share)
      "type": "covered_call"
    }
  ],
  "screener_watchlist": ["PFE", "T", "BCE", "BNS", "VALE", "CLF"]
}
```

### Via the menu

Use **[6] Add/Edit Position** for a guided prompt. The position is saved
automatically to `positions.json`.

### Updating account values

Use **[u] Update account values** from the main menu.

---

## How Roll Analysis Works

For each short put position:
1. Fetches live options chains at ~30, 60, and 90 DTE
2. Calculates **net credit** = new premium received − cost to close current position
3. Evaluates same strike + strike reductions of $0.50 and $1.00
4. Sorts by net credit (best first) and highlights any net-credit roll in green

If no options data is available, falls back to a Black-Scholes approximation
using 30-day historical volatility.

---

## How CC Screener Works

For each ticker in `screener_watchlist`:
1. Finds the options expiry nearest to 30 DTE
2. Selects OTM call strikes with delta ≤ 0.40
3. Ranks by annualised yield = `(premium / strike) × (365 / DTE) × 100`
4. Adds dividend yield for **total annualised yield**
5. Flags as **★ HIGH PRI** if total yield ≥ 10%

---

## Updating to New Positions

After closing or rolling a position:
1. Edit `positions.json` — update `value`, `pnl`, `pnl_pct`, or remove the entry
2. Optionally add a `premium_history` entry with the credit received
3. Restart `portfolio.py` (or press **[r] Refresh** to clear the session cache)

---

## Notes

- **USD/CAD rate** is hardcoded at 1.36 in `analysis.py:SECTOR_MAP` — update
  `usd_cad = 1.36` in `compute_risk_summary()` to adjust.
- yfinance data can occasionally be delayed or missing. The tool falls back
  to Black-Scholes estimates using historical volatility when needed.
- CAD-traded tickers (`BCE`, `BNS`) are automatically tagged as CAD currency.
  Add more to `CAD_TICKERS` in `data_fetcher.py` if needed.
