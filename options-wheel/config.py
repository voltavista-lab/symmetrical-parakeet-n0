"""Configuration for Options Wheel Strategy Finder & Simulator."""

# Default watchlist - dividend aristocrats + high-IV blue chips
WATCHLIST = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "MCD", "HD", "ABBV",
    "T", "VZ", "XOM", "CVX", "PM", "MO", "IBM", "MMM", "CAT",
    "BA", "JPM", "BAC", "WFC", "C", "GS", "INTC", "CSCO", "QCOM",
    "TXN", "AMGN", "BMY", "PFE", "MRK", "UPS", "LOW", "TGT",
    "SCHD", "O", "MAIN", "JEPI", "JEPQ",
]

# Tastytrade-style parameters
STRATEGY = {
    "target_dte": 45,
    "dte_range": (30, 60),
    "profit_target_pct": 50,
    "max_loss_close_pct": 40,
    "roll_trigger_pct": 60,
    "min_iv_rank": 30,
    "min_dividend_yield": 1.0,
    "max_stock_price": 200,
    "min_open_interest": 100,
    "min_volume": 50,
    "delta_target_put": -0.30,
    "delta_range_put": (-0.20, -0.40),
    "delta_target_call": 0.30,
    "delta_range_call": (0.20, 0.40),
}
