"""
Week 6: multi-ticker portfolio version of the pullback strategy - 12 tickers
across equities + commodity ETF proxies (USO/GLD/SLV), daily inverse-ATR-vol
weighted, opened in a single window
with a searchable ticker switch on the left and portfolio-level stats on
the right (see `viz.chart.show_portfolio_pullback_backtest`).

Clipped to the overlapping date range every symbol's CSV actually has data
for (some were fetched ~10 days later than others), so every ticker's daily
weight/rebalance index lines up exactly - no partial-history symbol quietly
skewing early weights.
"""

import pandas as pd

from src.config import BacktestConfig
from src.engine.portfolio_pullback_backtest import run_portfolio_pullback_backtest
from src.viz.chart import show_portfolio_pullback_backtest

# ---------------------------------------------------------------------------
# Tuning knobs - edit these; nothing below the divider needs to change.
# Flip the two USE_* flags to False to recover the full backtest.
# ---------------------------------------------------------------------------
ALL_SYMBOLS = ["NVDA", "KO", "XOM", "JPM", "UNH", "NKE", "INTC", "FCX", "DAL", "USO", "GLD", "SLV"]
DEV_SYMBOLS = ["NVDA", "XOM", "GLD"]        # fast subset: one high-vol, one energy, one metal
USE_DEV_SUBSET = True                       # False -> full 12-ticker universe

FULL_RANGE = ("2024-07-18", "2026-07-17")
DEV_RANGE = ("2024-07-18", "2024-10-18")    # ~3 months, past the ~2.5-week 2h warm-up
USE_DEV_RANGE = True                        # False -> full 2-year sample

CONFIG = BacktestConfig()                   # strategy switches (see src/config.py)
SHOW_CHART = True                           # False -> print metrics only, skip the (blocking) chart
# ---------------------------------------------------------------------------

SYMBOLS = DEV_SYMBOLS if USE_DEV_SUBSET else ALL_SYMBOLS
START, END = DEV_RANGE if USE_DEV_RANGE else FULL_RANGE


def main():
    dfs = {}
    for symbol in SYMBOLS:
        df = pd.read_csv(f"data/raw/{symbol}_5m.csv", parse_dates=["timestamp"])
        df = df[(df["timestamp"] >= START) & (df["timestamp"] <= END)].reset_index(drop=True)
        dfs[symbol] = df

    result = run_portfolio_pullback_backtest(dfs, config=CONFIG)

    print(f"Portfolio: Return {result.portfolio_return:.2%}, Sharpe {result.portfolio_sharpe:.2f}")
    for symbol in sorted(result.per_ticker):
        r = result.per_ticker[symbol]
        print(
            f"  {symbol}: Return {r.total_return:.2%}, Sharpe {r.sharpe:.2f}, "
            f"batches {r.n_batches}, winrate {r.win_rate:.1%}"
        )

    if SHOW_CHART:
        show_portfolio_pullback_backtest(dfs, result)


if __name__ == "__main__":
    main()
