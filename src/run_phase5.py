"""
Phase 5 entry point: run the same Phase 3/4 portfolio backtest (MACD entry /
ATR TP-SL exit / cooldown re-entry, inverse-vol sized, daily rebalance) and
open the interactive chart - left: per-symbol candlesticks + entry/exit
markers (switchable), right: portfolio equity curve + drawdown, Sharpe/return
summary, and the per-day portfolio weight ranking, crosshair-synced to the
left pane.
"""

import pandas as pd

from src.engine.backtest import Strategy, run_backtest
from src.engine.portfolio_backtest import align_to_common_index, run_portfolio_backtest
from src.performance.sharpe import sharpe_ratio
from src.rules.entry import MACDGoldenCross
from src.rules.exit import ATRTakeProfitStopLoss
from src.rules.reentry import CooldownReentry
from src.rules.sizing import InverseVolatilitySizer
from src.viz.chart import show_portfolio_backtest

SYMBOLS = ["NVDA", "KO", "XOM", "JPM"]
HISTORY_WEEKS = 5  # MVP cap: ~25 trading days - still clears the 20-day vol lookback, keeps the run fast


def make_strategy() -> Strategy:
    return Strategy(
        entry_rule=MACDGoldenCross(),
        exit_rule=ATRTakeProfitStopLoss(),
        reentry_rule=CooldownReentry(),
    )


def main():
    dfs = {
        s: pd.read_csv(f"data/00-美股ETF历史/raw/{s}_5m.csv", parse_dates=["timestamp"]) for s in SYMBOLS
    }
    cutoff = min(df["timestamp"].max() for df in dfs.values()) - pd.Timedelta(weeks=HISTORY_WEEKS)
    dfs = {s: df[df["timestamp"] >= cutoff].reset_index(drop=True) for s, df in dfs.items()}

    result = run_portfolio_backtest(dfs, make_strategy(), InverseVolatilitySizer())
    aligned = align_to_common_index(dfs)

    # same Sharpe comparison as Phase 4: portfolio-level vs. each symbol run
    # standalone (same rules, same aligned bars, full notional)
    portfolio_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
    sharpe_stats = {"Portfolio": sharpe_ratio(result.equity_curve)}
    for s in SYMBOLS:
        single_result = run_backtest(aligned[s], make_strategy())
        sharpe_stats[s] = sharpe_ratio(single_result.equity_curve)

    show_portfolio_backtest(
        aligned,
        result.trades,
        result.weight_adjustments,
        result.equity_curve,
        result.weights_history,
        sharpe_stats,
        portfolio_return,
    )


if __name__ == "__main__":
    main()
