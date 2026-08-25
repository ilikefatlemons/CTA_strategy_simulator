"""
v1.1 Part D: run four independent strategy variants - 5m, 15m, 30m, 1h - and
open the multi-timeframe frontend.

Each variant is a genuinely different strategy, not the same trades redrawn
on a different candle grouping: the 5m variant is the unmodified v1.0
strategy (MACDGoldenCross/ATRTakeProfitStopLoss on native 5m bars); the
15m/30m/1h variants swap in HigherTFMACDGoldenCross/HigherTFATRTakeProfitStopLoss
(src/rules/higher_tf.py), which read that timeframe's own live, still-forming
MACD/ATR (src/data/higher_tf_indicators.py) instead of the 5m-native ones -
a complete replacement of the signal source, not a 5m+higher-TF combined
filter. Execution grain (5m bar-by-bar decisions/fills), reentry rule, and
position sizer are identical across all four, so any difference in the
resulting trades/equity/Sharpe is attributable to the entry/exit signal's
timeframe alone.
"""

import pandas as pd

from src.data.higher_tf_indicators import live_higher_tf_indicators
from src.data.resample import resample_ohlcv
from src.engine.backtest import Strategy, run_backtest
from src.engine.portfolio_backtest import align_to_common_index, run_portfolio_backtest
from src.performance.sharpe import sharpe_ratio
from src.rules.entry import MACDGoldenCross
from src.rules.exit import ATRTakeProfitStopLoss
from src.rules.higher_tf import HigherTFATRTakeProfitStopLoss, HigherTFMACDGoldenCross
from src.rules.reentry import CooldownReentry
from src.rules.sizing import InverseVolatilitySizer
from src.viz.chart import show_multi_timeframe_backtest

SYMBOLS = ["NVDA", "KO", "XOM", "JPM"]
HISTORY_WEEKS = 5  # same window as v1.0 Phase 5, for continuity
TIMEFRAMES = {"5m": None, "15m": "15min", "30m": "30min", "1h": "1h"}


def main():
    dfs = {
        s: pd.read_csv(f"data/00-美股ETF历史/raw/{s}_5m.csv", parse_dates=["timestamp"]) for s in SYMBOLS
    }
    cutoff = min(df["timestamp"].max() for df in dfs.values()) - pd.Timedelta(weeks=HISTORY_WEEKS)
    dfs = {s: df[df["timestamp"] >= cutoff].reset_index(drop=True) for s, df in dfs.items()}
    aligned = align_to_common_index(dfs)

    # Precompute each higher timeframe's live indicators once per symbol, on
    # the *aligned* (common-index) series - entry_bar_idx in the backtest
    # indexes into aligned[s], so this must match that indexing exactly.
    live_by_tf = {
        tf: {s: live_higher_tf_indicators(aligned[s], rule) for s in SYMBOLS}
        for tf, rule in TIMEFRAMES.items()
        if rule is not None
    }
    # Resampled candles for the higher-timeframe panes' visual display.
    display_candles = {
        tf: (aligned if rule is None else {s: resample_ohlcv(aligned[s], rule) for s in SYMBOLS})
        for tf, rule in TIMEFRAMES.items()
    }

    def make_strategy(tf: str) -> Strategy:
        if tf == "5m":
            return Strategy(
                entry_rule=MACDGoldenCross(), exit_rule=ATRTakeProfitStopLoss(), reentry_rule=CooldownReentry()
            )
        live = live_by_tf[tf]
        return Strategy(
            entry_rule=HigherTFMACDGoldenCross(live),
            exit_rule=HigherTFATRTakeProfitStopLoss(live),
            reentry_rule=CooldownReentry(),
        )

    variants = {}
    for tf in TIMEFRAMES:
        result = run_portfolio_backtest(dfs, make_strategy(tf), InverseVolatilitySizer())
        portfolio_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
        sharpe_stats = {"Portfolio": sharpe_ratio(result.equity_curve)}
        for s in SYMBOLS:
            single_result = run_backtest(aligned[s], make_strategy(tf))
            sharpe_stats[s] = sharpe_ratio(single_result.equity_curve)

        print(
            f"[{tf}] {len(result.trades)} trades, portfolio return {portfolio_return:.2%}, "
            f"Sharpe {sharpe_stats['Portfolio']:.2f}"
        )

        variants[tf] = {
            "candles": display_candles[tf],
            "rule": TIMEFRAMES[tf],
            "trades": result.trades,
            "weight_adjustments": result.weight_adjustments,
            "equity_curve": result.equity_curve,
            "weights_history": result.weights_history,
            "sharpe_stats": sharpe_stats,
            "portfolio_return": portfolio_return,
            "live_indicators": live_by_tf.get(tf),  # None for 5m - chart computes native 5m indicators itself
        }

    show_multi_timeframe_backtest(SYMBOLS, variants)


if __name__ == "__main__":
    main()
