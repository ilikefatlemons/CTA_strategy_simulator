"""
v1.1 Part D: entry/exit rules driven entirely by a higher-timeframe's own
live MACD/ATR, instead of the 5m-native ones in `rules/entry.py`/`exit.py`.

These are new, separate rule classes rather than edits to the v1.0 ones -
the v1.0 rules stay exactly as validated, and a "15m variant" of the
strategy is just this class swapped in for the entry/exit rule, keeping the
same execution grain (5m bar-by-bar decisions/fills), the same reentry rule,
and the same position sizer. Confirmed design: complete replacement, not a
5m+higher-TF combined filter - the 15m variant's signals depend only on the
15m (live-forming) MACD/ATR; the native 5m indicators don't participate.

Each higher-timeframe indicator series is precomputed once per symbol
before the backtest runs (`live_higher_tf_indicators` is truncation-invariant,
so precomputing ahead of time and indexing into it bar-by-bar is exactly
equivalent to recomputing it fresh on every call - see `src/run_phaseC.py`).
Since the v1.0 architecture's "one shared rule instance across every
symbol" constraint still applies here, this precomputed data is keyed by
symbol and looked up via `history["symbol"]` on each call, rather than
baking in a single symbol's series per instance.
"""

import numpy as np
import pandas as pd

from src.rules.base import EntryRule, ExitRule, Position, Side, Signal


class HigherTFMACDGoldenCross(EntryRule):
    def __init__(self, live_indicators_by_symbol: dict[str, pd.DataFrame]):
        self._macd = {s: df["macd"].to_numpy() for s, df in live_indicators_by_symbol.items()}
        self._signal = {s: df["signal"].to_numpy() for s, df in live_indicators_by_symbol.items()}

    def on_bar(self, history: pd.DataFrame) -> Signal:
        i = len(history) - 1
        if i < 1:
            return Signal.NONE
        symbol = history["symbol"].iloc[-1]
        macd, signal = self._macd[symbol], self._signal[symbol]
        prev_diff = macd[i - 1] - signal[i - 1]
        curr_diff = macd[i] - signal[i]
        if np.isnan(prev_diff) or np.isnan(curr_diff):
            return Signal.NONE
        if prev_diff <= 0 and curr_diff > 0:
            return Signal.LONG
        if prev_diff >= 0 and curr_diff < 0:
            return Signal.SHORT
        return Signal.NONE


class HigherTFATRTakeProfitStopLoss(ExitRule):
    def __init__(
        self, live_indicators_by_symbol: dict[str, pd.DataFrame], tp_mult: float = 2.0, sl_mult: float = 1.0
    ):
        self._atr = {s: df["atr"].to_numpy() for s, df in live_indicators_by_symbol.items()}
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult

    def exit_reason(self, history: pd.DataFrame, position: Position) -> str | None:
        symbol = history["symbol"].iloc[-1]
        entry_atr = self._atr[symbol][position.entry_bar_idx]
        if np.isnan(entry_atr):
            return None

        current_price = history["close"].iloc[-1]
        if position.side is Side.LONG:
            move = current_price - position.entry_price
        else:
            move = position.entry_price - current_price

        if move >= self.tp_mult * entry_atr:
            return "TP"
        if move <= -self.sl_mult * entry_atr:
            return "SL"
        return None
