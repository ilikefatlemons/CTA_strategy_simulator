"""Phase 2: MACD golden-cross entry rule."""

import pandas as pd

from src.rules.base import EntryRule, Signal


class MACDGoldenCross(EntryRule):
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def _macd_lines(self, closes: pd.Series) -> tuple[pd.Series, pd.Series]:
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        return macd_line, signal_line

    def on_bar(self, history: pd.DataFrame) -> Signal:
        min_bars = self.slow + self.signal_period
        if len(history) < min_bars + 1:
            return Signal.NONE

        macd_line, signal_line = self._macd_lines(history["close"])
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]

        if prev_diff <= 0 and curr_diff > 0:
            return Signal.LONG
        if prev_diff >= 0 and curr_diff < 0:
            return Signal.SHORT
        return Signal.NONE
