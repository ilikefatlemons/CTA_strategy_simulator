"""Phase 2: ATR-based take-profit / stop-loss exit rule."""

import pandas as pd

from src.rules.base import ExitRule, Position, Side


def atr(history: pd.DataFrame, period: int) -> float:
    high, low, close = history["high"], history["low"], history["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean().iloc[-1]


class ATRTakeProfitStopLoss(ExitRule):
    def __init__(self, atr_period: int = 14, tp_mult: float = 2.0, sl_mult: float = 1.0):
        self.atr_period = atr_period
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult

    def exit_reason(self, history: pd.DataFrame, position: Position) -> str | None:
        entry_history = history.iloc[: position.entry_bar_idx + 1]
        if len(entry_history) < self.atr_period + 1:
            return None

        entry_atr = atr(entry_history, self.atr_period)
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
