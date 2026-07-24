"""
止损: ATR止损与支撑/阻力平台止损二选一 - 取离入场价更近的那个(亏得更少的
那个)，再向不利方向小幅偏移(offset_pct)，防止被程序化扫损。

平台位取自30m K线的swing high/low(局部极值): 某根K线的high比左右各k根都
高即为swing high，low比左右各k根都低即为swing low。用30m而不是2h，是因为
入场本身就是在5m/15m小周期触发的，止损平台应该贴近实际入场时机的结构，
2h的swing点粒度太粗、离入场价太远。
"""

import pandas as pd


def _swing_points(klines: pd.DataFrame, k: int) -> tuple[list[float], list[float]]:
    highs, lows = klines["high"].to_numpy(), klines["low"].to_numpy()
    swing_highs, swing_lows = [], []
    for i in range(k, len(klines) - k):
        window_high = highs[i - k : i + k + 1]
        window_low = lows[i - k : i + k + 1]
        if highs[i] == window_high.max():
            swing_highs.append(highs[i])
        if lows[i] == window_low.min():
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def _nearest_platform_level(klines: pd.DataFrame, entry_price: float, direction: str, k: int) -> float | None:
    swing_highs, swing_lows = _swing_points(klines, k)
    if direction == "long":
        candidates = [lvl for lvl in swing_lows if lvl < entry_price]
        return max(candidates) if candidates else None  # 离入场价最近的低点
    candidates = [lvl for lvl in swing_highs if lvl > entry_price]
    return min(candidates) if candidates else None  # 离入场价最近的高点


class StopLossCalculator:
    def __init__(self, atr_mult: float = 1.5, offset_pct: float = 0.003, swing_k: int = 2):
        self.atr_mult = atr_mult
        self.offset_pct = offset_pct
        self.swing_k = swing_k

    def calc(self, entry_price: float, direction: str, atr_value: float, klines_30m: pd.DataFrame) -> float:
        atr_stop = (
            entry_price - self.atr_mult * atr_value if direction == "long"
            else entry_price + self.atr_mult * atr_value
        )
        platform_stop = _nearest_platform_level(klines_30m, entry_price, direction, self.swing_k)

        candidates = [atr_stop] + ([platform_stop] if platform_stop is not None else [])
        chosen = min(candidates, key=lambda lvl: abs(entry_price - lvl))

        offset = entry_price * self.offset_pct
        return chosen - offset if direction == "long" else chosen + offset
