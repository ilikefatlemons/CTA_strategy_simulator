"""
回调入场引擎 - 取代原MACD golden-cross信号源(现已注释掉，见 src/rules/entry.py
不再被 run_phaseE 使用)。

首仓(open)和回补(reentry)复用同一个 on_bar 判断路径 - 本类完全不追踪"这是
第几批仓位"，两者的唯一区别只在于调用方(回测引擎)如何给返回的 Signal 打
type 标签，判断逻辑本身对两种情况一视同仁，天然满足"同一函数处理两种场景"
的约束。
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.rules.cooldown import CooldownManager
from src.rules.ma_filter import Bias, MultiTimeframeFilter


@dataclass
class MarketSnapshot:
    tf_2h: pd.DataFrame
    small_tf: dict[str, pd.DataFrame]  # {"5m": df, "15m": df, "30m": df}, 均为已收盘K线
    close: float
    atr: float


@dataclass
class Signal:
    direction: Literal["long", "short"]
    entry_price: float
    type: Literal["open", "reentry"] = "open"


class PullbackEntryEngine:
    def __init__(self, filter: MultiTimeframeFilter, cooldown: CooldownManager):
        self.filter = filter
        self.cooldown = cooldown
        self._bias: Bias = "neutral"
        self._pullback_seen = False
        # Observability only (admin overlay on the chart) - never read by any
        # decision. True on the bar where 回调 confirmation fires.
        self.pullback_confirmed_this_bar = False
        self.last_pullback_bias: Bias = "neutral"

    def on_bar(self, snapshot: MarketSnapshot) -> Signal | None:
        self.pullback_confirmed_this_bar = False
        if self.cooldown.is_active():
            return None

        bias = self.filter.get_trend_bias(snapshot.tf_2h)
        if bias != self._bias:
            # bias变了(包括变成neutral) - 之前累积的回调状态作废，必须重新
            # 等一轮新的回调才能触发，不能用旧方向的回调去触发新方向的入场。
            self._bias = bias
            self._pullback_seen = False
        if bias == "neutral":
            return None

        if not self._pullback_seen:
            if self.filter.pullback_occurred(snapshot.small_tf, bias):
                self._pullback_seen = True
                self.pullback_confirmed_this_bar = True
                self.last_pullback_bias = bias
            return None

        if self.filter.get_entry_trigger(snapshot.small_tf, bias):
            self._pullback_seen = False  # 触发后清零，下次入场必须重新等一轮回调
            return Signal(direction=bias, entry_price=snapshot.close)
        return None
