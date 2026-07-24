"""
冷静期风控 - 全局单例, 被入场引擎(PullbackEntryEngine.is_active gate)和
回测引擎共享。

触发条件1: 同方向连续"纯止损"(reason == "SL", 即从未部分止盈过的批次)达到
阈值 - 说明连续判断错了方向。"护盈止损"(reason == "PROTECTIVE_SL", 部分
止盈后剩余仓位被吊灯止损打掉)不计入这个计数 - 那种情况方向判断本身是对
的，只是剩余仓位回吐了部分利润，性质跟"看错方向"不同。

触发条件2: 单根K线实体击穿多条均线 - 均线本身滞后，暴力单边行情下需要先
观望而不是继续按级联信号进场。

解除条件: 必须连续出现 `release_confirm_bars`(默认3)根2h K线都是同一个
干净排列(不要求方向一定要跟触发前一样，允许反转，也允许恢复原方向)，才
解除冷静期。

不能只判断"当前非neutral"就解除 - PullbackEntryEngine本身就只在2h非
neutral时才会产生信号，冷静期触发的那一刻2h几乎必然已经是某个干净排列了
(只是这个方向一直在出错)，"当前非neutral"这个条件在触发的同一时刻就成
立，等于形同虚设。也不能只要求"单根K线重新形成干净排列就解除" - 极端K线
触发时那根K线本身经常直接把排列砸成neutral，如果只需要下一根K线随便反弹
一下形成任意方向的排列就解除，等于没有真正"观望"，一根K线的反弹很可能只
是噪音。要求连续3根都稳定在同一个干净排列，才算真正确认结构已经站稳，
不是单根K线的一次性抖动 - 触发的当根K线不可能同时满足"连续3根"，所以也
天然不会在触发的同一时刻立刻解除。
"""

from dataclasses import dataclass, field

import pandas as pd

from src.rules.ma_filter import ArrayState, ma_array_state


@dataclass
class CooldownManager:
    consecutive_sl_threshold: int = 3
    release_confirm_bars: int = 3
    ma_periods: tuple[int, int, int] = (5, 20, 50)

    active: bool = field(default=False, init=False)
    _consecutive_sl: int = field(default=0, init=False)
    _last_sl_direction: str | None = field(default=None, init=False)
    _stable_state: ArrayState | None = field(default=None, init=False)
    _stable_count: int = field(default=0, init=False)

    def is_active(self) -> bool:
        return self.active

    def _arm(self) -> None:
        self.active = True
        self._stable_state = None
        self._stable_count = 0
        self._consecutive_sl = 0
        self._last_sl_direction = None

    def on_trade_closed(self, direction: str, reason: str) -> None:
        if reason != "SL":
            self._consecutive_sl = 0
            self._last_sl_direction = None
            return
        if direction == self._last_sl_direction:
            self._consecutive_sl += 1
        else:
            self._consecutive_sl = 1
            self._last_sl_direction = direction
        if self._consecutive_sl >= self.consecutive_sl_threshold:
            self._arm()

    def check_structure_break(self, klines: pd.DataFrame) -> bool:
        """最新一根K线的实体是否击穿了至少2条均线 - 判定暴力单边行情。"""
        fast, mid, slow = self.ma_periods
        if len(klines) < slow:
            return False
        closes = klines["close"]
        mas = [closes.rolling(p).mean().iloc[-1] for p in (fast, mid, slow)]
        if any(pd.isna(v) for v in mas):
            return False
        last = klines.iloc[-1]
        body_low, body_high = min(last["open"], last["close"]), max(last["open"], last["close"])
        return sum(1 for ma in mas if body_low <= ma <= body_high) >= 2

    def on_bar(self, klines_2h: pd.DataFrame) -> None:
        if not self.active:
            if self.check_structure_break(klines_2h):
                self._arm()
            return

        state = ma_array_state(klines_2h)
        if state == "neutral" or state != self._stable_state:
            self._stable_state = state
            self._stable_count = 1 if state != "neutral" else 0
        else:
            self._stable_count += 1

        if state != "neutral" and self._stable_count >= self.release_confirm_bars:
            self.active = False
            self._stable_state = None
            self._stable_count = 0
