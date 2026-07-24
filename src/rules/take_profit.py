"""
止盈: 第一层固定盈亏比部分止盈(锁定确定性利润), 剩余仓位用只上移不下移的
ATR吊灯止损跑趋势。

剩余仓位不用EMA跌破判断出场 - EMA有滞后性，等它被跌破时浮盈可能已经跌
没了甚至转为浮亏，这跟止损模块本该负责的下行保护职责矛盾。吊灯止损的初
始值就是原始止损位(`floor_stop`)，所以剩余仓位任何时刻的下行风险都不会
比原始止损更差 - 不存在"还没赚钱就被放飞导致亏损扩大"的情况，价格走出足
够空间之后吊灯线才会真正开始跟涨锁盈。
"""

from dataclasses import dataclass


@dataclass
class TakeProfitManager:
    partial_ratio: float = 0.5
    rr_trigger: float = 2.0  # 1.5~2R 区间, 取上限
    atr_multiple: float = 3.0

    def partial_trigger_price(self, entry_price: float, direction: str, stop_loss: float) -> float:
        r = abs(entry_price - stop_loss)
        return entry_price + self.rr_trigger * r if direction == "long" else entry_price - self.rr_trigger * r

    def chandelier_stop(
        self, direction: str, extreme_price_since_entry: float, atr_value: float, floor_stop: float
    ) -> float:
        """
        `extreme_price_since_entry`: 多头是入场以来最高价, 空头是最低价。
        `floor_stop`: 原始止损位 - 吊灯线永远不会比它更差(多头只上移/空头只下移)。
        """
        raw = (
            extreme_price_since_entry - self.atr_multiple * atr_value if direction == "long"
            else extreme_price_since_entry + self.atr_multiple * atr_value
        )
        return max(raw, floor_stop) if direction == "long" else min(raw, floor_stop)
