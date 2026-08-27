# -*- coding: utf-8 -*-
"""
lineA-03 的原语层。

策略规格出自 `docs/02-lineA-多周期回调/A3-单个自洽策略实施+改进/20260824-骨架+前端phase/goal.md`:

    大周期过滤   2h   收盘>MA21>MA55 -> 多; MA55>MA21>收盘 -> 空; 否则保持上一状态
    回调判断     15m  同一套三线排列, 但必须与 2h **完全反向**
    入场 锁1     15m  收盘[i-1] > MA21[i-1]              (空头镜像)
    入场 锁2     15m  水下金叉运行中 (做多) / 水上死叉运行中 (做空)
    SL           30m  入场价 ∓ 1.5 x ATR(14), 入场时冻结
    TP           30m  3 x ATR(14) 吊灯
    冷静期       2h   连续三次止损 -> 冻结到 2h 出现连续三根同方向的三线排列

本文件只放**无状态的纯函数**: 数组进, 数组出。级联的状态机在引擎里
(`src/engine/lineA_03_backtest.py`), 因为它要和成交、出场、冷静期交织。

------------------------------------------------------------------ 无未来函数 --

这里每个函数都**严格因果**: 下标 b 的输出只依赖 `close[:b+1]` / `dif[:b+1]`。
`rolling` 与 `ewm` 都是因果的, 逐根状态机只回看自己上一根。所以整段预算一次再按
`closed_pos` 查表, 与逐前缀重算逐位相同 —— 与 `src/indicators.py:1-11` 和
`src/strategy/pullback.py:52-75` 同一条论证。

**唯一需要显式守卫的是 MACD**: `ewm(adjust=False)` 无限记忆, 第一根之后**永远不返回
NaN**, 前几十根是垃圾但看不出来 (不像 `rolling(n).mean()` 会老实给 NaN)。挡住它的
只有 `MACD最少根数` 这一个数字, 见 `锁2_水位金叉运行`。这条守卫不能省。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 三值状态。与 `src/strategy/pullback.py:88` 同一套编码, 不另起一套。
未定, 多, 空 = np.int8(0), np.int8(1), np.int8(-1)

_状态名 = {0: "未定", 1: "多", -1: "空"}


def 状态名(s) -> str:
    return _状态名[int(s)]


# ---------------------------------------------------------------- 参数 ----
@dataclass(frozen=True)
class 策略参数:
    """
    goal.md 里出现过的每一个数字。字段名用中文, 与文档一字不差 —— 沿用
    `obsidian/00-总纲/参数总表.md` 的落地口径, 独立于 `src/config.py::BacktestConfig`。
    """

    大周期: str = "2h"
    回调周期: str = "15m"
    风险周期: str = "30m"          # SL / TP 的 ATR 从这里取

    # **驱动时钟**。信号层始终读各自周期自己的已收盘 bar, 这里只决定「多久看一眼
    # 价格」: 成交、出场触发、极值累积、mark-to-market 都在这个周期上。
    # 换更细的数据 (将来的逐 tick) 只需改这一个字段。
    驱动周期: str = "1m"

    快线: int = 21
    慢线: int = 55

    MACD快: int = 12
    MACD慢: int = 26
    MACD信号: int = 9
    # `ewm(adjust=False)` 第一根之后永不返回 NaN, 所以必须显式挡掉前面这些根。
    # = 慢线 26 + 信号线 9, 与 `src/rules/entry.py:8` 的 min_bars 同口径。
    MACD最少根数: int = 35

    ATR周期: int = 14
    止损ATR倍数: float = 1.5
    吊灯ATR倍数: float = 3.0

    冷静期_止损次数: int = 3
    冷静期_解冻根数: int = 3       # 2h 上连续几根同方向的干净排列才解冻

    def __post_init__(self) -> None:
        if not self.快线 < self.慢线:
            raise ValueError(f"快线({self.快线}) 必须小于 慢线({self.慢线})")
        if not self.MACD快 < self.MACD慢:
            raise ValueError(f"MACD快({self.MACD快}) 必须小于 MACD慢({self.MACD慢})")


@dataclass(frozen=True)
class 策略开关:
    """
    **只有一个真开关。**

    执行纪律那几项 (跳空补价 / 盘中触发 / 极值折入滞后一根) 在 v3 里是
    `PullbackConfig` 的具名开关, 这里刻意写死成 v3 的默认值 —— 初稿阶段它们不是
    「可选行为」而是**不变量**, 尤其是极值折入滞后一根: 关掉它就是引入未来函数。
    将来真要做 A/B 时, 在这里加字段、在引擎里对应的那一行读它。
    """

    # 开 = 同方向连续三次纯止损才计数 (v3 旧实现的口径)
    # 关 = 不分方向, 字面读 goal.md 的「连续三次止损」
    冷静期_按方向计数: bool = True

    # 一根驱动 bar 上只做一个动作: 本根开的仓, 最早下一根才可能平。
    #
    # 关掉就允许同根既开又平 —— 那需要知道 bar 内部「先到开盘价还是先到止损位」的
    # 次序, 而这一版的立场恰恰是**不知道**。15m 驱动时代实测 100 笔里有 1 笔是同根
    # 开平的 (2026-06-18 02:30), 那一笔本质上是假的。
    一根bar只做一个动作: bool = True

    # 往返成本(基点), 每批收一次。**默认 0 意味着所有数字都是毛收益。**
    成本_每笔基点: float = 0.0


# ------------------------------------------------------------ 三线排列 ----
def _滚动均值(close: np.ndarray, window: int) -> np.ndarray:
    """
    与 `src/strategy/pullback.py:205` 的 `_rolling_mean` 同一条路径 —— 刻意走 pandas
    而不是自己写 cumsum 差分: cumsum 的浮点误差累积方式与 pandas 的补偿求和不同,
    在 `收盘 == MA21` 这种极近平局处可能翻出不同的状态。
    """
    return np.asarray(pd.Series(close, dtype="float64").rolling(window).mean(),
                      dtype="float64")


def 三线状态(close, 快线: int = 21, 慢线: int = 55) -> np.ndarray:
    """
    逐根的「价格 + 双均线」三元排列, **不带粘性**。

        收盘 > MA快 > MA慢   ->  多
        MA慢 > MA快 > 收盘   ->  空
        其余 (含任意两者相等) ->  未定

    三元严格不等, **不许有等号** —— 平局一律落到「未定」, 交给 `粘住` 处理。

    返回 int8 数组, 长度与输入相同。前 `慢线-1` 根恒为「未定」(MA 还是 NaN)。
    """
    c = np.asarray(close, dtype="float64")
    n = c.size
    out = np.full(n, 未定, dtype=np.int8)
    if n == 0:
        return out
    快 = _滚动均值(c, 快线)
    慢 = _滚动均值(c, 慢线)
    可 = ~(np.isnan(快) | np.isnan(慢))
    out[可 & (c > 快) & (快 > 慢)] = 多
    out[可 & (慢 > 快) & (快 > c)] = 空
    return out


def 粘住(生: np.ndarray) -> np.ndarray:
    """
    前向填充「未定」—— goal.md 的「没有中间态，直到完全反向排列之前都不改变方向」。

    结果里的「未定」只可能出现在序列**开头**一段连续区间里 (一次干净排列都还没出现
    过的暖机期); 一旦离开就再也回不去。
    """
    out = np.asarray(生, dtype=np.int8).copy()
    上一个 = 未定
    for b in range(out.size):
        if out[b] == 未定:
            out[b] = 上一个
        else:
            上一个 = out[b]
    return out


def 大周期状态(close, 快线: int = 21, 慢线: int = 55) -> np.ndarray:
    """`三线状态` + `粘住`。这就是 goal.md 「大周期过滤」那一节。"""
    return 粘住(三线状态(close, 快线, 慢线))


def 完全反向(回调侧状态: np.ndarray, 大周期方向: int) -> np.ndarray:
    """
    回调判定: 小周期的三线排列与大周期**完全反向**。

    2h 多 -> 15m 必须是 `MA55 > MA21 > 收盘` (即 15m 状态 == 空)。
    「未定」不算反向 —— 排列不干净就不是回调。

    注意这里吃的是**未粘性**的 15m 状态: 回调要求的是「此刻确实完全反向」这个事实,
    不是「上一次干净排列是反向的」。粘性只属于大周期。
    """
    if 大周期方向 == 多:
        return 回调侧状态 == 空
    if 大周期方向 == 空:
        return 回调侧状态 == 多
    return np.zeros(len(回调侧状态), dtype=bool)


# --------------------------------------------------------------- MACD ----
def macd线(close, 快: int = 12, 慢: int = 26, 信号: int = 9):
    """
    返回 `(DIF, DEA)` 两个 float64 数组。

    **刻意不调 `src/indicators.py::macd`**: 那个函数读 `df["timestamp"]` 这一列,
    而 v3 数据层的 `TFBars.bars` 是 DatetimeIndex 且没有这一列, 直接喂会 KeyError。
    公式逐字相同 (`ewm(span=..., adjust=False)`), 由
    `tests/test_lineA_03_strategy.py::test_macd_与_indicators_逐位相同` 对拍钉死。
    """
    s = pd.Series(np.asarray(close, dtype="float64"), dtype="float64")
    快线 = s.ewm(span=快, adjust=False).mean()
    慢线 = s.ewm(span=慢, adjust=False).mean()
    dif = 快线 - 慢线
    dea = dif.ewm(span=信号, adjust=False).mean()
    return np.asarray(dif, dtype="float64"), np.asarray(dea, dtype="float64")


def 锁2_水位金叉运行(dif: np.ndarray, dea: np.ndarray, 最少根数: int = 35):
    """
    锁 2 —— 返回 `(多头锁开, 空头锁开)` 两个 bool 数组。

        多头锁开[b]  最近一次**金叉**发生在零轴下方 (DIF 与 DEA 都 < 0),
                     且从那根到 b 一直保持 DIF > DEA
        空头锁开[b]  最近一次**死叉**发生在零轴上方 (DIF 与 DEA 都 > 0),
                     且从那根到 b 一直保持 DIF < DEA

    ---------------------------------------------------------------- 为什么是状态 --

    goal.md 写的是「金叉」, 字面是一个事件; 但它同时写了「有两个锁，都解锁直接触发
    入场」—— 锁是状态, 两个锁才可能**先后**解开。如果按事件解释 (金叉必须恰好落在
    bar[i-1]), 那么锁1 与锁2 必须同根成立, 两个「锁」就退化成一个与门了。

    代价如实记在这里: **不设段龄上限**, 所以一次水下金叉能授权此后整段上涨里的任何
    时刻入场 ——「动量拐点刚发生」这层含义没有了。初稿刻意留着。

    ------------------------------------------------------------------ 暖机守卫 --

    `最少根数` 之前一律不给锁。`ewm(adjust=False)` 第一根之后**永不返回 NaN**, 前
    几十根的 DIF/DEA 是垃圾却看不出来 —— 没有这个守卫, 开头几十根会凭空冒出金叉。

    守卫是**两层**, 少一层就漏:

      1. `b < 最少根数` 的那些根不输出锁 —— 这是显而易见的一层
      2. **`b < 最少根数` 期间根本不累积「段起于水下」这个状态** —— 少了这一层,
         一次发生在垃圾区里的金叉会把标志置上, 于是守卫刚一放开, 锁就凭空开着了。
         实测: 400 根随机游走上, 只加第一层时 b=35 那一根的锁来自 b=19 的金叉,
         而 b=19 在垃圾区里。第一个锁必须由一次**发生在守卫之后**的交叉打开。
    """
    d, e = np.asarray(dif, "float64"), np.asarray(dea, "float64")
    n = d.size
    多锁 = np.zeros(n, dtype=bool)
    空锁 = np.zeros(n, dtype=bool)
    if n == 0:
        return 多锁, 空锁

    上行段起于水下 = False
    下行段起于水上 = False
    for b in range(n):
        if b < 最少根数 or np.isnan(d[b]) or np.isnan(e[b]):
            # 守卫之内: 不输出, 也**不累积状态** (见 docstring 第 2 层)
            上行段起于水下 = 下行段起于水上 = False
            continue
        上行 = d[b] > e[b]
        if b == 0 or np.isnan(d[b - 1]) or np.isnan(e[b - 1]):
            上行段起于水下 = 下行段起于水上 = False
            continue
        前上行 = d[b - 1] > e[b - 1]
        if 上行 and not 前上行:            # 金叉
            上行段起于水下 = d[b] < 0.0 and e[b] < 0.0
        elif (not 上行) and 前上行:        # 死叉
            下行段起于水上 = d[b] > 0.0 and e[b] > 0.0
        多锁[b] = 上行 and 上行段起于水下
        空锁[b] = (not 上行) and 下行段起于水上
    return 多锁, 空锁


def 金叉死叉(dif: np.ndarray, dea: np.ndarray):
    """
    逐根: 这一根是不是金叉 / 死叉。**只给图上标注用**, 引擎的判定走
    `锁2_水位金叉运行`。返回 `(是金叉, 是死叉)` 两个 bool 数组, 首根恒 False。
    """
    d, e = np.asarray(dif, "float64"), np.asarray(dea, "float64")
    n = d.size
    金 = np.zeros(n, dtype=bool)
    死 = np.zeros(n, dtype=bool)
    if n < 2:
        return 金, 死
    上行 = d > e
    金[1:] = 上行[1:] & ~上行[:-1]
    死[1:] = ~上行[1:] & 上行[:-1]
    return 金, 死


# ----------------------------------------------------------- 止损 / 止盈 ----
def 固定止损(入场价: float, 是多头: bool, atr: float, 倍数: float) -> float:
    """
    goal.md 的 SL: 「入场点反向 1.5 x ATR(14)」。**入场时冻结, 此后不再更新。**

    刻意不复用 `src/strategy/pullback.py::stop_level` —— 那个函数还会和 swing 平台位
    取近、再加一个 `sl_offset_pct` 偏移, 而 goal.md 要的就是纯 ATR 倍数。
    """
    return 入场价 - 倍数 * atr if 是多头 else 入场价 + 倍数 * atr


def 吊灯线(是多头: bool, 持仓极值: float, atr: float, 倍数: float, 地板: float) -> float:
    """
    goal.md 的 TP: 「30m 的 3 x ATR(14) 吊灯」。

    `地板` 是入场时冻结的固定止损 —— 吊灯线**永远不比它更差**, 于是:

        有效止损 = max(固定止损, 吊灯线)      (多头; 空头取 min)

    推论: 入场那一刻吊灯 ≈ `入场价 - 3 ATR`, 比固定止损 `入场价 - 1.5 ATR` 更差,
    所以**浮盈涨过 1.5 ATR 之前吊灯完全不起作用**。出场理由据此二分 (见引擎)。

    与 `pullback.py::chandelier_stop` 同形, 但不吃 `PullbackParams`。
    """
    raw = (持仓极值 - 倍数 * atr) if 是多头 else (持仓极值 + 倍数 * atr)
    return max(raw, 地板) if 是多头 else min(raw, 地板)
