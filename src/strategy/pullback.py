# -*- coding: utf-8 -*-
"""
v3.0 回调策略的原语层: 多周期 MA 排列状态、结构破坏、swing 平台位、止损/止盈几何。

这是 v2.1 phaseF 回调策略 (`src/rules/ma_filter.py`, `cooldown.py`, `stop_loss.py`,
`take_profit.py`) 的**向量化重写**。`src/rules/*` 一个字不改 —— 它们仍被 v2.1 引擎
和它自己的测试使用, 并且在 `tests/test_pullback_v3_states.py` 里作为**判等基准**。

周期角色映射 (一一对应, 不是重新设计):

    原策略 5m  -> 新 1m   执行/成交粒度; 回调与触发的快腿
    原策略 15m -> 新 5m   回调与触发的另一条快腿
    原策略 30m -> 新 30m  回调的**硬性**条件; 止损平台位; 风险 ATR (A6)
    原策略 2h  -> 新 1d   趋势方向 bias (MA5/20/50 严格排列); 冷静期

⚠ **注意: 这个重映射把策略的时间常数拉长了约 12 倍。** 趋势方向从一天变 3 次变成
一天变 1 次(日线只在交易日起点收一根), 冷静期解除从 6 小时变成 3 个交易日。级联的
**形状**逐条保真、每个谓词都可证等价, 但单位时间的批次数会远低于 v2.1 ——
**结果不能和 docs/v2.1 的账本比。**

===========================================================================
⚠⚠ 指标口径: 连续计算, 不分时段, 不设暖机带, 不做开盘跳空特判 ⚠⚠
===========================================================================
本模块所有 MA / ATR 都在**各自周期的整条 bar 序列上连续滚动**:

  * 不按 session / trading_date 重置。夜盘第一根的 MA20 会用到白盘最后 19 根。
  * ATR 的 TR 里那个 `|high - prev_close|` 会跨过夜盘收盘到次日开盘的跳空, 于是
    跳空当天的 ATR 被真实地抬高了 —— 这既可以说是"把跳空风险计入了波动率",
    也可以说是"日内策略不该为隔夜跳空买单"。
  * 因此**没有**「每个时段前 N 根不参与信号」这类暖机带。

这是刻意的简化 (决策 4), 不是疏漏。真要改的话, 改动点只有两处:
  1. 本文件的 `ma_*_states` / `structure_break_flags` 改成按 `session_id` 分组滚动;
  2. `src/engine/pullback_v3_backtest.py` 里加一个「本时段前 N 根不许入场」的闸门。
`PullbackConfig` 里已经留好了具名开关的位置 —— 加开关时沿用
`src/config.py::BacktestConfig` 的约定: 每个会改变结果的行为都要有一个具名字段,
好让新旧行为能在同一段数据上做 A/B, 而不是被静默烤进代码。

===========================================================================
为什么必须向量化重写, 而不能直接调用 src/rules/*
===========================================================================
原实现是 O(n^2): `ma_array_state` / `ma_fast_mid_state` 每次都在**不断增长的前缀
切片**上跑 `closes.rolling(p).mean().iloc[-1]`, 而 `PullbackEntryEngine.on_bar` 在
每一根空仓 bar 上被调用。v2.1 跑的是 3 个月 x 5m ~ 4,900 根, 勉强能忍; 这里是
一年 x 1m ~ 86,500 根, 每次 pandas 调用还有 ~50us 的固定开销 —— 光趋势方向一项就
是 86,500 x 3 次 rolling ~ 20-40 秒。R-Breaker 现在是 0.15 秒, UI 不能卡。

`CooldownManager.on_bar` 不在此列: A5 的早退发生在 `ma_array_state` 之前, 一天只
真跑一次。`StopLossCalculator._swing_points` 只在有信号那一根被调用, 但它同样是
O(入场数 x n), 窗口一拉长就退化, 而向量化只要三行, 所以一并处理。

---------------------------------------------------------------------------
逐前缀调用 == 全序列算一次再查表, 的证明
---------------------------------------------------------------------------
记 `klines = bars.iloc[:pos+1]`, 于是 `len(klines) == pos + 1`。

(a) **长度守卫与 NaN 守卫是同一个条件。** 原函数 `len(klines) < mid -> neutral`
    即 `pos + 1 < mid`; 而 `close.rolling(mid).mean()` 在下标 `pos` 处为 NaN 当且
    仅当 `pos < mid - 1`, 也就是 `pos + 1 < mid`。两者逐位重合, 所以"全序列
    rolling + 一个 `pos+1 < mid` 的守卫"与逐前缀调用完全等价。两个守卫都保留,
    是为了让代码读起来和原函数一一对应, 不是因为需要两个。

(b) **rolling 严格因果且逐位可复现。** pandas 的 `roll_mean` 是"加入 i、移出
    i-window"的在线算法, 补偿求和的状态沿下标单调推进; 前缀 `0..pos` 的处理序列
    与全序列前 `pos+1` 步逐操作相同, 所以下标 `pos` 处的值**逐位(bit-for-bit)
    相同**, 不是"近似相同"。同一条论证 `src/indicators.py` 顶部已经为 MACD/ATR
    写过一遍。

(c) **swing 点。** 原 `_swing_points` 的循环范围是 `range(k, len(klines) - k)`
    = `range(k, pos + 1 - k)`, 即候选下标只到 `pos - k`; 而 `is_swing_high[b]` 的
    判定窗口是 `b-k .. b+k`, 当 `b <= pos - k` 时该窗口整个落在 `0..pos` 内, 所以
    全序列上算出的标志位对每一个合法候选都与前缀上算出的相同。
    `nearest_platform(..., upto=pos-k)` 因此逐字节复现原结果。

以上三条都由 `tests/test_pullback_v3_states.py` 在**每一个前缀**上钉死。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# MA 排列状态。int8 编码, 与 v2.1 的字符串一一对应:
#   BULL  <-> "bullish" <-> bias "long"
#   BEAR  <-> "bearish" <-> bias "short"
#   NEUTRAL <-> "neutral" <-> bias "neutral"
NEUTRAL, BULL, BEAR = np.int8(0), np.int8(1), np.int8(-1)

_STATE_NAME = {0: "neutral", 1: "bullish", -1: "bearish"}
_BIAS_NAME = {0: "neutral", 1: "long", -1: "short"}


def state_name(s: int) -> str:
    """int8 状态 -> v2.1 的字符串, 给测试和调试用。"""
    return _STATE_NAME[int(s)]


def bias_name(s: int) -> str:
    """int8 状态 -> v2.1 的 bias 字符串。BULL 即 long, BEAR 即 short。"""
    return _BIAS_NAME[int(s)]


# --------------------------------------------------------------- 参数 ----
@dataclass(frozen=True)
class PullbackParams:
    """
    策略参数。**每一个默认值都逐字沿用 v2.1**, 出处标在右边。

    改这里的任何一个数都意味着"策略变了", 不再是移植。
    """

    # 均线: 2h/1d 用三线严格排列定方向, 小周期用快中两线找时机
    ma_fast: int = 5           # ma_filter.py:22,39
    ma_mid: int = 20           # ma_filter.py:22,39
    ma_slow: int = 50          # ma_filter.py:22

    atr_period: int = 14       # pullback_backtest.py:83

    # 止损: ATR 止损与 30m 平台止损取更近的那个, 再向不利方向偏移
    sl_atr_mult: float = 1.5   # stop_loss.py:37
    sl_offset_pct: float = 0.003
    swing_k: int = 2

    # 止盈: 2R 处平掉一半, 剩下一半用 3xATR 吊灯跑趋势
    partial_ratio: float = 0.5     # take_profit.py:17
    rr_trigger: float = 2.0        # take_profit.py:18  ("1.5~2R 区间, 取上限")
    chandelier_mult: float = 3.0   # take_profit.py:19

    # 冷静期。注意 release_bars 现在数的是**日线**根数 (原来是 2h 根数)
    cooldown_sl_threshold: int = 3   # cooldown.py:37
    cooldown_release_bars: int = 3   # cooldown.py:38

    def __post_init__(self) -> None:
        if not (self.ma_fast < self.ma_mid < self.ma_slow):
            raise ValueError("需要 ma_fast < ma_mid < ma_slow")
        # rr_trigger * sl_atr_mult == chandelier_mult 时, 部分止盈触发的那一刻
        # 吊灯线恰好落在保本位。这个恒等式是 A6 (两条腿共用同一个 ATR) 的理由,
        # 见 src/config.py:53-59。破坏它不是错, 但要知道自己在破坏什么。
        # 这里只提醒不报错 —— 它是一个设计巧合, 不是不变量。


@dataclass(frozen=True)
class PullbackConfig:
    """
    行为开关。沿用 `src/config.py::BacktestConfig` 的约定: 每一个会改变结果的行为
    都在这里有一个具名字段, 好让"旧行为 vs 新行为"能在同一段数据上做 A/B。
    默认值 == v2.1 Stage-2.1 定稿后的口径。
    """

    # ---- 入场时点 (2.1 entry refactor) ----
    # True (默认): 1m 的回调/触发只读**已收盘**的 bar (到 i-1 为止), 入场止损的
    #   ATR 用 i-1, 成交在 bar i **自己的 open**。整个决策在 bar i 开盘时已知。
    # False: pre-2.1 —— 1m 读到 close[i], ATR 用 i, 成交在 bar i+1 的 open。
    entry_on_completed_bar: bool = True

    # ---- 出场 ----
    # A2/Item3.5: 止损与吊灯止损按"线"和"本根 open"里更差的那个成交, 跳空穿越
    # 不按市场根本没成交过的线成交。止盈仍是挂在触发价的限价单, 不占便宜。
    gap_fill_exits: bool = True
    # A1/Item3: 用本根的 high/low 判定触发(盘中), 而不是只看收盘。
    range_based_exit_trigger: bool = True
    # Item3: 止损与止盈落在同一根里时, 悲观地先走止损。
    pessimistic_both_hit: bool = True
    # A3: 吊灯线由"到 i-1 为止"的极值算出, 本根 high/low 在测试**之后**才折入。
    trail_extreme_prev_bar: bool = True
    # A6: 风险 ATR (入场止损距离 + 吊灯) 读 30m 的 ATR(14), 不读 1m 的。
    risk_atr_on_30m: bool = True
    # A5: 冷静期是日频规则, 但引擎每根 1m 都调它 —— 门控成"每根新日线只评一次"。
    cooldown_counts_daily_bars: bool = True
    # 冷静期的**第二个**触发器: 单根日线实体吃穿 >=2 条均线 -> 暴力单边行情, 先
    # 观望 (`cooldown.py:10-11`)。关掉之后冷静期只剩"同方向连续三次纯止损"一个
    # 触发器 —— 而在时段锁死下那个触发器本来就很少响, 所以这个开关的影响比看上去大:
    # 实测 RB 一年 13 个冷静期区间里有 9 个是它 armed 的。
    #
    # 解除条件不受影响: 无论被哪个触发器 armed, 都要连续 3 根日线稳定排列才解除。
    cooldown_on_structure_break: bool = True

    # ---- v3.0 新增 ----
    # 时段模式只控制**入场闸门**(决策 2)。True: 不可交易的 bar 上不尝试入场, 但
    # 状态机(bias/回调/冷静期)照常推进, 且被挡住的触发**不消耗** _pullback_seen。
    # 这与 rbreaker_reversal_backtest.py:186-238 的先例同构(被挡的入场不清空
    # armed_dir)。False -> 忽略 tradable, 五种时段模式退化成同一个结果。
    session_gates_entry: bool = True
    # 时段**锁死**: 时段最后一根 1m bar 的 close 强制平仓。与 rbreaker_backtest.py:
    # 203-206 逐字同构 (成交价取 close, 不做跳空补价)。
    #
    # 与 session_gates_entry 正交 —— 那条挡入场, 这条管出场。**两条都开**才是用户
    # 要的"锁死"语义: 未选中的时段完全空仓且无交易。
    #
    # False -> 退回移植初版: 持仓穿越时段边界, 出场原因只有 SL/TP/PROTECTIVE_SL。
    # v2.1 判等测试走的正是这条路 —— 判等基准 (未改动的 run_pullback_backtest)
    # 没有强平, 不关掉必然全红。
    session_forces_flat: bool = True
    # 往返成本(基点), 逐批收一次, 按 size_fraction 分摊到两条腿上。
    # 仓库目前没有任何 tick_size / 合约乘数 / 费率表, 所以这里只能是一个旋钮。
    cost_bps: float = 0.0

    # ---- 数据层 (prepare_base 需要这两个字段, 与 RBreakerConfig 同名同义) ----
    min_ref_bars: int = 30
    max_reference_gap_days: int = 15


# ------------------------------------------------------- MA 排列状态 ----
def _rolling_mean(close: np.ndarray, window: int) -> np.ndarray:
    """
    与 v2.1 `closes.rolling(w).mean()` **同一条代码路径** —— 刻意走 pandas 而不是
    自己写 cumsum 差分, 因为 cumsum 的浮点误差累积方式与 pandas 的补偿求和不同,
    在 `MA5 == MA20` 的极近平局处可能翻出不同的状态。见模块 docstring (b)。
    """
    return np.asarray(pd.Series(close, dtype="float64").rolling(window).mean(),
                      dtype="float64")


def ma_fast_mid_states(close, fast: int = 5, mid: int = 20) -> np.ndarray:
    """
    逐根的两线排列状态, 对应 `src/rules/ma_filter.py::ma_fast_mid_state`。

    用于 1m / 5m / 30m 找时机。原函数的注释解释了为什么小周期不用三线严格排列:
    门槛太高会让整个级联几乎不可能凑齐 (2 年 NVDA 只完整走完 1 次)。

    返回 int8 数组; 下标 b 的值 == 在 `bars.iloc[:b+1]` 上调用原函数的结果。
    """
    c = np.asarray(close, dtype="float64")
    n = c.size
    out = np.full(n, NEUTRAL, dtype=np.int8)
    if n == 0:
        return out
    ma_f = _rolling_mean(c, fast)
    ma_m = _rolling_mean(c, mid)
    # 长度守卫: len(klines) < mid -> neutral, 即 b+1 < mid, 即 b < mid-1。
    ok = np.arange(n) >= (mid - 1)
    ok &= ~(np.isnan(ma_f) | np.isnan(ma_m))
    out[ok & (ma_f > ma_m)] = BULL
    out[ok & (ma_f < ma_m)] = BEAR
    return out


def ma_array_states(close, fast: int = 5, mid: int = 20, slow: int = 50) -> np.ndarray:
    """
    逐根的三线严格排列状态, 对应 `src/rules/ma_filter.py::ma_array_state`。

    只用于**日线**定方向 (原策略是 2h)。门槛高是刻意的 —— 大方向判断不该草率。
    """
    c = np.asarray(close, dtype="float64")
    n = c.size
    out = np.full(n, NEUTRAL, dtype=np.int8)
    if n == 0:
        return out
    ma_f = _rolling_mean(c, fast)
    ma_m = _rolling_mean(c, mid)
    ma_s = _rolling_mean(c, slow)
    ok = np.arange(n) >= (slow - 1)
    ok &= ~(np.isnan(ma_f) | np.isnan(ma_m) | np.isnan(ma_s))
    out[ok & (ma_f > ma_m) & (ma_m > ma_s)] = BULL
    out[ok & (ma_f < ma_m) & (ma_m < ma_s)] = BEAR
    return out


def structure_break_flags(
    bars: pd.DataFrame, periods: tuple[int, int, int] = (5, 20, 50),
) -> np.ndarray:
    """
    逐根: 该 bar 的**实体**是否包住了至少 2 条均线 —— 对应
    `src/rules/cooldown.py::check_structure_break`, 冷静期的第二个触发条件。

    判定的是"暴力单边行情": 均线本身滞后, 一根 K 线直接吃穿多条均线时应该先观望,
    而不是继续按级联信号进场。

    只在**日线**上算 (原策略是 2h)。
    """
    fast, mid, slow = periods
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    c = bars["close"].to_numpy("float64")
    o = bars["open"].to_numpy("float64")
    mas = np.column_stack([_rolling_mean(c, p) for p in (fast, mid, slow)])
    body_lo = np.minimum(o, c)[:, None]
    body_hi = np.maximum(o, c)[:, None]
    inside = (mas >= body_lo) & (mas <= body_hi)      # NaN 参与比较恒为 False
    ok = np.arange(n) >= (slow - 1)
    ok &= ~np.isnan(mas).any(axis=1)
    return ok & (inside.sum(axis=1) >= 2)


# --------------------------------------------------------- swing 平台 ----
def swing_flags(bars: pd.DataFrame, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    逐根: 是否 swing high / swing low —— 对应 `src/rules/stop_loss.py::_swing_points`。

    定义: `high[i] == max(high[i-k : i+k+1])` 即 swing high, low 对称。**用 `==`
    而不是 `>=` 于两侧, 与原实现逐字一致** —— 平台期里连续几根等高的 bar 会被全部
    判为 swing high, 这是原行为。

    首尾各 k 根永远为 False (原实现的 `range(k, len(klines)-k)`)。
    """
    n = len(bars)
    is_hi = np.zeros(n, dtype=bool)
    is_lo = np.zeros(n, dtype=bool)
    if n <= 2 * k:
        return is_hi, is_lo
    hi = bars["high"].to_numpy("float64")
    lo = bars["low"].to_numpy("float64")
    w = 2 * k + 1
    # sliding_window_view 给出形状 (n-w+1, w) 的视图, 第 j 行对应中心下标 j+k。
    win_hi = np.lib.stride_tricks.sliding_window_view(hi, w).max(axis=1)
    win_lo = np.lib.stride_tricks.sliding_window_view(lo, w).min(axis=1)
    is_hi[k:n - k] = hi[k:n - k] == win_hi
    is_lo[k:n - k] = lo[k:n - k] == win_lo
    return is_hi, is_lo


@dataclass(frozen=True)
class SwingIndex:
    """`swing_flags` 的查询友好形式: 下标数组 + 对应价位, 均按下标升序。"""

    high_idx: np.ndarray
    high_val: np.ndarray
    low_idx: np.ndarray
    low_val: np.ndarray

    @classmethod
    def build(cls, bars: pd.DataFrame, k: int = 2) -> "SwingIndex":
        is_hi, is_lo = swing_flags(bars, k)
        hi_i = np.flatnonzero(is_hi)
        lo_i = np.flatnonzero(is_lo)
        return cls(
            high_idx=hi_i, high_val=bars["high"].to_numpy("float64")[hi_i],
            low_idx=lo_i, low_val=bars["low"].to_numpy("float64")[lo_i],
        )


def nearest_platform(
    swings: SwingIndex, upto: int, entry_price: float, is_long: bool,
) -> float | None:
    """
    离入场价最近的平台位 —— 对应 `src/rules/stop_loss.py::_nearest_platform_level`。

    `upto`: 允许参与的最大 swing 下标 (闭区间)。调用方应传 `closed_pos - k`, 复现
    原实现 `range(k, len(klines)-k)` 的上界。见模块 docstring (c)。

    多头取**低于入场价的最高低点**, 空头取**高于入场价的最低高点**。都没有则 None。
    """
    if upto < 0:
        return None
    if is_long:
        j = np.searchsorted(swings.low_idx, upto, side="right")
        vals = swings.low_val[:j]
        vals = vals[vals < entry_price]
        return float(vals.max()) if vals.size else None
    j = np.searchsorted(swings.high_idx, upto, side="right")
    vals = swings.high_val[:j]
    vals = vals[vals > entry_price]
    return float(vals.min()) if vals.size else None


# ------------------------------------------------------ 止损 / 止盈 ----
def stop_level(
    entry_price: float, is_long: bool, atr_value: float,
    platform: float | None, p: PullbackParams,
) -> float:
    """
    入场止损 —— 对应 `src/rules/stop_loss.py::StopLossCalculator.calc`。

    ATR 止损与平台止损**取离入场价更近的那个**(亏得更少的那个), 再向不利方向偏移
    `sl_offset_pct`, 防止被程序化扫损。

    平局时 ATR 止损胜出 —— 原实现是 `min(candidates, key=...)` 且 atr_stop 排在
    列表首位, Python 的 min 在平局时返回第一个。逐字保留。
    """
    atr_stop = (entry_price - p.sl_atr_mult * atr_value) if is_long \
        else (entry_price + p.sl_atr_mult * atr_value)
    chosen = atr_stop
    if platform is not None and abs(entry_price - platform) < abs(entry_price - atr_stop):
        chosen = platform
    offset = entry_price * p.sl_offset_pct
    return chosen - offset if is_long else chosen + offset


def partial_trigger_price(
    entry_price: float, is_long: bool, stop_loss: float, p: PullbackParams,
) -> float:
    """部分止盈触发价 —— `take_profit.py::partial_trigger_price`。入场价 ± rr_trigger x R。"""
    r = abs(entry_price - stop_loss)
    return entry_price + p.rr_trigger * r if is_long else entry_price - p.rr_trigger * r


def chandelier_stop(
    is_long: bool, extreme_since_entry: float, atr_value: float,
    floor_stop: float, p: PullbackParams,
) -> float:
    """
    吊灯止损 —— `take_profit.py::chandelier_stop`。

    `floor_stop` 是原始止损位: 吊灯线**永远不会比它更差**, 所以剩余仓位任何时刻的
    下行风险都不超过原始止损。价格走出足够空间之后它才真正开始跟涨锁盈。
    """
    raw = (extreme_since_entry - p.chandelier_mult * atr_value) if is_long \
        else (extreme_since_entry + p.chandelier_mult * atr_value)
    return max(raw, floor_stop) if is_long else min(raw, floor_stop)


def gap_filled(level: float, is_long: bool, bar_open: float, enabled: bool) -> float:
    """
    跳空补价 —— `pullback_backtest.py:191-197`。

    止损按"线"和"本根 open"里更差的那个成交。10:15->10:31 / 11:30->13:31 以及每个
    时段边界都是零根 bar, 那里的跳空是真实的。**只用于止损/吊灯, 不用于止盈** ——
    "止损吃亏, 止盈不占便宜"。
    """
    if not enabled:
        return level
    return min(level, bar_open) if is_long else max(level, bar_open)
