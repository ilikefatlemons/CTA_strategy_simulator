# -*- coding: utf-8 -*-
"""
v3.0 回调策略的单品种逐根回测 —— `src/engine/pullback_backtest.py` 的中国期货移植。

**规则一个字没改。** 循环结构、判定顺序、每一个开关的语义都逐行对应 v2.1 那份,
右边的 `pb.py:NNN` 注释指向原文件的行号, 两边可以直接对读。变的只有三样:

  1. **周期**: 5m/15m/30m/2h -> 1m/5m/30m/1d (角色一一对应, 见 strategy/pullback.py)
  2. **实现**: 逐前缀 pandas 调用 -> 预计算数组 + numpy 标量循环 (纯性能, 语义
     等价由 tests/test_pullback_v3_states.py 在每个前缀上钉死)
  3. **时段锁死**: 新增两个正交开关 —— `session_gates_entry` 挡入场,
     `session_forces_flat` 在时段末根强制平仓 (见下)

---------------------------------------------------------------------------
无未来函数, 逐行可审计
---------------------------------------------------------------------------
    步骤                   读取的数据                                    触及的最新 bar
 1  冷静期(日频, A5 门控)  DSTATE[cd], DBREAK[cd]  cd = closed_pos_1d[i]  i-1
 2  趋势方向 bias          DSTATE[cd]                                     i-1
 3  回调 / 触发级联        S1[i-1], S5[c5[i]], S30[c30[i]]                i-1
 4  入场风险 ATR (A6)      ATR30[c30[i]]                                  i-1
 5  止损平台位             swing 标志, 下标 <= c30[i] - swing_k           i-1
 6  入场闸门               in_window[i] & tradable[i]                     i  (窗口形状, 与价格无关)
                           & !is_session_last[fill_idx]                   同上
 7  入场成交               open[i]                                        i  (开盘价)
 8  止损/止盈触发判定      low[i] / high[i]                               i  (事件本身)
 9  跳空补价               open[i]                                        i  (成交那一刻已知)
10  吊灯止损位             extreme(到 i-1 为止), ATR30[c30[i]]             i-1
10b 时段末强平             is_session_last[i], close[i]                   i  (形状静态; close 是成交价)
11  极值折入               high[i] / low[i] —— **在测试之后**             i, 但只影响 i+1
12  逐根 mark-to-market    close[i]                                       i
13  冷静期区间记录         cooldown.active                                i  (bar 末尾)

唯一结构上危险的是第 11 步(极值折入), 已经放在出场测试之后并带 `batch is not None`
守卫 —— 与 `pullback_backtest.py:296-300` 和 `rbreaker_backtest.py:230-231` 同形。

顺序上必须保住的三条 (全部来自 v2.1):
  * `cooldown.on_bar` 排在入场尝试**之前** (pb.py:205-210) —— 本根新收的日线所
    armed 的冷静期会挡住本根入场。
  * `cooldown.on_trade_closed` 在出场块里 (pb.py:253/289), 即入场尝试**之后** ——
    本根的止损不会挡住本根的入场。
  * 默认 `entry_on_completed_bar=True` 下入场尝试排在出场块**之前**, 成交价是
    **本根自己的 open** (pb.py:209-210)。

第 6 步和第 10b 步都不是未来函数: 时段划分和收盘时间都是窗口形状的属性, 循环开始前
就已确定, 不依赖任何价格 —— 与 `rbreaker_backtest.py:27-33` 的论证逐字同构。
(收盘时间是交易所提前公告的, 知道"这一根是本时段最后一根"不需要看见未来的价格。)

---------------------------------------------------------------------------
时段锁死 = 两个正交的开关
---------------------------------------------------------------------------
`session_gates_entry` 挡**入场**, `session_forces_flat` 管**出场**。两条都开
(默认) 才是"锁死": 未选中的时段完全空仓且无交易。

  入场   `tradable[i]` 为假不开仓; 外加**成交那一根**是时段末根时也不开仓
         —— 否则那一批既成交又立刻被强平, 会产生一条 0 根的 END_SESSION 腿。
         与 `rbreaker_backtest.py:242` 的 `not last_of_sess[i]` 同一个理由。
  出场   `is_session_last[i]` 那一根按 **close** 强制平仓, 不做跳空补价。
         与 `rbreaker_backtest.py:203-206` 逐字同构。

被挡住时**不消耗** `pullback_seen`: 级联好不容易凑齐的一次回调不该被一根不可交易
(或时段末) 的 bar 白白吃掉。这与 `rbreaker_reversal_backtest.py:186-238` 的先例
同构 —— 那里被挡的入场同样不清空 `armed_dir`。

**强平不是 `elif`。** 同一根上 2R 止盈刚平掉一半时, 剩下那一半也要在这根被强平,
于是这一批产生两条腿 (TP 50% + END_SESSION 50%), `Σ size_fraction` 仍然是 1.0。

**冷静期有两个触发器**, 各自一个开关, 解除条件共用 (连续 3 根日线稳定排列):

    触发器 1  同方向连续 3 次纯止损              —— 计数规则见下
    触发器 2  单根日线实体吃穿 >=2 条均线        —— `cooldown_on_structure_break`

时段锁死之后触发器 1 很少响 (七成批次在收盘钟被平掉, 走不到止损), 所以触发器 2
实际扛了大头: 实测 RB 一年 13 个冷静期区间里 **9 个**来自它。关掉它之前先知道
这一点 —— 那不是"少一个补充规则", 是砍掉了主要来源。

**「同方向连续看错」计数器只认两件事**:

    纯止损 SL        -> 同方向 +1; 够阈值就 arm, 并**在 arm 的同时清零**
    部分止盈 TP      -> 清零 (这一轮方向是对的)
    护盈止损 PSL     -> 什么也不做
    时段末强平 END   -> 什么也不做

后两条都是"什么也不做"而不是"清零", 理由不同:
  * PSL: 方向判断本身是对的, 只是剩余仓位回吐了利润 (`cooldown.py:5-8`)。它这里
    其实**恒等于空操作** —— 走到 PSL 必然已经 TP 过, 而 TP 刚清过零, 两条腿之间
    同一批一直持着仓, 没有别的批次能改计数器。写成"什么也不做"是为了别让人以为
    护盈止损对冷静期有话语权。
  * END: 日历驱动, 不含方向判断信息。

**已知代价, 不粉饰**: 锁死之后 END 占了七成批次却对计数器透明, 于是三笔真止损可以
横跨一周、中间夹着五个收盘批却仍被算成"连续三次" —— 实测 RB/split 下四次连续止损
冷静期全部是这种。试过"亏损的整批 END 算一次看错": 59 品种 x 5 模式下冷静期 +25%、
批次 -14%、逐组合收益 129 好 / 152 差 (噪声), 行为改变远大于它解决的问题, 已回退。
真要收紧, 该给"连续"加时间上限, 而不是把收盘平仓塞进这个计数器。

`is_session_last[i]` 和 `tradable[i]` 都不是未来函数: 时段划分是窗口形状的属性,
循环开始前就已确定, 不依赖任何价格 —— 与 `rbreaker_backtest.py:27-33` 同构。

裸 `is_session_last[i]` 就够, 不需要 `and tradable[i]`: 该列由 `session_id` 单独
算出、完全不看 `tradable` (`v3_sessions.py:340`)。持仓只可能开在可交易时段内, 而
那个时段自己的末根也带 `is_session_last` —— 走不到下一个不可交易时段。

与 v2.1 的 `warmup_discard_bars` 的一处**有意**差异: 那个开关是在调用状态机
**之前**返回的(pb.py:152), 于是暖机段结束时状态机是冷的。这里暖机段用同一个闸门,
但状态机照常推进 —— 暖机 padding 存在的全部意义就是让显示窗口第一根 bar 上策略
已经暖好。v2.1 的 `warmup_discard_bars` 默认是 0, 那条路径在真实策略里从不执行,
所以这里没有可"保真"的既有行为。

---------------------------------------------------------------------------
锁死改变了什么 (读结果之前必须知道)
---------------------------------------------------------------------------
1. **五种时段模式第一次两两不同。** 锁死之前 `split` / `night_day` / `day_night`
   逐字节相同 —— 它们的 `tradable` 全为真, 区别只在时段**分组**, 而当时没有任何
   逻辑读分组。强平读 `is_session_last`, 而它直接由分组决定, 于是三者立刻分叉:
   `split` 每天强平两次(夜盘收盘 + 15:00), `night_day` 只在 15:00,
   `day_night` 在夜盘收盘。
2. **这一层把策略推得离 v2.1 phaseF 更远。** 周期重映射 (5m/15m/30m/2h ->
   1m/5m/30m/1d) 已经把时间常数拉长约 12 倍; 现在再叠一层强制时段内平仓。
   `tests/test_pullback_v3_vs_v21.py` 判等走的是 `session_forces_flat=False`
   那条路径 —— 它证明的是**移植没把逻辑改坏**, 不是"线上跑的这套等于 v2.1"。
3. **锁死下收尾必然空仓** (帧末根的 `is_session_last` 恒为 True,
   `v3_sessions.py:340`), 所以有 `assert batch is None`。关掉开关则相反: 窗口
   最后一根仍持仓的那一批不产生 `PBTrade`, 浮盈只体现在权益曲线里。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.v3_sessions import Prepared
from src.data.v3_timeframes import TFBars, build_timeframes
from src.indicators import atr as atr_series
from src.strategy.pullback import (
    BEAR, BULL, NEUTRAL, PullbackConfig, PullbackParams, SwingIndex, chandelier_stop,
    gap_filled, ma_array_states, ma_fast_mid_states, nearest_platform,
    partial_trigger_price, stop_level, structure_break_flags,
)

SL = "SL"                        # 整批止损
TP = "TP"                        # 部分止盈 (partial_ratio)
PROTECTIVE_SL = "PROTECTIVE_SL"  # 护盈止损 (吊灯打掉剩余仓位)
END_SESSION = "END_SESSION"      # 时段末强平 (cfg.session_forces_flat)
EXIT_REASONS = (SL, TP, PROTECTIVE_SL, END_SESSION)

_NO_STATE = np.int8(2)           # 不可能的排列状态, 当 CooldownManager._stable_state 的 None


@dataclass
class PBTrade:
    """
    一条**腿**(不是一批)。一批仓位最多两条腿: 2R 处的部分止盈 + 剩余仓位的吊灯,
    或者整批一次性止损。同一批的腿共享 `entry_bar_idx` —— 统计层按它分组。
    """

    direction: str            # "long" | "short"
    entry_bar_idx: int
    entry_time: pd.Timestamp
    entry_price: float        # = open[entry_bar_idx], 实际成交价
    exit_bar_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    reason: str               # SL | TP | PROTECTIVE_SL
    size_fraction: float      # 这条腿占整批的比例; 同一批各腿之和 == 1.0
    signal_type: str          # "open" | "reentry"
    gapped: bool              # True = 按 bar 的 open 成交, 不是按止损线
    stop_loss: float          # 该批的**原始**止损位 (审计/展示用)
    trading_date: pd.Timestamp
    session_id: int
    cost_bps: float

    @property
    def pnl_pct(self) -> float:
        move = (self.exit_price - self.entry_price) if self.direction == "long" \
            else (self.entry_price - self.exit_price)
        return move / self.entry_price

    @property
    def pnl_net(self) -> float:
        """
        **单位仓位**的净收益。往返成本按整批收一次:
        `Σ_腿 size_fraction * pnl_net` 里的成本项恰好是 `1.0 * cost_bps/1e4`,
        因为同一批的 size_fraction 之和是 1。按腿各扣一次会重复收开仓费。

        统计口径一律用这个。
        """
        return self.pnl_pct - self.cost_bps / 1e4

    @property
    def bars_held(self) -> int:
        return self.exit_bar_idx - self.entry_bar_idx


@dataclass
class PullbackResult:
    symbol: str
    trades: list[PBTrade]
    # 权益曲线与 trading_date 只含**显示窗口**(暖机段被切掉), 起点归 1.0。
    equity_curve: pd.Series
    trading_date: pd.Series
    # 逐根, **全帧**(含暖机段): 图层要按 bin 分箱, 用全帧下标最省心。
    position: np.ndarray          # int8: -1 / 0 / +1
    stop_line: np.ndarray         # float64: 该根被测试的止损位; 空仓为 NaN
    # 仅观测, 没有任何决策读它们。图上「额外注释」用。
    pullback_points: list         # {"bar_idx", "direction", "price"}
    cooldown_spans: list          # [(start_bar_idx, end_bar_idx)] 闭区间
    warmup_bars: int
    warmup_days: int
    tradable_days: int
    params: PullbackParams
    config: PullbackConfig
    mode: str
    dropped_runs: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def in_position(self) -> np.ndarray:
        """止损线该画的那些根。"""
        return ~np.isnan(self.stop_line)


@dataclass
class _Batch:
    """`pullback_backtest.py:63-72` 的 `_OpenBatch`。"""

    direction: int            # +1 / -1
    entry_bar_idx: int
    entry_price: float
    stop_loss: float
    signal_type: str
    partial_taken: bool = False
    trailing_stop: float = float("nan")
    extreme_since_entry: float = 0.0


def _tf_states(tf: dict[str, TFBars], p: PullbackParams) -> dict[str, np.ndarray]:
    """
    四个周期的预计算数组。**全部在各自周期的整条序列上连续滚动** —— 不分时段、
    不设暖机带、不做开盘跳空特判 (决策 4, 见 strategy/pullback.py 的醒目段落)。
    """
    return {
        "S1": ma_fast_mid_states(tf["1m"].bars["close"], p.ma_fast, p.ma_mid),
        "S5": ma_fast_mid_states(tf["5m"].bars["close"], p.ma_fast, p.ma_mid),
        "S30": ma_fast_mid_states(tf["30m"].bars["close"], p.ma_fast, p.ma_mid),
        "DSTATE": ma_array_states(tf["1d"].bars["close"], p.ma_fast, p.ma_mid, p.ma_slow),
        "DBREAK": structure_break_flags(tf["1d"].bars, (p.ma_fast, p.ma_mid, p.ma_slow)),
        # 风险 ATR (A6): 只有 30m 这一个进决策。1m 的仅在 risk_atr_on_30m=False 时用到。
        "ATR30": np.asarray(atr_series(tf["30m"].bars, p.atr_period), dtype="float64"),
        "ATR1": np.asarray(atr_series(tf["1m"].bars, p.atr_period), dtype="float64"),
    }


def run_pullback_v3_backtest(
    prepared: Prepared,
    tf: dict[str, TFBars] | None = None,
    params: PullbackParams | None = None,
    config: PullbackConfig | None = None,
) -> PullbackResult:
    """
    逐根跑一遍。

    `tf` 作为参数而不是内部构造 —— 这正是 `tests/test_pullback_v3_vs_v21.py` 能把
    一个 1:3:6:24 的等构结构喂进来、和未改动的 v2.1 引擎逐笔对比的前提。省略时按
    真实的 1m/5m/30m/1d 现建。
    """
    p = params or PullbackParams()
    cfg = config or PullbackConfig()
    df = prepared.df
    n = len(df)

    if n == 0:
        return _empty_result(prepared, p, cfg)
    if tf is None:
        tf = build_timeframes(df)

    st = _tf_states(tf, p)
    S1, S5, S30 = st["S1"], st["S5"], st["S30"]
    DSTATE, DBREAK = st["DSTATE"], st["DBREAK"]
    ATR30, ATR1 = st["ATR30"], st["ATR1"]
    swings = SwingIndex.build(tf["30m"].bars, p.swing_k)

    c5, c30, cd = tf["5m"].closed_pos, tf["30m"].closed_pos, tf["1d"].closed_pos
    # 1m 的 closed_pos 恒等于 i-1 —— v2.1 的 `small_5m = df_5m.iloc[:i]`。
    c1 = tf["1m"].closed_pos

    op = df["open"].to_numpy("float64")
    hi = df["high"].to_numpy("float64")
    lo = df["low"].to_numpy("float64")
    cl = df["close"].to_numpy("float64")
    times = df.index.to_numpy()
    tds = df["trading_date"].to_numpy()
    sid = df["session_id"].to_numpy("int64")
    tradable = df["tradable"].to_numpy(bool)
    # 时段末根。锁死关掉时全部当作 False —— 这样两个开关真正正交, 关掉之后
    # 引擎里连一次比较都不做, 与移植初版逐字节相同。
    last_of_sess = (df["is_session_last"].to_numpy(bool) if cfg.session_forces_flat
                    else np.zeros(n, dtype=bool))
    forces_flat = cfg.session_forces_flat
    structure_break_arms = cfg.cooldown_on_structure_break
    in_window = prepared.in_window
    can_enter = (tradable & in_window) if cfg.session_gates_entry else in_window

    trail_extreme_prev = cfg.trail_extreme_prev_bar
    range_trigger = cfg.range_based_exit_trigger
    pessimistic = cfg.pessimistic_both_hit
    gap_fill = cfg.gap_fill_exits
    completed_bar = cfg.entry_on_completed_bar
    cost = cfg.cost_bps / 1e4

    equity = np.ones(n, dtype="float64")
    pos_arr = np.zeros(n, dtype=np.int8)
    stop_arr = np.full(n, np.nan, dtype="float64")
    trades: list[PBTrade] = []
    pullback_points: list = []
    cooldown_spans: list = []

    capital = 1.0
    batch: _Batch | None = None
    had_prior_batch = False
    cd_start: int | None = None

    # --- 入场状态机 (pullback_entry.PullbackEntryEngine) ---
    bias = NEUTRAL
    pullback_seen = False
    # --- 冷静期 (rules.cooldown.CooldownManager) ---
    cool_active = False
    consecutive_sl = 0
    last_sl_dir = 0
    stable_state = _NO_STATE
    stable_count = 0
    last_daily_pos = -1

    def count_directional_loss(direction: int) -> None:
        """
        「同方向连续看错」计数 +1, 够阈值就 arm 冷静期。

        `cooldown.py:61-72` 的 `on_trade_closed(direction, "SL")` 那一支, 外加
        `_arm()`(`:54-59`)。**arm 的同时把计数器清零** —— 所以冷静期解除之后是从
        0 重新数, 不会一进一出就再次触发。

        两个调用点共用: 纯止损, 以及**整批**亏损收盘 (见下面的 END_SESSION 块)。
        抽出来是因为两处必须逐字一致 —— 各写一遍迟早会漂。
        """
        nonlocal consecutive_sl, last_sl_dir, cool_active, stable_state, stable_count
        if direction == last_sl_dir:
            consecutive_sl += 1
        else:
            consecutive_sl, last_sl_dir = 1, direction
        if consecutive_sl >= p.cooldown_sl_threshold:
            cool_active = True
            stable_state, stable_count = _NO_STATE, 0
            consecutive_sl, last_sl_dir = 0, 0

    def risk_atr(i: int, prev_bar: bool) -> float:
        """pb.py:111-125。A6 下两条腿都读**最后一根已收盘 30m** 的 ATR(14)。"""
        if cfg.risk_atr_on_30m:
            j = c30[i]
            return ATR30[j] if j >= 0 else np.nan
        if prev_bar:
            return ATR1[i - 1] if i > 0 else np.nan
        return ATR1[i]

    def attempt_entry(i: int, D: int, P5: int, P30: int) -> _Batch | None:
        """
        `pullback_backtest.py:138-189` + `pullback_entry.PullbackEntryEngine.on_bar`
        合并成一个函数 —— 两种入场时点口径共用这一份, 免得复制两遍再各自漂移。

        级联 (ma_filter.MultiTimeframeFilter):
          日线三线严格排列定方向 -> 30m **必须**转反向 且 1m/5m 至少一个也转反向
          (回调确认) -> 1m 和 5m **都**转回与日线同向 (触发)。
        """
        nonlocal bias, pullback_seen
        # pb.py:150 —— 三个高周期都必须已经有收盘 K 线
        if D < 0 or P5 < 0 or P30 < 0:
            return None
        # pb.py:48 —— 冷静期闸门排在最前面, 连方向都不更新
        if cool_active:
            return None

        b = DSTATE[D]
        if b != bias:                    # 方向变了(含变成 neutral) -> 累积的回调作废
            bias, pullback_seen = b, False
        if b == NEUTRAL:
            return None

        if completed_bar:
            fill_idx, s1_pos = i, int(c1[i])    # c1[i] == i-1, 即 1m 只读到 i-1
        else:
            fill_idx, s1_pos = i + 1, i         # pre-2.1: 读到 close[i], 成交在 i+1
        s1 = S1[s1_pos] if s1_pos >= 0 else NEUTRAL
        s5, s30 = S5[P5], S30[P30]

        if not pullback_seen:
            opp = BEAR if b == BULL else BULL
            if s30 == opp and (s1 == opp or s5 == opp):
                pullback_seen = True
                pullback_points.append(
                    {"bar_idx": i, "direction": "long" if b == BULL else "short",
                     "price": float(cl[i])})
            # 回调确认的那一根**绝不**同时入场 —— 触发必须是更晚的一根 (pb.py:65)
            return None

        if not (s1 == b and s5 == b):
            return None
        if fill_idx >= n:                # pb.py:162
            return None
        # 时段/暖机闸门只挡动作。不消耗 pullback_seen —— 级联好不容易凑齐的一次
        # 回调不该被一根不可交易的 bar 白白吃掉。
        if not can_enter[i]:
            return None
        # 时段末根不开仓: 否则这一批既在这根成交、又在这根被强平, 产生一条 0 根的
        # END_SESSION 腿。与 rbreaker_backtest.py:242 的 `not last_of_sess[i]` 同一
        # 个理由。必须按 **fill_idx** 判定 —— 成形根口径下成交发生在 i+1。
        # 位置同样在 pullback_seen 清零**之前**: 被挡住的触发不消耗级联。
        if last_of_sess[fill_idx]:
            return None
        pullback_seen = False            # pb.py:68 触发后清零, 下次入场要重等一轮
        atr_dec = risk_atr(i, prev_bar=completed_bar)
        if np.isnan(atr_dec):            # pb.py:175 —— 信号仍被消耗掉, 但不开仓
            return None
        return _open(int(b), fill_idx, op, atr_dec, swings, P30, p,
                     "reentry" if had_prior_batch else "open")

    for i in range(n):
        D, P5, P30 = int(cd[i]), int(c5[i]), int(c30[i])

        # ------------------------------------------------ 1) 冷静期 (日频) ----
        # pb.py:205-206。v2.1 只在 klines_2h 就绪时调, 这里对应 D >= 0。
        if D >= 0:
            # A5: 每根**新日线**才真评一次。不门控的话 `stable_count` 会按调用次数
            # 推进 (3 根 1m = 3 分钟, 而不是 3 个交易日), 冷静期形同虚设。
            if (not cfg.cooldown_counts_daily_bars) or D != last_daily_pos:
                last_daily_pos = D
                if not cool_active:
                    # 触发器 2: 单根日线实体吃穿 >=2 条均线 (cooldown.py:10-11)。
                    if structure_break_arms and DBREAK[D]:
                        cool_active = True
                        stable_state, stable_count = _NO_STATE, 0
                        consecutive_sl, last_sl_dir = 0, 0
                else:
                    s = DSTATE[D]
                    if s == NEUTRAL or s != stable_state:
                        stable_state = s
                        stable_count = 1 if s != NEUTRAL else 0
                    else:
                        stable_count += 1
                    if s != NEUTRAL and stable_count >= p.cooldown_release_bars:
                        cool_active = False
                        stable_state, stable_count = _NO_STATE, 0

        # ------------------------------ 2) 入场尝试 (完成根口径: 排在出场之前) ----
        # 决策与成交都只用到 bar i 开盘那一刻已知的信息, 所以能排在出场之前。
        if completed_bar and batch is None:
            batch = attempt_entry(i, D, P5, P30)

        # -------------------------------------------------------- 3) 出场 ----
        if batch is not None and batch.entry_bar_idx <= i:
            is_long = batch.direction == BULL
            bo, bh, bl, bc = op[i], hi[i], lo[i], cl[i]
            if not trail_extreme_prev:
                # pre-A3: 本根极值在算吊灯**之前**折入 (乐观 —— 假设高点先于低点)
                batch.extreme_since_entry = (
                    max(batch.extreme_since_entry, bh) if is_long
                    else min(batch.extreme_since_entry, bl))

            if not batch.partial_taken:
                trigger = partial_trigger_price(
                    batch.entry_price, is_long, batch.stop_loss, p)
                stop_arr[i] = batch.stop_loss
                if range_trigger:          # A1: 本根 high/low 触及即算触发
                    hit_sl = bl <= batch.stop_loss if is_long else bh >= batch.stop_loss
                    hit_tp = bh >= trigger if is_long else bl <= trigger
                else:
                    hit_sl = bc <= batch.stop_loss if is_long else bc >= batch.stop_loss
                    hit_tp = bc >= trigger if is_long else bc <= trigger
                # 止损与止盈落在同一根里 -> 悲观地先走止损
                if hit_sl and (pessimistic or not hit_tp):
                    fill = gap_filled(batch.stop_loss, is_long, bo, gap_fill)
                    trades.append(_leg(batch, i, fill, SL, 1.0, times, tds, sid, cfg,
                                       gapped=fill != batch.stop_loss))
                    capital *= 1.0 + trades[-1].pnl_net
                    # 同方向连续**纯**止损计数; 止盈或护盈止损会把它清零
                    count_directional_loss(batch.direction)
                    batch, had_prior_batch = None, True
                elif hit_tp:
                    # 止盈是挂在触发价的限价单 —— 不做跳空补价, "止盈不占便宜"
                    trades.append(_leg(batch, i, trigger, TP, p.partial_ratio,
                                       times, tds, sid, cfg, gapped=False))
                    capital *= 1.0 + p.partial_ratio * trades[-1].pnl_net
                    consecutive_sl, last_sl_dir = 0, 0
                    batch.partial_taken = True
                    batch.trailing_stop = batch.stop_loss
            else:
                # 永远不用本根自己的 ATR: 它需要本根的收盘价, 而本根的价格马上就要
                # 拿去和一条依赖它的止损线比较。A6 下这读的是最后一根已收 30m。
                atr_prev = risk_atr(i, prev_bar=True)
                if not np.isnan(atr_prev):
                    batch.trailing_stop = chandelier_stop(
                        is_long, batch.extreme_since_entry, atr_prev, batch.stop_loss, p)
                stop_arr[i] = batch.trailing_stop
                if range_trigger:
                    hit = bl <= batch.trailing_stop if is_long else bh >= batch.trailing_stop
                else:
                    hit = bc <= batch.trailing_stop if is_long else bc >= batch.trailing_stop
                if hit:
                    remaining = 1.0 - p.partial_ratio
                    fill = gap_filled(batch.trailing_stop, is_long, bo, gap_fill)
                    trades.append(_leg(batch, i, fill, PROTECTIVE_SL, remaining,
                                       times, tds, sid, cfg,
                                       gapped=fill != batch.trailing_stop))
                    capital *= 1.0 + remaining * trades[-1].pnl_net
                    # 护盈止损**什么也不做**: 方向判断本身是对的, 只是剩余仓位回吐
                    # 了部分利润 —— 与"看错方向"不是一回事 (`cooldown.py:5-8`)。
                    #
                    # 这里曾经写的是清零。那行**恒等于空操作**: 走到这一支必然
                    # `partial_taken == True`, 而它只由 TP 那一支置位, TP 已经清过
                    # 零了; 两条腿之间同一批一直持着仓, 不可能有别的批次去改计数器。
                    # 删掉是为了别让人以为护盈止损对冷静期有话语权。
                    batch, had_prior_batch = None, True

            # 时段末强平 (cfg.session_forces_flat)。按 close 成交, 不做跳空补价 ——
            # 与 rbreaker_backtest.py:203-206 逐字同构。
            #
            # **不是 elif**: 同一根上 2R 止盈刚平掉一半时, 剩下那一半也要在这根被
            # 强平, 于是这一批产生两条腿 (TP 50% + END_SESSION 50%)。所以 frac 要
            # 看 partial_taken —— 写死 1.0 会让 Σ size_fraction 变成 1.5, 而
            # PBTrade.pnl_net 的成本口径和 pullback_stats.batch_pnls 都依赖它等于 1。
            if batch is not None and last_of_sess[i]:
                frac = (1.0 - p.partial_ratio) if batch.partial_taken else 1.0
                trades.append(_leg(batch, i, float(cl[i]), END_SESSION, frac,
                                   times, tds, sid, cfg, gapped=False))
                capital *= 1.0 + frac * trades[-1].pnl_net
                # 冷静期计数器: **什么也不做**。时段强平是日历驱动的, 不含任何方向
                # 判断信息 —— 既不像纯止损那样"确认看错了方向"(加一), 也不像止盈
                # 那样"这一轮赚到了"(清零)。
                #
                # 已知代价, 如实记在这里: 锁死之后收盘平仓占了七成批次, 而它对计数
                # 器透明, 于是三笔真止损可以横跨一周、中间夹着五个收盘批却仍被算成
                # "连续三次"。实测 RB/split 下四次连续止损冷静期**全部**是这种。
                # 曾试过"亏损的整批收盘算一次看错": 59 品种 x 5 模式下冷静期
                # +25%、批次 -14%, 逐组合收益 129 好 / 152 差 (噪声), 于是回退 ——
                # 换来的行为改变远大于它解决的问题。要收紧的话该给连续性加时间上限,
                # 而不是把收盘平仓塞进这个计数器。
                batch, had_prior_batch = None, True

            # A3: 本根 high/low 只在吊灯**测试之后**折入, 所以拿去和本根比较的那条
            # 线永远不依赖本根。带守卫 —— 上面可能刚把这一批平掉。
            if trail_extreme_prev and batch is not None:
                batch.extreme_since_entry = (
                    max(batch.extreme_since_entry, hi[i]) if batch.direction == BULL
                    else min(batch.extreme_since_entry, lo[i]))

        # -------------------------- 4) 入场尝试 (成形根口径: 排在出场之后) ----
        # pre-2.1 口径: 决策读到 close[i], 所以要等本根走完, 成交推到 i+1 的 open。
        if not completed_bar and batch is None:
            batch = attempt_entry(i, D, P5, P30)

        # ------------------------------------- 5) 逐根 mark-to-market ----
        # 成形根口径下, 本根刚建的批次 entry_bar_idx == i+1, 还没生效 —— 这个守卫
        # 同时覆盖两种口径。
        unrealized = 0.0
        if batch is not None and batch.entry_bar_idx <= i:
            pos_arr[i] = np.int8(batch.direction)
            move = (cl[i] - batch.entry_price) if batch.direction == BULL \
                else (batch.entry_price - cl[i])
            frac = (1.0 - p.partial_ratio) if batch.partial_taken else 1.0
            unrealized = frac * (move / batch.entry_price - cost)
        equity[i] = capital * (1.0 + unrealized)

        # --------------------------------------- 6) 冷静期区间 (仅观测) ----
        # 在 bar 末尾判, 所以本根出场 armed 的冷静期算在本根头上。
        if cool_active:
            if cd_start is None:
                cd_start = i
        elif cd_start is not None:
            cooldown_spans.append((cd_start, i - 1))
            cd_start = None

    if cd_start is not None:
        cooldown_spans.append((cd_start, n - 1))

    if forces_flat:
        # 帧末根的 is_session_last 恒为 True (v3_sessions.py:340), 且时段末根不入场
        # —— 锁死下不可能还留着仓位。与 rbreaker_backtest.py:271 同一条断言。
        assert batch is None, "时段锁死下回测结束仍有持仓 —— is_session_last 的构造被破坏了"

    w = prepared.warmup_bars
    idx = pd.DatetimeIndex(times[w:], name="datetime")
    eq = pd.Series(equity[w:], index=idx)
    ok = can_enter
    return PullbackResult(
        symbol=prepared.symbol, trades=trades,
        equity_curve=eq, trading_date=pd.Series(tds[w:], index=idx),
        position=pos_arr, stop_line=stop_arr,
        pullback_points=pullback_points, cooldown_spans=cooldown_spans,
        warmup_bars=w, warmup_days=prepared.warmup_days,
        tradable_days=int(pd.unique(tds[ok]).size) if ok.any() else 0,
        params=p, config=cfg, mode=prepared.mode,
        dropped_runs=prepared.dropped_runs,
    )


# ------------------------------------------------------------- helpers ----
def _open(
    direction: int, fill_idx: int, op: np.ndarray, atr_dec: float,
    swings: SwingIndex, p30: int, p: PullbackParams, signal_type: str,
) -> _Batch:
    """pb.py:176-188。成交价是 bar `fill_idx` 的 open, 不是信号那一刻的 close。"""
    is_long = direction == BULL
    entry_price = float(op[fill_idx])
    # 平台位只到下标 p30 - swing_k —— 复现 `range(k, len(klines)-k)` 的上界。
    plat = nearest_platform(swings, p30 - p.swing_k, entry_price, is_long)
    return _Batch(
        direction=direction, entry_bar_idx=fill_idx, entry_price=entry_price,
        stop_loss=stop_level(entry_price, is_long, atr_dec, plat, p),
        signal_type=signal_type, extreme_since_entry=entry_price,
    )


def _leg(
    batch: _Batch, i: int, exit_price: float, reason: str, size_fraction: float,
    times: np.ndarray, tds: np.ndarray, sid: np.ndarray, cfg: PullbackConfig,
    gapped: bool,
) -> PBTrade:
    return PBTrade(
        direction="long" if batch.direction == BULL else "short",
        entry_bar_idx=batch.entry_bar_idx, entry_time=pd.Timestamp(times[batch.entry_bar_idx]),
        entry_price=batch.entry_price,
        exit_bar_idx=i, exit_time=pd.Timestamp(times[i]), exit_price=float(exit_price),
        reason=reason, size_fraction=size_fraction, signal_type=batch.signal_type,
        # bool() 而不是裸表达式: 比较结果是 np.bool_, 会让 `t.gapped is True` 意外失败。
        gapped=bool(gapped), stop_loss=batch.stop_loss,
        trading_date=pd.Timestamp(tds[i]), session_id=int(sid[i]), cost_bps=cfg.cost_bps,
    )


def _empty_result(prepared: Prepared, p: PullbackParams, cfg: PullbackConfig) -> PullbackResult:
    return PullbackResult(
        symbol=prepared.symbol, trades=[],
        equity_curve=pd.Series(dtype=float),
        trading_date=pd.Series(dtype="datetime64[ns]"),
        position=np.zeros(0, dtype=np.int8), stop_line=np.zeros(0, dtype="float64"),
        pullback_points=[], cooldown_spans=[],
        warmup_bars=0, warmup_days=prepared.warmup_days, tradable_days=0,
        params=p, config=cfg, mode=prepared.mode, dropped_runs=prepared.dropped_runs,
    )
