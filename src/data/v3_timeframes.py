# -*- coding: utf-8 -*-
"""
v3.0 数据层: 把 1 分钟帧切成 1m / 5m / 30m / 1d 四套 bar, 并给出「截至 1m bar i,
各周期最后一根**已收盘** bar 是哪根」。

这是回调策略移植里唯一真正新增的数据结构。v2.1 的对应物是
`src/data/resample.py` 的 `resample_ohlcv` + `closed_bar_positions`, 但那两个
函数按**纽约交易日**分组、用**开盘时刻**打标、且假设时段内部没有断口 —— 三条
都不适用于中国期货, 所以重写而不是复用。

===========================================================================
一、bar 的标签口径 (最容易搞错的一条)
===========================================================================
v3.0 的 1 分钟 bar 打标在**分钟末端**: 标 09:01 的那根覆盖 [09:00, 09:01)。
(v2.1 的 `resample_ohlcv` 恰好相反, 用 `label="left"` 打在开盘时刻。)

所以这里统一成:

    bar i 的开盘时刻  start_i    = index[i] - 1min
    合成 bar 的标签   end_label  = 它**最后一根** 1m 成分的标签

于是 30m bar 标 09:30 覆盖 [09:00, 09:30), 与 1m bar 标 09:01 覆盖 [09:00, 09:01)
是同一套语义 —— 图上、markers、十字线吸附、以及"哪根已收盘"全部共用这一条。

===========================================================================
二、分箱: 段内按时钟, 键里带段号
===========================================================================
    bin 键 = (原子段号, start_i.floor(rule))

**段号必须进键。** 原子段 (`v3_sessions.derive_segments`) 是「trading_date 变化
或时钟间隔 > 3h」切出来的最小单位, 也就是夜盘段和白盘段。不带段号的话, 一根
30m bar 会横跨夜→日那 6.5 小时的断口 (AU/AG/SC 是 02:30 -> 09:01), 甚至横跨周末
那 58 小时。带上段号之后, **任何一根合成 bar 都不可能跨段**, 这一条由
`tests/test_v3_timeframes.py` 钉死。

**时段模式 (5 种) 不参与分箱。** 分箱永远用原子段 —— `night_day` / `day_night`
把两个原子段并成一个"时段"是**交易纪律**层面的事 (哪些 bar 能开仓), 不能让它
把 K 线也粘起来, 否则合并模式下会画出一根跨 6 小时空白的 30m bar。

段内按**时钟**而不是按根数 (决策 1): 日盘 225 根 1m 切 30m 得 8 根, 其中跨
10:15 小节休息的那一根只有 15 根 1m。这与国内行情软件的口径一致。日盘的三段
75/60/90 根都能被 5 整除, 所以 5m 永远不会被断口截短, 只有 30m 会。

**日线走单独分支**: 键直接用 `trading_date`, 不用时钟也不用段号 —— 一个交易日
= 夜盘段 + 白盘段, 这正是中国期货的交易日定义, 也是 `v3_sessions.daily_ohlc`
已经在用的口径。

---------------------------------------------------------------------------
集合竞价 bar: 段首孤儿箱要并进下一箱 (`merge_leading_singleton`, 默认开)
---------------------------------------------------------------------------
数据里每个时段的**第一根** bar 是集合竞价的撮合结果, 而不是一分钟的连续交易:

    RB 2025-07-01 21:00  o=h=l=c=3003.0  vol=4456    <- 20:59-21:00 的竞价撮合
    UR 2025-07-01 09:00  o=1707 h=1707 l=1706 c=1706 vol=1277
    IF 2025-07-01 09:30  o=h=l=c=3889.2  vol=59

它按"分钟末端"打标, 于是 `start = 20:59`, 被 floor 到 20:30 那一格 —— **自己独占
一个箱**。实测 2025-07 一个月里 RB/AU/UR/IF/CF 的 30m 序列各有 24-25 个这样的
孤儿箱, 占全部 30m 根数的 7.7%; 而且它们零振幅, 会同时污染 30m 的 MA 和 ATR ——
偏偏 30m 的 ATR 正是策略拿来定止损距离的那一个 (A6)。

对策: **一个原子段的首根若独占自己的时钟箱, 就并进下一箱。** 这条规则不读任何
时钟点(只比较相邻两根的箱键是否不同), 因此不违反"不许写死时段时间"的硬规则。
实测它恰好且仅仅命中集合竞价那一根 —— 上面五个品种的**全部**孤儿箱都是段首根。

并进去还顺带修对了一件事: 合成 bar 的 `open` 因此取的是**竞价撮合价**, 而这正是
交易所定义的开盘价 (见 docs/01-平台层/v3.0-PLAN-中国期货特殊规则应对方案.md
§1.3: "开盘价 = 集合竞价撮合价, 不是第一笔连续成交")。

关掉它 (`merge_leading_singleton=False`) 会退回逐字面的 resample —— 留着这个开关
是沿用仓库"每个会改变结果的行为都要有具名开关"的约定 (src/config.py 的抬头)。

===========================================================================
三、已收盘位置 (无未来函数的核心)
===========================================================================
    closed_pos[i] = searchsorted(end_labels, start_i, side="right") - 1

即 `end_labels[b] <= start_i` <=> 第 b 根在 bar i **开盘之前**已经收完。没有任何
一根已收盘时为 -1。三处自检 (都在测试里):

  * **1m**: 恒等于 `i - 1` —— 正是 v2.1 的 `small_5m = df_5m.iloc[:i]`。
  * **30m**: bar 10:31 (start 10:30) 的 closed_pos 指向标签 10:15 那根 (覆盖
    10:01-10:15)。断口处不会把还没开始的 10:30 那根误认为已收。
  * **1d**: bar i 所在的 trading_date 还在形成中, searchsorted 自动落到**上一个**
    trading_date —— 与 `prev_day_lines` 的 `agg.shift(1)` 同构。

推论: **日线 MA 状态每个交易日只在交易日起点变一次** (有夜盘品种是 21:00, 无夜盘
是 09:00)。这是「原策略 2h -> 新策略 1d」这个重映射的直接后果。

`day_night` 模式下一个时段跨两个 trading_date, 日线下标会在 21:00 处前进一格。
**不需要像 R-Breaker 的六线那样按时段钉住**: 六线要钉是因为线一跳、持仓中的止损位
就跟着平移; 而日线在这里只喂趋势方向, 方向变了既不移动止损 (止损在入场时定死、
吊灯读 30m ATR) 也不触发平仓 (本策略只有 SL/TP/吊灯三个出口)。逐根取"上一个已收
trading_date"本身就是因果的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.v3_sessions import derive_segments

# 支持的全部周期。键是对外的名字, 值是 pandas 的 floor 频率;
#   None -> 每根 1m 自成一箱 (恒等)
#   "D"  -> 走 trading_date 分支, 不读时钟 (唯一允许跨原子段的周期)
# 其余一律走 `(原子段号, start_i.floor(rule))` 分箱, `_bin_of` 对 rule 是通用的。
PB_RULES: dict[str, str | None] = {
    "1m": None, "5m": "5min", "15m": "15min", "30m": "30min", "2h": "2h", "1d": "D",
}

# v3.0 回调引擎与 chart_pullback 用的四周期集。**不要动它** —— 下游是按名字硬取的
# (engine/pullback_v3_backtest.py:243-250 取 1m/5m/30m/1d, viz/chart_pullback.py:72
# 的 GRID 写死 2x2)。要别的周期集就在调用点自己传 `tfs=(...)`, 不要改这个 tuple。
PB_TFS: tuple[str, ...] = ("1m", "5m", "30m", "1d")

# 15m 与 2h 在 AU 上的实测分箱形状 (2026-05-01 ~ 07-29, 58 个交易日):
#   15m  跨原子段 0 个; 箱大小 {15: 2046, 16: 56}   —— 干净, 日盘三段 75/60/90 都被 15 整除
#   2h   跨原子段 0 个; 箱大小 {120:112, 75:58, 61:56, 60:116, 30:114}
# **2h 有一半的箱是短的**, 一根实际含 30~120 分钟 (4 倍极差), 而 MA 对它们等权平均;
# 且每交易日的根数随夜盘档位从 4 到 8 不等 (AU 是 7.9)。这是分箱网格锚在午夜、与交易
# 时段形状无关造成的, 不是市场造成的。引用任何 2h 上的数字都必须声明这一条。
# 判据与钉死它的测试见 obsidian/01-周期/周期-2h.md 与 tests/test_v3_timeframes.py。

_BAR = pd.Timedelta("1min")
_BAR_NS = np.timedelta64(1, "m")
_OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class TFBars:
    """一个周期的合成 bar + 它与 1m 帧的两个映射。"""

    tf: str
    # index = 末端标签 (DatetimeIndex, 严格递增); 列 open/high/low/close/volume/trading_date
    bars: pd.DataFrame
    # 每根 1m bar 属于第几根合成 bar。0 基、稠密、非降 —— np.bincount 和
    # np.searchsorted 都依赖这三条, 构造时断言。
    bin_of: np.ndarray
    # 截至 1m bar i 的**开盘时刻**, 最后一根已收盘合成 bar 的下标; 一根都没有则 -1。
    closed_pos: np.ndarray

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def end_labels(self) -> np.ndarray:
        return self.bars.index.to_numpy()

    def first_bin_at_or_after(self, bar_idx: int) -> int:
        """
        1m 下标 -> 它所属的合成 bar 下标。用来把暖机段从显示帧里切掉。

        因为 bin 不跨原子段、原子段不跨 trading_date, 而暖机段的切点正好是一个
        trading_date 边界, 所以边界处不会出现"半根被切开"的 bin。
        """
        if bar_idx <= 0:
            return 0
        if bar_idx >= len(self.bin_of):
            return self.n_bars
        return int(self.bin_of[bar_idx])


# ------------------------------------------------------------- 分箱键 ----
def _bin_of(
    df: pd.DataFrame, rule: str | None, segments: np.ndarray,
    merge_leading_singleton: bool = True,
) -> np.ndarray:
    """0 基稠密的合成 bar 编号。见模块 docstring 第二节。"""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if rule is None:                      # 1m: 恒等
        return np.arange(n, dtype=np.int64)

    if rule == "D":                       # 日线: 交易日键, 不读时钟
        keys = df["trading_date"].to_numpy()
        changed = np.r_[True, keys[1:] != keys[:-1]]
    else:                                 # 段内按时钟 floor
        start = df.index.to_series().sub(_BAR).dt.floor(rule).to_numpy()
        new_seg = np.r_[True, segments[1:] != segments[:-1]]
        changed = np.r_[True, start[1:] != start[:-1]] | new_seg
        if merge_leading_singleton:
            # 段首根独占自己那一格时钟箱 <=> 它的下一根又开了一个新箱。
            # 把那个"新箱"标记取消, 于是段首根与下一箱合并。见 docstring。
            # 移位: lone[j] 为真表示**下标 j 的 changed 要抹掉**。
            lone = np.zeros(n, dtype=bool)
            lone[1:] = new_seg[:-1] & changed[1:] & ~new_seg[1:]
            changed = changed & ~lone
    return (np.cumsum(changed) - 1).astype(np.int64)


def _aggregate(df: pd.DataFrame, bin_of: np.ndarray) -> pd.DataFrame:
    """
    按 bin 聚合 OHLCV。用 `reduceat` 而不是 groupby —— 130 万根 1m 上实测快一个
    量级, 而切品种时这个差别是"瞬时"和"卡住"的区别 (同 rbreaker_backtest.py 用
    numpy 标量循环而不是 df.iloc 的理由)。
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=[*_OHLCV, "trading_date"],
            index=pd.DatetimeIndex([], name="datetime"),
        )
    edge = np.r_[True, bin_of[1:] != bin_of[:-1]]
    first = np.flatnonzero(edge)
    last = np.r_[first[1:] - 1, n - 1]

    hi = df["high"].to_numpy("float64")
    lo = df["low"].to_numpy("float64")
    vol = df["volume"].to_numpy("float64")
    out = pd.DataFrame(
        {
            "open": df["open"].to_numpy("float64")[first],
            "high": np.maximum.reduceat(hi, first),
            "low": np.minimum.reduceat(lo, first),
            "close": df["close"].to_numpy("float64")[last],
            "volume": np.add.reduceat(vol, first),
            # 末根的 trading_date: 日线分支下整箱同值; 其余周期下 bin 不跨段、
            # 段不跨 trading_date, 所以同样整箱同值。
            "trading_date": df["trading_date"].to_numpy()[last],
        },
        # 标签取**末根**的时刻 —— 见模块 docstring 第一节。
        index=pd.DatetimeIndex(df.index.to_numpy()[last], name="datetime"),
    )
    return out


def _closed_pos(df: pd.DataFrame, end_labels: np.ndarray) -> np.ndarray:
    """见模块 docstring 第三节。"""
    if len(df) == 0:
        return np.zeros(0, dtype=np.int64)
    starts = df.index.to_numpy() - _BAR_NS
    return np.searchsorted(end_labels, starts, side="right").astype(np.int64) - 1


# ------------------------------------------------------------- 构造 ----
def build_timeframe(
    df_1m: pd.DataFrame, tf: str, segments: np.ndarray | None = None,
    merge_leading_singleton: bool = True,
) -> TFBars:
    """单个周期。`segments` 省略时现算 —— 批量构造请用 `build_timeframes` 复用一次。"""
    if tf not in PB_RULES:
        raise ValueError(f"未知周期 {tf!r}; 可选: {list(PB_RULES)}")
    if segments is None:
        segments = derive_segments(df_1m) if len(df_1m) else np.zeros(0, dtype=np.int64)

    bin_of = _bin_of(df_1m, PB_RULES[tf], segments, merge_leading_singleton)
    bars = _aggregate(df_1m, bin_of)
    closed = _closed_pos(df_1m, bars.index.to_numpy())
    out = TFBars(tf=tf, bars=bars, bin_of=bin_of, closed_pos=closed)
    _assert_shape(out, segments, df_1m)
    return out


def build_timeframes(
    df_1m: pd.DataFrame, segments: np.ndarray | None = None,
    tfs: tuple[str, ...] = PB_TFS, merge_leading_singleton: bool = True,
) -> dict[str, TFBars]:
    """
    四个周期一次建好。`df_1m` 需要 DatetimeIndex (严格递增) + open/high/low/close/
    volume/trading_date 六列 —— 也就是 `v3_sessions.prepare_base` 输出的形状。
    """
    missing = [c for c in (*_OHLCV, "trading_date") if c not in df_1m.columns]
    if missing:
        raise ValueError(f"df_1m 缺列: {missing}")
    if len(df_1m) and not df_1m.index.is_monotonic_increasing:
        raise ValueError("df_1m 索引非单调递增 —— 所有分箱和 searchsorted 都会错")
    if segments is None:
        segments = derive_segments(df_1m) if len(df_1m) else np.zeros(0, dtype=np.int64)
    return {tf: build_timeframe(df_1m, tf, segments, merge_leading_singleton)
            for tf in tfs}


def _assert_shape(t: TFBars, segments: np.ndarray, df_1m: pd.DataFrame) -> None:
    """
    宁可炸也不要静默错分 —— 与 `v3_sessions.assert_no_ambiguous_gaps` /
    `assert_segment_shape` 同一个套路。
    """
    if t.n_bars == 0:
        return
    b = t.bin_of
    if b[0] != 0 or (np.diff(b) < 0).any() or int(b[-1]) + 1 != t.n_bars:
        raise ValueError(f"{t.tf}: bin_of 不是 0 基稠密非降的")
    if not t.bars.index.is_monotonic_increasing or t.bars.index.has_duplicates:
        raise ValueError(f"{t.tf}: 合成 bar 的末端标签不是严格递增的")
    # 日线**允许**跨段(一个交易日 = 夜盘段 + 白盘段); 其余周期一根都不许跨。
    if t.tf != "1d":
        edge = np.r_[True, b[1:] != b[:-1]]
        first = np.flatnonzero(edge)
        last = np.r_[first[1:] - 1, len(b) - 1]
        if (segments[first] != segments[last]).any():
            i = int(np.flatnonzero(segments[first] != segments[last])[0])
            raise ValueError(
                f"{t.tf}: 第 {i} 根合成 bar 跨了原子段 "
                f"({df_1m.index[first[i]]} -> {df_1m.index[last[i]]})"
            )
