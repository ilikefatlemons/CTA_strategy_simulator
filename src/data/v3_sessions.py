# -*- coding: utf-8 -*-
"""
v3.0 数据层: 交易时段推导 + 脏数据清洗 + 前一交易日 OHLC。

这些都是**交易所日历的事实**, 不是某个策略的选择 —— 后续每个 v3.0 策略都要
用, 所以放在 src/data/ 而不是 src/strategy/。`v3_` 前缀是为了和 v2.1 那套
美股 5 分钟的时段辅助 (resample.py / higher_tf_indicators.py) 区分开。

管线顺序是固定的, 反了就出错:

    加载 -> 丢伪造段 -> 切时段 -> 复权 -> 日聚合 -> 前日六线

先丢伪造段再聚合: 否则伪造段里的平价、以及被错标到 02:30 的集合竞价 bar
(单根 +4% 甚至 +13%) 会直接污染次日六条线的 H/L。

---------------------------------------------------------------------------
时段的定义
---------------------------------------------------------------------------
`trading_date` 是权威的交易日键 —— 夜盘 bar 已经带着**下一个**交易日的
trading_date (21:00 开的夜盘属于次日; 周五夜盘属于下周一)。一个 trading_date
内部再按「时钟间隔 > 3 小时」切成日盘 / 夜盘两段。

docs 的硬规则是「不许写死时段时间或每日 bar 数」(v3.0-SPEC-数据质量排查判据.md
:174)。这里两样都没写死: 没有时钟点, 没有 bar 数。写死的只是一个间隔阈值,
而它落在一条**实测空带**里。24 个品种全历史实测 (覆盖 23:00 / 01:00 / 02:30
三档夜盘 + 中金所 + 无夜盘):

    trading_date 内部出现过的间隔:  0.017h 0.033h 0.05h 0.1h 0.267h 0.283h
                                    2.017h  |  6.517h 8.017h 9.517h 10.017h
                                    54.517h 56.017h 57.517h 58.017h
                                              ^^^^^^^^^^^^^^^^^^^^ 夜->日 / 跨周末

    日内最大间隔      = 2.017h  (商品午休 11:31-13:30)
    夜->日最小间隔    = 6.517h  (AU/AG/SC 02:30 -> 09:01)
    落在 (2.25h, 6h) 的间隔 = **0 个**

空带两侧各有约 2.7 倍余量。配一条运行时断言 (`assert_no_ambiguous_gaps`)
盯着它, 真出现异常品种会立刻炸而不是被静默错切。

考虑过并否决的全推导方案:
  * 按间隔**比值**聚类 —— 0.267h->2.017h 是 7.5 倍, 比 2.017h->6.517h 的
    3.2 倍还大, 会把 10:16-10:30 的小节休息切成独立时段。
  * 按间隔**绝对差**聚类 —— 对 31 个无夜盘品种失效: 它们的间隔集合只有
    {0.017, 0.267, 2.017}, 最大跳变 1.75h 出现在午休处, 会把午休切开。
  任何补救都得引入一个下限常数, 引入之后推导本身就没有意义了。

这套切法自动覆盖:
  * 2020 年 2-5 月全市场停夜盘 —— 该 trading_date 没有 21:00 段, 日内最大
    间隔 2.02h < 3h, 自然退化成单时段。(实测 RB 有 1349 个单时段交易日)
  * 中金所 IF/IC/IH/IM/T/TF/TS/TL —— 唯一间隔 1.517h (11:30->13:01),
    单时段, 09:15 和 09:30 两个时代都对。
  * AU/AG/SC 跨零点 —— 21:00->02:30 是一条连续的 1 分钟段, 单时段, 末根 02:30。
  * 周五夜盘归下周一 —— 54.5h / 58.0h 的间隔, 正确切开。
  * 09:00 vs 09:01 的集合竞价差异 —— 无关, 没有任何逻辑读时钟。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.paths import CLEAN_DIR  # noqa: F401  (转出口: 既有调用点仍从这里 import)
from src.strategy.rbreaker import LINE_NAMES, RBreakerConfig, RBreakerParams, rbreaker_lines

# 时段切分阈值。见模块 docstring 里的实测分布。
SESSION_SPLIT_GAP = pd.Timedelta("3h")

# 断言守卫的「模糊带」。实测空带是 (2.02h, 6.52h), 这里取内缩的一段。
AMBIGUOUS_LO = pd.Timedelta("2h15m")
AMBIGUOUS_HI = pd.Timedelta("6h")

_RAW_COLS = ["open", "high", "low", "close", "volume", "K", "trading_date"]


# ------------------------------------------------------------------ 加载 ----
def load_minute_raw(
    symbol: str, start: str | None = None, end: str | None = None,
    clean_dir: str = CLEAN_DIR,
) -> pd.DataFrame:
    """
    读一个品种的分钟 parquet 并切窗口。**不复权** —— 复权是 `adjust_prices`
    的事, 因为复权因子 k0 要由显示窗口的首根决定, 而这里可能带着 padding。

    index = `datetime` (tz-naive 北京墙钟, bar 按分钟**末端**打标)。
    """
    path = os.path.join(clean_dir, f"{symbol}.parquet")
    df = pd.read_parquet(path, columns=_RAW_COLS)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        # 与 chart_minute.load_minute 同口径: end 当日整天都算在内
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: parquet 索引非单调递增, 后面所有 groupby/diff 都会错")
    return df


def adjust_prices(df: pd.DataFrame, k0: float) -> pd.DataFrame:
    """
    后复权: `adj = close / K * k0`。

    源数据本身是后复权的 (锚在上市首约), 直接画 close/K 纵轴会严重偏离真实价,
    所以再乘一个常数 k0。**不要乘 K_last** —— 那是前复权, 补新数据时整段历史
    都要重算。见 data/01-pkl层/一次排查/v3.0-SPEC-数据质量排查判据.md D2/D8。

    volume 不参与复权: K 是价格因子, 手数是手数。
    """
    out = df.copy()
    k = df["K"].to_numpy()
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy() / k * k0
    return out


# -------------------------------------------------------------- 脏数据 ----
def contiguous_runs(df: pd.DataFrame) -> np.ndarray:
    """
    `(trading_date, 连续 1 分钟段)` 的段编号。

    与 data/01-pkl层/一次排查/排查脚本/qc_fake_session_survivors.py:39-43 逐字同源 —— 那个
    QC 脚本就是用这个粒度找出 ETL 残留的 643 条伪造段的。
    """
    gap = df.index.to_series().diff() > pd.Timedelta("1min")
    tdc = df["trading_date"].ne(df["trading_date"].shift())
    # 首根的 tdc 必为 True (shift 出 NaT), 所以 cumsum 从 1 起 —— 减 1 转成
    # 0 基, 否则 bincount 会多出一个空的 0 号段, 被当成"零成交段"误报。
    return (gap | tdc).cumsum().to_numpy() - 1


def drop_dead_runs(
    df: pd.DataFrame, min_bars: int = 20, max_traded: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    丢掉「有 bar 但基本没成交」的伪造时段。返回 (清洗后的帧, 被丢段的明细)。

    判据: `bars >= min_bars 且 traded <= max_traded`, 或者整段成交量为 0。

    **必须用 `traded <= 2` 而不是 `vol == 0`。** ETL (prepare_v3_minute.py:77-84)
    的段粒度是 `(sym, trading_date, 日历日, 日/夜)`, 比正确的
    `(trading_date, 连续段)` 粗; 残留下来的 643 条伪造段的 `vol > 0`, 正是
    因为一根被错标到 02:30/01:00 的集合竞价 bar 带着真实成交量, 把整段
    零成交的夜盘碎片"救活"了。最恶劣的一例 AG 2026-02-24 02:30 单根 +13.00%。
    这种 bar 会在市场根本没开的时刻触发止损和突破信号。

    `min_bars` 是一道保险: 窗口边缘被切出来的十几根真实碎片不会被误吃。

    被丢的段一律返回明细, **绝不静默丢弃** —— 调用方应该打印或挂到结果上。
    """
    if df.empty:
        return df, pd.DataFrame(columns=["trading_date", "start", "end", "bars", "traded", "vol"])

    run = contiguous_runs(df)
    vol = df["volume"].to_numpy()
    traded = (vol > 0).astype(np.int64)

    n_runs = int(run.max()) + 1 if len(run) else 0
    bars_per = np.bincount(run, minlength=n_runs)
    vol_per = np.bincount(run, weights=vol, minlength=n_runs)
    traded_per = np.bincount(run, weights=traded, minlength=n_runs).astype(np.int64)

    dead = ((bars_per >= min_bars) & (traded_per <= max_traded)) | (vol_per == 0)
    mask = dead[run]
    if not mask.any():
        return df, pd.DataFrame(columns=["trading_date", "start", "end", "bars", "traded", "vol"])

    idx = np.flatnonzero(dead)
    times = df.index.to_numpy()
    tds = df["trading_date"].to_numpy()
    starts = np.searchsorted(run, idx, side="left")
    ends = np.searchsorted(run, idx, side="right") - 1
    dropped = pd.DataFrame({
        "trading_date": tds[starts],
        "start": times[starts],
        "end": times[ends],
        "bars": bars_per[idx],
        "traded": traded_per[idx],
        "vol": vol_per[idx],
    })
    return df[~mask], dropped


# -------------------------------------------------------------- 时段切分 ----
#
# 分两层:
#   **原子段 (segment)** —— 「trading_date 变化 或 时钟间隔 > 3h」切出来的最小
#     单位, 就是夜盘段和白盘段。五种模式都建立在它之上, 它本身不随模式变化。
#   **时段 (session)** —— 由原子段按模式分组而成。「一个时段最多一笔 + 时段
#     结束强制平仓」这两条纪律作用在时段上, 所以模式直接决定了一天做几笔、
#     持仓能多长、强平在哪一刻。

SESSION_MODES: dict[str, str] = {
    "night": "夜盘",              # 只有夜盘段可交易
    "day": "日盘",                # 只有白盘段可交易
    "night_day": "交易日 夜→日",  # 一个 trading_date 的夜+日合成一个时段
    "day_night": "自然日 日→夜",  # 白盘段 + 紧随其后的夜盘段合成一个时段
    "split": "日夜分开",          # 夜、日各算一个时段 (默认, 也是历史行为)
}
DEFAULT_SESSION_MODE = "split"


def derive_segments(
    df: pd.DataFrame, split_gap: pd.Timedelta = SESSION_SPLIT_GAP,
) -> np.ndarray:
    """原子段编号 (0 基, 沿时间单调递增)。"""
    changed = df["trading_date"].ne(df["trading_date"].shift())
    big_gap = df.index.to_series().diff() > split_gap
    return ((changed | big_gap).cumsum() - 1).to_numpy()


def segment_is_night(df: pd.DataFrame, seg: np.ndarray) -> np.ndarray:
    """
    逐根布尔: 该 bar 是否落在**夜盘段**里。

    规则: **每个 `trading_date` 的最后一个原子段是白盘段, 之前的都是夜盘段。**
    `trading_date` 总是以白盘收尾 (15:00 / 15:15), 所以这条不读任何时钟点,
    也自动覆盖了无夜盘品种、中金所、以及 2020 年 2-5 月全市场停夜盘 —— 那些
    trading_date 只有 1 段, 于是被判为白盘。实测每个 trading_date 最多 2 段。
    """
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    last_seg = pd.Series(seg).groupby(df["trading_date"].to_numpy()).transform("max")
    return (seg < last_seg.to_numpy())


def assert_segment_shape(
    df: pd.DataFrame, seg: np.ndarray, is_night: np.ndarray, symbol: str = "?",
) -> None:
    """
    守卫 `segment_is_night` 的「末段即白盘」规则: 夜盘段的首根必须落在
    [20:00, 04:00), 白盘段的首根必须落在 [08:00, 14:00)。

    时钟只用来**校验**, 不用来分类 —— 真正会漏的情形是某个 trading_date 的
    白盘段被 `drop_dead_runs` 整段清掉、只剩夜盘段, 那时「末段」会被误判成
    白盘。这条断言正好抓它。与 `assert_no_ambiguous_gaps` 同一个套路: 宁可炸
    也不要静默错分。
    """
    if len(df) == 0:
        return
    first_pos = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    t = df.index[first_pos]
    hour = t.hour + t.minute / 60.0
    night = is_night[first_pos]
    bad_n = night & ~((hour >= 20) | (hour < 4))
    bad_d = (~night) & ~((hour >= 8) & (hour < 14))
    bad = bad_n | bad_d
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        kind = "夜盘" if night[i] else "白盘"
        raise ValueError(
            f"{symbol}: {int(bad.sum())} 个原子段的首根时刻与「末段即白盘」规则"
            f"不符。首例 {t[i]} 被判为{kind}段 —— 该 trading_date 的段结构与实测"
            f"形状不同 (最可能是白盘段被整段清洗掉了)。"
        )


def derive_sessions(
    df: pd.DataFrame, split_gap: pd.Timedelta = SESSION_SPLIT_GAP,
) -> pd.DataFrame:
    """
    加三列: `session_id` (全局单调递增的整数)、`is_session_first`、`is_session_last`。

    这是 `apply_session_mode(df, "split")` 的等价写法, 保留是为了让既有调用点
    和测试不用改。

    末根的 `is_session_last` 由 `shift(-1)` 产生 NaN 再 `ne` 得到 True —— 这正是
    想要的: 窗口的最后一根强制平掉任何未平仓位。

    注意这三列都是**窗口形状的属性**, 在循环开始前就已知, 不是任何价格的函数。
    回测里用 `is_session_last[i]` 触发收盘平仓**不是未来函数**: 交易所的收盘
    时间是提前公告的, 实盘同样提前知道。
    """
    out = df.copy()
    same_td = df["trading_date"].ne(df["trading_date"].shift())
    big_gap = df.index.to_series().diff() > split_gap
    sid = (same_td | big_gap).cumsum()
    out["session_id"] = sid.to_numpy()
    out["is_session_first"] = sid.ne(sid.shift()).to_numpy()
    out["is_session_last"] = sid.ne(sid.shift(-1)).to_numpy()
    return out


def session_keys(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    按模式给每根 bar 算出 (分组键, 是否可交易)。

    分组键沿时间**非降**, 所以同一个键的 bar 必然连续 —— 时段因此永远是一段
    连续的 bar, 这是回测循环的前提。

    `day_night` 的键: 白盘段用自己的 trading_date 序号, 夜盘段用**前一个**
    trading_date 的序号。夜盘 T+1 紧接在白盘 T 之后 (白盘 T 收于 15:00, 夜盘
    T+1 开于当晚 21:00), 而 pos(T+1)-1 == pos(T), 所以两者拿到同一个键。
    周五同理: 周五白盘 + 周五晚夜盘 (其 trading_date 是下周一)。
    """
    if mode not in SESSION_MODES:
        raise ValueError(f"未知的时段模式 {mode!r}; 可选: {list(SESSION_MODES)}")
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    seg = derive_segments(df)
    is_night = segment_is_night(df, seg)

    if mode == "split":
        return seg, np.ones(n, dtype=bool)
    if mode == "night":
        return seg, is_night
    if mode == "day":
        return seg, ~is_night

    tds = pd.DatetimeIndex(sorted(pd.unique(df["trading_date"].to_numpy())))
    pos = tds.get_indexer(df["trading_date"]).astype(np.int64)
    if mode == "night_day":
        return pos, np.ones(n, dtype=bool)
    # day_night: 夜盘段挂到前一个 trading_date 上。窗口首个夜盘段的前一天不在
    # 帧内时键为 -1, 自成一个截断时段 —— 照常可交易、照常在其末根强平。
    return np.where(is_night, pos - 1, pos), np.ones(n, dtype=bool)


def apply_session_mode(df: pd.DataFrame, mode: str = DEFAULT_SESSION_MODE) -> pd.DataFrame:
    """加四列: `session_id` / `is_session_first` / `is_session_last` / `tradable`。"""
    out = df.copy()
    n = len(df)
    if n == 0:
        for c, dt in (("session_id", np.int64), ("is_session_first", bool),
                      ("is_session_last", bool), ("tradable", bool)):
            out[c] = np.zeros(0, dtype=dt)
        return out

    key, tradable = session_keys(df, mode)
    if (np.diff(key) < 0).any():
        raise ValueError(f"模式 {mode!r} 的分组键不是非降的 —— 时段会不连续")

    new_sess = np.r_[True, key[1:] != key[:-1]]
    sid = np.cumsum(new_sess)
    out["session_id"] = sid
    out["is_session_first"] = new_sess
    out["is_session_last"] = np.r_[sid[1:] != sid[:-1], True]
    out["tradable"] = tradable
    return out


def assert_no_ambiguous_gaps(
    df: pd.DataFrame, symbol: str = "?",
    lo: pd.Timedelta = AMBIGUOUS_LO, hi: pd.Timedelta = AMBIGUOUS_HI,
) -> None:
    """
    守卫 `SESSION_SPLIT_GAP` 那个常数: trading_date **内部**不得出现落在
    (lo, hi) 的间隔。跨 trading_date 的间隔不检查 —— 它们由 trading_date
    变化本身切开, 多大都无所谓。

    命中说明这个品种/时代的时段形状和实测的不一样, 3h 阈值可能切错。
    宁可炸也不要静默错切。
    """
    if len(df) < 2:
        return
    d = df.index.to_series().diff()
    same_td = df["trading_date"].eq(df["trading_date"].shift())
    bad = df.index[(d > lo) & (d < hi) & same_td]
    if len(bad):
        raise ValueError(
            f"{symbol}: {len(bad)} 个 trading_date 内部间隔落在模糊带 "
            f"({lo} , {hi}) —— SESSION_SPLIT_GAP={SESSION_SPLIT_GAP} 对这个品种"
            f"可能切错时段。首例: {bad[0]}"
        )


# ------------------------------------------------------------ 前日 OHLC ----
def daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 `trading_date` 聚合出 H/L/C/bars。

    聚合的是**完整交易日 = 夜盘 + 日盘** —— 夜盘 bar 本来就带着次日的
    trading_date, 所以这一步不需要任何特殊处理。`C` 取 `last()`, 因为日盘
    总是一个 trading_date 的收尾 (15:00 或 15:15), 所以它就是当日收盘。
    """
    return df.groupby("trading_date").agg(
        H=("high", "max"), L=("low", "min"), C=("close", "last"), bars=("close", "size"),
    ).sort_index()


def daily_ohlc_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    `daily_ohlc` 的带 **O** 版本, 列名是标准的 open/high/low/close/volume。

    另开一个函数而不是给 `daily_ohlc` 加列: 那个函数的列名 (H/L/C) 已经被
    `prev_day_lines` 和 R-Breaker 的全套测试写死了, 动它没有收益只有风险。

    回调策略需要 open, 因为冷静期的第二个触发条件 (`check_structure_break`) 读的是
    日线**实体** min/max(open, close) —— 见 src/rules/cooldown.py:74-85。
    """
    return df.groupby("trading_date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).sort_index()


def prev_day_lines(
    agg: pd.DataFrame, params: RBreakerParams, config: RBreakerConfig,
    tradable_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    把每个 trading_date 的六条线算出来 —— 用的是**上一个出现在帧里的**
    trading_date 的 H/L/C。

    `valid=False` 的日子六条线为 NaN, 回测的入场分支会被挡住:
      * 窗口首日没有参照 (`prev` 全 NaN) —— 无需特例。
      * 参照日 bar 数不足 `min_ref_bars` (被清洗掉大半, 或窗口边缘的碎片)。
      * 与上一个可用交易日的日历间隔超过 `max_reference_gap_days`。最长的
        合法休市是春节, 约 10 个日历日 (2024-02-09 -> 2024-02-19), 默认 15
        留了余量又能挡住真正的数据洞。
      * `tradable_from` 之前的日子 —— 那是为了给首个交易日提供参照而加载的
        padding, 不该产生交易。
    """
    prev = agg.shift(1)
    days = agg.index.to_series()
    gap_days = (days - days.shift(1)).dt.days

    valid = (
        prev["bars"].ge(config.min_ref_bars).fillna(False)
        & gap_days.le(config.max_reference_gap_days).fillna(False)
        & prev[["H", "L", "C"]].notna().all(axis=1)
    )
    if tradable_from is not None:
        valid &= agg.index >= tradable_from

    lines = rbreaker_lines(prev["H"], prev["L"], prev["C"], params)
    lines.loc[~valid.to_numpy(), :] = np.nan
    lines["ref_H"] = prev["H"].where(valid)
    lines["ref_L"] = prev["L"].where(valid)
    lines["ref_C"] = prev["C"].where(valid)
    lines["bars"] = agg["bars"]
    lines["valid"] = valid.to_numpy()
    return lines


# -------------------------------------------------------------- prepare ----
@dataclass
class PreparedBase:
    """
    与时段模式**无关**的那一半: 读盘 + 清洗 + 复权 + 日聚合 + 六条线。

    这一半贵 (读 parquet), 但五种模式共用 —— 决策 1: 线永远来自前一个完整
    `trading_date`, 模式只改变「哪些 bar 允许入场 / 何时强平」。所以切模式时
    只需要重跑 `apply_session_mode` 这个便宜的一半。
    """

    symbol: str
    df: pd.DataFrame          # index=datetime; 复权 OHLC + volume + trading_date
    lines: pd.DataFrame       # index=trading_date; 六线 + ref_H/L/C + bars + valid
    k0: float
    params: RBreakerParams
    config: RBreakerConfig
    dropped_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    # 帧首有多少根属于**暖机段** —— 只为把指标(日线 MA50 之类)算热, 不显示也不交易。
    # `warmup_days=0` (R-Breaker 路径) 时恒为 0, 帧与历史行为逐字节相同。
    warmup_bars: int = 0
    # 暖机段**实际**覆盖了多少个 trading_date。可能小于请求值 (品种上市晚、
    # 或 USABLE_FROM 裁过) —— 小于时统计面板要如实标注"暖机不足"。
    warmup_days: int = 0


@dataclass
class Prepared:
    """回测和图表共用的一份准备好的数据 (已应用某一种时段模式)。"""

    symbol: str
    df: pd.DataFrame          # index=datetime; 复权 OHLC + volume + trading_date + 四个时段列
    lines: pd.DataFrame       # index=trading_date; 六线 + ref_H/L/C + bars + valid
    # 每根 bar -> lines 的行号, **按时段钉住**: 取该 bar 所属时段**第一根**的
    # trading_date。对 split/night/day/night_day 四种模式, 时段必然落在一个
    # trading_date 内, 所以它就等于逐根的 trading_date 下标 —— 完全等价。
    # 只有 day_night 会跨 trading_date, 那时整个时段 (含夜盘部分) 统一用白盘
    # 那天的线。这是唯一无未来函数的选择: 钉到时段最后一根的 trading_date 需要
    # 在 09:00 就知道当天收盘后才能算出的线。
    line_idx: np.ndarray
    k0: float                 # 复权常数 = 显示窗口首根的 K
    params: RBreakerParams
    config: RBreakerConfig
    mode: str = DEFAULT_SESSION_MODE
    dropped_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    warmup_bars: int = 0      # 见 PreparedBase.warmup_bars
    warmup_days: int = 0      # 见 PreparedBase.warmup_days

    @property
    def mode_label(self) -> str:
        return SESSION_MODES.get(self.mode, self.mode)

    @property
    def in_window(self) -> np.ndarray:
        """逐根 bool: 该 bar 属于**显示/交易窗口**(不是暖机段)。"""
        return np.arange(len(self.df)) >= self.warmup_bars

    @property
    def tradable_days(self) -> int:
        """
        落在「可交易且 valid」时段内的 bar 覆盖了多少个不同的 `trading_date`。

        对 day/night_day/day_night/split 等于旧口径 (窗口内的合法交易日数);
        对 `night` 模式遇上无夜盘品种为 **0** —— 如实反映「这个品种在这个模式
        下无法交易」, 而不是假装有 258 天可交易却一笔不做。
        """
        if self.df.empty or self.lines.empty:
            return 0
        ok = self.df["tradable"].to_numpy() & self.lines["valid"].to_numpy()[self.line_idx]
        if not ok.any():
            return 0
        return int(pd.unique(self.df["trading_date"].to_numpy()[ok]).size)

    def ohlcv(self) -> pd.DataFrame:
        """lightweight-charts `set()` 的口径: time/open/high/low/close/volume。"""
        out = self.df[["open", "high", "low", "close", "volume"]].copy()
        out.insert(0, "time", self.df.index)
        return out.reset_index(drop=True)


def prepare_base(
    symbol: str, start: str | None = None, end: str | None = None,
    params: RBreakerParams = RBreakerParams(),
    config: RBreakerConfig = RBreakerConfig(),
    clean_dir: str = CLEAN_DIR,
    warmup_days: int = 0,
) -> PreparedBase:
    """
    跑到六条线为止 —— 与时段模式无关的那一半。切模式时不需要重跑这一段。

    **窗口 padding**: 实际加载 `[start - (max_reference_gap_days + 5) 天, end]`,
    好让 `start` 当天就有前日参照。但 `prev_day_lines` 的 `tradable_from`
    保证 padding 段永远 `valid=False`, 而输出帧又按 trading_date 边界切回
    `tradable_from` 起 —— 所以回测区间、统计区间、显示区间三者严格对齐,
    padding 对用户完全不可见。

    **`warmup_days` (默认 0 = 现状, 逐字节不变)**: >0 时输出帧**保留** `start`
    之前 `warmup_days` 个交易日的 bar, 并把根数记在 `PreparedBase.warmup_bars`
    上。给回调策略用 —— 它的趋势方向要日线 MA50, 也就是显示窗口开始前必须先有
    50 根日线; 上面那个 20 天的 padding 差得远。

    暖机段与上面的 padding 是**两件事**: padding 是为了六条线的前日参照, 用完就
    被切掉、用户看不见; 暖机段是**留在帧里**的, 因为指标要在它上面滚动。两者都
    `valid=False`, 所以两者都不产生交易。

    R-Breaker 路径不传这个参数 -> `warmup_bars=0` -> 帧、lines、k0 与历史行为
    逐字节相同 (tests/test_pullback_v3_warmup.py::test_warmup_zero_is_byte_identical)。

    **按 trading_date 边界切而不是按 `start` 时间戳切**: 否则一个交易日会被
    从中间劈开, 夜盘留在窗口外而日盘留在窗口内, 当天的日聚合就废了。副作用是
    有夜盘的品种, 显示窗口的第一根会是 `start` 前一晚的 21:00 —— 这是对的,
    那正是这个交易日的起点。

    **永远复权。** 不给 `adjust=False` 的开关: K 在 RB 全历史变了 50 次, 用
    裸价的话换月日的前日 H/L/C 和当日 bar 处在不同价格尺度上, 每个换月日都会
    制造一次必然突破。要看裸价请用 `src/run_v3_minute_chart.py` 那个纯浏览器。
    """
    # 暖机要的是**交易日**, 加载窗口给的是**日历日**。243/365 ~ 0.666, 所以
    # 1.6 倍再加 30 天足够覆盖周末 + 春节那个约 10 个日历日的最长休市。多加载的
    # 部分反正会被下面按 trading_date 精确裁掉, 宁可多不可少。
    pad_days = max(config.max_reference_gap_days + 5, int(warmup_days * 1.6) + 30)
    pad = pd.Timedelta(days=pad_days)
    load_start = None if start is None else (pd.Timestamp(start) - pad).strftime("%Y-%m-%d")

    raw = load_minute_raw(symbol, load_start, end, clean_dir)
    if raw.empty:
        return _empty_base(symbol, params, config)

    clean, dropped = drop_dead_runs(raw)
    if clean.empty:
        return _empty_base(symbol, params, config, dropped)
    assert_no_ambiguous_gaps(clean, symbol)

    # 显示/回测窗口的起点: 第一个 >= start 的 trading_date。
    tds = clean["trading_date"]
    tradable_from = None
    if start is not None:
        after = tds[tds >= pd.Timestamp(start)]
        if after.empty:
            return _empty_base(symbol, params, config, dropped)
        tradable_from = after.iloc[0]

    # k0 取**显示窗口**首根的 K (不是 padding 首根, 也不是暖机段首根), 这样窗口
    # 第一根恰好落在真实价上。padding / 暖机段用同一个 k0 复权, 所以参照日的
    # H/L/C、暖机段的指标、显示段三者同尺度。
    disp_mask = np.ones(len(clean), dtype=bool) if tradable_from is None else (tds >= tradable_from).to_numpy()
    k0 = float(clean["K"].to_numpy()[disp_mask][0])

    # 暖机段: 从 tradable_from 往前数 warmup_days 个**交易日**。切点必然落在
    # trading_date 边界上, 所以任何一根合成 K 线都不会被劈开 (v3_timeframes 依赖
    # 这一条)。品种上市晚 / USABLE_FROM 裁过时可能不够, 如实记在 warmup_days 上。
    keep_mask, warm_bars, warm_days = disp_mask, 0, 0
    if warmup_days > 0 and tradable_from is not None:
        uniq = pd.DatetimeIndex(sorted(pd.unique(tds.to_numpy())))
        j = int(uniq.get_indexer([tradable_from])[0])
        warm_from = uniq[max(0, j - warmup_days)]
        warm_days = j - max(0, j - warmup_days)
        keep_mask = (tds >= warm_from).to_numpy()
        warm_bars = int(keep_mask.sum() - disp_mask.sum())

    adj = adjust_prices(clean, k0)
    agg = daily_ohlc(adj)
    lines = prev_day_lines(agg, params, config, tradable_from)

    disp = adj[keep_mask]
    lines = lines[lines.index >= disp["trading_date"].iloc[0]] if len(disp) else lines.iloc[:0]
    keep = ["open", "high", "low", "close", "volume", "trading_date"]
    return PreparedBase(
        symbol=symbol, df=disp[keep], lines=lines, k0=k0,
        params=params, config=config, dropped_runs=dropped,
        warmup_bars=warm_bars, warmup_days=warm_days,
    )


def apply_mode(base: PreparedBase, mode: str = DEFAULT_SESSION_MODE) -> Prepared:
    """
    把某一种时段模式套到 `PreparedBase` 上, 得到可以直接喂给引擎的 `Prepared`。
    便宜 (纯 numpy/pandas 运算, 不碰磁盘), 切模式走这条路。
    """
    if base.df.empty:
        return _empty_prepared(base, mode)

    sess = apply_session_mode(base.df, mode)

    raw_idx = base.lines.index.get_indexer(sess["trading_date"]).astype(np.int32)
    if (raw_idx < 0).any():
        raise ValueError(
            f"{base.symbol}: {int((raw_idx < 0).sum())} 根 bar 的 trading_date 不在日聚合里")

    # 线的下标**按时段钉住**: 取该时段第一根的 trading_date 下标, 广播到整段。
    # 对四种不跨 trading_date 的模式这是恒等变换; 只有 day_night 会真正用到。
    sid = sess["session_id"].to_numpy()
    first_pos = np.flatnonzero(sess["is_session_first"].to_numpy())
    line_idx = raw_idx[first_pos][sid - 1]

    keep = ["open", "high", "low", "close", "volume", "trading_date",
            "session_id", "is_session_first", "is_session_last", "tradable"]
    return Prepared(
        symbol=base.symbol, df=sess[keep], lines=base.lines, line_idx=line_idx,
        k0=base.k0, params=base.params, config=base.config, mode=mode,
        dropped_runs=base.dropped_runs,
        warmup_bars=base.warmup_bars, warmup_days=base.warmup_days,
    )


def prepare(
    symbol: str, start: str | None = None, end: str | None = None,
    params: RBreakerParams = RBreakerParams(),
    config: RBreakerConfig = RBreakerConfig(),
    clean_dir: str = CLEAN_DIR,
    mode: str = DEFAULT_SESSION_MODE,
    warmup_days: int = 0,
) -> Prepared:
    """`prepare_base` + `apply_mode` 的便捷组合。一次性用法走这个。"""
    return apply_mode(
        prepare_base(symbol, start, end, params, config, clean_dir, warmup_days), mode)


_EMPTY_BAR_COLS = ["open", "high", "low", "close", "volume", "trading_date"]
_EMPTY_LINE_COLS = [*LINE_NAMES, "ref_H", "ref_L", "ref_C", "bars", "valid"]


def _empty_base(symbol, params, config, dropped=None) -> PreparedBase:
    return PreparedBase(
        symbol=symbol,
        df=pd.DataFrame(columns=_EMPTY_BAR_COLS, index=pd.DatetimeIndex([], name="datetime")),
        lines=pd.DataFrame(columns=_EMPTY_LINE_COLS),
        k0=1.0, params=params, config=config,
        dropped_runs=dropped if dropped is not None else pd.DataFrame(),
    )


def _empty_prepared(base: PreparedBase, mode: str) -> Prepared:
    cols = [*_EMPTY_BAR_COLS, "session_id", "is_session_first", "is_session_last", "tradable"]
    return Prepared(
        symbol=base.symbol,
        df=pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="datetime")),
        lines=base.lines, line_idx=np.zeros(0, dtype=np.int32), k0=base.k0,
        params=base.params, config=base.config, mode=mode, dropped_runs=base.dropped_runs,
    )
