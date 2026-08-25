# -*- coding: utf-8 -*-
"""
`src/data/v3_timeframes.py`: 1m -> 5m / 15m / 30m / 2h / 1d 的分箱与「哪根已收盘」。

用手搭的 bar 而不是读 parquet —— 失败时指向逻辑而不是数据。帧的形状严格复刻真实
schema: `datetime` 索引 (tz-naive 北京墙钟, 打标在分钟**末端**)、`trading_date`
是权威交易日键、夜盘 bar 带着**下一个**交易日的 trading_date。

这里钉死的三条不变量, 任何一条破了整个回调引擎就会读到未来数据或者错周期:
  1. 合成 bar 永不跨原子段 (夜->日那 6.5 小时的断口)
  2. 标签 = 末根成分的标签
  3. closed_pos 严格因果, 断口处不提前收
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import derive_segments
from src.data.v3_timeframes import (
    PB_RULES, PB_TFS, build_timeframe, build_timeframes,
)

# 真实的商品日盘形状: 09:01-10:15 (75) / 10:31-11:30 (60) / 13:31-15:00 (90) = 225 根
_DAY_RUNS = (("09:01", 75), ("10:31", 60), ("13:31", 90))
_NIGHT_RUNS = (("21:01", 120),)      # 21:01-23:00, 一档夜盘


def _bars(day: str, runs, trading_date: str, price0: float):
    """连续 1 分钟 bar; close 逐根 +1 好让聚合结果一眼可验。"""
    rows, px = [], price0
    for hhmm, n in runs:
        t0 = pd.Timestamp(f"{day} {hhmm}")
        for j in range(n):
            t = t0 + pd.Timedelta(minutes=j)
            rows.append((t, pd.Timestamp(trading_date), px, px + 0.5, px - 0.5, px, 1.0))
            px += 1.0
    return rows


def _frame(rows) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r[0] for r in rows], name="datetime")
    return pd.DataFrame(
        {
            "trading_date": [r[1] for r in rows],
            "open": [r[2] for r in rows], "high": [r[3] for r in rows],
            "low": [r[4] for r in rows], "close": [r[5] for r in rows],
            "volume": [r[6] for r in rows],
        },
        index=idx,
    )


@pytest.fixture
def two_days() -> pd.DataFrame:
    """
    两个完整交易日, 每个 = 夜盘段 + 日盘段。
      trading_date 2026-03-03: 夜盘 03-02 21:01-23:00, 日盘 03-03 09:01-15:00
      trading_date 2026-03-04: 夜盘 03-03 21:01-23:00, 日盘 03-04 09:01-15:00
    """
    rows = []
    rows += _bars("2026-03-02", _NIGHT_RUNS, "2026-03-03", 1000.0)
    rows += _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 2000.0)
    rows += _bars("2026-03-03", _NIGHT_RUNS, "2026-03-04", 3000.0)
    rows += _bars("2026-03-04", _DAY_RUNS, "2026-03-04", 4000.0)
    return _frame(rows)


# ----------------------------------------------------------- 分箱形状 ----
def test_segments_split_night_from_day(two_days):
    """前提自检: 每个 trading_date 恰好两个原子段。"""
    seg = derive_segments(two_days)
    assert seg[0] == 0
    assert len(np.unique(seg)) == 4
    # 日盘内部的 10:15->10:31 (16min) 和 11:30->13:31 (2h01) 都不许切段
    day1 = two_days.loc["2026-03-03 09:01":"2026-03-03 15:00"]
    assert len(np.unique(derive_segments(two_days)[120:120 + len(day1)])) == 1


def test_30m_day_session_bin_sizes_and_labels(two_days):
    """
    决策 1 的落地: 段内按时钟切, 跨 10:15 小节休息的那一根只有 15 根 1m。
    日盘 225 根 -> 8 根 30m, 尺寸 [30,30,15,30,30,30,30,30]。
    """
    t = build_timeframe(two_days, "30m")
    day1 = t.bars[(t.bars.index >= "2026-03-03 09:00") & (t.bars.index <= "2026-03-03 15:00")]
    assert [f"{x:%H:%M}" for x in day1.index] == [
        "09:30", "10:00", "10:15", "11:00", "11:30", "14:00", "14:30", "15:00"]
    sizes = np.bincount(t.bin_of)
    first_bin = int(t.bin_of[120])          # 日盘第一根 1m 的 bin
    assert list(sizes[first_bin:first_bin + 8]) == [30, 30, 15, 30, 30, 30, 30, 30]


def test_5m_never_gets_truncated_by_the_intraday_breaks(two_days):
    """日盘三段 75/60/90 根都能被 5 整除 -> 5m 全是满 5 根, 只有 30m 会被截短。"""
    t = build_timeframe(two_days, "5m")
    sizes = np.bincount(t.bin_of)
    assert set(sizes.tolist()) == {5}
    assert t.n_bars == (120 + 225) * 2 // 5


def test_bins_never_span_an_atomic_segment(two_days):
    """夜盘末根 23:00 与日盘首根 09:01 之间隔 10 小时, 绝不能被并进同一根。"""
    seg = derive_segments(two_days)
    for tf in ("5m", "30m"):
        t = build_timeframe(two_days, tf, seg)
        for b in range(t.n_bars):
            members = np.flatnonzero(t.bin_of == b)
            assert len(np.unique(seg[members])) == 1, f"{tf} bin {b} 跨段"


def test_bin_label_is_the_last_member(two_days):
    for tf in PB_TFS:
        t = build_timeframe(two_days, tf)
        for b in (0, t.n_bars // 2, t.n_bars - 1):
            members = np.flatnonzero(t.bin_of == b)
            assert t.bars.index[b] == two_days.index[members[-1]], tf


def test_ohlcv_aggregation(two_days):
    t = build_timeframe(two_days, "30m")
    for b in (0, 3, t.n_bars - 1):
        m = np.flatnonzero(t.bin_of == b)
        src = two_days.iloc[m]
        row = t.bars.iloc[b]
        assert row["open"] == src["open"].iloc[0]
        assert row["close"] == src["close"].iloc[-1]
        assert row["high"] == src["high"].max()
        assert row["low"] == src["low"].min()
        assert row["volume"] == src["volume"].sum()


# -------------------------------------------------- 集合竞价段首孤儿箱 ----
@pytest.fixture
def with_auction() -> pd.DataFrame:
    """
    每个时段前面加一根**集合竞价** bar (零振幅、带真实成交量), 复刻数据里的形状:
        RB 2025-07-01 21:00  o=h=l=c=3003.0  vol=4456
        UR 2025-07-01 09:00  o=1707 h=1707 l=1706 c=1706 vol=1277
    """
    rows = []
    rows.append((pd.Timestamp("2026-03-02 21:00"), pd.Timestamp("2026-03-03"),
                 999.0, 999.0, 999.0, 999.0, 4456.0))
    rows += _bars("2026-03-02", _NIGHT_RUNS, "2026-03-03", 1000.0)
    rows.append((pd.Timestamp("2026-03-03 09:00"), pd.Timestamp("2026-03-03"),
                 1999.0, 1999.0, 1999.0, 1999.0, 1277.0))
    rows += _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 2000.0)
    return _frame(rows)


def test_auction_bar_is_merged_into_the_next_bin(with_auction):
    """
    竞价 bar 的 start 是 20:59 / 08:59, 落进上一格时钟箱, 会独占一个箱。默认把它
    并进下一箱 —— 于是日盘仍是 8 根 30m, 与决策 1 的预览一致。
    """
    t = build_timeframe(with_auction, "30m")
    day = t.bars[t.bars.index >= "2026-03-03 09:00"]
    assert [f"{x:%H:%M}" for x in day.index] == [
        "09:30", "10:00", "10:15", "11:00", "11:30", "14:00", "14:30", "15:00"]
    assert (np.bincount(t.bin_of) >= 5).all(), "不该再有孤儿箱"
    # 并进去之后, 该 30m bar 的 open 变成**竞价撮合价** —— 这正是交易所的开盘价定义
    assert day["open"].iloc[0] == 1999.0
    first = np.flatnonzero(t.bin_of == int(t.bin_of[121]))
    assert len(first) == 31, "竞价那一根 + 30 根连续 bar"


def test_auction_merge_also_applies_to_5m(with_auction):
    t = build_timeframe(with_auction, "5m")
    sizes = np.bincount(t.bin_of)
    assert set(sizes.tolist()) == {5, 6}, "只有每段首箱是 6 根 (竞价 + 5)"
    assert (sizes == 6).sum() == 2, "两个时段各一根竞价 bar"


def test_auction_merge_can_be_switched_off(with_auction):
    """关掉 -> 退回逐字面的 resample, 孤儿箱回来。"""
    t = build_timeframe(with_auction, "30m", merge_leading_singleton=False)
    sizes = np.bincount(t.bin_of)
    assert (sizes == 1).sum() == 2
    day = t.bars[t.bars.index >= "2026-03-03 09:00"]
    assert [f"{x:%H:%M}" for x in day.index][:2] == ["09:00", "09:30"]


def test_merge_does_not_fire_when_the_segment_head_is_not_alone(two_days):
    """没有竞价 bar 的品种(RB 日盘从 09:01 起)不该被这条规则动到。"""
    a = build_timeframe(two_days, "30m", merge_leading_singleton=True)
    b = build_timeframe(two_days, "30m", merge_leading_singleton=False)
    assert list(a.bin_of) == list(b.bin_of)


def test_merge_never_crosses_a_segment(with_auction):
    """并箱只在段内进行 —— 夜盘的竞价 bar 绝不会被并进白盘。"""
    seg = derive_segments(with_auction)
    for tf in ("5m", "30m"):
        t = build_timeframe(with_auction, tf, seg)
        for b in range(t.n_bars):
            members = np.flatnonzero(t.bin_of == b)
            assert len(np.unique(seg[members])) == 1, f"{tf} bin {b} 跨段"


def test_merge_leaves_a_one_bar_segment_alone():
    """只有一根的时段没有"下一箱"可并 —— 不能越界并到下一个时段去。"""
    rows = [(pd.Timestamp("2026-03-02 21:00"), pd.Timestamp("2026-03-03"),
             999.0, 999.0, 999.0, 999.0, 10.0)]
    rows += _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 2000.0)
    df = _frame(rows)
    seg = derive_segments(df)
    assert len(np.unique(seg)) == 2
    t = build_timeframe(df, "30m", seg)
    assert np.bincount(t.bin_of)[0] == 1
    assert t.bars.index[0] == pd.Timestamp("2026-03-02 21:00")


# -------------------------------------------------------------- 日线 ----
def test_daily_bar_is_night_plus_day_of_one_trading_date(two_days):
    t = build_timeframe(two_days, "1d")
    assert t.n_bars == 2
    assert list(t.bars["trading_date"]) == [pd.Timestamp("2026-03-03"), pd.Timestamp("2026-03-04")]
    # 标签 = 该交易日最后一根 1m, 也就是日盘 15:00
    assert [f"{x:%Y-%m-%d %H:%M}" for x in t.bars.index] == [
        "2026-03-03 15:00", "2026-03-04 15:00"]
    # 第一根日线的 open 来自**夜盘**首根 (1000.0), 不是日盘首根
    assert t.bars["open"].iloc[0] == 1000.0
    assert t.bars["high"].iloc[0] == two_days.iloc[:345]["high"].max()


def test_daily_bar_is_allowed_to_span_two_segments(two_days):
    """日线是唯一允许跨原子段的周期 —— 夜盘段 + 白盘段本来就是一个交易日。"""
    seg = derive_segments(two_days)
    t = build_timeframe(two_days, "1d", seg)
    members = np.flatnonzero(t.bin_of == 0)
    assert len(np.unique(seg[members])) == 2


# ------------------------------------------------------- closed_pos ----
def test_1m_closed_pos_is_exactly_i_minus_1(two_days):
    """v2.1 的 `small_5m = df_5m.iloc[:i]` 逐字对应物。"""
    t = build_timeframe(two_days, "1m")
    assert list(t.closed_pos) == list(range(-1, len(two_days) - 1))


def test_30m_closed_pos_does_not_close_a_bar_early_at_the_break(two_days):
    t = build_timeframe(two_days, "30m")
    pos_of = {ts: i for i, ts in enumerate(two_days.index)}
    lbl = list(t.bars.index)

    # 10:31 (start 10:30): 最后一根已收的是标签 10:15 那根, 不是还没开始的 10:30
    i = pos_of[pd.Timestamp("2026-03-03 10:31")]
    assert lbl[t.closed_pos[i]] == pd.Timestamp("2026-03-03 10:15")
    # 10:15 那根自己 (start 10:14): 它还没收, 最后一根已收的是 10:00
    i = pos_of[pd.Timestamp("2026-03-03 10:15")]
    assert lbl[t.closed_pos[i]] == pd.Timestamp("2026-03-03 10:00")
    # 09:31 (start 09:30): 09:30 那根刚刚收完
    i = pos_of[pd.Timestamp("2026-03-03 09:31")]
    assert lbl[t.closed_pos[i]] == pd.Timestamp("2026-03-03 09:30")
    # 日盘首根 09:01 (start 09:00): 最后一根已收的是**夜盘**末根
    i = pos_of[pd.Timestamp("2026-03-03 09:01")]
    assert lbl[t.closed_pos[i]] == pd.Timestamp("2026-03-02 23:00")


def test_daily_closed_pos_points_at_the_previous_trading_date(two_days):
    t = build_timeframe(two_days, "1d")
    td = two_days["trading_date"].to_numpy()
    for i in range(len(two_days)):
        cp = int(t.closed_pos[i])
        if td[i] == np.datetime64("2026-03-03"):
            assert cp == -1, "窗口首个交易日没有上一日, 必须是 -1"
        else:
            assert t.bars["trading_date"].iloc[cp] == pd.Timestamp("2026-03-03")


def test_closed_pos_is_strictly_causal_for_every_timeframe(two_days):
    """通用不变量: 被指到的那根 bar 的末端标签必须 <= 本根的开盘时刻。"""
    starts = two_days.index.to_numpy() - np.timedelta64(1, "m")
    for tf in PB_TFS:
        t = build_timeframe(two_days, tf)
        ok = t.closed_pos >= 0
        assert (t.end_labels[t.closed_pos[ok]] <= starts[ok]).all(), tf
        # 而且是**最后**一根: 下一根(若存在)必须还没收
        nxt = t.closed_pos + 1
        live = nxt < t.n_bars
        assert (t.end_labels[nxt[live]] > starts[live]).all(), tf
        assert (np.diff(t.closed_pos) >= 0).all(), tf


# --------------------------------------------------------- 边界情形 ----
def test_empty_frame():
    empty = _frame([]).astype({"open": float, "high": float, "low": float,
                               "close": float, "volume": float})
    tfs = build_timeframes(empty)
    for tf in PB_TFS:
        assert tfs[tf].n_bars == 0
        assert tfs[tf].bin_of.shape == (0,)
        assert tfs[tf].closed_pos.shape == (0,)


def test_single_segment_symbol_without_night_session():
    """无夜盘品种 / 中金所: 一个 trading_date 只有一段, 照常工作。"""
    rows = _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 100.0)
    rows += _bars("2026-03-04", _DAY_RUNS, "2026-03-04", 200.0)
    df = _frame(rows)
    tfs = build_timeframes(df)
    assert tfs["1d"].n_bars == 2
    assert tfs["30m"].n_bars == 16
    assert tfs["1d"].bars["open"].iloc[0] == 100.0


def test_rejects_non_monotonic_index(two_days):
    bad = two_days.iloc[::-1]
    with pytest.raises(ValueError, match="非单调递增"):
        build_timeframes(bad)


def test_rejects_missing_columns(two_days):
    with pytest.raises(ValueError, match="缺列"):
        build_timeframes(two_days.drop(columns=["volume"]))


def test_first_bin_at_or_after_lands_on_a_whole_bar(two_days):
    """暖机切点落在 trading_date 边界上, 所以不会把一根合成 bar 劈开。"""
    warmup = 345                       # 第一个交易日的全部 1m 根数
    assert two_days["trading_date"].iloc[warmup] == pd.Timestamp("2026-03-04")
    for tf in PB_TFS:
        t = build_timeframe(two_days, tf)
        b = t.first_bin_at_or_after(warmup)
        members = np.flatnonzero(t.bin_of == b)
        assert members[0] == warmup, tf


# --------------------------------------------------------------- 15m 与 2h ----
# 这两个周期是后加的, 数据层只加了两条 PB_RULES。下面几条钉死「加两条就够了」这件事,
# 以及 2h 的碎箱是**已知且被承认**的形状, 不是可以静默改掉的实现细节。

_四周期 = ("1m", "5m", "15m", "2h")


def test_new_timeframes_are_registered_but_not_in_pb_tfs(two_days):
    """新周期进 PB_RULES, 但 PB_TFS 一个字不动 —— 下游是按名字硬取的。"""
    assert "15m" in PB_RULES and "2h" in PB_RULES
    assert PB_TFS == ("1m", "5m", "30m", "1d")
    # 两个集合都能构建, 互不影响
    assert set(build_timeframes(two_days, tfs=PB_TFS)) == set(PB_TFS)
    assert set(build_timeframes(two_days, tfs=_四周期)) == set(_四周期)


def test_15m_is_never_truncated_by_the_intraday_breaks(two_days):
    """日盘三段 75/60/90 都被 15 整除, 夜盘 120 也是 —— 每个 15m 箱恰好 15 根。"""
    tfb = build_timeframe(two_days, "15m")
    b = tfb.bin_of
    sizes = np.bincount(b)
    assert set(sizes.tolist()) == {15}
    assert tfb.n_bars == (120 + 225) * 2 // 15


def test_2h_bins_are_uneven_and_that_is_the_known_defect(two_days):
    """
    2h 的网格锚在午夜, 和交易时段形状无关, 所以箱大小不齐。

    这不是 bug, 是 obsidian/01-周期/周期-2h.md 已经点名并选择「承认它并在结果里
    分层」的东西。这条测试的作用是**让它不能被静默改掉**: 哪天分箱口径换成段首
    锚定, 这里会红, 逼人去改那份文档。
    """
    tfb = build_timeframe(two_days, "2h")
    sizes = np.bincount(tfb.bin_of).tolist()
    # 一档夜盘 21:01-23:00 -> [22:00) 60 根 + [23:00) 60 根;
    # 日盘 -> 10:00 的 60、11:30 的 75、14:00 的 30、15:00 的 60
    assert sizes == [60, 60, 60, 75, 30, 60] * 2
    assert min(sizes) * 2 < max(sizes), "极差没了就说明分箱口径变了"


def test_2h_never_spans_an_atomic_segment(two_days):
    """
    箱不齐可以忍, 跨段不行 —— 一根 2h 横跨夜->日那 6.5 小时断口是致命的。
    段号在分箱键里 (`_bin_of` 的 `changed |= new_seg`), 这里做端到端确认。
    """
    seg = derive_segments(two_days)
    for tf in ("15m", "2h"):
        b = build_timeframe(two_days, tf, segments=seg).bin_of
        edge = np.r_[True, b[1:] != b[:-1]]
        first = np.flatnonzero(edge)
        last = np.r_[first[1:] - 1, len(b) - 1]
        assert (seg[first] == seg[last]).all(), tf


def test_new_timeframes_closed_pos_is_strictly_causal(two_days):
    """
    `end_labels[closed_pos] <= start_i < end_labels[closed_pos+1]`, 且非降。
    这是任何策略读高周期的唯一合法通道, 破了就是未来函数。
    """
    starts = two_days.index.to_numpy() - np.timedelta64(1, "m")
    for tf in _四周期:
        tfb = build_timeframe(two_days, tf)
        cp, el = tfb.closed_pos, tfb.end_labels
        seen = cp >= 0
        assert (el[cp[seen]] <= starts[seen]).all(), tf
        nxt = cp + 1
        ok = nxt < len(el)
        assert (el[nxt[ok]] > starts[ok]).all(), tf
        assert (np.diff(cp) >= 0).all(), tf


def test_2h_bar_count_per_trading_day_depends_on_the_night_tier(two_days):
    """
    每交易日的 2h 根数随夜盘档位变 —— 这就是「同一个 MA55 在不同品种上不是同一段
    时间」的根源 (周期-2h.md: 无夜盘 4 / 至23:00 是 6 / 至01:00 是 7 / 至02:30 是 8)。
    夹具是「至 23:00」那一档, 所以应当是 6。
    """
    tfb = build_timeframe(two_days, "2h")
    per_day = tfb.bars.groupby("trading_date").size()
    assert per_day.tolist() == [6, 6]
