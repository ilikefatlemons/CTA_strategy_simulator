# -*- coding: utf-8 -*-
"""
`src/data/v3_timeframes.py`: 1m -> 5m / 15m / 30m / 2h / 1d 的分箱与「哪根已收盘」。

用手搭的 bar 而不是读 parquet —— 失败时指向逻辑而不是数据。帧的形状严格复刻真实
schema: `datetime` 索引 (tz-naive 北京墙钟, 打标在分钟**末端**)、`trading_date`
是权威交易日键、夜盘 bar 带着**下一个**交易日的 trading_date。

这里钉死的三条不变量, 任何一条破了整个回调引擎就会读到未来数据或者错周期:
  1. 合成 bar 永不跨 **trading_date**; 时钟分箱的周期 (5m/15m/30m) 还额外不跨原子段。
     **1d 与 2h 是例外, 允许跨原子段** —— 1d 因为一个交易日 = 夜盘段 + 白盘段;
     2h 因为它按交易分钟等分, 那一桶要横过夜->日的断口才凑得满 120 分钟。
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
# 15m 仍走段内时钟分箱。**2h 在 2026-08-26 改成了「交易日等分 120 个连续交易分钟」**,
# 下面几条钉死新形状: 桶大小、每交易日桶数、允许跨原子段但绝不跨 trading_date。

_四周期 = ("1m", "5m", "15m", "2h")

# AU/AG/SC 那一档夜盘: 21:00 竞价 + 21:01-02:30 共 330 根连续。
_夜盘至0230 = (("21:01", 330),)


@pytest.fixture
def au形状() -> pd.DataFrame:
    """
    实测的 AU 交易日形状 (556 根): 夜盘有 21:00 竞价根, **白盘没有** —— AU 的白盘
    数据就是从 09:01 开始的。这个不对称正是 `_竞价根` 必须逐段判、不能逐交易日判
    的原因。
    """
    rows = [(pd.Timestamp("2026-03-02 21:00"), pd.Timestamp("2026-03-03"),
             999.0, 999.0, 999.0, 999.0, 4456.0)]
    rows += _bars("2026-03-02", _夜盘至0230, "2026-03-03", 1000.0)
    rows += _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 2000.0)
    return _frame(rows)


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


def test_2h_是交易日等分而不是时钟网格(two_days):
    """
    旧口径(段内 floor("2h"), 网格锚在午夜)在这个夹具上给出 [60,60,60,75,30,60] ——
    一半的箱是短的。新口径按**连续交易分钟数**等分: 前几桶恰好 120, 末桶是余数。

    夹具是「夜盘至 23:00」那一档, 每交易日 345 根:
        桶0  21:01-23:00              120
        桶1  09:01-10:15 + 10:31-11:15  75+45 = 120
        桶2  11:16-11:30 + 13:31-15:00  15+90 = 105   <- 末桶, 余数
    """
    tfb = build_timeframe(two_days, "2h")
    assert np.bincount(tfb.bin_of).tolist() == [120, 120, 105] * 2
    assert [f"{x:%m-%d %H:%M}" for x in tfb.bars.index] == [
        "03-02 23:00", "03-03 11:15", "03-03 15:00",
        "03-03 23:00", "03-04 11:15", "03-04 15:00"]


def test_2h_在AU那一档落成五个整桶(au形状):
    """
    实测 AU 的目标形状。前四桶各恰好 120 个**连续交易分钟**, 末桶 75。
    桶0 是 121 根 = 120 个交易分钟 + 那根不占额度的 21:00 竞价根。
    """
    tfb = build_timeframe(au形状, "2h")
    assert np.bincount(tfb.bin_of).tolist() == [121, 120, 120, 120, 75]
    assert [f"{x:%H:%M}" for x in tfb.bars.index] == [
        "23:00", "01:00", "09:30", "13:45", "15:00"]
    # 桶0 的 open 必须是竞价撮合价 —— 交易所定义的开盘价
    assert tfb.bars["open"].iloc[0] == 999.0


def test_2h_那一桶刻意横跨夜日断口(au形状):
    """
    `01:00-09:30` 这一桶含 01:01-02:30 与 09:01-09:30 两截, 中间隔着 6.5 小时。
    **这是设计**, 不是漏网 —— 两截加起来正好 120 个交易分钟。15m 仍然一根都不许跨。
    """
    seg = derive_segments(au形状)
    tfb = build_timeframe(au形状, "2h", segments=seg)
    m = np.flatnonzero(tfb.bin_of == 2)
    assert len(np.unique(seg[m])) == 2, "这一桶本来就该跨段"
    assert au形状.index[m[0]] == pd.Timestamp("2026-03-03 01:01")
    assert au形状.index[m[-1]] == pd.Timestamp("2026-03-03 09:30")
    assert len(m) == 120
    # 15m 是对照组: 时钟分箱, 一根都不许跨
    b15 = build_timeframe(au形状, "15m", segments=seg).bin_of
    edge = np.r_[True, b15[1:] != b15[:-1]]
    first = np.flatnonzero(edge)
    last = np.r_[first[1:] - 1, len(b15) - 1]
    assert (seg[first] == seg[last]).all()


def test_没有任何周期的桶跨trading_date(au形状, two_days):
    """
    跨段可以放开, 跨交易日不行 —— 跨了它一根 bar 会横过周末那 58 小时, 而
    `first_bin_at_or_after` 切暖机段时正是切在交易日边界上。
    """
    for df in (au形状, two_days):
        td = df["trading_date"].to_numpy()
        for tf in ("1m", "5m", "15m", "30m", "2h", "1d"):
            b = build_timeframe(df, tf).bin_of
            edge = np.r_[True, b[1:] != b[:-1]]
            first = np.flatnonzero(edge)
            last = np.r_[first[1:] - 1, len(b) - 1]
            assert (td[first] == td[last]).all(), tf


def test_2h_竞价根不占额度(au形状):
    """
    竞价根是撮合结果不是一分钟连续交易。让它占一个额度, 所有桶边界会整体前移一根
    (23:00->22:59, 01:00->00:59, 09:30->09:29, 13:45->13:44) —— 那正是 AU 上天真地
    「每 120 根切一刀」会得到的错误结果。`merge_leading_singleton=False` 保留了那条
    字面路径, 供 A/B。
    """
    字面 = build_timeframe(au形状, "2h", merge_leading_singleton=False)
    assert [f"{x:%H:%M}" for x in 字面.bars.index] == [
        "22:59", "00:59", "09:29", "13:44", "15:00"]
    assert np.bincount(字面.bin_of).tolist() == [120, 120, 120, 120, 76]


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


@pytest.mark.parametrize("夜盘根数, 期望桶数", [
    (0, 2),      # 无夜盘: 225 交易分钟 -> 120 + 105
    (120, 3),    # 至 23:00: 345 -> 120 + 120 + 105
    (240, 4),    # 至 01:00: 465 -> 120*3 + 105
    (330, 5),    # 至 02:30 (AU/AG/SC): 555 -> 120*4 + 75
])
def test_2h_每交易日桶数只由夜盘档位决定(夜盘根数, 期望桶数):
    """
    等分之后每交易日的桶数是 `ceil(交易分钟数 / 120)`, 只由夜盘档位决定, 不再受
    「网格切在哪」影响。这仍然意味着**同一个 MA55 在不同品种上不是同一段时间**
    (无夜盘 2 桶/日 -> 55 根 ≈ 27.5 个交易日; AU 5 桶/日 -> 11 个交易日), 但至少
    每一桶现在都是等量的交易时间。
    """
    rows = []
    if 夜盘根数:
        rows += _bars("2026-03-02", (("21:01", 夜盘根数),), "2026-03-03", 1000.0)
    rows += _bars("2026-03-03", _DAY_RUNS, "2026-03-03", 2000.0)
    tfb = build_timeframe(_frame(rows), "2h")
    assert tfb.n_bars == 期望桶数
    sizes = np.bincount(tfb.bin_of).tolist()
    assert sizes[:-1] == [120] * (期望桶数 - 1), sizes
    assert 0 < sizes[-1] <= 120, sizes
