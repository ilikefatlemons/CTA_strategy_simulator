# -*- coding: utf-8 -*-
"""
`src/viz/chart_pullback.py` 的帧级检查 —— **不开窗口**, 只验推给
lightweight-charts 的那几张表。

窗口里的东西 (按钮、版面、可见性) 没法自动验, 只能照 docs 的清单人工过一遍;
但"喂进去的数据对不对"是可以钉死的, 而且恰恰是最容易错的部分:
标记有没有落在真实的 K 线上、暖机段有没有干净地切掉、冷静期涂色是不是恰好覆盖
被触及的那些箱。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import CLEAN_DIR, apply_mode, prepare_base
from src.data.v3_timeframes import PB_TFS, build_timeframes
from src.engine.pullback_v3_backtest import END_SESSION, run_pullback_v3_backtest
from src.viz.chart_pullback import (
    ATR_H, MA_PERIODS, MAIN_H, REASON_COLOR, REASON_LABEL, ROW_H, TOP, atr_frame,
    cooldown_frame, ma_frame, markers, stats_html, tile_frame,
)
from src.viz.lwc_helpers import price_dp
from src.performance.pullback_stats import compute_stats

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "AG.parquet")),
    reason="需要 data/v3.0/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)

# AG 而不是 RB: 见 test_pullback_v3_backtest.py 顶部的同一条注释 —— 时段锁死之后
# 持仓不跨时段边界, 而跳空只发生在边界上, RB/split 的跳空成交因此归零,
# `test_gapped_exits_are_labelled` 会退化成空真。
SYMBOL, START, END, WARM = "AG", "2025-07-01", "2026-07-29", 60


@pytest.fixture(scope="module")
def ctx():
    prep = apply_mode(prepare_base(SYMBOL, START, END, warmup_days=WARM), "split")
    tf = build_timeframes(prep.df)
    res = run_pullback_v3_backtest(prep, tf)
    return prep, tf, res


def _start(tf, name, warmup):
    return tf[name].first_bin_at_or_after(warmup)


# ------------------------------------------------------------ 显示切片 ----
@pytest.mark.parametrize("name", PB_TFS)
def test_tile_frame_drops_every_warmup_candle(ctx, name):
    prep, tf, _ = ctx
    tfb = tf[name]
    start = _start(tf, name, prep.warmup_bars)
    df = tile_frame(tfb, start)
    assert len(df) == tfb.n_bars - start > 0
    first_display_time = prep.df.index[prep.warmup_bars]
    # 第一根可见 K 线的**末端标签** >= 显示窗口第一根 1m 的时刻
    assert pd.Timestamp(df["time"].iloc[0]) >= first_display_time
    # 且它的前一根整根落在暖机段里
    assert tfb.bars.index[start - 1] < first_display_time


@pytest.mark.parametrize("name", PB_TFS)
def test_tile_frame_carries_no_volume_column(ctx, name):
    """带 volume 的话 `AbstractChart.set` 会自动建成交量副图, 挤掉 ATR 的位置。"""
    prep, tf, _ = ctx
    df = tile_frame(tf[name], _start(tf, name, prep.warmup_bars))
    assert list(df.columns) == ["time", "open", "high", "low", "close"]


@pytest.mark.parametrize("name", PB_TFS)
def test_ma_is_already_warm_on_the_first_visible_candle(ctx, name):
    """
    均线在**全帧(含暖机段)**上滚完再切显示段, 所以第一根可见 K 线上就有值 ——
    这正是暖机 padding 存在的意义。日线的 MA50 是最难满足的那一条。
    """
    prep, tf, _ = ctx
    start = _start(tf, name, prep.warmup_bars)
    for period in MA_PERIODS[name]:
        df = ma_frame(tf[name], period, start)
        assert len(df) == tf[name].n_bars - start
        assert not np.isnan(df[f"MA{period}"].iloc[0]), f"{name} MA{period} 首根是 NaN"


@pytest.mark.parametrize("name", PB_TFS)
def test_atr_is_warm_and_aligned(ctx, name):
    prep, tf, _ = ctx
    start = _start(tf, name, prep.warmup_bars)
    a = atr_frame(tf[name], start)
    c = tile_frame(tf[name], start)
    assert list(a["time"]) == list(c["time"])
    assert not np.isnan(a["atr"].iloc[0])
    assert (a["atr"].dropna() >= 0).all()


def test_daily_tile_has_three_mas_the_others_two():
    assert MA_PERIODS["1d"] == (5, 20, 50)
    for name in ("1m", "5m", "30m"):
        assert MA_PERIODS[name] == (5, 20)


# -------------------------------------------------------------- 标记 ----
@pytest.mark.parametrize("name", PB_TFS)
def test_markers_are_sorted_and_land_on_real_candles(ctx, name):
    """
    `marker_list` (abstract.py:249-273) 不排序, 而 `setMarkers` 对乱序会出错。
    而且每个标记的时刻必须**恰好**是某一根显示 K 线的时刻, 否则渲染器会丢掉它。
    """
    prep, tf, res = ctx
    start = _start(tf, name, prep.warmup_bars)
    ms = markers(res, tf[name], start, notes=True)
    assert ms
    times = [m["time"] for m in ms]
    assert times == sorted(times)
    valid = set(tile_frame(tf[name], start)["time"].to_numpy())
    for m in ms:
        assert np.datetime64(m["time"]) in valid, f"{name}: {m['time']} 不是一根真实 K 线"


@pytest.mark.parametrize("name", PB_TFS)
def test_notes_toggle_only_adds_the_pullback_circles(ctx, name):
    prep, tf, res = ctx
    start = _start(tf, name, prep.warmup_bars)
    off = markers(res, tf[name], start, notes=False)
    on = markers(res, tf[name], start, notes=True)
    assert all(m["shape"] != "circle" for m in off)
    circles = [m for m in on if m["shape"] == "circle"]
    assert len(on) == len(off) + len(circles)
    assert circles, "回调确认点一个都没画出来"
    assert all(m["text"].startswith("Pullback") for m in circles)


def test_entry_arrow_is_drawn_once_per_batch(ctx):
    """一批两条腿, 但入场箭头只画一个。"""
    prep, tf, res = ctx
    start = _start(tf, "1m", prep.warmup_bars)
    ms = markers(res, tf["1m"], start, notes=False)
    n_open = sum(1 for m in ms if " @ " in m["text"])
    assert n_open == len({t.entry_bar_idx for t in res.trades})


def test_entry_text_shows_direction_and_price(ctx):
    """`Long @ 3521.00` —— 方向必须一眼可读, 而不是靠箭头朝向猜。"""
    prep, tf, res = ctx
    start = _start(tf, "1m", prep.warmup_bars)
    ms = markers(res, tf["1m"], start, notes=False)
    entries = [m for m in ms if " @ " in m["text"]]
    assert entries
    want = {f"{'Long' if t.direction == 'long' else 'Short'} @ "
            f"{t.entry_price:.{price_dp(t.entry_price)}f}"
            for t in res.trades}
    assert {m["text"] for m in entries} <= want
    assert any(m["text"].startswith("Long ") for m in entries)
    assert any(m["text"].startswith("Short ") for m in entries)


def test_exit_text_carries_both_size_and_return(ctx):
    """
    `TP 50% +4.00%` —— 两个百分比含义不同: 前者是这条腿占整批的比例, 后者是
    **单位仓位**相对入场价的涨跌 (`pnl_net`), 不是已经乘过仓位的那个。
    """
    prep, tf, res = ctx
    start = _start(tf, "1m", prep.warmup_bars)
    by_exit = {m["time"]: m["text"] for m in markers(res, tf["1m"], start, notes=False)
               if " @ " not in m["text"]}
    checked = 0
    for t in res.trades:
        tail = f"{t.size_fraction:.0%} {t.pnl_net:+.2%}"
        want = f"{REASON_LABEL[t.reason]}{'(G)' if t.gapped else ''} {tail}"
        if want in by_exit.values():
            checked += 1
    assert checked >= len(res.trades) // 2, "离场文案格式对不上"
    # 收益那一位**不是**乘过仓位的 —— 拿一条 50% 的腿反证
    half = next(t for t in res.trades if t.size_fraction < 1.0 and abs(t.pnl_net) > 1e-4)
    assert f"{half.pnl_net:+.2%}" != f"{half.size_fraction * half.pnl_net:+.2%}"


def test_end_session_markers_are_labelled_and_greyed(ctx):
    """时段强平的箭头: 文字 `End Session`, 颜色是那个不带红绿语气的蓝灰。"""
    prep, tf, res = ctx
    start = _start(tf, "1m", prep.warmup_bars)
    ms = markers(res, tf["1m"], start, notes=False)
    ends = [m for m in ms if m["text"].startswith("End Session")]
    assert len(ends) == sum(1 for t in res.trades if t.reason == END_SESSION) > 0
    assert {m["color"] for m in ends} == {REASON_COLOR[END_SESSION]}
    # 收盘平仓永远不跳空 —— 按 close 成交, 不做补价
    assert not any("(G)" in m["text"] for m in ends)


def test_gapped_exits_are_labelled(ctx):
    prep, tf, res = ctx
    start = _start(tf, "1m", prep.warmup_bars)
    ms = markers(res, tf["1m"], start, notes=False)
    assert sum(1 for m in ms if "(G)" in m["text"]) == sum(1 for t in res.trades if t.gapped)
    assert any("(G)" in m["text"] for m in ms), "真实数据里应当有跳空成交"


def test_markers_never_reach_into_the_warmup_stretch(ctx):
    """暖机段不交易, 所以那一段上一个标记都不该有。"""
    prep, tf, res = ctx
    for name in PB_TFS:
        start = _start(tf, name, prep.warmup_bars)
        lo = tf[name].bars.index[start]
        for m in markers(res, tf[name], start, notes=True):
            assert m["time"] >= lo, name


# ---------------------------------------------------------- 冷静期带 ----
@pytest.mark.parametrize("name", PB_TFS)
def test_cooldown_shades_exactly_the_touched_bins(ctx, name):
    """
    某根合成 K 线只要包含**任意一根**处于冷静期的 1m bar 就整根涂色。
    这里对着 `cooldown_spans` 独立重算一遍箱集合, 逐个比对。
    """
    prep, tf, res = ctx
    tfb = tf[name]
    start = _start(tf, name, prep.warmup_bars)
    df = cooldown_frame(res, tfb, start)
    assert df is not None and len(df)
    assert (df["cooldown"] == 1.0).all()

    want = set()
    for s, e in res.cooldown_spans:
        want.update(int(b) for b in np.unique(tfb.bin_of[s:e + 1]) if b >= start)
    got = {int(np.searchsorted(tfb.bars.index.to_numpy(), np.datetime64(t)))
           for t in df["time"]}
    assert got == want


def test_cooldown_frame_is_none_when_there_is_nothing_to_draw(ctx):
    import dataclasses
    prep, tf, res = ctx
    empty = dataclasses.replace(res, cooldown_spans=[])
    assert cooldown_frame(empty, tf["1m"], 0) is None


# -------------------------------------------------------------- 版面 ----
def test_layout_fractions_tile_the_screen():
    assert TOP + 2 * ROW_H == pytest.approx(1.0)
    assert MAIN_H + ATR_H == pytest.approx(ROW_H)
    assert 0 < TOP < 0.15 and MAIN_H > ATR_H


def test_stats_html_contains_the_panel_rows(ctx):
    _, _, res = ctx
    html = stats_html(compute_stats(res, "日夜分开"))
    for label in ("可交易日", "批次", "胜率(按批)", "离场 止损/止盈/护盈"):
        assert label in html
    assert "回调入场" in html and "日夜分开" in html
