# -*- coding: utf-8 -*-
"""
`prepare_base(warmup_days=...)` 的暖机段。

回调策略的趋势方向要**日线 MA50**, 也就是显示窗口第一根 bar 上就得有 50 根日线
可用。`prepare_base` 原来只 pad 20 个日历日(给六条线的前日参照用), 远远不够。

这里钉两件事:
  1. `warmup_days=0` 时输出与历史行为**逐字节相同** —— 这是"没有破坏 R-Breaker"
     的凭据, 比 test_rbreaker_* 全绿更直接。
  2. `warmup_days>0` 时暖机段留在帧里(指标要在它上面滚), 但 `valid=False` 所以
     不产生交易, 且切点落在 trading_date 边界上。

用真实 parquet 而不是合成帧 —— 暖机段的意义就在于它必须能从**真实的**交易日历里
数出正确的天数 (周末、节假日、春节那个 10 天的洞)。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import CLEAN_DIR, apply_mode, prepare_base
from src.data.v3_timeframes import build_timeframes

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "RB.parquet")),
    reason="需要 data/01-pkl层/一次排查/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)

START, END = "2025-07-01", "2025-10-01"
WARM = 60


@pytest.fixture(scope="module")
def cold():
    return prepare_base("RB", START, END)


@pytest.fixture(scope="module")
def warm():
    return prepare_base("RB", START, END, warmup_days=WARM)


def test_warmup_zero_is_byte_identical(cold):
    """默认参数这条路径必须和加了 warmup_days=0 完全一样。"""
    again = prepare_base("RB", START, END, warmup_days=0)
    pd.testing.assert_frame_equal(cold.df, again.df)
    pd.testing.assert_frame_equal(cold.lines, again.lines)
    assert cold.k0 == again.k0
    assert again.warmup_bars == 0 and again.warmup_days == 0


def test_warm_frame_is_the_cold_frame_with_a_prefix(cold, warm):
    """
    暖机只是在前面接了一段 —— 显示窗口那部分必须**逐字节**不变, 包括复权后的
    价格 (即 k0 仍然取自显示窗口首根, 而不是暖机段首根)。
    """
    assert warm.warmup_bars > 0
    assert warm.k0 == cold.k0
    pd.testing.assert_frame_equal(
        warm.df.iloc[warm.warmup_bars:], cold.df, check_freq=False)


def test_warmup_covers_the_requested_number_of_trading_days(cold, warm):
    warm_part = warm.df.iloc[: warm.warmup_bars]
    days = pd.unique(warm_part["trading_date"].to_numpy())
    assert len(days) == WARM == warm.warmup_days
    assert warm_part["trading_date"].max() < cold.df["trading_date"].min()


def test_warmup_cut_lands_on_a_trading_date_boundary(warm):
    """
    切点必须整齐 —— 否则 v3_timeframes 会把一根合成 K 线劈成两半, 而它的
    `first_bin_at_or_after` 明确依赖这一条。
    """
    td = warm.df["trading_date"].to_numpy()
    i = warm.warmup_bars
    assert td[i - 1] != td[i]
    for t in build_timeframes(warm.df).values():
        members = np.flatnonzero(t.bin_of == t.first_bin_at_or_after(i))
        assert members[0] == i


def test_warmup_bars_are_never_tradable(warm):
    """
    `prev_day_lines` 的 `tradable_from` 已经把暖机段全部标成 valid=False,
    所以就算引擎忘了看 in_window, R-Breaker 那条路也不会在暖机段开仓。
    """
    prep = apply_mode(warm, "split")
    valid = prep.lines["valid"].to_numpy(bool)[prep.line_idx]
    assert not valid[: prep.warmup_bars].any()
    assert valid[prep.warmup_bars:].any()


def test_in_window_matches_warmup_bars(warm):
    prep = apply_mode(warm, "split")
    assert prep.in_window.sum() == len(prep.df) - prep.warmup_bars
    assert not prep.in_window[: prep.warmup_bars].any()
    assert prep.in_window[prep.warmup_bars:].all()


def test_tradable_days_ignores_the_warmup_stretch(cold, warm):
    """可交易日是**显示窗口**的属性, 暖机段不许把它撑大。"""
    for mode in ("split", "night", "day", "night_day", "day_night"):
        assert apply_mode(warm, mode).tradable_days == apply_mode(cold, mode).tradable_days, mode


def test_daily_bars_available_at_the_first_display_bar(warm):
    """
    暖机的目的: 显示窗口第一根 bar 上, 已收盘的日线根数 >= 50 (MA50 的需求)。
    """
    tfs = build_timeframes(warm.df)
    d = tfs["1d"]
    closed_at_first_display = int(d.closed_pos[warm.warmup_bars])
    assert closed_at_first_display + 1 >= 50


def test_short_history_is_reported_not_faked():
    """
    上市晚 / USABLE_FROM 裁过的品种凑不满暖机 —— 必须如实少给, 而不是报错也不是
    假装够了。LH (生猪) 2021-01-14 上市。
    """
    base = prepare_base("LH", "2021-01-20", "2021-03-01", warmup_days=WARM)
    assert 0 < base.warmup_days < WARM
    days = pd.unique(base.df["trading_date"].to_numpy()[: base.warmup_bars])
    assert len(days) == base.warmup_days
