# -*- coding: utf-8 -*-
"""
`src/strategy/pullback.py` 的向量化原语 vs **未改动的** v2.1 `src/rules/*`。

这是整个移植「策略逻辑一个字没变」的凭据。向量化是为了性能 (原实现在 1 分钟
数据上是 O(n^2), 见 pullback.py 的模块 docstring), 但性能重写最容易悄悄改掉语义,
所以每一个原语都在**每一个前缀**上和原函数逐位比对 —— 不是抽查几个点。

若这些测试红了, 说明 v3.0 引擎跑的已经不是 phaseF 那个策略了。
"""

import random

import numpy as np
import pandas as pd
import pytest

from src.rules.cooldown import CooldownManager
from src.rules.ma_filter import ma_array_state, ma_fast_mid_state
from src.rules.stop_loss import StopLossCalculator, _nearest_platform_level
from src.rules.take_profit import TakeProfitManager
from src.strategy.pullback import (
    PullbackParams,
    SwingIndex,
    chandelier_stop,
    gap_filled,
    ma_array_states,
    ma_fast_mid_states,
    nearest_platform,
    partial_trigger_price,
    state_name,
    stop_level,
    structure_break_flags,
    swing_flags,
)

P = PullbackParams()
N = 140          # 够长: 越过 MA50 的暖机线, 又不至于让 O(n^2) 的对照实现变慢


def _walk(seed: int, n: int = N, start: float = 100.0, vol: float = 0.6) -> pd.DataFrame:
    """
    随机游走 OHLC。刻意用**粗糙的价格网格** (0.5 一档) —— 中国期货的最小变动价位
    本来就是离散的, 而且离散网格会制造大量 `MA5 == MA20` 的平局和等高的 swing 点,
    正是最容易在重写中被改掉的那些分支。
    """
    rng = random.Random(seed)
    closes, px = [], start
    for _ in range(n):
        px = max(1.0, px + rng.choice([-2, -1, -1, 0, 1, 1, 2]) * vol)
        closes.append(round(px * 2) / 2)
    c = np.array(closes, dtype="float64")
    o = np.r_[c[0], c[:-1]]
    span = np.array([round(rng.uniform(0, 3 * vol) * 2) / 2 for _ in range(n)])
    hi = np.maximum(o, c) + span
    lo = np.minimum(o, c) - span
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": np.ones(n)})


# --------------------------------------------------------- MA 排列状态 ----
@pytest.mark.parametrize("seed", range(6))
def test_ma_fast_mid_states_matches_v21_on_every_prefix(seed):
    df = _walk(seed)
    got = ma_fast_mid_states(df["close"], P.ma_fast, P.ma_mid)
    for pos in range(len(df)):
        want = ma_fast_mid_state(df.iloc[: pos + 1], P.ma_fast, P.ma_mid)
        assert state_name(got[pos]) == want, f"seed={seed} pos={pos}"


@pytest.mark.parametrize("seed", range(6))
def test_ma_array_states_matches_v21_on_every_prefix(seed):
    df = _walk(seed)
    got = ma_array_states(df["close"], P.ma_fast, P.ma_mid, P.ma_slow)
    for pos in range(len(df)):
        want = ma_array_state(df.iloc[: pos + 1], P.ma_fast, P.ma_mid, P.ma_slow)
        assert state_name(got[pos]) == want, f"seed={seed} pos={pos}"


def test_warmup_branch_is_neutral_up_to_the_exact_bar():
    """
    `len(klines) < mid -> neutral` 与 `rolling(mid) 为 NaN` 是同一个条件, 所以
    转变必须恰好发生在下标 mid-1 / slow-1 上, 不能差一根。
    """
    df = _walk(11, n=400)
    fm = ma_fast_mid_states(df["close"], 5, 20)
    arr = ma_array_states(df["close"], 5, 20, 50)
    # 暖机区必须全 neutral, 一根都不能提前。
    assert (fm[:19] == 0).all()
    assert (arr[:49] == 0).all()
    # 而且暖机之后确实会出现非 neutral —— 否则上面两条是空真。
    assert fm[19:].any() and arr[49:].any()
    # 三线排列是两线排列的加严, 所以它永远不会更早开始。
    assert int(np.flatnonzero(arr)[0]) >= int(np.flatnonzero(fm)[0])


def test_exact_ties_are_neutral_on_both_sides():
    """完全恒定的序列 -> 三条均线相等 -> 两边都必须判 neutral, 不能出现 bullish。"""
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "volume": 1.0}, index=range(80))
    assert (ma_fast_mid_states(df["close"]) == 0).all()
    assert (ma_array_states(df["close"]) == 0).all()
    for pos in (25, 60, 79):
        assert ma_fast_mid_state(df.iloc[: pos + 1]) == "neutral"
        assert ma_array_state(df.iloc[: pos + 1]) == "neutral"


# ------------------------------------------------------------ 结构破坏 ----
@pytest.mark.parametrize("seed", range(6))
def test_structure_break_flags_matches_v21_on_every_prefix(seed):
    # 实体要够大才可能同时包住两条均线, 所以用一个振幅更大的游走。
    df = _walk(seed, vol=2.5)
    cd = CooldownManager()
    got = structure_break_flags(df, (P.ma_fast, P.ma_mid, P.ma_slow))
    for pos in range(len(df)):
        want = cd.check_structure_break(df.iloc[: pos + 1])
        assert bool(got[pos]) is bool(want), f"seed={seed} pos={pos}"


def test_structure_break_actually_fires_somewhere():
    """守卫上一个测试: 全 False 也能"通过", 那就什么都没验证。"""
    hits = sum(int(structure_break_flags(_walk(s, vol=2.5)).any()) for s in range(6))
    assert hits >= 3


# --------------------------------------------------------- swing 平台位 ----
@pytest.mark.parametrize("seed", range(6))
def test_nearest_platform_matches_v21_on_every_prefix(seed):
    df = _walk(seed)
    swings = SwingIndex.build(df, P.swing_k)
    k = P.swing_k
    for pos in range(len(df)):
        prefix = df.iloc[: pos + 1]
        for entry in (prefix["close"].iloc[-1], prefix["close"].iloc[-1] + 3.0):
            for is_long in (True, False):
                want = _nearest_platform_level(
                    prefix, entry, "long" if is_long else "short", k)
                got = nearest_platform(swings, pos - k, entry, is_long)
                if want is None:
                    assert got is None, f"seed={seed} pos={pos} long={is_long}"
                else:
                    assert got == pytest.approx(want, abs=0.0)


def test_swing_flags_uses_equality_like_the_original():
    """
    原实现是 `highs[i] == window.max()`, 所以一段等高的平台上**每一根**都是
    swing high。用 `>` 于两侧会只留一根 —— 那会改变平台止损选到哪一档。
    """
    df = pd.DataFrame({
        "open": [1, 2, 3, 3, 3, 2, 1], "high": [1, 2, 3, 3, 3, 2, 1],
        "low": [1, 2, 3, 3, 3, 2, 1], "close": [1, 2, 3, 3, 3, 2, 1],
        "volume": [1] * 7,
    }, dtype="float64")
    is_hi, _ = swing_flags(df, k=2)
    assert list(is_hi) == [False, False, True, True, True, False, False]


def test_swing_flags_empty_and_short_frames():
    for n in (0, 1, 4):
        df = _walk(0, n=n) if n else pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []}, dtype="float64")
        is_hi, is_lo = swing_flags(df, k=2)
        assert is_hi.shape == (n,) and not is_hi.any()
        assert is_lo.shape == (n,) and not is_lo.any()


# --------------------------------------------------------- 止损 / 止盈 ----
def test_stop_level_matches_v21_including_the_tie_rule():
    """
    平局(ATR 止损与平台位离入场价一样远)时原实现取 **ATR 止损** —— `min` 的
    candidates 列表里它排第一, Python 的 min 平局返回第一个。
    """
    calc = StopLossCalculator(P.sl_atr_mult, P.sl_offset_pct, P.swing_k)
    df = _walk(3)
    swings = SwingIndex.build(df, P.swing_k)
    rng = random.Random(7)
    for _ in range(300):
        pos = rng.randrange(2 * P.swing_k + 1, len(df))
        prefix = df.iloc[: pos + 1]
        entry = float(prefix["close"].iloc[-1]) + rng.uniform(-4, 4)
        atr_v = rng.uniform(0.2, 6.0)
        for is_long in (True, False):
            d = "long" if is_long else "short"
            want = calc.calc(entry, d, atr_v, prefix)
            plat = nearest_platform(swings, pos - P.swing_k, entry, is_long)
            got = stop_level(entry, is_long, atr_v, plat, P)
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

    # 构造一次精确平局: 平台位与 ATR 止损等距 -> 必须选 ATR 止损。
    entry, atr_v = 100.0, 2.0                       # atr_stop(long) = 97.0
    assert stop_level(entry, True, atr_v, 97.0, P) == pytest.approx(
        97.0 - entry * P.sl_offset_pct)


def test_take_profit_geometry_matches_v21():
    tp = TakeProfitManager(P.partial_ratio, P.rr_trigger, P.chandelier_mult)
    rng = random.Random(5)
    for _ in range(400):
        entry = rng.uniform(50, 5000)
        r = rng.uniform(0.1, 40)
        atr_v = rng.uniform(0.1, 40)
        for is_long in (True, False):
            d = "long" if is_long else "short"
            sl = entry - r if is_long else entry + r
            assert partial_trigger_price(entry, is_long, sl, P) == pytest.approx(
                tp.partial_trigger_price(entry, d, sl))
            ext = entry + rng.uniform(-3 * r, 8 * r) * (1 if is_long else -1)
            assert chandelier_stop(is_long, ext, atr_v, sl, P) == pytest.approx(
                tp.chandelier_stop(d, ext, atr_v, sl))


def test_chandelier_never_worse_than_the_original_stop():
    """吊灯线是"只朝有利方向移动"的 —— 多头不低于原始止损, 空头不高于。"""
    rng = random.Random(9)
    for _ in range(500):
        entry, r, atr_v = 1000.0, rng.uniform(1, 50), rng.uniform(1, 200)
        sl_long, sl_short = entry - r, entry + r
        ext_l = entry + rng.uniform(-5 * r, 5 * r)
        assert chandelier_stop(True, ext_l, atr_v, sl_long, P) >= sl_long
        assert chandelier_stop(False, entry - (ext_l - entry), atr_v, sl_short, P) <= sl_short


def test_gap_filled_is_asymmetric_and_switchable():
    # 多头止损 95, 跳空低开到 92 -> 按 92 成交 (更差)
    assert gap_filled(95.0, True, 92.0, True) == 92.0
    # 没跳空 -> 按线成交
    assert gap_filled(95.0, True, 97.0, True) == 95.0
    # 空头止损 105, 跳空高开到 108 -> 按 108 成交
    assert gap_filled(105.0, False, 108.0, True) == 108.0
    assert gap_filled(105.0, False, 103.0, True) == 105.0
    # 关掉 -> 永远按线
    assert gap_filled(95.0, True, 92.0, False) == 95.0


def test_params_reject_bad_ma_ordering():
    with pytest.raises(ValueError):
        PullbackParams(ma_fast=20, ma_mid=5, ma_slow=50)
