# -*- coding: utf-8 -*-
"""
冷静期「同方向连续看错」计数器的精确语义。

    纯止损 SL        -> 同方向 +1; 够阈值就 arm, 并**在 arm 的同时清零**
    部分止盈 TP      -> 清零
    护盈止损 PSL     -> 什么也不做
    时段末强平 END   -> 什么也不做

「什么也不做」= 既不加也不清零。这与「清零」在观察上是能区分的:
`SL, END, SL, SL` 必须 arm (END 没打断连续), 而 `SL, TP, SL, SL` 必须不 arm。
这组测试的核心就是这一对。

真实数据上验不了: 冷静期一 arm 就挡住后续入场, 两条时间线立刻整个岔开, 没法逐点
对比。所以这里手搭一段每个时段恰好开一批、且那一批怎么收尾由测试指定的帧 ——
`run_pullback_v3_backtest` 的 `tf` 是参数正是为了这种测试。
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import Prepared
from src.data.v3_timeframes import TFBars
from src.engine.pullback_v3_backtest import (
    END_SESSION, PROTECTIVE_SL, SL, TP, run_pullback_v3_backtest,
)
from src.strategy.pullback import PullbackConfig, PullbackParams

P = PullbackParams()
SESS_BARS = 8          # 每个时段 8 根: 0-2 回调确认, 3 入场, 4-6 持有, 7 时段末根
ENTRY_BAR = 3
FLAT = 250.0
LOSS, WIN = FLAT - 0.5, FLAT + 0.5

# 与 test_pullback_v3_entry_gate.py 共用的两把梯子。下标 -> 排列状态。
BEAR_POS, BULL_POS, DAY_POS = 99, 199, 59


def _ladder() -> pd.DataFrame:
    c = np.r_[np.linspace(300.0, 200.0, 100), np.linspace(200.0, 300.0, 100)]
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1.0,
                         "low": np.minimum(o, c) - 1.0, "close": c,
                         "volume": np.ones(c.size)})


def _daily() -> pd.DataFrame:
    """60 根单调上涨的日线 -> 末根 MA5 > MA20 > MA50, 多头方向。"""
    c = np.linspace(100.0, 160.0, 60)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": np.ones(c.size)})


def _frame(shapes: list[str]) -> tuple[Prepared, dict[str, TFBars]]:
    """
    一个时段一种收尾形状:

        "end-"   全程平在 FLAT, 末根收在 FLAT-0.5   -> 亏损的整批 END_SESSION
        "end+"   同上但末根收在 FLAT+0.5            -> 盈利的整批 END_SESSION
        "sl"     bar 5 砸穿止损                     -> 纯 SL
        "tp"     bar 5 冲过 2R                      -> TP, 下一根被吊灯打掉 -> PSL

    每个时段只可能开一批: 小周期在 bar 0-2 是空、bar 3 起转多, 一个时段里只有一次
    "空转多", 所以 `pullback_seen` 用掉之后不会在同一时段里再攒出一次。
    """
    n = len(shapes) * SESS_BARS
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-03-03 09:01") + pd.Timedelta(minutes=i) for i in range(n)],
        name="datetime")
    pos_in = np.arange(n) % SESS_BARS
    sess = np.arange(n) // SESS_BARS

    close = np.full(n, FLAT)
    high = np.full(n, FLAT)
    low = np.full(n, FLAT)
    for k, shape in enumerate(shapes):
        last, mid = k * SESS_BARS + SESS_BARS - 1, k * SESS_BARS + 5
        if shape == "end-":
            close[last], low[last] = LOSS, LOSS
        elif shape == "end+":
            close[last], high[last] = WIN, WIN
        elif shape == "sl":
            low[mid] = FLAT - 40.0
        elif shape == "tp":
            high[mid] = FLAT + 40.0
        else:                                     # pragma: no cover
            raise ValueError(shape)

    df = pd.DataFrame(
        {"open": np.full(n, FLAT), "high": high, "low": low, "close": close,
         "volume": np.ones(n), "trading_date": pd.Timestamp("2026-03-03"),
         "session_id": sess + 1,
         "is_session_first": pos_in == 0,
         "is_session_last": pos_in == SESS_BARS - 1,
         "tradable": True},
        index=idx)

    ladder, daily = _ladder(), _daily()
    ar = np.arange(n, dtype=np.int64)
    small = np.where(pos_in <= 2, BEAR_POS, BULL_POS).astype(np.int64)
    tf = {
        "1m": TFBars("1m", ladder, ar, small),
        "5m": TFBars("5m", ladder, ar, small),
        # 30m 全程空头: 回调确认要它是空, 触发根本不读它。
        "30m": TFBars("30m", ladder, ar, np.full(n, BEAR_POS, dtype=np.int64)),
        # 日线始终停在同一根 -> 冷静期只会 arm 不会 release, 本文件只关心 arm。
        "1d": TFBars("1d", daily, np.zeros(n, dtype=np.int64),
                     np.full(n, DAY_POS, dtype=np.int64)),
    }
    prep = Prepared(
        symbol="CD", df=df, lines=pd.DataFrame(),
        line_idx=np.zeros(n, dtype=np.int32), k0=1.0,
        params=None, config=None, mode="split",  # type: ignore[arg-type]
    )
    return prep, tf


def _run(shapes: list[str], **cfg):
    prep, tf = _frame(shapes)
    return run_pullback_v3_backtest(prep, tf, config=PullbackConfig(**cfg))


def _armed(shapes: list[str]) -> bool:
    return bool(_run(shapes).cooldown_spans)


def _kinds(res) -> list[tuple[int, frozenset]]:
    b: dict[int, set] = {}
    for t in res.trades:
        b.setdefault(t.entry_bar_idx, set()).add(t.reason)
    return [(e, frozenset(v)) for e, v in sorted(b.items())]


# ------------------------------------------------------------ fixture 自检 ----
@pytest.mark.parametrize("shape,want", [
    ("end-", frozenset({END_SESSION})),
    ("end+", frozenset({END_SESSION})),
    ("sl", frozenset({SL})),
    ("tp", frozenset({TP, PROTECTIVE_SL})),
])
def test_each_shape_produces_the_intended_exit(shape, want):
    """守卫: 下面每一条断言都建立在"这四种形状真的走了那条出场分支"上。"""
    res = _run([shape])
    assert _kinds(res) == [(ENTRY_BAR, want)], shape


def test_one_batch_per_session():
    res = _run(["end-"] * 3)
    assert [e for e, _ in _kinds(res)] == [
        ENTRY_BAR, ENTRY_BAR + SESS_BARS, ENTRY_BAR + 2 * SESS_BARS]


def test_losing_and_winning_bell_closes_are_distinguishable():
    """"亏损收盘"这个概念在 fixture 里是真的存在的 —— 否则下面的断言是空真。"""
    assert _run(["end-"]).trades[0].pnl_pct < 0
    assert _run(["end+"]).trades[0].pnl_pct > 0


# --------------------------------------------------------- 纯止损: 加一 ----
def test_three_same_direction_stops_arm_the_cooldown():
    assert _armed(["sl", "sl", "sl"])
    assert _run(["sl", "sl", "sl"]).cooldown_spans[0][0] == 2 * SESS_BARS + 5


def test_two_stops_are_not_enough():
    assert not _armed(["sl", "sl"])


def test_arming_zeroes_the_counter():
    """
    arm 的同时清零 —— 否则冷静期解除之后第一笔止损就会立刻再次 arm。

    这个 fixture 里冷静期不会解除(日线停着不动), 所以换个观察角度: 冷静期只该
    出现**一段**, 而不是每根 bar 都重新 arm 出一段。
    """
    assert len(_run(["sl"] * 6).cooldown_spans) == 1


# ------------------------------------------- END_SESSION: 什么也不做 ----
def test_bell_closes_alone_never_arm():
    assert not _armed(["end-"] * 6)
    assert not _armed(["end+"] * 6)


def test_a_bell_close_does_not_break_the_stop_streak():
    """
    **这是"什么也不做"与"清零"的判别测试。**

    `SL, END, SL, SL` -> 三笔同向止损, 中间那次收盘平仓不打断连续 -> 必须 arm。
    若 END 改成清零, 这里只会数到 2, 不 arm。
    """
    assert _armed(["sl", "end-", "sl", "sl"])
    assert _armed(["sl", "end+", "sl", "sl"]), "盈利的收盘平仓同样不该打断"


def test_a_bell_close_does_not_advance_the_stop_streak_either():
    """
    另一半: END 也不能算成一次止损。`SL, END, END` 只有一笔止损, 不许 arm。
    """
    assert not _armed(["sl", "end-", "end-"])
    assert not _armed(["end-", "end-", "end-"])


# --------------------------------------------------- TP: 清零 / PSL: 不动 ----
def test_a_take_profit_resets_the_stop_streak():
    """`SL, TP, SL, SL` -> 止盈把连续清零, 之后只数到 2 -> 不许 arm。"""
    assert not _armed(["sl", "tp", "sl", "sl"])
    # 对照: 把中间那一批换成收盘平仓就会 arm —— 差别只在 TP 清零而 END 不清
    assert _armed(["sl", "end-", "sl", "sl"])


def test_protective_stop_alone_neither_arms_nor_matters():
    """
    护盈止损什么也不做。它在观察上等价于"清零", 因为走到 PSL 必然已经 TP 过,
    而 TP 刚清过零 —— 两条腿之间同一批一直持着仓, 没有别的批次能改计数器。
    (59 品种 x 5 模式实测 594 个 TP+PSL 批次, 两条腿之间零个别的批次。)

    所以这里只钉住可观测的部分: 一串 TP+PSL 永远不 arm。
    """
    assert not _armed(["tp"] * 6)
    assert _kinds(_run(["tp"]))[0][1] == frozenset({TP, PROTECTIVE_SL})


# ------------------------------------------- 触发器 2: 日线结构破位 ----
def _frame_with_break() -> tuple[Prepared, dict[str, TFBars]]:
    """
    把日线**末根**换成一根实体横跨 MA5/MA20/MA50 的大阴线 -> `DBREAK` 为真。

    其余与 `_frame(["end-"] * 3)` 相同, 于是唯一能 arm 冷静期的就是结构破位这一条
    (三次收盘平仓什么都不算)。
    """
    prep, tf = _frame(["end-"] * 3)
    d = _daily().copy()
    # 前 59 根一路涨到 ~159, MA50 ≈ 130 出头。末根从 165 砸到 120: 实体
    # [120, 165] 把三条均线全包进去。
    d.loc[d.index[-1], ["open", "high", "low", "close"]] = [165.0, 166.0, 119.0, 120.0]
    n = len(prep.df)
    tf = dict(tf)
    tf["1d"] = TFBars("1d", d, np.zeros(n, dtype=np.int64),
                      np.full(n, DAY_POS, dtype=np.int64))
    return prep, tf


def test_the_break_fixture_really_breaks():
    """守卫: 这根日线的实体确实吃穿了 >=2 条均线, 否则下面两条是空真。"""
    from src.strategy.pullback import structure_break_flags
    _, tf = _frame_with_break()
    flags = structure_break_flags(tf["1d"].bars, (P.ma_fast, P.ma_mid, P.ma_slow))
    assert bool(flags[DAY_POS]), "末根日线没被判成结构破位 —— fixture 没搭对"
    # 对照: 原来那把单调上涨的梯子不该触发
    assert not bool(structure_break_flags(
        _daily(), (P.ma_fast, P.ma_mid, P.ma_slow))[DAY_POS])


def test_structure_break_arms_the_cooldown_on_its_own():
    """三次收盘平仓什么都不算, 所以 arm 只可能来自结构破位。"""
    prep, tf = _frame_with_break()
    res = run_pullback_v3_backtest(prep, tf)
    assert res.cooldown_spans, "结构破位没能 arm 冷静期"
    # 日频规则在第一根就评一次 -> 冷静期从 bar 0 起
    assert res.cooldown_spans[0][0] == 0
    assert res.trades == [], "冷静期从头挡到尾, 不该有任何成交"


def test_structure_break_switch_turns_it_off():
    prep, tf = _frame_with_break()
    res = run_pullback_v3_backtest(
        prep, tf, config=PullbackConfig(cooldown_on_structure_break=False))
    assert res.cooldown_spans == [], "开关关掉后结构破位还在 arm"
    assert res.trades, "关掉之后应该恢复正常交易"


def test_the_switch_does_not_touch_the_stop_streak_trigger():
    """关掉触发器 2 之后, 触发器 1 (连续三次纯止损) 必须照常工作。"""
    off = {"cooldown_on_structure_break": False}
    assert _run(["sl", "sl", "sl"], **off).cooldown_spans
    assert not _run(["sl", "sl"], **off).cooldown_spans
    # END/TP 的判别测试在关掉之后同样成立
    assert _run(["sl", "end-", "sl", "sl"], **off).cooldown_spans
    assert not _run(["sl", "tp", "sl", "sl"], **off).cooldown_spans


# --------------------------------------------------------------- 成本 ----
def test_cost_bps_never_enters_the_decision():
    """
    成本只能是记账层的叠加, 不能是信号 —— 调费率不许改变成交序列或冷静期。
    (曾经有一版用 `pnl_net` 判"收盘亏没亏", 就是踩在这上面。)
    """
    for shapes in (["end-"] * 4, ["sl", "end-", "sl", "sl"], ["sl", "tp", "sl", "sl"]):
        a, b = _run(shapes), _run(shapes, cost_bps=50.0)
        assert [(t.entry_bar_idx, t.exit_bar_idx, t.reason) for t in a.trades] == \
               [(t.entry_bar_idx, t.exit_bar_idx, t.reason) for t in b.trades], shapes
        assert a.cooldown_spans == b.cooldown_spans, shapes
