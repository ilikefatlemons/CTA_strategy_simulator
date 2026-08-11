# -*- coding: utf-8 -*-
"""
入场闸门的精确语义 (决策 2): **挡住动作, 不消耗 `pullback_seen`。**

这条在真实数据上观察不到 —— 闸门一挡, 策略就一直空仓、继续找新的回调, 与"不挡"
的那条时间线整个岔开, 没法逐点对比。所以这里手搭一段状态序列, 让四周期级联在
一根**不可交易**的 bar 上恰好走完, 下一根变成可交易:

    bar 0-2  30m=空 且 1m=空          -> 回调在 bar 0 确认
    bar 3    1m=多 且 5m=多, 不可交易  -> 触发被挡
    bar 4    1m=多 且 5m=多, 可交易    -> **必须**在这里入场

若被挡的那次把 `pullback_seen` 清了, bar 4 就不会入场 —— 级联得重新等一轮回调。
这与 `rbreaker_reversal_backtest.py:186-238` 的先例同构 (被挡的入场不清空
`armed_dir`)。

`run_pullback_v3_backtest` 的 `tf` 是参数正是为了这种测试: 直接喂进人造的
`TFBars`, 就能一根一根地指定"此刻各周期的排列状态是什么"。
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import Prepared
from src.data.v3_timeframes import TFBars
from src.engine.pullback_v3_backtest import run_pullback_v3_backtest
from src.strategy.pullback import (
    BEAR, BULL, PullbackConfig, ma_array_states, ma_fast_mid_states,
)

N_BARS = 8


def _ladder() -> pd.DataFrame:
    """
    前 100 根一路下跌、后 100 根一路上涨的 OHLC。于是:
      下标 99  -> MA5 < MA20 -> 空头排列
      下标 199 -> MA5 > MA20 -> 多头排列
    小周期 (1m/5m/30m) 三条共用这一份, 靠 `closed_pos` 指向不同下标来选状态。
    """
    c = np.r_[np.linspace(300.0, 200.0, 100), np.linspace(200.0, 300.0, 100)]
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1.0,
                         "low": np.minimum(o, c) - 1.0, "close": c,
                         "volume": np.ones(c.size)})


def _daily() -> pd.DataFrame:
    """60 根单调上涨的日线 -> 末根 MA5 > MA20 > MA50, 即多头方向。"""
    c = np.linspace(100.0, 160.0, 60)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": np.ones(c.size)})


BEAR_POS, BULL_POS, DAY_POS = 99, 199, 59


def _check_ladder_states() -> None:
    small = ma_fast_mid_states(_ladder()["close"])
    assert small[BEAR_POS] == BEAR and small[BULL_POS] == BULL
    assert ma_array_states(_daily()["close"])[DAY_POS] == BULL


def _build(tradable: list[bool]) -> tuple[Prepared, dict[str, TFBars]]:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-03-03 09:01") + pd.Timedelta(minutes=i) for i in range(N_BARS)],
        name="datetime")
    px = np.full(N_BARS, 250.0)
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": np.ones(N_BARS),
         "trading_date": pd.Timestamp("2026-03-03"),
         "session_id": 1, "is_session_first": False, "is_session_last": False,
         "tradable": tradable},
        index=idx)
    df.iloc[0, df.columns.get_loc("is_session_first")] = True   # type: ignore[index]
    df.iloc[-1, df.columns.get_loc("is_session_last")] = True   # type: ignore[index]

    ladder, daily = _ladder(), _daily()
    # bar 2 = 回调确认那一根 (小周期全空), bar >= 3 = 触发 (小周期全多)
    small = np.where(np.arange(N_BARS) <= 2, BEAR_POS, BULL_POS).astype(np.int64)
    tf = {
        "1m": TFBars("1m", ladder, np.arange(N_BARS, dtype=np.int64), small),
        "5m": TFBars("5m", ladder, np.arange(N_BARS, dtype=np.int64), small),
        # 30m 全程保持空头排列: 回调确认要它是空, 而触发根本不读它。
        "30m": TFBars("30m", ladder, np.arange(N_BARS, dtype=np.int64),
                      np.full(N_BARS, BEAR_POS, dtype=np.int64)),
        "1d": TFBars("1d", daily, np.zeros(N_BARS, dtype=np.int64),
                     np.full(N_BARS, DAY_POS, dtype=np.int64)),
    }
    prep = Prepared(
        symbol="GATE", df=df, lines=pd.DataFrame(),
        line_idx=np.zeros(N_BARS, dtype=np.int32), k0=1.0,
        params=None, config=None, mode="split",  # type: ignore[arg-type]
    )
    return prep, tf


def test_ladder_fixture_produces_the_intended_states():
    """守卫: 下面两个测试全靠这三个下标真的对应到预期的排列状态。"""
    _check_ladder_states()


def test_entry_lands_on_bar_3_when_nothing_is_gated():
    """基线: 不设闸门时, 触发的第一根 (bar 3) 就入场。"""
    prep, tf = _build([True] * N_BARS)
    res = run_pullback_v3_backtest(prep, tf)
    assert res.position[2] == 0
    assert res.position[3] == BULL, "级联走完了却没入场 —— fixture 没搭对"
    # 回调在**条件成立的第一根**上确认(bar 0), 且只确认一次 —— 没有重新等一轮
    assert [p["bar_idx"] for p in res.pullback_points] == [0]


def test_blocked_trigger_does_not_consume_the_pullback():
    """
    bar 3 不可交易 -> 触发被挡。bar 4 可交易且触发条件仍成立 -> **必须**入场。
    若 `pullback_seen` 被那次挡下的触发清掉了, bar 4 会一直空仓。
    """
    tradable = [True] * N_BARS
    tradable[3] = False
    prep, tf = _build(tradable)
    res = run_pullback_v3_backtest(prep, tf)
    assert res.position[3] == 0, "闸门没挡住"
    assert res.position[4] == BULL, "被挡的触发把回调状态吃掉了 —— 决策 2 被违反"
    # 回调只确认过一次, 没有重新等一轮
    # 回调在**条件成立的第一根**上确认(bar 0), 且只确认一次 —— 没有重新等一轮
    assert [p["bar_idx"] for p in res.pullback_points] == [0]


def test_gate_can_be_switched_off():
    tradable = [True] * N_BARS
    tradable[3] = False
    prep, tf = _build(tradable)
    res = run_pullback_v3_backtest(
        prep, tf, config=PullbackConfig(session_gates_entry=False))
    assert res.position[3] == BULL, "关掉闸门后 tradable 不该再起作用"


def test_pullback_confirmation_bar_never_enters_on_itself():
    """回调确认的那一根绝不同时入场 —— 触发必须是更晚的一根 (pb.py:65)。"""
    prep, tf = _build([True] * N_BARS)
    res = run_pullback_v3_backtest(prep, tf)
    assert res.position[2] == 0


@pytest.mark.parametrize("blocked", [[3], [3, 4], [3, 4, 5]])
def test_entry_lands_on_the_first_tradable_bar_after_the_trigger(blocked):
    tradable = [i not in blocked for i in range(N_BARS)]
    prep, tf = _build(tradable)
    res = run_pullback_v3_backtest(prep, tf)
    first = max(blocked) + 1
    assert (res.position[:first] == 0).all()
    assert res.position[first] == BULL
