# -*- coding: utf-8 -*-
"""
`src/performance/pullback_stats.py`。

最要紧的一条: **胜率和盈亏比按「批」算, 不是按「腿」。** 一批两条腿的结构下,
按腿统计会把"止盈那一半"记成一个独立的胜 —— 一批只要摸到 2R 就白送一个胜,
胜率被系统性抬高。这里同时对着 v2.1 的原实现判等。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import CLEAN_DIR, apply_mode, prepare_base
from src.data.v3_timeframes import build_timeframes
from src.engine.pullback_v3_backtest import (
    END_SESSION, PROTECTIVE_SL, SL, TP, PBTrade, PullbackResult, run_pullback_v3_backtest,
)
from src.engine.portfolio_pullback_backtest import _batch_pnls as v21_batch_pnls
from src.engine.pullback_backtest import Trade as V21Trade
from src.performance.pullback_stats import (
    batch_pnls, compute_stats, format_stats, panel_rows, panel_subtitle, panel_title,
)
from src.strategy.pullback import PullbackConfig, PullbackParams

needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "RB.parquet")),
    reason="需要 data/v3.0/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)


def _leg(entry, direction, pnl, size, reason, exit_idx=None, cost=0.0) -> PBTrade:
    """给定目标 pnl_pct 反推出场价, 省得每个用例手算价格。"""
    px = 100.0
    exit_px = px * (1 + pnl) if direction == "long" else px * (1 - pnl)
    t0 = pd.Timestamp("2026-03-03 09:01")
    return PBTrade(
        direction=direction, entry_bar_idx=entry, entry_time=t0, entry_price=px,
        exit_bar_idx=exit_idx if exit_idx is not None else entry + 10,
        exit_time=t0 + pd.Timedelta(minutes=10), exit_price=exit_px,
        reason=reason, size_fraction=size, signal_type="open", gapped=False,
        stop_loss=px * 0.99, trading_date=pd.Timestamp("2026-03-03"),
        session_id=1, cost_bps=cost,
    )


def _result(trades, equity=None) -> PullbackResult:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-03-03 09:01") + pd.Timedelta(minutes=i) for i in range(3)])
    eq = pd.Series(equity if equity is not None else [1.0, 1.0, 1.0], index=idx)
    return PullbackResult(
        symbol="T", trades=trades, equity_curve=eq,
        trading_date=pd.Series([pd.Timestamp("2026-03-03")] * 3, index=idx),
        position=np.zeros(3, dtype=np.int8), stop_line=np.full(3, np.nan),
        pullback_points=[], cooldown_spans=[], warmup_bars=0, warmup_days=60,
        tradable_days=1, params=PullbackParams(), config=PullbackConfig(), mode="split",
    )


# ------------------------------------------------------------ 按批统计 ----
def test_batch_pnls_matches_the_v21_definition():
    """对着 `portfolio_pullback_backtest._batch_pnls` 逐值判等。"""
    v3 = [
        _leg(10, "long", 0.04, 0.5, TP), _leg(10, "long", -0.005, 0.5, PROTECTIVE_SL),
        _leg(50, "short", -0.02, 1.0, SL),
        _leg(90, "short", 0.03, 0.5, TP), _leg(90, "short", 0.01, 0.5, PROTECTIVE_SL),
    ]
    v21 = [V21Trade(direction=t.direction, entry_bar_idx=t.entry_bar_idx,
                    entry_price=t.entry_price, exit_bar_idx=t.exit_bar_idx,
                    exit_price=t.exit_price, reason=t.reason,
                    size_fraction=t.size_fraction, signal_type=t.signal_type)
           for t in v3]
    got, want = batch_pnls(v3), v21_batch_pnls(v21)
    assert set(got) == set(want)
    for k in want:
        assert got[k][0] == want[k][0]
        assert got[k][1] == pytest.approx(want[k][1], rel=1e-12)


def test_win_rate_is_per_batch_not_per_leg():
    """
    一批 = 止盈 +4% 一半 + 护盈 -0.5% 一半 -> 批收益 +1.75%, 算 1 胜。
    另一批整批止损 -2% -> 1 负。**按批**胜率 50%; 按腿会是 2/3 = 66.7%。
    """
    trades = [
        _leg(10, "long", 0.04, 0.5, TP), _leg(10, "long", -0.005, 0.5, PROTECTIVE_SL),
        _leg(50, "long", -0.02, 1.0, SL),
    ]
    s = compute_stats(_result(trades))
    assert s.n_batches == 2 and s.n_legs == 3
    assert s.win_rate == pytest.approx(0.5)
    assert s.n_tp == 1 and s.n_psl == 1 and s.n_sl == 1


def test_flat_batch_counts_as_a_loss():
    """仓库约定: 亏损桶是 `p <= 0`, 持平算亏。胜率与盈亏比因此互相自洽。"""
    trades = [_leg(10, "long", 0.0, 1.0, SL), _leg(50, "long", 0.02, 1.0, TP)]
    s = compute_stats(_result(trades))
    assert s.win_rate == pytest.approx(0.5), "持平那一批必须算负"
    # 亏损桶里只有一笔 0 -> 平均亏损为 0 -> 盈亏比无意义, 按 trade_stats 的约定返回 nan
    assert pd.isna(s.payoff)


def test_direction_split_counts_batches():
    trades = [
        _leg(10, "long", 0.01, 0.5, TP), _leg(10, "long", 0.01, 0.5, PROTECTIVE_SL),
        _leg(50, "short", -0.01, 1.0, SL),
    ]
    s = compute_stats(_result(trades))
    assert (s.n_batches, s.n_long, s.n_short) == (2, 1, 1)


def test_avg_bars_held_is_per_batch_to_the_last_leg():
    trades = [
        _leg(10, "long", 0.01, 0.5, TP, exit_idx=30),
        _leg(10, "long", 0.01, 0.5, PROTECTIVE_SL, exit_idx=110),
    ]
    s = compute_stats(_result(trades))
    assert s.avg_bars_held == pytest.approx(100.0)


def test_empty_result_renders_as_n_a():
    s = compute_stats(_result([]))
    rows = dict(panel_rows(s))
    assert rows["批次"] == "0 (多0/空0)"
    assert rows["胜率(按批)"] == "n/a" and rows["盈亏比(按批)"] == "n/a"
    assert rows["平均持有"] == "n/a"
    assert rows["离场 止损/止盈/护盈/收盘"] == "0 / 0 / 0 / 0"


# -------------------------------------------------------------- 面板 ----
def test_panel_shape():
    s = compute_stats(_result([_leg(10, "long", 0.01, 1.0, SL)]), mode_label="夜盘")
    rows = panel_rows(s)
    assert len(rows) == 9
    assert [k for k, _ in rows][:2] == ["可交易日", "批次"]
    assert "回调入场" in panel_title(s)
    assert "夜盘" in panel_subtitle(s)
    assert "MA5/20/50" in panel_subtitle(s)
    assert "回调入场" in format_stats(s)


def test_short_warmup_is_flagged_in_the_subtitle():
    """「一笔不做」和「没暖好」在屏幕上长得一样, 必须标出来。"""
    import dataclasses
    r = _result([])
    ok = compute_stats(r)
    assert not ok.warmup_short and "暖机不足" not in panel_subtitle(ok)
    short = compute_stats(dataclasses.replace(r, warmup_days=12))
    assert short.warmup_short and "暖机不足 12/50" in panel_subtitle(short)


def test_exit_counts_are_four_not_two():
    """
    R-Breaker 面板是「止损/收盘」两个数; 回调是「止损/止盈/护盈/收盘」四个。

    四个数是**腿**数不是批数 —— 这里第一批就贡献了两条腿 (2R 止盈那一根正好
    是时段末根, 剩下一半被强平), 所以四数之和是 4 而批次只有 2。
    """
    trades = [_leg(10, "long", 0.04, 0.5, TP), _leg(10, "long", 0.01, 0.5, END_SESSION),
              _leg(50, "long", -0.02, 1.0, SL)]
    s = compute_stats(_result(trades))
    row = dict(panel_rows(s))["离场 止损/止盈/护盈/收盘"]
    assert row == "1 / 1 / 0 / 1"
    assert s.n_batches == 2 and s.n_legs == 3


# --------------------------------------------------------- 真实数据 ----
@needs_data
def test_on_real_data():
    prep = apply_mode(prepare_base("RB", "2025-07-01", "2026-07-29", warmup_days=60), "split")
    res = run_pullback_v3_backtest(prep, build_timeframes(prep.df))
    s = compute_stats(res, mode_label="日夜分开")
    assert s.n_batches >= 20
    assert s.n_legs == s.n_sl + s.n_tp + s.n_psl + s.n_end
    # 锁死之后收盘平仓是主要出场方式 —— 少了这条, 上面那个等式换成三项也能过
    assert s.n_end > 0
    assert s.n_batches == s.n_long + s.n_short
    assert 0.0 <= s.win_rate <= 1.0
    assert s.max_drawdown <= 0.0
    assert not s.warmup_short
    # 按腿的胜率会明显更高 —— 这正是为什么要按批
    per_leg = sum(1 for t in res.trades if t.pnl_net > 0) / len(res.trades)
    assert per_leg > s.win_rate
    assert len(format_stats(s).splitlines()) == 10
