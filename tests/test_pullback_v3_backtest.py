# -*- coding: utf-8 -*-
"""
`src/engine/pullback_v3_backtest.py` 的不变量。

判等测试 (`test_pullback_v3_vs_v21.py`) 证明"和 v2.1 是同一个策略"; 这里证明的是
**在中国期货数据上**这个引擎自身立得住: 没有未来函数、仓位纪律不破、跳空只吃亏
不占便宜、暖机段不交易、成本只收一次。

用真实 parquet: 无未来函数这条只有在真实的时段断口 (10:15 小节休息、午休、
夜->日 6.5 小时、跨周末 58 小时) 上验才有意义, 合成帧造不出那种形状。
"""

import os

import dataclasses

import numpy as np
import pytest

from src.data.v3_sessions import CLEAN_DIR, apply_mode, prepare_base
from src.data.v3_timeframes import build_timeframes
from src.engine.pullback_v3_backtest import (
    END_SESSION, EXIT_REASONS, PROTECTIVE_SL, SL, TP, run_pullback_v3_backtest,
)
from src.strategy.pullback import PullbackConfig, PullbackParams

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "AG.parquet")),
    reason="需要 data/01-pkl层/一次排查/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)

# 为什么是 AG (沪银) 而不是 RB: 时段锁死之后持仓不再跨时段边界, 而**跳空只发生在
# 边界上** —— RB/split 一整年的跳空成交因此归零, `test_gap_fill_*` 两条和
# `test_each_switch_is_actually_wired[gap_fill_exits]` 全部退化成空真。
# AG 的夜盘开到 02:30, 一天三个断口 (10:15 / 午休 / 夜盘收盘), 锁死之后仍有 5 笔
# 真跳空, 四种出场原因也都出现。**换 fixture 是为了保住覆盖, 不是为了让测试变绿。**
SYMBOL, START, END, WARM = "AG", "2025-07-01", "2026-07-29", 60
P = PullbackParams()


@pytest.fixture(scope="module")
def prep():
    return apply_mode(prepare_base(SYMBOL, START, END, warmup_days=WARM), "split")


@pytest.fixture(scope="module")
def tfs(prep):
    return build_timeframes(prep.df)


@pytest.fixture(scope="module")
def res(prep, tfs):
    return run_pullback_v3_backtest(prep, tfs)


def _batches(trades) -> dict[int, list]:
    out: dict[int, list] = {}
    for t in trades:
        out.setdefault(t.entry_bar_idx, []).append(t)
    return out


def test_the_fixture_is_worth_asserting_on(res):
    """守卫: 下面每一条不变量在零笔交易上都恒真。"""
    assert len(_batches(res.trades)) >= 20
    assert set(t.reason for t in res.trades) == set(EXIT_REASONS)
    assert any(t.gapped for t in res.trades), "真实数据里应当有跳空成交"
    # 锁死之后收盘平仓通常是**最主要**的出场方式 —— 如果它不是, 说明 fixture 里
    # 的时段太长, 强平那条分支没被真正压到。
    assert sum(1 for t in res.trades if t.reason == END_SESSION) >= 10


# --------------------------------------------------------- 无未来函数 ----
@pytest.mark.parametrize("frac", [0.35, 0.6, 0.85])
def test_no_lookahead_by_truncation(prep, res, frac):
    """
    把 bar i 之后的价格全部改成垃圾重跑, 所有 `exit_bar_idx < i` 的交易必须**逐字节
    相同**。照抄 `test_rbreaker_backtest.py::test_no_lookahead_by_truncation`。

    只改价格、不动索引和 trading_date, 所以分箱不变 —— 于是这条测试真正检验的是
    "决策有没有读到未来的**价格**", 而不是"分箱有没有变"。
    """
    n = len(prep.df)
    i = int(n * frac)
    dirty = prep.df.copy()
    rng = np.random.default_rng(0)
    for col in ("open", "high", "low", "close"):
        v = dirty[col].to_numpy("float64").copy()
        v[i:] = v[i:] * 3.0 + rng.normal(0, 50, size=n - i)
        dirty[col] = v
    dirty["high"] = np.maximum(dirty["high"], np.maximum(dirty["open"], dirty["close"]))
    dirty["low"] = np.minimum(dirty["low"], np.minimum(dirty["open"], dirty["close"]))

    dirty_prep = dataclasses.replace(prep, df=dirty)
    got = run_pullback_v3_backtest(dirty_prep, build_timeframes(dirty))

    def before(trades):
        return [(t.direction, t.entry_bar_idx, t.entry_price, t.exit_bar_idx,
                 t.exit_price, t.reason, t.size_fraction, t.gapped)
                for t in trades if t.exit_bar_idx < i]

    assert before(got.trades) == before(res.trades)
    assert len(before(res.trades)) >= 5, "截断点太靠前, 这条测试没验到东西"


def test_trailing_stop_is_never_worse_than_the_original_stop(res):
    """
    吊灯线以原始止损位为地板 (`max(raw, floor_stop)`), 所以剩余仓位任何时刻的下行
    风险都不超过原始止损 —— 不存在"还没赚钱就被放飞导致亏损扩大"。

    注意这里**不能**断言"止损位单调只上移" —— 那是 take_profit.py docstring 的措辞,
    但实现并不保证: 吊灯线是 `极值 - 3 x ATR` 每根重算的, 极值不降而 ATR 上升时
    它会回落 (仍不低于地板)。这是 v2.1 的既有行为, 移植不改它, 只如实记下来。
    """
    stop, pos = res.stop_line, res.position
    for entry, legs in _batches(res.trades).items():
        floor = legs[0].stop_loss
        end = max(t.exit_bar_idx for t in legs)
        seg = stop[entry:end + 1]
        seg = seg[~np.isnan(seg)]
        if pos[entry] == 1:
            assert (seg >= floor - 1e-9).all(), f"多头止损跌破原始止损 @ {entry}"
        else:
            assert (seg <= floor + 1e-9).all(), f"空头止损升破原始止损 @ {entry}"


@pytest.mark.parametrize("field", [
    "entry_on_completed_bar", "gap_fill_exits", "range_based_exit_trigger",
    "risk_atr_on_30m", "cooldown_counts_daily_bars", "session_forces_flat",
    "cooldown_on_structure_break",
])
def test_each_switch_is_actually_wired(prep, tfs, res, field):
    """
    每个开关关掉后结果必须变 —— 抓"加了字段但引擎根本没读"这类哑火。

    刻意**不**包含三个开关, 理由是它们在 1 分钟真实数据上不 bite, 不是没接上:
      * `pessimistic_both_hit` / `trail_extreme_prev_bar` —— 都要求两个事件落在
        **同一根** bar 上 (止损与止盈同根 / 新极值与触发同根)。1 分钟粒度下几乎
        不发生; 实测 RB 一整年零次。
      * `session_gates_entry` —— split 模式下 `tradable` 全 True, 自然无差别。
        它由 `test_pullback_v3_sessions.py` 在 night/day 模式下验。
    这三个的正确性由 `test_pullback_v3_vs_v21.py` 的八格配置矩阵与 v2.1 逐笔判等
    保证 —— 那份 5 分钟 fixture 的 bar 更粗, 同根事件真实发生。
    """
    flipped = run_pullback_v3_backtest(
        prep, tfs, config=PullbackConfig(**{field: False}))
    a = [(t.entry_bar_idx, t.exit_bar_idx, t.exit_price, t.reason) for t in res.trades]
    b = [(t.entry_bar_idx, t.exit_bar_idx, t.exit_price, t.reason) for t in flipped.trades]
    assert a != b, f"{field} 关掉后结果没变 —— 引擎可能根本没读这个字段"


# ------------------------------------------------------------ 仓位纪律 ----
def test_only_four_exit_reasons(res):
    assert {t.reason for t in res.trades} <= set(EXIT_REASONS)


def test_no_session_last_bar_ever_carries_a_position(prep, res):
    """
    锁死最强的那条机器不变量: **每一根时段末根上仓位都是 0**。

    两条路都通向它 —— 强平那一根在 mark-to-market 之前就把 batch 置了 None
    (于是 `pos_arr[i]` 保持 0), 而时段末根本来就不许开仓。合起来就是"时段结束的
    那一刻必然空仓", 也就是用户要的"未选中时段完全空仓"。

    注意**不能**断言"最后一条腿落在帧末根上": 末根空仓的常见原因是那之前根本没
    开仓, 不是强平失败。
    """
    last = prep.df["is_session_last"].to_numpy()
    assert last.any() and last[-1], "帧末根的 is_session_last 应恒为 True"
    assert (res.position[last] == 0).all(), "有时段末根仍持着仓位"
    assert res.position[-1] == 0


def test_one_batch_at_a_time(res):
    """一批没走完不许开下一批 —— 仓位管理不会出现分数级叠加。"""
    spans = sorted((e, max(t.exit_bar_idx for t in legs))
                   for e, legs in _batches(res.trades).items())
    for (e1, x1), (e2, _) in zip(spans, spans[1:]):
        assert e2 > x1, f"批次 {e1}-{x1} 与 {e2} 重叠"


def test_batch_legs_sum_to_one(res):
    """
    锁死之后一批只有**四种**收尾形状, 每一种的腿都恰好加到 1.0:

        [SL]                  整批一次止损
        [TP, PROTECTIVE_SL]   止盈一半 + 吊灯打掉剩下一半
        [END_SESSION]         没走到任何出场位, 整批被时段末强平
        [END_SESSION, TP]     止盈一半, 剩下一半被强平 (用户点名的那个场景)

    没有第五种: SL 和 PROTECTIVE_SL 都会把 `batch` 置 None, 所以它们不可能与
    END_SESSION 同批出现。腿和恒等于 1 是**载重不变量** —— `PBTrade.pnl_net` 的
    成本口径和 `pullback_stats.batch_pnls` 都建立在它上面。
    """
    seen = set()
    for entry, legs in _batches(res.trades).items():
        reasons = sorted(t.reason for t in legs)
        assert sum(t.size_fraction for t in legs) == pytest.approx(1.0), entry
        assert reasons in ([SL], [END_SESSION], [END_SESSION, TP],
                           sorted([TP, PROTECTIVE_SL])), (entry, reasons)
        seen.add(tuple(reasons))
    # 四种形状在这份 fixture 上都要真实出现过, 否则上面那条断言是空真
    assert len(seen) == 4, f"只出现了 {sorted(seen)}"


def test_tp_leg_fills_exactly_at_its_trigger(res):
    """止盈是挂在 2R 的限价单 —— 不做跳空补价, 成交价必须**恰好**等于触发价。"""
    from src.strategy.pullback import partial_trigger_price
    for t in res.trades:
        if t.reason != TP:
            continue
        want = partial_trigger_price(
            t.entry_price, t.direction == "long", t.stop_loss, P)
        assert t.exit_price == pytest.approx(want, rel=1e-12)
        assert not t.gapped


def test_exit_never_precedes_entry(res):
    for t in res.trades:
        assert t.exit_bar_idx >= t.entry_bar_idx
        assert t.exit_time >= t.entry_time


# ------------------------------------------------------------ 跳空补价 ----
def test_gap_fill_only_ever_makes_the_fill_worse(prep, tfs, res):
    """
    "止损吃亏, 止盈不占便宜": 打开跳空补价之后, 止损/护盈的成交价只可能比线更差,
    绝不可能更好; 止盈那一条腿一根手指都不许碰。
    """
    off = run_pullback_v3_backtest(prep, tfs, config=PullbackConfig(gap_fill_exits=False))
    # 关掉之后决策链不变(补价只影响成交价, 不影响触发), 所以两边逐笔一一对应
    assert [(t.entry_bar_idx, t.exit_bar_idx, t.reason) for t in off.trades] == \
           [(t.entry_bar_idx, t.exit_bar_idx, t.reason) for t in res.trades]
    n_worse = 0
    for a, b in zip(res.trades, off.trades):
        # 止盈是限价单; 时段强平按 close 成交。两者都与跳空补价无关, 价格必须逐位相同。
        if a.reason in (TP, END_SESSION):
            assert a.exit_price == b.exit_price
            assert not a.gapped
            continue
        if a.direction == "long":
            assert a.exit_price <= b.exit_price + 1e-12
        else:
            assert a.exit_price >= b.exit_price - 1e-12
        n_worse += int(a.exit_price != b.exit_price)
        assert a.gapped == (a.exit_price != b.exit_price)
    assert n_worse >= 1, "没有一笔真的跳空 -> 这条测试没验到东西"


def test_gapped_flag_matches_the_fill(res):
    for t in res.trades:
        if t.reason == TP:
            assert not t.gapped


# -------------------------------------------------------------- 暖机 ----
def test_no_trades_and_flat_equity_in_the_warmup_stretch(prep, res):
    assert prep.warmup_bars > 0
    assert min(t.entry_bar_idx for t in res.trades) >= prep.warmup_bars
    assert (res.position[: prep.warmup_bars] == 0).all()
    assert np.isnan(res.stop_line[: prep.warmup_bars]).all()
    # 权益曲线切掉暖机段, 起点归 1.0
    assert len(res.equity_curve) == len(prep.df) - prep.warmup_bars
    assert res.equity_curve.iloc[0] == pytest.approx(1.0)


def test_state_machine_is_already_warm_at_the_first_display_bar(prep, tfs):
    """
    暖机 padding 存在的全部意义: 显示窗口第一根 bar 上, 日线 MA50 已经算得出来,
    趋势方向不是"还在等暖机"的 neutral。
    """
    from src.strategy.pullback import ma_array_states
    d = tfs["1d"]
    states = ma_array_states(d.bars["close"], P.ma_fast, P.ma_mid, P.ma_slow)
    pos_at_first = int(d.closed_pos[prep.warmup_bars])
    assert pos_at_first >= P.ma_slow - 1
    assert not np.isnan(states[pos_at_first])


# -------------------------------------------------------------- 成本 ----
def test_cost_is_charged_once_per_batch_not_once_per_leg(prep, tfs, res):
    """
    一批两条腿, 按腿各扣一次会重复收开仓费。约定是
    `pnl_net = pnl_pct - cost/1e4`, 而批收益是 `Σ size_fraction * pnl_net`,
    于是成本项恰好是 `1.0 x cost`。
    """
    bps = 10.0
    with_cost = run_pullback_v3_backtest(prep, tfs, config=PullbackConfig(cost_bps=bps))
    # 成本不进任何决策 —— 逐笔必须完全一样
    assert [(t.entry_bar_idx, t.exit_bar_idx, t.reason, t.exit_price) for t in with_cost.trades] == \
           [(t.entry_bar_idx, t.exit_bar_idx, t.reason, t.exit_price) for t in res.trades]

    def batch_pnl(r):
        out = {}
        for t in r.trades:
            out[t.entry_bar_idx] = out.get(t.entry_bar_idx, 0.0) + t.size_fraction * t.pnl_net
        return out

    a, b = batch_pnl(res), batch_pnl(with_cost)
    for k in a:
        assert a[k] - b[k] == pytest.approx(bps / 1e4, abs=1e-12), k


# -------------------------------------------------------------- 边界 ----
def test_empty_frame_returns_an_empty_result(prep):
    empty = dataclasses.replace(prep, df=prep.df.iloc[:0], warmup_bars=0)
    r = run_pullback_v3_backtest(empty)
    assert r.trades == [] and len(r.equity_curve) == 0 and r.tradable_days == 0


def test_without_the_lock_an_open_batch_at_the_frame_end_produces_no_trade(prep, tfs):
    """
    `session_forces_flat=False` 那条路径的收尾语义 (移植初版, 也是 v2.1 的行为):
    没有强平 => 窗口末根可能仍持仓, 那一批不产生 PBTrade, 面板的「批次」因此可能
    比图上最后一个入场箭头少一个。

    锁死打开时相反 —— 见 `test_end_session_legs_are_flat_by_the_last_bar`。
    这两条一起证明那个开关**两个方向**都真的接上了。
    """
    off = run_pullback_v3_backtest(
        prep, tfs, config=PullbackConfig(session_forces_flat=False))
    last_leg_exit = max(t.exit_bar_idx for t in off.trades)
    still_open = off.position[last_leg_exit + 1:].any()
    if still_open:
        assert max(t.entry_bar_idx for t in off.trades) < len(prep.df) - 1
    assert END_SESSION not in {t.reason for t in off.trades}


def test_protective_stop_can_never_change_the_cooldown_counter(res):
    """
    护盈止损对冷静期计数器"什么也不做"是安全的 —— 因为它**根本没有机会**做什么:
    走到 PSL 必然已经 TP 过 (partial_taken 只由 TP 置位), 而 TP 刚把计数清零,
    两条腿之间同一批一直持着仓, 没有别的批次能插进来改计数器。

    这里把那个结构前提钉死: 任何 TP+PSL 批次的两条腿之间, 不存在别的批次的腿。
    (59 品种 x 5 模式实测 594 个这样的批次, 零个反例。)

    计数器本身的语义在 `tests/test_pullback_v3_cooldown.py` 上用手搭的帧验 ——
    真实数据上验不了, 冷静期一 arm 就挡住后续入场, 两条时间线立刻整个岔开。
    """
    pairs = 0
    for entry, legs in _batches(res.trades).items():
        if {t.reason for t in legs} != {TP, PROTECTIVE_SL}:
            continue
        pairs += 1
        tp = next(t for t in legs if t.reason == TP)
        psl = next(t for t in legs if t.reason == PROTECTIVE_SL)
        between = [t for t in res.trades
                   if t.entry_bar_idx != entry
                   and tp.exit_bar_idx <= t.exit_bar_idx <= psl.exit_bar_idx]
        assert not between, f"批 {entry} 的 TP 与 PSL 之间夹进了别的批次"
    assert pairs >= 3, "这份 fixture 里没有 TP+PSL 批次, 断言成了空真"
