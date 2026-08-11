# -*- coding: utf-8 -*-
"""
五种时段模式对回调策略意味着什么 —— **时段锁死**。

约定是两个正交的开关合起来生效:
  `session_gates_entry`  不可交易时段不开仓 (外加时段末根不开仓)
  `session_forces_flat`  时段最后一根 1m bar 的 close 强制平仓

于是「未选中时段完全空仓且无交易」。这里把它的可观测后果钉死, 其中最重要的一条是
**五种模式两两不同** —— 这与锁死之前恰好相反 (那时 split / night_day / day_night
逐字节相同), 所以两个方向都留了测试: 默认路径验锁死, `session_forces_flat=False`
的路径守住旧行为, 免得开关退化成惰性的。
"""

import os

import numpy as np
import pytest

from src.data.v3_sessions import (
    CLEAN_DIR, SESSION_MODES, apply_mode, derive_segments, prepare_base, segment_is_night,
)
from src.data.v3_timeframes import build_timeframes
from src.engine.pullback_v3_backtest import (
    END_SESSION, EXIT_REASONS, run_pullback_v3_backtest,
)
from src.strategy.pullback import PullbackConfig, PullbackParams

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "RB.parquet")),
    reason="需要 data/v3.0/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)

START, END, WARM = "2025-07-01", "2026-07-29", 60


@pytest.fixture(scope="module")
def rb_base():
    return prepare_base("RB", START, END, warmup_days=WARM)


def _run(base, mode, **cfg):
    prep = apply_mode(base, mode)
    return prep, run_pullback_v3_backtest(
        prep, build_timeframes(prep.df), config=PullbackConfig(**cfg))


def _key(r):
    return [(t.entry_bar_idx, t.exit_bar_idx, t.exit_price, t.reason) for t in r.trades]


def test_all_five_modes_now_produce_different_results(rb_base):
    """
    **五种模式两两不同。** 锁死之前不是这样: 那时只有 `night` / `day` 会把
    `tradable` 置 False, 另外三种的区别**只在时段如何分组**, 而当时没有任何逻辑
    读分组, 于是 split / night_day / day_night 逐字节相同。

    强平读 `is_session_last`, 而它直接由分组算出 (`v3_sessions.py:340`) —— 分组
    第一次变成可观测的东西, 三者立刻分叉。
    """
    keys = {m: _key(_run(rb_base, m)[1]) for m in SESSION_MODES}
    assert all(keys.values()), "某个模式一笔都没有 —— 这个断言就成了空真"
    seen = {}
    for m, k in keys.items():
        dup = seen.get(str(k))
        assert dup is None, f"{m} 与 {dup} 结果相同 —— 锁死没有让分组变成可观测的"
        seen[str(k)] = m


def test_without_the_lock_three_modes_still_collapse(rb_base):
    """
    反向守卫: 关掉强平, 上一条测的那个差异必须**消失**。

    没有这条的话, 上面那个测试无法区分"锁死起作用了"和"这三种模式本来就不同"。
    """
    keys = {m: _key(_run(rb_base, m, session_forces_flat=False)[1]) for m in SESSION_MODES}
    assert keys["split"] == keys["night_day"] == keys["day_night"]
    assert keys["night"] != keys["split"] and keys["day"] != keys["split"]


def test_night_mode_only_enters_on_night_bars(rb_base):
    prep, res = _run(rb_base, "night")
    seg = derive_segments(prep.df)
    is_night = segment_is_night(prep.df, seg)
    assert res.trades
    for t in res.trades:
        assert is_night[t.entry_bar_idx], f"夜盘模式却在白盘开仓 @ {t.entry_bar_idx}"


@pytest.mark.parametrize("mode", list(SESSION_MODES))
def test_no_position_ever_crosses_a_session_boundary(rb_base, mode):
    """
    锁死的**核心不变量**: 每一条腿的出场与它那一批的入场落在**同一个时段**里。

    这是"未选中时段完全空仓"的机器版本 —— 只要有一笔跨了时段, 就说明它在某个
    不该持仓的时刻还拿着仓位。
    """
    prep, res = _run(rb_base, mode)
    sid = prep.df["session_id"].to_numpy()
    assert res.trades, f"{mode}: 一笔都没有, 断言成了空真"
    for t in res.trades:
        assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx], \
            f"{mode}: 持仓跨了时段 {t.entry_bar_idx}->{t.exit_bar_idx}"
    assert res.position[-1] == 0, f"{mode}: 收尾还留着仓位"


def test_forced_flat_fills_at_the_session_last_close(rb_base):
    """
    强平的三个机械细节, 与 `rbreaker_backtest.py:203-206` 逐字对齐:
      * 成交价 == 时段末根的 **close** (不是 open, 不做跳空补价)
      * 那一根确实带 `is_session_last`
      * `gapped` 恒为 False
    """
    prep, res = _run(rb_base, "day")
    df = prep.df
    cl = df["close"].to_numpy()
    last = df["is_session_last"].to_numpy()
    ends = [t for t in res.trades if t.reason == END_SESSION]
    assert ends, "日盘模式一次强平都没有 —— 不可能"
    for t in ends:
        assert last[t.exit_bar_idx], "强平那一根不是时段末根"
        assert t.exit_price == pytest.approx(cl[t.exit_bar_idx]), "强平没按 close 成交"
        assert t.gapped is False, "强平不该标成跳空成交"


def test_forced_flat_never_produces_a_zero_bar_leg(rb_base):
    """
    时段末根**不开仓**的凭据。少了那道闸门, 时段末根上成交的那一批会在同一根被
    强平, 留下一条 `exit_bar_idx == entry_bar_idx` 的腿 —— 一根 bar 既开又平。
    """
    for mode in SESSION_MODES:
        _, res = _run(rb_base, mode)
        for t in res.trades:
            if t.reason == END_SESSION:
                assert t.exit_bar_idx > t.entry_bar_idx, f"{mode}: 0 根的强平腿"


def test_forced_flat_keeps_batch_fractions_summing_to_one(rb_base):
    """
    用户点名的那个场景: 2R 止盈那一根**正好**是时段末根时, 这一批要产生两条腿
    (TP 50% + END_SESSION 50%), 而不是把整仓平掉 (那会让腿和变成 1.5)。

    `PBTrade.pnl_net` 的成本口径和 `pullback_stats.batch_pnls` 都依赖腿和恒等于 1。
    """
    p = PullbackParams()
    both = 0
    for mode in SESSION_MODES:
        _, res = _run(rb_base, mode)
        batches: dict[int, list] = {}
        for t in res.trades:
            batches.setdefault(t.entry_bar_idx, []).append(t)
        for entry, legs in batches.items():
            assert sum(x.size_fraction for x in legs) == pytest.approx(1.0), \
                f"{mode}/{entry}: 腿和不是 1"
            reasons = {x.reason for x in legs}
            if reasons == {"TP", END_SESSION}:
                both += 1
                end_leg = next(x for x in legs if x.reason == END_SESSION)
                assert end_leg.size_fraction == pytest.approx(1.0 - p.partial_ratio)
    assert both, "没有任何一批走到 TP+强平 —— 这条断言成了空真"


def test_without_the_lock_positions_do_cross_the_session_close(rb_base):
    """
    反向守卫: 关掉强平, 夜盘开的仓必须能持有穿过夜盘收盘跑进白盘。
    这是移植初版的行为, 也是 `session_forces_flat=False` 唯一的用处。
    """
    prep, res = _run(rb_base, "night", session_forces_flat=False)
    seg = derive_segments(prep.df)
    is_night = segment_is_night(prep.df, seg)
    crossed = [t for t in res.trades
               if is_night[t.entry_bar_idx] and not is_night[t.exit_bar_idx]]
    assert crossed, "一笔都没跨过夜盘收盘 —— 那这个开关就没被验到"
    assert END_SESSION not in {t.reason for t in res.trades}
    assert {t.reason for t in res.trades} <= set(EXIT_REASONS)
    assert any(seg[t.exit_bar_idx] > seg[t.entry_bar_idx] for t in res.trades)


def test_the_gate_reduces_trading(rb_base):
    """
    闸门确实在起作用: 夜盘模式的成交数严格少于关掉闸门的同一段数据。

    这里**只**断言这一条。两个更强的说法都是错的, 记下来免得日后有人加回去:
      * "回调确认点集合不变" —— `attempt_entry` 只在空仓时被调用 (v2.1 行为),
        被挡住 -> 一直空仓 -> 继续记录后面的确认; 不挡则已进场、状态机冻结。
        两边必然岔开。
      * "冷静期区间不变" —— 冷静期的触发条件之一是"同方向连续 3 次纯止损",
        而止损来自成交。闸门改了成交, 冷静期自然跟着改。

    "被挡的触发不消耗 `pullback_seen`" 这条真正的不变量由
    `tests/test_pullback_v3_entry_gate.py` 用手搭的状态序列精确验证。
    """
    _, night = _run(rb_base, "night")
    _, free = _run(rb_base, "night", session_gates_entry=False)
    assert len(night.trades) < len(free.trades)


def test_both_switches_off_collapses_all_five_modes(rb_base):
    """
    **两个**开关都关掉 -> 时段模式对这个策略完全无影响, 五种结果合一。

    必须两个都关: 只关入场闸门的话强平仍读 `is_session_last`, 五种模式照样不同。
    这条同时证明了这两个开关是时段模式**仅有**的两个入口。
    """
    keys = {m: _key(_run(rb_base, m, session_gates_entry=False,
                         session_forces_flat=False)[1])
            for m in SESSION_MODES}
    assert len(set(map(str, keys.values()))) == 1


def test_entry_gate_alone_does_not_collapse_the_modes(rb_base):
    """只关入场闸门不够 —— 强平还在读分组。"""
    keys = {m: _key(_run(rb_base, m, session_gates_entry=False)[1]) for m in SESSION_MODES}
    assert len(set(map(str, keys.values()))) > 1


def test_no_night_symbol_in_night_mode_does_nothing():
    """
    UR (尿素) 没有夜盘。夜盘模式下必须如实报 `可交易日 0` + 零成交, 而不是假装
    有 258 天可交易却一笔不做。
    """
    base = prepare_base("UR", START, END, warmup_days=WARM)
    prep, res = _run(base, "night")
    assert not prep.df.empty
    assert res.tradable_days == 0
    assert res.trades == []
    # 但白盘模式下照常工作
    _, day = _run(base, "day")
    assert day.tradable_days > 200 and day.trades


def test_tradable_days_counts_the_display_window_only(rb_base):
    """
    夜盘模式下 RB 的可交易日比 split 少几天 —— 那几天是节后首日, 交易所本就不开
    夜盘 (与 R-Breaker 时段模式方案记录的现象一致)。
    """
    _, split = _run(rb_base, "split")
    _, night = _run(rb_base, "night")
    assert night.tradable_days < split.tradable_days
    assert split.tradable_days - night.tradable_days <= 15


def test_warmup_bars_are_never_enterable_in_any_mode(rb_base):
    for mode in SESSION_MODES:
        prep, res = _run(rb_base, mode)
        assert prep.warmup_bars > 0
        assert (res.position[: prep.warmup_bars] == 0).all(), mode
        if res.trades:
            assert min(t.entry_bar_idx for t in res.trades) >= prep.warmup_bars, mode


def test_higher_timeframe_bars_never_span_a_segment_in_any_mode(rb_base):
    """
    分箱永远用**原子段**, 与时段模式无关 —— 否则 night_day / day_night 会画出一根
    跨 6 小时空白的 30m K 线。
    """
    seg = derive_segments(rb_base.df)
    for mode in SESSION_MODES:
        prep = apply_mode(rb_base, mode)
        tfs = build_timeframes(prep.df, seg)
        for tf in ("5m", "30m"):
            b = tfs[tf].bin_of
            edge = np.r_[True, b[1:] != b[:-1]]
            first = np.flatnonzero(edge)
            last = np.r_[first[1:] - 1, len(b) - 1]
            assert (seg[first] == seg[last]).all(), f"{mode}/{tf} 有 bar 跨段"
