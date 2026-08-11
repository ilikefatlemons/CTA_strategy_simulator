# -*- coding: utf-8 -*-
"""
**判等主测**: v3.0 引擎 vs **未改动的** v2.1 `run_pullback_backtest`。

前面 `test_pullback_v3_states.py` 证明了每个**原语**等价; 这里证明把它们组装成
引擎之后**整条链**也等价 —— 判定顺序、状态机推进、止损/止盈/吊灯的先后、跳空补价、
冷静期的 arm/release, 一笔都不能差。

做法: 造一份 v2.1 口径的 5 分钟美股帧, 让两个引擎看到**逐字节相同的四层已收盘
K 线序列**。v3 引擎的 `tf` 是参数正是为此 —— 这里直接用 v2.1 自己的
`resample_ohlcv` / `closed_bar_positions` 搭出 1:3:6:24 的结构 (而不是真实的
1:5:30:1d), 于是"周期比例"这个变量被消掉, 剩下的差异只可能来自逻辑。

两边的差异只允许有四处, 都在测试里显式中和掉:
  * 资金基数 10_000 (v2.1) vs 1.0 (v3.0) —— 只比交易, 权益比**比值**
  * v3 的 `tradable` / `in_window` 闸门 —— 全 True
  * v3 的 `cost_bps` —— 0
  * v3 的**时段锁死** `session_forces_flat` —— **关掉** (见 `_pb` 下面那段)

---------------------------------------------------------------------------
这个文件证明的是什么, 不证明什么
---------------------------------------------------------------------------
它证明**移植没把逻辑改坏**。它**不**证明"线上跑的那套等于 v2.1" —— 线上默认开着
时段锁死 (`session_forces_flat=True`, 时段末根按 close 强平), 而判等基准
`run_pullback_backtest` 根本没有这个概念。再叠上周期重映射把时间常数拉长约 12 倍,
线上那套与 v2.1 的账本**完全不可比**。锁死本身的正确性由
`tests/test_pullback_v3_sessions.py` 钉。
"""

import numpy as np
import pandas as pd
import pytest

from src.config import BacktestConfig
from src.data.resample import closed_bar_positions, resample_ohlcv
from src.data.v3_sessions import Prepared
from src.data.v3_timeframes import TFBars
from src.engine.pullback_backtest import run_pullback_backtest
from src.engine.pullback_v3_backtest import run_pullback_v3_backtest
from src.strategy.pullback import PullbackConfig, PullbackParams

# v2.1 的三个高周期。base 是 5m。
_RULES = {"15m": "15min", "30m": "30min", "2h": "2h"}
# v3 引擎的四个槽位 <- v2.1 的四个周期。角色一一对应。
_SLOT = {"1m": None, "5m": "15m", "30m": "30m", "1d": "2h"}

CAL_DAYS = 90          # ~64 个交易日 x 78 根 = ~5000 根 5m
# 配置矩阵用的那一个种子。选它是因为三种出场都够多 (SL 5 / TP 7 / PSL 7),
# 回调确认 15 次、冷静期 arm 5 次 —— 判等因此覆盖到每一条分支而不是空真。
MATRIX_SEED = 3
AUX_SEEDS = [3, 9]


def _rth_5m(n_cal_days: int, seed: int, noise: float = 0.55,
            span_sd: float = 1.00, wave_a: float = 0.35,
            gap_sd: float = 2.50) -> pd.DataFrame:
    """
    美股 RTH 交易日的 5 分钟 bar (09:30-16:00 = 78 根/天), 跳过周末。

    价格路径是**调出来的**, 不是随便一条随机游走: 慢趋势(周期 5000 根, 让 2h
    的三线严格排列能站住) + 中周期回调(周期 320 根 ~ 13 根 2h bar, 足够把 30m
    的快中线打反) + 大噪音 + 宽振幅 (`span_sd=1.0`, 让盘中触及止损线成为常事)。

    这组参数是网格搜出来的, 目标只有一个: 让 **SL / TP / PROTECTIVE_SL 三种出场**、
    回调确认、冷静期 arm/release 全部在这段数据里真实发生过。默认那条平滑路径几乎
    只走 TP -> 吊灯, 纯止损一次都不出 —— 那样"判等"就是在拿两个空列表比大小。
    `test_the_fixture_actually_trades` 守着这一点。
    """
    rng = np.random.default_rng(seed)
    stamps, opens, closes, px = [], [], [], 100.0
    day = pd.Timestamp("2024-07-01 09:30", tz="America/New_York")
    k = 0
    for d in range(n_cal_days):
        start = day + pd.Timedelta(days=d)
        if start.weekday() >= 5:
            continue
        for j in range(78):
            stamps.append(start + pd.Timedelta(minutes=5 * j))
            # 隔夜跳空: 每个交易日的第一根 open 相对昨收跳一段。没有这一段的话
            # `open[i] == close[i-1]` 恒成立, 价格永远不会跳过止损线, gap_fill
            # 那条分支就一次都走不到 —— 而它正是 v2.1 里最贵的一个修正
            # (docs/v2.1: -20.42pp / -3.60 Sharpe)。
            op = px + (rng.normal(0, gap_sd) if (j == 0 and k > 0) else 0.0)
            px = op + (0.15 * np.sin(2 * np.pi * k / 5000.0)   # 慢趋势
                       + wave_a * np.sin(2 * np.pi * k / 320.0)  # 中周期回调
                       + rng.normal(0, noise))
            opens.append(op)
            closes.append(px)
            k += 1
    o, c = np.asarray(opens), np.asarray(closes)
    span = np.abs(rng.normal(0, span_sd, size=c.size)) + 0.02
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(stamps),
        "open": o, "high": np.maximum(o, c) + span, "low": np.minimum(o, c) - span,
        "close": c, "volume": np.full(c.size, 1000.0),
    })


def _as_v3(df_5m: pd.DataFrame) -> tuple[Prepared, dict[str, TFBars]]:
    """
    把 v2.1 的 5m 帧包装成 v3 引擎能吃的 (Prepared, TFBars)。

    `TFBars.bars` / `closed_pos` 直接取自 v2.1 自己的 `resample_ohlcv` 和
    `closed_bar_positions`, 所以两个引擎读到的高周期序列**是同一份数据**。
    `bin_of` 只有图层用, 引擎不读, 填恒等即可。
    """
    n = len(df_5m)
    df = df_5m.set_index(pd.DatetimeIndex(df_5m["timestamp"], name="datetime"))
    df = df.drop(columns=["timestamp"])
    df["trading_date"] = pd.DatetimeIndex(
        df.index.tz_convert("America/New_York").date)  # type: ignore[union-attr]
    df["session_id"] = pd.factorize(df["trading_date"])[0]
    df["is_session_first"] = ~df["trading_date"].duplicated(keep="first")
    df["is_session_last"] = ~df["trading_date"].duplicated(keep="last")
    df["tradable"] = True

    resampled = {tf: resample_ohlcv(df_5m, rule) for tf, rule in _RULES.items()}
    tf: dict[str, TFBars] = {}
    for slot, src in _SLOT.items():
        if src is None:                      # base: 每根自成一箱, closed_pos = i-1
            bars = df[["open", "high", "low", "close", "volume"]]
            closed = np.arange(n, dtype=np.int64) - 1
        else:
            bars = resampled[src].set_index(
                pd.DatetimeIndex(resampled[src]["timestamp"], name="datetime"))
            closed = np.asarray(
                closed_bar_positions(resampled[src], _RULES[src], df_5m["timestamp"]),
                dtype=np.int64)
        tf[slot] = TFBars(tf=slot, bars=bars, bin_of=np.arange(n, dtype=np.int64),
                          closed_pos=closed)

    prep = Prepared(
        symbol="TEST", df=df, lines=pd.DataFrame(),
        line_idx=np.zeros(n, dtype=np.int32), k0=1.0,
        params=None, config=None, mode="split",  # type: ignore[arg-type]
    )
    return prep, tf


def _legs(trades) -> list[tuple]:
    return [
        (t.direction, int(t.entry_bar_idx), round(float(t.entry_price), 10),
         int(t.exit_bar_idx), round(float(t.exit_price), 10), t.reason,
         round(float(t.size_fraction), 10), t.signal_type, bool(t.gapped))
        for t in trades
    ]


def _pb(**kw) -> PullbackConfig:
    """
    判等用的 v3 配置: **一律关掉时段锁死**。

    判等基准是未改动的 `run_pullback_backtest`, 它没有"时段末强平"这个概念 ——
    锁死开着的话每个 `is_session_last` 都会多平一次仓, 八个用例必然全红。这不是
    把测试将就着改绿, 而是这个文件本来就只负责"两个引擎在**同一套规则**下逐笔
    相同"; 锁死是 v3.0 独有的新规则, 归 `test_pullback_v3_sessions.py` 管。

    `_as_v3` 造的帧里 `is_session_last` 是按 trading_date 生成的 (每天末根为 True),
    所以这个开关在这里绝不是惰性的 —— 不关掉就会真的每天强平。
    """
    return PullbackConfig(session_forces_flat=False, **kw)


CASES = [
    ("默认口径", BacktestConfig(), _pb()),
    ("ATR 用 base 周期 (A6 关)",
     BacktestConfig(risk_atr_on_30m=False), _pb(risk_atr_on_30m=False)),
    ("不做跳空补价 (pre-2.1)",
     BacktestConfig(gap_fill_exits=False), _pb(gap_fill_exits=False)),
    ("只看收盘触发 (pre-2.1)",
     BacktestConfig(range_based_exit_trigger=False),
     _pb(range_based_exit_trigger=False)),
    ("双触时乐观",
     BacktestConfig(pessimistic_both_hit=False), _pb(pessimistic_both_hit=False)),
    ("极值提前折入 (pre-A3)",
     BacktestConfig(trail_extreme_prev_bar=False),
     _pb(trail_extreme_prev_bar=False)),
    ("冷静期按调用次数 (pre-A5)",
     BacktestConfig(cooldown_counts_2h_bars=False),
     _pb(cooldown_counts_daily_bars=False)),
    ("成形根口径 (pre-2.1 入场时点)",
     BacktestConfig(entry_on_completed_bar=False),
     _pb(entry_on_completed_bar=False)),
]


@pytest.mark.parametrize("label,cfg21,cfg3", CASES, ids=[c[0] for c in CASES])
def test_engines_agree_leg_by_leg(label, cfg21, cfg3):
    """
    八种开关组合各跑一遍。**只用一个种子** —— v2.1 引擎是 O(n^2), 5000 根一趟约
    3 秒, 全矩阵再乘种子会把整个测试套件从 2 秒拖到两分钟。种子维度由下面三个
    辅助测试覆盖 (默认口径 x 两个种子)。
    """
    df_5m = _rth_5m(CAL_DAYS, MATRIX_SEED)
    want = run_pullback_backtest(df_5m, initial_capital=10_000.0, config=cfg21)
    prep, tf = _as_v3(df_5m)
    got = run_pullback_v3_backtest(prep, tf, PullbackParams(), cfg3)
    assert _legs(got.trades) == _legs(want.trades), label
    assert len(got.trades) >= 5, f"{label}: 这组开关下没交易, 判等等于空真"


@pytest.mark.parametrize("seed", AUX_SEEDS)
def test_observability_streams_agree(seed):
    """
    「额外注释」画的两样东西 —— 回调确认点和冷静期区间 —— 也必须逐条相同, 否则
    图上标出来的和引擎实际经历的不是一回事。
    """
    df_5m = _rth_5m(CAL_DAYS, seed)
    want = run_pullback_backtest(df_5m, initial_capital=10_000.0)
    prep, tf = _as_v3(df_5m)
    got = run_pullback_v3_backtest(prep, tf, PullbackParams(), _pb())
    assert [(p["bar_idx"], p["direction"]) for p in got.pullback_points] == \
           [(p["bar_idx"], p["direction"]) for p in want.pullback_points]
    assert got.cooldown_spans == want.cooldown_spans


@pytest.mark.parametrize("seed", AUX_SEEDS)
def test_equity_curves_agree_up_to_the_capital_base(seed):
    """v2.1 起点 10_000, v3.0 起点 1.0 —— 比**比值**。"""
    df_5m = _rth_5m(CAL_DAYS, seed)
    want = run_pullback_backtest(df_5m, initial_capital=10_000.0)
    prep, tf = _as_v3(df_5m)
    got = run_pullback_v3_backtest(prep, tf, PullbackParams(), _pb())
    np.testing.assert_allclose(
        got.equity_curve.to_numpy(), want.equity_curve.to_numpy() / 10_000.0,
        rtol=1e-12, atol=1e-12)


def test_the_fixture_actually_trades():
    """
    守卫上面所有判等: 零笔交易的话 `[] == []` 也会通过, 等于什么都没验。

    跑的是**配置矩阵用的那一个种子**, 断言的是矩阵要覆盖到的每一条分支:
    三种出场都出现、回调确认和冷静期都真的 arm 过。

    跳空补价不在这里断言: "持仓恰好跨隔夜、且开盘跳过止损线"在随机路径上是稀有
    事件, 调噪音去凑它只会把 fixture 调成一个看不懂的魔法数。那条分支由
    `test_pullback_v3_backtest.py::test_gap_through_the_stop_fills_at_the_open`
    用手搭的 bar 精确覆盖。
    """
    df_5m = _rth_5m(CAL_DAYS, MATRIX_SEED)
    prep, tf = _as_v3(df_5m)
    got = run_pullback_v3_backtest(prep, tf, PullbackParams(), _pb())
    reasons = pd.Series([t.reason for t in got.trades]).value_counts()
    assert len({t.entry_bar_idx for t in got.trades}) >= 10
    assert set(reasons.index) == {"SL", "TP", "PROTECTIVE_SL"}
    assert reasons.min() >= 3, f"某种出场太少, 分支覆盖不足: {reasons.to_dict()}"
    assert len(got.pullback_points) >= 10 and len(got.cooldown_spans) >= 3
