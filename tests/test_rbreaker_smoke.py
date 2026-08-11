# -*- coding: utf-8 -*-
"""
v3.0 R-Breaker: 全品种管线冒烟, **两个策略一起跑**。

**不是 pytest 模块** —— 和 `test_v3_minute_smoke.py` 一样是脚本式的 `main()`,
因为它要读 1 GB 的 parquet、跑 59 个品种 × 2 个策略, 不适合塞进单元测试。
合成数据的不变量已经由 test_rbreaker_{lines,sessions,backtest,stats,reversal}.py
钉住了; 这里要抓的是**真实数据里 59 个品种中某一个的时段形状意外** —— 那是
任何合成 fixture 都造不出来的。

两个策略跑**同一套**结构断言。突破和反转的引擎各写了一遍循环骨架(约 60 行
重复), 这里是防止两边基线纪律偷偷分叉的主要手段。

    python -m tests.test_rbreaker_smoke
"""
import os
import time
import traceback

import numpy as np
import pandas as pd

from src.data.v3_sessions import CLEAN_DIR, SESSION_MODES, apply_mode, prepare_base
from src.engine.rbreaker_backtest import SESSION_END, TRAIL, run_rbreaker_backtest
from src.engine.rbreaker_reversal_backtest import run_rbreaker_reversal_backtest
from src.performance.rbreaker_stats import compute_stats
from src.strategy.rbreaker import LINE_NAMES, RBreakerConfig, RBreakerParams, ReversalConfig
from src.viz.chart_rbreaker import TRAIL_NAME, TRAIL_REV_NAME, build_markers, step_rows, trail_rows

START, END = "2025-07-01", "2026-07-29"
PARAMS = RBreakerParams()
CONFIG = RBreakerConfig()
REV_CONFIG = ReversalConfig()


def structural_checks(sym: str, p, r, label: str, trail_name: str) -> None:
    """两个策略共用的一套断言。任何一边的基线纪律走偏都会在这里炸。"""
    df = p.df
    sid = df["session_id"].to_numpy()
    first = df["is_session_first"].to_numpy()
    last = df["is_session_last"].to_numpy()
    tag = f"{sym}/{label}"

    sids = [t.session_id for t in r.trades]
    assert len(sids) == len(set(sids)), f"{tag}: 某个时段成交了不止一笔"
    for t in r.trades:
        assert t.exit_bar_idx > t.entry_bar_idx, f"{tag}: 同一根既入场又出场"
        assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx], f"{tag}: 持仓跨了时段"
        assert not first[t.entry_bar_idx], f"{tag}: 时段首根入场"
        assert not last[t.entry_bar_idx], f"{tag}: 时段末根入场"
        assert t.reason in (TRAIL, SESSION_END), f"{tag}: 未知离场原因 {t.reason}"
    for a, b in zip(r.trades, r.trades[1:]):
        assert b.entry_bar_idx > a.exit_bar_idx, f"{tag}: 交易重叠"
    assert r.position[-1] == 0, f"{tag}: 收尾还留着仓位"

    # 止损线单调收紧(方向相反, 但都不许放松)
    for t in r.trades:
        seg = r.trail_stop[t.entry_bar_idx:t.exit_bar_idx + 1]
        dd = np.diff(seg)
        if t.direction == "long":
            assert (dd >= -1e-9).all(), f"{tag}: 多头止损放松了"
        else:
            assert (dd <= 1e-9).all(), f"{tag}: 空头止损放松了"

    comp = float(np.prod([1 + t.pnl_net for t in r.trades])) if r.trades else 1.0
    assert abs(r.equity_curve.iloc[-1] - comp) < 1e-9, f"{tag}: 权益曲线与逐笔复利对不上"

    # 图表层的数据帧 —— 推给浏览器之前先在这里炸
    tl = trail_rows(p, r, trail_name)
    assert len(tl) == int(r.in_position.sum()), f"{tag}: 止损线点数对不上持仓根数"
    assert int((tl["color"] == "rgba(0,0,0,0)").sum()) == len(r.trades), \
        f"{tag}: 透明点数 != 笔数, 笔与笔之间会连出线来"
    mk = build_markers(r)
    times = [m["time"] for m in mk]
    assert times == sorted(times), f"{tag}: marker 未排序, setMarkers 会出错"
    assert len(mk) == 2 * len(r.trades), f"{tag}: marker 数不是笔数的两倍"


def check_symbol(sym: str) -> list[dict]:
    """一个品种 × 5 种时段模式 × 2 个策略 = 10 行。"""
    t0 = time.perf_counter()
    base = prepare_base(sym, START, END, PARAMS, CONFIG)
    t_prep = time.perf_counter() - t0

    if base.df.empty:
        return [{"sym": sym, "mode": "-", "strat": "-", "bars": 0, "note": "窗口内无数据"}]

    assert base.df.index.is_monotonic_increasing, f"{sym}: 索引乱序"

    rows = []
    for mode in SESSION_MODES:
        p = apply_mode(base, mode)
        df = p.df
        n_sess = int(df["session_id"].nunique())
        tag = f"{sym}/{mode}"

        # 数据层的不变量, 与策略无关
        assert int(df["is_session_first"].sum()) == n_sess == int(df["is_session_last"].sum()), \
            f"{tag}: 时段首/末标记数对不上"
        assert (p.line_idx >= 0).all(), f"{tag}: line_idx 有负值"
        assert (np.diff(df["session_id"].to_numpy()) >= 0).all(), f"{tag}: 时段编号非单调"
        sid = df["session_id"].to_numpy()
        assert (pd.Series(p.line_idx).groupby(sid).nunique() == 1).all(), \
            f"{tag}: line_idx 在时段内发生跳变"
        assert (pd.Series(df["tradable"].to_numpy()).groupby(sid).nunique() == 1).all(), \
            f"{tag}: tradable 在时段内发生跳变"
        span = int(pd.Series(df["trading_date"].to_numpy()).groupby(sid).nunique().max())
        assert span <= (2 if mode == "day_night" else 1), f"{tag}: 时段跨了 {span} 个 trading_date"
        for name in LINE_NAMES:
            f = step_rows(p, name)
            assert len(f) == n_sess + 1, f"{tag}/{name}: 阶梯点数 {len(f)} != {n_sess}+1"
            assert f[name].notna().all(), f"{tag}/{name}: 阶梯里混进了 NaN"

        for label, runner, cfg, tname in (
            ("突破", run_rbreaker_backtest, CONFIG, TRAIL_NAME),
            ("反转", run_rbreaker_reversal_backtest, REV_CONFIG, TRAIL_REV_NAME),
        ):
            t0 = time.perf_counter()
            r = runner(p, PARAMS, cfg)
            t_loop = time.perf_counter() - t0
            structural_checks(sym, p, r, f"{mode}/{label}", tname)
            s = compute_stats(r, label, p.mode_label)
            rows.append({
                "sym": sym, "mode": mode, "strat": label,
                "bars": len(df), "days": p.tradable_days, "sess": n_sess,
                "dropped": len(base.dropped_runs),
                "trades": s.n_trades, "L/S": f"{s.n_long}/{s.n_short}",
                "stop/close": f"{s.n_exit_trail}/{s.n_exit_session}",
                "gap": sum(1 for t in r.trades if t.gapped),
                "net": f"{s.total_return:+.2%}" if s.n_trades else "-",
                "maxdd": f"{abs(s.max_drawdown):.2%}" if s.n_trades else "-",
                "win": f"{s.win_rate:.1%}" if not pd.isna(s.win_rate) else "n/a",
                "payoff": f"{s.payoff:.2f}" if not pd.isna(s.payoff) else "n/a",
                "sharpe": f"{s.sharpe:+.2f}" if not pd.isna(s.sharpe) else "n/a",
                "hold": f"{s.avg_bars_held:.0f}" if not pd.isna(s.avg_bars_held) else "n/a",
                "_net": s.total_return, "_dd": abs(s.max_drawdown), "_n": s.n_trades,
                "_hold": s.avg_bars_held, "_stop": s.n_exit_trail, "_close": s.n_exit_session,
                "_gap": sum(1 for t in r.trades if t.gapped),
                "prep_s": round(t_prep, 2), "loop_s": round(t_loop, 3),
            })
    return rows


def main() -> None:
    man_path = os.path.join(CLEAN_DIR, "_manifest.csv")
    if not os.path.exists(man_path):
        raise SystemExit(f"找不到 {man_path}\n先跑: python -m src.data.prepare_v3_minute")
    symbols = sorted(pd.read_csv(man_path, index_col="sym").index.astype(str))

    print(f"{len(symbols)} 个品种 × {len(SESSION_MODES)} 种时段 × 2 个策略 = "
          f"{len(symbols) * len(SESSION_MODES) * 2} 次回测 | 窗口 {START} ~ {END} | "
          f"f1={PARAMS.f1:g} f2={PARAMS.f2:g} f3={PARAMS.f3:g} d={PARAMS.d:g} "
          f"cost={CONFIG.cost_bps:g}bp\n")

    rows, failed = [], []
    t0 = time.perf_counter()
    for sym in symbols:
        try:
            rows.extend(check_symbol(sym))
            print(f"  {sym:<4} ok", flush=True)
        except Exception:
            failed.append(sym)
            print(f"  {sym:<4} FAILED", flush=True)
            traceback.print_exc()
    total = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    n_syms = df["sym"].nunique()
    print(f"\n{n_syms} 个品种通过, {len(failed)} 个失败, 合计 {total:.1f}s")

    if "dropped" in df:
        dropped = df.groupby("sym")["dropped"].first().fillna(0)
        print(f"被清洗掉的伪造段合计 {int(dropped.sum())} 条"
              f" (非零的品种: {list(dropped.index[dropped > 0])})")

    # ---- 本次真正想看的东西: 按 (时段模式 × 策略) 聚合 ----
    if "_net" in df:
        print("\n=== 按 (时段模式 × 策略) 聚合, 59 个品种 ===")
        g = df.groupby(["mode", "strat"], sort=False)
        agg = pd.DataFrame({
            "成交/品种 中位": g["_n"].median().round(0),
            "成交合计": g["_n"].sum(),
            "净收益 中位": g["_net"].median().map(lambda v: f"{v:+.2%}"),
            "为正的品种": g["_net"].apply(lambda s: int((s > 0).sum())),
            "回撤 中位": g["_dd"].median().map(lambda v: f"{v:.2%}"),
            "持有 中位": g["_hold"].median().round(0),
            "止损:收盘": g.apply(
                lambda x: f"{int(x['_stop'].sum())}:{int(x['_close'].sum())}",
                include_groups=False),
            "跳空成交": g["_gap"].sum(),
        })
        print(agg.to_string())
        print("\n注意: 这是 10 种配置在**同一年**数据上的比较。最好看的那一格基本"
              "注定是噪声 —— 只看结构性差异(笔数/持有/离场构成/跳空), 不要挑赢家。")

    if failed:
        raise SystemExit(f"失败: {failed}")


if __name__ == "__main__":
    main()
