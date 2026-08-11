# -*- coding: utf-8 -*-
"""
v3.0 回调策略: 全品种管线冒烟。

**不是 pytest 模块** —— 和 `test_rbreaker_smoke.py` 一样是脚本式的 `main()`,
因为它要读 1 GB 的 parquet、跑 59 个品种 × 5 种时段模式。合成数据的不变量已经由
`test_pullback_v3_{states,backtest,sessions,entry_gate,chart}.py` 钉住了; 这里要抓的
是**真实数据里 59 个品种中某一个的时段/分箱意外** —— 那是任何合成 fixture 都造不
出来的, 比如某个品种的暖机段凑不满、某个时代的时段形状与实测不同。

顺带盯住**性能**: 单次回测 > 1.5s 就算回归。图表终端是交互式的, 引擎一慢窗口就卡,
而这类退化(比如不小心退回逐前缀 rolling)在功能测试里完全看不出来。

    python -m tests.test_pullback_v3_smoke
"""
import os
import time
import traceback

import numpy as np
import pandas as pd

from src.data.v3_sessions import (
    CLEAN_DIR, SESSION_MODES, apply_mode, derive_segments, prepare_base, segment_is_night,
)
from src.data.v3_timeframes import PB_TFS, build_timeframes
from src.engine.pullback_v3_backtest import (
    END_SESSION, EXIT_REASONS, PROTECTIVE_SL, SL, TP, run_pullback_v3_backtest,
)
from src.performance.pullback_stats import batch_pnls, compute_stats
from src.strategy.pullback import PullbackConfig, PullbackParams
from src.viz.chart_pullback import (
    MA_PERIODS, atr_frame, cooldown_frame, ma_frame, markers, tile_frame,
)

START, END = "2025-07-01", "2026-07-29"
WARMUP_DAYS = 60
PARAMS = PullbackParams()
CONFIG = PullbackConfig()
MAX_LOOP_S = 1.5          # 单次回测的上限。R-Breaker 是 0.15s, 这里留了 10 倍余量。


def structural_checks(sym: str, prep, tf, r, label: str) -> None:
    n = len(prep.df)
    tag = f"{sym}/{label}"

    for t in r.trades:
        assert t.reason in EXIT_REASONS, f"{tag}: 未知离场原因 {t.reason}"
        assert t.exit_bar_idx >= t.entry_bar_idx, f"{tag}: 出场早于入场"
        assert t.entry_bar_idx >= prep.warmup_bars, f"{tag}: 暖机段里开仓了"
        assert 0 < t.size_fraction <= 1.0, f"{tag}: 腿的比例越界"

    # 时段锁死的两条核心不变量, 59 个品种 x 5 种模式全覆盖:
    #   1) 每一根**时段末根**上仓位都是 0 (强平那一根在 mark-to-market 之前就把
    #      batch 置了 None; 时段末根本来就不许开仓)
    #   2) 每条腿的出场与它那一批的入场在**同一个时段**里
    # 上一版这里有一段"末批未平仓"的豁免 —— 锁死之后那条路径不可达, 删掉了。
    sid = prep.df["session_id"].to_numpy()
    last_of_sess = prep.df["is_session_last"].to_numpy(bool)
    assert (r.position[last_of_sess] == 0).all(), f"{tag}: 有时段末根仍持着仓位"
    assert r.position[-1] == 0, f"{tag}: 收尾还留着仓位"

    # 一批 = 一个 entry_bar_idx; 各腿比例之和恒为 1。锁死之后只有四种收尾形状。
    batches: dict[int, list] = {}
    for t in r.trades:
        batches.setdefault(t.entry_bar_idx, []).append(t)
    for entry, legs in batches.items():
        total = sum(t.size_fraction for t in legs)
        reasons = sorted(t.reason for t in legs)
        assert abs(total - 1.0) < 1e-12, f"{tag}: 批 {entry} 的腿加起来不是 1"
        assert reasons in ([SL], [END_SESSION], [END_SESSION, TP],
                           sorted([TP, PROTECTIVE_SL])), \
            f"{tag}: 批 {entry} 的出场组合不合法 {reasons}"
        for t in legs:
            assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx], \
                f"{tag}: 批 {entry} 的一条腿跨了时段"

    # 同一时间只允许一批
    spans = sorted((e, max(t.exit_bar_idx for t in ls)) for e, ls in batches.items())
    for (e1, x1), (e2, _) in zip(spans, spans[1:]):
        assert e2 > x1, f"{tag}: 批次 {e1}-{x1} 与 {e2} 重叠"

    # 权益曲线 == 逐腿复利。锁死之后收尾必然空仓, 所以这条对**每个**品种/模式都
    # 真的跑起来了 (上一版只在恰好平仓的那些组合上跑)。
    if r.position[-1] == 0 and len(r.equity_curve):
        comp = float(np.prod([1 + t.size_fraction * t.pnl_net for t in r.trades])) \
            if r.trades else 1.0
        assert abs(r.equity_curve.iloc[-1] - comp) < 1e-9, f"{tag}: 权益与逐笔复利对不上"

    assert (r.position[: prep.warmup_bars] == 0).all(), f"{tag}: 暖机段有仓位"
    assert np.isnan(r.stop_line[: prep.warmup_bars]).all(), f"{tag}: 暖机段画了止损线"
    assert len(r.equity_curve) == n - prep.warmup_bars, f"{tag}: 权益曲线长度对不上"
    for s, e in r.cooldown_spans:
        assert 0 <= s <= e < n, f"{tag}: 冷静期区间越界"

    # 图表层的数据帧 —— 推给浏览器之前先在这里炸
    for name in PB_TFS:
        tfb = tf[name]
        start = tfb.first_bin_at_or_after(prep.warmup_bars)
        cand = tile_frame(tfb, start)
        assert len(cand), f"{tag}/{name}: 显示帧空了"
        assert list(cand.columns) == ["time", "open", "high", "low", "close"], \
            f"{tag}/{name}: 带了 volume 会自动长出成交量副图"
        assert cand["time"].is_monotonic_increasing, f"{tag}/{name}: 显示帧乱序"
        atr = atr_frame(tfb, start)
        assert list(atr["time"]) == list(cand["time"]), f"{tag}/{name}: ATR 与 K 线错位"
        for period in MA_PERIODS[name]:
            ma = ma_frame(tfb, period, start)
            assert list(ma["time"]) == list(cand["time"]), \
                f"{tag}/{name}: MA{period} 与 K 线错位"
            # 暖机段够长的话第一根可见 K 线上均线就该有值 (暖机不足另说)
            if not r.warmup_days < PARAMS.ma_slow:
                assert not np.isnan(ma[f"MA{period}"].iloc[0]), \
                    f"{tag}/{name}: MA{period} 首根仍是 NaN"
        ms = markers(r, tfb, start, notes=True)
        times = [m["time"] for m in ms]
        assert times == sorted(times), f"{tag}/{name}: marker 未排序, setMarkers 会出错"
        valid = set(cand["time"].to_numpy())
        for m in ms:
            assert np.datetime64(m["time"]) in valid, \
                f"{tag}/{name}: marker 落在 {m['time']}, 那里没有 K 线"
        cd = cooldown_frame(r, tfb, start)
        if cd is not None:
            assert (cd["cooldown"] == 1.0).all(), f"{tag}/{name}: 冷静期带的值不是 1"


def check_symbol(sym: str) -> list[dict]:
    t0 = time.perf_counter()
    base = prepare_base(sym, START, END, warmup_days=WARMUP_DAYS)
    t_prep = time.perf_counter() - t0

    if base.df.empty:
        return [{"sym": sym, "mode": "-", "bars": 0, "note": "窗口内无数据"}]
    assert base.df.index.is_monotonic_increasing, f"{sym}: 索引乱序"

    # 有没有夜盘决定了"五种模式两两不同"成不成立 —— 见 main() 里那条断言。
    seg = derive_segments(base.df)
    has_night = bool(segment_is_night(base.df, seg).any())

    rows = []
    for mode in SESSION_MODES:
        prep = apply_mode(base, mode)
        t1 = time.perf_counter()
        tf = build_timeframes(prep.df)
        t_tf = time.perf_counter() - t1
        t1 = time.perf_counter()
        r = run_pullback_v3_backtest(prep, tf, PARAMS, CONFIG)
        t_loop = time.perf_counter() - t1
        assert t_loop < MAX_LOOP_S, \
            f"{sym}/{mode}: 单次回测 {t_loop:.2f}s > {MAX_LOOP_S}s —— 性能回归"
        structural_checks(sym, prep, tf, r, mode)

        s = compute_stats(r, prep.mode_label, PARAMS)
        pnls = [v for _, v in batch_pnls(r.trades).values()]
        rows.append({
            "sym": sym, "mode": mode, "bars": len(prep.df),
            "warm": prep.warmup_days, "days": s.tradable_days,
            "batches": s.n_batches, "L/S": f"{s.n_long}/{s.n_short}",
            "SL/TP/PSL/End": f"{s.n_sl}/{s.n_tp}/{s.n_psl}/{s.n_end}", "gap": s.n_gapped,
            "net": f"{s.total_return:+.2%}" if pnls else "-",
            "maxdd": f"{abs(s.max_drawdown):.2%}" if pnls else "-",
            "win": f"{s.win_rate:.1%}" if not pd.isna(s.win_rate) else "n/a",
            "payoff": f"{s.payoff:.2f}" if not pd.isna(s.payoff) else "n/a",
            "sharpe": f"{s.sharpe:+.2f}" if not pd.isna(s.sharpe) else "n/a",
            "hold": f"{s.avg_bars_held:.0f}" if not pd.isna(s.avg_bars_held) else "n/a",
            "_net": s.total_return, "_dd": abs(s.max_drawdown), "_n": s.n_batches,
            "_hold": s.avg_bars_held, "_sl": s.n_sl, "_tp": s.n_tp, "_psl": s.n_psl,
            "_end": s.n_end, "_night": int(bool(has_night)),
            "_gap": s.n_gapped, "_short_warm": int(s.warmup_short),
            "prep_s": round(t_prep, 2), "tf_s": round(t_tf, 3), "loop_s": round(t_loop, 3),
        })
    return rows


def main() -> None:
    man_path = os.path.join(CLEAN_DIR, "_manifest.csv")
    if not os.path.exists(man_path):
        raise SystemExit(f"找不到 {man_path}\n先跑: python -m src.data.prepare_v3_minute")
    symbols = sorted(pd.read_csv(man_path, index_col="sym").index.astype(str))

    print(f"{len(symbols)} 个品种 × {len(SESSION_MODES)} 种时段 = "
          f"{len(symbols) * len(SESSION_MODES)} 次回测 | 窗口 {START} ~ {END} | "
          f"暖机 {WARMUP_DAYS} 个交易日 | cost={CONFIG.cost_bps:g}bp\n")

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
    print(f"\n{df['sym'].nunique()} 个品种通过, {len(failed)} 个失败, 合计 {total:.1f}s")
    if "loop_s" in df:
        print(f"单次回测 中位 {df['loop_s'].median():.3f}s / 最慢 {df['loop_s'].max():.3f}s"
              f"  (上限 {MAX_LOOP_S}s)")
    if "_short_warm" in df:
        short = sorted(df.loc[df["_short_warm"] > 0, "sym"].unique())
        print(f"暖机不足的品种 ({len(short)}): {short}")

    if "_net" in df:
        print("\n=== 按时段模式聚合, 59 个品种 ===")
        g = df.groupby("mode", sort=False)
        agg = pd.DataFrame({
            "批次/品种 中位": g["_n"].median().round(0),
            "批次合计": g["_n"].sum(),
            "净收益 中位": g["_net"].median().map(lambda v: f"{v:+.2%}"),
            "为正的品种": g["_net"].apply(lambda s: int((s > 0).sum())),
            "回撤 中位": g["_dd"].median().map(lambda v: f"{v:.2%}"),
            "持有 中位": g["_hold"].median().round(0),
            "止损:止盈:护盈:收盘": g.apply(
                lambda x: f"{int(x['_sl'].sum())}:{int(x['_tp'].sum())}"
                          f":{int(x['_psl'].sum())}:{int(x['_end'].sum())}",
                include_groups=False),
            "跳空成交": g["_gap"].sum(),
        })
        print(agg.to_string())
        print("\n注意 1: 时段锁死之后五种模式**两两不同** —— 强平读 is_session_last, "
              "而它直接由时段分组算出, 于是分组第一次变成可观测的东西。"
              "唯一的例外是**无夜盘品种**: 它们每个 trading_date 只有一个原子段, "
              "seg / trading_date / day_night 三个分组键同步递增, 于是 "
              "split≡night_day≡day_night≡day, 而 night 零成交。")
        print("注意 2: 这是同一年数据上的多重比较, 最好看的那一格基本注定是噪声 —— "
              "只看结构性差异(批次/持有/离场构成/跳空), 不要挑赢家。")

        # 注意 1 的机器版本, 按有无夜盘分开断言。
        cols = ["_n", "_net", "_dd", "_sl", "_tp", "_psl", "_end", "_gap"]
        keyed = {m: df.loc[df["mode"] == m].set_index("sym")[cols + ["_night"]]
                 for m in SESSION_MODES}
        night_syms = sorted(keyed["split"].index[keyed["split"]["_night"] == 1])
        flat_syms = sorted(keyed["split"].index[keyed["split"]["_night"] == 0])
        print(f"\n有夜盘 {len(night_syms)} 个 / 无夜盘 {len(flat_syms)} 个: {flat_syms}")

        def sig(m, syms):
            return keyed[m].loc[syms, cols].round(10).to_csv()

        # 有夜盘: 五种模式两两不同
        for syms, label in ((night_syms, "有夜盘"),):
            if not syms:
                continue
            seen = {}
            for m in SESSION_MODES:
                s = sig(m, syms)
                assert s not in seen, f"{label}: {m} 与 {seen[s]} 结果相同 —— 锁死没生效"
                seen[s] = m
        # 无夜盘: split / night_day / day_night / day 必然合一, night 一笔不做
        if flat_syms:
            base = sig("split", flat_syms)
            for m in ("night_day", "day_night", "day"):
                assert sig(m, flat_syms) == base, f"无夜盘品种上 {m} 竟与 split 不同"
            assert int(keyed["night"].loc[flat_syms, "_n"].sum()) == 0, \
                "无夜盘品种在夜盘模式下竟然有成交"

    if failed:
        raise SystemExit(f"失败: {failed}")


if __name__ == "__main__":
    main()
