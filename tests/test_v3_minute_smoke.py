# -*- coding: utf-8 -*-
"""v3.0 分钟线数据链路冒烟测试 (不开窗口)。"""
import os

import numpy as np
import pandas as pd

from src.viz.chart_minute import CLEAN_DIR, load_minute

WIN = ("2025-07-29", "2026-07-29")


def _jumps(df, thr=0.01):
    r = np.abs(np.diff(np.log(df["close"].to_numpy())))
    return int((r > thr).sum()), float(r.max())


def main():
    man = pd.read_csv(os.path.join(CLEAN_DIR, "_manifest.csv"), index_col="sym")
    syms = sorted(man.index.astype(str))
    print(f"manifest: {len(syms)} symbols, {man.bars.sum():,} bars, {man.mb.sum():.0f} MB")

    print("\n--- 单品种加载耗时 / 行数 (窗口 %s ~ %s) ---" % WIN)
    import time
    for s in ["RB", "I", "AU", "IF", "T", "B"]:
        t0 = time.time()
        d = load_minute(s, *WIN, adjust=True)
        dt = time.time() - t0
        raw = load_minute(s, *WIN, adjust=False)
        nj_a, mx_a = _jumps(d)
        nj_r, mx_r = _jumps(raw)
        print(f"  {s:3s} {len(d):>7,} bars  {dt*1000:5.0f} ms   "
              f"复权后跳>1%: {nj_a:3d} (max {mx_a:.4f})   "
              f"裸价跳>1%: {nj_r:3d} (max {mx_r:.4f})   "
              f"首根 复权={d['close'].iloc[0]:.2f} 裸={raw['close'].iloc[0]:.2f}")

    print("\n--- 列/类型契约 ---")
    d = load_minute("RB", *WIN)
    print(f"  columns={list(d.columns)}  (须含 volume, 库靠它画成交量副图)")
    assert "volume" in d.columns, "缺 volume 列, 成交量副图画不出来"
    raw_v = load_minute("RB", *WIN, adjust=False)["volume"]
    same = bool((d["volume"].to_numpy() == raw_v.to_numpy()).all())
    print(f"  volume 未被复权污染: {same}  (K 是价格因子, 手数不该乘)")
    assert same
    print(f"  time dtype={d['time'].dtype}, 单调递增={d['time'].is_monotonic_increasing}")
    print(f"  NaN 数={int(d.isna().sum().sum())}, 非正价数={int((d[['open','high','low','close']]<=0).sum().sum())}")
    bad = ((d.high < d.low) | (d.high < d[["open", "close"]].max(axis=1) - 1e-6)
           | (d.low > d[["open", "close"]].min(axis=1) + 1e-6)).sum()
    print(f"  OHLC 逻辑破损={int(bad)}")

    print("\n--- 伪造 bar 是否已剔除 (节前夜盘 / 元旦) ---")
    full = pd.read_parquet(os.path.join(CLEAN_DIR, "RB.parquet"),
                           columns=["volume", "trading_date"])
    for td in ["2024-10-08", "2025-02-05", "2026-02-24", "2023-01-03", "2025-01-02"]:
        g = full[full.trading_date == pd.Timestamp(td)]
        cals = sorted({str(x.date()) for x in g.index})
        print(f"  trading_date={td}: {len(g):>3d} bars, 成交量 {g.volume.sum():>12,.0f}, "
              f"日历日 {cals}")

    print("\n--- 全池 59 个品种能否全部加载 ---")
    bad_syms = []
    for s in syms:
        try:
            x = load_minute(s, *WIN)
            if x.empty:
                bad_syms.append((s, "empty"))
        except Exception as e:                                   # noqa: BLE001
            bad_syms.append((s, repr(e)[:60]))
    print(f"  OK {len(syms)-len(bad_syms)}/{len(syms)}" +
          (f"  失败: {bad_syms}" if bad_syms else ""))


if __name__ == "__main__":
    main()
