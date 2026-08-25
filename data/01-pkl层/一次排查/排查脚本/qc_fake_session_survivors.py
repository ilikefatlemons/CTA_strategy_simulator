# -*- coding: utf-8 -*-
"""
查 clean_1m 里**残留的假 session**。

背景 (AU 2025-10-09 发现的):
清洗规则是「整个**日历日** volume==0 就删」。但一个假 session 可能和真 session
共享同一个日历日 —— 跨零点的夜盘品种 (AU/AG/SC/AL/CU/ZN/PB/SN/NI/SS/BC) 的
夜盘后半段 00:00~02:30 落在**下一个日历日**上, 而那个日历日还装着真实日盘。
于是日历日合计 volume > 0, 假夜盘就被留下来了。

  AU trading_date=2025-10-09 (国庆 10-01~10-08 后首个交易日):
    2025-10-08 21:00~23:59  180根 flat@874.4 vol=0   -> 已删 OK
    2025-10-09 00:00~02:29  150根 flat@874.4 vol=0   -> **残留**
    2025-10-09 02:30         1根 flat@909.96 vol=454 -> **残留, 且是"假期压缩bar"**

「假期压缩 bar」: 停牌期间的价格跳变被塞进 session 最后一分钟。
AU 那根 874.4 -> 909.96 = +4.06%, 而 909.96 几乎等于次日 09:01 的 open 909.98
—— 等于把**节后开盘价**标在了假期的时间戳上。回测里它会在一个根本无法交易的
时刻触发止损/止盈。

正确的清洗粒度是 (trading_date, session), 不是 calendar day。
"""
import os

import numpy as np
import pandas as pd

ROOT = r"D:\2026_Summer\TradingApp\data\03-pkl层"
CLEAN = os.path.join(ROOT, "clean_1m")
OUT = os.path.join(ROOT, "排查证据")

MIN_BARS = 20     # 一个 session 至少这么多根才算数(排除零碎)
MAX_TRADED = 2    # 有成交的 bar 数 <= 这个 -> 判为假 session
JUMP_TH = 0.01    # 压缩 bar 相对前一根的价格跳变阈值

pd.set_option("display.width", 240, "display.max_rows", 500, "display.max_columns", 40)


def sessions_of(df: pd.DataFrame) -> np.ndarray:
    """(trading_date, 连续时间段) 的组合 id。相邻 bar 相差 >1min 或换 trading_date 就断开。"""
    gap = df.index.to_series().diff() > pd.Timedelta("1min")
    tdc = df["trading_date"].ne(df["trading_date"].shift())
    return (gap | tdc).cumsum().to_numpy()


rows, comp = [], []
syms = sorted(f[:-8] for f in os.listdir(CLEAN) if f.endswith(".parquet"))
print(f"扫描 {len(syms)} 个品种 ...", flush=True)

for s in syms:
    df = pd.read_parquet(os.path.join(CLEAN, f"{s}.parquet")).sort_index()
    sid = sessions_of(df)
    v = df["volume"].to_numpy()
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    flat = (o == h) & (h == l) & (l == c)

    g = pd.DataFrame({"sid": sid, "v": v, "traded": v > 0, "flat": flat})
    agg = g.groupby("sid").agg(bars=("v", "size"), vol=("v", "sum"),
                               traded=("traded", "sum"), flat=("flat", "sum"))
    bad = agg[(agg["bars"] >= MIN_BARS) & (agg["traded"] <= MAX_TRADED)]
    if bad.empty:
        continue

    first_i = pd.Series(np.arange(len(df))).groupby(sid).min()
    last_i = pd.Series(np.arange(len(df))).groupby(sid).max()
    for sidv, r in bad.iterrows():
        a, b = int(first_i[sidv]), int(last_i[sidv])
        rows.append({
            "sym": s, "trading_date": df["trading_date"].iloc[a],
            "start": df.index[a], "end": df.index[b],
            "bars": int(r["bars"]), "traded_bars": int(r["traded"]),
            "flat_bars": int(r["flat"]), "volume": float(r["vol"]),
        })
        # 压缩 bar: 该假 session 里唯一(或最后)有成交的那根, 且相对前一根有大跳变
        tb = np.flatnonzero(v[a:b + 1] > 0)
        for j in tb:
            i = a + int(j)
            prev = c[i - 1] if i > 0 else np.nan
            ret = c[i] / prev - 1 if prev and np.isfinite(prev) and prev > 0 else np.nan
            if np.isfinite(ret) and abs(ret) >= JUMP_TH:
                nxt = df.iloc[i + 1:i + 2]
                comp.append({
                    "sym": s, "at": df.index[i], "trading_date": df["trading_date"].iloc[i],
                    "prev_close": prev, "close": c[i], "ret_pct": ret * 100,
                    "volume": v[i], "flat": bool(flat[i]),
                    "next_open": float(nxt["open"].iloc[0]) if len(nxt) else np.nan,
                    "next_at": nxt.index[0] if len(nxt) else pd.NaT,
                })
    print(f"  {s}: {len(bad)} 个残留假 session", flush=True)

F = pd.DataFrame(rows)
C = pd.DataFrame(comp)

print("\n" + "=" * 100)
print("残留假 session (clean_1m 里仍然存在)")
print("=" * 100)
if F.empty:
    print("  (无)")
else:
    print(f"总计 {len(F)} 个 session, 共 {int(F['bars'].sum()):,} 根 bar，涉及 {F['sym'].nunique()} 个品种")
    print("\n--- 逐品种 ---")
    per = F.groupby("sym").agg(sessions=("bars", "size"), bars=("bars", "sum"),
                               first=("start", "min"), last=("start", "max"))
    print(per.sort_values("bars", ascending=False).to_string())
    print("\n--- 起始时刻分布 (确认是不是都在跨零点夜盘的 00:00 段) ---")
    print(F["start"].dt.strftime("%H:%M").value_counts().head(10).to_string())
    F.to_csv(os.path.join(OUT, "qc_fake_session_survivors.csv"), index=False)

print("\n" + "=" * 100)
print(f"假期压缩 bar (假 session 内, 跳变 >= {JUMP_TH:.0%} 的有成交 bar)")
print("=" * 100)
if C.empty:
    print("  (无)")
else:
    C = C.sort_values("ret_pct", key=lambda s: s.abs(), ascending=False)
    print(f"总计 {len(C)} 根，涉及 {C['sym'].nunique()} 个品种")
    print("\n--- 时刻分布 ---")
    print(C["at"].dt.strftime("%H:%M").value_counts().to_string())
    print("\n--- 跳变最大的 25 根 ---")
    print(C.head(25).to_string(index=False))
    # 验证「压缩 bar 的收盘 ≈ 下一段真实 session 的开盘」
    ok = C.dropna(subset=["next_open"])
    if len(ok):
        rel = (ok["close"] / ok["next_open"] - 1).abs()
        print(f"\n--- 验证: 压缩 bar 的 close 与紧邻下一根 open 的相对差 ---")
        print(f"    中位数 {rel.median():.4%}   <0.1% 的占比 {(rel < 0.001).mean():.1%}"
              f"   (接近 0 => 压缩 bar 用的就是复市后的开盘价)")
    C.to_csv(os.path.join(OUT, "qc_holiday_compression_bars.csv"), index=False)

print(f"\n-> {OUT}\\qc_fake_session_survivors.csv")
print(f"-> {OUT}\\qc_holiday_compression_bars.csv")
