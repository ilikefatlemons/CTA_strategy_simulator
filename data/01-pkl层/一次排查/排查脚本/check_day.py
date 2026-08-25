# -*- coding: utf-8 -*-
"""
单日数据体检: 把某个品种某个 trading_date 的全部 1m bar 摊开, 逐项查错,
并和原始 pkl 对比看清洗到底删了什么。

用法:
    python check_day.py                    # 默认 AU / 2025-10-09
    python check_day.py RB 2026-06-22
    python check_day.py AU 2025-10-09 --no-raw     # 跳过原始 pkl 对比(快)

原始 pkl 有 6.4GB, 第一次用到某个品种时会把它的切片缓存成
test_data/03-pkl层/排查证据/raw_1m/{sym}.parquet, 之后就不再读大文件了。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"D:\2026_Summer\TradingApp\data\03-pkl层"
PKL = os.path.join(ROOT, "all_symbol_min_full_main_close_k_1.pkl")
CLEAN = os.path.join(ROOT, "clean_1m")
RAWCACHE = os.path.join(ROOT, "排查证据", "raw_1m")

pd.set_option("display.width", 240, "display.max_rows", 800, "display.max_columns", 40)

OHLCV = ["open", "high", "low", "close", "volume"]


# ----------------------------------------------------------------- 数据读取
def load_clean(sym: str) -> pd.DataFrame:
    p = os.path.join(CLEAN, f"{sym}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"没有清洗后的 {sym}: {p}")
    return pd.read_parquet(p)


def load_raw(sym: str) -> pd.DataFrame:
    """原始(未清洗)切片。第一次调用会读 6.4GB 大 pkl 并缓存该品种。"""
    os.makedirs(RAWCACHE, exist_ok=True)
    p = os.path.join(RAWCACHE, f"{sym}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    print(f"  (首次读取 {sym} 的原始数据, 需要加载 6.4GB pkl, 约 10-30s ...)", flush=True)
    big = pd.read_pickle(PKL)
    one = big.loc[big["underlying_symbol"] == sym].drop(columns=["underlying_symbol"]).copy()
    del big
    one.to_parquet(p)
    print(f"  (已缓存到 {p})", flush=True)
    return one


# ----------------------------------------------------------------- 分析工具
def runs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """把一天的 bar 按「相邻时间戳相差 >1min 就断开」切成连续段, 返回 [(起, 止)] 位置区间。"""
    if df.empty:
        return []
    gap = df.index.to_series().diff() > pd.Timedelta("1min")
    rid = gap.cumsum().to_numpy()
    return [(int(np.flatnonzero(rid == r)[0]), int(np.flatnonzero(rid == r)[-1]))
            for r in np.unique(rid)]


def describe_runs(df: pd.DataFrame, tag: str) -> None:
    print(f"\n  {tag}: {len(df)} 根")
    if df.empty:
        print("    (无数据)")
        return
    for k, (a, b) in enumerate(runs(df)):
        seg = df.iloc[a:b + 1]
        vs = float(seg["volume"].sum())
        flat = bool((seg["open"] == seg["high"]).all() and (seg["high"] == seg["low"]).all()
                    and (seg["low"] == seg["close"]).all())
        mark = ""
        if vs == 0:
            mark = "   <== 假 session (整段零成交, 是补出来的)"
        elif flat:
            mark = "   <== 整段 OHLC 全平"
        print(f"    run {k}: {seg.index[0]} -> {seg.index[-1]}  "
              f"({len(seg):>4} 根, volume合计={vs:>12,.0f}){mark}")


def quality_flags(df: pd.DataFrame) -> None:
    """逐项查错。每一项都打印结果, 没问题也打印 OK, 避免'沉默通过'。"""
    print("\n  --- 逐项检查 ---")
    if df.empty:
        print("    该 trading_date 没有任何 bar")
        return
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy()

    checks = [
        ("OHLC 有 NaN", int(np.isnan(o).sum() + np.isnan(h).sum()
                            + np.isnan(l).sum() + np.isnan(c).sum())),
        ("OHLC 有 <=0", int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())),
        ("high < low", int((h < l).sum())),
        ("high < max(open,close)", int((h < np.maximum(o, c)).sum())),
        ("low  > min(open,close)", int((l > np.minimum(o, c)).sum())),
        ("volume < 0", int((v < 0).sum())),
        ("volume == 0", int((v == 0).sum())),
        ("OHLC 全平 (o==h==l==c)", int(((o == h) & (h == l) & (l == c)).sum())),
        ("时间戳重复", int(df.index.duplicated().sum())),
        ("时间戳非单调", int(not df.index.is_monotonic_increasing)),
    ]
    for name, n in checks:
        print(f"    {'OK  ' if n == 0 else 'HIT '} {name:<26} {n}")

    print(f"    ---- contract: {sorted(map(str, df['contract'].unique()))}")
    if "K" in df:
        k = df["K"].to_numpy()
        print(f"    ---- K: 取值 {sorted(set(np.round(k[np.isfinite(k)], 8)))[:5]}"
              f"  nan={int(np.isnan(k).sum())} inf={int(np.isinf(k).sum())} zero={int((k == 0).sum())}")
    print(f"    ---- 覆盖日历日: {sorted({str(d) for d in df.index.normalize().date})}")
    print(f"    ---- volume 合计 {v.sum():,.0f} | 有成交的 bar {int((v > 0).sum())}/{len(v)} "
          f"({(v > 0).mean():.1%})")
    print(f"    ---- 价格区间 low {l.min():g} ~ high {h.max():g} | "
          f"首 open {o[0]:g} 末 close {c[-1]:g}")


def context(df: pd.DataFrame, day: pd.Timestamp, n: int = 3) -> None:
    """前后各 n 个交易日的 bar 数与成交量, 用来判断这一天是不是异常。"""
    per = df.groupby("trading_date").agg(bars=("close", "size"), volume=("volume", "sum"))
    if day not in per.index:
        print(f"\n  {day.date()} 不在该品种的 trading_date 里")
        near = per.index[np.argsort(np.abs((per.index - day).days))[:5]]
        print(f"  最近的交易日: {sorted(str(d.date()) for d in near)}")
        return
    i = per.index.get_loc(day)
    win = per.iloc[max(0, i - n):i + n + 1].copy()
    win["是否本日"] = ["<<<<" if d == day else "" for d in win.index]
    win["bars众数差"] = win["bars"] - int(per["bars"].mode().iloc[0])
    print(f"\n  --- 上下文 (该品种 bars/日 众数 = {int(per['bars'].mode().iloc[0])}) ---")
    print(win.to_string())


# ----------------------------------------------------------------- 主流程
def check_day(sym: str = "AU", date: str = "2025-10-09", compare_raw: bool = True) -> None:
    day = pd.Timestamp(date)
    print("=" * 100)
    print(f"单日数据体检:  {sym}   trading_date = {day.date()}")
    print("=" * 100)

    clean = load_clean(sym)
    cd = clean.loc[clean["trading_date"] == day].sort_index()

    print(f"\n[清洗后]  {sym}.parquet  共 {len(clean):,} 行  "
          f"{clean.index.min()} .. {clean.index.max()}")
    describe_runs(cd, "清洗后该日 bar")
    quality_flags(cd)
    context(clean, day)

    if not compare_raw:
        return

    raw = load_raw(sym)
    rd = raw.loc[raw["trading_date"] == day].sort_index()
    print("\n" + "-" * 100)
    print(f"[原始]  共 {len(raw):,} 行 (清洗共删除 {len(raw) - len(clean):,} 行)")
    describe_runs(rd, "原始该日 bar")

    removed = rd.index.difference(cd.index)
    added = cd.index.difference(rd.index)
    print(f"\n  --- 清洗对这一天做了什么 ---")
    print(f"    原始 {len(rd)} 根 -> 清洗后 {len(cd)} 根 (删除 {len(removed)}, 新增 {len(added)})")
    if len(removed):
        seg = rd.loc[removed]
        print(f"    被删除的 bar: {seg.index[0]} .. {seg.index[-1]}, "
              f"volume合计={seg['volume'].sum():,.0f}, "
              f"全平={bool((seg['open'] == seg['close']).all() and (seg['high'] == seg['low']).all())}")
        describe_runs(seg, "被删除的部分")
    if len(added):
        print(f"    !! 清洗后凭空多出 {len(added)} 根, 这不应该发生: {list(added)[:10]}")

    # 值是否被改过(共同的时间戳上 OHLCV 是否一致)
    common = cd.index.intersection(rd.index)
    if len(common):
        diff = (cd.loc[common, OHLCV].to_numpy() != rd.loc[common, OHLCV].to_numpy())
        nd = int(diff.any(axis=1).sum())
        print(f"    共同时间戳 {len(common)} 根, 其中 OHLCV 被改动过的: {nd}"
              + ("  (清洗只做删除, 不改值 -> 符合预期)" if nd == 0 else "  <== 有值被改写!"))

    out = os.path.join(ROOT, "排查证据", f"check_{sym}_{day.date()}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rd.assign(in_clean=rd.index.isin(cd.index)).to_csv(out)
    print(f"\n  完整逐 bar 明细 -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_day(
        sym=args[0] if len(args) > 0 else "AU",
        date=args[1] if len(args) > 1 else "2025-10-09",
        compare_raw="--no-raw" not in sys.argv,
    )
