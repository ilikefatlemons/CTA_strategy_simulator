# -*- coding: utf-8 -*-
"""
决定性检验 + 具体 K 线证据。

核心问题: 节后首日出现在 02:30 / 01:00 的那根"平 bar + 大成交量", 到底是
  (a) 前一交易日收盘的 padding, 还是
  (b) 当日 09:00 日盘集合竞价被错标到了夜盘尾根?

判据是价格:
  设节后首日 T 的 02:30 bar 收盘价 = X
      同日 09:01 bar 开盘价       = Y   (日盘第一根连续交易 bar)
      上一交易日 15:00 收盘价     = Z
  X == Y  -> 是日盘竞价被错标 (b)
  X == Z  -> 是前收盘 padding   (a)
"""
import os
import time

import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\03-pkl层\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\03-pkl层\排查证据"
EXCH = {
    "SHFE": "AG AL AO AU BR BU CU FU HC NI PB RB RU SN SP SS WR ZN",
    "INE": "BC EC LU NR SC",
    "DCE": "A B BB C CS EB EG FB I J JD JM L LH M P PG PP RR V Y",
    "CZCE": "AP CF CJ CY FG JR LR MA OI PF PK PM PX RI RM RS SA SF SH SM SR TA UR WH ZC",
    "CFFEX": "IC IF IH IM T TF TL TS",
    "GFEX": "LC SI",
}
SYM2EX = {s: e for e, ss in EXCH.items() for s in ss.split()}
_t0 = time.time()
def hr(t):
    print("\n" + "=" * 96, flush=True); print(t, flush=True); print("=" * 96, flush=True)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)

df = pd.read_pickle(PKL)
print(f"[{time.time()-_t0:.1f}s] loaded", flush=True)
sym = df["underlying_symbol"].astype("category")
sc = sym.cat.codes.to_numpy(); SYMS = np.asarray(sym.cat.categories)
EXOF = np.array([SYM2EX.get(str(s), "?") for s in SYMS])
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
op = df["open"].to_numpy(); hi = df["high"].to_numpy()
lo = df["low"].to_numpy(); cl = df["close"].to_numpy(); vol = df["volume"].to_numpy()
d_codes, d_uniq = pd.factorize(df["trading_date"], sort=True)
d_uniq = pd.DatetimeIndex(d_uniq); ND = len(d_uniq)
night = (mod >= 21 * 60) | (mod <= 179)

key = sc.astype(np.int64) * ND + d_codes
b = np.flatnonzero(np.diff(key)) + 1
gs = np.concatenate([[0], b]); ge = np.concatenate([b, [len(key)]])
gkey = key[gs]; g_sym = (gkey // ND).astype(int); g_dc = (gkey % ND).astype(int)
g_date = d_uniq[g_dc]; g_ex = EXOF[g_sym]
night_vol = np.add.reduceat(np.where(night, vol, 0.0), gs)
same_prev = np.concatenate([[False], g_sym[1:] == g_sym[:-1]])
gap = np.where(same_prev, np.concatenate([[np.nan],
              np.diff(g_date.to_numpy()) / np.timedelta64(1, "D")]), np.nan)

def bar_at(m):
    pos = np.full(len(gs), -1, np.int64)
    hits = np.flatnonzero(mod == m)
    if len(hits):
        pos[np.searchsorted(gs, hits, side="right") - 1] = hits
    return pos

P0230, P0100, P0901, P1500, P2100 = (bar_at(150), bar_at(60), bar_at(541),
                                     bar_at(900), bar_at(21 * 60))

# ================================================== 决定性检验
hr("决定性检验: 节后首日夜盘尾根的价格 == 当日 09:01 开盘价, 还是 == 前收盘?")
rows = []
for label, P, syms in [("02:30", P0230, ["AU", "AG", "SC"]),
                       ("01:00", P0100, ["CU", "AL", "ZN", "PB", "NI", "SN", "SS", "BC"])]:
    sel = np.flatnonzero((P >= 0) & (gap > 3) & np.isin(SYMS[g_sym], syms) & (vol[np.maximum(P, 0)] > 0))
    for i in sel:
        p_tail, p_day = P[i], P0901[i]
        if p_day < 0 or i == 0 or g_sym[i - 1] != g_sym[i]:
            continue
        p_prev_close = P1500[i - 1]
        if p_prev_close < 0:
            continue
        rows.append({
            "尾根": label, "sym": SYMS[g_sym[i]], "date": g_date[i], "gap": gap[i],
            "尾根close_X": cl[p_tail], "日盘0901open_Y": op[p_day],
            "前日1500close_Z": cl[p_prev_close],
            "尾根vol": vol[p_tail], "0901vol": vol[p_day],
            "X==Y": bool(cl[p_tail] == op[p_day]),
            "X==Z": bool(cl[p_tail] == cl[p_prev_close]),
        })
e = pd.DataFrame(rows)
print(f"可检验样本 (节后首日 + 尾根有成交 + 当日有 09:01 + 前日有 15:00): {len(e)}")
if len(e):
    print(f"\n  X == Y (等于当日日盘开盘价, 说明是日盘竞价被错标) : "
          f"{int(e['X==Y'].sum())} / {len(e)}  = {e['X==Y'].mean():.1%}")
    print(f"  X == Z (等于前一交易日收盘, 说明是 padding)        : "
          f"{int(e['X==Z'].sum())} / {len(e)}  = {e['X==Z'].mean():.1%}")
    print("\n按尾根类型:")
    print(e.groupby("尾根")[["X==Y", "X==Z"]].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n完整样本:")
    print(e.sort_values("尾根vol", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    e.to_csv(os.path.join(OUT, "qc_auction_mislabel_evidence.csv"), index=False)

# ================================================== 逐根 K 线证据
hr("逐根 K 线证据")
def dump(symbol, date, lo_m=None, hi_m=None, prev=False):
    si = int(np.flatnonzero(SYMS == symbol)[0])
    dt = np.datetime64(date)
    gi = np.flatnonzero((g_sym == si) & (g_date.to_numpy() == dt))
    if not len(gi):
        print(f"  {symbol} {date}: 无此交易日"); return
    gi = int(gi[0])
    if prev:
        gi -= 1
    s, en = gs[gi], ge[gi]
    sub = pd.DataFrame({
        "datetime": idx[s:en], "mod": mod[s:en], "open": op[s:en], "high": hi[s:en],
        "low": lo[s:en], "close": cl[s:en], "volume": vol[s:en],
    })
    if lo_m is not None:
        sub = sub[(sub["mod"] >= lo_m) & (sub["mod"] <= hi_m)]
    print(f"\n  {symbol}  trading_date={pd.Timestamp(g_date[gi]).date()}  "
          f"(共 {en-s} 根, 夜盘成交 {night_vol[gi]:,.0f})")
    print(sub.drop(columns="mod").to_string(index=False))

print("\n【证据 1】AG 白银, 2025-05-06 (劳动节后首个交易日, 距上一交易日 6 天)")
print("  节前 5/5 晚的夜盘被取消, 但数据里 00:00-02:30 有 151 根 bar")
dump("AG", "2025-05-06", 145, 155)
dump("AG", "2025-05-06", 541, 545)
print("\n  对照: 同品种一个正常交易日 2025-05-07 的同一时段")
dump("AG", "2025-05-07", 145, 155)
dump("AG", "2025-05-07", 541, 545)

print("\n\n【证据 2】BC 国际铜, 2024-01-02 (元旦后首个交易日)")
dump("BC", "2024-01-02", 55, 65)
dump("BC", "2024-01-02", 541, 545)

print("\n\n【证据 3】节后夜盘 padding 的形态 —— AG 2025-05-06 夜盘段前 8 根")
dump("AG", "2025-05-06", 1260, 1268)

hr("郑商所竞价 bar 混入连续交易 —— 逐根对比")
print("\n【证据 4】TA PTA, 21:00 夜盘竞价 bar 不是平 bar")
for d in ["2026-07-28", "2026-07-27"]:
    dump("TA", d, 1260, 1263)
print("\n  对照: 上期所 RB 螺纹钢同期 21:00 bar (标准平 bar)")
for d in ["2026-07-28", "2026-07-27"]:
    dump("RB", d, 1260, 1263)

hr("③ 节后夜盘 padding: 按 gap 拆分, 区分'伪造'与'僵尸品种真实无成交'")
fake = (np.add.reduceat(night.astype(np.int64), gs) > 0) & (night_vol == 0)
fk = pd.DataFrame({"ex": g_ex, "sym": SYMS[g_sym], "date": g_date, "gap": gap,
                   "nbars": np.add.reduceat(night.astype(np.int64), gs)})[fake]
fk["类型"] = np.where(fk.gap > 3, "长假后(伪造)", "普通日/周末后")
print(fk.groupby("类型").agg(组合数=("nbars", "size"), bar数=("nbars", "sum"),
                              品种数=("sym", "nunique")).to_string())
print("\n'普通日/周末后' 那批集中在哪些品种 (前 15):")
z = fk[fk.gap <= 3].groupby("sym").agg(组合数=("nbars", "size"), bar数=("nbars", "sum"))
print(z.sort_values("组合数", ascending=False).head(15).to_string())
print("\n'长假后(伪造)' 那批按交易所:")
print(fk[fk.gap > 3].groupby("ex").agg(组合数=("nbars", "size"), bar数=("nbars", "sum"),
                                        品种数=("sym", "nunique")).to_string())

hr("横向扫描: 各交易所各时段 bar 的形态差异")
for label, m in [("09:00 日盘竞价", 540), ("09:01 日盘首根", 541),
                 ("21:00 夜盘竞价", 1260), ("21:01 夜盘首根", 1261),
                 ("15:00 日盘收盘", 900), ("11:30 上午收盘", 690)]:
    P = bar_at(m)
    ok = P >= 0
    if not ok.any():
        print(f"\n{label}: 全库无此分钟 bar"); continue
    pos = P[ok]
    t = pd.DataFrame({
        "ex": g_ex[ok],
        "flat": (op[pos] == hi[pos]) & (hi[pos] == lo[pos]) & (lo[pos] == cl[pos]),
        "vol": vol[pos],
    })
    g = t.groupby("ex").agg(n=("flat", "size"), 平bar比例=("flat", "mean"),
                            成交量中位=("vol", "median"))
    print(f"\n{label}:")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

hr("DONE")
print(f"[{time.time()-_t0:.1f}s] done")
