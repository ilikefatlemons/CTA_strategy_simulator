# -*- coding: utf-8 -*-
"""
按交易所核查集合竞价 / 夜盘 bar 的异常。

要核实的四条:
  1. 2023-05 起夜盘品种开了双集合竞价, 但数据里没有 09:00 日盘竞价 bar
  2. 郑商所夜盘品种的 21:00 竞价 bar 不是平 bar (O=H=L=C), 而是有长度
  3. 上期所夜盘品种节后有夜盘 padding (实际无夜盘), 价格不变且成交量恒为 0
  4. 上期所夜盘品种节后 09:00 竞价数据被标到 02:30

外加: 横向扫描各交易所, 找同类型的其它异常。
"""
import os
import time

import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"

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
def log(*a): print(f"[{time.time()-_t0:6.1f}s]", *a, flush=True)
def hr(t):
    print("\n" + "=" * 96, flush=True); print(t, flush=True); print("=" * 96, flush=True)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)

df = pd.read_pickle(PKL); log("loaded")
sym = df["underlying_symbol"].astype("category")
sc = sym.cat.codes.to_numpy(); SYMS = np.asarray(sym.cat.categories)
SID = {s: i for i, s in enumerate(SYMS)}
EXOF = np.array([SYM2EX.get(str(s), "?") for s in SYMS])
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
cal = idx.normalize().to_numpy()
td = df["trading_date"].to_numpy()
op = df["open"].to_numpy(); hi = df["high"].to_numpy()
lo = df["low"].to_numpy(); cl = df["close"].to_numpy(); vol = df["volume"].to_numpy()
d_codes, d_uniq = pd.factorize(df["trading_date"], sort=True)
d_uniq = pd.DatetimeIndex(d_uniq); ND = len(d_uniq)
yr = d_uniq.year.to_numpy()[d_codes]
log(f"{len(df):,} rows, {len(SYMS)} symbols")
print("未归类的品种:", [str(s) for s in SYMS if SYM2EX.get(str(s)) is None])

night = (mod >= 21 * 60) | (mod <= 179)
# (品种,交易日) 分组
key = sc.astype(np.int64) * ND + d_codes
assert (np.diff(key) >= 0).all()
b = np.flatnonzero(np.diff(key)) + 1
gs = np.concatenate([[0], b]); ge = np.concatenate([b, [len(key)]])
gkey = key[gs]; g_sym = (gkey // ND).astype(int); g_dc = (gkey % ND).astype(int)
g_date = d_uniq[g_dc]; g_ex = EXOF[g_sym]; g_year = pd.DatetimeIndex(g_date).year

has_night = np.add.reduceat(night.astype(np.int64), gs) > 0
night_vol = np.add.reduceat(np.where(night, vol, 0.0), gs)
# 每个 (品种,交易日) 是否有某个具体分钟的 bar
def has_min(m):
    return np.add.reduceat((mod == m).astype(np.int64), gs) > 0
def bar_at(m):
    """返回该分钟 bar 在原数组里的位置, 没有则 -1"""
    pos = np.full(len(gs), -1, np.int64)
    hitmask = mod == m
    hits = np.flatnonzero(hitmask)
    if len(hits):
        gi = np.searchsorted(gs, hits, side="right") - 1
        pos[gi] = hits
    return pos

# 节后首日: 距上一交易日间隔
prev_date = np.full(len(gs), np.datetime64("NaT"), "datetime64[ns]")
same_prev = np.concatenate([[False], g_sym[1:] == g_sym[:-1]])
prev_date[1:] = g_date.to_numpy()[:-1]
gap = np.where(same_prev, (g_date.to_numpy() - prev_date) / np.timedelta64(1, "D"), np.nan)

# ================================================== 1. 09:00 日盘竞价 bar
hr("① 09:00 日盘集合竞价 bar 是否存在")
h0900 = has_min(9 * 60)
t = pd.DataFrame({"ex": g_ex, "sym": SYMS[g_sym], "year": g_year,
                  "night": has_night, "h0900": h0900})
print("按 交易所 × 是否有夜盘 × 年份 统计 '当日存在 09:00 bar' 的比例:")
piv = t.pivot_table(index=["ex", "night"], columns="year", values="h0900", aggfunc="mean")
print(piv.round(3).to_string())

print("\n只看有夜盘的品种, 逐品种逐年 (2021 之后):")
tn = t[t.night]
piv2 = tn.pivot_table(index=["ex", "sym"], columns="year", values="h0900", aggfunc="mean")
cols = [c for c in piv2.columns if c >= 2021]
print(piv2[cols].round(3).to_string())

# ================================================== 2. 21:00 夜盘竞价 bar 形态
hr("② 21:00 夜盘集合竞价 bar 的形态 (是否为 O=H=L=C 的平 bar)")
p2100 = bar_at(21 * 60)
m = p2100 >= 0
pos = p2100[m]
flat = (op[pos] == hi[pos]) & (hi[pos] == lo[pos]) & (lo[pos] == cl[pos])
rng = np.where(cl[pos] > 0, (hi[pos] - lo[pos]) / cl[pos], np.nan)
a = pd.DataFrame({"ex": g_ex[m], "sym": SYMS[g_sym[m]], "year": g_year[m],
                  "flat": flat, "rng": rng, "vol": vol[pos]})
print("按交易所: 21:00 bar 为平 bar 的比例 / 平均振幅 / 成交量中位")
print(a.groupby("ex").agg(n=("flat", "size"), flat_rate=("flat", "mean"),
                          mean_range=("rng", "mean"), med_vol=("vol", "median")
                          ).to_string(float_format=lambda x: f"{x:.5f}"))
print("\n逐品种 (平 bar 比例最低的 20 个):")
per = a.groupby(["ex", "sym"]).agg(n=("flat", "size"), flat_rate=("flat", "mean"),
                                   mean_range=("rng", "mean"), med_vol=("vol", "median"))
print(per.sort_values("flat_rate").head(20).to_string(float_format=lambda x: f"{x:.5f}"))

print("\n对照: 无夜盘品种的 09:00 bar 形态 (这是标准竞价 bar 应有的样子)")
p0900 = bar_at(9 * 60)
m2 = (p0900 >= 0) & (~has_night)
pos2 = p0900[m2]
flat2 = (op[pos2] == hi[pos2]) & (hi[pos2] == lo[pos2]) & (lo[pos2] == cl[pos2])
c2 = pd.DataFrame({"ex": g_ex[m2], "flat": flat2,
                   "rng": np.where(cl[pos2] > 0, (hi[pos2]-lo[pos2])/cl[pos2], np.nan)})
print(c2.groupby("ex").agg(n=("flat", "size"), flat_rate=("flat", "mean"),
                           mean_range=("rng", "mean")).to_string(float_format=lambda x: f"{x:.5f}"))

# ================================================== 3. 节后夜盘 padding
hr("③ 节后被取消的夜盘却有 bar (成交量恒为 0) —— 按交易所")
fake_night = has_night & (night_vol == 0)
f = pd.DataFrame({"ex": g_ex, "sym": SYMS[g_sym], "year": g_year, "date": g_date,
                  "gap": gap, "fake": fake_night,
                  "nbars": np.add.reduceat(night.astype(np.int64), gs)})
ff = f[f.fake]
print(f"命中 (品种,交易日) 组合: {len(ff):,}, 涉及 bar {int(ff.nbars.sum()):,}")
print("\n按交易所:")
print(ff.groupby("ex").agg(组合数=("fake", "size"), bar数=("nbars", "sum"),
                           品种数=("sym", "nunique")).to_string())
print("\n按交易所 × 年份 (组合数):")
print(ff.pivot_table(index="ex", columns="year", values="fake", aggfunc="size",
                     fill_value=0).to_string())
print("\n间隔天数分布 (gap = 距上一交易日):")
print(ff.gap.value_counts().sort_index().head(12).to_string())

# ================================================== 4. 节后 02:30 / 01:00 尾根异常
hr("④ 夜盘尾根 (02:30 / 01:00) 在节后是否出现异常成交")
for label, minute, syms in [("02:30", 150, ["AU", "AG", "SC"]),
                            ("01:00", 60, ["CU", "AL", "ZN", "PB", "NI", "SN", "SS", "BC"])]:
    pm = bar_at(minute)
    ok = pm >= 0
    sel = ok & np.isin(SYMS[g_sym], syms)
    pos = pm[sel]
    sub = pd.DataFrame({
        "sym": SYMS[g_sym[sel]], "date": g_date[sel], "gap": gap[sel],
        "vol": vol[pos], "night_vol": night_vol[sel],
        "flat": (op[pos] == hi[pos]) & (hi[pos] == lo[pos]) & (lo[pos] == cl[pos]),
    })
    sub["holiday"] = sub.gap > 3
    print(f"\n--- {label} 尾根, {len(syms)} 个品种, n={len(sub):,} ---")
    print(sub.groupby("holiday").agg(n=("vol", "size"), 成交量中位=("vol", "median"),
                                     成交量均值=("vol", "mean"),
                                     零成交比例=("vol", lambda x: float((x == 0).mean())),
                                     平bar比例=("flat", "mean"),
                                     ).to_string(float_format=lambda x: f"{x:.4f}"))
    big = sub[sub.holiday & (sub.vol > 0)].sort_values("vol", ascending=False)
    print(f"  节后首日 {label} 尾根有成交的 (品种,交易日): {len(big)} 个")
    if len(big):
        print(big.head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

hr("DONE"); log("done")
