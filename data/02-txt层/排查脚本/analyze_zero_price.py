"""零价 K 线（OHLC 出现 0）的重新定性。

按根计数会严重误导。实际有两种形态：

  形态 A「四价全 0」—— close 也是 0，被「无成交沿用上一价」的逻辑一路复制，
      一次故障污染几十到几百根 K 线。起点是一根真实成交的 bar，
      volume / value / vwap 全对，只有 OHLC 是 0。
  形态 B「只有 open 和 low 为 0」—— close 有效，不传染，孤立单根。

所以要分开数三件事：
  · 独立故障次数（块数 + 孤立根数）
  · 真正丢失价格的成交 bar（本该有 OHLC 却是 0）
  · 被传染的 bar（本该复制上一有效价，却复制了 0）

用法：python data/02-txt层/排查脚本/analyze_zero_price.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TD = ROOT / "data" / "02-txt层" / "排查证据"
OUT = ROOT / "report" / "数据" / "04-两层对账" / "明细"
MKT = {"XSGE": "上期所", "XSIE": "能源中心", "XDCE": "大商所",
       "XZCE": "郑商所", "XGFE": "广期所", "CCFX": "中金所"}
FIELD = {1: "open", 2: "high", 4: "low", 8: "close"}


def head(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def mask_str(m: int) -> str:
    return "+".join(v for k, v in FIELD.items() if m & k) or "-"


b = pd.read_csv(TD / "zero_price_blocks.csv",
                dtype={"clearing_day": str, "contract": str, "start_tdate": str,
                       "start_bt": str, "end_tdate": str, "end_bt": str})
p = pd.read_csv(TD / "zero_price_isolated.csv",
                dtype={"clearing_day": str, "contract": str, "tdate": str, "bartime": str})
d = pd.read_csv(TD / "zero_price_daily.csv", dtype={"clearing_day": str})

print(f"扫描覆盖 {len(d):,} 个交易日文件")
print(f"OHLC 出现 0 的 bar 合计 {d['n_bars_any_zero'].sum():,} 根，分成两种形态：")
print(f"  形态 A（close 也是 0，会传染）：{d['n_bars_close_zero'].sum():,} 根，"
      f"归入 {d['n_blocks'].sum():,} 个块")
print(f"  形态 B（close 有效，不传染）：{d['n_isolated'].sum():,} 根，孤立单根")
assert d["n_bars_close_zero"].sum() + d["n_isolated"].sum() == d["n_bars_any_zero"].sum()


# ---------------------------------------------------------------- 合并跨文件的块
b = b.sort_values(["contract", "clearing_day"]).reset_index(drop=True)
b["prev_contract"] = b["contract"].shift(1)
b["prev_at_end"] = b["at_file_end"].shift(1).fillna(0).astype(int)
b["prev_day"] = b["clearing_day"].shift(1)
# 只有【相邻的两个交易日文件】才可能是同一个块被切开，所以要按交易日序号相差 1 来卡
day_idx = {day: i for i, day in enumerate(sorted(d["clearing_day"].tolist()))}
b["adjacent"] = (b["clearing_day"].map(day_idx) - b["prev_day"].map(day_idx)) == 1
b["is_cont"] = ((b["contract"] == b["prev_contract"]) & (b["prev_at_end"] == 1) &
                (b["at_file_start"] == 1) & b["adjacent"].fillna(False))
b["grp"] = (~b["is_cont"]).cumsum()

g = b.groupby("grp").agg(
    market=("market", "first"), product=("product", "first"), contract=("contract", "first"),
    start_day=("clearing_day", "first"), start_tdate=("start_tdate", "first"),
    start_bt=("start_bt", "first"), end_day=("clearing_day", "last"),
    end_tdate=("end_tdate", "last"), end_bt=("end_bt", "last"),
    n_files=("clearing_day", "nunique"), n_bars=("n_bars", "sum"),
    n_bars_with_vol=("n_bars_with_vol", "sum"), sum_vol=("sum_vol", "sum"),
    seed_vol=("seed_vol", "first"), seed_vwap=("seed_vwap", "first"),
    seed_mask=("seed_mask", "first"), prev_close=("prev_close", "first"),
    next_open=("next_open", "last"),
).reset_index(drop=True)
g["market_cn"] = g["market"].map(MKT)
g["year"] = g["start_day"].str[:4]

head("1  形态 A：独立故障次数与传染放大")
print(f"独立块数：{len(g):,}（合并前 {len(b):,}，其中 {int(b['is_cont'].sum()):,} 个是被文件边界切开的续块）")
print(f"跨越多个交易日文件的块：{(g['n_files'] > 1).sum():,} 个")
lost = int(g["n_bars_with_vol"].sum())
tot = int(g["n_bars"].sum())
print(f"\n零价 bar {tot:,} 根，拆成：")
print(f"  · 真正丢失价格的成交 bar（块内 volume > 0）：{lost:,} 根")
print(f"  · 被传染的无成交 bar（本该复制上一有效价）：{tot - lost:,} 根")
print(f"  传染放大倍数：{tot / max(lost, 1):.1f} 倍")
print(f"\n块的种子字段形态：")
print(g["seed_mask"].map(mask_str).value_counts().rename("块数").to_string())

head("2  块长分布（一次故障污染多少根 K 线）")
print(g["n_bars"].describe(percentiles=[.25, .5, .75, .9, .99]).to_string())
bins = [0, 1, 5, 20, 60, 120, 240, 480, 10 ** 9]
lbl = ["1 根", "2–5", "6–20", "21–60", "61–120", "121–240", "241–480", ">480"]
print("\n分档：")
print(g.groupby(pd.cut(g["n_bars"], bins, labels=lbl), observed=True)
      .agg(块数=("n_bars", "size"), 累计bar=("n_bars", "sum")).to_string())
print(f"\n跨到次日的块（起止不在同一日历日）：{(g['start_tdate'] != g['end_tdate']).sum():,} 个")

head("3  种子是不是一笔真实成交，价格能否佐证")
print(f"种子 bar 带成交量：{(g['seed_vol'] > 0).sum():,} / {len(g):,}")
print(f"种子 bar 的 vwap > 0：{(g['seed_vwap'] > 0).sum():,} / {len(g):,}")
ok = g[(g["seed_vwap"] > 0) & g["prev_close"].notna() & (g["prev_close"] > 0)]
dev = (ok["seed_vwap"] - ok["prev_close"]).abs() / ok["prev_close"]
print(f"种子 vwap 与块前最后有效价的相对偏离：中位 {dev.median():.5f}，p95 {dev.quantile(.95):.5f}")
print("  -> vwap 是这一分钟的真实成交均价，说明行情数据本身没丢，是 OHLC 字段没写进去")

head("4  形态 A 的影响范围")
print("按交易所：")
print(g.groupby("market_cn").agg(块数=("n_bars", "size"), 零价bar=("n_bars", "sum"),
                                 丢价成交bar=("n_bars_with_vol", "sum"),
                                 品种数=("product", "nunique"), 合约数=("contract", "nunique"))
      .sort_values("零价bar", ascending=False).to_string())
print("\n按年：")
print(g.groupby("year").agg(块数=("n_bars", "size"), 零价bar=("n_bars", "sum"),
                            丢价成交bar=("n_bars_with_vol", "sum"),
                            涉及品种=("product", "nunique")).to_string())
print("\n按品种（前 15）：")
print(g.groupby(["market_cn", "product"])
      .agg(块数=("n_bars", "size"), 零价bar=("n_bars", "sum"), 丢价成交bar=("n_bars_with_vol", "sum"))
      .sort_values("零价bar", ascending=False).head(15).to_string())

head("5  块起始时刻 —— 故障发生在什么时候")
print(g["start_bt"].value_counts().head(10).rename("块数").to_string())
sess = pd.cut(g["start_bt"].str[:2].astype(int), [-1, 8, 11, 14, 20, 23],
              labels=["00–08 夜盘后半", "09–11 上午", "12–14 下午", "15–20 收盘后", "21–23 夜盘前半"])
print("\n按时段：")
print(sess.value_counts().rename("块数").to_string())

head("6  形态 B：孤立坏根")
if len(p):
    p["market_cn"] = p["market"].map(MKT)
    p["year"] = p["clearing_day"].str[:4]
    print(f"合计 {len(p):,} 根，涉及 {p['clearing_day'].nunique():,} 个交易日、"
          f"{p['product'].nunique()} 个品种、{p['contract'].nunique():,} 个合约")
    print("\n字段形态（哪几列为 0）：")
    print(p["mask"].map(mask_str).value_counts().rename("根数").to_string())
    print("\n按交易所：")
    print(p.groupby("market_cn").size().sort_values(ascending=False).rename("根数").to_string())
    print("\n按年：")
    print(p.groupby("year").size().rename("根数").to_string())
    print("\n起始时刻（前 8）：")
    print(p["bartime"].value_counts().head(8).rename("根数").to_string())
    print(f"\n这些 bar 全部带成交量：{(p['volume'] > 0).all()}；"
          f"high/close 均有效：{((p['high'] > 0) & (p['close'] > 0)).all()}")
else:
    print("无")

head("7  打在活跃合约还是清淡远月上（决定影响有多大）")
rw = pd.read_csv(TD / "raw_scan_by_contract.csv", dtype={"clearing_day": str, "contract": str},
                 usecols=["clearing_day", "product", "contract", "sum_vol"])
tot_v = rw.groupby(["clearing_day", "product"])["sum_vol"].sum().rename("prod_vol")
rw = rw.join(tot_v, on=["clearing_day", "product"])
rw["share"] = rw["sum_vol"] / rw["prod_vol"].replace(0, np.nan)
g = g.merge(rw[["clearing_day", "contract", "sum_vol", "share"]].rename(
    columns={"clearing_day": "start_day", "sum_vol": "contract_day_vol", "share": "vol_share"}),
    on=["start_day", "contract"], how="left")
g["合约地位"] = pd.cut(g["vol_share"].fillna(0), [-.01, .01, .1, .4, 1.01],
                       labels=["清淡(<1%)", "次要(1-10%)", "活跃(10-40%)", "主力(>40%)"])
print(g.groupby("合约地位", observed=True)
      .agg(块数=("n_bars", "size"), 零价bar=("n_bars", "sum"),
           丢价成交bar=("n_bars_with_vol", "sum"),
           该合约当日成交中位=("contract_day_vol", "median")).to_string())

head("8  受影响的交易日、合约与最长的块")
print(f"至少有一处零价的交易日：{d[(d['n_bars_any_zero'] > 0)]['clearing_day'].nunique():,} / {len(d):,}")
print(f"形态 A 受影响合约 {g['contract'].nunique():,} 个 / 品种 {g['product'].nunique()} 个")
print("\n单块最长的 8 个：")
print(g.nlargest(8, "n_bars")[["market_cn", "product", "contract", "start_tdate", "start_bt",
                               "end_tdate", "end_bt", "n_bars", "n_bars_with_vol",
                               "seed_vwap", "prev_close", "next_open"]].to_string(index=False))

cols = ["market", "market_cn", "product", "contract", "start_day", "start_tdate", "start_bt",
        "end_day", "end_tdate", "end_bt", "n_files", "n_bars", "n_bars_with_vol", "sum_vol",
        "seed_vol", "seed_vwap", "prev_close", "next_open", "contract_day_vol", "vol_share",
        "合约地位"]
g[cols].sort_values(["start_day", "market", "contract"]).to_csv(
    OUT / "P1-9a_零价K线块明细.csv", index=False, encoding="utf-8-sig")
print(f"\n-> P1-9a_零价K线块明细.csv（{len(g):,} 行）")
if len(p):
    p.drop(columns=["stream"]).to_csv(OUT / "P1-9b_零价K线孤立坏根.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"-> P1-9b_零价K线孤立坏根.csv（{len(p):,} 行）")
