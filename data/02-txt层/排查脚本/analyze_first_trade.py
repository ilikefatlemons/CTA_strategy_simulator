"""判定零价缺陷是随机发生还是有条件发生。

已知：零价块的种子 100% 是「某合约某交易时段的第一笔成交」。但这只是充分条件的一半 ——
绝大多数首笔成交是正常的。要给数据端一个能落地排查的范围，必须回答：

  在全部「时段首笔成交」里，出错的比率是多少？这个比率受哪些变量支配？

依次检验：交易所 / 品种 / 年份 / 时段 / gap（首笔成交距开盘隔了几根 bar）/ 首笔成交手数。
若某个变量下比率是 0 与 1 的分野，就是条件；若各处都接近同一个小比率，就是随机。

用法：python data/02-txt层/排查脚本/analyze_first_trade.py
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


def head(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


f = pd.read_csv(TD / "first_trade_summary.csv", dtype={"clearing_day": str})
z = pd.read_csv(TD / "first_trade_zero.csv",
                dtype={"clearing_day": str, "contract": str, "tdate": str, "bartime": str})
f["market_cn"] = f["market"].map(MKT)
z["market_cn"] = z["market"].map(MKT)
f["year"] = f["clearing_day"].str[:4]
z["year"] = z["clearing_day"].str[:4]

# 块的延续（上一根已是零价）不算一次独立故障，单独剔出去
z_seed = z[z["is_continuation"] == 0]

head("0  总基数与总体出错率")
N, Z = f["n_first"].sum(), f["n_zero"].sum()
print(f"全历史「某合约某时段的第一笔成交」共 {N:,} 次")
print(f"其中 OHLC 出现 0 的 {Z:,} 次，总体出错率 = {Z / N:.5f}（约 {Z / N * 10000:.1f} 万分之一）")
print(f"剔除块延续后的独立故障 {len(z_seed):,} 次，独立故障率 = {len(z_seed) / N:.5f}")

head("1  条件一：交易所 —— 是不是所有交易所都会发生")
t = f.groupby("market_cn").agg(首笔成交次数=("n_first", "sum"), 出错=("n_zero", "sum"))
t["出错率"] = (t["出错"] / t["首笔成交次数"]).round(6)
print(t.sort_values("出错率", ascending=False).to_string())
print("\n-> 出错率为 0 的交易所即可完全排除")

head("2  条件二：品种 —— 受影响交易所内部是不是所有品种都会发生")
aff = f[f["market_cn"].isin(["大商所", "广期所"])]
t = aff.groupby(["market_cn", "product"]).agg(首笔成交次数=("n_first", "sum"), 出错=("n_zero", "sum"))
t["出错率"] = (t["出错"] / t["首笔成交次数"]).round(6)
t = t.sort_values("出错率", ascending=False)
print(f"大商所 + 广期所共 {len(t)} 个品种，其中出错率 > 0 的 {(t['出错'] > 0).sum()} 个：")
print(t[t["出错"] > 0].to_string())
print(f"\n出错率恒为 0 的品种（{(t['出错'] == 0).sum()} 个）：",
      ", ".join(t[t["出错"] == 0].reset_index()["product"].tolist()))

head("3  条件三：gap —— 首笔成交是否落在时段第一分钟")
g1_first = f["n_first_gap1"].sum()
g1_zero = f["n_zero_gap1"].sum()
gx_first = N - g1_first
gx_zero = Z - g1_zero
print(f"gap = 1（开盘第一分钟就成交）：{g1_first:,} 次，出错 {g1_zero:,} 次，"
      f"出错率 {g1_zero / max(g1_first, 1):.6f}")
print(f"gap ≥ 2（开盘后隔了几分钟才第一笔）：{gx_first:,} 次，出错 {gx_zero:,} 次，"
      f"出错率 {gx_zero / max(gx_first, 1):.6f}")
if g1_zero == 0:
    print("\n-> 开盘第一分钟就有成交的合约【从不】出错，这是一条硬条件")

head("4  条件四：只看受影响品种 + gap ≥ 2 时，出错率有多高")
a2_first = aff["n_first"].sum() - aff["n_first_gap1"].sum()
a2_zero = aff["n_zero"].sum() - aff["n_zero_gap1"].sum()
print(f"大商所+广期所、gap ≥ 2 的首笔成交：{a2_first:,} 次，出错 {a2_zero:,} 次，"
      f"出错率 {a2_zero / max(a2_first, 1):.5f}")
onlyaff = t[t["出错"] > 0].reset_index()["product"].tolist()
a3 = aff[aff["product"].isin(onlyaff)]
a3_first = a3["n_first"].sum() - a3["n_first_gap1"].sum()
a3_zero = a3["n_zero"].sum() - a3["n_zero_gap1"].sum()
print(f"再限定到出错过的 {len(onlyaff)} 个品种：{a3_first:,} 次，出错 {a3_zero:,} 次，"
      f"出错率 {a3_zero / max(a3_first, 1):.5f}")

head("5  条件五：gap 的具体分布 —— 出错的首笔成交隔了多久")
zz = z_seed.copy()
print(zz["gap"].describe(percentiles=[.25, .5, .75, .9]).to_string())
bins = [0, 1, 2, 5, 15, 60, 200, 10 ** 9]
lbl = ["1", "2", "3–5", "6–15", "16–60", "61–200", ">200"]
print("\n出错样本的 gap 分档：")
print(zz.groupby(pd.cut(zz["gap"], bins, labels=lbl), observed=True).size().rename("次数").to_string())

head("6  条件六：首笔成交的手数")
print(zz["vol"].describe(percentiles=[.25, .5, .75, .9, .99]).to_string())

head("7  条件七：时段与年份")
print("按时段：")
t = f.groupby("session").agg(首笔成交=("n_first", "sum"), 出错=("n_zero", "sum"))
t["出错率"] = (t["出错"] / t["首笔成交"]).round(6)
print(t.rename(index={"N": "夜盘", "D": "日盘"}).to_string())
print("\n按年（仅大商所+广期所）：")
t = aff.groupby("year").agg(首笔成交=("n_first", "sum"), 出错=("n_zero", "sum"))
t["出错率"] = (t["出错"] / t["首笔成交"]).round(5)
print(t.to_string())

head("8  条件八：是否集中在少数交易日")
byday = z_seed.groupby("clearing_day").size()
alldays = f["clearing_day"].nunique()
print(f"有故障的交易日 {len(byday):,} / {alldays:,} = {len(byday) / alldays:.3f}")
print(f"单日故障次数：中位 {byday.median():.0f}，p90 {byday.quantile(.9):.0f}，max {byday.max()}")
print("\n故障最多的 10 个交易日：")
print(byday.sort_values(ascending=False).head(10).rename("故障次数").to_string())

head("9  结论：随机还是有条件")
print("按上面的分层逐条看：出错率在哪一层出现 0 / 非 0 的分野，哪一层就是条件；")
print("在最细的一层里如果比率仍然稳定在同一个小数值，那一层内部就是随机的。")

z_seed.drop(columns=["stream"]).to_csv(OUT / "P1-9d_出错的首笔成交明细.csv",
                                       index=False, encoding="utf-8-sig")
print(f"\n-> P1-9d_出错的首笔成交明细.csv（{len(z_seed):,} 行）")
