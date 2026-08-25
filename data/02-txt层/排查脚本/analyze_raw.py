"""把 scan_raw.awk 的三条聚合流拆成按检查项分类的明细清单。

输入  data/02-txt层/排查证据/raw_scan_daily.csv          (流 A)
      data/02-txt层/排查证据/raw_scan_contract_value.csv (流 B)
      data/02-txt层/排查证据/raw_scan_prebar.csv         (流 C)
输出  report/数据/02-txt层/明细/01..08_*.csv
      并在 stdout 打印报告正文要引用的全部汇总数字。

用法：python data/02-txt层/排查脚本/analyze_raw.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TD = ROOT / "data" / "02-txt层" / "排查证据"
OUT = ROOT / "report" / "数据" / "02-txt层" / "明细"
OUT.mkdir(parents=True, exist_ok=True)

# 2023-05-26 起，上期所 / 能源中心 / 大商所对有夜盘的品种新增日盘集合竞价；
# 郑商所只开了 08:55-08:59 撤单窗口，没有日盘集合竞价撮合，故不在此列。
AUCTION_RULE_DAY = "20230526"
DAY_AUCTION_MARKETS = ["XSGE", "XSIE", "XDCE"]
SCHEMA_SWITCH_DAY = "20220922"
TRUNCATED_9 = ["AO", "BR", "EC", "IM", "LC", "PX", "SH", "SI", "TL"]
MKT_CN = {"XSGE": "上期所", "XSIE": "能源中心", "XDCE": "大商所",
          "XZCE": "郑商所", "XGFE": "广期所", "CCFX": "中金所"}
WD_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 中国期货市场休市日（各交易所每年的休市安排公告）。只覆盖 2022-09-22 换版之后，
# 因为"文件缺失"这项检查只在这个区间内下结论。脚本会把这张表跟数据里的直接判据
# 对照，冲突会被打印出来 —— 表错了会立刻暴露，不会静悄悄地污染结论。
HOLIDAYS_CN = {
    "2022国庆": ("20221001", "20221007"),
    "2023元旦": ("20230101", "20230102"), "2023春节": ("20230121", "20230127"),
    "2023清明": ("20230405", "20230405"), "2023劳动": ("20230429", "20230503"),
    "2023端午": ("20230622", "20230624"), "2023中秋国庆": ("20230929", "20231006"),
    "2024元旦": ("20240101", "20240101"), "2024春节": ("20240209", "20240217"),
    "2024清明": ("20240404", "20240406"), "2024劳动": ("20240501", "20240505"),
    "2024端午": ("20240608", "20240610"), "2024中秋": ("20240915", "20240917"),
    "2024国庆": ("20241001", "20241007"),
    "2025元旦": ("20250101", "20250101"), "2025春节": ("20250128", "20250204"),
    "2025清明": ("20250404", "20250406"), "2025劳动": ("20250501", "20250505"),
    "2025端午": ("20250531", "20250602"), "2025国庆中秋": ("20251001", "20251008"),
    "2026元旦": ("20260101", "20260102"), "2026春节": ("20260216", "20260223"),
    "2026清明": ("20260404", "20260406"), "2026劳动": ("20260501", "20260505"),
    "2026端午": ("20260619", "20260621"),
}


def holiday_of(day: str) -> str:
    for name, (a, b) in HOLIDAYS_CN.items():
        if a <= day <= b:
            return name
    return ""


def dump(df: pd.DataFrame, name: str) -> pd.DataFrame:
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> {name}  ({len(df):,} 行)")
    return df


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- 读入 + 公共派生
A = pd.read_csv(TD / "raw_scan_daily.csv", dtype={"clearing_day": str, "tdate": str,
                                                  "min_bt": str, "max_bt": str,
                                                  "min_pre_voltime": str,
                                                  "max_pre_voltime": str})
B = pd.read_csv(TD / "raw_scan_contract_value.csv", dtype={"clearing_day": str})
C = pd.read_csv(TD / "raw_scan_prebar.csv", dtype={"clearing_day": str, "pre_bt": str,
                                                   "pad_bt": str, "day0_bt": str})
A["dt"] = pd.to_datetime(A["clearing_day"], format="%Y%m%d")
A["year"] = A["dt"].dt.year
A["market_cn"] = A["market"].map(MKT_CN)
A["is_night_block"] = A["tdate"] != A["clearing_day"]

days = pd.Index(sorted(A["clearing_day"].unique()))
day_dt = pd.to_datetime(days, format="%Y%m%d")
print(f"扫描覆盖 {len(days):,} 个交易日文件：{days[0]} ~ {days[-1]}")
print(f"流 A {len(A):,} 行 / 流 B {len(B):,} 行 / 流 C {len(C):,} 行")

# 每个文件的"非当日 TDATE"块（= 夜盘块）总 bar 数与总成交量
night = (A[A["is_night_block"]]
         .groupby("clearing_day")
         .agg(night_bars=("bars", "sum"), night_vol=("sum_vol", "sum"))
         .reindex(days).fillna(0))
night["padded"] = (night["night_bars"] > 0) & (night["night_vol"] == 0)

prev_day = pd.Series(days, index=days).shift(1)
gap = (pd.Series(day_dt, index=days) - pd.Series(pd.to_datetime(prev_day, format="%Y%m%d").values,
                                                 index=days)).dt.days


# ================================================================ 05 缺失交易日文件
# 先算，因为 02 要用它把"假期"和"文件缺失"分开
head("05 原始文件整日缺失")
all_wd = pd.date_range(day_dt[0], day_dt[-1], freq="B")          # 全部工作日
have = set(days)
missing = [d for d in all_wd if d.strftime("%Y%m%d") not in have]

rec = []
for d in missing:
    x = d.strftime("%Y%m%d")
    later = days[days > x]
    nxt = str(later[0]) if len(later) else None
    earlier = days[days < x]
    prv = str(earlier[-1]) if len(earlier) else None
    # 判据：缺失日 X 的次一个可用文件里，TDATE=X 的夜盘块有真实成交量
    #      -> X 当晚开过夜盘 -> X 是交易日 -> 文件缺失
    #      成交量为 0 -> 该块是 padding -> X 当天休市（真假期）
    blk = A[(A["clearing_day"] == nxt) & (A["tdate"] == x)] if nxt is not None else A.iloc[:0]
    if len(blk):
        verdict = "已证实是交易日（文件缺失）" if blk["sum_vol"].sum() > 0 else "已证实是假期（次日文件里该块是 padding）"
        ev = blk["sum_vol"].sum()
    else:
        verdict, ev = "无直接判据（连续缺失区间内部）", np.nan
    rec.append(dict(missing_day=x, weekday=WD_CN[d.weekday()], prev_file=prv, next_file=nxt,
                    tdate_block_vol_in_next_file=ev, direct_verdict=verdict,
                    holiday=holiday_of(x)))
miss = pd.DataFrame(rec)

# 直接判据只能覆盖"紧邻下一个可用文件"的那一天；连续缺失区间的内部日子要靠休市公告
# （HOLIDAYS_CN）来分。两者重叠的部分正好用来互相校验。
conflict = miss[(miss["direct_verdict"].str.startswith("已证实是交易日")) & (miss["holiday"] != "")]
print(f"休市表与数据直接判据的冲突：{len(conflict)} 条"
      + ("（休市表有误，需修正）" if len(conflict) else "（无冲突）"))
if len(conflict):
    print(conflict[["missing_day", "holiday", "tdate_block_vol_in_next_file"]].to_string(index=False))

after_switch = miss["missing_day"] >= SCHEMA_SWITCH_DAY
miss["verdict"] = np.where(
    miss["holiday"] != "", "休市（交易所公告）",
    np.where(miss["direct_verdict"].str.startswith("已证实是交易日"), "交易日 · 文件缺失（数据直接证实）",
             np.where(after_switch, "交易日 · 文件缺失（不在休市公告内）", "休市或缺失（2022-09-22 前，未逐日判定）")))

md = pd.to_datetime(miss["missing_day"], format="%Y%m%d")
miss["run_id"] = ((md - md.shift(1)).dt.days.fillna(99) > 4).cumsum().values

miss_real = miss[miss["verdict"].str.startswith("交易日")].copy()
dump(miss, "05_缺失交易日文件.csv")
print(f"\n区间内全部工作日 {len(all_wd):,}，有文件 {len(days):,}，无文件 {len(missing):,}")
print(miss["verdict"].value_counts().to_string())
print(f"\n判定为「交易日但文件缺失」共 {len(miss_real)} 天，按年份：")
print(miss_real.groupby(miss_real["missing_day"].str[:4].values).size().to_string())
print("\n连续缺失区间（≥2 个交易日）：")
for _, g in miss_real.groupby("run_id"):
    if len(g) >= 2:
        d0, d1 = g["missing_day"].tolist()[0], g["missing_day"].tolist()[-1]
        print(f"  {d0} ~ {d1}  缺失交易日 {len(g)} 个")
print("\n全部缺失交易日：")
print("  " + "  ".join(f"{r.missing_day}({r.weekday})" for r in miss_real.itertuples()))

# 逐年文件数 —— 中国期货市场每年约 242~250 个交易日，明显偏低的年份就是丢了文件。
# 旧格式期的缺失日几乎都无法用夜盘块直接判定（那时节后首日根本不生成夜盘块），
# 所以用"逐年文件数"这个独立口径来交叉验证旧口径是否完整。
fy = pd.DataFrame({"year": days.str[:4], "d": 1}).groupby("year")["d"].sum()
fy = fy.to_frame("文件数")
fy["缺失交易日"] = miss_real.groupby(miss_real["missing_day"].str[:4].values).size().reindex(fy.index).fillna(0).astype(int)
dump(fy.reset_index(), "05b_逐年文件数.csv")
print("\n逐年文件数（2026 只到 08-11，2010 起于 01-04）：")
print(fy.T.to_string())

missing_real = set(miss_real["missing_day"])


# ================================================================ 02 节后夜盘 padding
head("02 节假日后的夜盘被 padding")
# padding 的判据必须下在【交易日】这一层：整个文件的非当日 TDATE 块成交量全为 0。
# 只看单个品种会把"僵尸品种整夜真的没人交易"也算进来 —— 那是市场事实，不是伪造。
padded_days = set(night.index[night["padded"]])
pad = A[A["is_night_block"] & A["clearing_day"].isin(list(padded_days))].copy()
pad["weekday"] = pad["dt"].dt.weekday.map(lambda i: WD_CN[i])
pad["prev_file"] = pad["clearing_day"].map(prev_day)
pad["gap_days"] = pad["clearing_day"].map(gap)
pad_out = pad[["clearing_day", "weekday", "tdate", "market", "market_cn", "product",
               "n_contracts", "bars", "min_bt", "max_bt", "prev_file", "gap_days"]] \
    .sort_values(["clearing_day", "market", "product"])
dump(pad_out, "02_节后夜盘padding.csv")

pad_day = (pad.groupby("clearing_day")
           .agg(bars=("bars", "sum"), n_prod=("product", "nunique"),
                n_mkt=("market", "nunique"), tdates=("tdate", lambda s: ",".join(sorted(set(s)))))
           .reset_index())
pad_day["weekday"] = pd.to_datetime(pad_day["clearing_day"], format="%Y%m%d").dt.weekday.map(lambda i: WD_CN[i])
pad_day["prev_file"] = pad_day["clearing_day"].map(prev_day)
pad_day["era"] = np.where(pad_day["clearing_day"] < SCHEMA_SWITCH_DAY, "旧格式(含Offset)", "新格式")
dump(pad_day, "02b_节后夜盘padding_按交易日汇总.csv")

print(f"出现 padding 夜盘块的交易日：{len(pad_day)} 个，累计伪造 {pad['bars'].sum():,} 根 K 线")
print(f"其中旧格式期（< {SCHEMA_SWITCH_DAY}）：{(pad_day['era'] == '旧格式(含Offset)').sum()} 个")
print(f"    新格式期（>= {SCHEMA_SWITCH_DAY}）：{(pad_day['era'] == '新格式').sum()} 个")
print("\n按交易所拆分：")
print(pad.groupby("market_cn").agg(组合数=("bars", "size"), K线数=("bars", "sum"),
                                   品种数=("product", "nunique")).sort_values("K线数", ascending=False).to_string())
print("\n逐个 padding 交易日：")
print(pad_day[["clearing_day", "weekday", "prev_file", "tdates", "n_mkt", "n_prod", "bars"]].to_string(index=False))

# 新旧口径对比：节后首日里有多少被 padding 了。
# 「节后首日」完全从数据推导：与上一个可用文件的日历间隔超过正常值（周一 3 天、其余 1 天），
# 且中间隔的不是"文件缺失"的交易日。不依赖任何硬编码的假期表。
brk = pd.DataFrame({"clearing_day": days, "dt": day_dt})
brk["weekday"] = brk["dt"].dt.weekday
brk["gap"] = brk["clearing_day"].map(gap)
brk["is_break"] = brk["gap"] > np.where(brk["weekday"] == 0, 3, 1)
brk["prev_file"] = brk["clearing_day"].map(prev_day)


def _spans_missing_file(row) -> bool:
    # 注意用 row["dt"] 而不是 row.dt —— 后者会撞上 pandas 的 .dt 访问器
    if not row["is_break"] or row["prev_file"] is None or pd.isna(row["prev_file"]):
        return False
    lo = (pd.to_datetime(row["prev_file"], format="%Y%m%d") + pd.Timedelta(days=1)).strftime("%Y%m%d")
    hi = (row["dt"] - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return any(lo <= m <= hi for m in missing_real)


brk["spans_missing"] = brk.apply(_spans_missing_file, axis=1)
after_break = brk[brk["is_break"] & ~brk["spans_missing"]].copy()
after_break["era"] = np.where(after_break["clearing_day"] < SCHEMA_SWITCH_DAY, "旧格式", "新格式")
after_break["padded"] = after_break["clearing_day"].isin(list(padded_days))
dump(after_break[["clearing_day", "weekday", "gap", "prev_file", "era", "padded"]],
     "02d_节后首日清单与padding情况.csv")
print("\n【新旧口径对比】节后首日中被 padding 的比例：")
print(after_break.groupby("era")["padded"].agg(节后首日数="size", 被padding="sum").assign(
    比例=lambda d: (d["被padding"] / d["节后首日数"]).round(4)).to_string())
print("旧格式期被 padding 的那几天：",
      ", ".join(after_break.loc[after_break["padded"] & (after_break["era"] == "旧格式"), "clearing_day"]))
print("新格式期未被 padding 的节后首日：",
      ", ".join(after_break.loc[~after_break["padded"] & (after_break["era"] == "新格式"), "clearing_day"]) or "（无）")

# 形似但性质完全不同：普通交易日里个别品种整夜零成交 —— 那是真的没人交易（僵尸品种），
# 不是伪造。单列出来，避免跟 padding 混为一谈。
zombie = A[A["is_night_block"] & (A["bars"] > 0) & (A["sum_vol"] == 0)
           & ~A["clearing_day"].isin(list(padded_days))]
dump(zombie[["clearing_day", "tdate", "market", "market_cn", "product", "n_contracts",
             "bars", "min_bt", "max_bt"]].sort_values(["clearing_day", "market", "product"]),
     "02c_普通交易日整夜零成交_非伪造.csv")
print(f"\n【对照，不算问题】普通交易日里整夜零成交的 (品种×交易日)：{len(zombie):,} 组 / "
      f"{zombie['bars'].sum():,} 根，集中在：")
print(zombie.groupby(["market_cn", "product"]).size().sort_values(ascending=False).head(8).to_string())


# ================================================================ 06 元旦被伪造成整日
head("06 假期当天被伪造成完整交易日")
# 该块跨过了日盘时段（既有 09:30 之前的开头，又延伸到 15:00 之后）-> 是被伪造出来的
# 一整个交易日，而不是普通的夜盘 padding（后者恒为 21:00~23:59 或 00:00~02:30）
fake_day = pad[(pad["min_bt"] <= "09:30") & (pad["max_bt"] >= "15:00")].copy()
fake_out = (fake_day.groupby(["clearing_day", "tdate"])
            .agg(bars=("bars", "sum"), n_prod=("product", "nunique"),
                 n_mkt=("market", "nunique"), min_bt=("min_bt", "min"), max_bt=("max_bt", "max"))
            .reset_index())
dump(fake_day[["clearing_day", "tdate", "market", "market_cn", "product", "n_contracts",
               "bars", "min_bt", "max_bt"]].sort_values(["clearing_day", "market", "product"]),
     "06_假期被伪造成整日.csv")
print(fake_out.to_string(index=False))


# ================================================================ 03 竞价被错标到夜盘末根
head("03 节后首日的夜盘末根带成交量 = 被错标的日盘集合竞价")
padded_days = set(pad_day["clearing_day"])
c = C[C["clearing_day"].isin(list(padded_days))].copy()
c = c[(c["day0_open"] > 0) & (c["pre_close"] > 0)]
c["dev_to_day_open"] = (c["pre_close"] - c["day0_open"]).abs() / c["day0_open"]
c["dev_to_padding"] = np.where(c["pad_close"] > 0,
                               (c["pre_close"] - c["pad_close"]).abs() / c["pad_close"], np.nan)
c["closer_to"] = np.where(c["dev_to_day_open"] < c["dev_to_padding"], "当日日盘开盘价", "padding价(前收盘)")
c["market_cn"] = c["market"].map(MKT_CN)
c_out = c[["clearing_day", "market", "market_cn", "product", "contract", "pre_bt", "pre_vol",
           "pre_close", "pad_bt", "pad_close", "day0_bt", "day0_open", "day0_vol",
           "dev_to_day_open", "dev_to_padding", "closer_to"]].sort_values(["clearing_day", "market", "contract"])
dump(c_out, "03_竞价被错标到夜盘末根.csv")

both = c.dropna(subset=["dev_to_padding"])
print(f"样本（padding 交易日 × 盘前仅 1~3 根带量 bar 的合约）：{len(c)}，其中可与 padding 价对比的 {len(both)}")
print(f"更接近【当日日盘开盘价】的比例：{(both['closer_to'] == '当日日盘开盘价').mean():.3f}"
      f"  ({(both['closer_to'] == '当日日盘开盘价').sum()}/{len(both)})")
print(f"距日盘开盘价的相对偏离中位数：{both['dev_to_day_open'].median():.5f}")
print(f"距 padding 价（前收盘）的相对偏离中位数：{both['dev_to_padding'].median():.5f}")
print(f"偏离 < 0.1% 的占比：日盘开盘价 {(both['dev_to_day_open'] < 1e-3).mean():.3f}，"
      f"padding 价 {(both['dev_to_padding'] < 1e-3).mean():.3f}")
print("\n涉及的品种与该根的时间：")
print(c.groupby(["market_cn", "product", "pre_bt"]).size().rename("样本数").to_string())

# 周一特例：padding 日里完全没有盘前带量 bar -> 竞价数据被直接丢弃
lost = sorted(padded_days - set(c["clearing_day"]))
lost_df = pd.DataFrame({"clearing_day": lost})
if len(lost_df):
    lost_df["weekday"] = pd.to_datetime(lost_df["clearing_day"], format="%Y%m%d").dt.weekday.map(lambda i: WD_CN[i])
    lost_df["padding_tdates"] = lost_df["clearing_day"].map(pad_day.set_index("clearing_day")["tdates"])
dump(lost_df, "03b_节后首日竞价直接丢失.csv")
print(f"\npadding 交易日中盘前无任何带量 bar（竞价数据直接丢失）：{len(lost)} 个")
if len(lost_df):
    print(lost_df["weekday"].value_counts().to_string())
    print(lost_df.to_string(index=False))


# ================================================================ 01 缺 09:00 日盘集合竞价
head("01 有夜盘的品种缺 09:00 日盘集合竞价")
pp = (A.groupby(["clearing_day", "market", "product"])
      .agg(n0900=("n0900", "sum"), n0900v=("n0900v", "sum"), n0900flat=("n0900flat", "sum"),
           n2100=("n2100", "sum"), n2100v=("n2100v", "sum"), n2100flat=("n2100flat", "sum"),
           n0930=("n0930", "sum"), n0930v=("n0930v", "sum"), n0930flat=("n0930flat", "sum"),
           n_contracts=("n_contracts", "max"), sum_vol=("sum_vol", "sum"))
      .reset_index())
pp["market_cn"] = pp["market"].map(MKT_CN)
pp["has_night"] = pp["n2100"] > 0
pp["has_0900"] = pp["n0900"] > 0

# 互斥性：有夜盘的品种是不是一根 09:00 都没有
recent = pp[pp["clearing_day"] >= AUCTION_RULE_DAY]
x = pd.crosstab(recent["has_night"], recent["has_0900"])
print("2023-05-26 起，(品种×交易日) 的 has_2100 × has_0900 交叉表：")
print(x.to_string())

miss0900 = recent[recent["has_night"] & recent["market"].isin(DAY_AUCTION_MARKETS) &
                  ~recent["clearing_day"].isin(list(padded_days))].copy()
dump(miss0900[["clearing_day", "market", "market_cn", "product", "n_contracts",
               "n2100", "n2100v", "n0900"]].sort_values(["clearing_day", "market", "product"]),
     "01_缺09点日盘集合竞价.csv")
print(f"\n受规则约束（上期所/能源中心/大商所 有夜盘品种，2023-05-26 起，剔除 padding 日）：")
print(f"  组合数 {len(miss0900):,}（品种×交易日），其中有 09:00 K 线的：{(miss0900['n0900'] > 0).sum()}")
print(f"  涉及交易日 {miss0900['clearing_day'].nunique()} 个，品种 {miss0900['product'].nunique()} 个")
print("\n按交易所×品种：")
sm = (miss0900.groupby(["market_cn", "product"])
      .agg(交易日数=("clearing_day", "nunique"), 合约数中位=("n_contracts", "median"),
           有0900的天数=("n0900", lambda s: (s > 0).sum()))
      .reset_index())
print(sm.to_string(index=False))
dump(sm, "01b_缺09点日盘集合竞价_按品种汇总.csv")

# 对照：旧口径下，夜盘取消的交易日里 09:00 是存在的
print("\n对照 — 旧格式期节后首日（夜盘取消）的 09:00 集合竞价存在情况：")
for d in ["20160215", "20180222", "20191008", "20201009", "20210218", "20220104", "20220207", "20220505"]:
    if d not in have:
        continue
    sub = pp[(pp["clearing_day"] == d) & (pp["market"].isin(DAY_AUCTION_MARKETS))]
    nn = sub[sub["n0900"] > 0]
    print(f"  {d}: 有 09:00 的品种 {len(nn)}/{len(sub)}，"
          f"夜盘块 bar 数 {int(night.loc[d, 'night_bars'])}")


# ================================================================ 04 郑商所竞价 bar 不是平 bar
head("04 郑商所集合竞价 K 线不是平 K 线")
ny = (A.groupby(["year", "market"])
      .agg(n2100v=("n2100v", "sum"), n2100flat=("n2100flat", "sum"),
           n0900v=("n0900v", "sum"), n0900flat=("n0900flat", "sum"),
           n0930v=("n0930v", "sum"), n0930flat=("n0930flat", "sum"))
      .reset_index())
ny["flat_2100"] = ny["n2100flat"] / ny["n2100v"].replace(0, np.nan)
ny["flat_0900"] = ny["n0900flat"] / ny["n0900v"].replace(0, np.nan)
ny["market_cn"] = ny["market"].map(MKT_CN)
dump(ny[["year", "market", "market_cn", "n2100v", "n2100flat", "flat_2100",
         "n0900v", "n0900flat", "flat_0900"]], "04_竞价K线平bar比例_按年按交易所.csv")
print("21:00 夜盘集合竞价平 K 线比例（分母 = 有成交的竞价 bar）：")
print(ny.pivot(index="year", columns="market_cn", values="flat_2100").round(3).to_string())
print("\n09:00 日盘集合竞价平 K 线比例：")
print(ny.pivot(index="year", columns="market_cn", values="flat_0900").round(3).to_string())

npd = (A[A["market"] == "XZCE"].groupby("product")
       .agg(n2100v=("n2100v", "sum"), n2100flat=("n2100flat", "sum"),
            n0900v=("n0900v", "sum"), n0900flat=("n0900flat", "sum"))
       .reset_index())
npd["flat_2100"] = npd["n2100flat"] / npd["n2100v"].replace(0, np.nan)
npd["flat_0900"] = npd["n0900flat"] / npd["n0900v"].replace(0, np.nan)
dump(npd.sort_values("n2100v", ascending=False), "04b_郑商所竞价平bar比例_按品种.csv")
print("\n郑商所逐品种（夜盘竞价样本 ≥ 1000 的）：")
print(npd[npd["n2100v"] >= 1000].sort_values("flat_2100").to_string(index=False))


# ================================================================ 07 郑商所成交额凑整万元
head("07 郑商所 value / vwap 被「凑整万元」污染")
B["year"] = B["clearing_day"].str[:4].astype(int)
B["market_cn"] = B["market"].map(MKT_CN)
by = (B.groupby(["year", "market_cn"])
      .agg(合约日数=("n_contracts", "sum"), 整万元的=("n_value_whole_10k", "sum"))
      .reset_index())
by["整万元比例"] = by["整万元的"] / by["合约日数"].replace(0, np.nan)
dump(by, "07_成交额整万元比例_按年按交易所.csv")
print("逐合约全日成交额恰好是整数万元的比例（已剔除全日无成交额的合约）：")
print(by.pivot(index="year", columns="market_cn", values="整万元比例").round(4).to_string())

neg = (A.groupby(["year", "market_cn"])
       .agg(负成交额bar=("n_neg_value", "sum"), 负成交量bar=("n_neg_vol", "sum"),
            负持仓量bar=("n_neg_oi", "sum"),
            vwap越界bar_1pct=("n_vwap_out_1pct", "sum"), vwap越界bar_01pct=("n_vwap_out", "sum"))
       .reset_index())
dump(neg, "07b_负值与vwap越界_按年按交易所.csv")
print("\n负成交额 bar（全历史，按交易所）：")
print(A.groupby("market_cn")["n_neg_value"].sum().to_string())
print("\nvwap 落在该根 [low, high] 之外且偏离 >1% 的 bar（按交易所）：")
print(A.groupby("market_cn")[["n_vwap_out_1pct"]].sum().to_string())
print("\nvwap 最大偏离（按交易所）：")
print(A.groupby("market_cn")["max_vwap_dev"].max().round(4).to_string())

vw = (A[A["n_vwap_out_1pct"] > 0]
      .groupby(["market_cn", "product"])
      .agg(bar数=("n_vwap_out_1pct", "sum"), 最大偏离=("max_vwap_dev", "max"),
           涉及交易日=("clearing_day", "nunique"))
      .reset_index().sort_values("bar数", ascending=False))
dump(vw, "07c_vwap越界_按品种.csv")
print("\nvwap 越界最严重的 15 个品种：")
print(vw.head(15).to_string(index=False))


# ================================================================ 09 RU 的 vwap 用错了合约乘数
head("09 上期所天然橡胶 RU 的 vwap 用错了合约乘数")
ru = A[(A["market"] == "XSGE") & (A["product"] == "RU")].groupby("clearing_day").agg(
    bad=("n_vwap_out_1pct", "sum"), bars=("bars", "sum"), maxdev=("max_vwap_dev", "max")).reset_index()
ru["ratio"] = ru["bad"] / ru["bars"]
dump(ru, "09_RU_vwap越界_逐日.csv")
bad_days = ru[ru["bad"] > 0]
print(f"RU 有 vwap 越界的交易日：{len(bad_days)} / {len(ru)}")
print(f"集中区间：{bad_days['clearing_day'].min()} ~ ", end="")
dense = bad_days[bad_days["ratio"] > 0.01]
print(f"{dense['clearing_day'].max()}（占比 >1% 的日子共 {len(dense)} 天，越界 bar {int(dense['bad'].sum()):,} 根）")
print(ru[ru["clearing_day"].str[:4].isin(["2010", "2011", "2012", "2013"])]
      .groupby(ru["clearing_day"].str[:4]).agg(交易日=("clearing_day", "size"),
                                               越界bar=("bad", "sum"), 总bar=("bars", "sum")).to_string())

# 其余交易所的 vwap 越界是什么量级 —— 用来说明 RU 和郑商所是两类不同的问题
other = A[(A["n_vwap_out_1pct"] > 0) & ~((A["market"] == "XSGE") & (A["product"] == "RU"))
          & (A["market"] != "XZCE")]
print(f"\n【对照】除 RU 和郑商所外，其余 vwap 越界 bar 共 {int(other['n_vwap_out_1pct'].sum()):,} 根，"
      f"最大偏离 {other['max_vwap_dev'].max():.4f}，集中在：")
print(other.groupby("clearing_day")["n_vwap_out_1pct"].sum().sort_values(ascending=False).head(6).to_string())

# 旧格式期是否真的没有文件缺失：只看能直接判定的那些天
pre = miss[(miss["missing_day"] < SCHEMA_SWITCH_DAY) & miss["direct_verdict"].str.startswith("已证实")]
print(f"\n【对照】2022-09-22 之前可直接判定的缺失工作日 {len(pre)} 天，"
      f"其中判为交易日（=文件缺失）的：{(pre['direct_verdict'].str.startswith('已证实是交易日')).sum()} 天")


# ================================================================ 08 九品种在 raw 中的覆盖
head("08 AO/BR/EC/IM/LC/PX/SH/SI/TL 在 raw 中的覆盖")
nine = (A[A["product"].isin(TRUNCATED_9)]
        .groupby(["clearing_day", "market", "product"])
        .agg(n_contracts=("n_contracts", "max"), sum_vol=("sum_vol", "sum"))
        .reset_index())
nine["market_cn"] = nine["market"].map(MKT_CN)
dump(nine.sort_values(["product", "clearing_day"]), "08_九品种raw覆盖.csv")
cov = (nine.groupby(["market_cn", "product"])
       .agg(首个交易日=("clearing_day", "min"), 末个交易日=("clearing_day", "max"),
            交易日数=("clearing_day", "nunique"), 合约数中位=("n_contracts", "median"))
       .reset_index())
# 上市后应有的交易日数 = 区间内文件数，用来确认中间没有断档
cov["区间内文件数"] = cov.apply(
    lambda r: int(((days >= r["首个交易日"]) & (days <= r["末个交易日"])).sum()), axis=1)
cov["缺口"] = cov["区间内文件数"] - cov["交易日数"]
dump(cov, "08b_九品种raw覆盖汇总.csv")
print(cov.to_string(index=False))
print("\n2024-04-22 前后各 5 个交易日的合约数（确认无断点）：")
around = days[(days >= "20240415") & (days <= "20240429")]
print(nine[nine["clearing_day"].isin(around)]
      .pivot_table(index="clearing_day", columns="product", values="n_contracts").to_string())

print("\n完成。明细清单目录：", OUT)
