"""raw（通联日频文件）与二级数据（主力连续 pkl）的逐合约对账。

回答三个问题：
  1. 同一个 (合约, 交易日) 在两层的 bar 数是否一致 —— 不一致就是某一层多生成或少生成了 K 线。
  2. 有没有【只在二级数据里出现的日历日】—— 那是 raw 里根本不存在、由二级加工环节
     凭空生成的 bar。判据很干净：raw 的 TDATE 就是 bar 的日历日，取差集即可。
  3. 二级数据是否丢了 raw 有的交易日（FB / LR 那类）。

6GB pkl 只加载一次，聚合结果落盘缓存；重跑时若缓存在就直接读缓存。

用法：python data/00-跨层/compare_layers.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKL = ROOT / "data" / "01-pkl层" / "all_symbol_min_full_main_close_k_1.pkl"
TD = ROOT / "data" / "00-跨层" / "对账证据"          # 对账产物（本脚本自己的结论）
RAW = ROOT / "data" / "02-txt层" / "排查证据"        # raw 侧扫描证据（既是输入，也是 §4 产物的归属）
TD.mkdir(parents=True, exist_ok=True)

AGG_CONTRACT = TD / "_pkl_agg_by_contract.csv"
AGG_CALENDAR = TD / "_pkl_agg_by_calendar.csv"

_t0 = time.time()


def log(*a):
    print(f"[{time.time() - _t0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- 二级数据侧聚合（带缓存）
def build_pkl_aggregates() -> None:
    if AGG_CONTRACT.exists() and AGG_CALENDAR.exists():
        log("二级数据聚合缓存已存在，跳过 6GB 加载")
        return

    log("加载 6GB pkl ...")
    df = pd.read_pickle(PKL)
    log(f"已加载 {len(df):,} 行，列 = {list(df.columns)}")

    # 只留对账需要的列，尽早把内存压下来。
    # 日期一律用 yyyymmdd 整数而不是 strftime —— 后者在 6,600 万行上是 Python 级循环，
    # 实测要跑十几分钟且额外吃掉几个 GB 的字符串；年月日相乘是向量化的，秒级完成。
    idx = pd.DatetimeIndex(df.index)
    td = pd.DatetimeIndex(pd.to_datetime(df["trading_date"].values))
    out = pd.DataFrame({
        "calendar_day": (idx.year * 10000 + idx.month * 100 + idx.day).astype("int32"),
        "trading_day": (td.year * 10000 + td.month * 100 + td.day).astype("int32"),
        # 合约代码转 category：6,600 万行的 object 列做 groupby 会慢一个数量级
        "contract": pd.Categorical(df["contract"].values),
        "product": pd.Categorical(df["underlying_symbol"].values),
        "volume": df["volume"].astype("float64").values,
    })
    del df, idx, td
    log(f"已抽出对账列，{out['contract'].nunique():,} 个合约")

    (out.groupby(["contract", "trading_day"], observed=True)
        .agg(pkl_bars=("volume", "size"), pkl_vol=("volume", "sum"))
        .reset_index()
        .to_csv(AGG_CONTRACT, index=False))
    log(f"-> {AGG_CONTRACT.name}")

    (out.groupby("calendar_day", observed=True)
        .agg(pkl_bars=("volume", "size"), pkl_vol=("volume", "sum"),
             n_product=("product", "nunique"))
        .reset_index()
        .to_csv(AGG_CALENDAR, index=False))
    log(f"-> {AGG_CALENDAR.name}")


build_pkl_aggregates()

pk = pd.read_csv(AGG_CONTRACT, dtype={"contract": str, "trading_day": str})
pcal = pd.read_csv(AGG_CALENDAR, dtype={"calendar_day": str})
log(f"二级数据：{len(pk):,} 个 (合约×交易日)，{len(pcal):,} 个日历日")

rw = pd.read_csv(RAW / "raw_scan_by_contract.csv",
                 dtype={"clearing_day": str, "contract": str, "min_bt": str, "max_bt": str})
log(f"raw：{len(rw):,} 个 (合约×结算日)")


# ================================================================ 问题 1 逐合约 bar 数比对
head = lambda s: print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")

head("1  逐 (合约, 交易日) 的 bar 数比对")
m = rw.merge(pk, left_on=["contract", "clearing_day"], right_on=["contract", "trading_day"],
             how="outer", indicator=True)
both = m[m["_merge"] == "both"].copy()
both["d_bars"] = both["pkl_bars"] - both["bars"]
diff = both[both["d_bars"] != 0]
print(f"两层都有的 (合约×交易日)：{len(both):,}，其中 bar 数不一致：{len(diff):,} "
      f"（{len(diff) / max(len(both), 1):.4%}）")
if len(diff):
    print("\n按交易日汇总（前 15）：")
    byday = (diff.groupby("clearing_day")
             .agg(合约数=("contract", "nunique"), 二级多出的bar=("d_bars", "sum"),
                  品种数=("product", "nunique"))
             .sort_values("二级多出的bar", ascending=False))
    print(byday.head(15).to_string())
    diff.sort_values(["clearing_day", "contract"]).to_csv(
        TD / "layer_diff_by_contract.csv", index=False, encoding="utf-8-sig")
    log(f"-> layer_diff_by_contract.csv（{len(diff):,} 行）")

only_raw = m[m["_merge"] == "left_only"]
only_pkl = m[m["_merge"] == "right_only"]
print(f"\n只在 raw 出现的 (合约×交易日)：{len(only_raw):,}"
      f"（正常 —— 二级只保留主力合约）")
print(f"只在二级出现的 (合约×交易日)：{len(only_pkl):,}")
if len(only_pkl):
    print(only_pkl.groupby(only_pkl["trading_day"].str[:4]).size().rename("天数").to_string())


# ================================================================ 问题 2 只在二级出现的日历日
head("2  只在二级数据里出现的日历日（= raw 里根本没有的 bar）")
# raw 里 bar 的日历日就是 TDATE：夜盘块 TDATE=当晚日历日，跨午夜段与日盘 TDATE=当日
A = pd.read_csv(RAW / "raw_scan_daily.csv",
                dtype={"clearing_day": str, "tdate": str})
raw_cal = set(A["tdate"].unique())
print(f"raw 覆盖的日历日：{len(raw_cal):,}   二级覆盖的日历日：{len(pcal):,}")

ghost = pcal[~pcal["calendar_day"].isin(raw_cal)].copy()
# 只看二级数据覆盖期内的，避免把 raw 尚未交付的尾部算进来
ghost = ghost[ghost["calendar_day"] <= A["tdate"].max()]
ghost = ghost.sort_values("calendar_day")
ghost.to_csv(TD / "layer_diff_calendar_days.csv", index=False, encoding="utf-8-sig")
print(f"\n只在二级数据出现的日历日：{len(ghost)} 个")
if len(ghost):
    print(ghost.to_string(index=False))
    print(f"\n这些日历日的成交量合计 = {ghost['pkl_vol'].sum():,.0f}"
          f"  （若为 0 -> 全部是伪造；若非 0 -> 是真实数据，结论要推翻）")
log("-> layer_diff_calendar_days.csv")


# ================================================================ 问题 3 二级数据丢了 raw 有的交易日
head("3  二级数据缺、raw 有的交易日（按品种，限品种在二级数据里的存活区间内）")
rw_prod = rw.groupby(["product", "clearing_day"]).size().reset_index(name="n")
pk_prod = pk.merge(rw[["contract", "clearing_day", "product"]].drop_duplicates(),
                   left_on=["contract", "trading_day"], right_on=["contract", "clearing_day"],
                   how="left")
rows = []
for prod, g in pk_prod.groupby("product"):
    if pd.isna(prod):
        continue
    lo, hi = g["trading_day"].min(), g["trading_day"].max()
    pdays = set(g["trading_day"])
    rdays = set(rw_prod.loc[(rw_prod["product"] == prod) &
                            (rw_prod["clearing_day"] >= lo) &
                            (rw_prod["clearing_day"] <= hi), "clearing_day"])
    miss = sorted(rdays - pdays)
    if miss:
        rows.append(dict(product=prod, 存活区间=f"{lo}~{hi}", raw天数=len(rdays),
                         二级天数=len(pdays), 二级缺失=len(miss),
                         首个缺失=miss[0], 末个缺失=miss[-1],
                         样例="  ".join(miss[:6])))
lost = pd.DataFrame(rows).sort_values("二级缺失", ascending=False)
lost.to_csv(TD / "layer_missing_days_by_product.csv", index=False, encoding="utf-8-sig")
print(f"有缺失的品种：{len(lost)}")
print(lost.head(20).to_string(index=False))
log("-> layer_missing_days_by_product.csv")


# ================================================================ 4 raw 侧硬错误（v3.1 未查）
head("4  raw 侧硬错误：OHLC <= 0 与量额矛盾（v3.1 漏查，v3.0 只在二级层查过）")
print(f"OHLC <= 0 的 bar：{int(rw['n_ohlc_nonpos'].sum()):,}")
np_ = rw[rw["n_ohlc_nonpos"] > 0]
print(np_.groupby(["market", "product"])["n_ohlc_nonpos"].sum().sort_values(ascending=False).head(10).to_string())
print(f"\n涉及交易日 {np_['clearing_day'].nunique()} 个，"
      f"区间 {np_['clearing_day'].min()} ~ {np_['clearing_day'].max()}")
print("\n按年：")
print(np_.groupby(np_["clearing_day"].str[:4])["n_ohlc_nonpos"].sum().to_string())

print(f"\n量额矛盾 volume==0 但 value!=0：{int(rw['n_vol0_val'].sum()):,}")
print(rw.groupby("market")["n_vol0_val"].sum().sort_values(ascending=False).to_string())
print(f"\n量额矛盾 volume!=0 但 value==0：{int(rw['n_val0_vol'].sum()):,}")
print(rw.groupby("market")["n_val0_vol"].sum().sort_values(ascending=False).to_string())
print("\nvolume!=0 但 value==0 的按品种（前 10）：")
print(rw[rw["n_val0_vol"] > 0].groupby(["market", "product"])["n_val0_vol"].sum()
      .sort_values(ascending=False).head(10).to_string())

hard = (rw[(rw["n_ohlc_nonpos"] > 0) | (rw["n_vol0_val"] > 0) | (rw["n_val0_vol"] > 0)]
        [["clearing_day", "market", "product", "contract", "bars",
          "n_ohlc_nonpos", "n_vol0_val", "n_val0_vol"]]
        .sort_values(["clearing_day", "market", "contract"]))
hard.to_csv(RAW / "raw_hard_errors.csv", index=False, encoding="utf-8-sig")
log(f"-> raw_hard_errors.csv（{len(hard):,} 行）")

log("完成")
