# -*- coding: utf-8 -*-
"""
v3.0 一分钟数据的一次性 ETL: 6GB 单体 pkl -> 59 个干净的分品种 parquet。

为什么要落地成分品种文件: 画图时一次只看一个品种, 但源文件是 6600 万行的
单体 pkl, 每次加载 8 秒 + 6GB 内存。拆开之后单品种加载在 0.1 秒量级, 图表
才能做到切换 ticker 即时响应。

清洗规则全部来自 data/v3.0/data_quality_checklist.md 的排查结论:
  1. 品种池 = 59 (见 ticker_exclusion_log.md Step 1-4)
  2. usable_from 裁掉 B/SF/SM/CY 的早期无成交段 (Step 5)
  3. 剔除伪造 bar: 任一 (品种,交易日,日历日,日/夜段) 的成交量恒为 0 -> 整段丢弃
     这一条同时命中"节前被取消的夜盘"和"元旦被伪造成完整交易日"两种伪造
  4. 剔除硬错误: OHLC<=0 或 NaN、vol/turnover/OI 负值、量额矛盾
  5. 保留 K 原样 (后复权因子)。复权价 = close/K, 在展示层再乘窗口首根的 K。
     不在这里预先复权, 是为了让存储层保持后复权语义: 补新数据只追加, 永不重算。

用法: python -m src.data.prepare_v3_minute
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

SRC = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
DST = r"D:\2026_Summer\TradingApp\data\v3.0\clean_1m"

POOL = [
    "A", "AG", "AL", "AP", "AU", "B", "BC", "BU", "C", "CF", "CJ", "CS", "CU", "CY",
    "EB", "EG", "FG", "FU", "HC", "I", "IC", "IF", "IH", "J", "JD", "JM", "L", "LH",
    "LU", "M", "MA", "NI", "NR", "OI", "P", "PB", "PF", "PG", "PK", "PP", "RB", "RM",
    "RU", "SA", "SC", "SF", "SM", "SN", "SP", "SR", "SS", "T", "TA", "TF", "TS", "UR",
    "V", "Y", "ZN",
]
# 早期整日零成交的死区 (data_quality_checklist.md F4)
USABLE_FROM = {
    "B": "2018-01-01", "SF": "2018-01-01", "SM": "2017-01-01", "CY": "2019-01-01",
}

_t0 = time.time()


def log(*a):
    print(f"[{time.time() - _t0:6.1f}s]", *a, flush=True)


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    log("loading source pkl ...")
    df = pd.read_pickle(SRC)
    n0 = len(df)
    log(f"loaded {n0:,} rows")

    # ---- 1. 品种池 ------------------------------------------------------
    df = df[df["underlying_symbol"].isin(POOL)]
    log(f"pool filter  -> {len(df):,} rows ({len(POOL)} symbols)")

    sym = df["underlying_symbol"].astype("category")
    sc = sym.cat.codes.to_numpy()
    SYMS = np.asarray(sym.cat.categories)
    vol = df["volume"].to_numpy()
    tov = df["total_turnover"].to_numpy()
    oi = df["open_interest"].to_numpy()
    op = df["open"].to_numpy(); hi = df["high"].to_numpy()
    lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    idx = df.index
    mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)

    # ---- 2. 伪造 bar: 按 (品种,交易日,日历日,日/夜段) 判成交量是否恒为 0 ----
    # 夜盘定义 21:00-02:59; 其余算日盘。日历日必须进 key, 否则元旦那种
    # "伪造日盘 + 真实日盘同属一个 trading_date" 的情形会被真实成交掩盖。
    # 键全部用整数拼: 5700 万行上做 strftime + 字符串拼接会慢到分钟级并吃掉
    # 十几 GB 内存, 整数算术是毫秒级。
    is_night = (mod >= 21 * 60) | (mod <= 179)
    day_unit = np.timedelta64(1, "D")
    td_i = (df["trading_date"].to_numpy().astype("datetime64[D]")
            - np.datetime64("2000-01-01")) // day_unit
    cal_i = (idx.to_numpy().astype("datetime64[D]")
             - np.datetime64("2000-01-01")) // day_unit
    raw = ((sc.astype(np.int64) * 20000 + td_i) * 20000 + cal_i) * 2 + is_night
    seg_key = pd.factorize(raw)[0]
    seg_vol = np.bincount(seg_key, weights=vol)
    fake = seg_vol[seg_key] == 0
    log(f"fabricated segments -> drop {int(fake.sum()):,} bars "
        f"({fake.mean():.4%}); {int(pd.Series(seg_key[fake]).nunique()):,} segments")

    # ---- 3. 硬错误 ------------------------------------------------------
    bad_px = (np.isnan(op) | np.isnan(hi) | np.isnan(lo) | np.isnan(cl)
              | (op <= 0) | (hi <= 0) | (lo <= 0) | (cl <= 0))
    bad_neg = (vol < 0) | (tov < 0) | (oi < 0)
    bad_mix = ((vol == 0) & (tov > 0)) | ((vol > 0) & (tov == 0))
    hard = bad_px | bad_neg | bad_mix
    log(f"hard errors -> drop {int(hard.sum()):,} bars "
        f"(zero-px {int(bad_px.sum()):,}, neg {int(bad_neg.sum()):,}, "
        f"vol/turnover mismatch {int(bad_mix.sum()):,})")

    keep = ~(fake | hard)
    df = df[keep]
    sc = sc[keep]
    log(f"after cleaning -> {len(df):,} rows (dropped {n0 - len(df):,} total)")

    # ---- 4. 逐品种落地 --------------------------------------------------
    cols = ["open", "high", "low", "close", "volume", "total_turnover",
            "open_interest", "contract", "K", "trading_date"]
    rows = []
    for code in range(len(SYMS)):
        s = str(SYMS[code])
        sub = df[sc == code][cols].copy()
        n_before = len(sub)
        cut = USABLE_FROM.get(s)
        if cut:
            sub = sub[sub["trading_date"] >= pd.Timestamp(cut)]
        sub["contract"] = sub["contract"].astype("category")
        sub.index.name = "datetime"
        path = os.path.join(DST, f"{s}.parquet")
        sub.to_parquet(path, compression="zstd", index=True)
        rows.append({
            "sym": s, "bars": len(sub), "trimmed_by_usable_from": n_before - len(sub),
            "first": sub.index.min(), "last": sub.index.max(),
            "sessions": sub["trading_date"].nunique(),
            "mb": os.path.getsize(path) / 1e6,
        })
        if len(rows) % 10 == 0:
            log(f"  {len(rows)}/{len(SYMS)} written")

    man = pd.DataFrame(rows).set_index("sym")
    man.to_csv(os.path.join(DST, "_manifest.csv"))
    log(f"done. {len(man)} files, {man.bars.sum():,} bars, {man.mb.sum():.0f} MB")
    print(man.to_string())


if __name__ == "__main__":
    main()
