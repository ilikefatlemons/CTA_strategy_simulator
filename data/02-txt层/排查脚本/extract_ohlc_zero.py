"""导出 raw 里全部「OHLC 出现 0」的问题行清单（P1-9）。

扫描 data/02-txt层/原始数据 下每一个 .txt 文件，挑出 closeprice / openprice / highprice / lowprice
四列中任意一列等于 0 的行，汇总成一份 CSV。

输出列（顺序固定）：
    TDATE, bartime, MARKET, CONTRACTID, ZERO_OHLC

ZERO_OHLC 标出这一行是哪几个价格为 0，字母恒按 O -> H -> L -> C 的顺序拼接：
    "OHLC"  四价全为 0
    "OL"    只有 openprice 与 lowprice 为 0
    "C"     只有 closeprice 为 0
    以此类推

行按 TDATE -> bartime -> MARKET -> CONTRACTID 排序。

用法：python data/02-txt层/排查脚本/extract_ohlc_zero.py
输出：data/02-txt层/排查证据/ohlc_zero_rows.csv

关于两种表头
------------
raw 全历史只有两套表头，以 2022-09-22 为界：
    旧（<= 20220921）TAB 分隔、14 列，第 5 列是 Offset
    新（>= 20220922）逗号分隔、13 列，无 Offset
两者的**列名逐字相同**，差异只有多出来的 Offset 一列。所以这里用 usecols 按【列名】
取列，Offset 造成的列位移由 pandas 自动吸收 —— 不需要写两套列索引，也不需要偏移量。
唯一需要判别的是分隔符。
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "02-txt层" / "原始数据"
OUT_CSV = Path(__file__).resolve().parent / "ohlc_zero_rows.csv"

# 标识列：既是输出的前四列，也是排序键
KEY_COLS = ["TDATE", "bartime", "MARKET", "CONTRACTID"]
# 价格列与其在 ZERO_OHLC 里的字母，顺序即拼接顺序 O -> H -> L -> C
PRICE_MARKS = [("openprice", "O"), ("highprice", "H"),
               ("lowprice", "L"), ("closeprice", "C")]
PRICE_COLS = [c for c, _ in PRICE_MARKS]
FLAG_COL = "ZERO_OHLC"
OUT_COLS = KEY_COLS + [FLAG_COL]
NEED = KEY_COLS + PRICE_COLS

# 所有列一律按字符串读入，价格列读进来之后再 to_numeric(errors="coerce")。
# 两条都必须这么做的理由：
#   1. 价格列若交给 pandas 自行推断：raw 同一列里既有 "0" 又有 "4485.0"，C 解析器会在不同
#      chunk 上分别推断出 int64 与 float64，触发 mixed-dtype 告警分支；而那个分支与 usecols
#      同时使用时，会拿【原始列号】去索引【过滤后的列名表】，直接 IndexError 崩掉。
#   2. 价格列若钉死 float64：raw 里确实存在字段错位的坏行（已知 future_pricemin20220901.txt
#      第 213566 行，合约 M2301、bartime 09:38，字段数 20、行长 39 万字符），会把合约代码
#      之类的字符串落进价格列，导致【整个文件】读不进来。
# 读成字符串再 coerce，坏值变 NaN（NaN != 0，不会被误判成零价），文件其余几十万行照常处理。
# low_memory=False 再兜一层（整文件一次读入，根本不走 chunk 拼接）。
DTYPES = {c: "str" for c in NEED}

# 设成 1 可退回单进程，便于排查
WORKERS = min(12, max(1, cpu_count() - 2))


def scan_one(path_str: str) -> tuple[pd.DataFrame, str | None]:
    """扫一个 raw 文件，返回 (命中的行, 结构异常说明)。单个文件坏掉不拖垮整轮扫描。"""
    path = Path(path_str)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        sep = "\t" if "\t" in header else ","

        # on_bad_lines="skip"：字段数与表头对不上的行直接跳过，下面用行数对账把跳过量报出来
        df = pd.read_csv(path, sep=sep, usecols=NEED, dtype=DTYPES,
                         low_memory=False, on_bad_lines="skip")
        n_raw = sum(1 for _ in open(path, "rb")) - 1        # 减掉表头
        for col in PRICE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(sum(df[c].isna().sum() for c in PRICE_COLS))
        note = None
        if n_raw != len(df) or n_bad:
            note = (f"{path.name}: 文件 {n_raw:,} 行 / 解析 {len(df):,} 行"
                    f"（跳过 {n_raw - len(df):,} 行），价格字段非数字 {n_bad:,} 个")
    except Exception as exc:
        return pd.DataFrame(columns=OUT_COLS), f"{path.name}: 读取失败 {exc!r}"

    # 按需求判 == 0。正常数据里四个价格列没有空值、没有非数字，且全历史不存在负价，
    # 所以 == 0 与 <= 0 在这里等价；被 coerce 成 NaN 的坏值不会命中。
    is_zero = {col: df[col] == 0 for col in PRICE_COLS}
    mask = is_zero[PRICE_COLS[0]]
    for col in PRICE_COLS[1:]:
        mask = mask | is_zero[col]
    if not mask.any():
        return pd.DataFrame(columns=OUT_COLS), note

    out = df.loc[mask, KEY_COLS].copy()
    # 恒按 O -> H -> L -> C 拼字母；.map 出来是 object 列，+ 即字符串拼接
    flag = None
    for col, mark in PRICE_MARKS:
        part = is_zero[col][mask].map({True: mark, False: ""})
        flag = part if flag is None else flag + part
    out[FLAG_COL] = flag
    return out, note


def main() -> int:
    files = sorted(RAW.rglob("*.txt"))
    if not files:
        print(f"没有在 {RAW} 下找到任何 .txt 文件", file=sys.stderr)
        return 1

    top = sum(1 for f in files if f.parent == RAW)
    print(f"待扫描 {len(files):,} 个 .txt 文件"
          f"（{RAW} 顶层 {top:,} 个，子目录 {len(files) - top:,} 个），"
          f"并行度 {WORKERS}", flush=True)

    t0 = time.time()
    parts: list[pd.DataFrame] = []
    notes: list[str] = []
    with Pool(WORKERS) as pool:
        for i, (part, note) in enumerate(
                pool.imap(scan_one, [str(f) for f in files], chunksize=8), 1):
            if note:
                notes.append(note)
            if len(part):
                parts.append(part)
            if i % 200 == 0 or i == len(files):
                hit = sum(len(p) for p in parts)
                print(f"  [{time.time() - t0:6.1f}s] {i:>5,}/{len(files):,} 个文件，"
                      f"已命中 {hit:,} 行", flush=True)

    out = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=OUT_COLS))
    out = out.sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("")
    print(f"完成，用时 {time.time() - t0:.1f}s")
    print(f"-> {OUT_CSV}   共 {len(out):,} 行")
    if len(out):
        print(f"TDATE 范围：{out['TDATE'].min()} ~ {out['TDATE'].max()}")
        print("")
        print("按 ZERO_OHLC 取值：")
        print(out[FLAG_COL].value_counts().rename("行数").to_string())
        print("")
        print("按交易所：")
        print(out["MARKET"].value_counts().rename("行数").to_string())
        print("")
        print(f"涉及合约 {out['CONTRACTID'].nunique():,} 个，"
              f"涉及 TDATE {out['TDATE'].nunique():,} 天")
    if notes:
        print("")
        print(f"[!] {len(notes)} 个文件存在结构异常（已尽量解析，异常行不计入结果）：")
        for n in notes[:20]:
            print(f"    {n}")
    return 0


if __name__ == "__main__":
    # Windows 用 spawn 启动子进程，子进程会重新 import 本模块，必须有这层保护
    sys.exit(main())
