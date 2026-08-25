"""从 data/02-txt层/原始数据 直接抓原始行，生成报告里引用的 K 线证据块。

raw 只有两套表头（2022-09-22 分界），`read_raw` 把两套都归一成同一组列名，
下游不必再关心分隔符和 Offset 列。

用法：python data/02-txt层/排查脚本/extract_evidence.py > ../test_data/evidence.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "02-txt层" / "原始数据"

COLS_NEW = [
    "CLEARINGDAY", "TDATE", "CONTRACTID", "MARKET", "bartime",
    "close", "open", "high", "low", "volume", "value", "vwap", "OPENINTS",
]
COLS_OLD = COLS_NEW[:4] + ["Offset"] + COLS_NEW[4:]


def read_raw(day: str) -> pd.DataFrame:
    """读一个交易日的原始文件，返回归一化后的 DataFrame。"""
    path = RAW / f"future_pricemin{day}.txt"
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    old = "Offset" in header
    df = pd.read_csv(
        path,
        sep="\t" if old else ",",
        header=0,
        names=COLS_OLD if old else COLS_NEW,
        dtype={"CLEARINGDAY": str, "TDATE": str, "CONTRACTID": str,
               "MARKET": str, "bartime": str},
    )
    if old:
        df = df.drop(columns=["Offset"])
    return df


def rows(day: str, contract: str, t0: str = "00:00", t1: str = "23:59") -> pd.DataFrame:
    df = read_raw(day)
    m = (df["CONTRACTID"] == contract) & (df["bartime"] >= t0) & (df["bartime"] <= t1)
    return df.loc[m].sort_values(["TDATE", "bartime"])


def show(title: str, df: pd.DataFrame, note: str = "") -> None:
    print(f"### {title}")
    if note:
        print(f"# {note}")
    if df.empty:
        print("(无数据)")
        print()
        return
    for _, r in df.iterrows():
        # .10g 而不是 .4g —— 后者会把 98380 印成 9.838e+04，价格必须原样可读
        print(
            f"{r.CONTRACTID:<8} {r.TDATE} {r.bartime}  O={r.open:<10.10g} H={r.high:<10.10g} "
            f"L={r.low:<10.10g} C={r.close:<10.10g} vol={int(r.volume):<8d} "
            f"value={r.value:<16.10g} vwap={r.vwap:<10.10g} oi={int(r.OPENINTS)}"
        )
    print()


def main() -> None:
    # P0-3 节后 02:30 那根带成交量的 padding bar = 被错标的日盘集合竞价
    show(
        "AG2506 @ 2025-05-06（劳动节后首日，夜盘本已取消）",
        pd.concat([
            rows("20250506", "AG2506", "02:25", "02:30"),
            rows("20250506", "AG2506", "09:00", "09:02"),
        ]),
        "02:29 及以前是 padding（价格恒为上一交易日收盘 8182）；02:30 突变成平 bar + 12,226 手；"
        "09:01 开盘 8158 —— 02:30 的 8156 距日盘开盘 2 点，距 padding 价 26 点",
    )
    show(
        "AG2506 @ 2025-05-07（对照：正常交易日，夜盘真实存在）",
        pd.concat([
            rows("20250507", "AG2506", "02:28", "02:30"),
            rows("20250507", "AG2506", "09:00", "09:02"),
        ]),
    )

    # P0-1 / P0-2 新旧口径对照
    show(
        "RB2205 @ 2022-01-04（旧格式期，元旦后首日）",
        rows("20220104", "RB2205", "09:00", "09:02"),
        "旧口径：无 padding 夜盘块，且 09:00 日盘集合竞价平 bar 真实存在（4,643 手）",
    )
    show(
        "RB2305 @ 2023-01-03（新格式期，元旦后首日）",
        pd.concat([
            rows("20230103", "RB2305", "22:58", "23:00"),
            rows("20230103", "RB2305", "09:00", "09:02"),
        ]),
        "新口径：凭空多出 21:00–23:00 的零成交 padding 夜盘，09:00 那根反而消失，日盘直接从 09:01 开始",
    )

    # P0-4 郑商所集合竞价 bar 不是平 bar
    show(
        "郑商所 vs 其他所 21:00 夜盘集合竞价 bar @ 2026-07-10",
        pd.concat([
            rows("20260710", c, "21:00", "21:01")
            for c in ("TA2609", "FG2609", "RB2610", "M2609")
        ]),
        "TA / FG（郑商所）的 21:00 带振幅；RB（上期所）、M（大商所）是标准平 bar",
    )

    # P0-2 周一回退逻辑
    show(
        "CU2603 @ 2026-01-05（元旦后首个交易日，周一）",
        pd.concat([
            rows("20260105", "CU2603", "22:58", "23:00"),
            rows("20260105", "CU2603", "09:00", "09:02"),
        ]),
        "padding 夜盘块被标成 TDATE=20260102（周五，实为假期），"
        "00:00–02:30 段被标成 TDATE=20260103（周六）—— 简单 weekday/weekend 回退的直接证据",
    )

    # P1-7 郑商所成交额凑整万元
    show(
        "FG2504 @ 2024-08-27（塞值并进有成交的 bar，vwap 失真）",
        pd.concat([
            rows("20240827", "FG2504", "14:58", "15:00"),
            rows("20240827", "FG2504", "09:00", "09:02"),
        ]),
        "15:00 那根 1 手、value=34,940、vwap=1747，而真实价 1329（20 吨/手 × 1329 = 26,580）",
    )
    show(
        "CY2505 @ 2024-08-27（全日仅 11 手，两根塞值 bar 清晰可见）",
        rows("20240827", "CY2505").query("volume > 0 or value != 0"),
        "真实成交额 1,113,350；09:01 塞 −8,775、15:00 塞 +5,425 → 全日 1,110,000 = 111.0000 万元",
    )


if __name__ == "__main__":
    sys.exit(main())
