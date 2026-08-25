# -*- coding: utf-8 -*-
"""
v3.0: 一分钟 K 线浏览器。

整屏只有 K 线, 无任何指标; 左上角一个可搜索 + 键盘上下选择的 ticker 选择器。

前置: 先跑一次 ETL 生成分品种 parquet
    python -m src.data.prepare_v3_minute

再跑
    python -m src.run_v3_minute_chart
"""
import os

import pandas as pd

from src.viz.chart_minute import CLEAN_DIR, show_minute_candles

# ---------------------------------------------------------------------------
# 调参区
# ---------------------------------------------------------------------------
START = "2025-09-30"      # None -> 各品种全历史 (单品种上百万根, 会明显变慢)
END = "2025-10-10"
DEFAULT_SYMBOL = "AU"
ADJUST = True             # True -> close/K*K0 后复权, 消除换月跳空; False -> 裸价

# 品种池。**当前只做黄金主力连续一个品种**（2026-08-25 决定）。
#
# 为什么收到一个: 59 个品种的横截面统计会把「策略本身对不对」和「参数在哪些品种上
# 恰好合适」搅在一起 —— 而后者在样本外为零的前提下基本是噪声
# (`docs/02-lineA-多周期回调/Archive-美股测试/Archive-审计与修正-v2.0~v2.1/`
#  `v2.1-LEDG-试验次数与过拟合折扣.md` 记着已跑过 40+ 组配置)。
# 先在一个品种上把一条策略走通、走对, 再谈推广。
#
# 要放回全部品种: 把下面这行改成 None, 品种池就重新从 _manifest.csv 读全表。
# ⚠ lineB (R-Breaker) 的立论依赖跨品种统计 (突破/反转逐品种净收益相关系数 +0.145),
#   做 lineB 的横截面工作时必须先放回去。
品种池: list[str] | None = ["AU"]

# ---------------------------------------------------------------------------


def main() -> None:
    man_path = os.path.join(CLEAN_DIR, "_manifest.csv")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"找不到 {man_path}\n先跑: python -m src.data.prepare_v3_minute"
        )
    man = pd.read_csv(man_path, index_col="sym")
    全部 = sorted(man.index.astype(str))
    symbols = 全部 if 品种池 is None else [s for s in 品种池 if s in 全部]
    if not symbols:
        raise SystemExit(f"品种池 {品种池} 一个都不在 _manifest.csv 里; 可选: {全部}")

    default = DEFAULT_SYMBOL if DEFAULT_SYMBOL in symbols else symbols[0]
    print(f"{len(symbols)} symbols {symbols if len(symbols) <= 5 else ''} "
          f"| window {START} ~ {END} | adjust={ADJUST}")
    show_minute_candles(symbols, default=default, start=START, end=END, adjust=ADJUST)


if __name__ == "__main__":
    main()
