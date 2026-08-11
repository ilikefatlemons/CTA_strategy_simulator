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
# ---------------------------------------------------------------------------


def main() -> None:
    man_path = os.path.join(CLEAN_DIR, "_manifest.csv")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"找不到 {man_path}\n先跑: python -m src.data.prepare_v3_minute"
        )
    man = pd.read_csv(man_path, index_col="sym")
    symbols = sorted(man.index.astype(str))

    default = DEFAULT_SYMBOL if DEFAULT_SYMBOL in symbols else symbols[0]
    print(f"{len(symbols)} symbols | window {START} ~ {END} | adjust={ADJUST}")
    show_minute_candles(symbols, default=default, start=START, end=END, adjust=ADJUST)


if __name__ == "__main__":
    main()
