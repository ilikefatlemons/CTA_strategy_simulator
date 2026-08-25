# -*- coding: utf-8 -*-
"""
数据路径的唯一真源。

在这之前 `CLEAN_DIR` 是三份互相复制粘贴的硬编码字面量 (v3_sessions.py /
prepare_v3_minute.py / chart_minute.py), 而且写死了盘符。2026 年那次 data/
重组把 `03-pkl层` 换成了 `01-pkl层/一次排查`, 三处全部失效 —— 更糟的是
6 个测试文件 guard 在 `os.path.exists(CLEAN_DIR/"RB.parquet")` 上, 于是它们
**静默跳过**, `pytest -q` 看着全绿, 实际只有手搭 frame 的那两个在跑。

所以这里有两条约定:
  1. 路径由 `__file__` 往上推, 不写盘符 —— 换机器、换盘、放进容器都不用改。
  2. 只有这一个文件知道目录长什么样。别处一律 import, 不许再抄字面量。

同样的写法先例: data/00-跨层/compare_layers.py:22。
"""
from __future__ import annotations

from pathlib import Path

# src/data/paths.py -> src/data -> src -> 仓库根
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

# 二级数据: 主力连续 + 复权因子 K, 79 品种 6600 万行
RAW_PKL = DATA_DIR / "01-pkl层" / "all_symbol_min_full_main_close_k_1.pkl"

# 清洗后的 59 个分品种 parquet + _manifest.csv —— 回测与画图只读这里
CLEAN_DIR = str(DATA_DIR / "01-pkl层" / "一次排查" / "clean_1m")

MANIFEST = str(Path(CLEAN_DIR) / "_manifest.csv")

# 出现在 skip reason / SystemExit 文案里, 让人一眼看到该去哪
CLEAN_DIR_HINT = "data/01-pkl层/一次排查/clean_1m"
