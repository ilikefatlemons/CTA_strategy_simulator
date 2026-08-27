# -*- coding: utf-8 -*-
"""
lineA-03 · 多周期回调 —— 标注型测试窗口。

    python -m src.run_lineA_03

四张图 5m / 15m / 30m / 2h, 每格三层 (主图 + ATR + MACD)。左上角品种下拉与三个指标
开关, 右上角统计面板与五个标注开关。**默认只显示 Entry 与出场箭头**, 其余全关。

策略规格见 `docs/02-lineA-多周期回调/A3-单个自洽策略实施+改进/goal.md`:

    大周期过滤   2h   收盘>MA21>MA55 -> 多; 反之空; 否则保持上一状态 (无中间态)
    回调判断     15m  与 2h **完全反向**
    入场 锁1     15m  收盘[i-1] > MA21[i-1]
    入场 锁2     15m  水下金叉运行中 (做多) / 水上死叉运行中 (做空)
    SL           30m  入场价 ∓ 1.5 x ATR(14), 入场时冻结
    TP           30m  3 x ATR(14) 吊灯 (**同一个 ATR₀**, 也是冻结的)
    CD           2h   连续三次止损 -> 冻结到 2h 出现连续三根同方向的三线排列

**这是给人看的调试窗口, 不是用来挑参数的。** 单品种、无样本外; 任何「换个参数看哪个
好」的动作之前, 得先有一个测试证明它要修的问题真实存在。
"""
from __future__ import annotations

import os
import time

import pandas as pd

from src.data.paths import CLEAN_DIR, CLEAN_DIR_HINT, MANIFEST
from src.data.v3_sessions import derive_segments, prepare
from src.data.v3_timeframes import build_timeframes
from src.engine.lineA_03_backtest import 计算周期, 跑回测
from src.performance.lineA_03_stats import 算统计, 终端版
from src.strategy.lineA_03 import 策略参数, 策略开关
from src.viz.chart_lineA_03 import show_lineA_03

# ---------------------------------------------------------------------------
# 调参区
# ---------------------------------------------------------------------------
START = "2025-09-01"
END = "2026-07-29"

# 品种池。**当前只做黄金主力连续一个品种**, 左上角下拉里就一个。
# 要放回全部品种: 改成 None, 就从 _manifest.csv 读全表。
品种池: list[str] | None = ["AU"]
默认品种 = "AU"

# 开窗时只画最后这么多个交易日 —— 全区间是 50 万根 1m, 全量渲染会卡。
# **这只影响看到什么, 不影响面板上的统计**: 两个口径是分开的。
可见交易日 = 60

# 回调策略需要 2h 的 MA55 暖机。AU 每交易日约 8 根 2h, 55 根 ≈ 7 个交易日;
# 无夜盘品种每天只有 4 根 -> 约 14 个交易日。60 是从老配置继承的, 宽松但无害。
暖机天数 = 60

参数 = 策略参数()
开关 = 策略开关()

DEBUG = False             # True -> 打开 webview devtools, 调 JS 时用
# ---------------------------------------------------------------------------


def 加载(品种: str):
    """`(周期字典, 回测结果, 统计, 参数)` —— 图表层要的四样。"""
    t0 = time.time()
    prep = prepare(品种, start=START, end=END, warmup_days=暖机天数)
    tf = build_timeframes(prep.df, segments=derive_segments(prep.df), tfs=计算周期)
    结果 = 跑回测(prep, tf, 参数=参数, 开关=开关)
    每日 = tf[参数.大周期].n_bars / max(prep.df["trading_date"].nunique(), 1)
    统 = 算统计(结果, 每日)
    print(f"\n[{品种}] {len(prep.df):,} 根 1m · {prep.df['trading_date'].nunique()} 交易日 "
          f"· 暖机 {prep.warmup_bars:,} 根 · {time.time() - t0:.1f}s")
    print(终端版(统))
    return tf, 结果, 统, 参数


def main() -> None:
    if not os.path.exists(MANIFEST):
        raise SystemExit(
            f"找不到 {MANIFEST}\n"
            f"清洗后的分品种 parquet 应该在 {CLEAN_DIR_HINT}/;\n"
            f"没有的话先跑: python -m src.data.prepare_v3_minute"
        )
    全部 = sorted(pd.read_csv(MANIFEST, index_col="sym").index.astype(str))
    symbols = 全部 if 品种池 is None else [s for s in 品种池 if s in 全部]
    if not symbols:
        raise SystemExit(f"品种池 {品种池} 一个都不在 _manifest.csv 里; 可选: {全部}")
    默认 = 默认品种 if 默认品种 in symbols else symbols[0]

    print("=" * 72)
    print("lineA-03 · 多周期回调（标注型测试窗口）")
    print(f"  品种      {默认}（{len(symbols)} 个可选）")
    print(f"  区间      {START} ~ {END}   开窗可见最后 {可见交易日} 个交易日")
    print(f"  周期      {参数.大周期} 定向 · {参数.回调周期} 回调+入场 · "
          f"{参数.风险周期} 风险   （驱动时钟 {参数.回调周期}）")
    print(f"  数据      {CLEAN_DIR}")
    print("=" * 72)

    show_lineA_03(symbols, 加载, default=默认, 可见交易日=可见交易日, debug=DEBUG)


if __name__ == "__main__":
    main()
