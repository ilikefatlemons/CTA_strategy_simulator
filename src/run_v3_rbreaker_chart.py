# -*- coding: utf-8 -*-
"""
v3.0: 中国商品期货的图表终端 —— 三个策略共用一个窗口。

整屏一分钟 K 线 + 叠加层 + 中文统计面板。左上角 ticker 搜索框 (键盘上下 + Enter),
下面一个「指标线 六条」开关; 右上角「策略结果统计」/ Archive-Pullback /
Breakout / Reversal, 后三个三向互斥。

**三个互相独立的策略**:
  Breakout          —— 上穿 Bbreak 做多 / 下破 Sbreak 做空 (只读两条线)
  Reversal          —— 冲到 Ssetup 武装后回落跌破 Senter 做空, 反之做多 (读四条线)
  Archive-Pullback  —— v2.1 phaseF 那套四周期 MA 级联回调入场, 整屏换成
                       1m/5m/30m/1d 四张 K 线 + 各自的 ATR 副图

前两个共用同一套基线执行纪律, 追踪止损用同一个 giveback 公式, 因此初始止损距离
逐字节相同、结果天然可比。

**第三个不一样, 用之前要知道的三件事**:
  1. 周期角色是 5m/15m/30m/2h -> 1m/5m/30m/1d 的一一映射, 策略逻辑一个字没改
     (`tests/test_pullback_v3_vs_v21.py` 对着未改动的 v2.1 引擎逐笔判等)。但这个
     重映射把时间常数拉长了约 12 倍 —— 趋势方向从一天变 3 次变成一天变 1 次。
     **结果不能和 docs/v2.1 的账本比。**
  2. **时段是锁死的**: 未选中的时段完全空仓且无交易, 持仓在本时段最后一根 1m bar
     的 close 被强制平掉 (出场原因 `End Session`)。这和 R-Breaker 两个引擎是同一
     套纪律 (`rbreaker_backtest.py:203-206`)。因此五种时段模式两两不同 —— RB 一年
     的强平点数: 日夜分开 510 (每天两次)、自然日 日→夜 259、交易日 夜→日 258、
     日盘 258、夜盘 252。**换时段就是换策略, 不是换视图。**
     想看"持仓穿越时段边界"的旧行为: `PullbackConfig(session_forces_flat=False)`。
  3. 它需要 60 个交易日的暖机段 (日线 MA50), 由 `PB_WARMUP_DAYS` 控制。暖机段
     不显示也不交易; 凑不满时统计面板副标题会标「暖机不足」。

前置: 先跑一次 ETL 生成分品种 parquet
    python -m src.data.prepare_v3_minute

再跑
    python -m src.run_v3_rbreaker_chart

只想看裸 K 线不带策略的话用 `python -m src.run_v3_minute_chart`。
"""
import os

import pandas as pd

from src.data.v3_sessions import CLEAN_DIR
from src.strategy.pullback import PullbackConfig, PullbackParams
from src.strategy.rbreaker import RBreakerConfig, RBreakerParams, ReversalConfig
from src.viz.chart_rbreaker import show_rbreaker_chart

# ---------------------------------------------------------------------------
# 调参区
# ---------------------------------------------------------------------------
START = "2023-01-01"      # 约 250 个交易日。窗口越长, 首次选中一个品种越慢:
END = "2025-07-02"        # 瓶颈不是回测(130 万根 0.34s), 是往图表推数据
DEFAULT_SYMBOL = "AU"     # (js_data 实测 8.5 万行 0.36s / 16MB, 100 万行 3.7s / 193MB)

PARAMS = RBreakerParams(
    f1=0.35,   # Ssetup / Bsetup
    f2=0.07,   # Senter / Benter
    f3=0.25,   # Bbreak / Sbreak
    d=3.0,     # 追踪止损分母; 极限保留 MFE 的 1/d
)
# 突破策略 (右上角 Breakout)
CONFIG = RBreakerConfig(
    entry_trigger_on_range=True,   # bar i-1 的任意价格穿线即触发 (非收盘价)
    one_trade_per_session=True,    # 白盘一笔 + 夜盘一笔, 一天最多两笔
    gap_fill_exits=True,
    cost_bps=0.0,                  # 往返成本(基点)。0 = 毛收益, 见下面的提醒
)

# 反转策略 (右上角 Reversal)。与突破共用 PARAMS 的 f1/f2/f3/d ——
# 两者的初始止损距离因此逐字节相同, 结果天然可比。
REV_CONFIG = ReversalConfig(
    arm_must_precede_trigger=True,  # 允许同一根既武装又触发
    entry_trigger_on_range=True,
    one_trade_per_session=True,
    gap_fill_exits=True,
    cost_bps=0.0,
)

# 时段划分模式的初值 (窗口里可以随时用搜索框右边那个选择器改)。
#   "night"      只做夜盘
#   "day"        只做日盘
#   "night_day"  一个交易日的夜+日合成一个时段
#   "day_night"  白盘 + 紧随其后的夜盘合成一个时段
#   "split"      夜、日各算一个时段 (默认, 也是历史行为)
SESSION_MODE = "split"

# 回调策略 (右上角 Archive-Pullback)。**默认值逐字沿用 v2.1**, 改任何一个都意味着
# "策略变了"、不再是移植 —— 每个数的出处标在 src/strategy/pullback.py 里。
PB_PARAMS = PullbackParams()
PB_CONFIG = PullbackConfig(
    cost_bps=0.0,             # 与另外两个策略同口径, 见下面的提醒
    # 两条一起才是"时段锁死"。都是默认值, 显式写出来是为了好做 A/B ——
    # 把 session_forces_flat 改成 False 就退回"持仓穿越时段边界"的旧行为。
    session_gates_entry=True,   # 不可交易时段不开仓
    session_forces_flat=True,   # 时段末根 close 强制平仓
    # 冷静期的第二个触发器: 单根日线实体吃穿 >=2 条均线。改成 False 直接对比。
    # 它比看上去重的多 —— 开着时策略约**一半时间**在冷静期里 (59 品种均值 45~51%),
    # 关掉降到 10~25%, 批次 +14%。
    cooldown_on_structure_break=True,
)
# 暖机段的交易日数。趋势方向要日线 MA50, 所以至少 50; 60 留了 10 根余量。
# 这一段不显示也不交易, 只为把指标算热。调大不会改变结果, 只是多读一点 parquet。
PB_WARMUP_DAYS = 60

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

DEBUG = False             # True -> 打开 webview devtools, 调 JS 时用
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
          f"| window {START} ~ {END} | 时段 {SESSION_MODE}")
    print(f"params f1={PARAMS.f1:g} f2={PARAMS.f2:g} f3={PARAMS.f3:g} d={PARAMS.d:g}"
          f" | cost {CONFIG.cost_bps:g} bp")
    if CONFIG.cost_bps == 0:
        # 按每时段一笔算, 一年 100~200 笔; 2bp 的往返成本就是 2~4% 的累计
        # 拖累 —— 对一个年收益在 ±5% 量级的策略, 这一项能直接决定正负号。
        print("提醒: cost_bps=0, 面板上的「累计净收益」是**毛**收益。")

    show_rbreaker_chart(
        symbols, default=default, start=START, end=END,
        params=PARAMS, config=CONFIG, rev_config=REV_CONFIG,
        session_mode=SESSION_MODE, debug=DEBUG,
        pb_params=PB_PARAMS, pb_config=PB_CONFIG, pb_warmup_days=PB_WARMUP_DAYS,
    )


if __name__ == "__main__":
    main()
