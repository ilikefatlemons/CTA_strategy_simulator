# -*- coding: utf-8 -*-
"""
R-Breaker 的六条线 + 追踪止损几何。纯数学, 不碰数据也不碰图表。

六条线由**前一交易日**的 H/L/C 投射出来, 当日固定不动, 次日整套重算:

    Ssetup = H + f1*(C - L)                Bsetup = L - f1*(H - C)
    Senter = ((1+f2)/2)*(H + C) - f2*L     Benter = ((1+f2)/2)*(L + C) - f2*H
    Bbreak = Ssetup + f3*(Ssetup - Bsetup) Sbreak = Bsetup - f3*(Ssetup - Bsetup)

自上而下的顺序是 Bbreak > Ssetup > Senter > Benter > Bsetup > Sbreak
(证明见 tests/test_rbreaker_lines.py::test_strict_line_ordering)。

**出处声明**: R-Breaker 没有作者本人发表的规范文档, 网上流传的公式版本彼此
有出入 —— 见 docs/03-lineB-R-Breaker/B0-原理与出处/v3.0-RSCH-R-Breaker与Turtle运行逻辑.md 第十节。这里用的是
最常见的那一版重构, 系数 f1=0.35 / f2=0.07 / f3=0.25。引用任何回测结果时
都应该带上这组系数。

本次只实现**突破**半边: 价格上穿 Bbreak 做多、下破 Sbreak 做空。反转半边
(持仓时触及 Ssetup 后跌破 Senter 反手) 不实现, 但 Senter/Benter 仍然参与
追踪止损, 另外四条线也照常计算并可在图上显示。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 自上而下。图表按这个顺序建 series, 面板/图例也按这个顺序。
LINE_NAMES = ("Bbreak", "Ssetup", "Senter", "Benter", "Bsetup", "Sbreak")

# 突破策略实际读的两条线。
BREAK_LINES = ("Bbreak", "Sbreak")

# 反转策略实际读的四条线。Bbreak/Sbreak 不参与入场, 但作为止损的天花板/地板
# 出现在 trail_stop_rev_* 里 —— 见那两个函数的推导。
REVERSAL_LINES = ("Ssetup", "Senter", "Benter", "Bsetup")


@dataclass(frozen=True)
class RBreakerParams:
    """六条线的系数 + 追踪止损的分母。"""

    f1: float = 0.35   # Ssetup / Bsetup
    f2: float = 0.07   # Senter / Benter
    f3: float = 0.25   # Bbreak / Sbreak
    # 追踪止损分母。极限情况下锁定 MFE 的 1/d —— d=3 即"最多回吐峰值浮盈的
    # 三分之二"。注意 d 越大止损越松: d=3 时初始止损在入场价下方约 0.6 倍
    # 前日振幅, 要到 MFE ≈ 1.79 倍前日振幅才回到保本。
    d: float = 3.0


@dataclass(frozen=True)
class RBreakerConfig:
    """
    行为开关。沿用 `src/config.py::BacktestConfig` 的约定: 每一个会改变结果的
    行为都在这里有一个具名字段, 好让"旧行为 vs 新行为"能在同一段数据上做 A/B,
    而不是被静默烤进代码。默认值就是当前采用的口径。
    """

    # ---- 入场 ----
    # True: 只要 bar i-1 的**任意价格**穿过线就触发 (high > Bbreak / low < Sbreak)。
    # False: 要求 bar i-1 **收盘**站上/跌破。两者都成交在 bar i 的 open。
    entry_trigger_on_range: bool = True
    # 每个交易时段最多一笔 (白盘一笔、夜盘一笔, 一天最多两笔)。
    # False -> 同一时段内止损后可以无限次再入场。
    one_trade_per_session: bool = True

    # ---- 出场 ----
    # 跳空穿越止损时按 bar 的 open 成交(更差), 而不是按市场根本没成交过的
    # 止损线成交。10:15->10:31 / 11:30->13:31 以及每个时段边界都是零根 bar,
    # 那里的跳空是真实的。
    gap_fill_exits: bool = True

    # ---- 参照日 ----
    min_ref_bars: int = 30          # 参照交易日清洗后至少这么多根, 否则当日不交易
    max_reference_gap_days: int = 15  # 与上一个可用交易日的日历间隔上限(春节约 10 天)

    # ---- 成本 ----
    # 单次往返成本(基点), 直接从每笔 pnl 里扣。0 = 毛收益。
    # 仓库目前没有任何 tick_size / 合约乘数 / 费率表, 所以这里只能是一个旋钮,
    # 接到真实合约参数后再填。参考 docs/v3.0/中国商品期货特殊规则/
    # v3.0-PLAN-中国期货特殊规则应对方案.md 11.2 的测算路径。
    cost_bps: float = 0.0


@dataclass(frozen=True)
class ReversalConfig:
    """
    反转策略的开关。字段与 `RBreakerConfig` 大部分同名同义 —— 两个策略共用
    `src/data/v3_sessions.py::prepare()`, 所以数据层那两个字段必须在这里也有,
    否则 `prepare()` 拿不到。

    刻意不做成 RBreakerConfig 的子类: 两个策略是独立的, 一边加开关不应该
    悄悄改变另一边的默认行为。
    """

    # ---- 武装 ----
    # True: 要求武装严格发生在触发**之前**的某一根 (运行极值退到 i-2)。
    # False(默认): 运行极值取到 i-1 为止(含), 所以一根 high>=Ssetup 且
    #   low<Senter 的 bar 会当场武装并触发, 次根开盘成交。这需要单根跨越
    #   Ssetup-Senter (RB 约 0.44%), 1 分钟里罕见但存在。
    arm_must_precede_trigger: bool = False

    # ---- 入场 ----
    # True: bar i-1 的**任意价格**穿过反转线即触发 (low < Senter / high > Benter)。
    # False: 要求 bar i-1 **收盘**跌破/站上。两者都成交在 bar i 的 open。
    entry_trigger_on_range: bool = True
    one_trade_per_session: bool = True

    # ---- 出场 ----
    gap_fill_exits: bool = True

    # ---- 参照日 (给 prepare() 用, 与 RBreakerConfig 同值) ----
    min_ref_bars: int = 30
    max_reference_gap_days: int = 15

    # ---- 成本 ----
    cost_bps: float = 0.0


def rbreaker_lines(high, low, close, params: RBreakerParams = RBreakerParams()):
    """
    六条线。标量和 pandas.Series 输入都接受 —— 全是广播运算。

    Series 进 -> DataFrame 出(列名 = LINE_NAMES, 索引沿用输入);
    标量进 -> dict 出。
    """
    p = params
    ssetup = high + p.f1 * (close - low)
    bsetup = low - p.f1 * (high - close)
    senter = ((1 + p.f2) / 2) * (high + close) - p.f2 * low
    benter = ((1 + p.f2) / 2) * (low + close) - p.f2 * high

    # 恒等式: spread == (1 + f1) * (H - L)。因此
    #     Bbreak - Ssetup == f3 * (1 + f1) * (H - L) == 0.3375 * R
    # 这就是追踪止损公式里那个 0.3375 的来源 —— 它不是一个裸常数, 而是
    # f3*(1+f1) 乘上前日振幅。见 trail_stop_long 的推导。
    spread = ssetup - bsetup
    bbreak = ssetup + p.f3 * spread
    sbreak = bsetup - p.f3 * spread

    out = {
        "Bbreak": bbreak, "Ssetup": ssetup, "Senter": senter,
        "Benter": benter, "Bsetup": bsetup, "Sbreak": sbreak,
    }
    if isinstance(high, pd.Series):
        return pd.DataFrame(out, index=high.index)[list(LINE_NAMES)]
    return out


# --------------------------------------------------------------- 追踪止损 ----
#
# 原始写法(giveback from peak):
#
#     giveback = (Bbreak - Senter) - (Bbreak - Ssetup)/d + MFE_geo*(1 - 1/d)
#     stop     = peak - giveback
#
# 其中 MFE_geo = peak - Bbreak, 是一个**几何量**: 它只驱动止损位, 和实际成交价
# 无关。实际成交价 open[i] 与 Bbreak 之间的差额全额计入盈亏、零传导到止损位。
#
# 代入化简:
#
#     stop = peak - (Bbreak - Senter) + (Bbreak - Ssetup)/d - (peak - Bbreak)(1 - 1/d)
#          = peak - Bbreak + Senter + (Bbreak - Ssetup)/d - (peak - Bbreak) + (peak - Bbreak)/d
#          = Senter + (Bbreak - Ssetup + peak - Bbreak)/d
#          = Senter + (peak - Ssetup)/d
#
# 空头由反射 p -> -p, (H,L,C) -> (-L,-H,-C) 得到(该反射把 Ssetup <-> -Bsetup、
# Senter <-> -Benter、Bbreak <-> -Sbreak 互换):
#
#     stop = Benter - (Bsetup - trough)/d
#
# peak 单调不减 => 多头 stop 单调不减; trough 单调不增 => 空头 stop 单调不增。
# 也就是说这是一个真正的追踪止损, 永不放松。
#
# 注意**没有保本地板**: v2.1 的吊灯止损有 floor_stop, 这个刻意没有。


def trail_stop_long(senter, ssetup, peak, d: float):
    """多头追踪止损位。`peak` = 入场以来 high 的运行最大值, 下限 Bbreak。"""
    return senter + (peak - ssetup) / d


def trail_stop_short(benter, bsetup, trough, d: float):
    """空头追踪止损位。`trough` = 入场以来 low 的运行最小值, 上限 Sbreak。"""
    return benter - (bsetup - trough) / d


def trail_stop_long_from_giveback(bbreak, senter, ssetup, peak, d: float):
    """
    上面那个闭式解的**参照实现** —— 原始公式一字不改地照抄。

    存在的唯一理由是让 tests/test_rbreaker_lines.py 能对着它验证闭式解, 把
    "0.3375 = f3*(1+f1)" 这个化简钉死。生产路径不调用它。
    """
    mfe_geo = peak - bbreak
    giveback = (bbreak - senter) - (bbreak - ssetup) / d + mfe_geo * (1 - 1 / d)
    return peak - giveback


def trail_stop_short_from_giveback(sbreak, benter, bsetup, trough, d: float):
    """空头的参照实现, 与 trail_stop_long_from_giveback 对称。"""
    mfe_geo = sbreak - trough
    giveback = (benter - sbreak) - (bsetup - sbreak) / d + mfe_geo * (1 - 1 / d)
    return trough + giveback


# ------------------------------------------------------- 反转策略的追踪止损 ----
#
# **和突破半边是同一个 giveback 表达式、同一组常数, 只换入场线。**
#
#     giveback = (Bbreak - Senter) - (Bbreak - Ssetup)/d + MFE_geo*(1 - 1/d)
#
# 突破**多**头从 Bbreak 入场, 反转**空**头从 Senter 入场 —— 两者代进去的是
# 同一个 giveback。于是初始风险逐字节相同:
#
#     初始风险_突破多 = Bbreak - stop = (Bbreak - Senter) - (Bbreak - Ssetup)/d
#     初始风险_反转空 = stop - Senter = (Bbreak - Senter) - (Bbreak - Ssetup)/d
#
# 实测 (RB, 参照日 H/L/C = 3085/3053/3061):
#     突破多 入场 3098.60 止损 3078.00 风险 20.60
#     反转空 入场 3074.40 止损 3095.00 风险 20.60   <- 同距
# 同理 突破空 与 反转多 共用另一个常数, 风险同为 17.64。
# 两个策略因此**天然风险可比**, 这正是 A/B 两个策略时想要的性质。
#
# 化简 (空):
#     stop = trough + (Bbreak-Senter) - (Bbreak-Ssetup)/d + (Senter-trough)(1-1/d)
#          = Bbreak - [(Bbreak - Ssetup) + (Senter - trough)]/d
#          = Bbreak - [(Bbreak - Ssetup) + MFE_geo]/d
#
# 多头由反射 p -> -p, (H,L,C) -> (-L,-H,-C) 得到。
#
# 语义上也是对的: 反转空的止损落在 Bbreak 正下方 (Bbreak-Ssetup)/d 处 ——
# 逆势空单错在哪里? 错在市场真的向上突破了。这个位置是镜像自动给出的。
#
# MFE_geo 从**理论线**量起 (空: Senter - trough; 多: peak - Benter), 入场时
# 为 0, 单调不减 => 止损单调收紧, 永不放松。同样**不设保本地板**。


def trail_stop_rev_short(bbreak, ssetup, mfe_geo, d: float):
    """反转空头的止损位。`mfe_geo = Senter - trough`, 入场时为 0。"""
    return bbreak - ((bbreak - ssetup) + mfe_geo) / d


def trail_stop_rev_long(sbreak, bsetup, mfe_geo, d: float):
    """反转多头的止损位。`mfe_geo = peak - Benter`, 入场时为 0。"""
    return sbreak + ((bsetup - sbreak) + mfe_geo) / d


def trail_stop_rev_short_from_giveback(bbreak, ssetup, senter, trough, d: float):
    """
    反转空的**参照实现** —— giveback 公式一字不改地照抄, 入场线换成 Senter。

    存在的唯一理由是让测试对着它验证上面的闭式解。注意它和
    `trail_stop_long_from_giveback` (突破多) 用的是**同一个** giveback 表达式。
    """
    mfe_geo = senter - trough
    giveback = (bbreak - senter) - (bbreak - ssetup) / d + mfe_geo * (1 - 1 / d)
    return trough + giveback


def trail_stop_rev_long_from_giveback(sbreak, bsetup, benter, peak, d: float):
    """反转多的参照实现。与 `trail_stop_short_from_giveback` (突破空) 同式。"""
    mfe_geo = peak - benter
    giveback = (benter - sbreak) - (bsetup - sbreak) / d + mfe_geo * (1 - 1 / d)
    return peak - giveback
