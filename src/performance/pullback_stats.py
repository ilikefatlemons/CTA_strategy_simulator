# -*- coding: utf-8 -*-
"""
回调策略的结果统计 + 图上那个中文面板的行内容。

版式沿用 `src/performance/rbreaker_stats.py` (v3.0 现行那一版); **老 phaseF 那套
统计 UI —— 三页表格、组合权益、权重饼图 —— 全部丢弃**。

统计口径沿用仓库既有约定 (`src/performance/trade_stats.py`):
  * 亏损桶是 `p <= 0` —— 持平的一笔算亏, 这样胜率和盈亏比互相自洽。
  * 收益率一律是**净的**(扣掉 cost_bps), 存成普通小数, 格式化层再乘 100。
  * 最大回撤是**负**分数; 面板显示时取绝对值。
  * 最大回撤基于**逐根 mark-to-market** 权益曲线, 不是逐笔平仓权益。

---------------------------------------------------------------------------
胜率和盈亏比按「批」算, 不是按「腿」
---------------------------------------------------------------------------
一批仓位内部有两条腿: 2R 处平掉 50% (TP), 剩余 50% 用吊灯跑趋势 (PROTECTIVE_SL);
或者整批一次性止损 (SL)。按腿统计会把"止盈那一半"和"回吐那一半"记成两笔独立
交易, 胜率被系统性抬高 —— 一批只要走到 2R, 就白送一个"胜"。

所以一批 = 一个 `entry_bar_idx`, 收益 = `Σ size_fraction x pnl_net`。这是
`portfolio_pullback_backtest._batch_pnls` (:123-131) 的定义, 逐字沿用。
面板上的标签写明「按批」, 免得和 R-Breaker 面板的「成交」混着看。

---------------------------------------------------------------------------
两条要如实标注的东西
---------------------------------------------------------------------------
1. **暖机不足**: 趋势方向要日线 MA50, 显示窗口开始前必须先有 50 根日线。上市晚
   或被 `USABLE_FROM` 裁过的品种凑不满, 于是方向一直 neutral、一笔不做 ——
   屏幕上"没信号"和"没暖好"长得一模一样。凑不满时副标题追加「暖机不足 N/50」。
2. **收盘平仓的笔数要单独列出来**: 时段锁死 (`session_forces_flat`, 默认开) 下,
   持仓在时段最后一根 1m bar 的 close 被强平, 出场原因是 `END_SESSION`。在
   「日盘」「日夜分开」这类时段短的模式下它会是**最主要**的出场方式 —— 混进
   止损或止盈里会让人误判策略的实际行为, 所以面板离场那行是四个数字。
   注意一批可能**同时**有 TP 和 END_SESSION 两条腿 (2R 止盈那一根正好是时段末根),
   所以 `n_tp + n_sl + n_psl + n_end` 不等于批次数, 只等于腿数。
   开关关掉时窗口最后一根可能仍持仓, 那一批不产生 `PBTrade` —— 浮盈在权益曲线里
   但不进交易表, 「批次」会比图上最后一个入场箭头少一个。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.pullback_v3_backtest import (
    END_SESSION, PROTECTIVE_SL, SL, TP, PullbackResult,
)
from src.performance.sharpe import sharpe_ratio_cn
from src.performance.trade_stats import max_drawdown, payoff, win_rate
from src.strategy.pullback import PullbackParams


@dataclass(frozen=True)
class PullbackStats:
    symbol: str
    mode_label: str          # 时段划分模式的中文名, 进副标题
    tradable_days: int
    n_batches: int
    n_long: int
    n_short: int
    n_legs: int
    total_return: float      # 普通小数
    max_drawdown: float      # 负分数
    win_rate: float          # **按批**
    payoff: float            # **按批**
    sharpe: float
    n_sl: int
    n_tp: int
    n_psl: int
    n_end: int               # 时段末强平的腿数 (session_forces_flat 关掉时恒为 0)
    n_gapped: int
    avg_bars_held: float     # 按批: 入场 -> 最后一条腿离场
    warmup_days: int
    warmup_needed: int
    ma: tuple[int, int, int] = (5, 20, 50)
    rr_trigger: float = 2.0
    sl_atr_mult: float = 1.5
    chandelier_mult: float = 3.0

    @property
    def warmup_short(self) -> bool:
        return self.warmup_days < self.warmup_needed


def batch_pnls(trades) -> dict[int, tuple[str, float]]:
    """
    `entry_bar_idx -> (方向, Σ size_fraction x pnl_net)`。

    `portfolio_pullback_backtest._batch_pnls` (:123-131) 的逐字重写 —— 那边不能
    直接 import, 它会把整个 v2.1 引擎栈 (src.config / src.rules.* /
    src.data.resample) 拖进 v3.0。判等由 `tests/test_pullback_stats.py` 保证。
    """
    out: dict[int, tuple[str, float]] = {}
    for t in trades:
        direction, acc = out.get(t.entry_bar_idx, (t.direction, 0.0))
        out[t.entry_bar_idx] = (direction, acc + t.size_fraction * t.pnl_net)
    return out


def _avg_batch_bars(trades) -> float:
    spans: dict[int, int] = {}
    for t in trades:
        spans[t.entry_bar_idx] = max(spans.get(t.entry_bar_idx, 0),
                                     t.exit_bar_idx - t.entry_bar_idx)
    return float(np.mean(list(spans.values()))) if spans else float("nan")


def compute_stats(
    result: PullbackResult, mode_label: str = "",
    params: PullbackParams | None = None,
) -> PullbackStats:
    p = params or result.params
    tr = result.trades
    batches = batch_pnls(tr)
    pnls = [v for _, v in batches.values()]
    eq = result.equity_curve
    return PullbackStats(
        symbol=result.symbol,
        mode_label=mode_label,
        tradable_days=result.tradable_days,
        n_batches=len(batches),
        n_long=sum(1 for d, _ in batches.values() if d == "long"),
        n_short=sum(1 for d, _ in batches.values() if d == "short"),
        n_legs=len(tr),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else float("nan"),
        max_drawdown=max_drawdown(eq),
        win_rate=win_rate(pnls),
        payoff=payoff(pnls),
        sharpe=sharpe_ratio_cn(eq, result.trading_date),
        n_sl=sum(1 for t in tr if t.reason == SL),
        n_tp=sum(1 for t in tr if t.reason == TP),
        n_psl=sum(1 for t in tr if t.reason == PROTECTIVE_SL),
        n_end=sum(1 for t in tr if t.reason == END_SESSION),
        n_gapped=sum(1 for t in tr if t.gapped),
        avg_bars_held=_avg_batch_bars(tr),
        warmup_days=result.warmup_days,
        warmup_needed=p.ma_slow,
        ma=(p.ma_fast, p.ma_mid, p.ma_slow),
        rr_trigger=p.rr_trigger, sl_atr_mult=p.sl_atr_mult,
        chandelier_mult=p.chandelier_mult,
    )


# ---------------------------------------------------------------- 格式化 ----
def _pct(x: float, signed: bool = True) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.2%}" if signed else f"{abs(x):.2%}"


def _num(x: float, fmt: str) -> str:
    return "n/a" if pd.isna(x) else format(x, fmt)


def panel_title(s: PullbackStats) -> str:
    return f"{s.symbol} · 回调入场 (Archive)"


def panel_subtitle(s: PullbackStats) -> str:
    f, m, sl = s.ma
    parts = [f"MA{f}/{m}/{sl} · {s.rr_trigger:g}R+{s.chandelier_mult:g}ATR吊灯"]
    if s.mode_label:
        parts.append(f"时段 {s.mode_label}")
    if s.warmup_short:
        # 「一笔不做」和「没暖好」在屏幕上长得一样, 必须标出来
        parts.append(f"⚠ 暖机不足 {s.warmup_days}/{s.warmup_needed}")
    return " · ".join(parts)


def panel_rows(s: PullbackStats) -> list[tuple[str, str]]:
    """面板的九行 (标签, 值)。顺序固定, 与 R-Breaker 面板对齐好并排看。"""
    return [
        ("可交易日", f"{s.tradable_days}"),
        ("批次", f"{s.n_batches} (多{s.n_long}/空{s.n_short})"),
        ("累计净收益", _pct(s.total_return)),
        ("最大回撤", _pct(s.max_drawdown, signed=False)),
        ("胜率(按批)", "n/a" if pd.isna(s.win_rate) else f"{s.win_rate:.1%}"),
        ("盈亏比(按批)", _num(s.payoff, ".2f")),
        ("策略夏普比", _num(s.sharpe, "+.2f")),
        # 四个数是**腿**数不是批数: 2R 止盈那一根正好是时段末根时, 一批会同时贡献
        # 一条 TP 腿和一条收盘腿。
        ("离场 止损/止盈/护盈/收盘", f"{s.n_sl} / {s.n_tp} / {s.n_psl} / {s.n_end}"),
        ("平均持有", "n/a" if pd.isna(s.avg_bars_held) else f"{s.avg_bars_held:.0f} 根"),
    ]


def format_stats(s: PullbackStats) -> str:
    """终端里打一份, 方便不开窗口时看结果。"""
    head = f"{panel_title(s)}   [{panel_subtitle(s)}]"
    body = "\n".join(f"  {k:<20}{v}" for k, v in panel_rows(s))
    return f"{head}\n{body}"
