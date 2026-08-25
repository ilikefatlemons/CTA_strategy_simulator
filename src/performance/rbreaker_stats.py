# -*- coding: utf-8 -*-
"""
R-Breaker 突破策略的结果统计 + 图上那个中文面板的行内容。

统计口径沿用仓库既有约定 (src/performance/trade_stats.py):
  * 亏损桶是 `p <= 0` —— 持平的一笔算亏, 这样胜率和盈亏比互相自洽。
  * 收益率一律是**净的**(扣掉 cost_bps), 存成普通小数, 格式化层再乘 100。
  * 最大回撤是**负**分数; 面板显示时取绝对值。
  * 最大回撤基于**逐根 mark-to-market** 权益曲线, 不是逐笔平仓权益 ——
    后者看不见笔内回撤。

面板行序是定死的(和参考截图一致), 但**去掉了「平均 R」和「止损宽度」,
加上了「策略夏普比」**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.rbreaker_backtest import SESSION_END, TRAIL, RBreakerResult
from src.performance.sharpe import sharpe_ratio_cn
from src.performance.trade_stats import max_drawdown, payoff, win_rate


@dataclass(frozen=True)
class RBreakerStats:
    symbol: str
    # 策略名, 进面板标题: "{symbol} · R-Breaker {label}"。突破和反转是两个
    # 独立策略但共用这一套统计, 靠这个字段区分。
    label: str
    # 时段划分模式的中文名, 进副标题。空串表示不显示(单元测试等场合)。
    mode_label: str
    tradable_days: int
    n_trades: int
    n_long: int
    n_short: int
    total_return: float       # 普通小数
    max_drawdown: float       # 负分数
    win_rate: float
    payoff: float
    sharpe: float
    n_exit_trail: int
    n_exit_session: int
    avg_bars_held: float
    # 六线系数, 带进面板副标题 —— R-Breaker 没有权威公式版本, 任何结果都
    # 应该声明用的是哪一组系数。见 v3.0-RSCH-R-Breaker与Turtle运行逻辑.md 第十节。
    f1: float = 0.35
    f2: float = 0.07
    f3: float = 0.25
    d: float = 3.0


def compute_stats(result: RBreakerResult, label: str = "突破",
                  mode_label: str = "") -> RBreakerStats:
    """
    `label` 进面板标题(突破 / 反转); `mode_label` 进副标题(时段划分模式)。
    """
    tr = result.trades
    pnls = [t.pnl_net for t in tr]
    eq = result.equity_curve
    p = result.params
    return RBreakerStats(
        symbol=result.symbol,
        label=label,
        mode_label=mode_label,
        tradable_days=result.tradable_days,
        n_trades=len(tr),
        n_long=sum(1 for t in tr if t.direction == "long"),
        n_short=sum(1 for t in tr if t.direction == "short"),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else float("nan"),
        max_drawdown=max_drawdown(eq),
        win_rate=win_rate(pnls),
        payoff=payoff(pnls),
        sharpe=sharpe_ratio_cn(eq, result.trading_date),
        n_exit_trail=sum(1 for t in tr if t.reason == TRAIL),
        n_exit_session=sum(1 for t in tr if t.reason == SESSION_END),
        avg_bars_held=float(np.mean([t.bars_held for t in tr])) if tr else float("nan"),
        f1=p.f1, f2=p.f2, f3=p.f3, d=p.d,
    )


# ---------------------------------------------------------------- 格式化 ----
def _pct(x: float, signed: bool = True) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.2%}" if signed else f"{abs(x):.2%}"


def _num(x: float, fmt: str) -> str:
    return "n/a" if pd.isna(x) else format(x, fmt)


def panel_title(stats: RBreakerStats) -> str:
    return f"{stats.symbol} · R-Breaker {stats.label}"


def panel_subtitle(stats: RBreakerStats) -> str:
    coeff = f"f1={stats.f1:g} f2={stats.f2:g} f3={stats.f3:g} d={stats.d:g}"
    return f"{coeff} · 时段 {stats.mode_label}" if stats.mode_label else coeff


def panel_rows(stats: RBreakerStats) -> list[tuple[str, str]]:
    """面板的九行 (标签, 值)。顺序固定。"""
    s = stats
    return [
        ("可交易日", f"{s.tradable_days}"),
        ("成交", f"{s.n_trades} (多{s.n_long}/空{s.n_short})"),
        ("累计净收益", _pct(s.total_return)),
        ("最大回撤", _pct(s.max_drawdown, signed=False)),
        ("胜率", "n/a" if pd.isna(s.win_rate) else f"{s.win_rate:.1%}"),
        ("盈亏比", _num(s.payoff, ".2f")),
        ("策略夏普比", _num(s.sharpe, "+.2f")),
        ("离场 止损/收盘", f"{s.n_exit_trail} / {s.n_exit_session}"),
        ("平均持有", "n/a" if pd.isna(s.avg_bars_held) else f"{s.avg_bars_held:.0f} 根"),
    ]


def format_stats(stats: RBreakerStats) -> str:
    """终端里打一份, 方便不开窗口时看结果。"""
    head = f"{panel_title(stats)}   [{panel_subtitle(stats)}]"
    body = "\n".join(f"  {k:<14}{v}" for k, v in panel_rows(stats))
    return f"{head}\n{body}"
