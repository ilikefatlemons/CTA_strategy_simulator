# -*- coding: utf-8 -*-
"""
lineA-03 的统计面板。

版式与 `src/performance/pullback_stats.py` / `rbreaker_stats.py` 逐字对齐, 好把三个
策略的面板并排看。口径也沿用 `src/performance/trade_stats.py` 的既有约定:

  * 亏损桶是 `p <= 0` —— 持平的一笔算亏, 这样胜率和盈亏比互相自洽
  * 收益率一律是**净的**(扣掉成本), 存成普通小数, 格式化层再乘 100
  * 最大回撤是**负**分数; 面板显示时取绝对值
  * 最大回撤基于**逐根 mark-to-market** 权益曲线, 不是逐笔平仓权益

------------------------------------------------------------- 与 v3 的两处不同 --

1. **一批只有一条腿。** 本策略无部分止盈, 所以「批」与「腿」是同一个东西, 不需要
   `batch_pnls` 那样先按 `入场下标` 归约。

2. **吊灯出场必须把盈/亏分开报。** 吊灯结构上是追踪止损而不是止盈: 浮盈在 +1R 到
   +2R 之间时它生效但坐在入场价下方, 那一段被它打掉的交易**结构性地是亏损单**
   (几何见 `src/engine/lineA_03_backtest.py` 的模块 docstring)。只报一个「吊灯 220」
   会被读成 220 笔赚钱的 —— 实测 AU 上其中 40% 是亏的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.lineA_03_backtest import 止损, 吊灯, 回测结果
from src.performance.sharpe import sharpe_ratio_cn
from src.performance.trade_stats import max_drawdown, payoff, win_rate
from src.strategy.lineA_03 import 策略参数


@dataclass(frozen=True)
class 统计:
    品种: str
    时段模式: str
    可交易日: int
    笔数: int
    多: int
    空: int
    累计净收益: float
    最大回撤: float          # 负分数
    胜率: float
    盈亏比: float
    夏普: float
    固定止损笔数: int
    吊灯笔数: int
    吊灯亏损笔数: int        # 吊灯出场里净收益 <= 0 的
    跳空笔数: int
    平均持有根数: float          # 驱动周期的根数, 单位见 参数.驱动周期
    末尾未平仓: int
    暖机天数: int
    每交易日大周期根数: float   # 2h 上每个交易日几根 —— 分箱警告要用
    参数: 策略参数

    @property
    def 暖机需要(self) -> int:
        """2h 的慢线是最慢的一条。AU 每交易日 5 根 2h, 所以 55 根 = 11 个交易日。"""
        if not np.isfinite(self.每交易日大周期根数) or self.每交易日大周期根数 <= 0:
            return self.参数.慢线
        return int(np.ceil(self.参数.慢线 / self.每交易日大周期根数))

    @property
    def 暖机不足(self) -> bool:
        return self.暖机天数 < self.暖机需要


def 算统计(结果: 回测结果, 每交易日大周期根数: float = float("nan")) -> 统计:
    tr = 结果.交易
    收益 = [t.净收益 for t in tr]
    eq = 结果.权益曲线
    吊 = [t for t in tr if t.理由 == 吊灯]
    return 统计(
        品种=结果.品种,
        时段模式=结果.时段模式,
        可交易日=结果.可交易日,
        笔数=len(tr),
        多=sum(1 for t in tr if t.方向 == "多"),
        空=sum(1 for t in tr if t.方向 == "空"),
        累计净收益=float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else float("nan"),
        最大回撤=max_drawdown(eq) if len(eq) else float("nan"),
        胜率=win_rate(收益),
        盈亏比=payoff(收益),
        夏普=sharpe_ratio_cn(eq, 结果.交易日) if len(eq) else float("nan"),
        固定止损笔数=sum(1 for t in tr if t.理由 == 止损),
        吊灯笔数=len(吊),
        吊灯亏损笔数=sum(1 for t in 吊 if t.净收益 <= 0),
        跳空笔数=sum(1 for t in tr if t.跳空成交),
        平均持有根数=float(np.mean([t.持有根数 for t in tr])) if tr else float("nan"),
        末尾未平仓=结果.末尾未平仓,
        暖机天数=结果.暖机天数,
        每交易日大周期根数=每交易日大周期根数,
        参数=结果.参数,
    )


# ---------------------------------------------------------------- 格式化 ----
_每根分钟 = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "2h": 120}


def _分钟(周期: str) -> int:
    """驱动周期一根几分钟。只给「平均持有」那一行换算小时用。"""
    return _每根分钟.get(周期, 1)


def _pct(x: float, 带号: bool = True) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.2%}" if 带号 else f"{abs(x):.2%}"


def _num(x: float, fmt: str) -> str:
    return "n/a" if pd.isna(x) else format(x, fmt)


def 面板标题(s: 统计) -> str:
    return f"{s.品种} · lineA-03 多周期回调"


def 面板副标题(s: 统计) -> str:
    p = s.参数
    段 = [f"{p.大周期}定向 · {p.回调周期}回调+入场 · {p.风险周期}风险",
          f"MA{p.快线}/{p.慢线} · {p.止损ATR倍数:g}ATR止损 + {p.吊灯ATR倍数:g}ATR吊灯",
          "跨时段持有"]
    if s.暖机不足:
        # 「一笔不做」和「没暖好」在屏幕上长得一样, 必须标出来
        段.append(f"⚠ 暖机不足 {s.暖机天数}/{s.暖机需要}")
    return " · ".join(段)


def 面板脚注(s: 统计) -> list[str]:
    """
    必须跟着结果一起显示的声明。**这不是装饰** —— 每一条都会改变数字怎么读。
    """
    脚 = ["成本 0 基点 → 以上都是**毛**收益"]
    if np.isfinite(s.每交易日大周期根数):
        脚.append(
            f"{s.参数.大周期} 按交易日等分 120 个交易分钟, 每交易日 "
            f"{s.每交易日大周期根数:.1f} 根 (末根是余数, 且 01:00-09:30 那根横跨夜→日断口) "
            f"—— MA{s.参数.慢线} 实际覆盖约 {s.参数.慢线 / s.每交易日大周期根数:.1f} 个交易日"
        )
    脚.append("涨跌停未实现: 跨时段持有能穿过封板, 回测里止损照常成交 → 单向高估收益")
    return 脚


def 面板行(s: 统计) -> list[tuple[str, str]]:
    """
    面板的九行 (标签, 值)。顺序固定, 与 R-Breaker / v3 回调面板对齐好并排看。
    末尾未平仓时追加第十行。
    """
    盈 = s.吊灯笔数 - s.吊灯亏损笔数
    行 = [
        ("可交易日", f"{s.可交易日}"),
        ("交易", f"{s.笔数} (多{s.多}/空{s.空})"),
        ("累计净收益", _pct(s.累计净收益)),
        ("最大回撤", _pct(s.最大回撤, 带号=False)),
        ("胜率", "n/a" if pd.isna(s.胜率) else f"{s.胜率:.1%}"),
        ("盈亏比", _num(s.盈亏比, ".2f")),
        ("策略夏普比", _num(s.夏普, "+.2f")),
        # 吊灯**不是止盈**: 盈/亏必须分开, 否则「吊灯 220」会被读成 220 笔赚钱的
        ("离场 固定止损/吊灯", f"{s.固定止损笔数} / {s.吊灯笔数}（盈{盈}·亏{s.吊灯亏损笔数}）"),
        ("平均持有", "n/a" if pd.isna(s.平均持有根数)
         else f"{s.平均持有根数:.0f} 根 {s.参数.驱动周期} ≈ "
              f"{s.平均持有根数 * _分钟(s.参数.驱动周期) / 60:.1f} 小时"),
    ]
    if s.末尾未平仓:
        行.append(("末尾未平仓", f"{s.末尾未平仓} 批（**未计入统计**）"))
    return 行


def 终端版(s: 统计) -> str:
    """跑脚本、不开窗口时看的那一份。"""
    头 = f"{面板标题(s)}   [{面板副标题(s)}]"
    体 = "\n".join(f"  {k:<20}{v}" for k, v in 面板行(s))
    脚 = "\n".join(f"  ⚠ {x}" for x in 面板脚注(s))
    return f"{头}\n{体}\n{脚}"
