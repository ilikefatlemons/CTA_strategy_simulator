# -*- coding: utf-8 -*-
"""
R-Breaker 突破策略的单品种逐根回测。

规则(全部来自 docs/03-lineB-R-Breaker/B1-拆分成两个策略+时段/v3.0-PLAN-R-Breaker突破实施方案.md):

  入场   bar i-1 的价格穿过 Bbreak -> 做多 / 穿过 Sbreak -> 做空,
         **成交在 bar i 的 open**。每个交易时段最多一笔。
  出场   只有追踪止损 `Senter + (peak - Ssetup)/d`, 外加时段结束强制平仓。
         没有止盈。
  纪律   一根 bar 只做一件事(入场 或 出场, 不可兼得)。

---------------------------------------------------------------------------
无未来函数, 逐行可审计
---------------------------------------------------------------------------
    步骤            读取的数据                              触及的最新 bar
    止损位          SE[D]/SS[D](来自 D-1 日 OHLC) + peak    i-1
    止损触发判定    low[i] / high[i]                        i  (事件本身)
    跳空补价        open[i]                                 i  (成交那一刻已知)
    收盘平仓        is_session_last[i], close[i]            i  (形状静态; close 是成交价)
    极值折入        high[i]/low[i] —— **在测试之后**        i, 但只影响 i+1
    入场信号        high[i-1]/low[i-1], BB[D]/SBK[D]        i-1
    入场成交        open[i]                                 i  (开盘价)

唯一结构上危险的是极值折入, 已经放在出场测试之后并带 `pos != 0` 守卫 ——
与 `pullback_backtest.py:296-300` 同形。

`is_session_last[i]` **不是**未来函数: 收盘时间是交易所提前公告的, 实盘同样
提前知道。它是窗口形状的属性, 循环开始前就已确定, 不依赖任何价格。

tests/test_rbreaker_backtest.py::test_no_lookahead_by_truncation 用截断法把
这条不变量钉死: 把 bar i 之后的数据全部改成垃圾重跑, 所有 exit_bar_idx < i
的交易必须逐字节相同。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.v3_sessions import Prepared
from src.strategy.rbreaker import RBreakerConfig, RBreakerParams, ReversalConfig

TRAIL = "TRAIL"
SESSION_END = "SESSION_END"


@dataclass
class RBTrade:
    """一笔完整的交易(开 -> 平)。"""

    direction: str            # "long" | "short"
    entry_bar_idx: int
    entry_time: pd.Timestamp
    entry_price: float        # = open[entry_bar_idx], 实际成交价
    exit_bar_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    reason: str               # TRAIL | SESSION_END
    trading_date: pd.Timestamp
    session_id: int
    # 当日的 Bbreak(多) / Sbreak(空)。追踪止损的几何基准 —— MFE 从这里量,
    # 不从实际成交价量。成交价与它的差额全额落进盈亏、零传导到止损位。
    theo_entry: float
    extreme: float            # 出场时的 peak(多) / trough(空)
    stop_at_exit: float       # 出场那一根被测试的止损位
    gapped: bool              # True = 按 bar 的 open 成交, 不是按止损线
    cost_bps: float

    @property
    def pnl_gross(self) -> float:
        move = (self.exit_price - self.entry_price) if self.direction == "long" \
            else (self.entry_price - self.exit_price)
        return move / self.entry_price

    @property
    def pnl_net(self) -> float:
        """扣掉往返成本后的收益。**统计口径一律用这个。**"""
        return self.pnl_gross - self.cost_bps / 1e4

    @property
    def bars_held(self) -> int:
        return self.exit_bar_idx - self.entry_bar_idx


@dataclass
class RBreakerResult:
    symbol: str
    trades: list[RBTrade]
    equity_curve: pd.Series        # 逐根 mark-to-market, index = bar datetime
    trading_date: pd.Series        # 与 equity_curve 同 index, 供夏普按交易日分组
    trail_stop: np.ndarray         # 逐根: 该根被测试的止损位; 空仓为 NaN
    position: np.ndarray           # 逐根 int8: -1 / 0 / +1
    lines: pd.DataFrame            # index=trading_date, 六线 + valid
    tradable_days: int
    params: RBreakerParams
    # 反转引擎 (`rbreaker_reversal_backtest.py`) 复用这个容器, 它带的是
    # ReversalConfig —— 两个策略的结果形状完全一样, 统计层和图表层因此不用分叉。
    config: RBreakerConfig | ReversalConfig
    dropped_runs: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def in_position(self) -> np.ndarray:
        """止损线该画的那些根。"""
        return ~np.isnan(self.trail_stop)


def run_rbreaker_backtest(
    prepared: Prepared,
    params: RBreakerParams | None = None,
    config: RBreakerConfig | None = None,
) -> RBreakerResult:
    """
    逐根跑一遍。所有数据先取成 numpy 数组 —— 实测 130 万根用 numpy 标量循环
    0.34s, 用 `df.iloc[i]` 逐行取 10.4s (30 倍差距), 而切品种时这个差别是
    "瞬时"和"卡住"的区别。
    """
    p = params or prepared.params
    cfg = config or prepared.config
    df = prepared.df
    n = len(df)

    equity = np.ones(n, dtype=np.float64)
    trail_arr = np.full(n, np.nan, dtype=np.float64)
    pos_arr = np.zeros(n, dtype=np.int8)
    trades: list[RBTrade] = []

    if n == 0:
        return RBreakerResult(
            symbol=prepared.symbol, trades=trades,
            equity_curve=pd.Series(dtype=float), trading_date=pd.Series(dtype="datetime64[ns]"),
            trail_stop=trail_arr, position=pos_arr, lines=prepared.lines,
            tradable_days=prepared.tradable_days, params=p, config=cfg,
            dropped_runs=prepared.dropped_runs,
        )

    op = df["open"].to_numpy(np.float64)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    sid = df["session_id"].to_numpy(np.int64)
    first_of_sess = df["is_session_first"].to_numpy(bool)
    last_of_sess = df["is_session_last"].to_numpy(bool)
    # 时段模式决定的可交易性 (如「只做夜盘」下白盘段整段为 False)。时段内恒定。
    tradable = df["tradable"].to_numpy(bool)
    times = df.index.to_numpy()
    tds = df["trading_date"].to_numpy()

    ln = prepared.lines
    BB = ln["Bbreak"].to_numpy(np.float64)
    SBK = ln["Sbreak"].to_numpy(np.float64)
    SS = ln["Ssetup"].to_numpy(np.float64)
    BS = ln["Bsetup"].to_numpy(np.float64)
    SE = ln["Senter"].to_numpy(np.float64)
    BE = ln["Benter"].to_numpy(np.float64)
    valid = ln["valid"].to_numpy(bool)
    # **按时段钉住**的线下标, 不是逐根的 trading_date 下标 —— 见 Prepared.line_idx。
    # 对除 day_night 之外的四种模式两者相同; day_night 下整个时段共用白盘那天的线,
    # 所以持仓中的止损位不会因为跨了 trading_date 而突然平移。
    day = prepared.line_idx

    d = p.d
    cost = cfg.cost_bps / 1e4
    gap_fill = cfg.gap_fill_exits
    range_trigger = cfg.entry_trigger_on_range
    one_per_session = cfg.one_trade_per_session

    capital = 1.0
    pos = 0
    entry_i = -1
    entry_px = np.nan
    entry_t = None
    theo = np.nan
    peak = np.nan          # 多头是运行最大值, 空头是运行最小值
    used_sid = -1          # 本时段已用掉的那一笔

    for i in range(n):
        D = day[i]
        S = sid[i]
        acted = False      # 一根 bar 只做一件事

        # ---------------------------------------------------------- 出场 ----
        # `entry_i < i`: 成交那一根不测试止损。成交和止损是两个动作, 一根
        # bar 只允许一个 —— 这也保证了每笔 exit_bar_idx > entry_bar_idx。
        if pos != 0 and entry_i < i:
            is_long = pos == 1
            if is_long:
                stop = SE[D] + (peak - SS[D]) / d
                hit = lo[i] <= stop
            else:
                stop = BE[D] - (BS[D] - peak) / d
                hit = hi[i] >= stop
            trail_arr[i] = stop

            px, reason = np.nan, None
            if hit:
                # 跳空: 按 open 和止损线里更差的那个成交, 而不是按市场根本
                # 没成交过的止损线。10:15->10:31 / 11:30->13:31 以及每个时段
                # 边界都是零根 bar, 那里的跳空是真实的。
                px = (min(stop, op[i]) if is_long else max(stop, op[i])) if gap_fill else stop
                reason = TRAIL
            elif last_of_sess[i]:
                # 同一根既触止损又是时段末根时走上面那支: 止损机械上更早发生,
                # 价格通常也更差。对应 v2.1 的 pessimistic_both_hit。
                px, reason = cl[i], SESSION_END
            else:
                reason = None

            if reason is not None:
                trades.append(RBTrade(
                    direction="long" if is_long else "short",
                    entry_bar_idx=entry_i, entry_time=entry_t, entry_price=entry_px,
                    exit_bar_idx=i, exit_time=pd.Timestamp(times[i]), exit_price=float(px),
                    reason=reason, trading_date=pd.Timestamp(tds[i]), session_id=int(S),
                    theo_entry=float(theo), extreme=float(peak), stop_at_exit=float(stop),
                    # bool() 而不是裸表达式: px/stop 是 numpy 标量, 比较结果是
                    # np.bool_, 会让 `t.gapped is True` 这种断言意外失败。
                    gapped=bool(reason == TRAIL and px != stop), cost_bps=cfg.cost_bps,
                ))
                capital *= 1.0 + trades[-1].pnl_net
                pos, entry_i, entry_px, entry_t = 0, -1, np.nan, None
                theo = peak = np.nan
                acted = True

        # -------------------------------------------- 折入本根极值(测试之后) ----
        # 放在出场测试之后, 所以"测试 bar i 时用的止损位"永远只依赖到 i-1 的
        # 极值。带 pos != 0 守卫: 上面可能刚把仓位平掉。
        # 与 pullback_backtest.py:296-300 同形。
        if pos != 0:
            peak = max(peak, hi[i]) if pos == 1 else min(peak, lo[i])

        # ---------------------------------------------------------- 入场 ----
        #   not acted          —— 出场那一根不再入场
        #   valid[D]           —— 当日必须有合法的前日参照
        #   S != used_sid      —— 每个时段最多一笔
        #   not last_of_sess   —— 会话末根不入场, 否则这根既入场又被强平
        #   not first_of_sess  —— bar i-1 属于上一个时段(可能隔 6~58 小时,
        #                         而且可能是另一个 trading_date 的另一套线)
        if (pos == 0 and not acted and valid[D] and tradable[i]
                and not (one_per_session and S == used_sid)
                and not last_of_sess[i] and not first_of_sess[i]):
            j = i - 1
            if range_trigger:
                up, dn = hi[j] > BB[D], lo[j] < SBK[D]
            else:
                up, dn = cl[j] > BB[D], cl[j] < SBK[D]
            # up 和 dn 不可能同时为真: Bbreak - Sbreak = (1+2*f3)(Ssetup-Bsetup) > 0
            if up or dn:
                pos = 1 if up else -1
                entry_i, entry_px, entry_t = i, float(op[i]), pd.Timestamp(times[i])
                theo = BB[D] if up else SBK[D]
                # peak 种在**理论线**上而不是成交价上 —— 这是 MFE_geo 的定义。
                peak = theo
                trail_arr[i] = (SE[D] + (theo - SS[D]) / d) if up else (BE[D] - (BS[D] - theo) / d)
                used_sid = S
                # 本根的 high/low 立刻折进 peak, 所以 bar i+1 的止损位由
                # max(theo, high[i]) 决定 —— 与上面那段折入逻辑一致。
                peak = max(peak, hi[i]) if pos == 1 else min(peak, lo[i])

        # ------------------------------------------ 逐根 mark-to-market ----
        pos_arr[i] = pos
        if pos != 0:
            move = (cl[i] - entry_px) if pos == 1 else (entry_px - cl[i])
            equity[i] = capital * (1.0 + move / entry_px - cost)
        else:
            equity[i] = capital

    # 窗口末根的 is_session_last 恒为 True(shift(-1) 出 NaN), 且会话末根不许
    # 入场 —— 所以循环结束时不可能还留着仓位。
    assert pos == 0, "回测结束仍有未平仓位 —— is_session_last 的构造被破坏了"

    idx = pd.DatetimeIndex(times, name="datetime")
    return RBreakerResult(
        symbol=prepared.symbol, trades=trades,
        equity_curve=pd.Series(equity, index=idx),
        trading_date=pd.Series(tds, index=idx),
        trail_stop=trail_arr, position=pos_arr, lines=prepared.lines,
        tradable_days=prepared.tradable_days, params=p, config=cfg,
        dropped_runs=prepared.dropped_runs,
    )
