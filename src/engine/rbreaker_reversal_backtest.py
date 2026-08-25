# -*- coding: utf-8 -*-
"""
R-Breaker **反转**策略的单品种逐根回测。与突破半边是两个独立的策略。

    武装空  本时段运行最高价 >= Ssetup            —— 「观察卖」
    触发空  已武装 且 low[i-1] < Senter           —— 「反转卖」-> 成交在 open[i]
    武装多  本时段运行最低价 <= Bsetup            —— 「观察买」
    触发多  已武装 且 high[i-1] > Benter          —— 「反转买」-> 成交在 open[i]

    出场    只有追踪止损 + 时段结束强制平仓, 无止盈
    纪律    一根 bar 只做一件事; 每个交易时段最多一笔

「观察价的作用是武装(arming)」是 docs/03-lineB-R-Breaker/B0-原理与出处/v3.0-RSCH-R-Breaker与Turtle运行逻辑.md
第 38 行的原话: *它把一个静态价格变成"必须先走过一段路才生效"的条件, 用于
排除开盘即贴着该价位来回震荡的情形。*

---------------------------------------------------------------------------
版本声明 (重要)
---------------------------------------------------------------------------
该研究笔记第 12、33 行两次写明**反转线只在「持仓时」生效** —— 在它引用的那一
版重构里, 反转是"把突破半边开出的仓位反手平掉"的事件, **不能从空仓直接入场**。

本模块允许**从空仓武装并入场**, 超出了那一版重构。之所以合法, 是因为同文第
218 行已声明 R-Breaker 没有作者本人发表的规范文档、流传的版本彼此有出入。
**任何对外引用本策略的回测结果都必须一并声明这一条。**

---------------------------------------------------------------------------
与突破引擎的关系
---------------------------------------------------------------------------
出场块 / MFE 折入 / MTM / 时段守卫与 `rbreaker_backtest.py` **逐行同构** ——
基线执行纪律(跳空补价、一根一动作、时段强平、极值折入的位置)必须完全一致,
否则两个策略的对比就不干净了。**动了任何一边的基线, 另一边要同步。**

刻意没有把两者合并成一个带 mode 参数的引擎: 入场状态机真的不同(武装闩锁 +
运行极值 + 双向候选), 塞进已经测通的突破循环有回归风险。代价是约 60 行骨架
重复, 由 tests/test_rbreaker_smoke.py 对两个策略跑同一套结构断言来补偿。

---------------------------------------------------------------------------
无未来函数, 逐行可审计
---------------------------------------------------------------------------
    步骤            读取的数据                              触及的最新 bar
    止损位          BB[D]/SS[D](来自 D-1 日 OHLC) + ext     i-1
    止损触发判定    high[i] / low[i]                        i  (事件本身)
    跳空补价        open[i]                                 i  (成交那一刻已知)
    收盘平仓        is_session_last[i], close[i]            i  (形状静态)
    MFE 折入        high[i]/low[i] —— **在测试之后**        i, 只影响 i+1
    **武装判定**    run_hi/run_lo —— **只到 i-1**           i-1
    入场触发        low[i-1]/high[i-1], SE[D]/BE[D]         i-1
    入场成交        open[i]                                 i  (开盘价)

本策略比突破多一处未来函数风险点: **运行极值必须在入场块之后才折入本根**
(见循环里的第 5 步), 否则 bar i 的 high 会参与决定 bar i 自己的武装状态。
与第 2 步同形, 用同一套截断测试钉住。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.v3_sessions import Prepared
from src.engine.rbreaker_backtest import SESSION_END, TRAIL, RBTrade, RBreakerResult
from src.strategy.rbreaker import RBreakerParams, ReversalConfig


def run_rbreaker_reversal_backtest(
    prepared: Prepared,
    params: RBreakerParams | None = None,
    config: ReversalConfig | None = None,
) -> RBreakerResult:
    """逐根跑一遍反转策略。所有数据先取成 numpy 数组, 理由同突破引擎。"""
    p = params or prepared.params
    cfg = config or ReversalConfig()
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
    # **按时段钉住**的线下标 —— 见 Prepared.line_idx 与突破引擎的同一处注释。
    day = prepared.line_idx

    d = p.d
    cost = cfg.cost_bps / 1e4
    gap_fill = cfg.gap_fill_exits
    range_trigger = cfg.entry_trigger_on_range
    one_per_session = cfg.one_trade_per_session
    arm_first = cfg.arm_must_precede_trigger

    capital = 1.0
    pos = 0
    entry_i = -1
    entry_px = np.nan
    entry_t = None
    theo = np.nan
    ext = np.nan            # MFE_geo 本身(非负), 不是极值价
    used_sid = -1

    # 武装状态与运行极值 —— 决策 1: **按时段重置**
    cur_sid = -1
    armed_s = armed_l = False      # 各方向的阈值是否已被越过(每时段各至多一次)
    armed_dir = 0                  # 当前**唯一**生效的方向: -1 空 / +1 多 / 0 无
    run_hi = -np.inf
    run_lo = np.inf
    ambiguous_arm = 0              # 同一根同时武装两个方向(需单根跨越 1.35R)

    for i in range(n):
        D = day[i]
        S = sid[i]
        acted = False

        # 换时段 -> 武装与运行极值全部清空。夜盘碰到的高点不能武装白盘。
        if S != cur_sid:
            cur_sid = S
            armed_s = armed_l = False
            armed_dir = 0
            run_hi = -np.inf
            run_lo = np.inf

        # ---------------------------------------------------------- 出场 ----
        # `entry_i < i`: 成交那一根不测试止损 —— 一根 bar 只做一件事。
        if pos != 0 and entry_i < i:
            if pos == -1:
                stop = BB[D] - ((BB[D] - SS[D]) + ext) / d
                hit = hi[i] >= stop
            else:
                stop = SBK[D] + ((BS[D] - SBK[D]) + ext) / d
                hit = lo[i] <= stop
            trail_arr[i] = stop

            px, reason = np.nan, None
            if hit:
                px = (max(stop, op[i]) if pos == -1 else min(stop, op[i])) if gap_fill else stop
                reason = TRAIL
            elif last_of_sess[i]:
                px, reason = cl[i], SESSION_END

            if reason is not None:
                extreme = (theo - ext) if pos == -1 else (theo + ext)
                trades.append(RBTrade(
                    direction="short" if pos == -1 else "long",
                    entry_bar_idx=entry_i, entry_time=entry_t, entry_price=entry_px,
                    exit_bar_idx=i, exit_time=pd.Timestamp(times[i]), exit_price=float(px),
                    reason=reason, trading_date=pd.Timestamp(tds[i]), session_id=int(S),
                    theo_entry=float(theo), extreme=float(extreme), stop_at_exit=float(stop),
                    gapped=bool(reason == TRAIL and px != stop), cost_bps=cfg.cost_bps,
                ))
                capital *= 1.0 + trades[-1].pnl_net
                pos, entry_i, entry_px, entry_t = 0, -1, np.nan, None
                theo = ext = np.nan
                acted = True

        # ------------------------------------ 折入本根 MFE (测试之后) ----
        if pos != 0:
            ext = max(ext, SE[D] - lo[i]) if pos == -1 else max(ext, hi[i] - BE[D])

        # ---------------------------------------------------------- 武装 ----
        # run_hi/run_lo 此刻只覆盖到 bar i-1 (本根在第 5 步才折入)。
        # `armed_dir_before` 是"到 i-2 为止"的武装方向 —— 上一根做这一步时用的
        # run_hi 正好只覆盖到 i-2。arm_must_precede_trigger 就靠它实现。
        armed_dir_before = armed_dir
        if valid[D]:
            # run_hi 只增、run_lo 只减, 所以每个阈值每时段至多被越过一次 ——
            # 下面两个 new_* 各自至多为真一次。
            new_s = (not armed_s) and run_hi >= SS[D]
            new_l = (not armed_l) and run_lo <= BS[D]
            armed_s = armed_s or new_s
            armed_l = armed_l or new_l
            # **最近发生的那个极值胜出。** 你要 fade 的是最新的那一段推动:
            # 刚冲到 Ssetup -> 等回落跌破 Senter 做空; 刚砸到 Bsetup -> 等回升
            # 突破 Benter 做多。两个方向因此永远不会同时生效。
            #
            # 这一条是必需的, 不是锦上添花: 空头触发条件是 low < Senter, 多头是
            # high > Benter, 而 Senter > Benter —— 一根**整根落在 [Benter,
            # Senter] 带内**的 bar 同时满足两者。这不需要跨越 0.605R, 相反是
            # 价格待在当日中段时的常态。实测 RB 一年内有 606 根这样的 bar。
            # 若不定方向而是跳过, 那些时段会一笔都不做。
            if new_s and new_l:
                # 单根同时越过 Ssetup 和 Bsetup, 即单根跨越 1.35 倍前日振幅。
                # 真发生了说明该根极不正常, 本时段不定方向。
                ambiguous_arm += 1
                armed_dir = 0
            elif new_s:
                armed_dir = -1
            elif new_l:
                armed_dir = 1
        use_dir = armed_dir_before if arm_first else armed_dir

        # ---------------------------------------------------------- 入场 ----
        if (pos == 0 and not acted and valid[D] and tradable[i]
                and not (one_per_session and S == used_sid)
                and not last_of_sess[i] and not first_of_sess[i]):
            j = i - 1
            # use_dir 至多是 -1 或 +1 之一, 所以 dn / up 互斥是构造性的。
            if range_trigger:
                dn = use_dir == -1 and lo[j] < SE[D]
                up = use_dir == 1 and hi[j] > BE[D]
            else:
                dn = use_dir == -1 and cl[j] < SE[D]
                up = use_dir == 1 and cl[j] > BE[D]

            if dn or up:
                pos = -1 if dn else 1
                entry_i, entry_px, entry_t = i, float(op[i]), pd.Timestamp(times[i])
                theo = SE[D] if dn else BE[D]
                ext = 0.0                      # MFE_geo 从理论线量起, 入场时为 0
                trail_arr[i] = (BB[D] - (BB[D] - SS[D]) / d) if dn \
                    else (SBK[D] + (BS[D] - SBK[D]) / d)
                used_sid = S
                # 本根的 low/high 立刻折进 MFE, 所以 bar i+1 的止损位已经含它。
                ext = max(ext, SE[D] - lo[i]) if dn else max(ext, hi[i] - BE[D])

        # ------------------------ 运行极值折入本根 (供下一根的武装判定) ----
        # **必须排在入场块之后** —— 否则 bar i 的 high 会决定 bar i 自己的武装。
        run_hi = max(run_hi, hi[i])
        run_lo = min(run_lo, lo[i])

        # ------------------------------------------ 逐根 mark-to-market ----
        pos_arr[i] = pos
        if pos != 0:
            move = (cl[i] - entry_px) if pos == 1 else (entry_px - cl[i])
            equity[i] = capital * (1.0 + move / entry_px - cost)
        else:
            equity[i] = capital

    assert pos == 0, "回测结束仍有未平仓位 —— is_session_last 的构造被破坏了"
    if ambiguous_arm:
        print(f"[reversal] {prepared.symbol}: {ambiguous_arm} 根 bar 单根同时武装多空"
              f"(跨越 1.35 倍前日振幅), 该时段不定方向")

    idx = pd.DatetimeIndex(times, name="datetime")
    return RBreakerResult(
        symbol=prepared.symbol, trades=trades,
        equity_curve=pd.Series(equity, index=idx),
        trading_date=pd.Series(tds, index=idx),
        trail_stop=trail_arr, position=pos_arr, lines=prepared.lines,
        tradable_days=prepared.tradable_days, params=p, config=cfg,
        dropped_runs=prepared.dropped_runs,
    )
