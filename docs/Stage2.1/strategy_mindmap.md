# 回调策略 Pullback Strategy v1.1 + Stage 2.1

## 进场 ENTRY
- 四周期级联 4-TF cascade (2h → 30m → 15m → 5m)
- Step 0 · 冷静期门 Cooldown gate
  - active → 不进场 no entry
- Step 1 · 2h 定方向 Bias (`get_trend_bias`)
  - `ma_array_state` 严格三线 strict MA5/20/50
  - 多头 long: MA5>20>50
  - 空头 short: MA5<20<50
  - 其余 neutral → 不交易
  - 用已收盘 2h K线 closed 2h bars
- Step 2 · 回调确认 Pullback (`pullback_occurred`)
  - 硬性 required: 30m 反向 (MA5/20 opposite)
  - 且 AND: 5m 或 15m 至少一个反向 (OR — 谁先反向不固定)
  - `ma_fast_mid_state` 两线 MA5/20 (小周期用两线, 三线太严)
- Step 3 · 触发入场 Trigger (`get_entry_trigger`)
  - 5m 且 15m 都翻回与 2h 同向
- 状态机 State machine (`PullbackEntryEngine`)
  - 只跟踪 `_bias` + `_pullback_seen`
  - bias 变化(含 neutral) → 作废回调, 回 Step 1
  - 首仓/回补同一 `on_bar` 路径 (调用方打 open/reentry 标签)
- 仓位 Position
  - 同时只允许 1 个 `_OpenBatch` / ticker
- 成交 Fill
  - 下一根开盘价 next-bar open (见 Stage 2.1 进场重构)

## 出场 EXIT
- 两条腿 Two legs (非"一把梭全平")
  - Leg A (50%): 固定盈亏比部分止盈 fixed 2R partial TP
  - Leg B (50%): 吊灯止损跑趋势 chandelier trailing
- 止损 Stop-loss (`stop_loss.py`)
  - min(1.5×ATR, 最近 30m swing high/low 平台)
  - + 0.3% offset 逆方向 (防程序化扫损)
- 止盈 Take-profit (`take_profit.py`)
  - Leg A: 2R 部分止盈 (rr_trigger=2.0, 50%)
  - Leg B: 3×ATR 吊灯 chandelier
    - 只上移不下移 favorable-only
    - 永不差于原止损 floor_stop
  - 吊灯 ATR 用 bar i-1 (非 i) — 无未来函数
- 冷静期 Cooldown (`cooldown.py`) — 只挡新单, 不强平
  - 触发1: 同向连续 3 次纯止损 (护盈止损不计)
  - 触发2: 单根 2h 实体吞没 ≥2 条均线
  - 解除: 连续 3 根 2h 同一干净排列

## 权重分配 WEIGHT ASSIGNMENT
- 组合层 Portfolio (`portfolio_pullback_backtest.py`)
  - 各 ticker 独立引擎, 无跨标的耦合
  - 只决定每个 ticker 当日 %收益计入多少
- 逆波动率定权 InverseVolatilitySizer (`sizing.py`)
  - 权重 ∝ 1/vol, 归一化 renormalize
  - 剔除 0 波动 (否则 1/vol 爆掉)
- 波动率估计 rolling_daily_atr_vol (`vol_estimator.py`)
  - ATR/close (%-of-close, 跨标的可比)
  - `.shift(1)` — 当日 D 只用 ≤ D-1 收盘 (无未来)
  - 窗口 14 天
- 每日再平衡 daily rebalance (非 buy-and-hold)
- 空仓 ticker 仍得全额权重 → 当日 0 收益 (设计如此, 非 bug)
- 贡献分解 contribution: 精确日度美元分解 (weight × return × equity_before)

## Stage 2.1 修正 CHANGES  ← 本次改动
- 全部开关在 `src/config.py`; 默认 ON = 诚实; OFF 复现旧行为
- 进场重构 Entry refactor (`entry_on_completed_bar`)
  - 决策只用已收盘K线 ≤ i-1, 成交在 bar i 开盘
  - 旧: 用 close[i], 成交 i+1 开盘
  - 两者都无未来函数; Δ 很小 (只重排 HTF 新鲜度)
- ★ 3.5 跳空逻辑 Gap-fill (`gap_fill_exits`)  ← 主导 dominant
  - 止损/吊灯成交取"更差侧" worse-of(line, bar open)
    - 多头 long: min(stop, open)
    - 空头 short: max(stop, open)
  - 隔夜跳空穿损 overnight gap → 按(更差的)开盘价成交
  - 止盈保持 limit @ trigger — 不给跳空便宜
    - "止损吃亏, 止盈不占便宜" 保守不对称 (留 live buffer)
  - 证据: 26% 的止损单开盘已跳过 stop
  - Δ: return −20.42pp, Sharpe −3.60 (≈65% 收益是它)
- ★ 3.1 区间触发 Range-based trigger (`range_based_exit_trigger`)
  - 止损/止盈/吊灯用 bar high/low 判定(盘内), 不只用 close
  - 抓回 close-only 漏掉的盘内触损
  - 证据: 6% 批次触损后被 close-only 放跑
  - Δ: return −7.98pp, Sharpe −1.21
- ★ 3.1 双触优先 Both-hit ordering (`pessimistic_both_hit`)
  - 一根K内同时触损+触盈 (仅区间触发下可能)
  - 悲观 pessimistic: 先算止损 stop-first
  - 证据: 仅 0.4% 批次; Δ −0.19pp (可忽略, 但严格逻辑仍修)
- 其他项已核查 = 无变化 verified NO-CHANGE
  - Item 1 时间戳 timestamp: open-time 一致守卫
  - Item 2 同根收盘成交 same-bar fill: 已是次开盘
  - Item 4 确认K泄漏 confirmation leak: 信号只读 ≤ i-1
  - Item 5 环境过滤泄漏 context filter: 全因果 (swing-pivot 居中但受限 → watch)
- 结果 RESULT
  - +31.40% / Sharpe 5.25  →  +3.00% / Sharpe 0.44
  - "edge" ~90% 来自出场成交模型 (exit-fill artifact)
  - maxDD −0.81% → −1.95% (残余平滑 = 组合稀释, 归 sizing 另议)
- 测量脚手架 (非fix, 默认关) measurement only
  - `entry_lag_bars` (滞后诊断), `batch_audit` (审计钩子)

## 数据与守卫 DATA & GUARDS  (背景)
- 数据源 Alpaca IEX 5m, `Adjustment.ALL` (复权 split+div)
- 时间戳 = K线开盘时间 open-time UTC; RTH 09:30–16:00 ET
- 重采样 resample 5m → 15m/30m/2h, 按 session 锚定 09:30
- `closed_bar_positions` 守卫: t+rule ≤ open_time (绝不用未收盘K)
- 商品 ticker USO/GLD/SLV = ETF 代理 (Alpaca 无期货数据)
