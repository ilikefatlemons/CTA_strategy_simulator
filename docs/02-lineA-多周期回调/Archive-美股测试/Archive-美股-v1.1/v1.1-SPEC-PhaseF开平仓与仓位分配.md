PhaseF 开仓/平仓逻辑与仓位分配公式整理

一、方向判断 (2h bias)

ma_filter.py::ma_array_state，用 2h K线的 MA5/MA20/MA50：
- 多头：MA5 > MA20 > MA50（严格排列）→ bias = long
- 空头：MA5 < MA20 < MA50（严格排列）→ bias = short
- 其余 → neutral（neutral 时不开仓）

二、开仓触发（三级联动，pullback_entry.py + ma_filter.py）

1. 定方向：2h bias 非 neutral
2. 回调确认（pullback_occurred）：用 MA5/MA20（不含MA50）判断 30m + (5m或15m) 反向排列
  - 30m 必须反向排列（硬性条件）
  - 5m 和 15m 至少一个也反向排列（OR，不要求两个都反）
3. 入场触发（get_entry_trigger）：5m 和 15m 都重新转回与 2h 同向排列（AND，两个都要）
4. 触发后，entry_price = df_5m["open"].iloc[i+1]，即下一根5m K线的开盘价成交（不用触发那根K线自己的价格，避免未来函数）

三、止损公式（stop_loss.py）

两种候选取"离入场价更近"的一个：

- ATR止损：atr_stop = entry_price ∓ 1.5 × ATR(14)（多头减、空头加）
- 平台止损：30high/low（左右各2根K线的局部极值，swing_k=2）

chosen = min(candidates, key=|entry_price - 候选价|)   # 取更近的那个
final_stop = c            #再留 0.3% 缓冲，防止扫损

四、止盈/离场（两条腿）

腿A：固定盈亏比部分止盈
R = |entry_price - stop_loss|
trigger_price  亏比2:1，取1.5~2R区间上限）
触发后平掉 50% 仓位（partial_ratio =
0.5），并把剩 损位。

腿B：剩余50%用当前ATR吊灯止损跑趋势
raw_trail = ex × ATR(14)
trailing_stop = max(raw_trail, floor_stop)   # 多头只上移，floor_stop=原始止损，永不比原始止损更差
- extreme_pric价（多头）/最低价（空头），逐bar更新
- 价格触及 trailing_stop → 平掉剩余50%，reason =
"PROTECTIVE_SL

原始止损未部分止盈前触发 → 整批100%平仓，reason = "SL"

五、仓位分配（组合层，portfolio_pullback_backtest.py + sizing.py + vol_estimator.py）

波动率估计（rolling_daily_atr_vol，14日窗口）：
daily_ATR% = ATR(daily OHLC, 14) / daily_close
weight权重用的vol = daily_ATR%.shift(1)   # D日权重只用D-1及之前收盘算出的值，避免未来函数

逆波动率加权（InverseVolatilitySizer）：
weight_i = (1  有 vol_j > 0的ticker j
波动率越大，分配权重越小；每日按当天权重重新配比（daily
rebalance，等价于每天重新调仓）。

---
核心防未来函数设计：入场用次根K线开盘价成交、止损/止盈用触发时已知的ATR  T-1日收盘算出的波动率——全链路没有用到"当前bar收盘前才能确定"的未来信息。