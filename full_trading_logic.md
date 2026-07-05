Phase 2:

1. Entry fill: if we're FLAT and a signal was queued on the previous bar, open a position at bar i's open price. Side = LONG or SHORT per the queued signal. State → ENTERED.
2. Cooldown tick: if in COOLDOWN, increment bars-since-exit; if CooldownReentry.can_reenter() (≥5 bars) returns true, state → FLAT.
3. Exit check: if ENTERED and held ≥ min_holding_bars (2), ask ATRTakeProfitStopLoss.should_exit():
  - ATR(14) is computed once, frozen at the entry bar.
  - Exit if unrealized move ≥ +2×ATR (take-profit) or ≤ −1×ATR (stop-loss), measured against bar i's close.
  - On exit: realize return_pct = pnl/entry_price, compound into capital, state → COOLDOWN.
4. Entry signal generation: if FLAT, ask MACDGoldenCross.on_bar() — MACD(12,26,9) computed on closes up to and including bar i; a zero-crossing of (MACD − signal) between bar i-1 and i produces LONG (cross up) or SHORT (cross down). Signal is queued, not filled yet, so it fills at i+1's open — no lookahead.
5. Equity mark: capital compounded by realized trades so far, times (1 + unrealized_return) if a position is currently open.

Phase 3:
Setup (once):
- Load all 4 symbols' 5m bars, inner-join on timestamp so every symbol steps through the same aligned bar sequence.
- For each symbol, compute a daily volatility series: take the last bar's close of each calendar date → daily % returns → rolling 20-day std → shift by 1 day (so today's weight only ever uses data through yesterday's close, no lookahead).

Per bar, in order:
Setup (once):
- Load all 4 symbols' 5m bars, inner-join on timestamp so every symbol steps through the same aligned bar sequence.
- For each symbol, compute a daily volatility series: take the last bar's close of each calendar date → daily % returns → rolling 20-day std → shift by 1 day (so today's weight only ever uses data through yesterday's close, no lookahead).

Per bar, in order:
1. New-day check: if this bar starts a new calendar date:
  - Pull each symbol's vol estimate for today (computed from data through yesterday). Symbols still in their 20-day warmup are simply excluded from vols_today.
  - InverseVolatilitySizer.weights(): weight_i = (1/σ_i) / Σ(1/σ_j) over whichever symbols have a valid vol estimate — this is today's fixed target allocation.
  - Resize already-open positions: for every symbol currently ENTERED, recompute target_shares = sign(side) × weight_i × current_portfolio_equity / today's_open_price, then resize() moves the position from its old share count to the new one, realizing P&L on the delta immediately at today's open price. Symbols that are flat or in cooldown aren't touched.
2. Per symbol (same 4 steps as Phase 2, just now touching the shared ledger instead of compounding a % return):
  - Fill any pending entry signal at this bar's open → size it using today's weight → resize() from 0 shares to the target (this is a normal buy/short-sell, expressed with the same function used for rebalancing).
  - Tick cooldown / check reentry.
  - Check exit (ATR TP/SL, gated by min_holding_bars=2) → resize() to 0 shares, realizing final P&L, → COOLDOWN.
  - If flat, evaluate MACD for a new signal, queued for next bar's open.
3. Mark equity: cash + Σ(shares_i × close_i) across all 4 symbols — this is the single number that drives all sizing calculations, so today's rebalance and tomorrow's entries both scale with the portfolio's actual current value, not the original $10,000.

Note: 
2024-08-28,0.0777377560406031,0.3999849651457752,0.28205395212202594,0.24022332669159585
2024-08-29,0.08871340283536701,0.41728959404049276,0.26312063375689154,0.23087636936724854
->nvda weight sudden spike since the 20-day range does not count in the day with high volatility anymore. This could be fixed later using a smoothing filter or something.

Phase 4:
意思是:vol-targeting(按波动率倒数分配仓位)这套仓位分配逻辑的设计目的,是让组合里每个标的对整体风险的贡献差不多大——不会出现"NVDA波动大,一有事组合就跟着大起大落"这种单一标的主导整个组合风险的情况。理想效果是:组合的风险调整后收益(Sharpe)应该比最差的单标的更好,至少不会比所有单标的都差,因为拉平风险贡献本身就该带来一定的分散化收益。

但现在看到的数据是:
- 组合 Sharpe = -0.61
- 单标的分别是 NVDA 0.62、KO -0.32、XOM -0.64、JPM -0.08

组合的 Sharpe 比四个标的里三个(NVDA、KO、JPM)都差,只比最差的 XOM (-0.64) 好一点点。这说明"拉平风险贡献"没有转化成组合层面更好的风险调整收益——换句话说,仓位分配机制在数学上按波动率倒数分配了权重,但没有产生预期中"分散化让组合更稳"的效果。

可能的原因(还没细查,留给后续):
1. 四个标的的收益之间相关性不够低甚至是正相关,分散化本身收益有限
2. 每日 rebalance 本身有交易成本类的隐性损耗(虽然当前没建模手续费/滑点,但频繁调仓改变了每笔交易的实际持仓规模和时点)
3. NVDA(表现最好、Sharpe最高)权重被按低波动率标准压得最低,组合反而没吃到它的正收益

One comparison caveat worth knowing, not a bug: portfolio Sharpe is computed over common_idx (the intersection of all 4 symbols' timestamps), while each standalone per-symbol Sharpe uses that symbol's own full bar history. Row counts differ slightly (NVDA 38688 vs JPM 38401 bars) due to scattered missing intraday bars, not missing days — so the windows are close but not bar-for-bar identical. Doesn't change the qualitative conclusion, just means the two Sharpe numbers aren't computed over an exactly identical bar set. (✅fixed: align_to_common_index computes the intersection of all four symbols' exact timestamps and keeps only bars present in every symbol — any bar that's missing in even one ticker gets dropped from all of them.)