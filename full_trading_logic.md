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