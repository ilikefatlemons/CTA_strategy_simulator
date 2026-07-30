# orientation.md — Item 0 boundary map (read-only, no code changed)

**Stage:** 2.1 correctness pass. **Produced by:** Item 0 orientation pass.
**Rule followed:** nothing fixed here; every defect-shaped thing is logged, not touched.

---

## 0. Scope of the audited system

The live strategy is **`run_phaseF.py` → `portfolio_pullback_backtest` → `pullback_backtest`**. Everything traced below is that path.

**Dormant code that is NOT in the live path** (confirmed by import/grep, listed so it is not mistakenly audited or "fixed"):

- `src/rules/entry.py::MACDGoldenCross`, `src/rules/exit.py`, `src/rules/higher_tf.py` — used only by `run_phase2/3/4/5/D`, earlier phases.
- `src/data/higher_tf_indicators.py::live_higher_tf_indicators` (the incremental "live forming-candle" MACD/ATR) — used only by `run_phaseC` (verification demo) and `run_phaseD`. **The live strategy never calls it.**
- `src/indicators.py::macd` — used by `viz/chart.py` for display and by dormant phases. Not a signal input in `run_phaseF`.

Consequence: the "live forming candle / truncation-invariance" machinery the repo docs emphasise is **not exercised by the current strategy**. The live strategy's higher-timeframe reads are all **frozen, fully-closed** resampled bars via `resample_ohlcv` + `closed_bar_positions`.

---

## 1. Boundary A — Ingestion (where external price data enters)

| Attribute | Finding | Source line |
|---|---|---|
| External source | Alpaca Market Data API, `StockHistoricalDataClient`, **IEX** feed | `fetch_alpaca.py:44-60` |
| Bar interval | **5 minutes** (`TimeFrame(5, Minute)`) — finest granularity fetched; 15m/30m/2h are resampled from it, never re-fetched | `fetch_alpaca.py:38`, `resample.py` |
| Timestamp field | Alpaca bar `timestamp`, carried through unchanged to the CSV | `fetch_alpaca.py:67-88` |
| Timestamp meaning | **Bar OPEN / interval-start time.** First RTH bar of a session = `13:30:00+00:00` = 09:30 ET (see raw CSV). Whole codebase treats it as open-time: `resample` uses `label="left"`, and the closed-bar guard adds `+ rule` to derive close time. | `NVDA_5m.csv:2`, `resample.py:52,90-91` |
| Timezone | **UTC, tz-aware** (e.g. `2024-07-18 13:30:00+00:00`); converted to `America/New_York` only for the RTH session filter and per-session resample anchoring | `fetch_alpaca.py:84`, `resample.py:42` |
| Source documents convention? | Alpaca's documented convention is **timestamp = start of the bar interval**; the CSV data is consistent with that. **Not independently/empirically verified in code.** Empirical event check is deferred to Item 1. | — |
| OHLC adjustment | **`Adjustment.ALL` — split + dividend ADJUSTED.** One adjusted series serves both pattern logic and P&L. | `fetch_alpaca.py:58` |
| Session filter | Keeps 09:30 ≤ ET < 16:00 (RTH only); drops pre/post-market | `fetch_alpaca.py:84-86` |
| Backtest ingestion | `run_phaseF.main` reads `data/raw/{symbol}_5m.csv`, `parse_dates=["timestamp"]`, clips to `START,END = 2024-07-18 … 2026-07-17` | `run_phaseF.py:26-27` |

**One convention, applied where?** Open-time is native from Alpaca and never re-converted; `resample`/`closed_bar_positions` account for it in one shared place. So there is one convention — but it is **open-time, not close-time** (G1 asks for close-time). That gap is Item 1's subject.

---

## 2. Boundary B — Signal computation (where a decision value is produced)

Decision bar = the 5m bar `i` currently being evaluated in the backtest loop (`pullback_backtest.py:101`). "Max index read" is relative to that.

| Decision value | What it reads | Assigned to | Max bar index read | Causal? |
|---|---|---|---|---|
| Higher-tf closed bars (15m/30m/2h) | `resample_ohlcv` (open-time, per-session 09:30-anchored) gated by `closed_bar_positions` (`t+rule ≤ open_time[i]`) | bar `i` | last bar **fully closed** as of `i`'s open time | yes |
| 2h trend bias | `get_trend_bias`→`ma_array_state` (MA5/20/50 strict, `rolling().mean().iloc[-1]`) on closed 2h | bar `i` | last closed 2h bar | yes |
| Pullback confirm | `pullback_occurred`→`ma_fast_mid_state` (MA5/20): 30m closed **and** (5m or 15m) | bar `i` | 5m = **`i` (current bar)**; 15m/30m = last closed | yes |
| Entry trigger | `get_entry_trigger`→`ma_fast_mid_state`: 5m **and** 15m align to bias | bar `i` | 5m = `i`; 15m = last closed | yes |
| ATR (stop distance + partial-TP R) | `indicators.atr` = `TR.rolling(14).mean()`, full-series precompute, `atr_now = atr[i]` | entry sized for `i+1` | `i` | yes (rolling mean is strictly causal; precompute ≡ incremental) |
| Trailing ATR | `atr_full[i-1]` (deliberately `i-1`, since `i`'s ATR needs `i`'s close, unknown when `i`'s price is tested) | bar `i` exit test | `i-1` | yes |
| Signal state machine | `PullbackEntryEngine` carries `_bias`,`_pullback_seen` forward; bias change (incl. neutral) resets pullback | bar `i` | `i` | yes (forward-only state) |
| Cooldown | `CooldownManager.on_bar(closed 2h)`: consecutive-SL count + 2h structure-break / release on 3 stable clean arrays | bar `i` | last closed 2h bar | yes |
| Portfolio daily vol weight | `rolling_daily_atr_vol` = daily(ATR%/close) then **`.shift(1)`**; `InverseVolatilitySizer` = 1/vol renormalised | trading day `d` | `d-1` | yes |

No negative shifts, no successor-bar comparisons, no centred/symmetric smoothing, and no full-sample z-score / percentile / rank found in the **signal** path on this pass. (Item 4 and Item 5 will re-verify by test and by line-by-line read respectively — this is a first-pass reading, not their verdict.)

---

## 3. Boundary C — Execution (where an order becomes a fill)

| Event | Trigger tested against | Fill price | Bar | Order type | Line |
|---|---|---|---|---|---|
| **Entry** | signal fires at close of `i` | `df_5m["open"][i+1]` | **`i+1`** | next-bar market-on-open | `pullback_backtest.py:182,185` |
| Exit — hard SL (pre-partial) | `close[i] ≤ stop` (long) | **`stop_loss`** (the line itself) | `i` | stop, close-triggered | `:124-135` |
| Exit — partial TP leg A | `close[i] ≥ 2R trigger` | **`trigger`** (the line) | `i` | limit, close-triggered | `:123,136-144` |
| Exit — trailing SL leg B | `close[i] ≤ chandelier` | **`trailing_stop`** (the line) | `i` | trailing stop, close-triggered | `:150-168` |

**Entry** satisfies the t→t+1 separation (decision from `close[i]`, fill at `open[i+1]`). Equity is held flat until `entry_bar_idx ≤ i` so there is no same-bar mark-to-market (`:196-203`).

**Exits are CLOSE-ONLY** — this is the single most important execution finding:

- Every exit breach is tested against **`close[i]` only**, never `high[i]`/`low[i]`. Intrabar wick touches of a stop or target that the bar then recovers from are **invisible** to the engine.
- When an exit does fire, it fills at the **exact line price** (stop / trigger / trailing), which for a stop is *better* than the close that triggered it.
- SL is checked **before** TP (`if hit_sl … elif hit_tp`), i.e. already the pessimistic ordering — **but moot**, because a single `close` value cannot be simultaneously `≤ stop` and `≥ target`. **A "both levels hit in one bar" event is undetectable by construction here.**

Directional bias of the close-only model: stops under-trigger (favourable — fewer stop-outs, wick escapes), and fills beat the close (favourable); the partial-TP under-triggers too (unfavourable). Net sign is not assumed here.

---

## 4. Library / computed indicators — flags

| Indicator | Config | In live path? | Centre / displace / symmetric / retroactive? |
|---|---|---|---|
| `indicators.atr` | `TR.rolling(14).mean()` | **yes** (stops, sizing) | No. Trailing, causal. |
| `ma_filter` MAs | `rolling(5/20/50).mean().iloc[-1]` | **yes** (all signals) | No. Trailing, causal. |
| `stop_loss._swing_points` | pivot if `high[i]==max(high[i-k…i+k])`, `k=2` | **yes** (platform stop) | **CENTRED pivot** — window includes `k` bars *after* the candidate. ⚠️ Contained today: applied only to **closed** 30m klines with the last `k` bars excluded (`range(k, len-k)`), so every bar it reads is already closed ≤ signal bar → causal *as wired*. **Would leak instantly if ever fed a series that includes the forming/current bar.** Flagged for Item 5's table. |
| `indicators.macd` | `ewm(span, adjust=False)` | no (chart + dormant) | causal anyway; out of live path |
| `live_higher_tf_indicators` | incremental EWM/ATR on forming candle | **no** (dormant) | causal by its own construction; out of live path |

No Ichimoku, ZigZag, fractal, centred EMA, or full-sample normalisation anywhere in the live path.

---

## 5. Preliminary per-item implication (neutral — not a verdict, not a fix)

| Item | First-pass expectation to test at its own gate |
|---|---|
| 1.1 timestamp | Convention is **open-time, consistently guarded**, not close-time. Question is whether open-time-with-correct-guards satisfies G1's intent or must be restamped. Needs the empirical hard-timestamped-event check. Plausibly "no change." |
| 3.2 same-bar fill | Entry **already** fills `open[i+1]`. Likely "no change"; confirm with the 0/1/2/3-bar lag curve. |
| 3.1 intrabar path | **HALT CANDIDATE.** The doc presupposes a high/low range-based exit engine and wants the optimistic-vs-pessimistic both-hit spread. This engine is **close-only**; both-hit share is 0 by construction and the requested spread is **not computable without first converting exits to range-based checking — a large, separate modelling change.** Must be raised, not silently built. |
| 2.1 confirmation leakage | Signal dated at trigger bar `i`, reads ≤ `i`, no negative shift. Likely "no change"; confirm with the extra-forward-shift test. |
| 2.4 context leakage | Reading task. Main object of interest = 2h MA-array bias (causal) and the centred swing-pivot stop (causal as wired). Deliver the full filter→max-index table. |
| 1.2 split/div | All data is **adjusted** (one series for shape + P&L). But the signal uses MA relationships, **not candlestick shapes**; no `open==close` doji tests exist. Assess N/A-vs-applicable at its gate. |
| 2.6 warm-up | Rolling/`adjust=False` seeds, no full-sample seed; `len<slow→neutral` gives implicit warm-up. No explicit discard-stretch. Low risk; verify at gate. |
| 1.4 universe | Hardcoded 12-ticker list, all current survivors, chosen at authoring time → survivorship present by construction. Likely a documented-caveat deliverable. |

---

## 6. Open questions / uncertainties (flagged, not resolved)

1. **Alpaca timestamp = open-time** is strongly indicated (CSV + `label="left"` + docs) but **not yet empirically confirmed** against a hard-timestamped event. → Item 1.
2. **Item 3 scope**: is converting the close-only exit model to range-based checking in scope for this stage, or does the close-only model itself get logged to `found_not_fixed.md` while Item 3 is marked not-applicable-as-specified? This is a scope decision for the owner, raised at Item 3's gate.
3. **Ledger mechanics**: `run_phaseF.main` prints portfolio/per-ticker return+Sharpe but then opens the blocking chart. A headless metrics runner (return, CAGR, Sharpe+√252, maxDD, #trades, win rate, avg-hold-bars, avg-bps) will be written as measurement infra (not a strategy change) to populate `fix_ledger.md`. Fixed ledger range = `2024-07-18 … 2026-07-17`, full 12-ticker universe. Runnability of the environment not yet tested.
4. **Toggle mechanism**: no central config exists; engine components are dependency-injected with defaults (`StopLossCalculator`, `TakeProfitManager`, `CooldownManager`, `PullbackEntryEngine`). The per-fix toggle design will be proposed at Item 1's gate, not invented here.
