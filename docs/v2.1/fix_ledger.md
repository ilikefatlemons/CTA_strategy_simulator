# fix_ledger.md — Stage 2.1 before/after metrics

Before/after on a fixed data range per change, so each change's isolated effect
is visible. Range unless noted: **full sample, 2024-07-18 → 2026-07-17, all 12
tickers**. Sharpe annualised √252. Toggle names refer to `src/config.py`.
"before" = pre-2.1 behaviour (toggle OFF); "after" = new default (toggle ON).

## Standing deliverables (no metric row)
- **Item 0 — orientation:** `orientation.md`. Read-only boundary map, no code.
- **Item 1 — 1.1 bar timestamp:** verified **no change**. Convention is
  open-time, consistently guarded (`t + rule ≤ open_time`); confirmed
  empirically — every session's first RTH bar is 09:30 ET (500/500), and the
  FOMC 2024-09-18 14:00 ET reaction lands in the 14:00 bar (vol 124,959 / range
  2.51 vs the quiet 13:55 bar). No code touched → metrics identical.

## Entry-timing refactor (pre-Item-2 convention change)
- **Toggle:** `entry_on_completed_bar` (default `True`). **Date:** 2026-07-25.
- **What / why:** the 5m pullback/trigger now read only *completed* candles
  (≤ i-1), the entry-stop ATR uses i-1, and the fill is bar i's *own* open — the
  decision is knowable at bar i's open. Pre-2.1 read `close[i]` and filled i+1's
  open. **Both are lookahead-free**; the change realigns higher-TF context to the
  entry bar. User-directed convention change, **not a leak fix** — the drop below
  is incidental, not a removed leak.

| Metric | before (forming-bar, OFF) | after (completed-bar, ON) | Δ |
|---|---|---|---|
| Portfolio return | +34.33% | +31.40% | −2.93 pp |
| Portfolio Sharpe (√252) | +5.65 | +5.25 | −0.41 |
| Max drawdown | −0.82% | −0.81% | +0.01 pp |
| Batches | 1592 | 1616 | +24 |
| Legs (fills) | 2242 | 2249 | +7 |

Core metrics only; CAGR / avg-holding-bars / avg-bps deferred to the audit-item
rows (would need a re-run). Delta is small and boundary-driven, as predicted.

## Item 2 — 3.2 Same-bar close fill (entry) — VERIFIED NO-CHANGE
Date: 2026-07-25. Full sample, 12 tickers. No strategy code changed.

By inspection the entry fills at `open[fill_idx]` (open[i] completed / open[i+1]
forming), never `close[i]` — G3 satisfied. Verification = entry lag curve, fill
delayed 0/1/2/3 bars (`entry_lag_bars` diagnostic; lag 0 = real strategy, and it
reproduces +31.40% / +5.25 / 1616 exactly):

| lag | return | sharpe | maxDD | batches |
|---|---|---|---|---|
| 0 | +31.40% | +5.25 | −0.81% | 1616 |
| 1 | +32.51% | +5.42 | −0.75% | 1604 |
| 2 | +32.90% | +5.48 | −0.67% | 1586 |
| 3 | +33.09% | +5.58 | −0.58% | 1575 |

**No collapse at lag 1** → the entry is not a lookahead artifact (confirms the
inspection). **No steep decay** — result ~flat under a 0–3 bar fill delay.

Scope of the claim (corrected — the earlier draft overreached): the test perturbs
only the entry **fill timing**, holding the signal fixed. It therefore rules out
entry fill-timing as a profit source — and nothing more. It does **not** prove the
profit "comes from the exits": flat-under-lag is equally consistent with the exit
model, a long-biased survivor universe, or noise run through asymmetric exits. On
an un-tuned model the slight uptick with lag is noise, not signal. It also does
**not** test the entry *signal* (direction selection) — only the fill — so the
signal is not eliminated either; that needs a random-entry benchmark (Item 5 /
1.4). Localizing the actual source waits on running Items 3 / 3.5 (and 1.4 / 2.4).
The sub-1% maxDD across all lags is *suggestive* of exit-model drawdown truncation,
but only suggestive.

## Item 3 (3.1) + Item 3.5 — exit-model fixes
Date: 2026-07-26. Full sample, 12 tickers. Toggles in `src/config.py`, all default
True (honest). Baseline (all toggles off) reproduces the pre-fix number exactly.

| config | return | sharpe | maxDD | batches | legs |
|---|---|---|---|---|---|
| baseline (all off) | +31.40% | +5.25 | −0.81% | 1616 | 2249 |
| +3.5 gap-fill | +10.98% | +1.65 | −1.78% | 1616 | 2249 |
| +3.5 +range-based (ALL-ON, honest) | +3.00% | +0.44 | −1.95% | 1673 | 2301 |
| +range-based, optimistic both-hit | +3.19% | +0.48 | −1.91% | 1673 | 2305 |

Isolated deltas:
- **3.5 gap-fill (`gap_fill_exits`): −20.42 pp return, −3.60 Sharpe — dominant.**
  Filling overnight-gapped stops at the exact stop (not the gap-open) was ~65% of
  the reported return. Same trades (1616/2249), worse fills.
- **Range-based trigger (`range_based_exit_trigger`, A1): −7.98 pp, −1.21 Sharpe.**
  Intrabar stop-outs the close-only trigger let escape (+57 batches, +52 legs).
- **Both-hit (`pessimistic_both_hit`): −0.19 pp, −0.04 Sharpe** (pessimistic vs
  optimistic). Negligible (0.4% both-hit), fixed anyway for strict logic; pessimistic
  is the honest default.
- **All-on vs baseline: −28.40 pp return (−90%), Sharpe 5.25 → 0.44.**

Verdict: the reported edge was overwhelmingly an **exit-fill artifact.** Sharpe drops
from an absurd 5.25 to a mundane 0.44 — the §9 outcome, past the top of the 30–70%
band. Vindicates the Item 2 red flag: the profit was downstream of the entry.

Residual: maxDD only −0.81% → −1.95%, not the teammate's −4~−8%. The gap is
**portfolio dilution** (12 inverse-vol tickers, intermittent positions) — a
sizing/portfolio-layer trait, not the exit model. Separate thread (sizing / 1.4 / 2.4).

Decision (2026-07-26): applying gap-fill to the **TP** was considered and
**deliberately declined**. TP stays a limit at its trigger. Under-crediting a
gapped-up TP reports *worse* than reality (a conservative bias, not a leak), so it
leaves a live buffer — realized TP fills on a gap can only beat the plan. Gap-fill
stays on the loss side only (SL + trailing). Not to be re-opened.

## Item 4 — 2.1 Confirmation-bar leakage — VERIFIED NO-CHANGE
Date: 2026-07-26. No code changed.

Causal by construction: 5m read is `df_5m.iloc[:i]` (≤ i-1), higher-TFs are closed
bars ≤ `open_time[i]`, and the confirmation is dated at the **trigger** bar, not the
setup bar (the doc's correct behaviour). No negative shifts / successor comparisons /
fwd-back-fill. Shift-test (signal shifted 0/1/2/3 bars, current honest strategy):

| lag | return | sharpe | maxDD | batches |
|---|---|---|---|---|
| 0 | +3.00% | +0.44 | −1.95% | 1673 |
| 1 | +0.26% | −0.01 | −3.01% | 1659 |
| 2 | +1.83% | +0.27 | −2.22% | 1649 |
| 3 | +2.92% | +0.48 | −2.03% | 1635 |

Non-monotonic/noisy — lag 1 dips, lag 2/3 recover near lag 0. **Not** a sustained
collapse (a real leak stays dead at every lag ≥ 1), so no leak signature; at Sharpe
0.44 the curve just re-samples noise (uninformative on magnitude). Verdict rests on
inspection. No leak, no change.

## Item 5 — 2.4 Context-filter leakage — VERIFIED NO-CHANGE (by reading, per §10.6)
Date: 2026-07-26. No code changed. A reading audit of every input, not a test.

| input | module:line | max read (rel. signal bar i) | verdict |
|---|---|---|---|
| 2h bias (MA5/20/50) | ma_filter:22-36,75 | ≤ i-1 (last closed 2h) | causal |
| 30m/15m/5m states | ma_filter:39-88 | ≤ i-1 | causal |
| ATR (stop dist / R) | indicators:32-38 | i-1 (trailing) | causal |
| swing-pivot stop | stop_loss:14-24 | ≤ i-1 (last closed 30m) | ⚠ centred-but-contained |
| cooldown | cooldown:70-99 | ≤ i-1 | causal |
| portfolio vol / sizer | vol_estimator:39-55, sizing:7-12 | d-1 (shift(1); cross-sectional) | causal |

No input reads past the signal bar; **no full-sample statistic anywhere** — no
centred/symmetric smoothing in the signal, no ZigZag/fractal/Ichimoku, no
percentile/z-score/normalisation over the sample, no hindsight regime/date exclusion.

Watch item (not a fix — causal as wired): `_swing_points` is a *centred* pivot (reads
k=2 bars after the candidate), safe only because it runs on closed 30m klines with the
last k excluded (`range(k, len-k)`). Would leak instantly if fed a series that includes
the forming bar — a guard-comment is recommended.

Verdict: the context filter the doc expected to carry the largest leak does not. The
residual ~0.44 Sharpe is honest (mundane), not a 2.4 artifact. Remaining real threads:
survivor universe (1.4) and portfolio smoothing (sizing) — both outside 2.4.

## Item 6 — 1.2 Split/dividend vs. candle shape — N/A (documented)
Date: 2026-07-26. Data is `Adjustment.ALL` (split+div adjusted), one series for
classification and P&L. But the strategy classifies **MA relationships, not
candlestick shapes** — no exact-equality shape tests (`ma_array_state` uses strict
`>`/`<`; the only `==` is legitimate swing-pivot max detection on adjusted data).
The specific defect (corporate-action fake candles, doji equality) does not bind.
**N/A — caveat only, no code.** Would bind if candlestick-shape patterns are added.

## Item 7 — 2.6 Warm-up leakage — VERIFIED CLEAN (mechanism added, default off)
Date: 2026-07-26. By inspection: ATR/MAs are trailing `rolling()` (NaN until full,
never full-sample-seeded); array checks return `neutral` until the window fills. No
warm-up *leakage*. A behavioural drop-stretch run would be noise-dominated at Sharpe
0.44 (cf. Item 4), so inspection is the verification. Added `warmup_discard_bars`
(config, **default 0 = no change**) as an optional explicit discard buffer; a
non-zero value (~5–10× the longest lookback = 50-bar 2h MA / 20-day vol) is a
**parameter choice left to the owner (§1.4)**, not set here.

## Item 8 — 1.4 Universe survivorship — documented caveat + benchmark
Date: 2026-07-26. The 12-ticker list is hand-picked *today*; point-in-time data is
unavailable, so per the doc the deliverable is a written caveat. Verification
(buy&hold, full sample):

- **0 / 12 names fell ≥ 70% from a peak** → confirmed **survivor list**.
- **Equal-weight buy&hold of the universe: +54.9%** vs the honest strategy's +3.0%.

So the universe drifted up ~55% while the strategy captured almost none of it — the
strategy **massively underperforms passive holding** of its own survivor-biased
universe. Survivorship still inflates the long side (no delistings to lose on); a
realistic universe with failures would make it worse. **Caveat, no code fix** —
cannot reconstruct point-in-time from the current data. With 4.1 (no OOS, N ≈ 40+),
this is the sobering bottom line: **no demonstrated edge.**

## A3 — chandelier extreme intrabar circularity — FIXED
Date: 2026-07-26. Toggle `trail_extreme_prev_bar` (default True = honest).
Full sample, 12 tickers. Toggle-off reproduces the prior baseline exactly.

**Defect:** `extreme_since_entry` was updated with the *current* bar's high/low, the
chandelier trail computed from it, and that same bar then tested against the trail —
so a long's exit level was `high_i − 3·ATR`, dependent on bar i's own high. Knowable
only if the high preceded the low: an intrabar order the engine cannot know (G4).
**Fix:** trail computed from the extreme through **i−1**; bar i's high/low folded in
only *after* the test. Pure reordering, no parameter.

| config | return | sharpe | maxDD | batches | legs | trail exits |
|---|---|---|---|---|---|---|
| A3 OFF (baseline) | +3.00% | +0.44 | −1.95% | 1673 | 2301 | 628 |
| **A3 ON (honest)** | **+4.43%** | **+0.68** | −1.71% | 1673 | 2301 | 628 |

**Effect is POSITIVE (+1.43 pp, +0.24 Sharpe) — opposite to the finding author's
predicted 略降收益.** The defect had two opposing effects and A3 counted only one:
- *price*: raising the trail records long exits **higher** = optimistic (A3's point);
- *timing*: a raised trail is **easier to hit**, so the old code exited **earlier**,
  cutting winners short.
The timing effect dominates here, so honest accounting nets out positive. Counts are
identical because the chandelier sits pinned at `floor_stop` most of its life (the
extreme shift is absorbed by `max(raw, floor_stop)`), and where it lifts, exits are
delayed only a bar or two — never long enough to block the next entry.

**Gain scrutinised, not banked** (a performance-*raising* fix is where bias enters):
under ON the trail level is known **before bar i opens** — a stop that could actually
rest in the market; under OFF it depended on bar i's own high, which no one could
have placed. ON is strictly more realistic, so the gain is genuine.
**Does not change the stage conclusion:** Sharpe 0.68 raw is still ≈0 after the 4.1
haircut (N≈40+, no clean OOS), and +4.43% still trails +54.9% buy&hold by ~50 pp.

## A5 — cooldown counted 5m calls as if they were 2h bars — FIXED
Date: 2026-07-26. Toggle `cooldown_counts_2h_bars` (default True = honest).
Full sample, 12 tickers. Toggle-off reproduces the post-A3 baseline exactly.

**Defect:** `pullback_backtest.py:169` calls `cooldown.on_bar` on every 5m bar
(~19.5x per 2h bar, since a 390-min session yields four 2h bins: 24/24/24/6 bars).
`cooldown.py:94` advanced `_stable_count` per *call*, so the release floor was
**3 x 5m = 15 min** instead of the documented **3 x 2h = 6 hours**. Secondary defect
(same root cause, not in the original write-up): `check_structure_break` was also
re-evaluated per 5m bar, so the cooldown re-armed off the same 2h bar immediately
after releasing — an on/off flicker.

**Fix:** `CooldownManager` remembers the last 2h timestamp and returns early when it
has not advanced — gating the whole rule (arm check *and* release counter) to one
evaluation per new 2h bar. No parameter changed (`release_confirm_bars` stays 3).

Measured cooldown behaviour (dev slice, 3 tickers / 3 months):

| | before | after |
|---|---|---|
| arms | 98 | 20 |
| mean active duration | 18.2 bars = 91 min | 157 bars = 786 min |
| share of time active | 12.0% | 21.2% |

| config | return | sharpe | maxDD | batches | legs |
|---|---|---|---|---|---|
| A5 OFF (baseline) | +4.43% | +0.68 | −1.71% | 1673 | 2301 |
| **A5 ON (honest)** | **+5.92%** | **+0.97** | −1.77% | 1507 | 2083 |

**Batches −9.9%** (predicted −10~20%). Return **+1.49 pp**, Sharpe **+0.29**.

**Tests:** the two cooldown unit tests fed the *same* last timestamp on every call
(`_trend_klines` always ended at 2024-01-02 05:30), encoding the one-call-one-bar
assumption that hid this bug. Timestamps now advance (`bar=` param), and a new
regression test `test_cooldown_does_not_release_on_repeated_same_2h_bar` asserts that
50 repeated same-2h-bar calls do **not** release. 10/10 pass.

### Cross-cutting finding: the "improvements" are trade-reduction
A3 and A5 both **raised** reported performance, and both did so by making the
strategy trade less (A3: fewer premature trail exits; A5: −166 entries). Removing
166 trades *gained* 1.49 pp — evidence the **per-trade edge is negative**, i.e. the
strategy pays to trade and any throttle flatters it. Read these as the strategy being
throttled toward inactivity, not as it getting better. Still +5.92% vs +54.9% for
buy&hold of the same names.

**Caveat on record:** the cooldown's parameters (3 consecutive SLs, 3 stable bars,
≥2-MA structure break) were chosen while the rule was **inert** — never evaluated in
a working state. Any benefit now is untested, possibly luck. Same applies to the
3xATR chandelier after A3. Neither touched (§1.4).

## A6 — risk ATR moved from 5m to 30m — FIXED
Date: 2026-07-26. Toggle `risk_atr_on_30m` (default True). Full sample, 12 tickers.
Toggle-off reproduces the A5 baseline exactly.

**Defect:** the entry stop and the chandelier trail both sized off ATR(14) on **5m**,
while the stop's other candidate — the swing platform — is measured on **30m**. One
`min()` comparing two timeframes, in a strategy whose thesis is a 2h-trend pullback.
**Fix:** both risk legs read ATR(14) on 30m, from the last fully CLOSED 30m bar
(`closed_idx["30m"][i]` already excludes the forming bar — no extra shift, no
per-bar recompute). `extreme_since_entry` kept as-is. No multiplier changed.

Both legs moved together deliberately: `rr_trigger(2) x sl_atr_mult(1.5) ==
chandelier_mult(3.0)` places the trail at breakeven when the partial TP fires, which
only holds if both use the same ATR. Splitting into two toggles would have measured
two incoherent configurations.

**Measured magnitude:** mean ATR(14) 30m / 5m = **2.84x** (range 2.63–2.99 over all
12 symbols) — *above* the √6 = 2.45 random-walk expectation, since the 5m ATR is
damped by microstructure noise. Stops widen ~2.8x.

| config | return | sharpe | maxDD | batches | legs |
|---|---|---|---|---|---|
| A6 OFF (baseline) | +5.92% | +0.97 | −1.77% | 1507 | 2083 |
| **A6 ON (honest)** | **+9.35%** | **+1.17** | **−2.98%** | 1418 | 1966 |

**The risk measures disagree — report both.** Sharpe improves (0.97 → 1.17), but
maxDD worsens materially (−1.77% → −2.98%) and **return/maxDD degrades 3.34 → 3.14**.
The +58% relative return gain is roughly fully paid for by additional risk. Cause:
**no risk-based position sizing anywhere in this engine** — each batch is 100% of the
ticker's allocation regardless of stop distance, so a 2.8x wider stop is a 2.8x
larger loss per stop-out with no compensating size reduction. A6 exposes this
structural gap rather than causing it. Logged, not acted on.

**Methodology note:** the dev slice (3 tickers / 3 months) gave the OPPOSITE sign
(−1.42% → −2.62%) from the full sample. Dev is a wiring smoke-test only; it must not
be used to judge the direction of a change.

### Cross-cutting finding (updated): every fix profits by trading less
| after | return | batches |
|---|---|---|
| 3.1 + 3.5 | +3.00% | 1673 |
| A3 | +4.43% | — |
| A5 | +5.92% | 1507 |
| A6 | +9.35% | 1418 |

255 fewer batches, +6.35 pp — roughly **+2.5 bps of portfolio return per trade
removed**. This is the signature of a **negative per-trade edge**: if each additional
trade subtracts value, the profit-maximising trade count is zero, and these
"improvements" are the machine being throttled toward inactivity. Still far below
+54.9% for passively holding the same 12 names.

## ⚠️ Verification gaps — recorded, NOT performed (2026-07-26)
Disclosed so no reader assumes these were done. None can overturn the stage
conclusion (strategy +3.0% vs +54.9% buy&hold of its own universe; N≈40+; no OOS).

1. **1.4 — random-entry benchmark NOT run.** The ≥70%-drawdown count was run
   (0/12 → survivor list confirmed). The doc's second test — random entries after
   any multi-day decline on the same universe, to see whether the *universe* rather
   than the strategy produces the result — was never executed.
2. **2.6 — drop-the-opening-stretch re-run NOT performed.** Verdict was reached by
   inspection (trailing `rolling()`, neutral-until-full, no full-sample seed) plus
   the argument that a behavioural run is noise-dominated at Sharpe 0.44. Defensible,
   but it is not the verification the doc specifies.
3. **§3 ledger schema partial.** Rows carry return / Sharpe / maxDD / batches / legs.
   **CAGR, average holding period in bars, and average P&L per trade in bps were
   never computed** for any item. Would require a re-run plus new metric code.
4. Minor: 1.1 was verified on **one ticker (NVDA)** — there is only one data source
   (Alpaca), so "repeat per source" is satisfied, but not per instrument.

## ⚠️ Red flag on ABSOLUTE numbers (independent of this change)
Sharpe **~5.3–5.7** with max drawdown **~0.8%** on **+31–34%** return is not a
realistic result — the §9 "look harder" signal. The entry refactor did not touch
it (not a leak fix). Prime suspects, queued: **Item 3 (3.1)** — close-only /
intrabar exits (stops trigger only on a close beyond the line and fill at the
exact line, structurally erasing drawdown); **Item 5 (2.4)** — context filter.
Treat these absolute figures as provisional until 3 and 5 run; expect a large drop.
