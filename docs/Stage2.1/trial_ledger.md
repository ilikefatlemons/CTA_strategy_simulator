# trial_ledger.md — Item 4.1 (Stage 2.1)

Honest count of distinct configurations ever evaluated, for a multiple-testing
haircut on the reported Sharpe. Reconstructed as a **lower bound** — an
underestimate stated as one is useful; a missing number is not.

## Reconstructed trial count (lower bound)

**Strategy development (phases 2→F, from repo history + READMEs):**
- Entry-rule variants: MACD golden-cross (v1.0, phases 2–5); higher-TF MACD on
  15m/30m/1h (phase D, 4 variants); MA-array pullback cascade (v1.1/F). **≥ 6**
- Cascade composition (documented in `ma_filter.py`): "30m + any of 5m/15m" was
  chosen over "all three", "any 2", and "no 30m" — **≥ 4** compared, 4 tickers each.
- Parameter values, each of which could have been otherwise: MA 5/20/50, stop
  1.5×ATR, offset 0.3%, swing k=2, TP 2R, partial 50%, chandelier 3×ATR, cooldown
  3-SL / 3-stable, vol window 14/20d, ATR 14. **≥ 11 knobs.**

**This audit stage (2.1), configs actually run:**
- Entry convention A/B (completed vs forming): 2
- Entry lag curves (0/1/2/3), run twice (pre- and post-exit-fix): 8
- Exit-fix comparison (baseline / +3.5 / +range / optimistic): 4
- Item 3 probe, Item 1.4 universe check, dev-slice iterations: **≥ 6**

**Lower-bound N ≈ 40+ distinct configurations run, on top of ≥ 20 parameter/variant
choices in development.** True N is higher — abandoned attempts not in git are
unrecoverable. State it as an underestimate.

## Out-of-sample status

**There is effectively no out-of-sample data.** The full 2024-07 → 2026-07 sample
was used for development *and* repeatedly inspected + modified against during this
stage — every A/B, lag curve, and exit toggle was evaluated on it and the strategy
changed based on the result. Per the DEPLOYER doc, modification after inspection
converts a segment to in-sample permanently. So the entire sample is in-sample.

## Haircut

Raw Sharpe (honest all-on strategy) = **0.44**. With N ≈ 40+ trials and **zero clean
OOS**, a Deflated-/Harvey–Liu haircut lands well below that — plausibly **≈ 0**. The
doc's placeholder (halve when N is uncertain) gives ~0.22, which is *generous* here
given the missing OOS.

**Honest read: no demonstrated, out-of-sample edge.** Corroborated by 1.4 — a
buy&hold of the same universe returned +54.9% vs the strategy's +3.0%.
