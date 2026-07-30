# found_not_fixed.md — observed, out of scope, unrepaired

Logged, not fixed, so per-item performance deltas stay clean (DEPLOYER §1.6).

1. **Swing-pivot is a centred detector (fragility, not a live bug).**
   `stop_loss._swing_points` reads k=2 bars *after* each candidate. Causal only
   because it runs on closed 30m klines with the last k excluded (`range(k, len-k)`).
   Would leak instantly if ever fed a series that includes the forming bar. A
   guard-comment is recommended. (Item 5, fix_ledger.)

2. **O(n²) backtest cost.** `closed_klines` copies a growing higher-TF slice every
   bar, so full 12-ticker × 2-year runs take ~14 min. Correctness: fine. Speed only
   — flagged because it forced the dev-slice workflow.

3. **Portfolio maxDD stays low (~−1.95%) after the exit fixes**, vs the per-trade
   −4~−8% the exit-fix author expected. The residual smoothness is portfolio
   dilution (12 inverse-vol tickers, intermittent positions) — a sizing/portfolio-
   layer trait, not the exit model. A separate thread if pursued.

4. **Strategy underperforms buy&hold of its own universe** (+3.0% vs +54.9%,
   Item 1.4). Not a code defect — a strategy-quality finding, recorded so it is not
   lost.
