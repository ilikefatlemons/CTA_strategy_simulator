# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# run all tests
python -m pytest tests/ -q

# run a single test
python -m pytest tests/test_pullback_entry.py::test_open_and_reentry_are_the_same_bound_method -q

# refetch raw market data (split+dividend-adjusted 5m bars, ~2yr lookback) for all 12 symbols in SYMBOLS
python -m src.data.fetch_alpaca

# run the current (v1.1-pullback) multi-ticker portfolio backtest + open the interactive chart
python -m src.run_phaseF
```

Everything is invoked as `python -m src.<module>` from the repo root — the code imports via `from src....` throughout, so running a script file directly (`python src/run_phaseF.py`) will fail with `ModuleNotFoundError: No module named 'src'`.

No `pyproject.toml`/`requirements.txt` exists yet; dependencies (`pandas`, `alpaca-py`, `python-dotenv`, `lightweight-charts`, `pytest`) are assumed already installed in the active environment. Alpaca credentials (`ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`) are read from a local `.env` (gitignored).

## Architecture

This is a bar-by-bar (not vectorized) backtesting engine for a multi-ticker CTA-style trend/pullback strategy, evolved through several numbered phases (`src/run_phase2.py` … `run_phaseF.py`). **`run_phaseF.py` is the current/active entry point** — everything before it (`run_phase2`–`run_phaseD`) is earlier-phase scaffolding kept for reference, not the live strategy. Two long-form docs in the repo root (`README_MVP.md`, `README_v1.1.md`) record the phase-by-phase design history and rationale in detail; `trading_logic_full_walkthrough.md` is a from-scratch, code-cited walkthrough of the *current* strategy end to end.

### Data flow (current strategy, `run_phaseF.py`)

```
fetch_alpaca.py (Adjustment.ALL, split+dividend adjusted 5m bars → data/raw/{symbol}_5m.csv)
  → resample.py (5m → 15m/30m/2h, per-session-anchored)
  → indicators.py (shared ATR/MACD formulas, used by both single-ticker and portfolio layers)
  → rules/ (signal + risk modules, see below)
  → engine/pullback_backtest.py (single-ticker bar-by-bar execution)
  → engine/portfolio_pullback_backtest.py (daily inverse-vol weighting + rebalance across tickers)
  → viz/chart.py (lightweight-charts UI: candles + equity curve + stats panel)
```

### Strict no-lookahead discipline (the one invariant everything else is built around)

- **Higher-timeframe bars must be fully closed** before use. `resample.py::closed_bar_positions` is the single shared boundary calculation (`t + rule <= open_time`) — both the backtest engine and the vectorized higher-tf-indicator path call into it, so there is only one place that can get this boundary wrong.
- **Signals fill at the *next* bar's open**, never the signal bar's own price (`entry_price = df_5m["open"].iloc[i + 1]` in `pullback_backtest.py`).
- **Trailing-stop ATR uses bar `i-1`, not bar `i`** — bar i's own ATR needs bar i's own close, which isn't known yet at the moment bar i's price is being checked against that stop. This is the single easiest place to accidentally introduce a future-function bug if touched.
- Portfolio-layer daily vol/weights are `.shift(1)`'d (`vol_estimator.py`) so day D's weight only ever uses data through D-1's close.
- There was an earlier, since-abandoned "wait for full close" design for higher-timeframe indicators (`README_v1.1.md` §3.2) — it was correct but too conservative (threw away in-progress-candle signal). The current design uses live-updating forming candles instead, verified via truncation-invariance rather than a "staircase" check. If you're touching `higher_tf_indicators.py`, read that section before changing the update logic.

### Signal cascade (`src/rules/`)

Four-timeframe cascade, all built on two MA-array-state primitives in `ma_filter.py` (`ma_array_state` = strict 3-line MA5/20/50 for 2h bias only; `ma_fast_mid_state` = 2-line MA5/20 for the 30m/15m/5m pullback/trigger checks — 3-line was too strict to ever complete on the smaller timeframes):

1. **2h bias** (`get_trend_bias`) — strict bullish/bearish MA array or neutral (no trade).
2. **Pullback confirmation** (`pullback_occurred`) — 30m must reverse (hard requirement) AND at least one of 5m/15m also reverses (OR, not AND — 5m and 15m don't reverse in a fixed order).
3. **Entry trigger** (`get_entry_trigger`) — 5m AND 15m must both flip back to align with the 2h bias.

State machine (`pullback_entry.py::PullbackEntryEngine.on_bar`) tracks only `_bias` and `_pullback_seen` across bars; a bias change (including to neutral) invalidates any accumulated pullback state. Open ("first batch") and re-entry signals go through the exact same `on_bar` path — the *caller* (the backtest loop) decides the `open`/`reentry` label based on whether a batch has closed before, not the entry engine itself.

### Risk management

- **Stop-loss** (`stop_loss.py`): min(1.5×ATR stop, nearest 30m swing high/low), plus a 0.3% offset against the trade direction to avoid exact-tick stop-hunting.
- **Take-profit is two legs** (`take_profit.py`): leg A takes 50% off at a fixed 2R (`rr_trigger`), leg B trails the remaining 50% with a 3×ATR chandelier stop that only ever moves in the favorable direction and never goes worse than the original stop (`floor_stop`).
- **Cooldown** (`cooldown.py`) blocks *new* entries only — it never force-closes an existing open batch. Triggers on 3 consecutive pure stop-losses (protective stop-losses after a partial take-profit don't count — that's a different failure mode) or a single 2h bar whose body swallows ≥2 MAs. Releases only after 3 consecutive bars holding the same clean MA array (not just "non-neutral" — that's true the instant it triggers and would be a no-op release condition).
- Only one `_OpenBatch` may be open at a time per ticker; a batch's two legs must both exit (or the whole batch stop out) before a new signal is accepted.

### Portfolio layer (`engine/portfolio_pullback_backtest.py`)

Each ticker runs its own fully independent `PullbackEntryEngine`/`CooldownManager` instance — no cross-ticker coupling in entry/exit decisions. The portfolio layer only decides how much of each ticker's *own* daily % return counts, via `InverseVolatilitySizer` (`sizing.py`, weight ∝ 1/vol, renormalized) applied to `rolling_daily_atr_vol` (`vol_estimator.py`, ATR-as-%-of-close, `.shift(1)`'d), **recomputed every day** (daily rebalance, not buy-and-hold weights). A ticker with no open position on a given day still gets its full weight allocation — that idle capital just contributes 0 return that day, by design (not a bug — this was previously mis-flagged as a "dilution" bug and corrected).

Contribution-per-ticker-per-direction (`contribution_pct`) is computed via an exact daily-dollar decomposition (`weights_today(s) * daily_return_s(d) * equity_before`), not a naive sum of batch-level % PnL against compounded total return — the latter doesn't share a common denominator and can blow past ±100%. Stored as a plain fraction (matching `win_rate`'s convention) since the chart layer's `:.1%` format spec already multiplies by 100.

### Commodity tickers are ETF proxies, not futures

`SYMBOLS` in `fetch_alpaca.py` includes `USO`/`GLD`/`SLV` as stand-ins for WTI crude / COMEX gold / COMEX silver continuous futures — Alpaca has no futures data, so these ETFs are used as a liquidity/price proxy. If a real continuous-contract data source is ever wired in, expect a roll-gap adjustment step that doesn't exist anywhere in this codebase yet.
