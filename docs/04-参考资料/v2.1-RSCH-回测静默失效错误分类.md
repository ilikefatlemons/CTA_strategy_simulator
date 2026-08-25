# Silent Failure Taxonomy — OHLC Pattern & Indicator Strategies

Scope: errors that leave the code running and the equity curve attractive.

---

## Stage 1 — Data Acquisition and Preparation

**Rank order (P × damage × detection difficulty): 1.1 Bar-label convention → 1.2 Back-adjustment distortion of candle geometry → 1.3 Vendor restatement / non-PIT history → 1.4 Survivorship & delisting truncation → 1.5 Stale and synthetic bars → 1.6 Continuous-contract splicing**

### 1.1 Bar timestamp / label convention mismatch
- **(a)** Open-labelled vs. close-labelled bars (look-ahead by timestamp convention).
- **(b)** Vendor stamps bar `t` with its opening time while your code treats the stamp as the closing time, so `close[t]` is treated as known one full bar before it exists.
- **(c)** Inflates. Severe: routinely converts a null rule into Sharpe 1.5–3, with a smooth, plausible curve.
- **(d)** Take a scheduled hard-timestamped event (CPI/FOMC release, session open). Locate the volume/range spike. Confirm the spike sits in the bar whose *label window* contains the event, not the one before.
- **(e)** [DOCUMENTED]

### 1.2 Split/dividend back-adjustment distorting candle geometry
- **(a)** Back-adjustment artefact / non-point-in-time price levels.
- **(b)** Adjustment rescales O, H, L, C off the traded tick grid, so exact equalities (`C == O`, `H == max(O,C)`) that define doji and marubozu vanish or appear, and gap tests fire on prices that never printed.
- **(c)** Bidirectional; changes pattern *counts* by tens of percent on high-dividend or low-priced names. Silently makes your tolerance parameter, not the market, the classifier.
- **(d)** Count exact-equality patterns in adjusted vs. raw unadjusted OHLC. If adjusted data contains near-zero exact doji, adjustment has erased the discrete structure you claim to trade.
- **(e)** [INFERRED] (back-adjustment bias itself is [DOCUMENTED])

### 1.3 Vendor restatement and silent backfill
- **(a)** Non-point-in-time (non-PIT) data.
- **(b)** Bad prints are retroactively corrected and late/adjusted bars rewritten, so the OHLC you test is a cleaned version that never existed live — and corrections cluster on outlier bars, which is where patterns fire.
- **(c)** Inflates modestly overall, heavily on the extreme-range bars that carry most signal.
- **(d)** Snapshot the identical date range twice, months apart. Diff bar-by-bar. Count changed OHLC values and check whether they concentrate on high-range days.
- **(e)** [DOCUMENTED]

### 1.4 Survivorship and delisting truncation
- **(a)** Survivorship bias / delisting-return omission.
- **(b)** Universe drawn from *current* constituents, so bars only exist for names that survived; bullish reversal patterns look predictive because terminal decliners were removed from the sample.
- **(c)** Inflates long-side hit rate; equity-anomaly literature puts the drag at roughly 1–4% p.a., larger for small caps.
- **(d)** Count distinct symbols whose last bar precedes the sample end. If that count is near zero, the dataset is survivorship-contaminated by construction.
- **(e)** [DOCUMENTED]

### 1.5 Stale, halted, and synthetic bars
- **(a)** Nonsynchronous trading / stale prices.
- **(b)** Illiquid or halted sessions carry the prior close forward, producing `O=H=L=C` or zero-volume bars that your classifier reads as legitimate doji/inside bars; the next bar's "reversion" is the stale quote catching up.
- **(c)** Strongly inflates apparent mean reversion. In a cross-sectional scan the best-performing names are frequently the stalest.
- **(d)** Tabulate the share of bars with zero volume or `H == L`. Then bucket strategy returns by prior-bar volume decile; a monotone edge concentrated in the bottom decile is staleness, not alpha.
- **(e)** [DOCUMENTED]

### 1.6 Continuous futures / perpetual contract splicing
- **(a)** Roll-adjustment artefact.
- **(b)** Back-adjusted series shift historical O/H/L/C by a constant at each roll, so the O–C relationship across the splice is synthetic and gap-based patterns fire on roll mechanics.
- **(c)** Inflates and injects spurious trend; long histories can produce negative or near-zero prices that break ratio indicators without erroring.
- **(d)** Flag roll dates. Compare pattern frequency and per-trade P&L within ±2 bars of a roll against the rest of the sample.
- **(e)** [DOCUMENTED]

---

## Stage 2 — Pattern / Indicator Definition and Signal Construction

**Rank order: 2.1 Confirmation-bar leakage → 2.2 Whole-sample normalisation → 2.3 Repainting indicators → 2.4 Context/trend filter leakage → 2.5 Threshold data-mining on a step function → 2.6 Warm-up and alignment leakage**

### 2.1 Confirmation-bar leakage into the pattern label
- **(a)** Look-ahead bias in pattern definition.
- **(b)** The pattern is defined as "engulfing *followed by* a confirming close," but the signal is indexed to bar `t` while the confirmation uses `close[t+1]`, so the trade is entered on information from its own outcome bar.
- **(c)** Inflates severely; hit rate approaches the confirmation criterion itself, often 65–80%.
- **(d)** Re-run with the entire signal series shifted forward one additional bar. If performance collapses rather than degrades, the definition was consuming the confirmation bar.
- **(e)** [DOCUMENTED]

### 2.2 Whole-sample normalisation of thresholds
- **(a)** Full-sample scaling / in-sample standardisation leakage.
- **(b)** Body size, range, or indicator values are z-scored or percentile-ranked over the *entire* series, so the classification of bar `t` depends on volatility realised after `t`.
- **(c)** Inflates. Concentrated in high-volatility regimes; typically adds 30–60% to reported returns without deforming the curve's shape.
- **(d)** Recompute all normalising statistics on expanding windows only, then diff the signal series bar-by-bar. Count and date the flipped classifications; they will cluster around volatility regime changes.
- **(e)** [DOCUMENTED]

### 2.3 Repainting / revising indicators
- **(a)** Repainting (ZigZag, fractals, pivot highs, displaced Ichimoku lines, centred filters).
- **(b)** The indicator's value at `t` is recomputed once bars `t+1 … t+k` arrive, so the historical series you backtest is the *final* revision and no live bar ever had that value.
- **(c)** Inflates catastrophically for pivot- and swing-based rules; equity curves look near-perfect but are unremarkable in aggregate statistics, which is why they pass review.
- **(d)** Recompute the indicator on truncated histories ending at each `t`, store the value, and compare that vector against the full-history version. Any mismatch is repainting.
- **(e)** [DOCUMENTED]

### 2.4 Trend/context filter leakage
- **(a)** Contaminated conditioning set.
- **(b)** Classical reversal patterns require a prior trend; if "downtrend" is defined by a centred moving average, a completed swing, or any window extending past `t`, the prior condition encodes the subsequent move.
- **(c)** Inflates; often the single largest contributor, since the filter is where most of the discrimination sits.
- **(d)** For every filter, print the maximum bar index it touches relative to the signal bar. Any value greater than `t` is disqualifying. Then re-run with the filter randomly permuted across time — if performance survives, the filter was noise; if it dies, verify the index window.
- **(e)** [DOCUMENTED]

### 2.5 Threshold data-mining on a discrete step function
- **(a)** Specification search / parameter overfitting under discrete classification.
- **(b)** Body-to-range and shadow ratio cutoffs are tuned, and because classification is a step function, a 1% cutoff change swaps whole cohorts of trades in and out — high sample variance masquerading as parameter sensitivity analysis.
- **(c)** Inflates. Reported P&L varies smoothly with the threshold even when the trade *sample* changes wholesale, giving false evidence of robustness.
- **(d)** Plot the number of matched patterns, not P&L, against the threshold. If the count moves by more than ~20% across the "robust plateau," the plateau is a plateau in aggregation, not in signal.
- **(e)** [INFERRED]

### 2.6 Warm-up and alignment leakage
- **(a)** Initialisation / index-alignment bias.
- **(b)** Recursive indicators seeded with full-sample means, `min_periods` permitting values before enough history exists, or a sign error in a shift, all embed future information into the earliest bars.
- **(c)** Inflates, usually small and confined to the sample's start — which is exactly where a walk-forward's first in-sample fold sits.
- **(d)** Drop the first N bars (N = 10× the longest lookback) and re-run. A meaningful change indicates warm-up contamination rather than sample-size effects.
- **(e)** [DOCUMENTED]

---

## Stage 3 — Backtest Engine Mechanics

**Rank order: 3.1 Intrabar path ambiguity → 3.2 Same-bar signal-and-fill → 3.3 Bid-ask bounce in close-to-close returns → 3.4 Touch-implies-fill for stops and limits → 3.5 Signal-conditional cost misspecification → 3.6 Capital and concurrency accounting**

### 3.1 Intrabar path ambiguity
- **(a)** Intrabar path / stop-target resolution ambiguity.
- **(b)** When both stop and target lie inside `[L[t], H[t]]`, the engine must assume an ordering; the default assumption (target first, or "no stop hit if close is favourable") is optimistic and unverifiable from OHLC alone.
- **(c)** Inflates. For tight stops on daily bars this is frequently the *entire* reported edge; win rate can shift 10–20 points.
- **(d)** Re-run under the pessimistic convention (stop always first when both are inside the bar). Report the spread between optimistic and pessimistic P&L. If the strategy is only viable under one, you have no result.
- **(e)** [DOCUMENTED]

### 3.2 Same-bar signal-and-fill
- **(a)** Zero-latency close execution.
- **(b)** Signal computed from `close[t]` is filled at `close[t]`, requiring you to transact at a price that is only known once transacting is no longer possible.
- **(c)** Inflates. Magnitude scales with bar-level autocorrelation; on daily bars typically 20–50% of returns, far more intraday.
- **(d)** Build a lag-sensitivity curve: performance at execution lags 0, 1, 2, 3 bars. A cliff between lag 0 and lag 1 means the edge lives inside the unobservable instant, not in the market.
- **(e)** [DOCUMENTED]

### 3.3 Bid-ask bounce in close-to-close returns
- **(a)** Bid-ask bounce bias (Blume–Stambaugh).
- **(b)** Closing prints alternate between bid and ask, inducing negative serial correlation in `close[t]/close[t-1]`; any pattern conditioning on a down-close then measuring the next close harvests spread, not price.
- **(c)** Inflates reversal strategies; the bias is order (spread/2)² in variance terms and rises sharply as price level falls.
- **(d)** Estimate the effective spread from the OHLC series itself (Corwin–Schultz high–low estimator) and sort assets by it. If per-asset edge ranks with estimated spread, you are trading the bounce.
- **(e)** [DOCUMENTED]

### 3.4 Touch-implies-fill for stops and limits
- **(a)** Optimistic fill / no queue position; gap-through fills.
- **(b)** The engine fills a limit whenever `L[t] ≤ price ≤ H[t]`, ignoring that a single touch of the extreme may not reach your queue position; and fills stops *at* the stop level even when `open[t]` gapped through it.
- **(c)** Inflates. Limit-based entries are hit hardest — precisely the fills you never get are the ones that would have been profitable.
- **(d)** Require the level to be exceeded by k ticks rather than touched, and fill gapped stops at `open[t]`. Compare. Separately, report the fraction of fills that occurred at the exact bar extreme; anything above a few percent is fiction.
- **(e)** [DOCUMENTED]

### 3.5 Signal-conditional cost misspecification
- **(a)** Endogenous transaction cost.
- **(b)** A flat cost per trade is applied, but patterns defined by large ranges, gaps, or engulfing bodies fire disproportionately on the highest-volatility bars, where realised spread and slippage are widest.
- **(c)** Inflates. A flat 5 bp assumption can understate true cost by 2–5× on exactly the bars that generate signals.
- **(d)** Bucket trades by the signal bar's range/ATR. Confirm cost assumptions were constant across buckets, then re-run with cost scaled to the signal bar's own range and compare.
- **(e)** [INFERRED]

### 3.6 Capital and concurrency accounting
- **(a)** Unconstrained sizing / phantom leverage.
- **(b)** Multiple simultaneous signals each sized against full notional equity, with intrabar compounding on equity that isn't yet realised, so effective leverage silently exceeds 1 without triggering any margin logic.
- **(c)** Inflates return more than volatility, so Sharpe rises. Magnitude tracks the peak number of concurrent positions.
- **(d)** Log the time series of gross exposure divided by equity. Report its maximum and the distribution of concurrent open trades. If max gross exposure exceeds your intended limit at any bar, the sizing logic is not binding.
- **(e)** [DOCUMENTED]

---

## Stage 4 — Performance Evaluation and Statistical Inference

**Rank order: 4.1 Multiple-testing inflation → 4.2 Trade overlap and clustering in the t-statistic → 4.3 Consumed out-of-sample → 4.4 Cross-sectional dependence of trials → 4.5 Sharpe blind to trade-return skew → 4.6 Uncontrolled directional exposure**

### 4.1 Multiple-testing / data-snooping inflation
- **(a)** Data-snooping bias; backtest overfitting.
- **(b)** Every pattern × parameter × asset × holding-period combination examined — including abandoned ones — is a trial, and the maximum Sharpe over N noise trials grows roughly with √(2 ln N).
- **(c)** Inflates. With N ≈ 1,000, an expected maximum Sharpe near 1.0 arises from pure noise on a decade of daily data.
- **(d)** State N explicitly, counting discarded variants. Compute the Deflated Sharpe Ratio or the Harvey–Liu haircut. If you cannot reconstruct N, treat the result as untested.
- **(e)** [DOCUMENTED]

### 4.2 Trade overlap and clustering
- **(a)** Non-IID returns / overlapping outcomes; serial correlation in the Sharpe ratio.
- **(b)** Multi-bar holds share outcome bars, and pattern signals cluster in volatile regimes, so effective sample size is far below trade count while √252 annualisation assumes independence.
- **(c)** Inflates the t-statistic and annualised Sharpe, commonly by 1.3–2×.
- **(d)** Compute average label uniqueness (mean fraction of a trade's bars not shared with another trade). Re-estimate significance with Newey–West standard errors and a stationary block bootstrap; compare with the naive t-statistic.
- **(e)** [DOCUMENTED]

### 4.3 Consumed out-of-sample
- **(a)** Hold-out contamination / iterated walk-forward.
- **(b)** The reserved segment is examined, the rule is revised, and the segment is re-examined; after the first look it is in-sample, and walk-forward repeated over many configurations is a search, not a validation.
- **(c)** Inflates. Empirically the dominant reason strategies that pass validation still fail live.
- **(d)** Keep a dated log of every out-of-sample evaluation. If the count exceeds one per structural revision, compute Probability of Backtest Overfitting via CSCV rather than relying on the hold-out result.
- **(e)** [DOCUMENTED]

### 4.4 Cross-sectional dependence of trials
- **(a)** Correlated trials / illusory breadth.
- **(b)** The same rule tested on 500 equities is not 500 independent tests, because the assets share a common factor and the pattern fires on the same days across the panel.
- **(c)** Inflates significance and understates drawdown severity; effective independent trials may be under 10.
- **(d)** Build the per-asset daily return streams and compute average pairwise correlation ρ. Effective breadth ≈ N / (1 + (N−1)ρ). Report it beside the naive N.
- **(e)** [DOCUMENTED]

### 4.5 Sharpe blind to trade-return skew
- **(a)** Higher-moment blindness of the Sharpe ratio.
- **(b)** Fixed-target/stop pattern rules generate many small wins and rare large losses (or the inverse), a shape the second moment cannot represent.
- **(c)** Overstates attractiveness on negatively-skewed profiles; understates it on positively-skewed trend-following. Sharpe is uninformative when a handful of trades dominate.
- **(d)** Report skewness and kurtosis of trade returns, plus the fraction of total P&L contributed by the top 1% and top 5 trades. If five trades exceed half the P&L, the reported Sharpe carries no information.
- **(e)** [DOCUMENTED]

### 4.6 Uncontrolled directional exposure
- **(a)** Benchmark contamination / unhedged beta.
- **(b)** Bullish patterns fire more often in uptrends, so the strategy holds net long exposure and inherits the sample period's drift while being presented as a pattern edge.
- **(c)** Inflates in any bull sample. Frequently the whole result: alpha vanishes against buy-and-hold.
- **(d)** Regress daily strategy returns on the underlying's daily returns. Report intercept, slope, and average net exposure. Separately re-run with entry timing randomised but exposure profile preserved.
- **(e)** [DOCUMENTED]

---

## Stage 5 — Backtest-to-Live Divergence

**Rank order: 5.1 Print availability at the decision price → 5.2 Alpha decay and non-stationarity → 5.3 Feed and bar-construction mismatch → 5.4 Capacity and market impact → 5.5 Venue-level constraints → 5.6 Operational state divergence**

### 5.1 Print availability at the decision price
- **(a)** Signal-to-execution latency; official close vs. tradable close.
- **(b)** A daily rule needs `close[t]`, but the official close is a settlement or auction price published after the auction, so live entry occurs at the next open — across a gap that the backtest never crossed.
- **(c)** Inflates. On daily equity bars, a large share of next-day return arrives in the overnight gap, which the backtest silently captures.
- **(d)** Decompose historical P&L into the close-to-open and open-to-close components of the holding period. If the majority sits in the overnight gap, the live version cannot access it.
- **(e)** [DOCUMENTED]

### 5.2 Alpha decay and non-stationarity
- **(a)** Structural break / crowding.
- **(b)** Candlestick geometry depends on microstructure — tick size, decimalisation, auction rules — so a 25-year daily series contains bars generated by materially different price grids.
- **(c)** Inflates the full-sample result via a strong early sub-period; the recent sub-period is usually the honest estimate.
- **(d)** Split the sample into four chronological quartiles and report Sharpe, trade count, and average pattern frequency per quartile. A monotone decline in *frequency* indicates a microstructure break, not merely a return break.
- **(e)** [DOCUMENTED]

### 5.3 Feed and bar-construction mismatch
- **(a)** Vendor divergence in H/L construction.
- **(b)** Your live feed builds bars from a different venue set, with different outlier filtering and session boundaries, so the highs and lows that define your patterns differ from the backtest vendor's.
- **(c)** Directionally random but large in dispersion: pattern classification is a threshold decision, so small H/L differences flip discrete labels.
- **(d)** Pull the same period from a second independent vendor. Diff H and L per bar, then count how many pattern classifications flip. A flip rate above a few percent means your signal set is vendor-specific.
- **(e)** [INFERRED]

### 5.4 Capacity and market impact
- **(a)** Market impact / capacity constraint.
- **(b)** Backtest fills the full intended size at one price inside `[L[t], H[t]]`, while live execution consumes depth and moves the price, particularly on the wide-range bars that trigger patterns.
- **(c)** Inflates, growing roughly with the square root of participation rate.
- **(d)** Compute intended order size as a fraction of the signal bar's volume. Above roughly 1–2%, apply a square-root impact model and re-run. Report the size at which the edge reaches zero.
- **(e)** [DOCUMENTED]

### 5.5 Venue-level constraints
- **(a)** Short-sale, tick, and halt constraints.
- **(b)** Bearish patterns require shorting names that may be hard-to-borrow, and both directions assume tradability during limit-up/limit-down, halts, or after gaps that skip your entry level.
- **(c)** Inflates the short book asymmetrically; the least borrowable names carry the largest apparent edge.
- **(d)** Split reported P&L by side. If short-side contribution is disproportionate, cross-check the signal dates against halt and limit-state records for those symbols.
- **(e)** [DOCUMENTED]

### 5.6 Operational state divergence
- **(a)** State drift between simulator and live process.
- **(b)** The backtest holds full position state across every bar, while the live process restarts, misses bars, reconciles differently, or accumulates rounding in position size.
- **(c)** Inflates modestly but with high variance; typically shows up as a small persistent tracking error that compounds.
- **(d)** Run the live engine in paper mode on the same period and replay the backtest bar-for-bar. Diff the two order sequences. Any non-empty diff is a specification gap, not a rounding issue.
- **(e)** [INFERRED]

---

## Errors Most Often ABSENT From Public Tutorials

1. Bid-ask bounce as a manufactured source of reversal edge (3.3).
2. Intrabar path ambiguity when stop and target both sit inside one bar (3.1).
3. Back-adjustment destroying the exact-equality structure that defines doji and marubozu (1.2).
4. Trade overlap collapsing effective sample size in the t-statistic (4.2).
5. Cross-sectional dependence making a 500-stock panel far fewer than 500 tests (4.4).
6. Non-point-in-time vendor restatement concentrated on outlier bars (1.3).
7. Repainting as a *systematic* class rather than a quirk of ZigZag (2.3).

## Errors a Developer Is Least Likely to Think to Ask About

1. Whether the vendor's bar timestamp is the open or the close (1.1).
2. That threshold tuning on a discrete classifier changes the *sample*, not just the parameter, so a smooth P&L plateau is not evidence of robustness (2.5).
3. That the trend-context filter, not the pattern, carries almost all of the discrimination and almost all of the leakage risk (2.4).
4. That the sample of firing patterns is endogenous to volatility, so costs, spreads, and fill quality are all conditional on the signal (3.5).
5. That the out-of-sample segment was consumed on the first look (4.3).
6. That the official close is often not a tradable price (5.1).

## Where Conventional Wisdom Is Wrong or Overstated

**"Survivorship bias is the first thing to check."** Correct for cross-sectional equity scans, irrelevant for single-instrument index, FX, crypto, or futures work — which is most candlestick development. It dominates tutorial discussion because it is easy to explain, not because it is usually binding here.

**"Walk-forward analysis protects against overfitting."** Walk-forward run repeatedly across configurations is a search procedure with extra steps. CSCV work shows overfit strategies survive walk-forward routinely. The protection comes from counting trials, not from the folding scheme.

**"A conservative flat cost assumption is safe."** Flat costs are not conservative for pattern strategies, because signals concentrate on wide-range, wide-spread bars. The correct treatment is cost conditional on the signal bar's own range.

**"Look-ahead bias is obvious because results become absurd."** False, and it is the reason half-bar leakage persists. Leakage of one bar produces Sharpe 1.5–2.5 and a curve indistinguishable from a good strategy. Absurdity only appears with gross multi-bar leakage.

**"Longer history is better."** Beyond roughly two decades, daily bars come from different tick-size and auction regimes; the candle geometry itself is not comparable. More data buys statistical power in exchange for pooling incommensurable microstructures.

**"Candlestick patterns have no edge."** Overstated in the opposite direction. Marshall, Young and Rose find no value for DJIA daily bars after bootstrap correction, and that is the most careful test available — but it is a statement about a specific market, frequency, and horizon, not a proof that OHLC geometry is information-free. Lo, Mamaysky and Wang find statistically detectable information in chart patterns that is nonetheless small relative to trading costs. The defensible claim is that the effect size is below friction at daily frequency in liquid large caps.

---

## References

**Overfitting and multiple testing**
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio Management, 40(5), 94–107. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2016). *The Probability of Backtest Overfitting.* Journal of Computational Finance, 20(4). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.* Notices of the AMS, 61(5), 458–471. http://www.ams.org/notices/201405/rnoti-p458.pdf
- Harvey, C. R., & Liu, Y. (2015). *Backtesting.* Journal of Portfolio Management, 42(1), 13–28. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489 (code: https://people.duke.edu/~charvey/backtesting/)
- Sullivan, R., Timmermann, A., & White, H. (1999). *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap.* Journal of Finance, 54(5), 1647–1691.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica, 68(5), 1097–1126.
- Aronson, D. (2006). *Evidence-Based Technical Analysis.* Wiley. — data-mining bias applied specifically to chart rules.

**Statistical inference on returns**
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal, 58(4), 36–52. — annualisation under serial correlation.
- Newey, W. K., & West, K. D. (1987). *A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. — label uniqueness, overlapping outcomes, purged cross-validation, triple-barrier labelling.

**Microstructure and data artefacts**
- Blume, M. E., & Stambaugh, R. F. (1983). *Biases in Computed Returns: An Application to the Size Effect.* Journal of Financial Economics, 12(3), 387–404. — bid-ask bounce.
- Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market.* Journal of Finance, 39(4), 1127–1139.
- Corwin, S. A., & Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices.* Journal of Finance, 67(2), 719–760. https://doi.org/10.1111/j.1540-6261.2012.01729.x — the detection test in 3.3.
- Brown, S. J., Goetzmann, W. N., & Ross, S. A. (1995). *Survival.* Journal of Finance, 50(3), 853–873.
- Shumway, T. (1997). *The Delisting Bias in CRSP Data.* Journal of Finance, 52(1), 327–340.
- Almgren, R., & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk, 3(2), 5–39. — impact model for 5.4.

**Candlestick and technical-pattern evidence**
- Marshall, B. R., Young, M. R., & Rose, L. C. (2006). *Candlestick Technical Trading Strategies: Can They Create Value for Investors?* Journal of Banking & Finance, 30(8), 2303–2323. https://doi.org/10.1016/j.jbankfin.2005.08.001
- Marshall, B. R., Young, M. R., & Cahan, R. (2008). *Are Candlestick Technical Trading Strategies Profitable in the Japanese Equity Market?* Review of Quantitative Finance and Accounting, 31(2), 191–207. https://doi.org/10.1007/s11156-007-0068-1
- Lo, A. W., Mamaysky, H., & Wang, J. (2000). *Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation.* Journal of Finance, 55(4), 1705–1765. https://www.nber.org/papers/w7613
- Ready, M. J. (2002). *Profits from Technical Trading Rules.* Financial Management, 31(3), 43–61.
