"""
Trade-level statistics: average win/loss, payoff ratio, max drawdown.

Extracted from `src/engine/portfolio_pullback_backtest.py` so the v3.0 China
futures stack can reuse the exact definitions without importing the whole v2.1
engine (that module pulls in `src.config`, `src.rules.*`, `src.data.resample`).
The originals keep their private names there as aliases, so no v2.1 call site
changed. Same move, same reason as `src/viz/chart.py:392-394` -> `legend_patch`.

These are definitions, not just formulas - the conventions matter and are shared
across every strategy in the repo:

  * The loss bucket is `p <= 0`, not `p < 0`. A flat trade counts as a loss for
    the average and does NOT count as a win for the rate, so win rate and payoff
    ratio stay consistent with each other.
  * Returns are plain fractions. Formatting layers apply the `%`.
  * Max drawdown is a NEGATIVE fraction. Display layers take abs().
"""

from __future__ import annotations

import pandas as pd


def win_loss_avgs(pnls: list[float]) -> tuple[float, float]:
    """
    (avg winning trade, avg losing trade), each as a fraction of capital.

    These are what turn a payoff ratio into money: an expectancy of -0.31R says
    "loses 0.31 average-losses per trade", which is only -0.16% of capital if
    the average loss is 0.52%. The ratio alone doesn't tell you the size.
    """
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return (
        sum(wins) / len(wins) if wins else float("nan"),
        sum(losses) / len(losses) if losses else float("nan"),
    )


def payoff(pnls: list[float]) -> float:
    """avg winning trade / abs(avg losing trade); nan unless both sides exist."""
    avg_win, avg_loss = win_loss_avgs(pnls)
    if pd.isna(avg_win) or pd.isna(avg_loss) or avg_loss == 0:
        return float("nan")
    return avg_win / abs(avg_loss)


def win_rate(pnls: list[float]) -> float:
    """Share of trades with a strictly positive result. nan on an empty list."""
    if not pnls:
        return float("nan")
    return sum(1 for p in pnls if p > 0) / len(pnls)


def max_drawdown(curve: pd.Series) -> float:
    """
    Deepest peak-to-trough decline of an equity curve, as a NEGATIVE fraction.

    Must be measured on the combined curve, never as max() of per-component
    drawdowns - components trough at different times, so the aggregate is
    always shallower than the worst individual one.
    """
    return float((curve / curve.cummax() - 1.0).min()) if len(curve) else float("nan")
