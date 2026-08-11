"""
Phase 4: Sharpe ratio, computed from daily equity snapshots.

Bar-level (5m) returns are noisy and autocorrelated (open positions carry
unrealized P&L across bars within the same trade), so the ratio is computed
on day-over-day returns of the equity curve and annualized with sqrt(252),
not on raw bar returns.
"""

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Pct-change of each day's last equity snapshot."""
    index = pd.DatetimeIndex(equity_curve.index)
    daily_equity = equity_curve.groupby(index.date).last()
    return daily_equity.pct_change().dropna()


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio from an equity curve.

    risk_free_rate is an annual rate, converted to a daily rate before being
    subtracted from daily returns.
    """
    returns = daily_returns(equity_curve)
    if len(returns) < 2:
        return float("nan")

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    if excess.std() == 0:
        return float("nan")

    return (excess.mean() / excess.std()) * (TRADING_DAYS_PER_YEAR ** 0.5)


# ---------------------------------------------------------------------------
# China futures variant (v3.0). Kept separate rather than parameterizing the
# functions above, because v2.1 depends on their exact behaviour.
# ---------------------------------------------------------------------------

# Measured, not assumed: RB has 4,017 distinct trading_dates over
# 2010-01-04..2026-07-29 (16.57 years) -> 242.4/yr. CFFEX and the commodity
# exchanges share the same calendar.
TRADING_DAYS_PER_YEAR_CN = 243


def session_returns(equity_curve: pd.Series, trading_date: pd.Series) -> pd.Series:
    """
    Day-over-day returns grouped by `trading_date`, NOT by calendar date.

    Grouping by calendar date is wrong for China futures. A trading day starts
    with the previous evening's 21:00 night session, so night bars carry a
    calendar date one day BEFORE their trading_date, and Friday-night bars
    belong to Monday. `index.date` grouping would split one trading day in two
    and scramble the weekend ordering.

    `trading_date` must be aligned to `equity_curve` (same length, same order).
    """
    if len(equity_curve) != len(trading_date):
        raise ValueError("equity_curve and trading_date must be the same length")
    daily = equity_curve.groupby(trading_date.to_numpy()).last()
    return daily.pct_change().dropna()


def sharpe_ratio_cn(
    equity_curve: pd.Series,
    trading_date: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR_CN,
) -> float:
    """
    Annualized Sharpe for a China-futures equity curve.

    Same shape as `sharpe_ratio`, with two differences: returns are grouped by
    trading day (see `session_returns`), and the annualization uses the
    measured 243 sessions/year rather than the US 252. The annualization
    change is small (sqrt(243/252) = 0.982); the grouping change is not.

    Flat days must be present in the curve so they contribute zero-return days
    - `run_rbreaker_backtest` writes `capital` on every bar, including while
    flat, so this holds.
    """
    returns = session_returns(equity_curve, trading_date)
    if len(returns) < 2:
        return float("nan")

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    if excess.std() == 0:
        return float("nan")

    return (excess.mean() / excess.std()) * (periods_per_year ** 0.5)
