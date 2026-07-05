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
