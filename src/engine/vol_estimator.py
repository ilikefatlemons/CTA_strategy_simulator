"""Phase 3: rolling daily realized volatility, used to feed InverseVolatilitySizer."""

import pandas as pd


def daily_closes(df: pd.DataFrame) -> pd.Series:
    """Last close of each calendar date, indexed by date."""
    dates = df["timestamp"].dt.date
    return df.groupby(dates)["close"].last()


def rolling_daily_vol(closes: pd.Series, window: int = 20) -> pd.Series:
    """
    Std of daily pct-change returns over a trailing window, shifted by 1 day so
    day D's estimate only uses data through D-1 close (no lookahead).
    """
    daily_returns = closes.pct_change()
    vol = daily_returns.rolling(window).std()
    return vol.shift(1)
