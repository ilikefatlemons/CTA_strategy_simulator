"""Phase 3/6: rolling daily realized volatility, used to feed InverseVolatilitySizer."""

import pandas as pd

from src.indicators import atr as atr_series


def daily_closes(df: pd.DataFrame) -> pd.Series:
    """Last close of each calendar date, indexed by date."""
    dates = df["timestamp"].dt.date
    return df.groupby(dates)["close"].last()


def _daily_ohlc(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 5m bars to one daily OHLC row per calendar date. Every 5m bar
    already belongs to exactly one session, so this is a plain groupby (no
    need for resample_ohlcv's per-session-anchored resample, which exists to
    handle sub-daily bin boundaries within a session - here the whole session
    IS the bin).
    """
    dates = df_5m["timestamp"].dt.date
    daily = df_5m.groupby(dates).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    return daily


def rolling_daily_vol(closes: pd.Series, window: int = 20) -> pd.Series:
    """
    Std of daily pct-change returns over a trailing window, shifted by 1 day so
    day D's estimate only uses data through D-1 close (no lookahead).
    """
    daily_returns = closes.pct_change()
    vol = daily_returns.rolling(window).std()
    return vol.shift(1)


def rolling_daily_atr_vol(df_5m: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR-based vol estimate for the pullback strategy's inverse-vol sizer -
    resamples 5m bars up to daily OHLC, runs the same ATR used for the
    strategy's own stop-loss/take-profit, then expresses it as %-of-close
    (ATR / close) rather than raw price units. Raw ATR isn't comparable
    across symbols priced wildly differently (e.g. NVDA ~$150 vs XOM ~$110) -
    dividing by close puts every symbol's vol estimate on the same
    dimensionless scale, matching rolling_daily_vol's pct-return basis so the
    two are drop-in interchangeable for InverseVolatilitySizer.

    Shifted by 1 day so day D's weight only uses data through D-1's close,
    same no-lookahead convention as rolling_daily_vol.
    """
    daily = _daily_ohlc(df_5m)
    pct_atr = atr_series(daily, window) / daily["close"]
    return pct_atr.shift(1)
