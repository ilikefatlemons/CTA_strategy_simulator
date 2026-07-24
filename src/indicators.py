"""
v1.1: full-series MACD/ATR, vectorized over an entire OHLCV dataframe.

Same formulas as `rules.entry.MACDGoldenCross`/`rules.exit.atr` (kept as
those are validated v1.0 strategy code, not touched here), but computed once
over the whole series rather than re-evaluated bar-by-bar for a live signal
decision. Safe to precompute in one pass because both are strictly causal -
row j only ever depends on rows <= j, never a future row - so computing the
full series up front and later looking up row j is identical to having
computed it incrementally through row j.
"""

import pandas as pd


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    closes = df["close"]
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean()
