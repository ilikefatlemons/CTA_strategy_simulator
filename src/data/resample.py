"""
v1.1 Phase A: synthesize 15m/30m/1h candles from the 5m bars.

5m is the finest granularity we fetch, and every coarser timeframe here is an
exact integer multiple of it (15m=3 bars, 30m=6 bars, 1h=12 bars), so there's
no need to hit Alpaca again - resampling the existing 5m CSVs is sufficient
and guarantees every timeframe is perfectly aligned to the same underlying
bars (no separate-fetch timestamp drift between timeframes).

This module only produces the *complete historical* coarser series - it does
not decide what a live 5m bar in the backtest loop is allowed to see. That
lookahead guard is Phase B (`latest_closed_bar`). Resampling itself carries
no lookahead risk since it's just re-grouping bars that already happened.
"""

import pandas as pd

RULES = {"15m": "15min", "30m": "30min", "1h": "1h"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate 5m bars up to a coarser timeframe.

    Resampling is done one trading session (one ET calendar date) at a time,
    with each day's bins anchored to that day's first bar (09:30 ET) rather
    than to midnight. Two reasons:
      - Anchoring per-day stops a bin from ever spanning the overnight gap
        between sessions (a naive continuous resample would otherwise try to
        bridge e.g. 15:55 one day and 09:30 the next into one candle).
      - 390 minutes/session doesn't divide evenly into 60 (6.5h), so a
        midnight-anchored '1h' resample would misalign every session's
        hourly bins against the actual 09:30 open (first bin labeled 09:00
        but only holding 30 minutes of real data). Anchoring to 09:30 keeps
        every non-final bin a genuine full-length bar; only the last bin of
        the day is legitimately shorter (e.g. 15:30-16:00 for '1h').
    """
    if "timestamp" not in df.columns:
        raise ValueError("expected a 'timestamp' column")

    df = df.sort_values("timestamp")
    et_dates = df["timestamp"].dt.tz_convert("America/New_York").dt.date

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    extra_cols = [c for c in ("symbol", "sector") if c in df.columns]

    sessions = []
    for _, day_df in df.groupby(et_dates):
        day_df = day_df.set_index("timestamp")
        # pandas' bundled type stubs (no pandas-stubs installed) don't model
        # resample().agg(dict) correctly, though it's valid at runtime.
        session = day_df.resample(rule, origin="start", label="left", closed="left").agg(agg)  # type: ignore[arg-type]
        # Bins beyond the last bar of the day (or any gap) carry no source
        # rows and come back all-NaN - drop them rather than synthesizing a
        # candle out of nothing.
        session = session.dropna(subset=["open"])
        for col in extra_cols:
            session[col] = day_df[col].iloc[0]
        sessions.append(session)

    out = pd.concat(sessions).reset_index()
    cols = ["timestamp", *extra_cols, "open", "high", "low", "close", "volume"]
    return out[cols]


def resample_all_timeframes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """5m passthrough plus every coarser timeframe in `RULES`."""
    return {"5m": df, **{tf: resample_ohlcv(df, rule) for tf, rule in RULES.items()}}


def closed_bar_positions(higher_tf_df: pd.DataFrame, rule: str, open_times):
    """
    The lookahead guard's boundary math, shared by the single-bar accessor
    below and the vectorized indicator join in `higher_tf_indicators.py` so
    the two can't silently drift apart on the one rule that matters here.

    A bar labeled with open time `t` (label="left" in `resample_ohlcv`) spans
    [t, t + rule) and only closes at t + rule. So a bar is usable exactly
    when `t + rule <= open_time` - using `<` here would wrongly withhold a
    bar for one extra 5m tick after it has genuinely closed; comparing
    `t <= open_time` instead (no shift) would leak the still-forming bar for
    the entire time it's open.

    `open_times` may be a single Timestamp or an array-like of them -
    `Series.searchsorted` supports both, returning a scalar or an array of
    positions respectively (-1 meaning nothing has closed yet). Since
    `higher_tf_df` is sorted ascending (by construction in
    `resample_ohlcv`), this is O(log n) per lookup rather than a full rescan.
    """
    close_times = higher_tf_df["timestamp"] + pd.Timedelta(rule)
    return close_times.searchsorted(open_times, side="right") - 1


def latest_closed_bar(
    higher_tf_df: pd.DataFrame, rule: str, current_bar_open_time: pd.Timestamp
) -> pd.Series | None:
    """
    Return the most recent `higher_tf_df` row that has fully closed as of
    `current_bar_open_time` - the open time of the 5m bar currently being
    evaluated in the backtest loop - or None if no higher-timeframe bar has
    closed yet (e.g. the first few 5m bars of the whole series). See
    `closed_bar_positions` for the boundary rule itself.
    """
    idx = closed_bar_positions(higher_tf_df, rule, current_bar_open_time)
    if idx < 0:
        return None
    return higher_tf_df.iloc[idx]
