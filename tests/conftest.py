"""
Shared synthetic-frame builders for the v3.0 R-Breaker tests.

Every test here builds its bars by hand rather than reading parquet, so a
failure points at the logic and not at the data. The builders mirror the real
schema exactly: `datetime` index (tz-naive Beijing, bar stamped at the minute
END), `trading_date` as the authoritative session key, and night bars carrying
the NEXT trading day's `trading_date`.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import DEFAULT_SESSION_MODE, apply_session_mode, derive_sessions  # noqa: F401
from src.strategy.rbreaker import LINE_NAMES, RBreakerConfig, RBreakerParams, rbreaker_lines


def _raw_frame(bars) -> pd.DataFrame:
    """
    bars: iterable of (time, trading_date, open, high, low, close, volume).
    `time`/`trading_date` are anything pd.Timestamp accepts.
    """
    bars = list(bars)
    idx = pd.DatetimeIndex([pd.Timestamp(b[0]) for b in bars], name="datetime")
    return pd.DataFrame(
        {
            "open": [float(b[2]) for b in bars],
            "high": [float(b[3]) for b in bars],
            "low": [float(b[4]) for b in bars],
            "close": [float(b[5]) for b in bars],
            "volume": [float(b[6]) for b in bars],
            "K": 1.0,
            "trading_date": [pd.Timestamp(b[1]) for b in bars],
        },
        index=idx,
    )


def _run(start, n, trading_date, price=100.0, volume=10.0, freq="1min"):
    """n consecutive bars, flat OHLC at `price`. The boring filler."""
    times = pd.date_range(pd.Timestamp(start), periods=n, freq=freq)
    return [(t, trading_date, price, price, price, price, volume) for t in times]


@pytest.fixture
def raw_frame():
    return _raw_frame


@pytest.fixture
def bar_run():
    return _run


def _make_prepared(bars, refs, params=None, config=None, symbol="TEST",
                   mode=DEFAULT_SESSION_MODE):
    """
    Build a `Prepared` straight from explicit bars + explicit reference-day
    OHLC, bypassing `prepare()`. Lets a test pin an exact price path against
    exact lines, which reading real parquet never could.

    refs: {trading_date_str: (H, L, C) or None}. None -> that day is invalid
    (no reference), so the engine must not trade it.
    mode: any key of `SESSION_MODES`; defaults to the historical "split".
    """
    from src.data.v3_sessions import Prepared

    params = params or RBreakerParams()
    config = config or RBreakerConfig()
    df = apply_session_mode(_raw_frame(bars), mode)

    days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in refs), name="trading_date")
    rows = []
    for d in days:
        ref = refs[[k for k in refs if pd.Timestamp(k) == d][0]]
        if ref is None:
            rows.append({**{n: np.nan for n in LINE_NAMES},
                         "ref_H": np.nan, "ref_L": np.nan, "ref_C": np.nan,
                         "bars": 0, "valid": False})
        else:
            H, L, C = ref
            rows.append({**rbreaker_lines(H, L, C, params),
                         "ref_H": H, "ref_L": L, "ref_C": C,
                         "bars": 100, "valid": True})
    lines = pd.DataFrame(rows, index=days)

    raw_idx = lines.index.get_indexer(df["trading_date"]).astype(np.int32)
    assert (raw_idx >= 0).all(), "a bar's trading_date is missing from refs"
    # pin the line index per session, exactly as `v3_sessions.apply_mode` does
    sid = df["session_id"].to_numpy()
    first_pos = np.flatnonzero(df["is_session_first"].to_numpy())
    line_idx = raw_idx[first_pos][sid - 1]

    keep = ["open", "high", "low", "close", "volume", "trading_date",
            "session_id", "is_session_first", "is_session_last", "tradable"]
    return Prepared(symbol=symbol, df=df[keep], lines=lines, line_idx=line_idx,
                    k0=1.0, params=params, config=config, mode=mode,
                    dropped_runs=pd.DataFrame())


@pytest.fixture
def make_prepared():
    return _make_prepared
