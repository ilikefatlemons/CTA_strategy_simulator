"""
Session derivation and data hygiene (`src/data/v3_sessions.py`).

The 3h split threshold is the one hardcoded constant in the whole v3.0 stack,
and the docs forbid hardcoding session shapes. These tests pin the cases the
threshold was chosen against, measured over 24 symbols and every night tier:
the largest intra-trading_date gap is 2h01m (commodity lunch) and the smallest
night->day gap is 6h31m (AU 02:30 -> 09:01), leaving (2.02h, 6.52h) empty.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import (
    assert_no_ambiguous_gaps,
    contiguous_runs,
    daily_ohlc,
    derive_sessions,
    drop_dead_runs,
    prev_day_lines,
)
from src.strategy.rbreaker import RBreakerConfig, RBreakerParams


def _sessions(frame):
    out = derive_sessions(frame)
    return out, int(out["session_id"].nunique())


def test_night_and_day_split_into_two_sessions(raw_frame, bar_run):
    """
    A night-session commodity: 21:00-23:00 the previous calendar evening, then
    09:01-10:15 | 10:31-11:30 | 13:31-15:00 the next day - all one
    `trading_date`. The 10h gap (23:00 -> 09:01) splits; the 15m and 2h breaks
    inside the day do not.
    """
    td = "2025-07-02"
    bars = (
        bar_run("2025-07-01 21:00", 121, td)      # night
        + bar_run("2025-07-02 09:01", 75, td)     # day 1
        + bar_run("2025-07-02 10:31", 60, td)     # day 2 (after the 15m break)
        + bar_run("2025-07-02 13:31", 90, td)     # day 3 (after lunch)
    )
    out, n = _sessions(raw_frame(bars))
    assert n == 2, "night and day must be two sessions, the intraday breaks must not split"
    last = out.index[out["is_session_last"]]
    assert list(last) == [pd.Timestamp("2025-07-01 23:00"), pd.Timestamp("2025-07-02 15:00")]
    first = out.index[out["is_session_first"]]
    assert list(first) == [pd.Timestamp("2025-07-01 21:00"), pd.Timestamp("2025-07-02 09:01")]


def test_commodity_without_night_is_one_session(raw_frame, bar_run):
    """
    31 of the 59 symbols have no night session, and Feb-May 2020 had none
    market-wide. Three day runs, max gap 2h01m < 3h -> a single session.
    """
    td = "2025-07-02"
    bars = (bar_run("2025-07-02 09:00", 76, td)
            + bar_run("2025-07-02 10:31", 60, td)
            + bar_run("2025-07-02 13:31", 90, td))
    _out, n = _sessions(raw_frame(bars))
    assert n == 1


def test_cffex_lunch_break_does_not_split(raw_frame, bar_run):
    """IF/IC/IH: 09:30-11:30 | 13:01-15:00. Only gap is 1h31m -> one session."""
    td = "2025-07-02"
    bars = bar_run("2025-07-02 09:30", 121, td) + bar_run("2025-07-02 13:01", 120, td)
    _out, n = _sessions(raw_frame(bars))
    assert n == 1


def test_au_night_crossing_midnight_is_one_session(raw_frame, bar_run):
    """
    AU/AG/SC run 21:00 -> 02:30 as one contiguous 1-minute run. Crossing
    midnight must not split it, and the session's last bar is 02:30.
    """
    td = "2025-07-02"
    bars = bar_run("2025-07-01 21:00", 331, td) + bar_run("2025-07-02 09:01", 75, td)
    out, n = _sessions(raw_frame(bars))
    assert n == 2
    assert out.index[out["is_session_last"]][0] == pd.Timestamp("2025-07-02 02:30")


def test_friday_night_belongs_to_monday_and_splits(raw_frame, bar_run):
    """
    Friday's 21:00 night session carries MONDAY's `trading_date`, so one
    trading_date can contain a 58-hour internal gap. It must still split into
    two sessions.
    """
    td = "2025-07-07"                                  # a Monday
    bars = (bar_run("2025-07-04 21:00", 121, td)       # Friday night
            + bar_run("2025-07-07 09:01", 75, td))     # Monday day
    out, n = _sessions(raw_frame(bars))
    assert n == 2
    assert out.index.to_series().diff().max() > pd.Timedelta("54h")
    assert out["trading_date"].nunique() == 1, "Friday night must carry Monday's trading_date"


def test_ambiguous_gap_raises(raw_frame, bar_run):
    """
    A 4h gap inside one trading_date sits in the empty band the 3h threshold
    relies on. That means this symbol/era has a session shape nobody measured,
    so the split may be wrong - fail loudly rather than mis-split silently.
    """
    td = "2025-07-02"
    bars = bar_run("2025-07-02 09:00", 30, td) + bar_run("2025-07-02 13:30", 30, td)
    with pytest.raises(ValueError, match="模糊带"):
        assert_no_ambiguous_gaps(raw_frame(bars), "TEST")


def test_normal_shapes_do_not_trip_the_guard(raw_frame, bar_run):
    td = "2025-07-02"
    bars = (bar_run("2025-07-01 21:00", 121, td)
            + bar_run("2025-07-02 09:01", 75, td)
            + bar_run("2025-07-02 10:31", 60, td)
            + bar_run("2025-07-02 13:31", 90, td))
    assert_no_ambiguous_gaps(raw_frame(bars), "TEST")     # must not raise


# ------------------------------------------------------------- hygiene ----
def test_fake_zero_volume_run_is_dropped(raw_frame, bar_run):
    """
    Reproduces the real AU 2025-10-09 defect: a 151-bar 00:00-02:30 run in
    which every bar is untraded except one mislabeled auction bar that carries
    real volume AND a +4% price jump.

    The ETL's coarser `(sym, trading_date, calendar_day, day/night)` segment
    key lets this through because the segment's total volume is > 0 - which is
    exactly why the criterion here must be `traded <= 2`, not `vol == 0`.
    Left in, a backtest fires stops and breakouts at a time the market was shut.
    """
    td = "2025-10-09"
    fake = bar_run("2025-10-09 00:00", 151, td, price=770.0, volume=0.0)
    fake[-1] = (fake[-1][0], td, 770.0, 801.0, 770.0, 801.0, 454.0)   # the +4% bar
    real = bar_run("2025-10-09 09:01", 75, td, price=770.0, volume=100.0)

    clean, dropped = drop_dead_runs(raw_frame(fake + real))
    assert len(dropped) == 1
    assert dropped.iloc[0]["bars"] == 151 and dropped.iloc[0]["traded"] == 1
    assert len(clean) == 75
    # and the poisoned high never reaches the day aggregate
    assert daily_ohlc(clean)["H"].iloc[0] == 770.0


def test_whole_zero_volume_run_is_dropped(raw_frame, bar_run):
    td = "2025-01-02"
    bars = (bar_run("2025-01-02 09:00", 40, td, volume=0.0)
            + bar_run("2025-01-02 13:31", 90, td, volume=5.0))
    clean, dropped = drop_dead_runs(raw_frame(bars))
    assert len(dropped) == 1 and len(clean) == 90


def test_short_fragment_is_kept(raw_frame, bar_run):
    """
    A 10-bar sliver with one traded bar is BELOW `min_bars`, so it survives.
    Window edges routinely produce these; the filter must not eat them.
    """
    td = "2025-07-02"
    bars = bar_run("2025-07-02 14:51", 10, td, volume=0.0)
    bars[-1] = (bars[-1][0], td, 100.0, 100.0, 100.0, 100.0, 7.0)
    clean, dropped = drop_dead_runs(raw_frame(bars))
    assert len(dropped) == 0 and len(clean) == 10


def test_no_false_positives_on_a_normal_day(raw_frame, bar_run):
    td = "2025-07-02"
    bars = bar_run("2025-07-02 09:00", 226, td, volume=50.0)
    clean, dropped = drop_dead_runs(raw_frame(bars))
    assert len(dropped) == 0 and len(clean) == 226


def test_contiguous_runs_are_zero_based(raw_frame, bar_run):
    """
    Off-by-one guard: `(gap | tdc).cumsum()` starts at 1 because the first bar
    always changes trading_date. Left un-shifted, `np.bincount` would report a
    phantom empty run 0 and `drop_dead_runs` would emit a bogus dropped row.
    """
    bars = bar_run("2025-07-02 09:00", 5, "2025-07-02") + bar_run("2025-07-02 13:31", 5, "2025-07-02")
    runs = contiguous_runs(raw_frame(bars))
    assert runs[0] == 0
    assert list(np.unique(runs)) == [0, 1]


# ---------------------------------------------------- previous-day lines ----
def _agg(raw_frame, bar_run, days):
    """days: {trading_date: (price_low, price_high)} -> one 40-bar run each."""
    bars = []
    for i, (td, (lo, hi)) in enumerate(days.items()):
        run = bar_run(f"{td} 09:00", 40, td, price=lo, volume=10.0)
        run[20] = (run[20][0], td, lo, hi, lo, hi, 10.0)       # the day's high
        bars += run
    return daily_ohlc(raw_frame(bars))


def test_first_day_has_no_reference_and_is_invalid(raw_frame, bar_run):
    agg = _agg(raw_frame, bar_run, {"2025-07-01": (100, 110), "2025-07-02": (101, 111)})
    lines = prev_day_lines(agg, RBreakerParams(), RBreakerConfig())
    assert not bool(lines["valid"].iloc[0]), "day 1 has no previous day"
    assert bool(lines["valid"].iloc[1])
    assert pd.isna(lines["Bbreak"].iloc[0])
    assert not pd.isna(lines["Bbreak"].iloc[1])


def test_stale_reference_beyond_max_gap_is_rejected(raw_frame, bar_run):
    """
    A month-long hole must not silently supply a reference. Chinese New Year
    is the longest legitimate break at ~10 calendar days, so the 15-day
    default lets holidays through and blocks real data gaps.
    """
    agg = _agg(raw_frame, bar_run,
               {"2025-07-01": (100, 110), "2025-09-01": (101, 111)})
    lines = prev_day_lines(agg, RBreakerParams(), RBreakerConfig())
    assert not bool(lines["valid"].iloc[1]), "62-day gap must invalidate the reference"


def test_thin_reference_day_is_rejected(raw_frame, bar_run):
    td1, td2 = "2025-07-01", "2025-07-02"
    bars = (bar_run(f"{td1} 14:51", 10, td1, price=100.0)      # only 10 bars
            + bar_run(f"{td2} 09:00", 40, td2, price=101.0))
    lines = prev_day_lines(daily_ohlc(raw_frame(bars)), RBreakerParams(), RBreakerConfig())
    assert not bool(lines["valid"].iloc[1]), "a 10-bar reference day must be rejected"


def test_tradable_from_blocks_the_padding_window(raw_frame, bar_run):
    """
    `prepare()` loads extra days before START so day 1 has a reference. Those
    padding days must never themselves become tradable.
    """
    agg = _agg(raw_frame, bar_run,
               {"2025-06-28": (100, 110), "2025-07-01": (101, 111), "2025-07-02": (102, 112)})
    lines = prev_day_lines(agg, RBreakerParams(), RBreakerConfig(),
                           tradable_from=pd.Timestamp("2025-07-02"))
    assert list(lines["valid"]) == [False, False, True]


def test_reference_ohlc_spans_the_night_session(raw_frame, bar_run):
    """
    The reference is the FULL trading day, night + day. A high set at 22:00
    the previous evening must reach the next day's lines - night bars already
    carry the next trading day's `trading_date`, so plain `groupby` does it.
    """
    td1, td2 = "2025-07-01", "2025-07-02"
    night = bar_run("2025-06-30 21:00", 60, td1, price=100.0)
    night[30] = (night[30][0], td1, 100.0, 130.0, 100.0, 100.0, 10.0)   # night high
    day = bar_run(f"{td1} 09:01", 60, td1, price=100.0)
    nxt = bar_run(f"{td2} 09:01", 60, td2, price=100.0)

    agg = daily_ohlc(raw_frame(night + day + nxt))
    assert agg.loc[pd.Timestamp(td1), "H"] == 130.0
    lines = prev_day_lines(agg, RBreakerParams(), RBreakerConfig())
    assert lines.loc[pd.Timestamp(td2), "ref_H"] == 130.0
