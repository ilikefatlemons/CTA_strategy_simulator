"""
The six R-Breaker lines and the trailing-stop geometry.

The load-bearing test here is `test_trail_closed_form_matches_the_giveback_definition`:
the strategy spec was written as a "giveback from peak" expression, and the
engine implements the algebraically-simplified closed form. If those two ever
disagree, every stop in every backtest is wrong and nothing else would catch it.
"""

import random

import pytest

from src.strategy.rbreaker import (
    LINE_NAMES,
    RBreakerParams,
    rbreaker_lines,
    trail_stop_long,
    trail_stop_long_from_giveback,
    trail_stop_short,
    trail_stop_short_from_giveback,
)

P = RBreakerParams()


def _random_hlc(rng):
    """A plausible previous day: L < C < H, all positive."""
    low = rng.uniform(50, 5000)
    rng_span = rng.uniform(0.005, 0.08) * low
    high = low + rng_span
    close = rng.uniform(low, high)
    return high, low, close


def test_six_lines_against_hand_computed_values():
    """
    H=110, L=90, C=100 with f1=0.35, f2=0.07, f3=0.25:

        Ssetup = 110 + 0.35*(100-90)              = 113.5
        Bsetup = 90  - 0.35*(110-100)             = 86.5
        Senter = 0.535*(110+100) - 0.07*90        = 106.05
        Benter = 0.535*(90+100)  - 0.07*110       = 93.95
        Bbreak = 113.5 + 0.25*(113.5-86.5)        = 120.25
        Sbreak = 86.5  - 0.25*(113.5-86.5)        = 79.75
    """
    got = rbreaker_lines(110.0, 90.0, 100.0, P)
    want = {"Ssetup": 113.5, "Bsetup": 86.5, "Senter": 106.05,
            "Benter": 93.95, "Bbreak": 120.25, "Sbreak": 79.75}
    assert set(got) == set(LINE_NAMES)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=1e-9), k


def test_setup_spread_identity():
    """Ssetup - Bsetup == (1 + f1) * (H - L), for any L < C < H."""
    rng = random.Random(1)
    for _ in range(200):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        assert L["Ssetup"] - L["Bsetup"] == pytest.approx((1 + P.f1) * (h - low), rel=1e-12)


def test_break_offset_is_f3_times_one_plus_f1():
    """
    Bbreak - Ssetup == f3*(1+f1)*(H-L) == 0.3375*(H-L), and Bsetup - Sbreak
    is the mirror.

    This pins the `0.3375` that appears in the written stop formula to
    f3*(1+f1) rather than leaving it an unexplained bare constant - which is
    exactly the ambiguity the whole trailing stop hinges on.
    """
    assert P.f3 * (1 + P.f1) == pytest.approx(0.3375, abs=1e-12)
    rng = random.Random(2)
    for _ in range(200):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        r = h - low
        assert L["Bbreak"] - L["Ssetup"] == pytest.approx(0.3375 * r, rel=1e-12)
        assert L["Bsetup"] - L["Sbreak"] == pytest.approx(0.3375 * r, rel=1e-12)


def test_strict_line_ordering():
    """Bbreak > Ssetup > Senter > Benter > Bsetup > Sbreak whenever H > L."""
    rng = random.Random(3)
    for _ in range(500):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        order = [L["Bbreak"], L["Ssetup"], L["Senter"],
                 L["Benter"], L["Bsetup"], L["Sbreak"]]
        assert order == sorted(order, reverse=True), (h, low, c, L)


def test_long_and_short_entries_can_never_both_trigger():
    """
    Bbreak - Sbreak = (1 + 2*f3)*(Ssetup - Bsetup) > 0, so the two breakout
    conditions are mutually exclusive by construction. The engine relies on
    this (it uses `if up ... elif dn`, never checking for a conflict).
    """
    rng = random.Random(4)
    for _ in range(200):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        assert L["Bbreak"] > L["Sbreak"]
        assert L["Bbreak"] - L["Sbreak"] == pytest.approx(
            (1 + 2 * P.f3) * (L["Ssetup"] - L["Bsetup"]), rel=1e-12)


def test_scalar_and_series_inputs_agree():
    import pandas as pd

    highs = pd.Series([110.0, 210.0, 55.0])
    lows = pd.Series([90.0, 190.0, 45.0])
    closes = pd.Series([100.0, 205.0, 50.0])
    frame = rbreaker_lines(highs, lows, closes, P)
    assert list(frame.columns) == list(LINE_NAMES)
    for i in range(3):
        scalar = rbreaker_lines(highs[i], lows[i], closes[i], P)
        for name in LINE_NAMES:
            assert frame[name].iloc[i] == pytest.approx(scalar[name], rel=1e-12)


def test_trail_closed_form_matches_the_giveback_definition():
    """
    The spec was written as

        giveback = (Bbreak - Senter) - (Bbreak - Ssetup)/d + MFE_geo*(1 - 1/d)
        stop     = peak - giveback,     MFE_geo = peak - Bbreak

    The engine implements the simplified `Senter + (peak - Ssetup)/d`. These
    must agree to machine precision, for any day and any peak. Short side
    likewise against `Benter - (Bsetup - trough)/d`.
    """
    rng = random.Random(5)
    for _ in range(500):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        span = h - low
        # peak ranges from "just entered" (== Bbreak) to a big run
        peak = L["Bbreak"] + rng.uniform(0.0, 4.0) * span
        trough = L["Sbreak"] - rng.uniform(0.0, 4.0) * span

        closed = trail_stop_long(L["Senter"], L["Ssetup"], peak, P.d)
        literal = trail_stop_long_from_giveback(
            L["Bbreak"], L["Senter"], L["Ssetup"], peak, P.d)
        assert closed == pytest.approx(literal, rel=1e-12, abs=1e-9)

        closed_s = trail_stop_short(L["Benter"], L["Bsetup"], trough, P.d)
        literal_s = trail_stop_short_from_giveback(
            L["Sbreak"], L["Benter"], L["Bsetup"], trough, P.d)
        assert closed_s == pytest.approx(literal_s, rel=1e-12, abs=1e-9)


def test_trail_is_monotone_in_the_running_extreme():
    """
    A higher peak may never produce a looser long stop (and vice versa for
    shorts). This is what makes it a trailing stop rather than a wandering one.
    """
    L = rbreaker_lines(110.0, 90.0, 100.0, P)
    prev = -float("inf")
    for k in range(50):
        s = trail_stop_long(L["Senter"], L["Ssetup"], L["Bbreak"] + k, P.d)
        assert s >= prev
        prev = s
    prev = float("inf")
    for k in range(50):
        s = trail_stop_short(L["Benter"], L["Bsetup"], L["Sbreak"] - k, P.d)
        assert s <= prev
        prev = s


def test_initial_stop_sits_below_entry_for_a_long():
    """
    At entry (peak == Bbreak) the stop is Senter + (Bbreak - Ssetup)/d, which
    must be strictly below Bbreak - otherwise every trade would stop out on
    the bar after it filled.

        risk = (Bbreak - Senter) - (Bbreak - Ssetup)/d
    """
    rng = random.Random(6)
    for _ in range(200):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        stop0 = trail_stop_long(L["Senter"], L["Ssetup"], L["Bbreak"], P.d)
        assert stop0 < L["Bbreak"]
        assert stop0 == pytest.approx(
            L["Bbreak"] - ((L["Bbreak"] - L["Senter"]) - (L["Bbreak"] - L["Ssetup"]) / P.d),
            rel=1e-12)
        stop0_s = trail_stop_short(L["Benter"], L["Bsetup"], L["Sbreak"], P.d)
        assert stop0_s > L["Sbreak"]
