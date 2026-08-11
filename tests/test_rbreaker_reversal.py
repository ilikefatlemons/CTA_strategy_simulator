"""
The R-Breaker **reversal** strategy (`src/engine/rbreaker_reversal_backtest.py`).

Reference day H=110 L=90 C=100 throughout, which gives

    Bbreak 120.25 | Ssetup 113.50 | Senter 106.05 | Benter 93.95 | Bsetup 86.50 | Sbreak 79.75

so the initial reversal-short stop is Bbreak - (Bbreak-Ssetup)/3 = 118.00 and the
initial reversal-long stop is Sbreak + (Bsetup-Sbreak)/3 = 82.00.

The load-bearing test here is `test_initial_risk_mirrors_the_breakout_exactly`:
the whole design rests on the reversal reusing the breakout's giveback constant,
so `breakout-long risk == reversal-short risk` must hold identically.
"""

import random

import numpy as np
import pytest

from src.engine.rbreaker_backtest import SESSION_END, TRAIL, run_rbreaker_backtest
from src.engine.rbreaker_reversal_backtest import run_rbreaker_reversal_backtest
from src.strategy.rbreaker import (
    RBreakerConfig,
    RBreakerParams,
    ReversalConfig,
    rbreaker_lines,
    trail_stop_long,
    trail_stop_rev_long,
    trail_stop_rev_long_from_giveback,
    trail_stop_rev_short,
    trail_stop_rev_short_from_giveback,
    trail_stop_short,
)

P = RBreakerParams()
REF = (110.0, 90.0, 100.0)                 # symmetric: all four risks coincide
REF_ASYM = (110.0, 90.0, 105.0)            # asymmetric: the two pairs separate
SSETUP, SENTER = 113.5, 106.05
BENTER, BSETUP = 93.95, 86.5
BBREAK, SBREAK = 120.25, 79.75
STOP0_SHORT = 118.00
STOP0_LONG = 82.00
TD = "2025-07-02"


def _bars(prices, start=f"{TD} 09:00", td=TD):
    import pandas as pd
    times = pd.date_range(pd.Timestamp(start), periods=len(prices), freq="1min")
    return [(t, td, o, h, low, c, 10.0) for t, (o, h, low, c) in zip(times, prices)]


def _flat(price, n):
    return [(price, price, price, price)] * n


def _touch_high(price, high):
    """One bar that pokes up to `high` without its low going below Senter."""
    return (price, high, price - 0.5, price)


def _touch_low(price, low):
    return (price, price + 0.5, low, price)


def _random_hlc(rng):
    low = rng.uniform(50, 5000)
    high = low + rng.uniform(0.005, 0.08) * low
    return high, low, rng.uniform(low, high)


# ------------------------------------------------------------- 几何/公式 ----
def test_reversal_closed_form_matches_the_giveback_definition():
    """
    The reversal reuses the breakout's giveback expression verbatim, only the
    entry line changes. The engine implements the simplified closed form; these
    must agree to machine precision for any day and any excursion.
    """
    rng = random.Random(11)
    for _ in range(500):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        span = h - low
        mfe = rng.uniform(0.0, 4.0) * span

        closed = trail_stop_rev_short(L["Bbreak"], L["Ssetup"], mfe, P.d)
        literal = trail_stop_rev_short_from_giveback(
            L["Bbreak"], L["Ssetup"], L["Senter"], L["Senter"] - mfe, P.d)
        assert closed == pytest.approx(literal, rel=1e-12, abs=1e-9)

        closed_l = trail_stop_rev_long(L["Sbreak"], L["Bsetup"], mfe, P.d)
        literal_l = trail_stop_rev_long_from_giveback(
            L["Sbreak"], L["Bsetup"], L["Benter"], L["Benter"] + mfe, P.d)
        assert closed_l == pytest.approx(literal_l, rel=1e-12, abs=1e-9)


def test_initial_risk_mirrors_the_breakout_exactly():
    """
    The core claim of the whole design: the giveback constant is a DISTANCE, so
    it does not depend on which line you enter from.

        breakout-long  risk == reversal-short risk == (Bbreak-Senter) - (Bbreak-Ssetup)/d
        breakout-short risk == reversal-long  risk == (Benter-Sbreak) - (Bsetup-Sbreak)/d

    That makes the two strategies risk-comparable by construction, which is the
    whole point of running them side by side.
    """
    rng = random.Random(12)
    for _ in range(300):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)

        brk_long_risk = L["Bbreak"] - trail_stop_long(L["Senter"], L["Ssetup"], L["Bbreak"], P.d)
        rev_short_risk = trail_stop_rev_short(L["Bbreak"], L["Ssetup"], 0.0, P.d) - L["Senter"]
        assert rev_short_risk == pytest.approx(brk_long_risk, rel=1e-12)

        brk_short_risk = trail_stop_short(L["Benter"], L["Bsetup"], L["Sbreak"], P.d) - L["Sbreak"]
        rev_long_risk = L["Benter"] - trail_stop_rev_long(L["Sbreak"], L["Bsetup"], 0.0, P.d)
        assert rev_long_risk == pytest.approx(brk_short_risk, rel=1e-12)


def test_the_two_mirror_pairs_are_genuinely_different():
    """
    Guard against a test that passes because everything happens to be equal.
    On an asymmetric reference day (C above the midpoint) the long-pair and
    short-pair constants must differ, so the pairing is a real claim.
    """
    L = rbreaker_lines(*REF_ASYM, P)
    pair_a = (L["Bbreak"] - L["Senter"]) - (L["Bbreak"] - L["Ssetup"]) / P.d
    pair_b = (L["Benter"] - L["Sbreak"]) - (L["Bsetup"] - L["Sbreak"]) / P.d
    assert pair_a == pytest.approx(11.025)
    assert pair_b == pytest.approx(12.875)
    assert abs(pair_a - pair_b) > 1.0


def test_initial_stops_sit_on_the_correct_side_of_entry():
    """A reversal short's stop must be ABOVE Senter, a long's BELOW Benter."""
    rng = random.Random(13)
    for _ in range(300):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        assert trail_stop_rev_short(L["Bbreak"], L["Ssetup"], 0.0, P.d) > L["Senter"]
        assert trail_stop_rev_long(L["Sbreak"], L["Bsetup"], 0.0, P.d) < L["Benter"]


def test_reversal_trail_tightens_monotonically():
    L = rbreaker_lines(*REF, P)
    prev = float("inf")
    for k in range(50):
        s = trail_stop_rev_short(L["Bbreak"], L["Ssetup"], float(k), P.d)
        assert s <= prev
        prev = s
    prev = -float("inf")
    for k in range(50):
        s = trail_stop_rev_long(L["Sbreak"], L["Bsetup"], float(k), P.d)
        assert s >= prev
        prev = s


def test_senter_benter_gap_is_0605R():
    """
    `Senter - Benter = 0.605*(H-L)`. Both arms can only be live once the session
    range has covered Bsetup..Ssetup (1.35R); once that happens, ANY bar inside
    the [Benter, Senter] band satisfies both trigger conditions at once, which
    is why the engine needs an explicit most-recent-arm-wins rule.
    """
    rng = random.Random(14)
    for _ in range(200):
        h, low, c = _random_hlc(rng)
        L = rbreaker_lines(h, low, c, P)
        assert L["Senter"] - L["Benter"] == pytest.approx(0.605 * (h - low), rel=1e-12)
        assert L["Ssetup"] - L["Bsetup"] == pytest.approx(1.35 * (h - low), rel=1e-12)


# ------------------------------------------------------------- 状态机 ----
def _short_setup():
    """rise to Ssetup (arm), hold, then dip through Senter -> short fills next bar."""
    return (
        _flat(110.0, 2)
        + [_touch_high(112.0, 114.0)]              # bar 2: high 114 >= Ssetup -> arms
        + [(112.0, 112.0, 105.0, 106.0)]           # bar 3: low 105 < Senter -> trigger
        + [(106.0, 106.0, 105.5, 106.0)]           # bar 4: FILL at open 106.0
        + _flat(106.0, 8)
    )


def test_short_arms_then_fires(make_prepared):
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(_short_setup()), {TD: REF}))
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.direction == "short"
    assert t.entry_bar_idx == 4, "signal on bar 3 -> fill on bar 4"
    assert t.entry_price == 106.0, "fill is bar 4's own open"
    assert t.theo_entry == pytest.approx(SENTER), "trail is anchored on Senter, not the fill"
    assert r.trail_stop[4] == pytest.approx(STOP0_SHORT)


def test_unarmed_dip_below_senter_does_not_fire(make_prepared):
    """Same dip, but the high never reaches Ssetup - the observation gate holds."""
    prices = (
        _flat(110.0, 2)
        + [_touch_high(112.0, 113.0)]              # 113 < Ssetup 113.5 -> NOT armed
        + [(112.0, 112.0, 105.0, 106.0)]
        + _flat(106.0, 9)
    )
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(prices), {TD: REF}))
    assert r.trades == [], "fired without ever touching the observation line"


def test_long_side_is_the_mirror(make_prepared):
    """Reflect the short fixture through p -> 200 - p and expect a mirrored long."""
    short_prices = _short_setup()
    long_prices = [(200 - o, 200 - low, 200 - h, 200 - c) for o, h, low, c in short_prices]
    a = run_rbreaker_reversal_backtest(make_prepared(_bars(short_prices), {TD: REF})).trades
    b = run_rbreaker_reversal_backtest(make_prepared(_bars(long_prices), {TD: REF})).trades
    assert len(a) == len(b) == 1
    assert a[0].direction == "short" and b[0].direction == "long"
    assert (a[0].entry_bar_idx, a[0].exit_bar_idx) == (b[0].entry_bar_idx, b[0].exit_bar_idx)
    assert a[0].entry_price == pytest.approx(200 - b[0].entry_price)
    assert a[0].exit_price == pytest.approx(200 - b[0].exit_price)
    assert b[0].theo_entry == pytest.approx(BENTER)


def test_arming_resets_at_the_session_boundary(make_prepared):
    """
    Decision 1: the running extreme is per-SESSION, not per trading day. A high
    touched during the night session must not arm the day session - otherwise
    what happened overnight silently decides whether the day session trades.
    """
    night = _bars(_flat(110.0, 2) + [_touch_high(112.0, 114.0)] + _flat(110.0, 20),
                  start="2025-07-01 21:00", td=TD)
    day = _bars([(110.0, 110.0, 105.0, 106.0)] + _flat(106.0, 20),
                start=f"{TD} 09:01", td=TD)
    r = run_rbreaker_reversal_backtest(make_prepared(night + day, {TD: REF}))
    day_entries = [t for t in r.trades if t.entry_bar_idx >= len(night)]
    assert day_entries == [], "night-session high armed the day session"


def test_arming_survives_a_breakout_within_the_session(make_prepared):
    """
    Decision 4: no disarm. Price arms short at Ssetup, then runs clean through
    Bbreak, then collapses back under Senter -> the short still fires. That is
    precisely the failed-breakout-reversal shape R-Breaker exists to trade.
    """
    prices = (
        _flat(110.0, 2)
        + [_touch_high(112.0, 114.0)]              # arms short
        + [_touch_high(118.0, 122.0)]              # blows through Bbreak 120.25
        + [(120.0, 120.0, 105.0, 106.0)]           # collapses under Senter
        + [(106.0, 106.0, 105.5, 106.0)]           # FILL
        + _flat(106.0, 8)
    )
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(prices), {TD: REF}))
    assert len(r.trades) == 1 and r.trades[0].direction == "short"


def test_most_recent_arm_wins(make_prepared):
    """
    Once the session range spans Bsetup..Ssetup both arms are live, and every
    bar inside [Benter, Senter] satisfies BOTH trigger conditions. The tiebreak
    is the most recent extreme - you fade the latest push. Here the low comes
    last, so the long is the live direction.
    """
    # The arming bar's low must stay ABOVE Senter, or the short fires first and
    # the long arm never gets a chance - which is a different scenario entirely.
    prices = (
        _flat(110.0, 2)
        + [_touch_high(110.0, 114.0)]              # arms SHORT (low 109.5 > Senter)
        + [_touch_low(100.0, 86.0)]                # then arms LONG (more recent)
        + [(90.0, 91.0, 90.0, 91.0)]               # FILL
        + _flat(91.0, 8)
    )
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(prices), {TD: REF}))
    assert len(r.trades) == 1
    assert r.trades[0].direction == "long", "the older short arm won the tiebreak"
    assert r.trades[0].theo_entry == pytest.approx(BENTER)


def test_arm_must_precede_trigger_switch(make_prepared):
    """
    With the switch on, a bar that both arms and triggers must not fill; the
    running extreme is held back one extra bar.
    """
    prices = (
        _flat(110.0, 2)
        + [(112.0, 114.0, 105.0, 106.0)]           # single bar: arms AND triggers
        + [(106.0, 106.0, 105.5, 106.0)]
        + _flat(106.0, 9)
    )
    bars = _bars(prices)
    loose = run_rbreaker_reversal_backtest(make_prepared(bars, {TD: REF}))
    assert len(loose.trades) == 1 and loose.trades[0].entry_bar_idx == 3

    strict = run_rbreaker_reversal_backtest(
        make_prepared(bars, {TD: REF}), config=ReversalConfig(arm_must_precede_trigger=True))
    assert [t.entry_bar_idx for t in strict.trades] != [3], "same-bar arm+trigger still filled"


def test_close_trigger_mode_ignores_a_wick(make_prepared):
    prices = (
        _flat(110.0, 2)
        + [_touch_high(112.0, 114.0)]
        + [(112.0, 112.0, 105.0, 112.0)]           # wick under Senter, closes back above
        + _flat(112.0, 9)
    )
    bars = _bars(prices)
    assert len(run_rbreaker_reversal_backtest(make_prepared(bars, {TD: REF})).trades) == 1
    cfg = ReversalConfig(entry_trigger_on_range=False)
    assert run_rbreaker_reversal_backtest(make_prepared(bars, {TD: REF}), config=cfg).trades == []


def test_invalid_day_never_trades(make_prepared):
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(_short_setup()), {TD: None}))
    assert r.trades == []


# ------------------------------------------- baseline discipline (inherited) ----
def test_trailing_stop_uses_the_closed_form_each_bar(make_prepared):
    p = make_prepared(_bars(_short_setup()), {TD: REF})
    r = run_rbreaker_reversal_backtest(p)
    t = r.trades[0]
    lows = p.df["low"].to_numpy()
    ext = 0.0
    for i in range(t.entry_bar_idx, t.exit_bar_idx + 1):
        if i > t.entry_bar_idx:
            want = trail_stop_rev_short(BBREAK, SSETUP, ext, P.d)
            assert r.trail_stop[i] == pytest.approx(want, abs=1e-9), f"bar {i}"
        ext = max(ext, SENTER - lows[i])


def test_entry_bars_own_low_is_folded_in_before_the_next_bar(make_prepared):
    """MFE is seeded at 0 on the theoretical line, then the fill bar's own
    excursion folds in at the END of that bar - so bar i+1 is already tighter."""
    p = make_prepared(_bars(_short_setup()), {TD: REF})
    r = run_rbreaker_reversal_backtest(p)
    e = r.trades[0].entry_bar_idx
    assert r.trail_stop[e] == pytest.approx(STOP0_SHORT)
    ext = SENTER - p.df["low"].to_numpy()[e]
    assert r.trail_stop[e + 1] == pytest.approx(trail_stop_rev_short(BBREAK, SSETUP, ext, P.d))
    assert r.trail_stop[e + 1] < r.trail_stop[e]


def test_no_lookahead_by_truncation(make_prepared):
    """
    Overwrite everything from bar m onward with garbage and re-run. Trades that
    already closed must be byte-identical, and so must every earlier stop.

    This also pins the arming ordering: the running extreme is folded in AFTER
    the entry block, so bar i's own high can never arm bar i itself.
    """
    prices = (
        _flat(110.0, 2)
        + [_touch_high(112.0, 114.0)]
        + [(112.0, 112.0, 105.0, 106.0)]
        + [(106.0, 106.0, 105.5, 106.0)]
        + [(106.0, 119.0, 106.0, 118.0)]           # stops out
        + _flat(118.0, 8)
    )
    bars = _bars(prices)
    m = 7
    full = run_rbreaker_reversal_backtest(make_prepared(bars, {TD: REF}))
    rng = np.random.default_rng(0)
    garbled = list(bars)
    for i in range(m, len(garbled)):
        tt, td, *_rest = garbled[i]
        v = float(rng.uniform(10, 900))
        garbled[i] = (tt, td, v, v * 1.4, v * 0.6, v, 10.0)
    part = run_rbreaker_reversal_backtest(make_prepared(garbled, {TD: REF}))

    a = [t for t in full.trades if t.exit_bar_idx < m]
    b = [t for t in part.trades if t.exit_bar_idx < m]
    assert len(a) == len(b) >= 1
    for x, y in zip(a, b):
        assert (x.entry_bar_idx, x.exit_bar_idx, x.entry_price, x.exit_price, x.reason) == \
               (y.entry_bar_idx, y.exit_bar_idx, y.entry_price, y.exit_price, y.reason)
    assert np.allclose(full.trail_stop[:m], part.trail_stop[:m], equal_nan=True)


def test_this_bars_own_high_cannot_arm_this_bar(make_prepared):
    """
    Pins the ordering of the running-extreme update. A bar whose high reaches
    Ssetup must not make itself the armer for a decision taken on that same bar.
    """
    base = (_flat(110.0, 3) + [(112.0, 112.0, 105.0, 106.0)] + _flat(106.0, 9))
    spiked = list(base)
    spiked[3] = (112.0, 999.0, 105.0, 106.0)       # bar 3 arms AND dips, same bar
    a = run_rbreaker_reversal_backtest(make_prepared(_bars(base), {TD: REF}))
    b = run_rbreaker_reversal_backtest(make_prepared(_bars(spiked), {TD: REF}))
    assert a.trades == [], "control fixture never reaches Ssetup, so must not trade"
    assert len(b.trades) == 1, "the spike should arm, and the dip should then fire"
    assert b.trades[0].entry_bar_idx > 3, "bar 3's own high armed bar 3 itself"


def test_one_action_per_bar_and_one_trade_per_session(make_prepared):
    p = make_prepared(_bars(_short_setup()), {TD: REF})
    r = run_rbreaker_reversal_backtest(p)
    sid = p.df["session_id"].to_numpy()
    for t in r.trades:
        assert t.exit_bar_idx > t.entry_bar_idx
        assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx]
    assert len({t.session_id for t in r.trades}) == len(r.trades)


def test_gap_fill_exit_takes_the_bars_open(make_prepared):
    prices = (
        _flat(110.0, 2)
        + [_touch_high(112.0, 114.0)]
        + [(112.0, 112.0, 105.0, 106.0)]
        + [(106.0, 106.0, 105.5, 106.0)]           # FILL at 106
        + [(119.0, 119.5, 119.0, 119.0)]           # opens ABOVE the short stop
        + _flat(119.0, 8)
    )
    p = make_prepared(_bars(prices), {TD: REF})
    t = run_rbreaker_reversal_backtest(p).trades[0]
    assert t.reason == TRAIL and t.gapped is True
    assert t.exit_price == 119.0, "must fill at the open, not at the stop line"

    off = ReversalConfig(gap_fill_exits=False)
    t2 = run_rbreaker_reversal_backtest(make_prepared(_bars(prices), {TD: REF}), config=off).trades[0]
    assert t2.gapped is False and t2.exit_price < 119.0


def test_forced_close_at_session_end(make_prepared):
    p = make_prepared(_bars(_short_setup()), {TD: REF})
    t = run_rbreaker_reversal_backtest(p).trades[0]
    assert t.reason == SESSION_END
    assert t.exit_bar_idx == len(p.df) - 1


def test_no_position_survives_the_end_of_the_frame(make_prepared):
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(_short_setup()), {TD: REF}))
    assert r.position[-1] == 0


def test_cost_is_charged_once_per_round_trip(make_prepared):
    bars = _bars(_short_setup())
    free = run_rbreaker_reversal_backtest(make_prepared(bars, {TD: REF})).trades[0]
    paid = run_rbreaker_reversal_backtest(
        make_prepared(bars, {TD: REF}), config=ReversalConfig(cost_bps=5.0)).trades[0]
    assert paid.pnl_gross == pytest.approx(free.pnl_gross)
    assert paid.pnl_net == pytest.approx(free.pnl_net - 5.0 / 1e4)


def test_equity_curve_matches_compounded_trades(make_prepared):
    r = run_rbreaker_reversal_backtest(make_prepared(_bars(_short_setup()), {TD: REF}))
    comp = float(np.prod([1 + t.pnl_net for t in r.trades])) if r.trades else 1.0
    assert r.equity_curve.iloc[-1] == pytest.approx(comp, abs=1e-12)


def test_the_two_strategies_are_independent(make_prepared):
    """
    Same data, same Prepared: running one must not perturb the other. Guards the
    shared `Prepared` against accidental mutation by either engine.
    """
    p = make_prepared(_bars(_short_setup()), {TD: REF})
    rev_a = run_rbreaker_reversal_backtest(p)
    brk = run_rbreaker_backtest(p, config=RBreakerConfig())
    rev_b = run_rbreaker_reversal_backtest(p)
    assert [t.entry_bar_idx for t in rev_a.trades] == [t.entry_bar_idx for t in rev_b.trades]
    assert np.allclose(rev_a.trail_stop, rev_b.trail_stop, equal_nan=True)
    assert np.allclose(rev_a.equity_curve.to_numpy(), rev_b.equity_curve.to_numpy())
    del brk
