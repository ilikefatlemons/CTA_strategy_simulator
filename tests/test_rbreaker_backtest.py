"""
The R-Breaker bar loop (`src/engine/rbreaker_backtest.py`).

Every test builds an exact price path against exact lines, so a failure names
the rule that broke. The reference day is always H=110 L=90 C=100, which gives

    Bbreak 120.25 | Ssetup 113.5 | Senter 106.05 | Benter 93.95 | Bsetup 86.5 | Sbreak 79.75

so the initial long stop is Senter + (Bbreak - Ssetup)/3 = 106.05 + 2.25 = 108.30
and the initial short stop is Benter - (Bsetup - Sbreak)/3 = 93.95 - 2.25 = 91.70.
"""

import numpy as np
import pandas as pd
import pytest

from src.engine.rbreaker_backtest import SESSION_END, TRAIL, run_rbreaker_backtest
from src.strategy.rbreaker import RBreakerConfig, RBreakerParams

REF = (110.0, 90.0, 100.0)
BBREAK, SBREAK = 120.25, 79.75
SSETUP, SENTER = 113.5, 106.05
BSETUP, BENTER = 86.5, 93.95
STOP0_LONG = 108.30           # Senter + (Bbreak - Ssetup)/3, i.e. peak == Bbreak
STOP0_SHORT = 91.70           # Benter - (Bsetup - Sbreak)/3
TD = "2025-07-02"


def _stop_long(peak):
    """
    The level a bar is tested against, given the running peak THROUGH THE
    PREVIOUS BAR. Note the entry bar's own high is folded into `peak` at the
    end of the entry bar, so the bar right after a fill is already tested
    against max(Bbreak, high[entry]) - not against STOP0_LONG.
    """
    return SENTER + (peak - SSETUP) / 3.0


def _bars(prices, start=f"{TD} 09:00", td=TD):
    """
    prices: list of (open, high, low, close). One bar per minute.
    """
    times = pd.date_range(pd.Timestamp(start), periods=len(prices), freq="1min")
    return [(t, td, o, h, low, c, 10.0) for t, (o, h, low, c) in zip(times, prices)]


def _flat(price, n):
    return [(price, price, price, price)] * n


# --------------------------------------------------------------- entries ----
def test_long_entry_fills_at_the_next_bars_open(make_prepared):
    """
    The signal is read off the fully-closed bar i-1 (its high pierced Bbreak);
    the fill is bar i's OWN open. Never the signal bar's own price.
    """
    prices = _flat(100.0, 3) + [(100.0, 121.0, 100.0, 100.0)] + [(117.0, 118.0, 116.0, 117.0)] + _flat(117.0, 10)
    p = make_prepared(_bars(prices), {TD: REF})
    r = run_rbreaker_backtest(p)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.direction == "long"
    assert t.entry_bar_idx == 4, "signal on bar 3 -> fill on bar 4"
    assert t.entry_price == 117.0, "fill is bar 4's open, not bar 3's high"
    assert t.theo_entry == BBREAK, "the trail is anchored on Bbreak, not on the fill"


def test_short_entry_is_the_mirror(make_prepared):
    prices = _flat(100.0, 3) + [(100.0, 100.0, 79.0, 100.0)] + [(82.0, 83.0, 81.0, 82.0)] + _flat(82.0, 10)
    p = make_prepared(_bars(prices), {TD: REF})
    r = run_rbreaker_backtest(p)
    assert len(r.trades) == 1 and r.trades[0].direction == "short"
    assert r.trades[0].entry_price == 82.0
    assert r.trades[0].theo_entry == SBREAK


def test_close_trigger_mode_ignores_a_wick(make_prepared):
    """
    With `entry_trigger_on_range=False` a bar that pierced Bbreak intrabar but
    closed back below it must NOT fire.
    """
    prices = _flat(100.0, 3) + [(100.0, 121.0, 100.0, 100.0)] + _flat(100.0, 10)
    bars = _bars(prices)
    assert len(run_rbreaker_backtest(make_prepared(bars, {TD: REF})).trades) == 1
    cfg = RBreakerConfig(entry_trigger_on_range=False)
    assert len(run_rbreaker_backtest(make_prepared(bars, {TD: REF}, config=cfg)).trades) == 0


def test_invalid_day_never_trades(make_prepared):
    """A day with no usable previous-day reference must produce no trades."""
    prices = _flat(100.0, 3) + [(100.0, 500.0, 100.0, 400.0)] + _flat(400.0, 10)
    p = make_prepared(_bars(prices), {TD: None})
    assert run_rbreaker_backtest(p).trades == []


def test_no_entry_on_the_first_bar_of_a_session(make_prepared):
    """
    Bar i-1 belongs to the PREVIOUS session - possibly 6 to 58 hours earlier,
    possibly under a different day's lines. A signal must never cross that
    boundary. Here the night session's last bar pierces Bbreak; the day
    session's first bar must not fill on it.
    """
    night = _bars(_flat(100.0, 59) + [(100.0, 125.0, 100.0, 125.0)],
                  start="2025-07-01 21:00", td=TD)
    day = _bars(_flat(126.0, 30), start=f"{TD} 09:01", td=TD)
    r = run_rbreaker_backtest(make_prepared(night + day, {TD: REF}))
    entries = [t.entry_bar_idx for t in r.trades]
    assert 60 not in entries, "filled on the day session's first bar off a night-session signal"


def test_no_entry_on_the_last_bar_of_a_session(make_prepared):
    """
    Otherwise that bar would both enter and be force-closed - two actions on
    one bar, and a guaranteed zero-length trade.
    """
    prices = _flat(100.0, 8) + [(100.0, 121.0, 100.0, 121.0)] + [(121.0, 121.0, 121.0, 121.0)]
    p = make_prepared(_bars(prices), {TD: REF})
    r = run_rbreaker_backtest(p)
    assert r.trades == [], "entry filled on the frame's final (session-last) bar"


# ------------------------------------------------------- one per session ----
def test_only_one_trade_per_session(make_prepared):
    """
    A stop-out does not reopen the session. Three separate breakouts in one
    day session must yield exactly one trade.
    """
    prices = (
        _flat(100.0, 2)
        + [(100.0, 121.0, 100.0, 121.0)]          # signal 1
        + [(121.0, 121.0, 100.0, 100.0)]          # fill, then...
        + [(100.0, 100.0, 100.0, 100.0)]          # ...stopped out here
        + [(100.0, 122.0, 100.0, 122.0)]          # signal 2 - must be ignored
        + _flat(122.0, 3)
        + [(122.0, 123.0, 122.0, 123.0)]          # signal 3 - must be ignored
        + _flat(123.0, 5)
    )
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}))
    assert len(r.trades) == 1, [(t.entry_bar_idx, t.exit_bar_idx) for t in r.trades]


def test_each_session_gets_its_own_trade(make_prepared):
    """Night and day are separate sessions, so one trade each is allowed."""
    night = _bars(_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 20),
                  start="2025-07-01 21:00", td=TD)
    day = _bars(_flat(100.0, 2) + [(100.0, 122.0, 100.0, 122.0)] + _flat(122.0, 20),
                start=f"{TD} 09:01", td=TD)
    r = run_rbreaker_backtest(make_prepared(night + day, {TD: REF}))
    assert len(r.trades) == 2
    assert r.trades[0].session_id != r.trades[1].session_id


def test_reentry_allowed_when_the_switch_is_off(make_prepared):
    prices = (
        _flat(100.0, 2)
        + [(100.0, 121.0, 100.0, 121.0)]
        + [(121.0, 121.0, 100.0, 100.0)]
        + [(100.0, 100.0, 100.0, 100.0)]
        + [(100.0, 122.0, 100.0, 122.0)]
        + _flat(122.0, 8)
    )
    cfg = RBreakerConfig(one_trade_per_session=False)
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}, config=cfg))
    assert len(r.trades) == 2


# ----------------------------------------------------------------- exits ----
def test_trailing_stop_uses_the_closed_form(make_prepared):
    """`trail_stop[i] == Senter + (peak through i-1 - Ssetup)/d` on every bar."""
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 125.0, 121.0, 124.0)]
              + [(124.0, 130.0, 124.0, 129.0)]
              + [(129.0, 131.0, 128.0, 130.0)]
              + _flat(130.0, 6))
    p = make_prepared(_bars(prices), {TD: REF})
    r = run_rbreaker_backtest(p)
    t = r.trades[0]
    peak = t.theo_entry
    highs = p.df["high"].to_numpy()
    for i in range(t.entry_bar_idx, t.exit_bar_idx + 1):
        if i > t.entry_bar_idx:
            want = SENTER + (peak - SSETUP) / 3.0
            assert r.trail_stop[i] == pytest.approx(want, abs=1e-9), f"bar {i}"
        peak = max(peak, highs[i])
    assert r.trail_stop[t.entry_bar_idx] == pytest.approx(STOP0_LONG, abs=1e-9)


def test_entry_bars_own_high_is_folded_in_before_the_next_bar(make_prepared):
    """
    `peak` is seeded at Bbreak and then the entry bar's own high is folded in
    at the END of that bar. So the bar right after a fill is tested against
    max(Bbreak, high[entry]), not against the seed.

    Still lookahead-free: bar i+1's level depends on bar i, which has closed.
    """
    prices = (_flat(100.0, 2)
              + [(100.0, 121.0, 100.0, 121.0)]      # signal, high 121 > Bbreak 120.25
              + [(121.0, 121.0, 121.0, 121.0)]      # fill here; its high is also 121
              + _flat(121.0, 8))
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}))
    e = r.trades[0].entry_bar_idx
    assert r.trail_stop[e] == pytest.approx(STOP0_LONG), "entry bar records the seed level"
    assert r.trail_stop[e + 1] == pytest.approx(_stop_long(121.0)), \
        "the next bar must already include the entry bar's own high"
    assert r.trail_stop[e + 1] > r.trail_stop[e]


def test_trail_ignores_this_bars_own_high(make_prepared):
    """
    The running extreme is folded in only AFTER the stop has been tested, so
    an enormous high on bar i cannot move the level bar i is checked against.
    This is the single easiest place to introduce a future-function bug.
    """
    base = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
            + [(121.0, 121.0, 121.0, 121.0)] + _flat(121.0, 8))
    spiked = list(base)
    spiked[5] = (121.0, 900.0, 121.0, 121.0)          # absurd high on bar 5
    a = run_rbreaker_backtest(make_prepared(_bars(base), {TD: REF}))
    b = run_rbreaker_backtest(make_prepared(_bars(spiked), {TD: REF}))
    assert a.trail_stop[5] == pytest.approx(b.trail_stop[5], abs=1e-12), \
        "bar 5's own high leaked into the level bar 5 was tested against"


def test_no_lookahead_by_truncation(make_prepared):
    """
    Overwrite every bar from index m onward with garbage and re-run. Any trade
    that had already closed before m must come back byte-identical, and so
    must every stop level before m.

    Stronger than perturbing a single bar: it proves the loop's state at bar i
    is a function of bars <= i only.
    """
    rng = np.random.default_rng(0)
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 122.0, 121.0, 121.5)] + [(121.5, 121.5, 100.0, 100.0)]
              + _flat(100.0, 4)
              + [(100.0, 79.0, 79.0, 79.0)] + _flat(85.0, 10))
    bars = _bars(prices)
    m = 7
    full = run_rbreaker_backtest(make_prepared(bars, {TD: REF}))

    garbled = list(bars)
    for i in range(m, len(garbled)):
        t, td, *_rest = garbled[i]
        v = float(rng.uniform(10, 900))
        garbled[i] = (t, td, v, v * 1.4, v * 0.6, v, 10.0)
    part = run_rbreaker_backtest(make_prepared(garbled, {TD: REF}))

    a = [t for t in full.trades if t.exit_bar_idx < m]
    b = [t for t in part.trades if t.exit_bar_idx < m]
    assert len(a) == len(b) and len(a) >= 1
    for x, y in zip(a, b):
        assert (x.entry_bar_idx, x.exit_bar_idx, x.entry_price, x.exit_price, x.reason) == \
               (y.entry_bar_idx, y.exit_bar_idx, y.entry_price, y.exit_price, y.reason)
    assert np.allclose(full.trail_stop[:m], part.trail_stop[:m], equal_nan=True)


def test_gap_fill_exit_takes_the_bars_open(make_prepared):
    """
    If a bar OPENS beyond the stop, the fill is that open, not the stop line
    the market never traded at. Real across the 10:15/11:30 breaks and every
    session boundary, all of which contain zero bars.
    """
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 121.0, 121.0, 121.0)]
              + [(104.0, 104.0, 103.0, 103.5)]          # opens below STOP0_LONG
              + _flat(103.0, 6))
    p = make_prepared(_bars(prices), {TD: REF})
    t = run_rbreaker_backtest(p).trades[0]
    assert t.reason == TRAIL and t.gapped is True
    assert t.exit_price == 104.0, "must fill at the open, not at 108.30"

    off = RBreakerConfig(gap_fill_exits=False)
    t2 = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}, config=off)).trades[0]
    assert t2.exit_price == pytest.approx(_stop_long(121.0)) and t2.gapped is False


def test_forced_close_at_the_end_of_a_session(make_prepared):
    """A position that never hits its stop exits at the session's last close."""
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 6) + [(121.0, 122.0, 121.0, 121.5)]
    p = make_prepared(_bars(prices), {TD: REF})
    t = run_rbreaker_backtest(p).trades[0]
    assert t.reason == SESSION_END
    assert t.exit_bar_idx == len(prices) - 1
    assert t.exit_price == 121.5 and t.gapped is False


def test_trailing_stop_wins_over_session_close_on_the_same_bar(make_prepared):
    """
    When the final bar of a session both breaches the stop and would force a
    close, the stop takes it: mechanically earlier, and the worse price.
    """
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 5) + [(121.0, 121.0, 100.0, 120.0)]
    p = make_prepared(_bars(prices), {TD: REF})
    t = run_rbreaker_backtest(p).trades[0]
    assert t.reason == TRAIL
    assert t.exit_price == pytest.approx(_stop_long(121.0)), \
        "took the close (120.0) instead of the stop"


def test_position_never_survives_a_session_boundary(make_prepared):
    night = _bars(_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 20),
                  start="2025-07-01 21:00", td=TD)
    day = _bars(_flat(121.0, 30), start=f"{TD} 09:01", td=TD)
    p = make_prepared(night + day, {TD: REF})
    r = run_rbreaker_backtest(p)
    sid = p.df["session_id"].to_numpy()
    for t in r.trades:
        assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx]
    assert (r.position[len(night) - 1:] == 0).any()


# ------------------------------------------------------ bar-level hygiene ----
def test_one_action_per_bar(make_prepared):
    """
    A bar may enter or exit, never both. So no zero-length trade, and the next
    entry can only fill strictly after the previous exit.
    """
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 121.0, 100.0, 100.0)] + _flat(100.0, 3)
              + [(100.0, 79.0, 79.0, 79.0)] + _flat(85.0, 8))
    cfg = RBreakerConfig(one_trade_per_session=False)
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}, config=cfg))
    assert len(r.trades) >= 1
    for t in r.trades:
        assert t.exit_bar_idx > t.entry_bar_idx
    for a, b in zip(r.trades, r.trades[1:]):
        assert b.entry_bar_idx > a.exit_bar_idx


def test_entry_bar_is_never_also_an_exit_bar(make_prepared):
    """
    Even when the entry bar plunges straight through the stop, the exit waits
    for the next bar - one action per bar.
    """
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 121.0, 90.0, 95.0)] + _flat(95.0, 6))
    p = make_prepared(_bars(prices), {TD: REF})
    t = run_rbreaker_backtest(p).trades[0]
    assert t.entry_bar_idx == 3 and t.exit_bar_idx == 4


def test_trail_stop_is_nan_exactly_when_flat(make_prepared):
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 5) + [(121.0, 121.0, 121.0, 121.0)]
    p = make_prepared(_bars(prices), {TD: REF})
    r = run_rbreaker_backtest(p)
    t = r.trades[0]
    assert r.in_position[t.entry_bar_idx:t.exit_bar_idx + 1].all()
    assert not r.in_position[:t.entry_bar_idx].any()
    assert int(r.in_position.sum()) == t.bars_held + 1


# ------------------------------------------------------------------ cost ----
def test_cost_is_charged_once_per_round_trip(make_prepared):
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 5) + [(121.0, 122.0, 121.0, 121.0)]
    free = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF})).trades[0]
    cfg = RBreakerConfig(cost_bps=5.0)
    paid = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}, config=cfg)).trades[0]
    assert paid.pnl_gross == pytest.approx(free.pnl_gross)
    assert paid.pnl_net == pytest.approx(free.pnl_net - 5.0 / 1e4)


def test_equity_curve_matches_the_compounded_trades(make_prepared):
    prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 121.0, 100.0, 100.0)] + _flat(100.0, 3)
              + [(100.0, 79.0, 79.0, 79.0)] + _flat(85.0, 8))
    cfg = RBreakerConfig(one_trade_per_session=False)
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}, config=cfg))
    comp = float(np.prod([1 + t.pnl_net for t in r.trades]))
    assert r.equity_curve.iloc[-1] == pytest.approx(comp, abs=1e-12)


def test_short_side_is_the_reflection_of_the_long_side(make_prepared):
    """
    Reflecting prices through p -> 200 - p maps the long fixture onto the
    short one (and REF is symmetric about 100), so the mirrored run must
    produce mirrored trades with identical P&L up to sign conventions.
    """
    long_prices = (_flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)]
                   + [(121.0, 124.0, 121.0, 123.0)] + [(123.0, 123.0, 100.0, 105.0)] + _flat(105.0, 5))
    short_prices = [(200 - o, 200 - low, 200 - h, 200 - c) for o, h, low, c in long_prices]
    a = run_rbreaker_backtest(make_prepared(_bars(long_prices), {TD: REF})).trades
    b = run_rbreaker_backtest(make_prepared(_bars(short_prices), {TD: REF})).trades
    assert len(a) == len(b) == 1
    assert a[0].direction == "long" and b[0].direction == "short"
    assert (a[0].entry_bar_idx, a[0].exit_bar_idx) == (b[0].entry_bar_idx, b[0].exit_bar_idx)
    assert a[0].entry_price == pytest.approx(200 - b[0].entry_price)
    assert a[0].exit_price == pytest.approx(200 - b[0].exit_price)
    assert a[0].reason == b[0].reason


def test_no_position_survives_the_end_of_the_frame(make_prepared):
    """The final bar is always `is_session_last`, so the loop must end flat."""
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 20)
    r = run_rbreaker_backtest(make_prepared(_bars(prices), {TD: REF}))
    assert r.position[-1] == 0
    assert r.trades[-1].exit_bar_idx == len(prices) - 1


def test_params_and_config_can_be_overridden_at_call_time(make_prepared):
    """A looser d must produce a stop closer to Senter (further from entry)."""
    prices = _flat(100.0, 2) + [(100.0, 121.0, 100.0, 121.0)] + _flat(121.0, 8)
    p = make_prepared(_bars(prices), {TD: REF})
    tight = run_rbreaker_backtest(p, params=RBreakerParams(d=1.0))
    loose = run_rbreaker_backtest(p, params=RBreakerParams(d=10.0))
    i = 4
    assert tight.trail_stop[i] > loose.trail_stop[i]
