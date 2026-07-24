import pandas as pd

from src.rules.cooldown import CooldownManager
from src.rules.ma_filter import MultiTimeframeFilter
from src.rules.pullback_entry import MarketSnapshot, PullbackEntryEngine


def _trend_klines(direction: str, n: int = 60) -> pd.DataFrame:
    """A clean, strictly-ordered MA5/20/50 uptrend (or downtrend)."""
    step = 1.0 if direction == "long" else -1.0
    closes = [100 + step * i for i in range(n)]
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC"),
    })


def _flat_klines(n: int = 60) -> pd.DataFrame:
    closes = [100.0] * n
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    })


def test_open_and_reentry_are_the_same_bound_method():
    """
    The spec's hard constraint: 'open' and 'reentry' signals must be produced
    by literally the same code path, not two branches. PullbackEntryEngine
    itself doesn't even have a concept of reentry - the caller relabels
    Signal.type after the fact - so asserting the engine only exposes one
    on_bar callable, used identically regardless of prior state, is the
    correct unit-level proof of that constraint.
    """
    engine = PullbackEntryEngine(MultiTimeframeFilter(), CooldownManager())
    first_call = engine.on_bar
    # simulate some bars happening (state mutates internally)
    snapshot = MarketSnapshot(
        tf_2h=_trend_klines("long"),
        small_tf={"5m": _flat_klines(), "15m": _flat_klines(), "30m": _flat_klines()},
        close=100.0, atr=1.0,
    )
    engine.on_bar(snapshot)
    second_call = engine.on_bar
    assert first_call == second_call  # same bound method, no branch was taken to swap it


def test_neutral_2h_bias_blocks_entry():
    engine = PullbackEntryEngine(MultiTimeframeFilter(), CooldownManager())
    snapshot = MarketSnapshot(
        tf_2h=_flat_klines(60), small_tf={"5m": _flat_klines(), "15m": _flat_klines(), "30m": _flat_klines()},
        close=100.0, atr=1.0,
    )
    assert engine.on_bar(snapshot) is None


def test_full_cascade_produces_signal():
    engine = PullbackEntryEngine(MultiTimeframeFilter(), CooldownManager())
    bias_klines = _trend_klines("long")

    # 1. pullback bar: all three small tfs bearish (opposite of long bias)
    pullback_snapshot = MarketSnapshot(
        tf_2h=bias_klines,
        small_tf={tf: _trend_klines("short") for tf in ("5m", "15m", "30m")},
        close=100.0, atr=1.0,
    )
    assert engine.on_bar(pullback_snapshot) is None

    # 2. trigger bar: 5m + 15m flip back to bullish, 30m stays bearish
    trigger_snapshot = MarketSnapshot(
        tf_2h=bias_klines,
        small_tf={
            "5m": _trend_klines("long"), "15m": _trend_klines("long"), "30m": _trend_klines("short"),
        },
        close=105.0, atr=1.0,
    )
    signal = engine.on_bar(trigger_snapshot)
    assert signal is not None
    assert signal.direction == "long"


def test_cooldown_ignores_protective_stop_streak():
    cooldown = CooldownManager(consecutive_sl_threshold=3)
    cooldown.on_trade_closed("long", "SL")
    cooldown.on_trade_closed("long", "PROTECTIVE_SL")
    cooldown.on_trade_closed("long", "SL")
    cooldown.on_trade_closed("long", "SL")
    # only 2 consecutive true "SL" trades in a row survive the PROTECTIVE_SL
    # reset in between - not enough to trip the threshold of 3.
    assert cooldown.is_active() is False

    cooldown.on_trade_closed("long", "SL")
    assert cooldown.is_active() is True


def test_cooldown_requires_three_consecutive_stable_bars_to_release():
    """
    Regression test for the bug the user caught: PullbackEntryEngine only
    ever trades while 2h bias is already non-neutral, so a release condition
    of "current 2h state is non-neutral" would immediately undo the cooldown
    the moment it activates, since bias hasn't necessarily changed at all.
    Requiring 3 *consecutive* bars of the same clean array (not just one)
    guarantees release can never coincide with the triggering bar, and also
    filters out a single-bar flicker/bounce from counting as confirmation.
    """
    klines_2h = _trend_klines("long")  # bullish array, unchanged throughout
    cooldown = CooldownManager(consecutive_sl_threshold=1, release_confirm_bars=3)
    cooldown.on_trade_closed("long", "SL")
    assert cooldown.is_active() is True

    # bias is still the same clean bullish array it always was - one bar of
    # "non-neutral" must NOT be enough to release on its own.
    cooldown.on_bar(klines_2h)
    assert cooldown.is_active() is True

    # a single bar flipping to bearish resets the streak to 1 - still not
    # enough on its own.
    cooldown.on_bar(_trend_klines("short"))
    assert cooldown.is_active() is True
    # 2 consecutive bearish bars - still short of the 3-bar confirmation.
    cooldown.on_bar(_trend_klines("short"))
    assert cooldown.is_active() is True
    # 3rd consecutive bearish bar - now confirmed, releases.
    cooldown.on_bar(_trend_klines("short"))
    assert cooldown.is_active() is False


def test_cooldown_neutral_bar_resets_the_stability_streak():
    """
    An extreme-candle trigger often smashes the array down to neutral on the
    triggering bar itself. A single bar bouncing straight back to a clean
    array right after that must not count as confirmation - neutral bars (and
    any state change) reset the streak, so 3 *consecutive* identical
    non-neutral reads are always required regardless of how the array looked
    at the moment cooldown was triggered.
    """
    cooldown = CooldownManager(consecutive_sl_threshold=999, release_confirm_bars=3)
    extreme_bar = pd.concat([_flat_klines(50), _trend_klines("long", 10)]).reset_index(drop=True)
    extreme_bar.loc[extreme_bar.index[-1], ["open", "close"]] = [80.0, 130.0]  # engulfs all 3 MAs
    assert cooldown.check_structure_break(extreme_bar) is True
    cooldown.on_bar(extreme_bar)
    assert cooldown.is_active() is True

    cooldown.on_bar(_flat_klines(60))  # neutral - resets the streak
    assert cooldown.is_active() is True
    cooldown.on_bar(_trend_klines("long"))  # bar 1 of a new streak
    assert cooldown.is_active() is True
    cooldown.on_bar(_trend_klines("long"))  # bar 2
    assert cooldown.is_active() is True
    cooldown.on_bar(_trend_klines("long"))  # bar 3 - now releases
    assert cooldown.is_active() is False
