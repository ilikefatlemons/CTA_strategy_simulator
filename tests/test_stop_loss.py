import pandas as pd

from src.rules.stop_loss import StopLossCalculator


def _klines(lows, highs):
    n = len(lows)
    return pd.DataFrame({
        "open": lows, "close": highs, "high": highs, "low": lows,
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC"),
    })


def test_picks_closer_of_atr_and_platform_long():
    # swing low at index 5 (value 95) is closer to entry(100) than the
    # ATR stop (100 - 1.5*3 = 95.5) is farther... construct so platform (98)
    # is nearer than ATR stop (95.5).
    lows = [100, 99, 98, 99, 100, 101, 102, 103, 104, 105]
    highs = [l + 1 for l in lows]
    lows[2] = 98  # swing low candidate near entry
    klines = _klines(lows, highs)
    calc = StopLossCalculator(atr_mult=1.5, offset_pct=0.0)
    stop = calc.calc(entry_price=100.0, direction="long", atr_value=3.0, klines_30m=klines)
    atr_stop = 100.0 - 1.5 * 3.0  # 95.5
    # platform-based candidate must have been considered and, being nearer
    # to entry than the ATR stop, must be what got chosen.
    assert stop > atr_stop


def test_falls_back_to_atr_when_no_platform_level():
    n = 10
    flat = [100.0] * n
    klines = _klines(flat, flat)  # no swing points at all in a flat series
    calc = StopLossCalculator(atr_mult=1.5, offset_pct=0.0)
    stop = calc.calc(entry_price=100.0, direction="long", atr_value=3.0, klines_30m=klines)
    assert stop == 100.0 - 1.5 * 3.0


def test_offset_pushes_stop_further_away():
    n = 10
    flat = [100.0] * n
    klines = _klines(flat, flat)
    calc = StopLossCalculator(atr_mult=1.5, offset_pct=0.01)
    stop_long = calc.calc(entry_price=100.0, direction="long", atr_value=3.0, klines_30m=klines)
    stop_short = calc.calc(entry_price=100.0, direction="short", atr_value=3.0, klines_30m=klines)
    assert stop_long < 100.0 - 1.5 * 3.0  # pushed further down (away from entry)
    assert stop_short > 100.0 + 1.5 * 3.0  # pushed further up (away from entry)
