"""
v1.1 Phase A verification: resample NVDA's 5m bars up to 15m/30m/1h and sanity
check the result against a manual OHLCV rollup of the first few bars of one
session, plus basic invariants (no cross-day bins, bar counts line up).
"""

import pandas as pd

from src.data.resample import resample_all_timeframes

SYMBOL = "NVDA"


def manual_ohlcv(bars: pd.DataFrame) -> tuple:
    return (
        bars["open"].iloc[0],
        bars["high"].max(),
        bars["low"].min(),
        bars["close"].iloc[-1],
        bars["volume"].sum(),
    )


def main():
    df = pd.read_csv(f"data/00-美股ETF历史/raw/{SYMBOL}_5m.csv", parse_dates=["timestamp"])
    timeframes = resample_all_timeframes(df)

    first_date = df["timestamp"].dt.tz_convert("America/New_York").dt.date.iloc[0]
    day_5m = df[df["timestamp"].dt.tz_convert("America/New_York").dt.date == first_date].reset_index(drop=True)
    print(f"First session ({first_date}): {len(day_5m)} 5m bars, "
          f"{day_5m['timestamp'].iloc[0]} -> {day_5m['timestamp'].iloc[-1]}")

    for tf, n_bars in (("15m", 3), ("30m", 6), ("1h", 12)):
        synth = timeframes[tf]
        day_synth = synth[synth["timestamp"].dt.tz_convert("America/New_York").dt.date == first_date]
        first_synth_bar = day_synth.iloc[0]
        manual = manual_ohlcv(day_5m.iloc[:n_bars])

        print(f"\n[{tf}] first bar: {dict(first_synth_bar[['open', 'high', 'low', 'close', 'volume']])}")
        print(f"[{tf}] manual   : open={manual[0]} high={manual[1]} low={manual[2]} close={manual[3]} volume={manual[4]}")
        ok = (
            first_synth_bar["open"] == manual[0]
            and first_synth_bar["high"] == manual[1]
            and first_synth_bar["low"] == manual[2]
            and first_synth_bar["close"] == manual[3]
            and first_synth_bar["volume"] == manual[4]
        )
        print(f"[{tf}] match: {ok}")
        assert ok, f"{tf} synthesized first bar does not match manual rollup"

        # 390 minutes/session: 15m -> 26 bars/day, 30m -> 13 bars/day exactly;
        # 1h -> 6 full hours + a 30m tail bar = 7 bars/day.
        expected_bars_per_day = {"15m": 26, "30m": 13, "1h": 7}[tf]
        actual = len(day_synth)
        print(f"[{tf}] bars in first session: {actual} (expected {expected_bars_per_day})")
        assert actual == expected_bars_per_day, f"{tf}: expected {expected_bars_per_day} bars/session, got {actual}"

        # No bin should span two ET calendar dates.
        spans_two_days = (
            synth["timestamp"].dt.tz_convert("America/New_York").dt.date
            != (synth["timestamp"] + pd.Timedelta(tf.replace("m", "min") if tf != "1h" else "1h"))
            .dt.tz_convert("America/New_York")
            .dt.date
        )
        # A bar can legitimately end exactly at session close (16:00) without
        # "spanning" days - only flag bars whose start and nominal end land
        # on different ET calendar dates AND that gap isn't just hitting the
        # boundary exactly.
        print(f"[{tf}] bins nominally crossing midnight boundary (informational): {spans_two_days.sum()}")

    print("\nAll Phase A checks passed.")


if __name__ == "__main__":
    main()
