"""
v1.1 Phase B verification: `latest_closed_bar` boundary tests.

This is the one function standing between the strategy and a lookahead leak,
so it gets checked more deliberately than a typical helper: the exact instant
a higher-timeframe bar closes, one tick before that instant, the very start
of the series (nothing closed yet), and a full-day "staircase" check that
the value returned only ever changes at a real close boundary.
"""

import pandas as pd

from src.data.resample import latest_closed_bar, resample_all_timeframes

SYMBOL = "NVDA"


def main():
    df = pd.read_csv(f"data/raw/{SYMBOL}_5m.csv", parse_dates=["timestamp"])
    timeframes = resample_all_timeframes(df)
    df_5m = timeframes["5m"]

    first_date = df_5m["timestamp"].dt.tz_convert("America/New_York").dt.date.iloc[0]
    day_5m = df_5m[df_5m["timestamp"].dt.tz_convert("America/New_York").dt.date == first_date].reset_index(drop=True)

    for tf, rule, n_bars in (("15m", "15min", 3), ("30m", "30min", 6), ("1h", "1h", 12)):
        higher = timeframes[tf]
        day_higher = higher[higher["timestamp"].dt.tz_convert("America/New_York").dt.date == first_date].reset_index(drop=True)
        first_bar = day_higher.iloc[0]
        close_time = first_bar["timestamp"] + pd.Timedelta(rule)

        print(f"\n[{tf}] first bar opens {first_bar['timestamp']}, closes {close_time}")

        # Before the first higher-tf bar has closed, nothing is usable yet -
        # covers every 5m bar of that first window, not just bar 0.
        for i in range(n_bars):
            open_time = day_5m["timestamp"].iloc[i]
            result = latest_closed_bar(higher, rule, open_time)
            assert result is None, f"[{tf}] expected None before first close, got {result} at bar {i}"
        print(f"[{tf}] None for all {n_bars} bars before first close: OK")

        # One tick (5m) before the close instant: still not closed.
        one_tick_before = close_time - pd.Timedelta("5min")
        result = latest_closed_bar(higher, rule, one_tick_before)
        assert result is None, f"[{tf}] expected None one tick before close, got {result}"
        print(f"[{tf}] None exactly one 5m tick before close ({one_tick_before}): OK")

        # Exactly at the close instant: now usable (bar spans [open, close), so
        # the instant it closes is the first moment it's fair game).
        result = latest_closed_bar(higher, rule, close_time)
        assert result is not None and result["timestamp"] == first_bar["timestamp"], (
            f"[{tf}] expected first bar to become usable exactly at its close time, got {result}"
        )
        print(f"[{tf}] usable exactly at close instant ({close_time}): OK")

        # Staircase check over the whole day: the value returned should only
        # change at the moments higher-tf bars actually close, and should
        # never expose a bar before its close time.
        seen_values = []
        for _, row in day_5m.iterrows():
            bar = latest_closed_bar(higher, rule, row["timestamp"])
            bar_ts = bar["timestamp"] if bar is not None else None
            if bar is not None:
                assert bar["timestamp"] + pd.Timedelta(rule) <= row["timestamp"], (
                    f"[{tf}] lookahead leak: bar closing at "
                    f"{bar['timestamp'] + pd.Timedelta(rule)} exposed at {row['timestamp']}"
                )
            seen_values.append(bar_ts)
        transitions = sum(1 for a, b in zip(seen_values, seen_values[1:]) if a != b)
        print(f"[{tf}] staircase check over {len(day_5m)} bars: {transitions} transitions, no lookahead leaks")

    print("\nAll Phase B checks passed.")


if __name__ == "__main__":
    main()
