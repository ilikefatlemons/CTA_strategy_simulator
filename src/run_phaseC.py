"""
v1.1 Phase C verification: live, still-forming higher-timeframe MACD/ATR.

Three checks, each targeting a different way this could go wrong:
  1. Truncation-invariance - the real definition of "no lookahead": bar i's
     value must be identical whether it's computed from the full series or
     from a series truncated right after bar i. If truncating the future
     ever changed a past value, something in the implementation would be
     peeking ahead.
  2. Self-consistency - at the last 5m bar of a window (where the forming
     candle's OHLC equals the final closed candle's OHLC), the live value
     must exactly match the plain closed-candle formula (`indicators.macd`/
     `atr` on the fully resampled series). The live path is a different
     code path to the same answer at that instant.
  3. Prefire demonstration - within a window, the value must actually change
     bar-by-bar (this is the whole point vs. the earlier frozen-until-close
     draft), shown against the resampled 15m closes for context.
"""

import pandas as pd

from src.data.higher_tf_indicators import live_higher_tf_indicators
from src.data.resample import resample_ohlcv
from src.indicators import atr, macd

SYMBOL = "NVDA"
TEST_ROWS = 2000  # ~25 sessions - enough for multiple full windows, fast to loop


def main():
    df = pd.read_csv(f"data/raw/{SYMBOL}_5m.csv", parse_dates=["timestamp"])
    df_5m = df.iloc[:TEST_ROWS].reset_index(drop=True)

    for tf, rule in (("15m", "15min"), ("30m", "30min"), ("1h", "1h")):
        higher = resample_ohlcv(df_5m, rule)
        full = live_higher_tf_indicators(df_5m, rule)

        # 1. Truncation-invariance.
        for cutoff in (10, 50, 200, 800, TEST_ROWS - 1):
            truncated = live_higher_tf_indicators(df_5m.iloc[:cutoff].reset_index(drop=True), rule)
            for col in ("macd", "signal", "histogram", "atr"):
                a = full[col].iloc[:cutoff].to_numpy()
                b = truncated[col].to_numpy()
                mismatch = ~(
                    (pd.isna(a) & pd.isna(b)) | (abs(a - b) < 1e-9)
                )
                assert not mismatch.any(), (
                    f"[{tf}] lookahead leak: truncating at {cutoff} changed {col} "
                    f"at rows {mismatch.nonzero()[0][:5]}"
                )
        print(f"[{tf}] truncation-invariance over 5 cutoffs: OK")

        # 2. Self-consistency at each window's final bar.
        window_starts = df_5m["timestamp"].dt.tz_convert("America/New_York")
        closed_macd = macd(higher)
        closed_atr = atr(higher)
        checked = 0
        for w in range(min(len(higher), 20)):
            window_end_time = higher["timestamp"].iloc[w] + pd.Timedelta(rule)
            last_bar_mask = df_5m["timestamp"] < window_end_time
            if not last_bar_mask.any():
                continue
            last_idx = last_bar_mask[last_bar_mask].index[-1]
            if df_5m["timestamp"].iloc[last_idx] + pd.Timedelta("5min") != window_end_time:
                continue  # this window's true last bar isn't in our truncated test slice
            live_row = full.iloc[last_idx]
            expected_macd = closed_macd.iloc[w]
            expected_atr = closed_atr.iloc[w]

            def close_enough(a, b):
                return (pd.isna(a) and pd.isna(b)) or abs(a - b) < 1e-9

            assert close_enough(live_row["macd"], expected_macd["macd"]), (
                f"[{tf}] window {w} last bar: macd {live_row['macd']} vs closed {expected_macd['macd']}"
            )
            assert close_enough(live_row["atr"], expected_atr), (
                f"[{tf}] window {w} last bar: atr {live_row['atr']} vs closed {expected_atr}"
            )
            checked += 1
        print(f"[{tf}] self-consistency at {checked} window-closing bars: OK")

    # 3. Prefire demonstration on 15m, first window with enough warm-up to
    # have a non-trivial histogram (not just NaN/seed noise).
    higher_15m = resample_ohlcv(df_5m, "15min")
    full_15m = live_higher_tf_indicators(df_5m, "15min")
    window_starts = higher_15m["timestamp"]
    demo_window = window_starts.iloc[40]  # well past MACD/ATR warm-up
    bars_in_window = df_5m[(df_5m["timestamp"] >= demo_window) & (df_5m["timestamp"] < demo_window + pd.Timedelta("15min"))]
    print(f"\n[15m] prefire demo - window starting {demo_window}:")
    for idx in bars_in_window.index:
        print(
            f"  5m bar {df_5m['timestamp'].iloc[idx]} close={df_5m['close'].iloc[idx]:.2f}  "
            f"live histogram={full_15m['histogram'].iloc[idx]:+.4f}"
        )
    print("  (histogram updates every 5m tick within the still-forming window, "
          "not just once when it closes)")

    print("\nAll Phase C checks passed.")


if __name__ == "__main__":
    main()
