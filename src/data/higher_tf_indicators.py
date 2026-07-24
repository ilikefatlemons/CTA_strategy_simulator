"""
v1.1 Phase C: live, still-forming higher-timeframe MACD/ATR.

Deliberately NOT "wait until the higher-timeframe candle fully closes, then
expose one frozen value until the next close" (that was this module's first
draft, and it's wrong for the intended use: it can lag a genuine signal by
up to a full higher-timeframe window). Instead, this recomputes the
higher-timeframe candle - and therefore the indicator - on every single 5m
tick, using only the 5m bars that have printed so far in the current,
still-forming window. That's exactly what a live chart's "current candle"
does: it updates continuously and only turns into a fixed historical bar
once its window ends.

This is still lookahead-safe: at 5m bar i, the forming candle's
open/high/low/close is built only from 5m bars <= i. Nothing about a bar
that hasn't happened yet is ever touched - what changes vs. the frozen
version is *when* a signal becomes visible within an already-printed
higher-timeframe window, not *what* data it's allowed to see.
"""

import pandas as pd


def session_window_starts(df_5m: pd.DataFrame, rule: str) -> pd.Series:
    """
    For every 5m bar, the start timestamp of the higher-timeframe window it
    belongs to - anchored per ET session to that session's first bar (09:30
    ET), matching `resample_ohlcv`'s bin edges exactly so a "closed" window
    here is always the same window `resample_ohlcv` would produce. Public
    since Part D's frontend also uses this to snap a trade's 5m timestamp
    onto the higher-timeframe candle it should be marked on.
    """
    duration = pd.Timedelta(rule)
    et_dates = df_5m["timestamp"].dt.tz_convert("America/New_York").dt.date
    day_start = df_5m.groupby(et_dates)["timestamp"].transform("first")
    elapsed = df_5m["timestamp"] - day_start
    window_index = elapsed // duration
    return day_start + window_index * duration


def _true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def live_higher_tf_indicators(
    df_5m: pd.DataFrame,
    rule: str,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    For every 5m bar, MACD/signal/histogram/ATR computed off the
    still-forming higher-timeframe candle as of that bar's close - not
    frozen until the candle fully closes. NaN until enough *closed*
    higher-timeframe history exists to seed each indicator (same warm-up
    as the plain single-timeframe versions - a forming candle's live value
    is only meaningful once the EMA chain behind it has actually started).

    Implemented as one incremental EWM/rolling step per 5m bar rather than
    re-running `ewm()`/`rolling()` over the whole history each tick: only
    the state from the *last closed* higher-timeframe candle plus the
    forming candle's current OHLC is needed to reproduce the same value
    `pandas.ewm(adjust=False)` would give if that forming candle were the
    newest point in the series.
    """
    window_starts = session_window_starts(df_5m, rule).to_numpy()
    opens = df_5m["open"].to_numpy()
    highs = df_5m["high"].to_numpy()
    lows = df_5m["low"].to_numpy()
    closes = df_5m["close"].to_numpy()
    n = len(df_5m)

    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    alpha_signal = 2 / (signal_period + 1)

    # State carried forward from the last fully-closed higher-timeframe
    # candle only - never from the currently-forming one.
    closed_ema_fast: float | None = None
    closed_ema_slow: float | None = None
    closed_signal: float | None = None
    closed_close: float | None = None
    closed_true_ranges: list[float] = []  # capped at atr_period, oldest first

    current_window_start = None
    window_high = window_low = None

    out_macd = [float("nan")] * n
    out_signal = [float("nan")] * n
    out_hist = [float("nan")] * n
    out_atr = [float("nan")] * n

    for i in range(n):
        if current_window_start is None or window_starts[i] != current_window_start:
            if current_window_start is not None:
                # The window that was forming is now fully closed - fold it
                # into the closed-candle state exactly once, the same way
                # `indicators.macd`/`indicators.atr` would from a plain
                # resampled series.
                window_close = closes[i - 1]
                closed_ema_fast = (
                    window_close if closed_ema_fast is None
                    else alpha_fast * window_close + (1 - alpha_fast) * closed_ema_fast
                )
                closed_ema_slow = (
                    window_close if closed_ema_slow is None
                    else alpha_slow * window_close + (1 - alpha_slow) * closed_ema_slow
                )
                closed_macd = closed_ema_fast - closed_ema_slow
                closed_signal = (
                    closed_macd if closed_signal is None
                    else alpha_signal * closed_macd + (1 - alpha_signal) * closed_signal
                )
                closed_true_ranges.append(_true_range(window_high, window_low, closed_close))
                if len(closed_true_ranges) > atr_period:
                    closed_true_ranges.pop(0)
                closed_close = window_close

            current_window_start = window_starts[i]
            window_high = highs[i]
            window_low = lows[i]
        else:
            window_high = max(window_high, highs[i])
            window_low = min(window_low, lows[i])

        forming_close = closes[i]

        # Live MACD: one incremental EWM step from the last *closed*
        # candle's state, using this bar's forming (not yet closed) close.
        live_fast = (
            forming_close if closed_ema_fast is None
            else alpha_fast * forming_close + (1 - alpha_fast) * closed_ema_fast
        )
        live_slow = (
            forming_close if closed_ema_slow is None
            else alpha_slow * forming_close + (1 - alpha_slow) * closed_ema_slow
        )
        live_macd = live_fast - live_slow
        live_signal = (
            live_macd if closed_signal is None
            else alpha_signal * live_macd + (1 - alpha_signal) * closed_signal
        )
        out_macd[i] = live_macd
        out_signal[i] = live_signal
        out_hist[i] = live_macd - live_signal

        # Live ATR: same rolling mean, with the forming candle's true range
        # so far standing in for its still-incomplete final bar.
        live_true_range = _true_range(window_high, window_low, closed_close)
        window_true_ranges = closed_true_ranges + [live_true_range]
        if len(window_true_ranges) >= atr_period:
            out_atr[i] = sum(window_true_ranges[-atr_period:]) / atr_period

    return pd.DataFrame(
        {
            "timestamp": df_5m["timestamp"].to_numpy(),
            "macd": out_macd,
            "signal": out_signal,
            "histogram": out_hist,
            "atr": out_atr,
        }
    )
