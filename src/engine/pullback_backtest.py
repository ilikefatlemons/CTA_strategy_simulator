"""
回调模型回测引擎 - 单ticker, 5m为执行/数据粒度, 2h/15m/30m均由5m重采样得到
的"已收盘"K线序列驱动(不是live forming candle - MA排列判断需要用确认过
的收盘K线，不需要像MACD/ATR那样争分夺秒地在未收盘蜡烛上抢跑)。

仓位不是v1.0引擎那种"一把梭全平"的二元状态机 - 一批仓位内部有两条腿:
50%固定盈亏比部分止盈(leg A), 剩余50%用吊灯止损跑趋势(leg B)。同一时间
只允许一个"批次"存活 - 批次内两条腿都出场(或触发原始止损整批离场)之前
不接受新信号，无论首仓还是回补都是清空后才能开下一批，仓位管理不会出现
分数级叠加。
"""

from dataclasses import dataclass, field

import pandas as pd

from src.config import BacktestConfig
from src.data.resample import closed_bar_positions, resample_ohlcv
from src.indicators import atr as atr_series
from src.rules.cooldown import CooldownManager
from src.rules.ma_filter import MultiTimeframeFilter
from src.rules.pullback_entry import MarketSnapshot, PullbackEntryEngine
from src.rules.stop_loss import StopLossCalculator
from src.rules.take_profit import TakeProfitManager

_HIGHER_TF_RULES = {"15m": "15min", "30m": "30min", "2h": "2h"}


@dataclass
class Trade:
    direction: str  # "long" | "short"
    entry_bar_idx: int
    entry_price: float
    exit_bar_idx: int
    exit_price: float
    reason: str  # "SL" | "TP"(部分止盈) | "PROTECTIVE_SL"(护盈止损)
    size_fraction: float  # 这条腿占整批仓位的比例
    signal_type: str  # "open" | "reentry"

    @property
    def pnl_pct(self) -> float:
        move = (
            (self.exit_price - self.entry_price) if self.direction == "long"
            else (self.entry_price - self.exit_price)
        )
        return move / self.entry_price


@dataclass
class PullbackBacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


@dataclass
class _OpenBatch:
    direction: str
    entry_bar_idx: int
    entry_price: float
    stop_loss: float
    signal_type: str
    partial_taken: bool = False
    trailing_stop: float | None = None
    extreme_since_entry: float = 0.0


def run_pullback_backtest(
    df_5m: pd.DataFrame,
    initial_capital: float = 10_000.0,
    filter: MultiTimeframeFilter | None = None,
    entry_engine: PullbackEntryEngine | None = None,
    stop_calc: StopLossCalculator | None = None,
    tp_manager: TakeProfitManager | None = None,
    cooldown: CooldownManager | None = None,
    atr_period: int = 14,
    config: BacktestConfig | None = None,
    batch_audit: list | None = None,
) -> PullbackBacktestResult:
    df_5m = df_5m.reset_index(drop=True)
    # cfg first - the default CooldownManager below is constructed from it.
    cfg = config or BacktestConfig()
    cooldown = cooldown or CooldownManager(
        count_distinct_2h_bars=cfg.cooldown_counts_2h_bars
    )
    filter = filter or MultiTimeframeFilter()
    entry_engine = entry_engine or PullbackEntryEngine(filter, cooldown)
    stop_calc = stop_calc or StopLossCalculator()
    tp_manager = tp_manager or TakeProfitManager()

    resampled = {tf: resample_ohlcv(df_5m, rule) for tf, rule in _HIGHER_TF_RULES.items()}
    closed_idx = {
        tf: closed_bar_positions(resampled[tf], rule, df_5m["timestamp"])
        for tf, rule in _HIGHER_TF_RULES.items()
    }
    atr_full = atr_series(df_5m, atr_period)

    def closed_klines(tf: str, i: int) -> pd.DataFrame | None:
        pos = closed_idx[tf][i]
        return None if pos < 0 else resampled[tf].iloc[: pos + 1]

    n = len(df_5m)
    trades: list[Trade] = []
    equity = [0.0] * n
    capital = initial_capital
    batch: _OpenBatch | None = None
    had_prior_batch = False

    def attempt_entry(
        i: int, row: pd.Series,
        klines_2h: pd.DataFrame | None,
        klines_15m: pd.DataFrame | None,
        klines_30m: pd.DataFrame | None,
    ) -> _OpenBatch | None:
        # One code path, two conventions (see src/config.py):
        #   completed-bar (default): 5m reads only closed candles (<= i-1),
        #     entry-stop ATR uses i-1, fill is THIS bar's open - knowable at
        #     bar i's open, so called before the exit block.
        #   forming-bar (pre-2.1): 5m read includes close[i], ATR uses i, fill
        #     is the NEXT bar's open - called after the exit block.
        if klines_2h is None or klines_15m is None or klines_30m is None:
            return None
        if i < cfg.warmup_discard_bars:  # Item 2.6: no entries during the discard stretch
            return None
        if cfg.entry_on_completed_bar:
            fill_idx = i
            small_5m = df_5m.iloc[:i]
            atr_dec = atr_full.iloc[i - 1] if i > 0 else float("nan")
        else:
            fill_idx = i + 1
            small_5m = df_5m.iloc[: i + 1]
            atr_dec = atr_full.iloc[i]
        fill_idx += cfg.entry_lag_bars  # Item 2 lag-curve diagnostic; 0 in the real strategy
        if fill_idx >= n:
            return None
        snapshot = MarketSnapshot(
            tf_2h=klines_2h,
            small_tf={"5m": small_5m, "15m": klines_15m, "30m": klines_30m},
            close=row["close"], atr=atr_dec,
        )
        signal = entry_engine.on_bar(snapshot)
        if signal is not None and pd.notna(atr_dec):
            entry_price = df_5m["open"].iloc[fill_idx]
            stop = stop_calc.calc(entry_price, signal.direction, atr_dec, klines_30m)
            if batch_audit is not None:  # measurement only; no effect on results
                batch_audit.append({
                    "entry_bar_idx": fill_idx, "direction": signal.direction,
                    "entry_price": entry_price, "stop": stop,
                    "target": tp_manager.partial_trigger_price(entry_price, signal.direction, stop),
                })
            return _OpenBatch(
                direction=signal.direction, entry_bar_idx=fill_idx, entry_price=entry_price,
                stop_loss=stop, signal_type="reentry" if had_prior_batch else "open",
                extreme_since_entry=entry_price,
            )
        return None

    def gap_filled(level: float, is_long: bool, bar_open: float) -> float:
        # Item 3.5 (A2): fill a stop at the worse of the line and the bar's
        # open, so a gap through the stop is not filled at the (better) line
        # the market never traded. Off -> exact line (pre-2.1).
        if not cfg.gap_fill_exits:
            return level
        return min(level, bar_open) if is_long else max(level, bar_open)

    for i in range(n):
        row = df_5m.iloc[i]
        klines_2h = closed_klines("2h", i)
        klines_15m = closed_klines("15m", i)
        klines_30m = closed_klines("30m", i)

        if klines_2h is not None:
            cooldown.on_bar(klines_2h)

        # completed-bar convention: decide & fill at THIS bar's open, before exits
        if cfg.entry_on_completed_bar and batch is None:
            batch = attempt_entry(i, row, klines_2h, klines_15m, klines_30m)

        if batch is not None and batch.entry_bar_idx <= i:
            # entry_bar_idx <= i: process exits only once the batch has filled.
            # No-op at lag 0 (a batch enters on the bar it is created); matters
            # only for the Item 2 lag diagnostic, where the batch is created
            # early but fills entry_lag_bars later.
            # invariant: once a batch is open, a 2h bar must have already
            # closed (attempt_entry returns None unless all klines are ready)
            # and closed_bar_positions is monotonically non-decreasing, so
            # klines_2h can never regress to None on a later bar.
            assert klines_2h is not None
            is_long = batch.direction == "long"
            bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
            if not cfg.trail_extreme_prev_bar:
                # pre-A3: fold this bar's own extreme in BEFORE the trail is
                # computed and tested against this same bar (optimistic - it
                # assumes the bar's high came before its low).
                batch.extreme_since_entry = (
                    max(batch.extreme_since_entry, bar_high) if is_long
                    else min(batch.extreme_since_entry, bar_low)
                )

            if not batch.partial_taken:
                trigger = tp_manager.partial_trigger_price(batch.entry_price, batch.direction, batch.stop_loss)
                if cfg.range_based_exit_trigger:  # Item 3 (A1): hit if the bar's range reaches the level
                    hit_sl = bar_low <= batch.stop_loss if is_long else bar_high >= batch.stop_loss
                    hit_tp = bar_high >= trigger if is_long else bar_low <= trigger
                else:#old, wrong
                    hit_sl = bar_close <= batch.stop_loss if is_long else bar_close >= batch.stop_loss
                    hit_tp = bar_close >= trigger if is_long else bar_close <= trigger
                # Item 3 (both-hit): stop and target in one bar (only possible
                # range-based) -> pessimistic takes the stop first.
                take_sl = hit_sl and (cfg.pessimistic_both_hit or not hit_tp)
                if take_sl:
                    trades.append(Trade(
                        direction=batch.direction, entry_bar_idx=batch.entry_bar_idx, entry_price=batch.entry_price,
                        exit_bar_idx=i, exit_price=gap_filled(batch.stop_loss, is_long, bar_open),
                        reason="SL", size_fraction=1.0, signal_type=batch.signal_type,
                    ))
                    capital *= 1 + trades[-1].pnl_pct
                    cooldown.on_trade_closed(batch.direction, "SL")
                    batch = None
                    had_prior_batch = True
                elif hit_tp:
                    trades.append(Trade(
                        direction=batch.direction, entry_bar_idx=batch.entry_bar_idx, entry_price=batch.entry_price,
                        exit_bar_idx=i, exit_price=trigger, reason="TP", size_fraction=tp_manager.partial_ratio,
                        signal_type=batch.signal_type,
                    ))
                    capital *= 1 + tp_manager.partial_ratio * trades[-1].pnl_pct
                    batch.partial_taken = True
                    batch.trailing_stop = batch.stop_loss
            else:
                # use i-1's ATR, not i's - i's ATR needs bar i's own close,
                # so it isn't known until bar i has already happened, and
                # can't be used to decide whether bar i's own price (checked
                # right below) breaches a stop line that depends on it.
                atr_prev = atr_full.iloc[i - 1] if i > 0 else float("nan")
                if pd.notna(atr_prev):
                    batch.trailing_stop = tp_manager.chandelier_stop(
                        batch.direction, batch.extreme_since_entry, atr_prev, batch.stop_loss
                    )
                if cfg.range_based_exit_trigger:
                    hit_trail = bar_low <= batch.trailing_stop if is_long else bar_high >= batch.trailing_stop
                else:#old, wrong
                    hit_trail = bar_close <= batch.trailing_stop if is_long else bar_close >= batch.trailing_stop
                if hit_trail:
                    remaining = 1.0 - tp_manager.partial_ratio
                    trades.append(Trade(
                        direction=batch.direction, entry_bar_idx=batch.entry_bar_idx, entry_price=batch.entry_price,
                        exit_bar_idx=i, exit_price=gap_filled(batch.trailing_stop, is_long, bar_open),
                        reason="PROTECTIVE_SL", size_fraction=remaining, signal_type=batch.signal_type,
                    ))
                    capital *= 1 + remaining * trades[-1].pnl_pct
                    cooldown.on_trade_closed(batch.direction, "PROTECTIVE_SL")
                    batch = None
                    had_prior_batch = True

            # A3: fold this bar's high/low in only AFTER the trail has been
            # tested, so the level tested against bar i never depends on bar i.
            # Guarded - an exit above may have closed the batch.
            if cfg.trail_extreme_prev_bar and batch is not None:
                batch.extreme_since_entry = (
                    max(batch.extreme_since_entry, bar_high) if is_long
                    else min(batch.extreme_since_entry, bar_low)
                )

        # forming-bar convention: decide off close[i], fill at the NEXT bar's open
        if not cfg.entry_on_completed_bar and batch is None:
            batch = attempt_entry(i, row, klines_2h, klines_15m, klines_30m)

        unrealized = 0.0
        # Forming-bar convention: a batch opened this bar has entry_bar_idx ==
        # i+1 and isn't live on bar i yet, so equity stays flat until the bar
        # it actually enters on (no swing on a fill price that hasn't happened).
        # Completed-bar convention: it enters at open[i] (entry_bar_idx == i)
        # and is marked from bar i. The `entry_bar_idx <= i` guard covers both.
        if batch is not None and batch.entry_bar_idx <= i:
            price = df_5m["close"].iloc[i]
            move = (
                (price - batch.entry_price) if batch.direction == "long" else (batch.entry_price - price)
            )
            frac = (1 - tp_manager.partial_ratio) if batch.partial_taken else 1.0
            unrealized = frac * move / batch.entry_price
        equity[i] = capital * (1 + unrealized)

    equity_curve = pd.Series(equity, index=df_5m["timestamp"])
    return PullbackBacktestResult(trades=trades, equity_curve=equity_curve)
