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
) -> PullbackBacktestResult:
    df_5m = df_5m.reset_index(drop=True)
    cooldown = cooldown or CooldownManager()
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

    for i in range(n):
        row = df_5m.iloc[i]
        klines_2h = closed_klines("2h", i)
        klines_15m = closed_klines("15m", i)
        klines_30m = closed_klines("30m", i)

        if klines_2h is not None:
            cooldown.on_bar(klines_2h)

        if batch is not None:
            # invariant: once a batch is open, a 2h bar must have already
            # closed (can_open below requires it before entry) and
            # closed_bar_positions is monotonically non-decreasing, so
            # klines_2h can never regress to None on a later bar.
            assert klines_2h is not None
            price = row["close"]
            batch.extreme_since_entry = (
                max(batch.extreme_since_entry, row["high"]) if batch.direction == "long"
                else min(batch.extreme_since_entry, row["low"])
            )

            if not batch.partial_taken:
                trigger = tp_manager.partial_trigger_price(batch.entry_price, batch.direction, batch.stop_loss)
                hit_tp = price >= trigger if batch.direction == "long" else price <= trigger
                hit_sl = price <= batch.stop_loss if batch.direction == "long" else price >= batch.stop_loss
                if hit_sl:
                    trades.append(Trade(
                        direction=batch.direction, entry_bar_idx=batch.entry_bar_idx, entry_price=batch.entry_price,
                        exit_bar_idx=i, exit_price=batch.stop_loss, reason="SL", size_fraction=1.0,
                        signal_type=batch.signal_type,
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
                hit_trail = (
                    price <= batch.trailing_stop if batch.direction == "long" else price >= batch.trailing_stop
                )
                if hit_trail:
                    remaining = 1.0 - tp_manager.partial_ratio
                    trades.append(Trade(
                        direction=batch.direction, entry_bar_idx=batch.entry_bar_idx, entry_price=batch.entry_price,
                        exit_bar_idx=i, exit_price=batch.trailing_stop, reason="PROTECTIVE_SL",
                        size_fraction=remaining, signal_type=batch.signal_type,
                    ))
                    capital *= 1 + remaining * trades[-1].pnl_pct
                    cooldown.on_trade_closed(batch.direction, "PROTECTIVE_SL")
                    batch = None
                    had_prior_batch = True

        if (
            batch is None and i + 1 < n
            and klines_2h is not None and klines_15m is not None and klines_30m is not None
        ):
            atr_now = atr_full.iloc[i]
            snapshot = MarketSnapshot(
                tf_2h=klines_2h,
                small_tf={"5m": df_5m.iloc[: i + 1], "15m": klines_15m, "30m": klines_30m},
                close=row["close"], atr=atr_now,
            )
            signal = entry_engine.on_bar(snapshot)
            if signal is not None and pd.notna(atr_now):
                entry_price = df_5m["open"].iloc[i + 1]
                stop = stop_calc.calc(entry_price, signal.direction, atr_now, klines_30m)
                batch = _OpenBatch(
                    direction=signal.direction, entry_bar_idx=i + 1, entry_price=entry_price,
                    stop_loss=stop, signal_type="reentry" if had_prior_batch else "open",
                    extreme_since_entry=entry_price,
                )

        unrealized = 0.0
        # `batch` may have just been opened above with entry_bar_idx == i+1
        # (fills at next bar's open, to avoid lookahead) - it isn't actually
        # live yet on bar i itself, so equity must stay flat until the bar
        # it really enters on, not swing based on a fill price that hasn't
        # happened yet.
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
