"""
Phase 3: multi-symbol portfolio backtest with inverse-volatility sizing and
daily rebalance (including resizing already-open positions).

Each symbol runs the exact same rule instances and the exact same per-bar
state machine as Phase 2 (FLAT -> ENTERED -> COOLDOWN -> FLAT), but instead of
compounding a % return on notional capital, positions are tracked as actual
signed share counts against one shared cash ledger. That's what lets a daily
rebalance resize an open position: `resize()` just moves current shares to a
new target and realizes the P&L on the delta, uniformly covering entries
(target != 0 from 0), exits (target == 0), and resizes (target != current).
"""

from dataclasses import dataclass, field

import pandas as pd

from src.engine.backtest import State, Strategy
from src.engine.vol_estimator import daily_closes, rolling_daily_vol
from src.rules.base import Position, PositionSizer, Side, Signal


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_bar_idx: int
    entry_price: float
    exit_bar_idx: int
    exit_price: float

    @property
    def pnl(self) -> float:
        if self.side is Side.LONG:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    @property
    def return_pct(self) -> float:
        return self.pnl / self.entry_price


@dataclass
class PortfolioBacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)


def align_to_common_index(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Restrict every symbol's bars to the timestamps present in all symbols.

    Exposed so callers comparing a standalone single-symbol backtest against
    the portfolio backtest can run both over the exact same bars - otherwise
    a standalone run would see extra bars the portfolio run drops, which can
    shift its entry/exit signals and confound the comparison with more than
    just the sizing difference.
    """
    symbols = list(dfs.keys())
    common_idx = None
    for s in symbols:
        idx = pd.DatetimeIndex(dfs[s]["timestamp"])
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    return {s: dfs[s].set_index("timestamp").loc[common_idx].reset_index() for s in symbols}


def run_portfolio_backtest(
    dfs: dict[str, pd.DataFrame],
    strategy: Strategy,
    sizer: PositionSizer,
    vol_window: int = 20,
    initial_capital: float = 10_000.0,
    min_holding_bars: int = 2,
) -> PortfolioBacktestResult:
    symbols = list(dfs.keys())

    # vol estimates use each symbol's full own history, not the aligned subset
    vol_series = {s: rolling_daily_vol(daily_closes(dfs[s]), vol_window) for s in symbols}

    aligned = align_to_common_index(dfs)
    common_idx = pd.DatetimeIndex(aligned[symbols[0]]["timestamp"])
    dates = common_idx.date

    T = len(common_idx)
    cash = initial_capital
    shares = {s: 0.0 for s in symbols}
    state = {s: State.FLAT for s in symbols}
    position: dict[str, Position | None] = {s: None for s in symbols}
    pending_signal = {s: Signal.NONE for s in symbols}
    bars_since_exit = {s: 0 for s in symbols}
    current_weights = {s: 0.0 for s in symbols}

    trades: list[Trade] = []
    equity = [0.0] * T
    weights_rows = []
    prev_date = None

    def current_equity(i: int) -> float:
        eq = cash
        for s in symbols:
            eq += shares[s] * aligned[s]["close"].iloc[i]
        return eq

    for i in range(T):
        today = dates[i]
        is_new_day = today != prev_date
        prev_date = today

        if is_new_day:
            vols_today = {
                s: vol_series[s].get(today)
                for s in symbols
                if pd.notna(vol_series[s].get(today))
            }
            weights_today = sizer.weights(vols_today) if vols_today else {}
            current_weights = {s: weights_today.get(s, 0.0) for s in symbols}
            weights_rows.append({"date": today, **current_weights})

            eq = current_equity(i)
            for s in symbols:
                if state[s] is State.ENTERED and position[s] is not None:
                    open_price = aligned[s]["open"].iloc[i]
                    weight = current_weights[s]
                    if weight <= 0:
                        # vol estimate unavailable/excluded today - can no longer
                        # size this position, so close it out properly instead of
                        # silently zeroing shares while leaving state ENTERED
                        delta = 0.0 - shares[s]
                        cash -= delta * open_price
                        shares[s] = 0.0
                        trades.append(
                            Trade(
                                symbol=s,
                                side=position[s].side,
                                entry_bar_idx=position[s].entry_bar_idx,
                                entry_price=position[s].entry_price,
                                exit_bar_idx=i,
                                exit_price=open_price,
                            )
                        )
                        position[s] = None
                        state[s] = State.COOLDOWN
                        bars_since_exit[s] = 0
                        continue
                    side_sign = 1 if position[s].side is Side.LONG else -1
                    target_shares = side_sign * weight * eq / open_price
                    delta = target_shares - shares[s]
                    cash -= delta * open_price
                    shares[s] = target_shares

        for s in symbols:
            df_s = aligned[s]
            history = df_s.iloc[: i + 1]
            open_price = df_s["open"].iloc[i]
            close_price = df_s["close"].iloc[i]

            if state[s] is State.FLAT and pending_signal[s] is not Signal.NONE:
                side = Side.LONG if pending_signal[s] is Signal.LONG else Side.SHORT
                weight = current_weights[s]
                if weight > 0:
                    eq = current_equity(i)
                    side_sign = 1 if side is Side.LONG else -1
                    target_shares = side_sign * weight * eq / open_price
                    delta = target_shares - shares[s]
                    cash -= delta * open_price
                    shares[s] = target_shares
                    position[s] = Position(side=side, entry_price=open_price, entry_bar_idx=i)
                    state[s] = State.ENTERED
                pending_signal[s] = Signal.NONE

            elif state[s] is State.COOLDOWN:
                bars_since_exit[s] += 1
                if strategy.reentry_rule.can_reenter(history, bars_since_exit[s]):
                    state[s] = State.FLAT

            if state[s] is State.ENTERED and position[s] is not None:
                held_bars = i - position[s].entry_bar_idx
                if held_bars >= min_holding_bars and strategy.exit_rule.should_exit(
                    history, position[s]
                ):
                    delta = 0.0 - shares[s]
                    cash -= delta * close_price
                    shares[s] = 0.0
                    trades.append(
                        Trade(
                            symbol=s,
                            side=position[s].side,
                            entry_bar_idx=position[s].entry_bar_idx,
                            entry_price=position[s].entry_price,
                            exit_bar_idx=i,
                            exit_price=close_price,
                        )
                    )
                    position[s] = None
                    state[s] = State.COOLDOWN
                    bars_since_exit[s] = 0

            if state[s] is State.FLAT:
                signal = strategy.entry_rule.on_bar(history)
                if signal is not Signal.NONE and i + 1 < T:
                    pending_signal[s] = signal

        equity[i] = current_equity(i)

    equity_curve = pd.Series(equity, index=common_idx)
    weights_history = (
        pd.DataFrame(weights_rows).set_index("date") if weights_rows else pd.DataFrame()
    )
    return PortfolioBacktestResult(trades=trades, equity_curve=equity_curve, weights_history=weights_history)
