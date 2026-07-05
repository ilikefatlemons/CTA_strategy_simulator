"""
Phase 2: bar-by-bar backtest engine.

Signals are decided on bar t's close and filled at bar t+1's open, to avoid
lookahead bias. State machine per symbol: FLAT -> ENTERED -> EXIT -> COOLDOWN -> FLAT.
No cross-symbol sizing yet (fixed 1 unit, full capital reinvested each trade)
and no slippage/commission - those land in later phases.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd

from src.rules.base import EntryRule, ExitRule, Position, ReentryRule, Side, Signal


class State(Enum):
    FLAT = auto()
    ENTERED = auto()
    COOLDOWN = auto()


@dataclass
class Strategy:
    entry_rule: EntryRule
    exit_rule: ExitRule
    reentry_rule: ReentryRule


@dataclass
class Trade:
    side: Side
    entry_bar_idx: int
    entry_price: float
    exit_bar_idx: int
    exit_price: float
    reason: str

    @property
    def pnl(self) -> float:
        if self.side is Side.LONG:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    @property
    def return_pct(self) -> float:
        return self.pnl / self.entry_price


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    min_holding_bars: int = 2,
) -> BacktestResult:
    """
    min_holding_bars: minimum bars a position must be held before the exit rule
    is even consulted. Without this, MACD-cross entries on noisy 5m bars can
    get stopped out within the same bar they were filled on.
    """
    df = df.reset_index(drop=True)
    trades: list[Trade] = []
    equity = [0.0] * len(df)

    state = State.FLAT
    position: Position | None = None
    pending_signal: Signal = Signal.NONE
    bars_since_exit = 0
    capital = initial_capital

    for i in range(len(df)):
        history = df.iloc[: i + 1]

        if state is State.FLAT and pending_signal is not Signal.NONE:
            side = Side.LONG if pending_signal is Signal.LONG else Side.SHORT
            position = Position(side=side, entry_price=df["open"].iloc[i], entry_bar_idx=i)
            state = State.ENTERED
            pending_signal = Signal.NONE

        elif state is State.COOLDOWN:
            bars_since_exit += 1
            if strategy.reentry_rule.can_reenter(history, bars_since_exit):
                state = State.FLAT

        if state is State.ENTERED and position is not None:
            held_bars = i - position.entry_bar_idx
            reason = strategy.exit_rule.exit_reason(history, position) if held_bars >= min_holding_bars else None
            if reason:
                exit_price = df["close"].iloc[i]
                trade = Trade(
                    side=position.side,
                    entry_bar_idx=position.entry_bar_idx,
                    entry_price=position.entry_price,
                    exit_bar_idx=i,
                    exit_price=exit_price,
                    reason=reason,
                )
                trades.append(trade)
                capital *= 1 + trade.return_pct
                position = None
                state = State.COOLDOWN
                bars_since_exit = 0

        if state is State.FLAT:
            signal = strategy.entry_rule.on_bar(history)
            if signal is not Signal.NONE and i + 1 < len(df):
                pending_signal = signal

        unrealized_return = 0.0
        if state is State.ENTERED and position is not None:
            current_price = df["close"].iloc[i]
            if position.side is Side.LONG:
                unrealized_return = (current_price - position.entry_price) / position.entry_price
            else:
                unrealized_return = (position.entry_price - current_price) / position.entry_price
        equity[i] = capital * (1 + unrealized_return)

    equity_curve = pd.Series(equity, index=df["timestamp"] if "timestamp" in df else df.index)
    return BacktestResult(trades=trades, equity_curve=equity_curve)
