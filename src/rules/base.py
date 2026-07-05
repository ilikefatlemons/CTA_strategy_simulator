"""
Phase 2: abstract base classes for the pluggable strategy rule framework.

Every symbol runs the exact same rule instances (see README section 3 —
execution rules never fork per-symbol), so these classes hold no per-symbol
state themselves; per-position state is passed in explicitly via `Position`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd


class Signal(Enum):
    LONG = auto()
    SHORT = auto()
    NONE = auto()


class Side(Enum):
    LONG = auto()
    SHORT = auto()


@dataclass
class Position:
    side: Side
    entry_price: float
    entry_bar_idx: int


class EntryRule(ABC):
    @abstractmethod
    def on_bar(self, history: pd.DataFrame) -> Signal:
        """Decide entry signal given all bars up to and including the current one."""


class ExitRule(ABC):
    @abstractmethod
    def exit_reason(self, history: pd.DataFrame, position: Position) -> str | None:
        """
        Decide whether an open position should be closed on the current bar.

        Returns a short reason code (e.g. "TP"/"SL") if it should, otherwise
        None. Concrete reason codes are defined by each rule implementation.
        """

    def should_exit(self, history: pd.DataFrame, position: Position) -> bool:
        return self.exit_reason(history, position) is not None


class ReentryRule(ABC):
    @abstractmethod
    def can_reenter(self, history: pd.DataFrame, bars_since_exit: int) -> bool:
        """Gate whether the entry rule is even consulted while in cooldown."""


class PositionSizer(ABC):
    @abstractmethod
    def weights(self, vols: dict[str, float]) -> dict[str, float]:
        """Map symbol -> trailing vol estimate to symbol -> target portfolio weight."""
