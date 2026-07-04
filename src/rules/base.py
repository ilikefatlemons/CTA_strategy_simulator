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
    def should_exit(self, history: pd.DataFrame, position: Position) -> bool:
        """Decide whether an open position should be closed on the current bar."""


class ReentryRule(ABC):
    @abstractmethod
    def can_reenter(self, history: pd.DataFrame, bars_since_exit: int) -> bool:
        """Gate whether the entry rule is even consulted while in cooldown."""
