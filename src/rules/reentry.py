"""Phase 2: fixed-bar cooldown re-entry rule."""

import pandas as pd

from src.rules.base import ReentryRule


class CooldownReentry(ReentryRule):
    def __init__(self, cooldown_bars: int = 5):
        self.cooldown_bars = cooldown_bars

    def can_reenter(self, history: pd.DataFrame, bars_since_exit: int) -> bool:
        return bars_since_exit >= self.cooldown_bars
