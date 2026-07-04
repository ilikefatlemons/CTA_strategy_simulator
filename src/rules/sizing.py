"""Phase 3: inverse-volatility position sizing (standard CTA vol-targeting)."""

from src.rules.base import PositionSizer


class InverseVolatilitySizer(PositionSizer):
    def weights(self, vols: dict[str, float]) -> dict[str, float]:
        # a zero vol estimate (e.g. a flat price for the whole window) would make
        # 1/vol blow up and dominate the portfolio - exclude it instead of sizing off it
        inv_vols = {symbol: 1 / vol for symbol, vol in vols.items() if vol > 0}
        total = sum(inv_vols.values())
        return {symbol: inv_vol / total for symbol, inv_vol in inv_vols.items()}
