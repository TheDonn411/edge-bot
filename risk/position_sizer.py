"""
PositionSizer — converts signal scores into dollar/share allocations.

Sizing method: fractional Kelly criterion.
  Kelly fraction = edge / odds  (simplified: score * kelly_fraction)
  Position size  = capped at max_portfolio_pct of current equity.
"""

from __future__ import annotations
from loguru import logger


class PositionSizer:
    def __init__(self, cfg: dict):
        self.risk_cfg = cfg["risk"]

    def size(
        self,
        symbol: str,
        score: float,
        price: float,
        equity: float,
    ) -> dict:
        """
        Returns a dict with keys: symbol, shares, dollar_value, weight.
        score: normalised signal score in [0, 1].
        """
        max_pct = self.risk_cfg["max_portfolio_pct"]
        kelly_frac = self.risk_cfg["kelly_fraction"]

        # Map score [0,1] to bet size, apply fractional Kelly
        raw_weight = (score - 0.5) * 2 * kelly_frac  # [-kelly_frac, +kelly_frac]
        weight = max(0.0, min(raw_weight, max_pct))   # floor at 0, cap at max_pct

        dollar_value = equity * weight
        shares = int(dollar_value / price) if price > 0 else 0

        result = {
            "symbol": symbol,
            "shares": shares,
            "dollar_value": round(dollar_value, 2),
            "weight": round(weight, 4),
        }
        logger.debug(f"Sized {symbol}: {result}")
        return result

    def size_all(
        self,
        picks: list[str],
        scores: dict[str, float],
        prices: dict[str, float],
        equity: float,
    ) -> list[dict]:
        return [
            self.size(sym, scores.get(sym, 0.5), prices.get(sym, 1.0), equity)
            for sym in picks
        ]
