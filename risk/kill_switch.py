"""
KillSwitch — monitors portfolio-level and position-level risk limits.

Triggers:
  1. Portfolio drawdown exceeds max_drawdown_pct  → halt all trading.
  2. Daily P&L loss exceeds daily_loss_limit_pct  → halt for the day.
  3. Individual position loss exceeds stop_loss_pct → close that position.
"""

from __future__ import annotations
from loguru import logger


class KillSwitch:
    def __init__(self, cfg: dict):
        self.risk_cfg = cfg["risk"]
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset(self):
        """Call at start of each trading day to clear the daily halt."""
        self._halted = False
        logger.info("KillSwitch reset for new trading day.")

    def check_portfolio(self, peak_equity: float, current_equity: float, daily_pnl: float) -> bool:
        """
        Returns True if trading should continue, False if halted.
        peak_equity: highest equity value since inception.
        daily_pnl:   today's P&L as a fraction of start-of-day equity.
        """
        if self._halted:
            return False

        max_dd = self.risk_cfg["max_drawdown_pct"]
        daily_limit = self.risk_cfg["daily_loss_limit_pct"]

        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown >= max_dd:
                logger.warning(
                    f"KillSwitch: max drawdown breached ({drawdown:.1%} >= {max_dd:.1%}). HALTING."
                )
                self._halted = True
                return False

        if daily_pnl <= -daily_limit:
            logger.warning(
                f"KillSwitch: daily loss limit breached ({daily_pnl:.1%}). HALTING for the day."
            )
            self._halted = True
            return False

        return True

    def check_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        commission_per_share: float = 0.0,
    ) -> bool:
        """
        Returns True if position should be held, False if stop-loss triggered.
        commission_per_share: entry commission paid per share (included in cost basis).
        """
        stop_loss = self.risk_cfg["stop_loss_pct"]
        if entry_price <= 0:
            return True

        cost_basis = entry_price + commission_per_share
        loss_pct = (cost_basis - current_price) / cost_basis
        if loss_pct >= stop_loss:
            logger.warning(
                f"KillSwitch: stop-loss triggered for {symbol} "
                f"(loss={loss_pct:.1%} >= {stop_loss:.1%})."
            )
            return False

        return True
