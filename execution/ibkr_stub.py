"""
IBKRStub — paper-trading stub that simulates IBKR order execution.

All orders are filled immediately at the last known price plus configurable
slippage. No real money or network calls are involved.

Replace with ibkr_live.py (using ib_insync) when ready for live trading.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from loguru import logger
from .broker_base import BrokerBase
from portfolio import PortfolioTracker


class IBKRStub(BrokerBase):
    def __init__(self, cfg: dict):
        self.cfg = cfg["execution"]
        self._equity = cfg["backtest"]["initial_capital"]
        self._cash = self._equity
        self._positions: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}
        self.tracker = PortfolioTracker(
            cfg["backtest"]["initial_capital"],
            cfg.get("portfolio", {}),
            mode="paper" if self.cfg.get("paper_trading", True) else "live",
        )

    def connect(self) -> None:
        logger.info("IBKRStub: connected (paper trading mode)")

    def get_account(self) -> dict:
        position_value = sum(
            p["shares"] * p["last_price"] for p in self._positions.values()
        )
        return {
            "equity": self._cash + position_value,
            "cash": self._cash,
            "positions": self._positions,
        }

    def place_order(
        self,
        symbol: str,
        shares: int,
        side: str,
        order_type: str = "MKT",
        limit_price: float | None = None,
    ) -> dict:
        if shares <= 0:
            logger.warning(f"IBKRStub: ignoring zero/negative share order for {symbol}")
            return {}

        slippage = self.cfg.get("slippage_bps", 5) / 10_000
        commission = self.cfg.get("commission_per_share", 0.005)

        # Use limit_price as fill price if provided, else use last known price
        fill_price = limit_price or self._positions.get(symbol, {}).get("last_price", 0.0)
        if fill_price == 0.0:
            logger.warning(f"IBKRStub: no price available for {symbol}, order skipped")
            return {}

        if side == "BUY":
            fill_price *= (1 + slippage)
            cost = fill_price * shares + commission * shares
            if cost > self._cash:
                logger.warning(f"IBKRStub: insufficient cash for {symbol} BUY")
                return {}
            self._cash -= cost
            if symbol in self._positions:
                existing = self._positions[symbol]
                total_shares = existing["shares"] + shares
                avg_cost = (existing["avg_cost"] * existing["shares"] + fill_price * shares) / total_shares
                self._positions[symbol] = {"shares": total_shares, "avg_cost": avg_cost, "last_price": fill_price}
            else:
                self._positions[symbol] = {"shares": shares, "avg_cost": fill_price, "last_price": fill_price}

        elif side == "SELL":
            fill_price *= (1 - slippage)
            pos = self._positions.get(symbol)
            if not pos or pos["shares"] < shares:
                logger.warning(f"IBKRStub: cannot sell {shares} shares of {symbol} (insufficient position)")
                return {}
            proceeds = fill_price * shares - commission * shares
            self._cash += proceeds
            remaining = pos["shares"] - shares
            if remaining == 0:
                del self._positions[symbol]
            else:
                self._positions[symbol]["shares"] = remaining

        self.tracker.record_fill(
            datetime.now(timezone.utc),
            symbol,
            side,
            shares,
            fill_price,
            commission=commission * shares,
            metadata={"order_type": order_type},
        )

        order_id = str(uuid.uuid4())[:8]
        order = {"order_id": order_id, "symbol": symbol, "shares": shares, "side": side, "fill_price": fill_price}
        self._orders[order_id] = order
        logger.info(f"IBKRStub: {side} {shares} {symbol} @ {fill_price:.2f} | order={order_id}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False

    def get_positions(self) -> dict[str, dict]:
        return dict(self._positions)

    def update_prices(self, prices: dict[str, float]):
        """Feed latest prices so the stub can compute position values."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol]["last_price"] = price
        self.tracker.mark_to_market(datetime.now(timezone.utc), prices)
