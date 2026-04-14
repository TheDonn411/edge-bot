"""
PortfolioTracker — lightweight trade ledger + equity tracker.

Works with paper trading, backtests, or live brokers as long as you feed it
buy/sell fills and periodic price marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0
    last_price: float = 0.0


class PortfolioTracker:
    def __init__(
        self,
        initial_capital: float,
        cfg: dict | None = None,
        mode: str = "generic",
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.mode = mode
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.trades: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []

        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.trades_file = Path(cfg.get("trades_file", "logs/trades.csv"))
        self.snapshots_file = Path(cfg.get("snapshots_file", "logs/portfolio_snapshots.csv"))
        self.auto_save = bool(cfg.get("auto_save", True))

        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_file.parent.mkdir(parents=True, exist_ok=True)

    def record_fill(
        self,
        timestamp,
        symbol: str,
        side: str,
        shares: int,
        fill_price: float,
        commission: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        side = side.upper()
        shares = int(shares)
        fill_price = float(fill_price)
        commission = float(commission)
        metadata = metadata or {}

        if shares <= 0 or fill_price <= 0:
            raise ValueError("shares and fill_price must be positive")

        pos = self.positions.get(symbol, Position())
        realized_delta = 0.0

        if side == "BUY":
            gross = fill_price * shares
            self.cash -= gross + commission
            total_shares = pos.shares + shares
            avg_cost = (
                (pos.avg_cost * pos.shares + fill_price * shares) / total_shares
                if total_shares > 0 else 0.0
            )
            pos.shares = total_shares
            pos.avg_cost = avg_cost
            pos.last_price = fill_price
            self.positions[symbol] = pos
        elif side == "SELL":
            if pos.shares < shares:
                raise ValueError(f"cannot sell {shares} {symbol}; only {pos.shares} held")
            gross = fill_price * shares
            realized_delta = (fill_price - pos.avg_cost) * shares - commission
            self.cash += gross - commission
            pos.shares -= shares
            pos.last_price = fill_price
            if pos.shares == 0:
                del self.positions[symbol]
            else:
                self.positions[symbol] = pos
        else:
            raise ValueError(f"unsupported side: {side}")

        self.realized_pnl += realized_delta
        trade = {
            "timestamp": pd.Timestamp(timestamp),
            "mode": self.mode,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "fill_price": round(fill_price, 4),
            "commission": round(commission, 4),
            "cash_after": round(self.cash, 2),
            "realized_pnl_delta": round(realized_delta, 2),
            **metadata,
        }
        self.trades.append(trade)
        self._autosave()
        return trade

    def mark_to_market(
        self,
        timestamp,
        prices: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prices = prices or {}
        metadata = metadata or {}

        market_value = 0.0
        unrealized = 0.0
        for symbol, pos in list(self.positions.items()):
            if symbol in prices and prices[symbol] is not None:
                pos.last_price = float(prices[symbol])
            market_value += pos.shares * pos.last_price
            unrealized += (pos.last_price - pos.avg_cost) * pos.shares

        equity = self.cash + market_value
        snapshot = {
            "timestamp": pd.Timestamp(timestamp),
            "mode": self.mode,
            "cash": round(self.cash, 2),
            "market_value": round(market_value, 2),
            "equity": round(equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(self.realized_pnl + unrealized, 2),
            "open_positions": len(self.positions),
            **metadata,
        }
        self.snapshots.append(snapshot)
        self._autosave()
        return snapshot

    def summary(self) -> dict[str, float]:
        if self.snapshots:
            latest = self.snapshots[-1]
            return {
                "equity": latest["equity"],
                "cash": latest["cash"],
                "market_value": latest["market_value"],
                "realized_pnl": latest["realized_pnl"],
                "unrealized_pnl": latest["unrealized_pnl"],
                "total_pnl": latest["total_pnl"],
            }

        market_value = sum(pos.shares * pos.last_price for pos in self.positions.values())
        unrealized = sum((pos.last_price - pos.avg_cost) * pos.shares for pos in self.positions.values())
        equity = self.cash + market_value
        return {
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "market_value": round(market_value, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(self.realized_pnl + unrealized, 2),
        }

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def snapshots_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots)

    def _autosave(self) -> None:
        if not self.enabled or not self.auto_save:
            return
        if self.trades:
            pd.DataFrame(self.trades).to_csv(self.trades_file, index=False)
        if self.snapshots:
            pd.DataFrame(self.snapshots).to_csv(self.snapshots_file, index=False)
