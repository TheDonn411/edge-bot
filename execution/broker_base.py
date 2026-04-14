"""
BrokerBase — abstract interface for all broker integrations.

Every concrete broker must implement these five methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BrokerBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection / authenticate."""
        ...

    @abstractmethod
    def get_account(self) -> dict:
        """Return account info: {'equity': float, 'cash': float, 'positions': {...}}"""
        ...

    @abstractmethod
    def place_order(self, symbol: str, shares: int, side: str, order_type: str = "MKT") -> dict:
        """
        Place an order.
        side: 'BUY' | 'SELL'
        Returns order confirmation dict.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, dict]:
        """Return open positions: {symbol: {'shares': int, 'avg_cost': float}}"""
        ...
