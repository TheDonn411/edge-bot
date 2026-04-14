"""
BaseSignal — interface every signal module must implement.

compute(df) returns a float score in [-1, 1]:
  -1 = strong sell, 0 = neutral, +1 = strong buy
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> float:
        """Compute a signal score in [-1, 1] from an OHLCV DataFrame."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
