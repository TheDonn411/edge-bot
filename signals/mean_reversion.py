"""
MeanReversionSignal — Bollinger Band z-score.

Positive score means price is below the lower band (buy the dip).
Negative score means price is above the upper band (fade the rip).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseSignal


class MeanReversionSignal(BaseSignal):
    def compute(self, df: pd.DataFrame) -> float:
        period = self.cfg.get("bb_period", 20)
        n_std = self.cfg.get("bb_std", 2.0)

        if len(df) < period:
            return 0.0

        close = df["Close"]
        sma = close.rolling(period).mean().iloc[-1]
        std = close.rolling(period).std().iloc[-1]

        if std == 0:
            return 0.0

        z = (close.iloc[-1] - sma) / std
        # Invert: negative z (below mean) → positive score
        score = float(np.clip(-z / n_std, -1.0, 1.0))
        return score
