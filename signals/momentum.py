"""
MomentumSignal — rate-of-change over a configurable lookback window.

Score is the normalised ROC clipped to [-1, 1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseSignal


class MomentumSignal(BaseSignal):
    def compute(self, df: pd.DataFrame) -> float:
        lookback = self.cfg.get("lookback", 20)
        if len(df) < lookback + 1:
            return 0.0

        close = df["Close"]
        roc = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]

        # Normalise: assume ±20% move over the window = saturated signal
        score = float(np.clip(roc / 0.20, -1.0, 1.0))
        return score
