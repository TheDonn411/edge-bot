"""
MLScoreSignal — XGBoost classifier trained on engineered features.

Returns probability of up-move in [0, 1] mapped to score in [-1, 1].
Disabled by default until a model is trained; returns 0.0 as neutral stub.

Training workflow (manual, outside this module):
  1. Run backtest/feature_builder.py to generate training data.
  2. Train with: xgb.train(...) and save to config['model_path'].
  3. Flip signals.ml_score.enabled = true in config.yaml.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from .base import BaseSignal


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature set used at inference time."""
    f = pd.DataFrame(index=df.index)
    f["roc_5"]  = df["Close"].pct_change(5)
    f["roc_20"] = df["Close"].pct_change(20)
    f["vol_10"] = df["Close"].pct_change().rolling(10).std()
    f["rsi_14"] = _rsi(df["Close"], 14)
    f["adv_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return f.dropna()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class MLScoreSignal(BaseSignal):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._model = None
        model_path = Path(cfg.get("model_path", ""))
        if model_path.exists():
            try:
                import xgboost as xgb
                self._model = xgb.Booster()
                self._model.load_model(str(model_path))
                logger.info(f"MLScoreSignal loaded model from {model_path}")
            except Exception as exc:
                logger.warning(f"MLScoreSignal: failed to load model — {exc}")

    def compute(self, df: pd.DataFrame) -> float:
        if self._model is None:
            return 0.0  # neutral stub

        import xgboost as xgb
        features = _build_features(df)
        if features.empty:
            return 0.0

        row = features.iloc[[-1]]
        dmatrix = xgb.DMatrix(row)
        prob = float(self._model.predict(dmatrix)[0])  # P(up)
        return float(np.clip((prob - 0.5) * 2, -1.0, 1.0))
