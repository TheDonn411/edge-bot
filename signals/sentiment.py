"""
SentimentSignal — FinBERT-based news sentiment score.

Disabled by default (requires ~500 MB model download on first run).
Feeds a list of recent headlines through ProsusAI/finbert and returns
the net positive-minus-negative score normalised to [-1, 1].

To enable:
  1. Set signals.sentiment.enabled = true in config.yaml.
  2. Provide a headline fetcher (e.g., yfinance news, NewsAPI, etc.).
"""

from __future__ import annotations
import numpy as np
from loguru import logger
from .base import BaseSignal


class SentimentSignal(BaseSignal):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            model_name = self.cfg.get("model_name", "ProsusAI/finbert")
            self._pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                top_k=None,
            )
            logger.info(f"SentimentSignal loaded {model_name}")
        except Exception as exc:
            logger.warning(f"SentimentSignal: could not load model — {exc}")

    def compute(self, df=None, headlines: list[str] | None = None) -> float:  # type: ignore[override]
        """
        headlines: list of news headline strings for the symbol.
        df is accepted but unused (kept for interface compatibility).
        """
        if not headlines:
            return 0.0

        self._load_model()
        if self._pipeline is None:
            return 0.0

        scores = []
        for headline in headlines[:20]:  # cap to avoid latency spikes
            result = self._pipeline(headline[:512])[0]
            label_map = {item["label"].lower(): item["score"] for item in result}
            net = label_map.get("positive", 0) - label_map.get("negative", 0)
            scores.append(net)

        return float(np.clip(np.mean(scores), -1.0, 1.0))
