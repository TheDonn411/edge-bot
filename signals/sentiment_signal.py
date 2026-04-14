"""
NewsSentimentSignal — multi-source news sentiment scorer.

Primary path  (no model download needed):
  Finnhub /news-sentiment endpoint → pre-computed bullish/bearish scores
  + Finnhub /company-news         → top headlines for logging

Fallback path (requires PyTorch + FinBERT ~500 MB):
  Alpha Vantage NEWS_SENTIMENT API → headlines
  ProsusAI/finbert                 → per-headline scoring

Recency and analyst-event weighting applied on the fallback path.
Results cached per ticker for cache_ttl_min minutes.
"""

from __future__ import annotations

import importlib.util
import time
from datetime import datetime, timezone, timedelta

import requests
from loguru import logger

_AV_NEWS_URL  = "https://www.alphavantage.co/query"
_FH_BASE_URL  = "https://finnhub.io/api/v1"

# Keywords that identify analyst rating events (weighted 2x)
_ANALYST_KEYWORDS = [
    "upgrade", "downgrade", "initiated", "overweight", "underweight",
    "outperform", "underperform", "buy rating", "sell rating",
    "price target", "analyst", "coverage",
]


class NewsSentimentSignal:
    def __init__(self, cfg: dict):
        sn_cfg = cfg.get("sentiment_news", cfg)
        self.finnhub_key: str  = sn_cfg.get("finnhub_key", "")
        self.av_key: str       = sn_cfg.get("alpha_vantage_key", "")
        self.model_name: str   = sn_cfg.get("model_name", "ProsusAI/finbert")
        self.cache_ttl: float  = float(sn_cfg.get("cache_ttl_min", 15)) * 60
        self.max_headlines: int = int(sn_cfg.get("max_headlines", 10))
        self.analyst_multiplier: float = float(sn_cfg.get("analyst_weight_multiplier", 2.0))

        bucket_cfg = sn_cfg.get("recency_buckets", {})
        self.bucket_hrs = [
            float(bucket_cfg.get("bucket_1_hrs", 2)),
            float(bucket_cfg.get("bucket_2_hrs", 24)),
            float(bucket_cfg.get("bucket_3_hrs", 168)),
        ]
        self.bucket_weights: list[float] = list(sn_cfg.get("bucket_weights", [1.0, 0.6, 0.3]))

        self._pipeline = None
        self._cache: dict[str, tuple[float, dict]] = {}

    # ── PRIMARY PATH: Alpha Vantage pre-scored sentiment ─────────────────────
    # AV's NEWS_SENTIMENT endpoint returns per-ticker sentiment scores
    # alongside each article — no local model required.

    def _av_compute(self, symbol: str) -> dict | None:
        """
        Alpha Vantage NEWS_SENTIMENT with pre-computed ticker_sentiment_score.
        Returns result dict or None on failure / missing key.
        """
        if not self.av_key:
            return None

        try:
            r = requests.get(
                _AV_NEWS_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": symbol,
                    "limit": self.max_headlines,
                    "apikey": self.av_key,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.debug(f"NewsSentimentSignal AV error [{symbol}]: {exc}")
            return None

        if "Information" in data:   # rate-limited
            logger.warning(f"NewsSentimentSignal: AV rate-limited for {symbol}")
            return None

        feed = data.get("feed", [])
        if not feed:
            return None

        weighted_scores: list[tuple[float, float, str]] = []  # (score, recency_w, title)

        for item in feed:
            title    = item.get("title", "")
            time_str = item.get("time_published", "")
            if not title:
                continue

            recency_w = self._recency_weight(time_str)
            if recency_w == 0.0:
                continue

            # Prefer ticker-specific score; fall back to overall article score
            ticker_score: float | None = None
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    try:
                        ticker_score = float(ts["ticker_sentiment_score"])
                    except (KeyError, ValueError):
                        pass
                    break

            raw = ticker_score if ticker_score is not None else float(
                item.get("overall_sentiment_score", 0.0)
            )
            # AV scores are roughly in [-0.35, +0.35]; normalise to [-1, 1]
            raw_norm = max(-1.0, min(1.0, raw * 3.0))

            analyst_w = self.analyst_multiplier if self._is_analyst_event(title) else 1.0
            weighted_scores.append((raw_norm * recency_w * analyst_w, recency_w, title))

        if not weighted_scores:
            return None

        weighted_scores.sort(key=lambda x: abs(x[0]), reverse=True)
        top_headlines = [t for _, _, t in weighted_scores[:3]]
        avg_raw = sum(s for s, _, _ in weighted_scores) / len(weighted_scores)
        normalised = float((avg_raw + 1) / 2)   # [-1,1] → [0,1]

        result = {
            "sentiment_score": round(normalised, 4),
            "raw_score":       round(avg_raw, 4),
            "headline_count":  len(weighted_scores),
            "top_headlines":   top_headlines,
            "symbol":          symbol,
            "source":          "alpha_vantage",
        }
        logger.info(
            f"NewsSentimentSignal [{symbol}] AV: "
            f"score={normalised:.3f} ({len(weighted_scores)} articles)"
        )
        return result

    def _is_analyst_event(self, title: str) -> bool:
        lower = title.lower()
        return any(kw in lower for kw in _ANALYST_KEYWORDS)

    # ── SECONDARY PATH: Finnhub company-news headlines ────────────────────────
    # Finnhub /news-sentiment is a premium endpoint (403 on free tier).
    # /company-news is free but returns raw headlines without sentiment scores.
    # We use it only as a fallback headlines source when AV is unavailable.

    def _finnhub_headlines_only(self, symbol: str) -> list[str]:
        """Return recent headline strings from Finnhub /company-news."""
        if not self.finnhub_key:
            return []
        try:
            today    = datetime.now(timezone.utc).date()
            week_ago = (today - timedelta(days=7)).isoformat()
            r = requests.get(
                f"{_FH_BASE_URL}/company-news",
                params={"symbol": symbol, "from": week_ago,
                        "to": today.isoformat(), "token": self.finnhub_key},
                timeout=10,
            )
            if r.status_code == 200:
                return [item["headline"] for item in r.json()[:self.max_headlines]
                        if item.get("headline")]
        except Exception:
            pass
        return []

    # ── FALLBACK PATH: Alpha Vantage + FinBERT ────────────────────────────────

    def _load_pipeline(self) -> bool:
        if self._pipeline is not None:
            return True
        # Avoid lengthy Hugging Face retries when the local ML runtime
        # needed for FinBERT is not installed.
        if not any(importlib.util.find_spec(pkg) for pkg in ("torch", "tensorflow", "flax")):
            logger.debug("NewsSentimentSignal: no local ML runtime for FinBERT fallback")
            return False
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
            )
            logger.info(f"NewsSentimentSignal: loaded {self.model_name}")
            return True
        except Exception as exc:
            logger.debug(f"NewsSentimentSignal: FinBERT unavailable — {exc}")
            return False

    def _fetch_av_headlines(self, symbol: str) -> list[dict]:
        if not self.av_key:
            return []
        try:
            resp = requests.get(
                _AV_NEWS_URL,
                params={"function": "NEWS_SENTIMENT", "tickers": symbol,
                        "limit": self.max_headlines, "apikey": self.av_key},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("feed", [])
        except Exception as exc:
            logger.debug(f"NewsSentimentSignal AV error [{symbol}]: {exc}")
            return []

    def _recency_weight(self, time_str: str) -> float:
        try:
            dt = datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            age_hrs = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            return self.bucket_weights[-1]
        for hrs, w in zip(self.bucket_hrs, self.bucket_weights):
            if age_hrs <= hrs:
                return w
        return 0.0

    def _finbert_compute(self, symbol: str) -> dict | None:
        """Alpha Vantage headlines + FinBERT scoring. Returns None if unavailable."""
        if not self._load_pipeline():
            return None
        items = self._fetch_av_headlines(symbol)
        if not items:
            return None

        weighted_scores: list[tuple[float, str]] = []
        for item in items:
            title = item.get("title", "")
            if not title:
                continue
            recency_w = self._recency_weight(item.get("time_published", ""))
            if recency_w == 0.0:
                continue
            analyst_w = self.analyst_multiplier if any(
                kw in title.lower() for kw in _ANALYST_KEYWORDS
            ) else 1.0
            res = self._pipeline(title[:512])[0]
            label_map = {x["label"].lower(): x["score"] for x in res}
            raw = label_map.get("positive", 0.0) - label_map.get("negative", 0.0)
            weighted_scores.append((raw * recency_w * analyst_w, title))

        if not weighted_scores:
            return None
        weighted_scores.sort(key=lambda x: abs(x[0]), reverse=True)
        avg_raw = sum(s for s, _ in weighted_scores) / len(weighted_scores)
        normalised = (avg_raw + 1) / 2
        return {
            "sentiment_score": round(float(normalised), 4),
            "raw_score":       round(float(avg_raw), 4),
            "headline_count":  len(weighted_scores),
            "top_headlines":   [t for _, t in weighted_scores[:3]],
            "symbol":          symbol,
            "source":          "alpha_vantage+finbert",
        }

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(self, symbol: str) -> dict:
        """
        Returns:
            {
              "sentiment_score":  float [0, 1],
              "raw_score":        float [-1, 1],
              "headline_count":   int,
              "top_headlines":    list[str],
              "symbol":           str,
              "source":           str,   # "finnhub" | "alpha_vantage+finbert" | "neutral"
            }
        """
        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if time.time() - ts < self.cache_ttl:
                logger.debug(f"NewsSentimentSignal: cache hit for {symbol}")
                return cached

        # Primary: Alpha Vantage pre-scored sentiment (no local model needed)
        result = self._av_compute(symbol)

        # Fallback: Alpha Vantage headlines + FinBERT scoring (requires PyTorch)
        if result is None:
            result = self._finbert_compute(symbol)

        # Final fallback: neutral
        if result is None:
            result = self._neutral(symbol)

        self._cache[symbol] = (time.time(), result)
        logger.info(
            f"NewsSentimentSignal [{symbol}]: score={result['sentiment_score']:.3f} "
            f"source={result.get('source','?')} headlines={result['headline_count']}"
        )
        return result

    def score_only(self, symbol: str) -> float:
        return self.compute(symbol)["sentiment_score"]

    @staticmethod
    def _neutral(symbol: str) -> dict:
        return {
            "sentiment_score": 0.5,
            "raw_score": 0.0,
            "headline_count": 0,
            "top_headlines": [],
            "symbol": symbol,
            "source": "neutral",
        }
