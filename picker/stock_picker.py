"""
StockPicker — parallel multi-signal composite ranker.

For each ticker in the watchlist, runs four signals concurrently via
ThreadPoolExecutor:
  - VolumeSignal      → volume_score      (25% default weight)
  - SpikePredictor    → spike_probability (30%)
  - NewsSentimentSignal → sentiment_score (20%)
  - FlowSignal        → flow_score        (25%)
    • congressional_score (40%) — Senate/House Stock Watcher
    • institutional_score (35%) — SEC EDGAR 13F
    • insider_score       (25%) — SEC EDGAR Form 4 open-market buys

Returns top-N tickers above a composite threshold.
Full score breakdown is logged to logs/picks.log.

Backward compatibility:
  score(price_data: dict) still works and runs the legacy momentum/
  mean-reversion signals on pre-fetched DataFrames (used by BacktestEngine).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

from signals import (
    MomentumSignal,
    MeanReversionSignal,
    MLScoreSignal,
    VolumeSignal,
    SpikePredictor,
    NewsSentimentSignal,
    FlowSignal,
)


class StockPicker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.picker_cfg = cfg["picker"]
        self.sig_cfg = cfg["signals"]
        # Long-only; short selling removed

        # ── New signal instances ──────────────────────────────────────────────
        self._volume = VolumeSignal(self.sig_cfg.get("volume", {}))
        self._spike = SpikePredictor(cfg)

        sn_cfg = self.sig_cfg.get("sentiment_news", {})
        self._sentiment = (
            NewsSentimentSignal(self.sig_cfg)
            if sn_cfg.get("enabled", False)
            else None
        )

        flow_cfg = self.sig_cfg.get("flow", {})
        self._flow = (
            FlowSignal(self.sig_cfg)
            if flow_cfg.get("enabled", True)
            else None
        )

        # ── Legacy signals (used by score() + BacktestEngine) ─────────────────
        self._legacy_signals: list[tuple[object, float]] = []
        self._add_legacy(MomentumSignal, "momentum")
        self._add_legacy(MeanReversionSignal, "mean_reversion")
        self._add_legacy(MLScoreSignal, "ml_score")

        # ── Composite weights ─────────────────────────────────────────────────
        w = self.picker_cfg.get("composite_weights", {})
        self._w_volume: float    = float(w.get("volume_score", 0.25))
        self._w_spike: float     = float(w.get("spike_probability", 0.30))
        self._w_sentiment: float = float(w.get("sentiment_score", 0.20))
        self._w_flow: float      = float(w.get("flow_score", 0.25))
        self._w_technical: float = float(w.get("technical_score", 0.20))

        # Logging
        log_file = self.picker_cfg.get("log_file", "logs/picks.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="INFO", rotation="5 MB", filter=lambda r: "PICK" in r["message"])

    # ── Legacy helper ────────────────────────────────────────────────────────

    def _add_legacy(self, cls, key: str):
        scfg = self.sig_cfg.get(key, {})
        if scfg.get("enabled", True):
            self._legacy_signals.append((cls(scfg), float(scfg.get("weight", 1.0))))

    # ── Per-ticker signal workers (run in thread pool) ────────────────────────

    def _run_volume(self, symbol: str, df: pd.DataFrame | None) -> float:
        try:
            src = df if df is not None else symbol
            return self._volume.score_only(src)
        except Exception as exc:
            logger.warning(f"VolumeSignal failed for {symbol}: {exc}")
            return 0.0

    def _run_spike(self, symbol: str, df: pd.DataFrame | None) -> float:
        try:
            src = df if df is not None else symbol
            if getattr(self._spike, "_model_v3", None) is not None:
                return self._spike.predict_v3(src)
            return self._spike.predict(src)
        except Exception as exc:
            logger.warning(f"SpikePredictor failed for {symbol}: {exc}")
            return 0.0

    def _run_sentiment(self, symbol: str) -> float:
        if self._sentiment is None:
            return 0.5  # neutral default
        try:
            return self._sentiment.score_only(symbol)
        except Exception as exc:
            logger.warning(f"NewsSentimentSignal failed for {symbol}: {exc}")
            return 0.5

    def _run_flow(self, symbol: str, flow_scores: dict[str, float]) -> float:
        return flow_scores.get(symbol.upper(), 0.0)

    def _legacy_score_components(self, df: pd.DataFrame) -> dict[str, float]:
        if df is None or df.empty:
            return {"technical_score": 0.5}

        total_w = 0.0
        weighted = 0.0
        components: dict[str, float] = {}
        for signal, w in self._legacy_signals:
            try:
                raw = float(signal.compute(df))
                normalized = round((raw + 1.0) / 2.0, 4)
                components[signal.name] = normalized
                weighted += w * raw
                total_w += w
            except Exception as exc:
                logger.warning(f"{signal.name} failed in unified scoring: {exc}")

        composite = (weighted / total_w) if total_w > 0 else 0.0
        components["technical_score"] = round((composite + 1.0) / 2.0, 4)
        return components

    # ── Direction bias helpers ────────────────────────────────────────────────

    @staticmethod
    def _direction_bias(sentiment_score: float) -> str:
        # Neutral is treated as BULLISH: sentiment API outages default to 0.5
        # and should not zero out the entire watchlist.
        if sentiment_score < 0.45:
            return "BEARISH"
        return "BULLISH"

    @staticmethod
    def _catalyst_score(spike: float, flow: float, sentiment: float) -> float:
        """
        Catalyst score — how strong is the upcoming move signal.
        Combines spike probability, flow signal, and sentiment strength.
        """
        sentiment_strength = abs(2.0 * (sentiment - 0.5))   # [0, 1]
        return round(0.50 * spike + 0.30 * flow + 0.20 * sentiment_strength, 4)

    # ── Live scoring (parallel) ──────────────────────────────────────────────

    def score_live(
        self,
        symbols: list[str],
        price_data: dict[str, pd.DataFrame] | None = None,
        use_external_data: bool = True,
    ) -> pd.DataFrame:
        """
        Score all symbols using all four new signals in parallel.
        price_data: optional pre-fetched {symbol: DataFrame}; if absent,
                    VolumeSignal and SpikePredictor fetch their own data.

        Returns DataFrame sorted by composite_score descending with columns:
          [symbol, composite_score, volume_score, spike_probability,
           sentiment_score, flow_score, catalyst_score, direction_bias,
           trade_type]
        """
        price_data = price_data or {}
        if not symbols:
            return pd.DataFrame(columns=[
                "symbol", "composite_score", "volume_score", "spike_probability",
                "sentiment_score", "flow_score", "catalyst_score",
                "direction_bias", "trade_type",
            ])

        # Flow scores are fetched in a single batch call (one HTTP round trip)
        flow_scores: dict[str, float] = {}
        if use_external_data and self._flow:
            try:
                flow_scores = self._flow.score_only(symbols)
            except Exception as exc:
                logger.warning(f"FlowSignal batch call failed: {exc}")

        rows = []
        max_workers = min(len(symbols), 8)

        # Sentiment calls are serialised to respect Alpha Vantage's 5 req/min
        # rate limit; cache hits (TTL=15 min) are free so the delay is skipped.
        sentiment_scores: dict[str, float] = {}
        _AV_INTER_CALL_DELAY = 13.0   # seconds → ~4.6 calls/min
        last_sentiment_source: str | None = None
        if use_external_data and self._sentiment:
            for i, sym in enumerate(symbols):
                if sym in getattr(self._sentiment, "_cache", {}):
                    ts, _ = self._sentiment._cache[sym]
                    if time.time() - ts < self._sentiment.cache_ttl:
                        sentiment_scores[sym] = self._run_sentiment(sym)
                        _, cached = self._sentiment._cache.get(sym, (0.0, {}))
                        last_sentiment_source = cached.get("source")
                        continue
                if i > 0 and last_sentiment_source in {"alpha_vantage", "alpha_vantage+finbert"}:
                    time.sleep(_AV_INTER_CALL_DELAY)
                sentiment_scores[sym] = self._run_sentiment(sym)
                _, cached = self._sentiment._cache.get(sym, (0.0, {}))
                last_sentiment_source = cached.get("source")
        else:
            sentiment_scores = {sym: 0.5 for sym in symbols}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for sym in symbols:
                df = price_data.get(sym)
                futures[pool.submit(self._run_volume, sym, df)] = (sym, "volume")
                futures[pool.submit(self._run_spike, sym, df)] = (sym, "spike")

            results: dict[str, dict[str, float]] = {s: {} for s in symbols}
            for future in as_completed(futures):
                sym, sig_type = futures[future]
                try:
                    results[sym][sig_type] = future.result()
                except Exception as exc:
                    logger.warning(f"Worker exception [{sym}/{sig_type}]: {exc}")
                    results[sym][sig_type] = 0.0

        min_catalyst = self.picker_cfg.get("min_catalyst_score", 0.60)
        require_bias  = self.picker_cfg.get("require_direction_bias", True)

        for sym in symbols:
            r = results[sym]
            vol   = r.get("volume", 0.0)
            spike = r.get("spike", 0.0)
            sent  = sentiment_scores.get(sym, 0.5)
            flow  = self._run_flow(sym, flow_scores)
            legacy = self._legacy_score_components(price_data.get(sym))
            technical = legacy.get("technical_score", 0.5)

            bias     = self._direction_bias(sent)
            catalyst = self._catalyst_score(spike, flow, sent)

            # Long-only: BEARISH sentiment skips the stock, everything else is LONG
            if require_bias and bias == "BEARISH":
                trade_type = "SKIP"
            else:
                trade_type = "LONG"

            weight_total = (
                self._w_volume + self._w_spike + self._w_sentiment
                + self._w_flow + self._w_technical
            ) or 1.0
            composite = (
                self._w_volume * vol
                + self._w_spike * spike
                + self._w_sentiment * sent
                + self._w_flow * flow
                + self._w_technical * technical
            ) / weight_total
            row = {
                "symbol":            sym,
                "composite_score":   round(composite, 4),
                "volume_score":      round(vol, 4),
                "spike_probability": round(spike, 4),
                "sentiment_score":   round(sent, 4),
                "flow_score":        round(flow, 4),
                "technical_score":   round(technical, 4),
                "catalyst_score":    catalyst,
                "direction_bias":    bias,
                "trade_type":        trade_type,
            }
            for name, value in legacy.items():
                if name != "technical_score":
                    row[name] = value
            rows.append(row)

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            return pd.DataFrame(columns=[
                "symbol", "composite_score", "volume_score", "spike_probability",
                "sentiment_score", "flow_score", "catalyst_score",
                "direction_bias", "trade_type",
            ])
        df_out = df_out.sort_values("composite_score", ascending=False)
        return df_out

    # ── Legacy scoring (backward compat with BacktestEngine) ─────────────────

    def score(
        self,
        price_data: dict[str, pd.DataFrame],
        headlines: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Legacy interface: compute momentum/mean-reversion/ML scores from
        pre-fetched price DataFrames. Returns DataFrame[symbol, score].
        """
        headlines = headlines or {}
        rows = []

        for symbol, df in price_data.items():
            if df is None or df.empty:
                continue
            total_w = 0.0
            weighted = 0.0
            for signal, w in self._legacy_signals:
                try:
                    s = signal.compute(df)
                    weighted += w * s
                    total_w += w
                except Exception as exc:
                    logger.warning(f"{signal.name} failed for {symbol}: {exc}")

            composite = (weighted / total_w) if total_w > 0 else 0.0
            normalised = (composite + 1) / 2
            rows.append({"symbol": symbol, "score": round(normalised, 4)})

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            return pd.DataFrame(columns=["symbol", "score"])
        return df_out.sort_values("score", ascending=False)

    # ── Pick ─────────────────────────────────────────────────────────────────

    def pick(
        self,
        symbols_or_data: list[str] | dict[str, pd.DataFrame],
        sector_map: dict[str, str] | None = None,
        headlines: dict[str, list[str]] | None = None,
        dry_run: bool | None = None,
        price_data: dict[str, pd.DataFrame] | None = None,
        use_external_data: bool = True,
    ) -> list[str]:
        """
        Main entry point. Accepts either:
          - list[str]                  → runs score_live (new signals)
          - dict[str, pd.DataFrame]    → runs legacy score (backtest compat)

        Returns top-N symbols above threshold.
        """
        # Live path uses min_composite_score; legacy backtest uses score_threshold
        is_live = isinstance(symbols_or_data, list)
        threshold  = (
            self.picker_cfg.get("min_composite_score", 0.45)
            if is_live else self.picker_cfg["score_threshold"]
        )
        sector_cap = self.picker_cfg["sector_cap"]
        top_n      = self.picker_cfg["top_n"]
        dry_run    = dry_run if dry_run is not None else self.picker_cfg.get("dry_run", False)
        sector_map = sector_map or {}

        # Dispatch to correct scoring path
        if isinstance(symbols_or_data, dict):
            df_scores = self.score(symbols_or_data, headlines)
            df_scores = df_scores.rename(columns={"score": "composite_score"})
        else:
            df_scores = self.score_live(
                symbols_or_data,
                price_data=price_data,
                use_external_data=use_external_data,
            )

        self._last_score_df = df_scores  # cache so callers avoid a second score_live() call

        if df_scores.empty:
            logger.info("[PICK] No scored symbols available.")
            return []

        # Exclude SKIP-labelled rows from candidates (direction gate)
        if "trade_type" in df_scores.columns:
            eligible = df_scores[
                (df_scores["composite_score"] >= threshold)
                & (df_scores["trade_type"] != "SKIP")
            ]
        else:
            eligible = df_scores[df_scores["composite_score"] >= threshold]

        selected: list[str] = []
        sector_counts: dict[str, int] = {}

        for row in eligible.itertuples():
            if len(selected) >= top_n:
                break
            sector = sector_map.get(row.symbol, "Unknown")
            if sector_counts.get(sector, 0) >= sector_cap:
                continue
            selected.append(row.symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        logger.info(f"[PICK] Selected {len(selected)}/{len(df_scores)} symbols (threshold={threshold})")
        for row in df_scores.itertuples():
            flag  = "✓" if row.symbol in selected else " "
            ttype = getattr(row, "trade_type", "LONG")
            bias  = getattr(row, "direction_bias", "")
            cat   = getattr(row, "catalyst_score", "")
            logger.info(
                f"[PICK] {flag} {row.symbol:<6} composite={row.composite_score:.4f} "
                f"type={ttype:<5} bias={bias:<8} catalyst={cat}"
            )

        if dry_run:
            print("\n=== DRY RUN: StockPicker Results ===")
            cols = [c for c in [
                "symbol", "composite_score", "volume_score", "spike_probability",
                "sentiment_score", "flow_score", "technical_score", "catalyst_score",
                "direction_bias", "trade_type",
            ] if c in df_scores.columns]
            print(df_scores[cols].to_string(index=False))
            print(f"\nPicks (top {top_n} above {threshold}): {selected}\n")

        return selected
