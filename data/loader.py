"""
DataLoader — fetches and caches OHLCV price data.

Supported provider: yfinance (default).
Cached as Parquet files under data/processed/<symbol>.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger


class DataLoader:
    def __init__(self, cfg: dict):
        self.cfg = cfg["data"]
        self.cache_dir = Path(self.cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol*. Reads from cache when available."""
        cache_path = self.cache_dir / f"{symbol}.parquet"

        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {symbol}")
            return pd.read_parquet(cache_path)

        lookback = self.cfg.get("lookback_days", 365)
        start = start or (pd.Timestamp.today() - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

        logger.info(f"Fetching {symbol} [{start} → {end}]")
        ticker = yf.Ticker(symbol)
        try:
            df = ticker.history(start=start, end=end, auto_adjust=True)
        except Exception as exc:
            if cache_path.exists():
                logger.warning(
                    f"Fetch failed for {symbol}; falling back to cached data: {exc}"
                )
                return pd.read_parquet(cache_path)
            logger.warning(f"Fetch failed for {symbol}: {exc}")
            return pd.DataFrame()

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        df.to_parquet(cache_path)
        return df

    def fetch_many(self, symbols: list[str], **kwargs) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols; returns {symbol: DataFrame}."""
        return {sym: self.fetch(sym, **kwargs) for sym in symbols}
