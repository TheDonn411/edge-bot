"""
VolumeSignal — composite volume pressure indicator.

Sub-components (all normalised to [0, 1] before weighting):
  1. volume_zscore   (50%) — rolling z-score of volume; high z = unusual activity
  2. vwap_deviation  (30%) — |current_price - VWAP| / VWAP; proximity to VWAP
  3. obv_slope       (20%) — linear slope of OBV over last N candles; rising = bullish

volume_score = 0.5 * z_norm + 0.3 * vwap_norm + 0.2 * obv_norm  ∈ [0, 1]

A signal flag is raised when volume_score >= threshold (default 0.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from scipy import stats as scipy_stats


class VolumeSignal:
    # Weights for the composite score
    WEIGHT_ZSCORE = 0.50
    WEIGHT_VWAP = 0.30
    WEIGHT_OBV = 0.20

    def __init__(self, cfg: dict):
        vol_cfg = cfg.get("volume", cfg)  # accept full cfg or sub-cfg
        self.window: int = int(vol_cfg.get("window", 20))
        self.obv_window: int = int(vol_cfg.get("obv_slope_window", 10))
        self.threshold: float = float(vol_cfg.get("threshold", 0.7))
        self.default_interval: str = vol_cfg.get("default_interval", "1d")

    # ── Data fetching ────────────────────────────────────────────────────────

    def fetch(self, symbol: str, interval: str = "1d", period: str = "1y") -> pd.DataFrame:
        """
        Fetch OHLCV from yfinance.
        interval: "1m" | "5m" | "1d" (etc.)
        period:   yfinance period string e.g. "1y", "60d", "7d"
        """
        # Adjust period for intraday limits
        if interval in ("1m",):
            period = "7d"
        elif interval in ("5m", "15m", "30m"):
            period = "60d"

        ticker = yf.Ticker(symbol)
        try:
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:
            logger.warning(f"VolumeSignal: fetch failed for {symbol} [{interval}] — {exc}")
            return pd.DataFrame()
        if df.empty:
            logger.warning(f"VolumeSignal: no data for {symbol} [{interval}]")
        return df

    # ── Sub-components ───────────────────────────────────────────────────────

    def volume_zscore(self, df: pd.DataFrame) -> float:
        """
        Rolling z-score of the most recent volume bar.
        Returns raw z-score (not clipped); caller normalises.
        """
        if len(df) < self.window + 1:
            return 0.0
        vol = df["Volume"].astype(float)
        roll_mean = vol.rolling(self.window).mean().iloc[-1]
        roll_std = vol.rolling(self.window).std().iloc[-1]
        if roll_std == 0:
            return 0.0
        return float((vol.iloc[-1] - roll_mean) / roll_std)

    def vwap_deviation(self, df: pd.DataFrame) -> float:
        """
        Typical price VWAP over the DataFrame window.
        Returns |price - VWAP| / VWAP as a positive fraction.
        A large deviation means price has moved significantly from fair value.
        """
        if df.empty:
            return 0.0
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        volume_sum = df["Volume"].sum()
        if volume_sum == 0:
            return 0.0
        vwap = (typical * df["Volume"]).sum() / volume_sum
        if vwap == 0 or np.isnan(vwap):
            return 0.0
        current = float(df["Close"].iloc[-1])
        return abs(current - vwap) / vwap

    def obv_slope(self, df: pd.DataFrame) -> float:
        """
        OBV over last obv_window candles; return the linear regression slope
        normalised by mean OBV so the result is scale-free.
        Positive = accumulation, negative = distribution.
        """
        if len(df) < self.obv_window + 1:
            return 0.0

        # Build full OBV series
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()

        window_obv = obv.iloc[-self.obv_window:]
        x = np.arange(len(window_obv))
        slope, _, _, _, _ = scipy_stats.linregress(x, window_obv.values)

        mean_obv = abs(window_obv.mean())
        return float(slope / mean_obv) if mean_obv != 0 else 0.0

    # ── Normalisation helpers ────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Map any real value to (0, 1). Used to normalise unbounded metrics."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _clip_norm(x: float, lo: float = 0.0, hi: float = 0.10) -> float:
        """Clip then normalise to [0, 1]."""
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(
        self,
        symbol_or_df: str | pd.DataFrame,
        interval: str | None = None,
    ) -> dict:
        """
        Compute composite volume_score for a symbol or a pre-fetched DataFrame.

        Returns:
            {
              "volume_score":   float [0, 1],
              "volume_zscore":  float (raw),
              "vwap_deviation": float (raw),
              "obv_slope":      float (raw, scale-free),
              "flagged":        bool,
              "symbol":         str | None,
            }
        """
        interval = interval or self.default_interval

        if isinstance(symbol_or_df, str):
            symbol = symbol_or_df
            df = self.fetch(symbol, interval=interval)
        else:
            symbol = None
            df = symbol_or_df

        if df is None or df.empty or len(df) < self.window + 1:
            logger.debug(f"VolumeSignal: insufficient data (len={len(df) if df is not None else 0})")
            return {
                "volume_score": 0.0,
                "volume_zscore": 0.0,
                "vwap_deviation": 0.0,
                "obv_slope": 0.0,
                "flagged": False,
                "symbol": symbol,
            }

        z = self.volume_zscore(df)
        vd = self.vwap_deviation(df)
        obs = self.obv_slope(df)

        # Normalise each component to [0, 1]
        z_norm = self._sigmoid(z)                        # z~N(0,1): sigmoid maps well
        vd_norm = self._clip_norm(vd, 0.0, 0.05)        # 0–5% deviation = full range
        obs_norm = self._sigmoid(obs * 10)               # scale OBV slope for sensitivity

        score = (
            self.WEIGHT_ZSCORE * z_norm
            + self.WEIGHT_VWAP * vd_norm
            + self.WEIGHT_OBV * obs_norm
        )

        result = {
            "volume_score": round(float(score), 4),
            "volume_zscore": round(z, 4),
            "vwap_deviation": round(vd, 6),
            "obv_slope": round(obs, 6),
            "flagged": bool(score >= self.threshold),
            "symbol": symbol,
        }
        logger.debug(f"VolumeSignal [{symbol or 'df'}]: {result}")
        return result

    def score_only(self, symbol_or_df: str | pd.DataFrame, interval: str | None = None) -> float:
        """Convenience wrapper — returns volume_score only."""
        return self.compute(symbol_or_df, interval)["volume_score"]
