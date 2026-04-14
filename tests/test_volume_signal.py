"""
Unit tests for VolumeSignal.

Tests cover:
  - volume_zscore computation (normal + edge cases)
  - vwap_deviation computation
  - obv_slope computation
  - compute() with a DataFrame (no network)
  - compute() with a symbol string (mocked yfinance)
  - Insufficient data handling
  - Flagging threshold
  - score_only() convenience wrapper
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path when running tests directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from signals.volume_signal import VolumeSignal


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(
    n: int = 50,
    base_price: float = 100.0,
    base_volume: float = 1_000_000.0,
    volume_spike: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = base_price + np.cumsum(rng.normal(0, 1, n))
    closes = np.maximum(closes, 1.0)
    highs  = closes * (1 + rng.uniform(0, 0.01, n))
    lows   = closes * (1 - rng.uniform(0, 0.01, n))
    opens  = closes * (1 + rng.normal(0, 0.005, n))
    volumes = np.full(n, base_volume)
    if volume_spike:
        volumes[-1] = base_volume * 10  # dramatic spike on last candle

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


@pytest.fixture
def cfg():
    return {
        "volume": {
            "window": 20,
            "obv_slope_window": 10,
            "threshold": 0.7,
            "default_interval": "1d",
        }
    }


@pytest.fixture
def signal(cfg):
    return VolumeSignal(cfg)


@pytest.fixture
def normal_df():
    return _make_df(n=50)


@pytest.fixture
def spiked_df():
    return _make_df(n=50, volume_spike=True)


@pytest.fixture
def short_df():
    """DataFrame shorter than the rolling window."""
    return _make_df(n=10)


# ── volume_zscore ─────────────────────────────────────────────────────────────

class TestVolumeZscore:
    def test_returns_float(self, signal, normal_df):
        z = signal.volume_zscore(normal_df)
        assert isinstance(z, float)

    def test_spike_gives_high_zscore(self, signal, spiked_df):
        z = signal.volume_zscore(spiked_df)
        # Volume on last candle is 10x normal → z should be well above 3
        assert z > 3.0, f"Expected z > 3 for spike, got {z}"

    def test_flat_volume_gives_near_zero_zscore(self, signal, cfg):
        # Perfectly flat volume → std = 0 → should return 0.0
        df = _make_df(n=50)
        df["Volume"] = 1_000_000.0
        z = signal.volume_zscore(df)
        assert z == 0.0

    def test_insufficient_data_returns_zero(self, signal, short_df):
        z = signal.volume_zscore(short_df)
        assert z == 0.0

    def test_normal_volume_zscore_in_range(self, signal, normal_df):
        # For normally distributed volume, |z| should be small
        z = signal.volume_zscore(normal_df)
        assert abs(z) < 5.0


# ── vwap_deviation ─────────────────────────────────────────────────────────────

class TestVwapDeviation:
    def test_returns_float(self, signal, normal_df):
        vd = signal.vwap_deviation(normal_df)
        assert isinstance(vd, float)

    def test_nonnegative(self, signal, normal_df):
        vd = signal.vwap_deviation(normal_df)
        assert vd >= 0.0

    def test_zero_volume_returns_zero(self, signal):
        df = _make_df(n=30)
        df["Volume"] = 0.0
        vd = signal.vwap_deviation(df)
        assert vd == 0.0

    def test_price_at_vwap_gives_zero_deviation(self, signal):
        """When all candles are identical, price == VWAP."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 30,
            "High": [100.0] * 30,
            "Low": [100.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [1_000_000.0] * 30,
        }, index=dates)
        vd = signal.vwap_deviation(df)
        assert vd == pytest.approx(0.0, abs=1e-8)

    def test_empty_df_returns_zero(self, signal):
        vd = signal.vwap_deviation(pd.DataFrame())
        assert vd == 0.0


# ── obv_slope ─────────────────────────────────────────────────────────────────

class TestObvSlope:
    def test_returns_float(self, signal, normal_df):
        obs = signal.obv_slope(normal_df)
        assert isinstance(obs, float)

    def test_rising_prices_positive_slope(self, signal):
        """Consistently rising prices → OBV accumulates → positive slope."""
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = np.linspace(100, 130, n)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.005,
            "Low":  prices * 0.995,
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
        }, index=dates)
        obs = signal.obv_slope(df)
        assert obs > 0, f"Expected positive OBV slope for rising prices, got {obs}"

    def test_falling_prices_negative_slope(self, signal):
        """Consistently falling prices → OBV declines → negative slope."""
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = np.linspace(130, 100, n)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.005,
            "Low":  prices * 0.995,
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
        }, index=dates)
        obs = signal.obv_slope(df)
        assert obs < 0, f"Expected negative OBV slope for falling prices, got {obs}"

    def test_insufficient_data_returns_zero(self, signal, short_df):
        obs = signal.obv_slope(short_df)
        assert obs == 0.0


# ── compute() with DataFrame ──────────────────────────────────────────────────

class TestComputeWithDataFrame:
    def test_returns_dict_keys(self, signal, normal_df):
        result = signal.compute(normal_df)
        expected_keys = {"volume_score", "volume_zscore", "vwap_deviation", "obv_slope", "flagged", "symbol"}
        assert expected_keys == set(result.keys())

    def test_score_in_range(self, signal, normal_df):
        result = signal.compute(normal_df)
        assert 0.0 <= result["volume_score"] <= 1.0

    def test_symbol_is_none_for_df_input(self, signal, normal_df):
        result = signal.compute(normal_df)
        assert result["symbol"] is None

    def test_insufficient_data_returns_zeros(self, signal, short_df):
        result = signal.compute(short_df)
        assert result["volume_score"] == 0.0
        assert result["flagged"] == False

    def test_spike_flags(self, signal, spiked_df):
        """A big volume spike should produce flagged=True."""
        result = signal.compute(spiked_df)
        assert result["flagged"] == True, f"Expected flagged=True, got score={result['volume_score']}"

    def test_normal_data_not_flagged(self, signal, normal_df):
        """Flat, unremarkable volume should not flag."""
        # Use perfectly flat volume to guarantee z=0
        df = _make_df(n=50)
        df["Volume"] = 1_000_000.0
        result = signal.compute(df)
        assert result["flagged"] == False


# ── compute() with symbol string (mocked yfinance) ────────────────────────────

class TestComputeWithSymbol:
    def test_fetches_and_scores(self, signal, normal_df):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = normal_df

        with patch("signals.volume_signal.yf.Ticker", return_value=mock_ticker):
            result = signal.compute("AAPL", interval="1d")

        assert result["symbol"] == "AAPL"
        assert 0.0 <= result["volume_score"] <= 1.0
        mock_ticker.history.assert_called_once()

    def test_empty_fetch_returns_zeros(self, signal):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("signals.volume_signal.yf.Ticker", return_value=mock_ticker):
            result = signal.compute("FAKE")

        assert result["volume_score"] == 0.0
        assert result["flagged"] == False


# ── score_only() ──────────────────────────────────────────────────────────────

class TestScoreOnly:
    def test_returns_float(self, signal, normal_df):
        score = signal.score_only(normal_df)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_matches_compute(self, signal, normal_df):
        score = signal.score_only(normal_df)
        full = signal.compute(normal_df)
        assert score == full["volume_score"]


# ── Configurable threshold ────────────────────────────────────────────────────

class TestThreshold:
    def test_custom_threshold_respected(self, spiked_df):
        cfg_low  = {"volume": {"window": 20, "obv_slope_window": 10, "threshold": 0.01}}
        cfg_high = {"volume": {"window": 20, "obv_slope_window": 10, "threshold": 0.99}}

        signal_low  = VolumeSignal(cfg_low)
        signal_high = VolumeSignal(cfg_high)

        result_low  = signal_low.compute(spiked_df)
        result_high = signal_high.compute(spiked_df)

        assert result_low["flagged"] == True    # threshold=0.01 → nearly always flagged
        assert result_high["flagged"] == False  # threshold=0.99 → almost never flagged

    def test_default_threshold_is_0_7(self):
        signal = VolumeSignal({})
        assert signal.threshold == 0.7


# ── Normalisation weights ─────────────────────────────────────────────────────

class TestWeights:
    def test_weights_sum_to_one(self):
        total = VolumeSignal.WEIGHT_ZSCORE + VolumeSignal.WEIGHT_VWAP + VolumeSignal.WEIGHT_OBV
        assert total == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
