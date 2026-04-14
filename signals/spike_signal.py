"""
SpikePredictor — XGBoost binary classifier for short-term price spikes.

v1 (original):
  Label: 1 if close[t+5] / close[t] - 1 >= 2%  (directional up)
  Features: RSI(14), MACD histogram, ATR(14), BB width percentile,
            5-candle momentum, volume z-score
  Training universe: AAPL, MSFT, NVDA (large-cap)
  Saved as: models/spike_model.pkl

v3 (small/mid-cap earnings catalyst):
  Label: 1 if abs(close[t+3] / close[t] - 1) >= 8%  (big move either direction)
  Features: all v1 features + days_to_earnings, eps_revision_pct,
            short_ratio, low52_proximity, pre_earnings_compression
  Training universe: large (20%) + mid (40%) + small FMP screened (40%)
  Saved as: models/spike_model_v3_smallcap.pkl

Usage:
    predictor = SpikePredictor(cfg)
    predictor.retrain(["AAPL", "MSFT"], lookback_days=730)     # v1
    predictor.retrain_v3(fmp_key="...", lookback_days=1095)    # v3
    prob = predictor.predict(df)          # float in [0, 1]
    prob = predictor.predict("AAPL")      # fetches fresh data then predicts
"""

from __future__ import annotations

import pickle
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from loguru import logger

from .volume_signal import VolumeSignal

_FMP_BASE = "https://financialmodelingprep.com/api/v3"


# ── Feature engineering ──────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_histogram(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bb_width_percentile(series: pd.Series, period: int = 20, pct_window: int = 100) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    width = (2 * std) / sma.replace(0, np.nan)
    return width.rolling(pct_window).rank(pct=True)


def _momentum_5(series: pd.Series) -> pd.Series:
    return series.pct_change(5)


# ── FMP helpers (v3) ─────────────────────────────────────────────────────────

def _fmp_get(endpoint: str, params: dict, fmp_key: str, timeout: int = 15) -> list | dict | None:
    try:
        r = requests.get(
            f"{_FMP_BASE}/{endpoint}",
            params={**params, "apikey": fmp_key},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug(f"SpikePredictor FMP [{endpoint}]: {exc}")
        return None


def _fmp_earnings_calendar(fmp_key: str, lookback_days: int = 1095) -> list[dict]:
    """
    Pull historical earnings events for the past `lookback_days` days.
    Returns raw list of {symbol, date, revenueEstimated, revenue, ...}.
    """
    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=lookback_days)).isoformat()
    data = _fmp_get(
        "earning_calendar",
        {"from": from_date, "to": today.isoformat()},
        fmp_key,
    )
    return data if isinstance(data, list) else []


def _fmp_small_cap_tickers(fmp_key: str, n: int = 20, lookback_days: int = 180) -> list[str]:
    """
    Pull up to `n` unique small-cap tickers (marketCap < $2B) that have
    had an earnings event in the past `lookback_days` days.
    """
    events = _fmp_earnings_calendar(fmp_key, lookback_days)
    tickers: list[str] = []
    seen: set[str] = set()
    for ev in events:
        sym = (ev.get("symbol") or "").strip().upper()
        if not sym or "." in sym or sym in seen:
            continue
        mc = ev.get("marketCap")
        try:
            mc = float(mc) if mc is not None else None
        except (TypeError, ValueError):
            mc = None
        if mc is not None and mc > 2_000_000_000:
            continue
        seen.add(sym)
        tickers.append(sym)
        if len(tickers) >= n * 3:   # over-fetch; we'll sample below
            break

    random.shuffle(tickers)
    return tickers[:n]


def _fmp_eps_revision(symbol: str, fmp_key: str) -> float:
    """
    Return the most recent EPS revision percentage for `symbol`.
    Positive = upward revision, negative = downward revision.
    Returns 0.0 on failure.
    """
    data = _fmp_get(f"analyst-estimates/{symbol}", {}, fmp_key)
    if not isinstance(data, list) or len(data) < 2:
        return 0.0
    try:
        latest   = float(data[0].get("estimatedEpsAvg") or 0)
        previous = float(data[1].get("estimatedEpsAvg") or 0)
        if previous == 0:
            return 0.0
        return round((latest - previous) / abs(previous), 4)
    except (TypeError, ValueError):
        return 0.0


def _fmp_next_earnings_date(symbol: str, fmp_key: str, horizon_days: int = 60) -> str | None:
    """Return the nearest upcoming earnings date (YYYY-MM-DD) or None."""
    today = datetime.now(timezone.utc).date()
    end   = today + timedelta(days=horizon_days)
    data  = _fmp_get(
        "earning_calendar",
        {"from": today.isoformat(), "to": end.isoformat(), "symbol": symbol},
        fmp_key,
    )
    if not isinstance(data, list):
        return None
    dates = [ev["date"][:10] for ev in data if ev.get("date") and ev.get("symbol","").upper() == symbol]
    return min(dates) if dates else None


# ── V3 feature builder ────────────────────────────────────────────────────────

def build_features_v3(
    df: pd.DataFrame,
    vol_signal: VolumeSignal | None = None,
    earnings_dates: list[str] | None = None,
    eps_revision_pct: float = 0.0,
    short_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Extends build_features() with earnings-catalyst features:
      - days_to_earnings       : candles until next earnings (normalised 0-1 over 30d window)
      - eps_revision_pct       : latest EPS estimate revision (constant per ticker)
      - short_ratio            : yfinance shortRatio (constant per ticker)
      - low52_proximity        : (price - 52wLow) / (52wHigh - 52wLow)
      - pre_earnings_compression: ATR_5 / ATR_30  (low = compressed, about to expand)
    """
    f = build_features(df, vol_signal)

    close = df["Close"].astype(float)
    index = df.index

    # ── days_to_earnings ─────────────────────────────────────────────────────
    # Convert earnings_dates strings to date objects
    earnings_dt_set: set = set()
    if earnings_dates:
        for ds in earnings_dates:
            try:
                earnings_dt_set.add(datetime.strptime(ds[:10], "%Y-%m-%d").date())
            except ValueError:
                pass

    dte_values: list[float] = []
    for idx_val in index:
        if hasattr(idx_val, "date"):
            row_date = idx_val.date()
        else:
            try:
                row_date = pd.Timestamp(idx_val).date()
            except Exception:
                dte_values.append(1.0)
                continue

        # Days to nearest future earnings date
        future_dates = [d for d in earnings_dt_set if d > row_date]
        if future_dates:
            days = (min(future_dates) - row_date).days
            dte_norm = min(days / 30.0, 1.0)   # normalise over 30-day window
        else:
            dte_norm = 1.0   # no upcoming earnings in record
        dte_values.append(dte_norm)

    f["days_to_earnings"] = dte_values

    # ── Constant per-ticker scalar features ──────────────────────────────────
    f["eps_revision_pct"] = float(eps_revision_pct)
    f["short_ratio"]      = min(float(short_ratio), 30.0) / 30.0   # normalise 0-1

    # ── 52-week low proximity ─────────────────────────────────────────────────
    high52 = close.rolling(252, min_periods=50).max()
    low52  = close.rolling(252, min_periods=50).min()
    rng    = (high52 - low52).replace(0, np.nan)
    f["low52_proximity"] = ((close - low52) / rng).clip(0, 1)

    # ── Pre-earnings ATR compression ─────────────────────────────────────────
    # ATR_5 / ATR_30 — values < 0.7 indicate compression (squeeze)
    atr5  = _atr(df, 5)
    atr30 = _atr(df, 30)
    f["pre_earnings_compression"] = (atr5 / atr30.replace(0, np.nan)).clip(0, 3)

    return f


def build_features(df: pd.DataFrame, vol_signal: VolumeSignal | None = None) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV DataFrame.
    Returns a DataFrame aligned to df's index (NaN rows at the start).
    """
    close = df["Close"].astype(float)
    f = pd.DataFrame(index=df.index)

    f["rsi_14"] = _rsi(close, 14)
    f["macd_hist"] = _macd_histogram(close)
    f["atr_14"] = _atr(df, 14)
    f["bb_width_pct"] = _bb_width_percentile(close, 20, 100)
    f["momentum_5"] = _momentum_5(close)

    # Volume Z-score (rolling, using VolumeSignal window)
    window = vol_signal.window if vol_signal else 20
    vol = df["Volume"].astype(float)
    roll_mean = vol.rolling(window).mean()
    roll_std = vol.rolling(window).std().replace(0, np.nan)
    f["volume_zscore"] = (vol - roll_mean) / roll_std

    return f


# ── SpikePredictor ───────────────────────────────────────────────────────────

class SpikePredictor:
    def __init__(self, cfg: dict):
        signals_cfg = cfg.get("signals", {}) if isinstance(cfg, dict) else {}
        spike_cfg = cfg.get("spike") or signals_cfg.get("spike") or cfg
        self.model_path     = Path(spike_cfg.get("model_path", "models/spike_model.pkl"))
        self.model_path_v3  = Path(spike_cfg.get("model_path_v3", "models/spike_model_v3_smallcap.pkl"))
        self.label_horizon: int   = int(spike_cfg.get("label_horizon", 5))
        self.label_horizon_v3: int = int(spike_cfg.get("label_horizon_v3", 3))
        self.label_pct: float     = float(spike_cfg.get("label_pct_threshold", 0.02))
        self.label_pct_v3: float  = float(spike_cfg.get("label_pct_v3", 0.08))
        self.predict_threshold: float = float(spike_cfg.get("predict_threshold", 0.60))
        self.retrain_lookback: int    = int(spike_cfg.get("retrain_lookback_days", 730))
        self.fmp_key: str = spike_cfg.get("fmp_key", "") or cfg.get("apis", {}).get("fmp_key", "")

        volume_cfg = cfg.get("volume") or signals_cfg.get("volume", {})
        self._vol_signal = VolumeSignal(volume_cfg)
        self._model    = self._load_model()
        self._model_v3 = self._load_model(v3=True)

    # ── Model persistence ────────────────────────────────────────────────────

    def _load_model(self, v3: bool = False):
        path = self.model_path_v3 if v3 else self.model_path
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"SpikePredictor: loaded {'v3' if v3 else 'v1'} model from {path}")
                return model
            except Exception as exc:
                logger.warning(f"SpikePredictor: failed to load model — {exc}")
        return None

    def _save_model(self, model, v3: bool = False):
        path = self.model_path_v3 if v3 else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"SpikePredictor: model saved to {path}")

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _fetch(self, symbol: str, lookback_days: int | None = None) -> pd.DataFrame:
        days = lookback_days or self.retrain_lookback
        ticker = yf.Ticker(symbol)
        try:
            df = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True)
        except Exception as exc:
            logger.warning(f"SpikePredictor: fetch failed for {symbol} — {exc}")
            return pd.DataFrame()
        return df

    def _build_labeled_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build (X, y) from a single symbol's OHLCV DataFrame."""
        features = build_features(df, self._vol_signal)
        close = df["Close"].astype(float)
        future_return = close.shift(-self.label_horizon) / close - 1
        labels = (future_return >= self.label_pct).astype(int)

        # Align and drop NaN rows
        combined = features.copy()
        combined["_label"] = labels
        combined = combined.dropna()

        X = combined.drop(columns=["_label"])
        y = combined["_label"]
        return X, y

    # ── Training ─────────────────────────────────────────────────────────────

    def retrain(self, symbols: list[str], lookback_days: int | None = None) -> None:
        """
        Fetch historical data for all symbols, build features+labels,
        train XGBoost, and save the model.
        Call weekly or whenever market regime shifts.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        all_X, all_y = [], []
        for sym in symbols:
            logger.info(f"SpikePredictor.retrain: fetching {sym}")
            df = self._fetch(sym, lookback_days)
            if len(df) < 150:
                logger.warning(f"SpikePredictor: skipping {sym} (insufficient data)")
                continue
            X, y = self._build_labeled_dataset(df)
            all_X.append(X)
            all_y.append(y)

        if not all_X:
            logger.error("SpikePredictor.retrain: no data collected, aborting.")
            return

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        pos_rate = y_all.mean()
        logger.info(f"Training set: {len(X_all)} rows, {pos_rate:.1%} positive labels")

        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, shuffle=False
        )

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info("Validation results:")
        preds = model.predict(X_val)
        logger.info("\n" + classification_report(y_val, preds))

        self._model = model
        self._save_model(model)

    # ── V3 training ──────────────────────────────────────────────────────────

    def retrain_v3(
        self,
        fmp_key: str | None = None,
        lookback_days: int | None = None,
        large_cap: list[str] | None = None,
        mid_cap:   list[str] | None = None,
        n_small_cap: int = 20,
    ) -> None:
        """
        Train the v3 small/mid-cap earnings catalyst model.

        Universe composition (configurable):
          large_cap (20%): AAPL, MSFT, NVDA  — baseline liquid tickers
          mid_cap   (40%): BYND, HIMS, SOUN, IONQ, OPEN, CLOV, WISH, SKLZ
          small_cap (40%): 20 random tickers from FMP earnings calendar < $2B

        Label: abs(close[t+3] / close[t] - 1) >= 8%
        (captures big moves in either direction — direction comes from
        sentiment + EPS revision, not this model)

        Extra features vs v1:
          days_to_earnings, eps_revision_pct, short_ratio,
          low52_proximity, pre_earnings_compression
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        fmp_key      = fmp_key or self.fmp_key
        lookback_days = lookback_days or self.retrain_lookback

        # ── Build training universe ───────────────────────────────────────────
        large  = large_cap or ["AAPL", "MSFT", "NVDA"]
        mid    = mid_cap   or ["BYND", "HIMS", "SOUN", "IONQ", "OPEN",
                                "CLOV", "WISH", "SKLZ"]
        small: list[str] = []
        if fmp_key:
            small = _fmp_small_cap_tickers(fmp_key, n=n_small_cap, lookback_days=lookback_days)
            logger.info(f"SpikePredictor v3: {len(small)} small-cap tickers from FMP")
        else:
            logger.warning("SpikePredictor v3: no fmp_key — skipping small-cap FMP pull")

        all_symbols = large + mid + small
        logger.info(
            f"SpikePredictor v3: training universe — "
            f"{len(large)} large, {len(mid)} mid, {len(small)} small "
            f"= {len(all_symbols)} total"
        )

        # ── Fetch historical earnings dates from FMP ──────────────────────────
        earnings_by_sym: dict[str, list[str]] = {}
        if fmp_key:
            calendar = _fmp_earnings_calendar(fmp_key, lookback_days)
            for ev in calendar:
                sym = (ev.get("symbol") or "").upper()
                dt  = ev.get("date", "")[:10]
                if sym and dt:
                    earnings_by_sym.setdefault(sym, []).append(dt)
            logger.info(f"SpikePredictor v3: earnings dates loaded for {len(earnings_by_sym)} tickers")

        # ── Build feature/label dataset ───────────────────────────────────────
        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series]    = []

        for sym in all_symbols:
            logger.info(f"SpikePredictor v3: processing {sym}")
            try:
                df = self._fetch(sym, lookback_days)
            except Exception as exc:
                logger.warning(f"  fetch failed: {exc}")
                continue
            if len(df) < 150:
                logger.warning(f"  insufficient data ({len(df)} rows), skipping")
                continue

            # Scalar per-ticker enrichment
            eps_rev    = _fmp_eps_revision(sym, fmp_key) if fmp_key else 0.0
            short_ratio = 0.0
            try:
                info = yf.Ticker(sym).fast_info
                # shortRatio not in fast_info — use full info
                short_ratio = float(yf.Ticker(sym).info.get("shortRatio") or 0)
            except Exception:
                pass

            # Build features with v3 extras
            features = build_features_v3(
                df,
                vol_signal=self._vol_signal,
                earnings_dates=earnings_by_sym.get(sym, []),
                eps_revision_pct=eps_rev,
                short_ratio=short_ratio,
            )

            # Label: abs 3-day return >= 8%
            close = df["Close"].astype(float)
            future_return = (close.shift(-self.label_horizon_v3) / close - 1).abs()
            labels = (future_return >= self.label_pct_v3).astype(int)

            combined = features.copy()
            combined["_label"] = labels
            combined = combined.dropna()
            if combined.empty:
                continue

            X = combined.drop(columns=["_label"])
            y = combined["_label"]
            all_X.append(X)
            all_y.append(y)
            time.sleep(0.2)   # polite yfinance rate-limiting

        if not all_X:
            logger.error("SpikePredictor v3: no data collected, aborting.")
            return

        X_all = pd.concat(all_X, ignore_index=True)
        y_all = pd.concat(all_y, ignore_index=True)

        pos_rate = y_all.mean()
        logger.info(
            f"SpikePredictor v3: training set {len(X_all)} rows, "
            f"{pos_rate:.1%} positive labels (abs ≥ 8%)"
        )
        logger.info(f"Features: {list(X_all.columns)}")

        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, shuffle=False
        )

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation metrics
        preds = model.predict(X_val)
        logger.info("SpikePredictor v3 — Validation results:")
        logger.info("\n" + classification_report(y_val, preds))

        # Feature importances
        importances = dict(zip(X_all.columns, model.feature_importances_))
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("Feature importances (top 10):")
        for feat, imp in ranked[:10]:
            logger.info(f"  {feat:<35} {imp:.4f}")

        self._model_v3 = model
        self._save_model(model, v3=True)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, symbol_or_df: str | pd.DataFrame) -> float:
        """
        Returns spike_probability ∈ [0, 1] for the most recent candle.
        Returns 0.0 if model is not trained.
        """
        if self._model is None:
            logger.debug("SpikePredictor: no model loaded, returning 0.0")
            return 0.0

        if isinstance(symbol_or_df, str):
            df = self._fetch(symbol_or_df, lookback_days=200)
        else:
            df = symbol_or_df

        if df is None or len(df) < 110:
            return 0.0

        features = build_features(df, self._vol_signal).dropna()
        if features.empty:
            return 0.0

        row = features.iloc[[-1]]
        prob = float(self._model.predict_proba(row)[0][1])
        logger.debug(f"SpikePredictor: spike_probability={prob:.4f}")
        return prob

    def is_flagged(self, symbol_or_df: str | pd.DataFrame) -> bool:
        return self.predict(symbol_or_df) >= self.predict_threshold

    def predict_v3(
        self,
        symbol_or_df: str | pd.DataFrame,
        eps_revision_pct: float = 0.0,
        short_ratio: float = 0.0,
        earnings_dates: list[str] | None = None,
    ) -> float:
        """
        Returns v3 spike_probability ∈ [0, 1] — probability of an abs ≥ 8%
        move within 3 candles.  Uses v3 feature set (earnings catalyst model).
        Falls back to v1 prediction if v3 model not yet trained.
        """
        if self._model_v3 is None:
            logger.debug("SpikePredictor: v3 model not loaded, falling back to v1")
            return self.predict(symbol_or_df)

        if isinstance(symbol_or_df, str):
            df = self._fetch(symbol_or_df, lookback_days=200)
        else:
            df = symbol_or_df

        if df is None or len(df) < 110:
            return 0.0

        features = build_features_v3(
            df,
            vol_signal=self._vol_signal,
            earnings_dates=earnings_dates or [],
            eps_revision_pct=eps_revision_pct,
            short_ratio=short_ratio,
        ).dropna()

        if features.empty:
            return 0.0

        # Align columns to what the model was trained on
        model_cols = self._model_v3.get_booster().feature_names
        missing = set(model_cols) - set(features.columns)
        for col in missing:
            features[col] = 0.0
        features = features[model_cols]

        row  = features.iloc[[-1]]
        prob = float(self._model_v3.predict_proba(row)[0][1])
        logger.debug(f"SpikePredictor v3: spike_probability={prob:.4f}")
        return prob
