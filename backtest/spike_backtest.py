"""
spike_backtest.py — precision/recall + PnL curve for SpikePredictor.

Workflow:
  1. Fetch OHLCV for a symbol over the backtest period.
  2. Build the same feature set used at inference time.
  3. Generate entry signals: spike_probability >= entry_threshold.
  4. Simulate trades: enter at next open, exit after `exit_candles` candles.
  5. Compute classification metrics (precision, recall, F1).
  6. Plot equity curve using vectorbt.

Usage:
    python backtest/spike_backtest.py              # uses config.yaml defaults
    python backtest/spike_backtest.py TSLA 1d
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from sklearn.metrics import classification_report, precision_score, recall_score

# Add project root to sys.path when run as a script
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import load_config
from signals.spike_signal import SpikePredictor, build_features
from signals.volume_signal import VolumeSignal


def run_spike_backtest(
    cfg: dict,
    symbol: str | None = None,
    interval: str | None = None,
) -> dict:
    """
    Run full spike backtest.

    Returns:
        {
          "precision":      float,
          "recall":         float,
          "f1":             float,
          "total_trades":   int,
          "win_rate":       float,
          "equity_series":  pd.Series,   # dollar equity over time
          "report_text":    str,
        }
    """
    bt_cfg = cfg["backtest"]
    spike_bt_cfg = bt_cfg.get("spike", {})

    symbol   = symbol   or spike_bt_cfg.get("symbol", "AAPL")
    interval = interval or spike_bt_cfg.get("interval", "1d")
    start    = bt_cfg["start"]
    end      = bt_cfg["end"]
    initial_capital = bt_cfg["initial_capital"]
    entry_threshold = float(spike_bt_cfg.get("entry_threshold", 0.60))
    exit_candles    = int(spike_bt_cfg.get("exit_candles", 5))
    label_horizon   = int(cfg["signals"]["spike"].get("label_horizon", 5))
    label_pct       = float(cfg["signals"]["spike"].get("label_pct_threshold", 0.02))

    logger.info(f"SpikeBacktest: {symbol} [{interval}] {start} → {end}")

    # ── Fetch data ────────────────────────────────────────────────────────────
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty or len(df) < 150:
        logger.error(f"SpikeBacktest: insufficient data for {symbol}")
        return {}

    logger.info(f"Fetched {len(df)} candles for {symbol}")

    # ── Build features ────────────────────────────────────────────────────────
    vol_signal = VolumeSignal(cfg["signals"].get("volume", {}))
    features = build_features(df, vol_signal).dropna()

    close = df["Close"].astype(float).reindex(features.index)

    # ── True labels ──────────────────────────────────────────────────────────
    future_return = close.shift(-label_horizon) / close - 1
    y_true = (future_return >= label_pct).astype(int)

    # Align; drop last `label_horizon` rows (no future data)
    valid_idx = features.index[:-label_horizon]
    features = features.loc[valid_idx]
    y_true = y_true.loc[valid_idx]
    close = close.loc[valid_idx]

    # ── Load or train model ───────────────────────────────────────────────────
    predictor = SpikePredictor(cfg)
    if predictor._model is None:
        logger.info("SpikeBacktest: no model found — training on this symbol first")
        predictor.retrain(
            [symbol],
            lookback_days=int(cfg["signals"]["spike"].get("retrain_lookback_days", 730)),
        )

    if predictor._model is None:
        logger.error("SpikeBacktest: model training failed, aborting")
        return {}

    # ── Generate predictions row by row ──────────────────────────────────────
    probs = predictor._model.predict_proba(features)[:, 1]
    y_pred = (probs >= entry_threshold).astype(int)

    # ── Classification metrics ────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── PnL simulation ────────────────────────────────────────────────────────
    equity = initial_capital
    equity_curve = []
    wins = 0
    total_trades = 0

    prices = close.values
    signals = y_pred

    i = 0
    while i < len(prices) - exit_candles:
        equity_curve.append(equity)
        if signals[i] == 1:
            entry_price = prices[i]
            exit_idx = min(i + exit_candles, len(prices) - 1)
            exit_price = prices[exit_idx]

            pct_return = (exit_price - entry_price) / entry_price
            # Size: invest 10% of equity per trade
            position_pct = 0.10
            trade_pnl = equity * position_pct * pct_return
            equity += trade_pnl
            total_trades += 1
            if pct_return > 0:
                wins += 1

            i = exit_idx + 1
        else:
            i += 1

    # Fill remaining equity (hold cash)
    while len(equity_curve) < len(prices):
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=close.index[: len(equity_curve)])
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # ── vectorbt portfolio view ───────────────────────────────────────────────
    try:
        import vectorbt as vbt

        # Build entry/exit arrays aligned to price series
        entries = pd.Series(False, index=close.index)
        exits   = pd.Series(False, index=close.index)
        j = 0
        while j < len(close.index) - exit_candles:
            if signals[j] == 1:
                entries.iloc[j] = True
                exits.iloc[min(j + exit_candles, len(close.index) - 1)] = True
                j += exit_candles + 1
            else:
                j += 1

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=0.001,
            freq="1D" if interval == "1d" else interval.upper(),
        )
        logger.info("\n" + str(pf.stats()))
    except Exception as exc:
        logger.warning(f"vectorbt portfolio view failed (non-critical): {exc}")

    # ── Print summary ─────────────────────────────────────────────────────────
    final_equity = equity_series.iloc[-1]
    cagr = (final_equity / initial_capital) ** (252 / max(len(equity_series), 1)) - 1

    print(f"\n{'='*50}")
    print(f"  Spike Backtest: {symbol} [{start} → {end}]")
    print(f"{'='*50}")
    print(f"  Total trades   : {total_trades}")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Precision       : {precision:.3f}")
    print(f"  Recall          : {recall:.3f}")
    print(f"  F1 score        : {f1:.3f}")
    print(f"  Initial capital : ${initial_capital:,.0f}")
    print(f"  Final equity    : ${final_equity:,.0f}")
    print(f"  CAGR (approx)   : {cagr:.2%}")
    print(f"\nClassification Report:\n{report}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "equity_series": equity_series,
        "report_text": report,
    }


if __name__ == "__main__":
    _symbol   = sys.argv[1] if len(sys.argv) > 1 else None
    _interval = sys.argv[2] if len(sys.argv) > 2 else None
    _cfg = load_config()
    run_spike_backtest(_cfg, symbol=_symbol, interval=_interval)
