# edge-bot

A modular Python trading bot with signal generation, stock selection, risk management, and backtesting.

## Requirements

- Python 3.11+ (project uses 3.9+ compatible syntax; upgrade recommended for production)
- See `requirements.txt` for package dependencies

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Score today's watchlist (dry run)
python main.py score

# Run a backtest over config.yaml date range
python main.py backtest

# Start the live scheduler loop (paper trading by default)
python main.py live
```

## Demo Dashboard

The repo includes a deployable static dashboard for medium-cap US stocks.
It uses the bundled watchlist for ticker discovery, then refreshes current
price, daily move, volume, market cap, sector, industry, and simple valuation
from Financial Modeling Prep when `FMP_API_KEY` is configured.

### Local Preview

```bash
cd /Users/seandonnelly/edge-bot\ copy
source .venv/bin/activate
python3 scripts/run_demo_dashboard.py
```

Open `http://127.0.0.1:8787`.

If that port is already in use:

```bash
EDGE_BOT_DEMO_PORT=8788 ./.venv/bin/python scripts/run_demo_dashboard.py
```

### Fresh Data Setup

1. Create a free API key at [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/quickstart).
2. Put it in `.env` as `FMP_API_KEY=...` or add it as a GitHub Actions secret.
3. Rebuild the static data:

```bash
./.venv/bin/python scripts/build_demo_dataset.py
```

Generated files:

- `demo_app/static/data/stocks.json`
- `demo_app/static/data/meta.json`

### GitHub Pages Deployment

This repo includes:

- `/.github/workflows/refresh_demo_data.yml`
- `/.github/workflows/deploy_demo_pages.yml`

To deploy publicly:

1. Push the repo to GitHub.
2. Add a repository secret named `FMP_API_KEY`.
3. In GitHub, open `Settings -> Pages`.
4. Set the source to `GitHub Actions`.
5. Push to your default branch.

The refresh workflow runs every weekday after the US market close and commits updated demo data.
That commit triggers the Pages deployment workflow automatically.

Free FMP accounts have a limited daily request budget. The dashboard uses
single-symbol quote calls, defaults to 240 refreshed tickers per run, and leaves
a small buffer for manual checks. Stocks outside that daily budget remain
searchable, but the UI marks them as seed watchlist rows rather than live quotes.
On the free/basic tier, treat this as end-of-day refreshed market data rather
than streaming real-time broker pricing.

To audit the deployed page against a fresh FMP quote:

```bash
./.venv/bin/python scripts/audit_live_data.py SM AAL
```

If a broker display such as eToro disagrees with the dashboard, check whether
the dashboard row says it was live-refreshed, compare the quote timestamp, and
remember that CFD/broker screens may show bid/ask, spreads, delayed quotes,
after-hours movement, or a different change baseline from FMP's quote endpoint.

### First Push To `TheDonn411/edge-bot`

If this folder is not already a git checkout, publish it with:

```bash
cd /Users/seandonnelly/edge-bot\ copy
git init
git branch -M main
git remote add origin https://github.com/TheDonn411/edge-bot.git
git add .
git status
git commit -m "Initial dashboard and bot import"
git push -u origin main
```

The included `.gitignore` keeps local secrets and the virtualenv out of Git.

After the push:

1. Open `https://github.com/TheDonn411/edge-bot/actions`.
2. Run `Refresh Demo Data` once with `Run workflow`.
3. Confirm it finishes successfully and that `demo_app/static/data/meta.json` shows `financial_modeling_prep`.
4. Run `Deploy Demo Site` once if it does not start automatically.
5. Open the Pages URL shown in the workflow summary.

Expected Pages URL:

```text
https://thedonn411.github.io/edge-bot/
```

## Project Structure

```
edge-bot/
├── data/                   # Market data layer
│   ├── raw/                # Untouched downloaded data (future use)
│   ├── processed/          # Parquet cache (auto-populated by DataLoader)
│   └── loader.py           # DataLoader: fetch + cache OHLCV via yfinance
├── signals/                # Individual signal modules
│   ├── base.py             # BaseSignal ABC — all signals implement compute() → [-1,1]
│   ├── momentum.py         # Rate-of-change over a configurable lookback window
│   ├── mean_reversion.py   # Bollinger Band z-score (buy dips, fade rips)
│   ├── ml_score.py         # XGBoost classifier on engineered features (disabled until trained)
│   └── sentiment.py        # FinBERT news sentiment (disabled until model downloaded)
├── picker/                 # Stock selection
│   └── stock_picker.py     # Combines signal scores → composite rank → top-N picks
│                           # Enforces score threshold + per-sector cap
├── risk/                   # Risk management
│   ├── position_sizer.py   # Fractional Kelly sizing, capped at max_portfolio_pct
│   └── kill_switch.py      # Portfolio drawdown, daily loss, and per-position stop-loss
├── execution/              # Broker integration
│   ├── broker_base.py      # Abstract interface: connect, place_order, get_positions, ...
│   └── ibkr_stub.py        # Paper-trading stub simulating IBKR with slippage + commission
├── backtest/               # Backtesting
│   └── engine.py           # Event-driven daily loop; prints CAGR, Sharpe, max drawdown
├── config.yaml             # All tunable parameters (see section below)
├── config.py               # Loads config.yaml; supports env-var overrides
└── main.py                 # Orchestrator: live | backtest | score modes
```

## Module Details

### `data/`
`DataLoader` fetches OHLCV data from yfinance and caches it as Parquet files in `data/processed/`. Cache is skipped in backtest and live modes to ensure fresh data.

### `signals/`
Each signal implements `compute(df) -> float` in **[-1, 1]** (−1 = strong sell, +1 = strong buy):

| Signal | Status | Description |
|---|---|---|
| `MomentumSignal` | ✅ enabled | Normalised rate-of-change over `lookback` days |
| `MeanReversionSignal` | ✅ enabled | Bollinger Band z-score (inverted) |
| `MLScoreSignal` | ⚙️ disabled | XGBoost P(up) — flip enabled after training |
| `SentimentSignal` | ⚙️ disabled | FinBERT over news headlines — flip enabled after download |

### `picker/`
`StockPicker.pick()` computes a weighted composite score per symbol, filters by `score_threshold`, enforces `sector_cap`, and returns the top `top_n` symbols. Scores are normalised to **[0, 1]**.

### `risk/`
- **`PositionSizer`** — fractional Kelly sizing: `weight = (score − 0.5) × 2 × kelly_fraction`, capped at `max_portfolio_pct`.
- **`KillSwitch`** — checks three conditions every cycle:
  - Portfolio drawdown ≥ `max_drawdown_pct` → halt all trading
  - Daily P&L loss ≥ `daily_loss_limit_pct` → halt for the day
  - Position loss ≥ `stop_loss_pct` → close that position

### `execution/`
`IBKRStub` simulates order fills with configurable slippage (bps) and per-share commission. No real money or network calls. Swap in `ibkr_live.py` (using `ib_insync`) when ready.

### `backtest/`
`BacktestEngine` runs a day-by-day simulation: fetch history → generate signals → pick → size → "execute" via stub → track equity. Outputs CAGR, Sharpe ratio, and max drawdown.

## Configuration (`config.yaml`)

Key sections:

```yaml
universe:
  watchlist: [AAPL, MSFT, ...]   # symbols to trade
  max_positions: 10

signals:
  momentum:    { enabled: true, lookback: 20, weight: 1.0 }
  ml_score:    { enabled: false, model_path: models/xgb_v1.json }
  sentiment:   { enabled: false }

risk:
  max_portfolio_pct: 0.10   # max single-position size
  kelly_fraction: 0.25
  max_drawdown_pct: 0.15    # portfolio kill switch
  stop_loss_pct: 0.07       # per-position stop

execution:
  broker: ibkr_stub
  paper_trading: true

backtest:
  start: "2022-01-01"
  end:   "2024-12-31"
  initial_capital: 100_000
```

**Env-var overrides** — any config leaf can be overridden at runtime:
```bash
EDGE_BOT__RISK__MAX_DRAWDOWN_PCT=0.20 python main.py live
```

## Enabling the ML Signal

1. Build features and train a model (see `backtest/engine.py` for feature logic in `ml_score.py`).
2. Save the model: `booster.save_model("models/xgb_v1.json")`.
3. Set `signals.ml_score.enabled: true` in `config.yaml`.

## Enabling Sentiment

1. Set `signals.sentiment.enabled: true` in `config.yaml`.
2. The model (~500 MB) auto-downloads on first run via HuggingFace.
3. Provide headlines to `StockPicker.pick()` via the `headlines={sym: [...]}` argument.

## Replacing the Stub Broker

Implement `BrokerBase` in `execution/ibkr_live.py` using [ib_insync](https://github.com/erdewit/ib_insync), then change `execution.broker: ibkr_live` in `config.yaml` and update the import in `execution/__init__.py`.
