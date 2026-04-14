"""
Live API diagnostic + 5-stock pick runner.

Usage:
    python tests/run_live_test.py

Tests every data source in isolation, then runs the full picker
on the watchlist defined in config.yaml and prints the top 5 picks
with a complete score breakdown.
"""

from __future__ import annotations

import sys, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
import yfinance as yf
from loguru import logger

from config import load_config

# ── Pretty print helpers ──────────────────────────────────────────────────────

def _hdr(title: str):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def _ok(msg):  print(f"  ✓  {msg}")
def _warn(msg): print(f"  ⚠  {msg}")
def _fail(msg): print(f"  ✗  {msg}")

# ── Individual source tests ───────────────────────────────────────────────────

def test_yfinance(symbol: str = "AAPL"):
    _hdr(f"yfinance — OHLCV fetch ({symbol})")
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="5d", interval="1d", auto_adjust=True)
        if df.empty:
            _fail("Returned empty DataFrame")
            return False
        _ok(f"{len(df)} rows  |  latest close: ${df['Close'].iloc[-1]:.2f}")
        _ok(f"Columns: {list(df.columns)}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_finnhub_news_sentiment(symbol: str, key: str):
    _hdr(f"Finnhub /news-sentiment ({symbol})")
    if not key:
        _warn("finnhub_key not configured — skipping"); return False
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news-sentiment",
            params={"symbol": symbol, "token": key},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        if "sentiment" not in d:
            _warn(f"Unexpected response: {d}"); return False
        s = d["sentiment"]
        _ok(f"bullishPercent={s.get('bullishPercent', '?'):.1%}  "
            f"bearishPercent={s.get('bearishPercent', '?'):.1%}")
        _ok(f"companyNewsScore={d.get('companyNewsScore', '?'):.3f}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_finnhub_company_news(symbol: str, key: str):
    _hdr(f"Finnhub /company-news ({symbol})")
    if not key:
        _warn("finnhub_key not configured — skipping"); return False
    today = datetime.now(timezone.utc).date()
    week_ago = (today - timedelta(days=7)).isoformat()
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": week_ago,
                    "to": today.isoformat(), "token": key},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
        if not isinstance(items, list):
            _warn(f"Unexpected response type: {type(items)}"); return False
        _ok(f"{len(items)} articles in last 7 days")
        for item in items[:3]:
            _ok(f"  → {item.get('headline', '')[:80]}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_finnhub_congressional(symbol: str, key: str):
    _hdr(f"Finnhub /stock/congressional-trading ({symbol})")
    if not key:
        _warn("finnhub_key not configured — skipping"); return False
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=365)).isoformat()
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/congressional-trading",
            params={"symbol": symbol, "from": start,
                    "to": today.isoformat(), "token": key},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        trades = d.get("data", []) if isinstance(d, dict) else []
        _ok(f"{len(trades)} trades found in last 365 days")
        purchases = [t for t in trades if "purchase" in str(t.get("transaction","")).lower()]
        _ok(f"  Purchases: {len(purchases)}")
        for t in purchases[:3]:
            _ok(f"  → {t.get('name','?')}: "
                f"{t.get('transaction','?')} ${t.get('amount',0):,.0f} "
                f"on {t.get('transactionDate','?')}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_finnhub_insider(symbol: str, key: str):
    _hdr(f"Finnhub /stock/insider-transactions ({symbol})")
    if not key:
        _warn("finnhub_key not configured — skipping"); return False
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": symbol, "token": key},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        txns = d.get("data", []) if isinstance(d, dict) else []
        buys = [t for t in txns
                if t.get("transactionCode") == "P" and not t.get("isDerivative", False)]
        _ok(f"{len(txns)} total transactions, {len(buys)} open-market purchases")
        for t in buys[:3]:
            _ok(f"  → {t.get('name','?')}: "
                f"P {t.get('share',0):,.0f} shares @ ${t.get('value',0)/max(t.get('share',1),1):.2f} "
                f"= ${t.get('value',0):,.0f}  ({t.get('transactionDate','?')})")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_finnhub_executive(symbol: str, key: str):
    _hdr(f"Finnhub /stock/executive ({symbol})")
    if not key:
        _warn("finnhub_key not configured — skipping"); return False
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/executive",
            params={"symbol": symbol, "token": key},
            timeout=10,
        )
        r.raise_for_status()
        execs = r.json().get("executive", [])
        _ok(f"{len(execs)} executives found")
        for e in execs[:4]:
            _ok(f"  → {e.get('name','?'):<30} {e.get('title','?')}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_alpha_vantage_news(symbol: str, key: str):
    _hdr(f"Alpha Vantage NEWS_SENTIMENT ({symbol})")
    if not key:
        _warn("alpha_vantage_key not configured — skipping"); return False
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "NEWS_SENTIMENT", "tickers": symbol,
                    "limit": 5, "apikey": key},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        if "Information" in d:
            _warn(f"Rate limited: {d['Information'][:80]}"); return False
        feed = d.get("feed", [])
        _ok(f"{len(feed)} articles returned")
        for item in feed[:3]:
            ts = item.get("time_published","")
            score = item.get("overall_sentiment_score", "?")
            _ok(f"  → [{ts[:8]}] score={score}  {item.get('title','')[:60]}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_sec_edgar_13f(cik: str = "1350694", fund: str = "Bridgewater"):
    _hdr(f"SEC EDGAR — 13F filings ({fund} CIK={cik})")
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
            headers={"User-Agent": "edge-bot research@example.com"},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        recent = d.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        thirteenf = [(f, a) for f, a in zip(forms, recent.get("accessionNumber", []))
                     if f.startswith("13F")]
        _ok(f"{len(thirteenf)} 13F filings found")
        if thirteenf:
            _ok(f"  Latest: {thirteenf[0]}")
        return True
    except Exception as e:
        _fail(str(e)); return False


def test_volume_signal(symbol: str, cfg: dict):
    _hdr(f"VolumeSignal ({symbol})")
    from signals import VolumeSignal
    vs = VolumeSignal(cfg["signals"]["volume"])
    result = vs.compute(symbol, interval="1d")
    _ok(f"volume_score={result['volume_score']:.4f}  "
        f"flagged={result['flagged']}")
    _ok(f"  z-score={result['volume_zscore']:.3f}  "
        f"vwap_dev={result['vwap_deviation']:.4f}  "
        f"obv_slope={result['obv_slope']:.4f}")
    return True


def test_spike_predictor(symbol: str, cfg: dict):
    _hdr(f"SpikePredictor ({symbol})")
    from signals import SpikePredictor
    sp = SpikePredictor(cfg)
    if sp._model is None:
        _warn("No trained model found — run retrain() first")
        return False
    prob = sp.predict(symbol)
    _ok(f"spike_probability={prob:.4f}  flagged={sp.is_flagged(symbol)}")
    return True


def test_sentiment_signal(symbol: str, cfg: dict):
    _hdr(f"NewsSentimentSignal ({symbol})")
    from signals import NewsSentimentSignal
    sn = NewsSentimentSignal(cfg["signals"])
    result = sn.compute(symbol)
    _ok(f"sentiment_score={result['sentiment_score']:.4f}  "
        f"source={result.get('source','?')}")
    if result.get("bullish_pct"):
        _ok(f"  bullish%={result['bullish_pct']:.1%}  "
            f"news_score={result.get('news_score','?'):.3f}")
    for h in result.get("top_headlines", [])[:2]:
        _ok(f"  → {h[:75]}")
    return True


def test_flow_signal(symbol: str, cfg: dict):
    _hdr(f"FlowSignal ({symbol})")
    from signals import FlowSignal
    fs = FlowSignal(cfg["signals"])
    result = fs.compute([symbol]).get(symbol, {})
    _ok(f"flow_score={result.get('flow_score', 0):.4f}")
    _ok(f"  congressional={result.get('congressional_score', 0):.4f}  "
        f"institutional={result.get('institutional_score', 0):.4f}  "
        f"insider={result.get('insider_score', 0):.4f}")
    if result.get("ceo_buy_alert"):
        _warn("  ⚠ CEO_BUY_ALERT active!")
    return True


# ── Full picker run ────────────────────────────────────────────────────────────

def run_picker(cfg: dict, top_n: int = 5) -> pd.DataFrame:
    _hdr(f"StockPicker — scoring {len(cfg['universe']['watchlist'])} stocks")
    from picker import StockPicker

    # Temporarily override top_n in config
    cfg = dict(cfg)
    cfg["picker"] = dict(cfg["picker"])
    cfg["picker"]["top_n"] = top_n
    cfg["picker"]["score_threshold"] = 0.40   # lower bar so we always get 5 picks
    cfg["picker"]["dry_run"] = False

    picker = StockPicker(cfg)
    symbols = cfg["universe"]["watchlist"]

    print(f"  Symbols: {symbols}")
    print(f"  Running parallel signal computation...")
    t0 = time.time()

    df_scores = picker.score_live(symbols)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s\n")
    print(f"  {'Symbol':<8} {'Composite':>10} {'Volume':>8} {'Spike':>8} "
          f"{'Sentiment':>10} {'Flow':>8}")
    print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for row in df_scores.itertuples():
        print(f"  {row.symbol:<8} {row.composite_score:>10.4f} "
              f"{row.volume_score:>8.4f} {row.spike_probability:>8.4f} "
              f"{row.sentiment_score:>10.4f} {row.flow_score:>8.4f}")

    # Derive picks directly from already-scored df to avoid a second score_live call
    threshold = cfg["picker"]["score_threshold"]
    top_df = df_scores[df_scores["composite_score"] >= threshold].head(top_n)
    picks = list(top_df["symbol"])

    print(f"\n{'═'*60}")
    print(f"  TOP {top_n} PICKS")
    print(f"{'═'*60}")
    for rank, row in enumerate(top_df.itertuples(), 1):
        print(f"\n  #{rank}  {row.symbol}")
        print(f"       composite_score  : {row.composite_score:.4f}")
        print(f"       volume_score     : {row.volume_score:.4f}  (25% weight)")
        print(f"       spike_probability: {row.spike_probability:.4f}  (30% weight)")
        print(f"       sentiment_score  : {row.sentiment_score:.4f}  (20% weight)")
        print(f"       flow_score       : {row.flow_score:.4f}  (25% weight)")
    print(f"\n{'═'*60}\n")

    return top_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()   # suppress loguru to stderr — we print our own output
    logger.add(sys.stderr, level="WARNING")

    cfg = load_config()
    fh_key = cfg.get("apis", {}).get("finnhub_key", "")
    av_key = cfg.get("apis", {}).get("alpha_vantage_key", "")

    print(f"\nedge-bot Live Diagnostic  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Finnhub key : {'✓ ' + fh_key[:8] + '...' if fh_key else '✗ not set'}")
    print(f"AV key      : {'✓ ' + av_key[:8] + '...' if av_key else '✗ not set'}")

    # ── API spot-checks (using AAPL as representative ticker) ─────────────────
    probe = "AAPL"
    results = {}
    results["yfinance"]            = test_yfinance(probe)
    results["finnhub_sentiment"]   = test_finnhub_news_sentiment(probe, fh_key)
    results["finnhub_news"]        = test_finnhub_company_news(probe, fh_key)
    results["finnhub_congress"]    = test_finnhub_congressional(probe, fh_key)
    results["finnhub_insider"]     = test_finnhub_insider(probe, fh_key)
    results["finnhub_executive"]   = test_finnhub_executive(probe, fh_key)
    results["alpha_vantage_news"]  = test_alpha_vantage_news(probe, av_key)
    results["sec_edgar_13f"]       = test_sec_edgar_13f()

    # ── Signal-level tests ────────────────────────────────────────────────────
    results["volume_signal"]       = test_volume_signal(probe, cfg)
    results["spike_predictor"]     = test_spike_predictor(probe, cfg)
    results["sentiment_signal"]    = test_sentiment_signal(probe, cfg)
    results["flow_signal"]         = test_flow_signal(probe, cfg)

    # ── Summary ───────────────────────────────────────────────────────────────
    _hdr("Diagnostic Summary")
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{len(results)} sources operational\n")

    # ── Full picker → 5 picks ─────────────────────────────────────────────────
    top5 = run_picker(cfg, top_n=5)
    print("Picks ready. Run 'python main.py score' to rescore interactively.\n")
