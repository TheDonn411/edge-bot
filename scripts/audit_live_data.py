from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_demo_dataset import FMP_BASE, _load_env_file

PAGES_BASE = "https://thedonn411.github.io/edge-bot"


def _get_json(url: str, **kwargs):
    response = requests.get(url, timeout=20, **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Request failed with HTTP {response.status_code} for {url.split('?')[0]}") from exc
    return response.json()


def _fmp_quote(symbol: str, api_key: str) -> dict:
    data = _get_json(f"{FMP_BASE}/quote/{symbol}", params={"apikey": api_key})
    return data[0] if isinstance(data, list) and data else {}


def _fmt(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GitHub Pages dashboard data against a fresh FMP quote.")
    parser.add_argument("symbols", nargs="*", default=["SM"], help="Ticker symbols to audit, e.g. SM AAL")
    args = parser.parse_args()

    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT / ".env.local")
    api_key = (
        os.environ.get("FMP_API_KEY")
        or os.environ.get("EDGE_BOT__APIS__FMP_KEY")
        or os.environ.get("EDGE_BOT_DEMO_FMP_KEY")
        or ""
    ).strip()
    if not api_key:
        raise SystemExit("Missing FMP_API_KEY in the environment or .env file.")

    meta = _get_json(f"{PAGES_BASE}/data/meta.json")
    stocks = _get_json(f"{PAGES_BASE}/data/stocks.json")
    stocks_by_symbol = {row["ticker"]: row for row in stocks}

    print(json.dumps({
        "pages_generated_at": meta.get("generated_at"),
        "pages_source": meta.get("source"),
        "quote_stats": meta.get("quote_stats", {}),
    }, indent=2))

    for raw_symbol in args.symbols:
        symbol = raw_symbol.upper()
        page_row = stocks_by_symbol.get(symbol)
        try:
            live_quote = _fmp_quote(symbol, api_key)
        except RuntimeError as exc:
            live_quote = {}
            print(f"\n{symbol}: fresh FMP check unavailable: {exc}")
        if not page_row:
            print(f"\n{symbol}: not present in the deployed dashboard dataset")
            continue

        if live_quote:
            print(f"\n{symbol}")
        print(f"  page_source:       {page_row.get('data_source')}")
        print(f"  page_live_quote:   {page_row.get('live_quote')}")
        print(f"  page_quote_time:   {page_row.get('quote_updated_at')}")
        print(f"  page_price:        {_fmt(page_row.get('price'))}")
        print(f"  fmp_price_now:     {_fmt(live_quote.get('price'))}")
        print(f"  page_change_pct:   {_fmt(page_row.get('change_pct'))}")
        print(f"  fmp_change_pct:    {_fmt(live_quote.get('changesPercentage'))}")
        print(f"  page_volume:       {_fmt(page_row.get('volume'))}")
        print(f"  fmp_volume_now:    {_fmt(live_quote.get('volume'))}")


if __name__ == "__main__":
    main()
