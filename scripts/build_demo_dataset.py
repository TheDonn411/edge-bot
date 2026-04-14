from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
SOURCE_WATCHLIST = ROOT / "data" / "demo" / "watchlist.csv"
OUTPUT_DIR = ROOT / "demo_app" / "static" / "data"
OUTPUT_STOCKS = OUTPUT_DIR / "stocks.json"
OUTPUT_META = OUTPUT_DIR / "meta.json"

MID_CAP_MIN = 2_000.0
MID_CAP_MAX = 10_000.0
FMP_BASE = "https://financialmodelingprep.com/api/v3"
# FMP free plan: ~250 calls/day. One call per ticker, no batch endpoint.
MAX_LIVE_QUOTES = 200


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text in {"-", "None", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_pct(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _score_from_bounds(value: float | None, lo: float, hi: float, invert: bool = False) -> float:
    if value is None:
        return 0.5
    scaled = max(0.0, min((value - lo) / (hi - lo), 1.0))
    return round(1.0 - scaled if invert else scaled, 4)


def _trade_setup_label(composite: float, momentum: float, volume: float) -> str:
    if composite >= 0.72:
        return "High Conviction Watch"
    if momentum >= 0.65 and volume >= 0.6:
        return "Momentum Watch"
    if momentum <= 0.35 and volume >= 0.55:
        return "Pullback Reversal Watch"
    if composite >= 0.55:
        return "Active Watch"
    return "Monitor"


def _interesting_badges(
    change_pct: float,
    volume_ratio: float,
    pe_ratio: float | None,
    dollar_volume_m: float,
) -> list[str]:
    badges: list[str] = []
    if change_pct >= 2.0:
        badges.append("Strong up day")
    elif change_pct <= -2.0:
        badges.append("Sharp pullback")
    else:
        badges.append("Moderate move")

    if volume_ratio >= 1.5:
        badges.append("Heavy volume")
    elif volume_ratio <= 0.8:
        badges.append("Light volume")

    if pe_ratio is not None and pe_ratio <= 15:
        badges.append("Lower P/E")
    elif pe_ratio is not None and pe_ratio >= 30:
        badges.append("Rich multiple")

    if dollar_volume_m >= 75:
        badges.append("Liquid")
    return badges[:3]


def _build_thesis(change_pct: float, volume_ratio: float, pe_ratio: float | None, dollar_volume_m: float) -> list[str]:
    notes: list[str] = []
    if change_pct >= 2.0:
        notes.append("Strong green day relative to the medium-cap watchlist.")
    elif change_pct <= -2.0:
        notes.append("Meaningful pullback today; worth checking for reversal or fresh negative news.")
    else:
        notes.append("Price action is moderate, so follow-through matters more than the raw daily move.")

    if volume_ratio >= 1.5:
        notes.append("Volume is well above the watchlist median, which can mean better sponsorship.")
    elif volume_ratio <= 0.8:
        notes.append("Volume is light against peers, so conviction should be lower.")

    if pe_ratio is None:
        notes.append("No simple P/E was available from the source, so valuation confidence is lower.")
    elif pe_ratio <= 15:
        notes.append("Valuation looks relatively cheap on a simple earnings multiple basis.")
    elif pe_ratio >= 30:
        notes.append("Valuation is rich, so timing and momentum matter more than value support.")

    if dollar_volume_m >= 75:
        notes.append("Dollar volume is strong enough for a cleaner demo entry and exit narrative.")
    else:
        notes.append("Liquidity is acceptable for a watchlist, but execution quality could still vary.")

    return notes


def _why_now(change_pct: float, volume_ratio: float, pe_ratio: float | None, dollar_volume_m: float) -> str:
    if change_pct <= -2.0 and volume_ratio >= 1.5:
        return "It is getting hit on real volume, which makes it interesting for bounce-or-break setups after the close."
    if change_pct >= 2.0 and volume_ratio >= 1.5:
        return "It is already moving and the volume is confirming it, so it stands out from quieter medium-cap peers."
    if dollar_volume_m >= 100 and pe_ratio is not None and pe_ratio <= 15:
        return "It combines tradability with a relatively inexpensive simple valuation, which makes it easier to justify follow-up work."
    if volume_ratio >= 1.2:
        return "Volume is picking up enough to put it back on the radar even without a huge daily move."
    return "It still fits the medium-cap US universe and has enough liquidity to stay worth watching."


def _read_seed_watchlist(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for row in rows:
        ticker = (row.get("Ticker") or "").strip().upper()
        if not ticker:
            continue
        parsed.append(
            {
                "ticker": ticker,
                "company": row.get("Company", "").strip(),
                "sector": row.get("Sector", "").strip(),
                "industry": row.get("Industry", "").strip(),
                "country": row.get("Country", "").strip(),
                "market_cap_m": _safe_float(row.get("Market Cap")),
                "price": _safe_float(row.get("Price")),
                "change_pct": _safe_pct(row.get("Change")),
                "volume": _safe_float(row.get("Volume")),
                "pe_ratio": _safe_float(row.get("P/E")),
            }
        )
    return parsed


def _fmp_quote(symbol: str, api_key: str) -> dict:
    """Fetch a single-symbol quote from FMP (works on all plan tiers)."""
    response = requests.get(
        f"{FMP_BASE}/quote/{symbol}",
        params={"apikey": api_key},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data[0] if isinstance(data, list) and data else {}


def _refresh_from_fmp(seed_rows: list[dict], api_key: str) -> list[dict]:
    # Sort by market cap desc so the most liquid names are fetched first
    # if we hit the daily call cap.
    sorted_seed = sorted(
        seed_rows,
        key=lambda r: r.get("market_cap_m") or 0,
        reverse=True,
    )
    to_fetch = sorted_seed[:MAX_LIVE_QUOTES]
    ticker_set = {row["ticker"] for row in to_fetch}

    quote_map: dict[str, dict] = {}
    for i, row in enumerate(to_fetch):
        ticker = row["ticker"]
        try:
            quote = _fmp_quote(ticker, api_key)
            if quote:
                quote_map[ticker] = quote
        except Exception as exc:
            print(f"  Quote failed for {ticker}: {exc}")
        # Small pause to avoid hitting per-second rate limits
        if i and i % 10 == 0:
            time.sleep(0.5)

    print(f"  Fetched live quotes for {len(quote_map)}/{len(to_fetch)} tickers")

    refreshed: list[dict] = []
    for seed in seed_rows:
        ticker = seed["ticker"]
        if ticker not in ticker_set:
            # Beyond MAX_LIVE_QUOTES — keep seed data as-is
            refreshed.append(seed)
            continue

        quote = quote_map.get(ticker, {})
        market_cap = _safe_float(quote.get("marketCap")) or seed["market_cap_m"]
        market_cap_m = market_cap / 1_000_000 if market_cap and market_cap > 100_000 else market_cap

        refreshed.append(
            {
                "ticker": ticker,
                # Seed already has accurate company/sector/industry/country;
                # quote.name is a fallback if seed is blank.
                "company": (seed["company"] or quote.get("name") or ticker).strip(),
                "sector": (seed["sector"] or "Unknown").strip(),
                "industry": (seed["industry"] or "Unknown").strip(),
                "country": (seed["country"] or "Unknown").strip(),
                "market_cap_m": market_cap_m,
                "price": _safe_float(quote.get("price")) or seed["price"],
                "change_pct": _safe_float(quote.get("changesPercentage")) or seed["change_pct"],
                "volume": _safe_float(quote.get("volume")) or seed["volume"],
                "pe_ratio": _safe_float(quote.get("pe")) or seed["pe_ratio"],
            }
        )

    return refreshed


def _enrich(rows: list[dict]) -> list[dict]:
    filtered = [
        row for row in rows
        if row["country"] == "USA"
        and row["market_cap_m"] is not None
        and MID_CAP_MIN <= row["market_cap_m"] <= MID_CAP_MAX
    ]
    if not filtered:
        return []

    volumes = sorted(row["volume"] for row in filtered if row["volume"] is not None and row["volume"] > 0)
    median_volume = volumes[len(volumes) // 2] if volumes else 1.0

    enriched = []
    for row in filtered:
        change_pct = row["change_pct"] or 0.0
        price = row["price"] or 0.0
        volume = row["volume"] or 0.0
        pe_ratio = row["pe_ratio"]
        volume_ratio = volume / median_volume if median_volume else 1.0
        dollar_volume_m = (price * volume) / 1_000_000 if price and volume else 0.0

        momentum_score = _score_from_bounds(change_pct, -6.0, 6.0)
        volume_score = _score_from_bounds(volume_ratio, 0.5, 2.5)
        liquidity_score = _score_from_bounds(dollar_volume_m, 5.0, 250.0)
        valuation_score = _score_from_bounds(pe_ratio, 8.0, 35.0, invert=True)
        size_score = _score_from_bounds(row["market_cap_m"], MID_CAP_MIN, MID_CAP_MAX)
        composite_score = round(
            0.30 * momentum_score
            + 0.25 * volume_score
            + 0.20 * liquidity_score
            + 0.15 * valuation_score
            + 0.10 * size_score,
            4,
        )

        enriched.append(
            {
                **row,
                "market_cap_b": round((row["market_cap_m"] or 0.0) / 1000.0, 2),
                "dollar_volume_m": round(dollar_volume_m, 2),
                "volume_ratio": round(volume_ratio, 2),
                "interesting_badges": _interesting_badges(change_pct, volume_ratio, pe_ratio, dollar_volume_m),
                "why_now": _why_now(change_pct, volume_ratio, pe_ratio, dollar_volume_m),
                "signals": {
                    "momentum_score": momentum_score,
                    "volume_score": volume_score,
                    "liquidity_score": liquidity_score,
                    "valuation_score": valuation_score,
                    "size_score": size_score,
                    "composite_score": composite_score,
                },
                "setup_label": _trade_setup_label(composite_score, momentum_score, volume_score),
                "thesis": _build_thesis(change_pct, volume_ratio, pe_ratio, dollar_volume_m),
            }
        )

    enriched.sort(key=lambda row: row["signals"]["composite_score"], reverse=True)
    return enriched


def build_dataset() -> tuple[list[dict], dict]:
    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT / ".env.local")
    seed = _read_seed_watchlist(SOURCE_WATCHLIST)
    api_key = (
        os.environ.get("FMP_API_KEY")
        or os.environ.get("EDGE_BOT__APIS__FMP_KEY")
        or os.environ.get("EDGE_BOT_DEMO_FMP_KEY")
        or ""
    ).strip()

    source = "watchlist_csv_seed"
    if api_key:
        try:
            seed = _refresh_from_fmp(seed, api_key)
            source = "financial_modeling_prep"
        except Exception as exc:
            print(f"FMP refresh failed, falling back to seed CSV: {exc}")

    dataset = _enrich(seed)
    sectors = sorted({row["sector"] for row in dataset})
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "count": len(dataset),
        "sectors": sectors,
        "refresh_schedule": "Weekdays after US market close",
        "market_cap_band": {"min_musd": MID_CAP_MIN, "max_musd": MID_CAP_MAX},
    }
    return dataset, meta


def main() -> None:
    dataset, meta = build_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STOCKS.write_text(json.dumps(dataset, indent=2))
    OUTPUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(dataset)} stocks to {OUTPUT_STOCKS}")
    print(f"Metadata written to {OUTPUT_META}")
    print(f"Source: {meta['source']}")


if __name__ == "__main__":
    main()
