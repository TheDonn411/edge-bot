from __future__ import annotations

import csv
import json
import math
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "demo_app" / "static"
DEFAULT_WATCHLIST = ROOT / "data" / "demo" / "watchlist.csv"

MID_CAP_MIN = 2_000.0
MID_CAP_MAX = 10_000.0


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_pct(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _score_from_bounds(value: float | None, lo: float, hi: float, invert: bool = False) -> float:
    if value is None:
        return 0.5
    scaled = _clamp((value - lo) / (hi - lo), 0.0, 1.0)
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


def _parse_watchlist(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    parsed: list[dict] = []
    for row in rows:
        market_cap = _safe_float(row.get("Market Cap"))
        price = _safe_float(row.get("Price"))
        volume = _safe_float(row.get("Volume"))
        change_pct = _safe_pct(row.get("Change"))
        pe_ratio = _safe_float(row.get("P/E"))

        if row.get("Country") != "USA":
            continue
        if market_cap is None or not (MID_CAP_MIN <= market_cap <= MID_CAP_MAX):
            continue
        if not row.get("Ticker") or not row.get("Company"):
            continue

        parsed.append(
            {
                "ticker": row["Ticker"].upper(),
                "company": row["Company"],
                "sector": row.get("Sector", "Unknown"),
                "industry": row.get("Industry", "Unknown"),
                "country": row.get("Country", ""),
                "market_cap_m": market_cap,
                "price": price,
                "change_pct": change_pct,
                "volume": volume,
                "pe_ratio": pe_ratio,
            }
        )
    return parsed


def _enrich(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    volumes = [r["volume"] for r in rows if r["volume"] is not None and r["volume"] > 0]
    median_volume = sorted(volumes)[len(volumes) // 2] if volumes else 1.0

    enriched: list[dict] = []
    for row in rows:
        change_pct = row["change_pct"] or 0.0
        price = row["price"] or 0.0
        volume = row["volume"] or 0.0
        pe_ratio = row["pe_ratio"]
        dollar_volume_m = (price * volume) / 1_000_000 if price and volume else 0.0
        volume_ratio = volume / median_volume if median_volume else 1.0

        momentum_score = _score_from_bounds(change_pct, -6.0, 6.0)
        volume_score = _score_from_bounds(volume_ratio, 0.5, 2.5)
        liquidity_score = _score_from_bounds(dollar_volume_m, 5.0, 250.0)
        valuation_score = _score_from_bounds(pe_ratio, 8.0, 35.0, invert=True)
        size_score = _score_from_bounds(row["market_cap_m"], MID_CAP_MIN, MID_CAP_MAX)
        composite_score = round(
            (
                0.30 * momentum_score
                + 0.25 * volume_score
                + 0.20 * liquidity_score
                + 0.15 * valuation_score
                + 0.10 * size_score
            ),
            4,
        )

        enriched.append(
            {
                **row,
                "market_cap_b": round(row["market_cap_m"] / 1000.0, 2),
                "dollar_volume_m": round(dollar_volume_m, 2),
                "volume_ratio": round(volume_ratio, 2),
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


def _build_thesis(change_pct: float, volume_ratio: float, pe_ratio: float | None, dollar_volume_m: float) -> list[str]:
    notes: list[str] = []
    if change_pct >= 2.0:
        notes.append("Strong green day relative to a medium-cap peer group.")
    elif change_pct <= -2.0:
        notes.append("Material pullback today; worth checking for mean-reversion or bad news.")
    else:
        notes.append("Daily move is moderate, so conviction should come from follow-through and liquidity.")

    if volume_ratio >= 1.5:
        notes.append("Volume is materially above the watchlist median, which usually means cleaner price discovery.")
    elif volume_ratio <= 0.8:
        notes.append("Volume is light versus peers, so the move may not be broadly sponsored.")

    if pe_ratio is None:
        notes.append("No simple P/E was available in the source file, so valuation confidence is lower.")
    elif pe_ratio <= 15:
        notes.append("Valuation screen looks relatively cheap on simple P/E.")
    elif pe_ratio >= 30:
        notes.append("Valuation is rich, so timing matters more than value support.")

    if dollar_volume_m >= 75:
        notes.append("Dollar volume is healthy enough for a clean demo entry/exit workflow.")
    else:
        notes.append("Liquidity is acceptable for a watchlist, but execution quality could still vary.")

    return notes


def load_dataset() -> list[dict]:
    source = Path(os.environ.get("EDGE_BOT_DEMO_WATCHLIST", DEFAULT_WATCHLIST))
    return _enrich(_parse_watchlist(source))


class DemoHandler(SimpleHTTPRequestHandler):
    dataset: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stocks":
            self._handle_stock_list(parsed.query)
            return
        if parsed.path.startswith("/api/stocks/"):
            ticker = parsed.path.rsplit("/", 1)[-1].upper()
            self._handle_stock_detail(ticker)
            return
        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def _handle_stock_list(self, query: str) -> None:
        params = parse_qs(query)
        search = params.get("q", [""])[0].strip().lower()
        sector = params.get("sector", [""])[0].strip().lower()

        filtered = self.dataset
        if search:
            filtered = [
                row for row in filtered
                if search in row["ticker"].lower() or search in row["company"].lower()
            ]
        if sector:
            filtered = [row for row in filtered if sector == row["sector"].lower()]

        payload = {
            "count": len(filtered),
            "stocks": filtered[:150],
            "sectors": sorted({row["sector"] for row in self.dataset}),
        }
        self._send_json(payload)

    def _handle_stock_detail(self, ticker: str) -> None:
        stock = next((row for row in self.dataset if row["ticker"] == ticker), None)
        if stock is None:
            self._send_json({"error": "Ticker not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(stock)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_server(host: str = "127.0.0.1", port: int = 8787) -> None:
    DemoHandler.dataset = load_dataset()
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Demo dashboard running at http://{host}:{port}")
    print(f"Loaded {len(DemoHandler.dataset)} medium-cap US stocks")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server(
        host=os.environ.get("EDGE_BOT_DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("EDGE_BOT_DEMO_PORT", "8787")),
    )
