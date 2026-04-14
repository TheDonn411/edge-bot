"""
Fetch fresh congressional trade data from Senate/House Stock Watcher APIs
and save to data/congressional_cache.json.

Run this on any machine that has external DNS access (e.g., your laptop
or GitHub Actions).  The edge-bot server can then load the cached file
even if it cannot resolve senatestockwatcher.com / housestockwatcher.com.

Usage:
    python scripts/fetch_congressional_cache.py

The file is also updated nightly at 18:00 ET by GitHub Actions
(.github/workflows/fetch_congress.yml).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "congressional_cache.json"

SENATE_URL = "https://senatestockwatcher.com/api/transactions"
HOUSE_URL  = "https://housestockwatcher.com/api/transactions"
TIMEOUT    = 30


def _fetch(url: str, label: str) -> list[dict]:
    """Fetch all pages from a Stock Watcher endpoint."""
    all_rows: list[dict] = []
    page = 1
    while True:
        try:
            r = requests.get(url, params={"page": page}, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"  ✗ {label} page {page}: {exc}", file=sys.stderr)
            break

        batch = data if isinstance(data, list) else data.get("data", [])
        if not batch:
            break
        all_rows.extend(batch)
        print(f"  {label} page {page}: {len(batch)} rows (total {len(all_rows)})")
        if len(batch) < 10:
            break
        page += 1

    return all_rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching congressional trade data → {OUTPUT_PATH}\n")

    senate = _fetch(SENATE_URL, "Senate")
    house  = _fetch(HOUSE_URL,  "House")

    cache = {
        "senate":     senate,
        "house":      house,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f)

    total = len(senate) + len(house)
    print(f"\nSaved {total} trades ({len(senate)} Senate, {len(house)} House) "
          f"to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
