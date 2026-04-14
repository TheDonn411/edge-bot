"""
EarningsScreener — nightly watchlist builder from FMP earnings calendar.

Replaces the static config.yaml watchlist each evening at 20:00 ET.

Pipeline
────────
1. Pull FMP /earning_calendar for today → today+14 days
2. Keep tickers with marketCap in [100M, 3B] and earnings 7–14 days out
3. Enrich each survivor with yfinance: beta, shortRatio, 52-week range
4. Require avgVolume > 300k
5. Score each with:
     fragility = (beta/5 * 0.3) + (min(shortRatio,15)/15 * 0.4)
                 + ((1 - low_proximity) * 0.3)
   where low_proximity = (price - 52wLow) / (52wHigh - 52wLow)
6. Return top 15 by fragility, save to config.yaml watchlist

Scheduling
──────────
run_scheduler() starts an APScheduler BlockingScheduler that fires at
20:00 ET every weekday (Mon–Fri).  Call it from main.py or run directly:

    python -m picker.screener
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import requests
import yaml
import yfinance as yf
from loguru import logger

ROOT = Path(__file__).parent.parent

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_CONFIG_PATH = ROOT / "config.yaml"


# ── Data containers ───────────────────────────────────────────────────────────

class Candidate(NamedTuple):
    symbol: str
    earnings_date: str          # YYYY-MM-DD
    days_to_earnings: int
    market_cap: float
    avg_volume: float
    beta: float
    short_ratio: float
    low_proximity: float        # 0 = at 52w low, 1 = at 52w high
    fragility: float


# ── FMP helpers ───────────────────────────────────────────────────────────────

def _fmp_get(endpoint: str, params: dict, fmp_key: str, timeout: int = 15) -> list | dict | None:
    params = {**params, "apikey": fmp_key}
    try:
        r = requests.get(f"{_FMP_BASE}/{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug(f"EarningsScreener FMP error [{endpoint}]: {exc}")
        return None


# ── Core screener ─────────────────────────────────────────────────────────────

class EarningsScreener:
    def __init__(self, cfg: dict):
        self.fmp_key: str = cfg.get("apis", {}).get("fmp_key", "")
        screen_cfg = cfg.get("screener", {})
        self.min_market_cap: float = float(screen_cfg.get("min_market_cap", 100_000_000))
        self.max_market_cap: float = float(screen_cfg.get("max_market_cap", 3_000_000_000))
        self.min_days_out: int     = int(screen_cfg.get("min_days_out", 7))
        self.max_days_out: int     = int(screen_cfg.get("max_days_out", 14))
        self.min_avg_volume: int   = int(screen_cfg.get("min_avg_volume", 300_000))
        self.top_n: int            = int(screen_cfg.get("top_n", 15))

    # ── Step 1: FMP earnings calendar ─────────────────────────────────────────

    def _fetch_earnings_calendar(self) -> list[dict]:
        today = datetime.now(timezone.utc).date()
        end   = today + timedelta(days=self.max_days_out + 1)
        data  = _fmp_get(
            "earning_calendar",
            {"from": today.isoformat(), "to": end.isoformat()},
            self.fmp_key,
        )
        if not isinstance(data, list):
            logger.warning("EarningsScreener: FMP returned unexpected type")
            return []
        logger.info(f"EarningsScreener: {len(data)} earnings events from FMP")
        return data

    # ── Step 2: marketCap + date filter ───────────────────────────────────────

    def _filter_events(self, events: list[dict]) -> list[dict]:
        today = datetime.now(timezone.utc).date()
        kept: list[dict] = []

        for ev in events:
            symbol = (ev.get("symbol") or "").strip().upper()
            if not symbol or "." in symbol:   # skip ADRs / preferred shares
                continue

            # Earnings date
            date_str = ev.get("date", "")
            try:
                earnings_dt = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            days_out = (earnings_dt - today).days
            if not (self.min_days_out <= days_out <= self.max_days_out):
                continue

            # marketCap (FMP includes it in the calendar response)
            mc = ev.get("marketCap")
            try:
                mc = float(mc) if mc is not None else None
            except (TypeError, ValueError):
                mc = None

            if mc is not None and not (self.min_market_cap <= mc <= self.max_market_cap):
                continue

            ev["_symbol"]       = symbol
            ev["_earnings_date"]= date_str[:10]
            ev["_days_out"]     = days_out
            ev["_market_cap"]   = mc or 0.0
            kept.append(ev)

        # De-duplicate by symbol (keep soonest earnings)
        seen: dict[str, dict] = {}
        for ev in kept:
            sym = ev["_symbol"]
            if sym not in seen or ev["_days_out"] < seen[sym]["_days_out"]:
                seen[sym] = ev

        logger.info(f"EarningsScreener: {len(seen)} candidates after date+marketCap filter")
        return list(seen.values())

    # ── Step 3+4: yfinance enrichment ─────────────────────────────────────────

    def _enrich(self, events: list[dict]) -> list[Candidate]:
        candidates: list[Candidate] = []

        for ev in events:
            symbol = ev["_symbol"]
            try:
                info = yf.Ticker(symbol).fast_info
                # fast_info is a lightweight dict-like object
                avg_vol    = getattr(info, "three_month_average_volume", None) or 0
                price      = getattr(info, "last_price", None) or 0
                high52     = getattr(info, "year_high", None) or 0
                low52      = getattr(info, "year_low", None) or 0
            except Exception as exc:
                logger.debug(f"EarningsScreener: yfinance fast_info failed for {symbol}: {exc}")
                avg_vol = price = high52 = low52 = 0

            if avg_vol < self.min_avg_volume:
                continue

            # Fall through to full info for beta + shortRatio
            try:
                full_info  = yf.Ticker(symbol).info
                beta       = float(full_info.get("beta") or 1.0)
                short_ratio= float(full_info.get("shortRatio") or 0.0)
                # Use full_info prices if fast_info didn't provide them
                if not price:
                    price  = float(full_info.get("currentPrice") or full_info.get("regularMarketPrice") or 0)
                if not high52:
                    high52 = float(full_info.get("fiftyTwoWeekHigh") or 0)
                if not low52:
                    low52  = float(full_info.get("fiftyTwoWeekLow") or 0)
                # Override market cap if FMP didn't include it
                if not ev["_market_cap"]:
                    mc = float(full_info.get("marketCap") or 0)
                    if not (self.min_market_cap <= mc <= self.max_market_cap):
                        continue
                    ev["_market_cap"] = mc
            except Exception as exc:
                logger.debug(f"EarningsScreener: yfinance full info failed for {symbol}: {exc}")
                beta = 1.0
                short_ratio = 0.0

            # 52-week range proximity (0 = at low, 1 = at high)
            rng = high52 - low52
            low_proximity = float((price - low52) / rng) if rng > 0 else 0.5

            # Fragility score
            fragility = (
                min(beta, 5.0) / 5.0 * 0.3
                + min(short_ratio, 15.0) / 15.0 * 0.4
                + (1.0 - low_proximity) * 0.3
            )

            candidates.append(Candidate(
                symbol       = symbol,
                earnings_date= ev["_earnings_date"],
                days_to_earnings= ev["_days_out"],
                market_cap   = ev["_market_cap"],
                avg_volume   = avg_vol,
                beta         = beta,
                short_ratio  = short_ratio,
                low_proximity= round(low_proximity, 4),
                fragility    = round(fragility, 4),
            ))

            # Polite rate limiting — yfinance full info can hammer Yahoo
            time.sleep(0.15)

        logger.info(f"EarningsScreener: {len(candidates)} candidates after volume + enrichment")
        return candidates

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> list[Candidate]:
        """
        Execute the full screening pipeline.
        Returns top-N candidates sorted by fragility descending.
        """
        if not self.fmp_key:
            logger.error("EarningsScreener: fmp_key not configured — aborting")
            return []

        events      = self._fetch_earnings_calendar()
        if not events:
            return []
        filtered    = self._filter_events(events)
        if not filtered:
            return []
        enriched    = self._enrich(filtered)
        if not enriched:
            return []

        enriched.sort(key=lambda c: c.fragility, reverse=True)
        top = enriched[:self.top_n]

        self._print_results(top)
        return top

    @staticmethod
    def _print_results(candidates: list[Candidate]) -> None:
        w = 72
        print(f"\n{'═'*w}")
        print(f"  EarningsScreener — {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
              f"({len(candidates)} picks)")
        print(f"{'═'*w}")
        print(f"  {'Symbol':<8} {'Earnings':>10} {'DTE':>4} {'MCap(M)':>9} "
              f"{'AvgVol':>9} {'Beta':>6} {'ShortR':>7} {'LowPrx':>7} {'Fragil':>7}")
        print(f"  {'─'*8} {'─'*10} {'─'*4} {'─'*9} {'─'*9} {'─'*6} {'─'*7} {'─'*7} {'─'*7}")
        for c in candidates:
            print(
                f"  {c.symbol:<8} {c.earnings_date:>10} {c.days_to_earnings:>4} "
                f"{c.market_cap/1e6:>9.0f} {c.avg_volume/1e3:>8.0f}K "
                f"{c.beta:>6.2f} {c.short_ratio:>7.1f} {c.low_proximity:>7.3f} "
                f"{c.fragility:>7.4f}"
            )
        print(f"{'═'*w}\n")


# ── Config update ─────────────────────────────────────────────────────────────

def update_watchlist(candidates: list[Candidate], config_path: Path = _CONFIG_PATH) -> None:
    """Overwrite config.yaml universe.watchlist with the screener output."""
    if not candidates:
        logger.warning("EarningsScreener: no candidates — watchlist unchanged")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    new_watchlist = [c.symbol for c in candidates]
    cfg["universe"]["watchlist"] = new_watchlist

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"EarningsScreener: watchlist updated → {new_watchlist}")


# ── APScheduler entry point ───────────────────────────────────────────────────

def run_scheduler(cfg: dict) -> None:
    """
    Start a BlockingScheduler that fires the screener at 20:00 ET
    every weekday.  Call from main.py or run this module directly.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    screener = EarningsScreener(cfg)

    def _job():
        logger.info("EarningsScreener: scheduled run starting")
        results = screener.run()
        update_watchlist(results)

    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        _job,
        CronTrigger(day_of_week="mon-fri", hour=20, minute=0,
                    timezone="America/New_York"),
        id="earnings_screener",
        name="Nightly earnings watchlist refresh",
        replace_existing=True,
    )

    logger.info("EarningsScreener: scheduler started — fires Mon–Fri 20:00 ET")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("EarningsScreener: scheduler stopped")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from config import load_config

    _cfg = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "schedule":
        run_scheduler(_cfg)
    else:
        _screener = EarningsScreener(_cfg)
        _results  = _screener.run()
        if _results:
            update_watchlist(_results)
