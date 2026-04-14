"""
FlowSignal — composite smart-money flow indicator.

Three sub-sources:
  1. Congressional trading  (Senate/House Stock Watcher APIs — free, no key)
  2. 13F hedge fund filings (SEC EDGAR API — free, no key)
  3. Options flow           (Unusual Whales API — stub/mock by default)

flow_score = congressional * 0.40 + institutional * 0.35 + options * 0.25
All sub-scores ∈ [0, 1].

Caches HTTP responses for the session to avoid redundant requests.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

_CONGRESSIONAL_CACHE_PATH = Path(__file__).parent.parent / "data" / "congressional_cache.json"
_CONGRESSIONAL_CACHE_MAX_AGE_H = 24   # warn if cache older than this

# ── Constants ────────────────────────────────────────────────────────────────

_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_HEADERS = {"User-Agent": "edge-bot research@example.com"}

# Maps 13F issuer names (as they appear in SEC XML) → ticker symbols.
# Extend as needed; unrecognised names fall back to first-token heuristic.
_ISSUER_TO_TICKER: dict[str, str] = {
    "APPLE INC": "AAPL", "APPLE INC.": "AAPL",
    "MICROSOFT CORP": "MSFT", "MICROSOFT CORPORATION": "MSFT",
    "NVIDIA CORP": "NVDA", "NVIDIA CORPORATION": "NVDA",
    "ALPHABET INC": "GOOGL", "ALPHABET INC-CL A": "GOOGL", "ALPHABET INC-CL C": "GOOG",
    "AMAZON COM INC": "AMZN", "AMAZON.COM INC": "AMZN",
    "META PLATFORMS INC": "META", "META PLATFORMS INC-CLASS A": "META",
    "TESLA INC": "TSLA", "TESLA MOTORS INC": "TSLA",
    "ADVANCED MICRO DEVICES": "AMD", "ADVANCED MICRO DEVICES INC": "AMD",
    "JPMORGAN CHASE & CO": "JPM", "JPMORGAN CHASE AND CO": "JPM",
    "BROADCOM INC": "AVGO",
    "ORACLE CORP": "ORCL", "ORACLE CORPORATION": "ORCL",
    "SALESFORCE INC": "CRM", "SALESFORCE.COM INC": "CRM",
    "NETFLIX INC": "NFLX",
    "EXXON MOBIL CORP": "XOM", "EXXON MOBIL CORPORATION": "XOM",
    "PALANTIR TECHNOLOGIES INC": "PLTR",
    "BERKSHIRE HATHAWAY INC": "BRK.B",
    "UNITEDHEALTH GROUP INC": "UNH",
    "JOHNSON & JOHNSON": "JNJ",
    "VISA INC": "V", "VISA INC-CLASS A SHARES": "V",
    "MASTERCARD INC": "MA", "MASTERCARD INCORPORATED": "MA",
    "WALMART INC": "WMT",
    "PROCTER & GAMBLE CO": "PG",
    "HOME DEPOT INC": "HD",
    "CHEVRON CORP": "CVX", "CHEVRON CORPORATION": "CVX",
    "ABBVIE INC": "ABBV",
    "COSTCO WHOLESALE CORP": "COST",
    "ELI LILLY & CO": "LLY",
}


# ── Helper: safe JSON fetch ───────────────────────────────────────────────────

def _get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 10) -> dict | list | None:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug(f"FlowSignal HTTP error [{url}]: {exc}")
        return None


# ── Source 1: Congressional Trading ─────────────────────────────────────────

class CongressionalSignal:
    """
    Fetches Senate and House stock trades from the free Stock Watcher APIs,
    scores each Purchase by politician tier, amount band, and time decay,
    and returns a normalised congressional_score ∈ [0, 1] per ticker.

    APIs (no auth required):
      Senate: https://senatestockwatcher.com/api/transactions[_by_ticker]
      House:  https://housestockwatcher.com/api/transactions[_by_ticker]

    Results are cached for cache_ttl_hours (default 4 h) since these
    datasets update at most once per day.
    """

    SENATE_ALL_URL    = "https://senatestockwatcher.com/api/transactions"
    SENATE_TICKER_URL = "https://senatestockwatcher.com/api/transactions_by_ticker"
    HOUSE_ALL_URL     = "https://housestockwatcher.com/api/transactions"
    HOUSE_TICKER_URL  = "https://housestockwatcher.com/api/transactions_by_ticker"

    # Tier weights (checked against full name, case-insensitive substring)
    TIER_1_NAMES: list[str] = ["pelosi", "tuberville", "collins", "paul", "wasserman schultz"]
    TIER_2_NAMES: list[str] = []   # add committee chairs here as needed

    # Amount band → score  (keys are lowercase-stripped for matching)
    AMOUNT_SCORES: dict[str, float] = {
        "$1,001 - $15,000":      0.2,
        "$15,001 - $50,000":     0.4,
        "$50,001 - $100,000":    0.6,
        "$100,001 - $250,000":   0.8,
        "$250,001 - $500,000":   0.9,
        "$500,001+":             1.0,
    }

    # Time-decay buckets: (max_days_inclusive, multiplier)
    DECAY_BUCKETS: list[tuple[int, float]] = [
        (7,  1.0),
        (21, 0.7),
        (45, 0.4),
    ]  # > 45 days → 0.0 (STOCK Act 45-day reporting window closed)

    def __init__(self, cfg: dict):
        self.tier1_w: float   = float(cfg.get("tier1_weight", 2.0))
        self.tier2_w: float   = float(cfg.get("tier2_weight", 1.5))
        self.tier3_w: float   = float(cfg.get("tier3_weight", 1.0))
        self.cache_ttl: float = float(cfg.get("cache_ttl_hours", 4.0)) * 3600
        self.max_pages: int   = int(cfg.get("max_pages", 3))
        self.finnhub_key: str = cfg.get("finnhub_key", "")

        # Cache: (fetched_at_ts, DataFrame)
        self._cache: tuple[float, pd.DataFrame] | None = None
        # Per-ticker Finnhub cache: {ticker: (ts, score)}
        self._fh_cache: dict[str, tuple[float, float]] = {}

    # ── Tier classification ───────────────────────────────────────────────────

    def _tier_weight(self, full_name: str) -> float:
        name_lower = full_name.lower()
        if any(t in name_lower for t in self.TIER_1_NAMES):
            return self.tier1_w
        if any(t in name_lower for t in self.TIER_2_NAMES):
            return self.tier2_w
        return self.tier3_w

    # ── Amount band parsing ───────────────────────────────────────────────────

    def _amount_score(self, amount_band: str) -> float:
        """Map an amount band string to a 0–1 score. Returns 0.1 for unknowns."""
        clean = amount_band.strip()
        # Exact match first
        if clean in self.AMOUNT_SCORES:
            return self.AMOUNT_SCORES[clean]
        # Fuzzy: find the entry whose key is a substring of the field (handles
        # minor spacing/capitalisation differences from the API)
        clean_lower = clean.lower().replace(" ", "")
        for key, score in self.AMOUNT_SCORES.items():
            if key.lower().replace(" ", "") in clean_lower:
                return score
        # Fallback: if it contains "500,001" or "million" treat as max band
        if "500,001" in clean or "million" in clean.lower() or "1,000,000" in clean:
            return 1.0
        logger.debug(f"CongressionalSignal: unknown amount band '{amount_band}', using 0.1")
        return 0.1

    # ── Time decay ────────────────────────────────────────────────────────────

    @staticmethod
    def _time_decay(transaction_date: str) -> float:
        """Parse YYYY-MM-DD and return decay multiplier."""
        try:
            dt = datetime.strptime(transaction_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - dt).days
        except Exception:
            return 0.0

        for max_days, multiplier in CongressionalSignal.DECAY_BUCKETS:
            if days_ago <= max_days:
                return multiplier
        return 0.0  # > 45 days

    # ── Fetching ──────────────────────────────────────────────────────────────

    def _fetch_chamber(self, base_url: str, chamber: str) -> list[dict]:
        """
        Paginate through `base_url` up to self.max_pages.
        Returns a flat list of raw trade dicts tagged with 'chamber'.
        Returns [] on failure — caller falls back to file cache.
        """
        rows: list[dict] = []
        for page in range(1, self.max_pages + 1):
            data = _get_json(base_url, params={"page": page}, timeout=15)
            if data is None:
                logger.warning(f"CongressionalSignal: {chamber} API unavailable (page {page})")
                break
            # Both APIs return a list directly or {"data": [...]}
            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break   # no more pages
            for row in batch:
                row["_chamber"] = chamber
            rows.extend(batch)
            if len(batch) < 10:
                break   # last page was short — no point fetching further
        return rows

    def _load_file_cache(self) -> tuple[list[dict], list[dict]]:
        """
        Load congressional trades from data/congressional_cache.json.
        Returns (senate_rows, house_rows).  Warns if cache is > 24 h old.
        """
        if not _CONGRESSIONAL_CACHE_PATH.exists():
            logger.debug("CongressionalSignal: no file cache found")
            return [], []
        try:
            with open(_CONGRESSIONAL_CACHE_PATH) as f:
                cache = json.load(f)
            fetched_at = cache.get("fetched_at", "")
            if fetched_at:
                try:
                    age_h = (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(fetched_at).replace(tzinfo=timezone.utc)
                    ).total_seconds() / 3600
                    if age_h > _CONGRESSIONAL_CACHE_MAX_AGE_H:
                        logger.warning(
                            f"CongressionalSignal: file cache is {age_h:.0f}h old "
                            f"(>{_CONGRESSIONAL_CACHE_MAX_AGE_H}h) — data may be stale"
                        )
                    else:
                        logger.info(f"CongressionalSignal: loaded file cache ({age_h:.1f}h old)")
                except Exception:
                    pass

            senate_rows = cache.get("senate", [])
            house_rows  = cache.get("house",  [])
            for r in senate_rows:
                r.setdefault("_chamber", "Senate")
            for r in house_rows:
                r.setdefault("_chamber", "House")
            return senate_rows, house_rows
        except Exception as exc:
            logger.warning(f"CongressionalSignal: failed to load file cache — {exc}")
            return [], []

    def _fetch_all(self) -> pd.DataFrame:
        """
        Fetch Senate + House in parallel.
        Falls back to data/congressional_cache.json if both live APIs fail.
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_senate = pool.submit(self._fetch_chamber, self.SENATE_ALL_URL, "Senate")
            f_house  = pool.submit(self._fetch_chamber, self.HOUSE_ALL_URL,  "House")
            senate_rows = f_senate.result()
            house_rows  = f_house.result()

        # If live APIs returned nothing, fall back to file cache
        if not senate_rows and not house_rows:
            logger.info("CongressionalSignal: live APIs empty — trying file cache")
            senate_rows, house_rows = self._load_file_cache()
        else:
            logger.debug("CongressionalSignal: using live API data")

        all_rows = senate_rows + house_rows
        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Normalise to unified column set
        # Senate/House field names match the spec exactly
        def _col(row, *keys):
            for k in keys:
                v = row.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
            return ""

        records = []
        for _, row in df.iterrows():
            first = _col(row, "first_name")
            last  = _col(row, "last_name")
            name  = f"{first} {last}".strip()
            records.append({
                "name":             name,
                "chamber":          row.get("_chamber", ""),
                "ticker":           _col(row, "ticker").upper(),
                "asset_description":_col(row, "asset_description"),
                "trade_type":       _col(row, "type"),
                "amount_band":      _col(row, "amount"),
                "transaction_date": _col(row, "transaction_date"),
                "date_received":    _col(row, "date_recieved"),   # API typo preserved
            })

        unified = pd.DataFrame(records)
        # Drop rows with no ticker
        unified = unified[unified["ticker"].str.len() > 0].copy()

        # Compute days_ago from transaction_date
        def _days_ago(date_str: str) -> int:
            try:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return max((datetime.now(timezone.utc) - dt).days, 0)
            except Exception:
                return 9999

        unified["days_ago"] = unified["transaction_date"].apply(_days_ago)
        return unified

    def _get_df(self) -> pd.DataFrame:
        """Return cached DataFrame, refreshing if the TTL has expired."""
        now = time.time()
        if self._cache is not None:
            fetched_at, df = self._cache
            if now - fetched_at < self.cache_ttl:
                return df

        logger.info("CongressionalSignal: fetching Senate + House trade data...")
        df = self._fetch_all()
        self._cache = (now, df)
        if not df.empty:
            logger.info(
                f"CongressionalSignal: loaded {len(df)} trades "
                f"({df['chamber'].value_counts().to_dict()})"
            )
        return df

    # ── Scoring ───────────────────────────────────────────────────────────────

    # ── Finnhub congressional (primary) ──────────────────────────────────────

    def _finnhub_score(self, ticker: str) -> float | None:
        """
        Fetch congressional trades for a single ticker from Finnhub.
        Returns a score ∈ [0, 1] or None on failure.
        Finnhub endpoint: GET /stock/congressional-trading
        """
        if not self.finnhub_key:
            return None

        # Check per-ticker cache
        if ticker in self._fh_cache:
            ts, score = self._fh_cache[ticker]
            if time.time() - ts < self.cache_ttl:
                return score

        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=180)).isoformat()

        data = _get_json(
            "https://finnhub.io/api/v1/stock/congressional-trading",
            params={"symbol": ticker, "from": start, "to": today.isoformat(),
                    "token": self.finnhub_key},
            timeout=10,
        )
        if not data:
            return None

        trades = data.get("data", []) if isinstance(data, dict) else []
        if not trades:
            self._fh_cache[ticker] = (time.time(), 0.0)
            return 0.0

        total = 0.0
        for trade in trades:
            tx_type = str(trade.get("transaction") or "").lower()
            if "purchase" not in tx_type:
                continue

            name = str(trade.get("name") or "")
            date_str = str(trade.get("transactionDate") or "")
            amount = float(trade.get("amount") or 0)

            decay = self._time_decay(date_str)
            if decay == 0.0:
                continue

            tier_w = self._tier_weight(name)
            # Map numeric amount to a 0–1 band score
            if amount >= 500_001:    amt_score = 1.0
            elif amount >= 250_001:  amt_score = 0.9
            elif amount >= 100_001:  amt_score = 0.8
            elif amount >= 50_001:   amt_score = 0.6
            elif amount >= 15_001:   amt_score = 0.4
            else:                    amt_score = 0.2

            total += tier_w * amt_score * decay

        # Normalise the same way as the Stock Watcher path
        score = round(min(total / max(self.tier1_w, 1.0), 1.0), 4)
        self._fh_cache[ticker] = (time.time(), score)
        if score > 0:
            logger.info(f"CongressionalSignal Finnhub [{ticker}]: score={score:.3f} ({len(trades)} trades)")
        return score

    def scores(self, tickers: list[str] | None = None) -> dict[str, float]:
        """
        Return {ticker: congressional_score ∈ [0, 1]}.
        Uses Finnhub per-ticker as primary; falls back to Stock Watcher bulk fetch.
        Only Purchase transactions within the 45-day decay window are counted.
        Logs the single top contributing trade per ticker.
        """
        # ── Finnhub primary path (per-ticker) ────────────────────────────────
        if self.finnhub_key and tickers:
            results: dict[str, float] = {}
            all_none = True
            for t in tickers:
                score = self._finnhub_score(t.upper())
                if score is not None:
                    results[t.upper()] = score
                    all_none = False
            if not all_none:
                return results
            # all calls failed → fall through to Stock Watcher

        # ── Stock Watcher fallback (bulk) ─────────────────────────────────────
        df = self._get_df()
        if df.empty:
            return {}

        # Filter to purchases only
        purchases = df[df["trade_type"].str.strip().str.lower() == "purchase"].copy()
        if tickers:
            upper_tickers = [t.upper() for t in tickers]
            purchases = purchases[purchases["ticker"].isin(upper_tickers)]

        if purchases.empty:
            return {}

        # Compute per-trade score component
        purchases["tier_w"]   = purchases["name"].apply(self._tier_weight)
        purchases["amt_score"] = purchases["amount_band"].apply(self._amount_score)
        purchases["decay"]     = purchases["transaction_date"].apply(self._time_decay)
        purchases["trade_score"] = (
            purchases["tier_w"] * purchases["amt_score"] * purchases["decay"]
        )

        # Drop expired trades
        active = purchases[purchases["decay"] > 0.0]
        if active.empty:
            return {}

        ticker_totals: dict[str, float] = defaultdict(float)
        ticker_top: dict[str, dict] = {}

        for _, row in active.iterrows():
            t = row["ticker"]
            ticker_totals[t] += row["trade_score"]
            if t not in ticker_top or row["trade_score"] > ticker_top[t]["score"]:
                ticker_top[t] = {
                    "name":    row["name"],
                    "amount":  row["amount_band"],
                    "days_ago":row["days_ago"],
                    "score":   row["trade_score"],
                    "chamber": row["chamber"],
                }

        if not ticker_totals:
            return {}

        # Normalise: divide by theoretical max a single trade can produce
        # (tier1_w * 1.0 * 1.0 = tier1_w), so scores > 1.0 mean multiple trades
        max_score = max(ticker_totals.values())
        result: dict[str, float] = {}
        for ticker, total in ticker_totals.items():
            score = round(min(total / max_score, 1.0), 4)
            result[ticker] = score
            top = ticker_top[ticker]
            logger.info(
                f"CongressionalSignal [{ticker}] top trade: "
                f"{top['name']} ({top['chamber']}): "
                f"Purchase {top['amount']}, {top['days_ago']}d ago "
                f"→ component={top['score']:.3f}, ticker_score={score:.3f}"
            )

        return result


# ── Source 2: 13F Institutional Filings ─────────────────────────────────────

class _InstitutionalSource:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.min_funds: int = int(cfg.get("min_funds_threshold", 3))
        self.target_funds: dict[str, str] = cfg.get("target_funds", {})
        self._cache: dict[str, dict] = {}

    def _fetch_latest_13f(self, cik: str) -> dict[str, int]:
        """
        Fetch the most recent 13F filing for a given CIK.
        Returns {ticker: shares} for all reported positions.
        """
        submissions = _get_json(
            _SEC_SUBMISSIONS_URL.format(cik=cik.zfill(10)),
            headers=_SEC_HEADERS,
        )
        if not submissions:
            return {}

        filings = submissions.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])

        # Find the most recent 13F-HR
        for form, accession in zip(forms, accessions):
            if form.startswith("13F"):
                # Fetch the primary document index
                acc_clean = accession.replace("-", "")
                index_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{acc_clean}/{accession}-index.htm"
                )
                # For simplicity, return the accession as proxy — full XML parse
                # would require beautifulsoup4 or lxml which are available in venv
                logger.debug(f"InstitutionalSource: found 13F {accession} for CIK {cik}")
                return self._parse_13f_holdings(cik, acc_clean)
        return {}

    def _find_holdings_xml_url(self, cik: str, acc_clean: str) -> str | None:
        """
        Discover the correct holdings XML filename from the filing directory.
        The cover sheet is primary_doc.xml; the actual holdings table is a
        separate XML file (name varies by filer, e.g. infotable.xml).
        """
        import re as _re
        dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"
        resp = requests.get(dir_url, headers=_SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        xml_files = _re.findall(
            rf'href="(/Archives/edgar/data/{cik}/{acc_clean}/[^"]+\.xml)"',
            resp.text,
        )
        for path in xml_files:
            fname = path.rsplit("/", 1)[-1]
            if fname != "primary_doc.xml":
                return f"https://www.sec.gov{path}"
        return None

    def _parse_13f_holdings(self, cik: str, acc_clean: str) -> dict[str, int]:
        """
        Download and parse the 13F XML holdings report.
        Returns {cusip_or_ticker: shares}.
        Note: 13F uses CUSIP; we return the name field as a proxy ticker.
        """
        try:
            import xml.etree.ElementTree as ET

            doc_url = self._find_holdings_xml_url(cik, acc_clean)
            if doc_url is None:
                return {}
            resp = requests.get(doc_url, headers=_SEC_HEADERS, timeout=15)
            if resp.status_code != 200:
                return {}

            root = ET.fromstring(resp.content)
            ns = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
            holdings = {}
            for entry in root.findall("ns:infoTable", ns):
                name_el   = entry.find("ns:nameOfIssuer", ns)
                shares_el = entry.find(".//ns:sshPrnamt", ns)
                cusip_el  = entry.find("ns:cusip", ns)
                if name_el is not None and shares_el is not None:
                    issuer = name_el.text.strip().upper()
                    # Map common issuer names to tickers
                    ticker = _ISSUER_TO_TICKER.get(issuer)
                    if ticker is None:
                        # Fallback: first token as rough proxy
                        ticker = issuer.split()[0]
                    try:
                        holdings[ticker] = int(shares_el.text.replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            return holdings
        except Exception as exc:
            logger.debug(f"InstitutionalSource: XML parse failed — {exc}")
            return {}

    def scores(self) -> dict[str, float]:
        """
        Return {ticker: score} where score = (num_funds_holding / min_funds_threshold).
        Capped at 1.0.
        """
        fund_holdings: dict[str, dict[str, int]] = {}

        for fund_name, cik in self.target_funds.items():
            if cik in self._cache:
                holdings = self._cache[cik]
            else:
                holdings = self._fetch_latest_13f(cik)
                self._cache[cik] = holdings
            fund_holdings[fund_name] = holdings

        # Count how many funds hold each ticker
        ticker_fund_count: dict[str, int] = defaultdict(int)
        for holdings in fund_holdings.values():
            for ticker in holdings:
                ticker_fund_count[ticker] += 1

        scores = {}
        for ticker, count in ticker_fund_count.items():
            if count >= self.min_funds:
                scores[ticker] = round(min(count / len(self.target_funds), 1.0), 4)

        return scores


# ── Source 3: Form 4 Insider Buying ──────────────────────────────────────────

class Form4InsiderSignal:
    """
    Fetches SEC EDGAR Form 4 filings (insider transactions) for a ticker
    over a configurable lookback window using the free EDGAR full-text search.

    Only open-market Purchase transactions (transactionCode == 'P') are scored.
    Awards ('A') and tax withholdings ('F') are ignored — they are not
    discretionary buys and carry no predictive signal.

    Insider role weights:
      CEO / President → 1.0    CFO / COO → 0.9
      10% Owner       → 0.8    Director  → 0.7

    insider_score per ticker = Σ(role_weight * amount_norm) / baseline
    capped at 1.0.

    Separately flags CEO_BUY_ALERT when a CEO or President makes a single
    open-market purchase exceeding ceo_buy_alert_threshold (default $500k).
    """

    # EDGAR full-text search for Form 4 filings mentioning a ticker
    _EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
    # EDGAR submission API used to fetch actual filing XML
    _EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    # Discretionary buy codes only
    _BUY_CODES = {"P"}
    _SKIP_CODES = {"A", "F", "M", "S", "D"}

    # Role keyword → weight (checked case-insensitively against rptOwnerRelationship)
    _ROLE_WEIGHTS: list[tuple[str, float]] = [
        ("president",  1.0),
        ("ceo",        1.0),
        ("chief executive", 1.0),
        ("cfo",        0.9),
        ("chief financial", 0.9),
        ("coo",        0.9),
        ("chief operating", 0.9),
        ("10%",        0.8),
        ("10 percent", 0.8),
        ("director",   0.7),
    ]

    def __init__(self, cfg: dict):
        insider_cfg = cfg if "ceo_buy_alert_threshold" in cfg else cfg.get("insider", cfg)
        self.lookback_days: int = int(insider_cfg.get("lookback_days", 30))
        self.alert_threshold: float = float(insider_cfg.get("ceo_buy_alert_threshold", 500_000))
        self.finnhub_key: str = insider_cfg.get("finnhub_key", "")
        rw = insider_cfg.get("role_weights", {})
        self._role_weights_cfg = {k.lower(): float(v) for k, v in rw.items()} if rw else {}
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl: float = 3600.0
        # Cached executive title map: {ticker: {name_lower: title}}
        self._exec_cache: dict[str, dict[str, str]] = {}

    # ── Role classification ───────────────────────────────────────────────────

    def _role_weight(self, relationship: str) -> float:
        rel = relationship.lower()
        # Config overrides first (keyed on simple role names)
        for keyword, weight in self._role_weights_cfg.items():
            if keyword in rel:
                return weight
        # Built-in defaults
        for keyword, weight in self._ROLE_WEIGHTS:
            if keyword in rel:
                return weight
        return 0.5  # unknown insider — give some credit

    # ── EDGAR fetching ────────────────────────────────────────────────────────

    def _search_form4(self, ticker: str) -> list[dict]:
        """
        Query EDGAR full-text search for recent Form 4 filings mentioning ticker.
        Returns a list of hit dicts from the search index.
        """
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=self.lookback_days)).isoformat()
        end   = today.isoformat()

        params = {
            "q": f'"{ticker}"',
            "forms": "4",
            "dateRange": "custom",
            "startdt": start,
            "enddt": end,
            "hits.hits.total.value": 1,
        }
        data = _get_json(self._EFTS_URL, params=params, headers=_SEC_HEADERS, timeout=15)
        if not data:
            return []
        hits = (data.get("hits") or {}).get("hits") or []
        return hits

    def _fetch_form4_xml(self, accession: str, cik: str) -> str | None:
        """Download the raw Form 4 XML for a given accession number."""
        acc_clean = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{acc_clean}/{accession}.xml"
        )
        try:
            resp = requests.get(url, headers=_SEC_HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp.text
            # Try alternate filename pattern
            url2 = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{acc_clean}/form4.xml"
            )
            resp2 = requests.get(url2, headers=_SEC_HEADERS, timeout=10)
            return resp2.text if resp2.status_code == 200 else None
        except Exception as exc:
            logger.debug(f"Form4InsiderSignal: XML fetch failed [{accession}]: {exc}")
            return None

    def _parse_transactions(self, xml_text: str) -> list[dict]:
        """
        Parse a Form 4 XML and extract non-derivative transactions.
        Returns list of dicts with keys:
          transactionCode, shares, pricePerShare, relationship, issuerTicker
        """
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.debug(f"Form4InsiderSignal: XML parse error — {exc}")
            return []

        relationship = ""
        rel_el = root.find(".//rptOwnerRelationship")
        if rel_el is not None:
            parts = []
            for tag in ("isOfficer", "isDirector", "isTenPercentOwner", "officerTitle"):
                el = rel_el.find(tag)
                if el is not None and el.text:
                    parts.append(el.text.strip())
            relationship = " ".join(parts)

        ticker_el = root.find(".//issuerTradingSymbol")
        issuer_ticker = ticker_el.text.strip().upper() if ticker_el is not None and ticker_el.text else ""

        transactions = []
        for tx in root.findall(".//nonDerivativeTransaction"):
            code_el = tx.find(".//transactionCode")
            shares_el = tx.find(".//transactionShares/value")
            price_el = tx.find(".//transactionPricePerShare/value")

            code = code_el.text.strip() if code_el is not None and code_el.text else ""
            if code not in self._BUY_CODES:
                continue

            try:
                shares = float(shares_el.text) if shares_el is not None else 0.0
                price  = float(price_el.text)  if price_el is not None else 0.0
            except (ValueError, TypeError):
                continue

            transactions.append({
                "transactionCode": code,
                "shares":          shares,
                "pricePerShare":   price,
                "total_value":     shares * price,
                "relationship":    relationship,
                "issuerTicker":    issuer_ticker,
            })

        return transactions

    # ── Finnhub insider (primary path) ───────────────────────────────────────

    def _finnhub_executives(self, ticker: str) -> dict[str, str]:
        """Fetch {name_lower: title} for a company's executives via Finnhub."""
        if ticker in self._exec_cache:
            return self._exec_cache[ticker]
        data = _get_json(
            "https://finnhub.io/api/v1/stock/executive",
            params={"symbol": ticker, "token": self.finnhub_key},
            timeout=10,
        )
        mapping: dict[str, str] = {}
        if data and isinstance(data, dict):
            for exec_item in data.get("executive", []):
                name  = str(exec_item.get("name") or "").lower().strip()
                title = str(exec_item.get("title") or "").lower()
                if name:
                    mapping[name] = title
        self._exec_cache[ticker] = mapping
        return mapping

    def _finnhub_insider_score(self, ticker: str) -> dict | None:
        """
        Fetch insider transactions from Finnhub /stock/insider-transactions.
        Returns a result dict or None on failure / missing key.
        """
        if not self.finnhub_key:
            return None

        data = _get_json(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": ticker, "token": self.finnhub_key},
            timeout=10,
        )
        if not data:
            return None

        transactions = [
            t for t in (data.get("data") or [])
            if t.get("transactionCode") == "P" and not t.get("isDerivative", False)
        ]
        if not transactions:
            return Form4InsiderSignal._neutral(ticker)

        # Build exec title map for role weighting
        exec_map = self._finnhub_executives(ticker)

        import math
        ceo_buy_alert = False
        alert_detail: str | None = None
        total_score = 0.0
        total_value = 0.0

        for tx in transactions:
            name   = str(tx.get("name") or "").strip()
            value  = float(tx.get("value") or 0)
            shares = float(tx.get("share") or 0)

            # Look up role from exec map (best-effort)
            title = exec_map.get(name.lower(), "")
            role_w = self._role_weight(title) if title else 0.75  # default

            amount_norm = math.tanh(value / 100_000)
            total_score += role_w * amount_norm
            total_value += value

            # CEO buy alert
            is_senior = any(k in title for k in ("ceo", "president", "chief executive"))
            if is_senior and value >= self.alert_threshold:
                ceo_buy_alert = True
                price = value / shares if shares else 0
                alert_detail = (
                    f"{name}: {title} purchased {shares:,.0f} shares "
                    f"@ ${price:.2f} (${value:,.0f}) — CEO_BUY_ALERT"
                )
                logger.warning(f"Form4InsiderSignal CEO_BUY_ALERT [{ticker}]: {alert_detail}")

        normalised = round(min(total_score / 4.0, 1.0), 4)
        result = {
            "insider_score":     normalised,
            "transaction_count": len(transactions),
            "total_value_usd":   round(total_value, 2),
            "ceo_buy_alert":     ceo_buy_alert,
            "alert_detail":      alert_detail,
            "ticker":            ticker,
            "source":            "finnhub",
        }
        logger.info(
            f"Form4InsiderSignal Finnhub [{ticker}]: score={normalised:.3f}, "
            f"txns={len(transactions)}, total=${total_value:,.0f}"
            + (" ⚠ CEO_BUY_ALERT" if ceo_buy_alert else "")
        )
        return result

    # ── Main scoring ──────────────────────────────────────────────────────────

    def score(self, ticker: str) -> dict:
        """
        Score insider buying for a single ticker.

        Returns:
            {
              "insider_score":   float [0, 1],
              "transaction_count": int,
              "total_value_usd": float,
              "ceo_buy_alert":   bool,
              "alert_detail":    str | None,
              "ticker":          str,
            }
        """
        # Cache check
        if ticker in self._cache:
            ts, cached = self._cache[ticker]
            if time.time() - ts < self._cache_ttl:
                return cached

        # ── Finnhub primary path ──────────────────────────────────────────────
        fh_result = self._finnhub_insider_score(ticker)
        if fh_result is not None:
            self._cache[ticker] = (time.time(), fh_result)
            return fh_result

        # ── EDGAR fallback ────────────────────────────────────────────────────
        hits = self._search_form4(ticker)
        if not hits:
            return self._neutral(ticker)

        all_txns: list[dict] = []
        ceo_buy_alert = False
        alert_detail: str | None = None

        for hit in hits[:20]:   # cap to limit HTTP requests
            source = hit.get("_source") or {}
            entity = source.get("entity_name", "")
            accession = (hit.get("_id") or "").replace(":", "-")
            # CIK is embedded in EDGAR filing URLs
            file_url = (hit.get("_source") or {}).get("file_num", "")
            # Extract CIK from the hit's nested path if available
            cik = ""
            for key in ("_source", "file_path"):
                val = str(hit.get(key) or "")
                if "/data/" in val:
                    parts = val.split("/data/")
                    if len(parts) > 1:
                        cik = parts[1].split("/")[0]
                        break

            if not cik or not accession:
                continue

            xml_text = self._fetch_form4_xml(accession, cik)
            if not xml_text:
                continue

            txns = self._parse_transactions(xml_text)
            for tx in txns:
                tx["entity"] = entity
            all_txns.extend(txns)

            # CEO buy alert check
            for tx in txns:
                rel = tx["relationship"].lower()
                is_senior = any(k in rel for k in ("ceo", "president", "chief executive"))
                if is_senior and tx["total_value"] >= self.alert_threshold:
                    ceo_buy_alert = True
                    alert_detail = (
                        f"{entity}: {tx['relationship']} purchased "
                        f"{tx['shares']:,.0f} shares @ ${tx['pricePerShare']:.2f} "
                        f"(${tx['total_value']:,.0f}) — CEO_BUY_ALERT"
                    )
                    logger.warning(f"Form4InsiderSignal CEO_BUY_ALERT [{ticker}]: {alert_detail}")

        if not all_txns:
            return self._neutral(ticker)

        # Score = sum of (role_weight * amount_norm) where amount_norm = tanh(value / 100k)
        import math
        total_score = sum(
            self._role_weight(tx["relationship"]) * math.tanh(tx["total_value"] / 100_000)
            for tx in all_txns
        )
        total_value = sum(tx["total_value"] for tx in all_txns)

        # Normalise: 4 director buys at $50k each ≈ score 1.0
        normalised = min(total_score / 4.0, 1.0)

        result = {
            "insider_score":     round(normalised, 4),
            "transaction_count": len(all_txns),
            "total_value_usd":   round(total_value, 2),
            "ceo_buy_alert":     ceo_buy_alert,
            "alert_detail":      alert_detail,
            "ticker":            ticker,
        }
        logger.info(
            f"Form4InsiderSignal [{ticker}]: score={normalised:.3f}, "
            f"txns={len(all_txns)}, total=${total_value:,.0f}"
            + (" ⚠ CEO_BUY_ALERT" if ceo_buy_alert else "")
        )
        self._cache[ticker] = (time.time(), result)
        return result

    def scores(self, tickers: list[str]) -> dict[str, float]:
        """Convenience: {ticker: insider_score} for a list of tickers."""
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(tickers), 4)) as pool:
            futures = {pool.submit(self.score, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()["insider_score"]
                except Exception as exc:
                    logger.warning(f"Form4InsiderSignal failed for {ticker}: {exc}")
                    results[ticker] = 0.0
        return results

    @staticmethod
    def _neutral(ticker: str) -> dict:
        return {
            "insider_score": 0.0,
            "transaction_count": 0,
            "total_value_usd": 0.0,
            "ceo_buy_alert": False,
            "alert_detail": None,
            "ticker": ticker,
        }


# ── Source 4: Options Flow ────────────────────────────────────────────────────

class _OptionsFlowSource:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stub: bool = bool(cfg.get("stub", True))
        self.api_key: str = cfg.get("unusual_whales_key", "")
        self.max_expiry_days: int = int(cfg.get("max_expiry_days", 30))
        self.min_premium: float = float(cfg.get("min_premium_usd", 10_000))

    def _mock_data(self) -> list[dict]:
        """
        Stub: returns a small hardcoded dataset that exercises the scoring logic.
        Replace with real Unusual Whales API when key is available.
        """
        return [
            {"ticker": "NVDA", "option_type": "call", "premium": 250_000, "days_to_expiry": 14, "strike_vs_spot": 1.03},
            {"ticker": "AAPL", "option_type": "call", "premium": 120_000, "days_to_expiry": 7, "strike_vs_spot": 1.01},
            {"ticker": "MSFT", "option_type": "call", "premium": 80_000, "days_to_expiry": 21, "strike_vs_spot": 1.05},
            {"ticker": "TSLA", "option_type": "put", "premium": 500_000, "days_to_expiry": 5, "strike_vs_spot": 0.95},
            {"ticker": "NVDA", "option_type": "call", "premium": 180_000, "days_to_expiry": 10, "strike_vs_spot": 1.02},
        ]

    def _fetch_sweeps(self) -> list[dict]:
        if self.stub or not self.api_key:
            return self._mock_data()

        data = _get_json(
            "https://api.unusualwhales.com/api/option-flows",
            params={"limit": 200},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return data if isinstance(data, list) else (data or {}).get("data", [])

    def scores(self) -> dict[str, float]:
        """
        Bullish signal: OTM call sweeps with expiry < max_expiry_days
        and premium >= min_premium. Score = normalised sum of premiums.
        """
        sweeps = self._fetch_sweeps()
        ticker_premium: dict[str, float] = defaultdict(float)

        for sweep in sweeps:
            ticker = (sweep.get("ticker") or "").upper()
            opt_type = (sweep.get("option_type") or "").lower()
            premium = float(sweep.get("premium", 0))
            dte = float(sweep.get("days_to_expiry", 999))
            strike_vs_spot = float(sweep.get("strike_vs_spot", 1.0))

            if (
                opt_type == "call"
                and strike_vs_spot > 1.0        # OTM call
                and dte <= self.max_expiry_days
                and premium >= self.min_premium
            ):
                ticker_premium[ticker] += premium

        if not ticker_premium:
            return {}

        max_prem = max(ticker_premium.values()) or 1.0
        return {t: round(p / max_prem, 4) for t, p in ticker_premium.items()}


# ── FlowSignal ───────────────────────────────────────────────────────────────

class FlowSignal:
    """
    Composite smart-money flow score.

    Sub-sources and weights:
      congressional_score  0.40  — Senate/House Stock Watcher APIs
      institutional_score  0.35  — SEC EDGAR 13F filings
      insider_score        0.25  — SEC EDGAR Form 4 open-market purchases

    flow_score = 0.40 * cong + 0.35 * inst + 0.25 * insider  ∈ [0, 1]
    """

    def __init__(self, cfg: dict):
        flow_cfg = cfg.get("flow", cfg)
        self.flow_cfg = flow_cfg
        cong_cfg    = flow_cfg.get("congressional", {})
        inst_cfg    = flow_cfg.get("institutional", {})
        insider_cfg = flow_cfg.get("insider", {})

        self._cong_w:    float = float(cong_cfg.get("weight",    0.40))
        self._inst_w:    float = float(inst_cfg.get("weight",    0.35))
        self._insider_w: float = float(insider_cfg.get("weight", 0.25))

        self._congressional = (
            CongressionalSignal(cong_cfg)
            if cong_cfg.get("enabled", True) else None
        )
        self._institutional = (
            _InstitutionalSource(inst_cfg)
            if inst_cfg.get("enabled", True) else None
        )
        self._insider = (
            Form4InsiderSignal(insider_cfg)
            if insider_cfg.get("enabled", True) else None
        )

    def compute(self, symbols: list[str] | None = None) -> dict[str, dict]:
        """
        Compute flow scores for all available tickers (or the supplied list).

        Returns:
            {
              ticker: {
                "flow_score":          float [0, 1],
                "congressional_score": float,
                "institutional_score": float,
                "insider_score":       float,
                "ceo_buy_alert":       bool,
              }
            }
        """
        cong_scores    = self._congressional.scores(tickers=symbols) if self._congressional else {}
        inst_scores    = self._institutional.scores() if self._institutional else {}

        # Form 4: fetch per-ticker (returns full dict with alert flags)
        insider_full: dict[str, dict] = {}
        if self._insider and symbols:
            for t in symbols:
                insider_full[t.upper()] = self._insider.score(t.upper())
        insider_scores = {t: d["insider_score"] for t, d in insider_full.items()}

        all_tickers = set(cong_scores) | set(inst_scores) | set(insider_scores)
        if symbols:
            all_tickers |= {s.upper() for s in symbols}

        results: dict[str, dict] = {}
        for ticker in all_tickers:
            c = cong_scores.get(ticker, 0.0)
            i = inst_scores.get(ticker, 0.0)
            ins = insider_scores.get(ticker, 0.0)
            composite = self._cong_w * c + self._inst_w * i + self._insider_w * ins
            ceo_alert = insider_full.get(ticker, {}).get("ceo_buy_alert", False)
            results[ticker] = {
                "flow_score":          round(composite, 4),
                "congressional_score": c,
                "institutional_score": i,
                "insider_score":       ins,
                "ceo_buy_alert":       ceo_alert,
            }
            logger.debug(f"FlowSignal [{ticker}]: {results[ticker]}")

        return results

    def score_only(self, symbols: list[str]) -> dict[str, float]:
        """Convenience: {ticker: flow_score}"""
        return {t: v["flow_score"] for t, v in self.compute(symbols).items()}
