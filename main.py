"""
main.py — edge-bot orchestrator.

Modes:
  python main.py live      → runs the live/paper-trading scheduler loop
  python main.py backtest  → runs a backtest and prints stats
  python main.py score     → scores today's watchlist and prints rankings (dry run)

All configuration is read from config.yaml.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import schedule
import time

_STATE_FILE = Path("logs/bot_state.json")
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)  # created once at import
from loguru import logger

from config import load_config
from data import DataLoader
from picker import StockPicker
from risk import PositionSizer, KillSwitch
from execution import IBKRStub

def _load_state(equity: float) -> dict:
    """Load persisted peak equity and day-start equity. Resets day_start on a new calendar day."""
    today = date.today().isoformat()
    try:
        state = json.loads(_STATE_FILE.read_text())
        if state.get("day_date") != today:
            state["day_start"] = equity
            state["day_date"] = today
        return state
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"Could not load bot state, resetting to defaults: {exc}")
    return {"peak": equity, "day_start": equity, "day_date": today}


def _save_state(state: dict) -> None:
    _STATE_FILE.write_text(json.dumps(state))


def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_cfg.get("level", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    log_dir = log_cfg.get("log_dir", "logs/")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}edge_bot.log",
        level=log_cfg.get("level", "INFO"),
        rotation=log_cfg.get("rotation", "10 MB"),
    )


def run_signal_cycle(
    cfg: dict,
    loader: DataLoader,
    picker: StockPicker,
    sizer: PositionSizer,
    kill: KillSwitch,
    broker: IBKRStub,
):
    """Single signal → pick → size → execute cycle."""
    logger.info(f"--- Signal cycle at {datetime.now().strftime('%H:%M:%S')} ---")

    account = broker.get_account()
    equity = account["equity"]

    state = _load_state(equity)
    state["peak"] = max(state["peak"], equity)
    _save_state(state)
    peak_equity = state["peak"]
    daily_pnl = (equity - state["day_start"]) / state["day_start"] if state["day_start"] > 0 else 0.0

    if not kill.check_portfolio(peak_equity, equity, daily_pnl=daily_pnl):
        logger.warning("Kill switch active — skipping cycle.")
        return

    # Fetch latest data for watchlist
    watchlist = cfg["universe"].get("watchlist", [])
    if not watchlist:
        logger.warning("Watchlist is empty. Add symbols to config.yaml → universe.watchlist")
        return

    price_data = loader.fetch_many(watchlist, use_cache=False)
    prices = {sym: float(df["Close"].iloc[-1]) for sym, df in price_data.items() if not df.empty}

    # Update broker with latest prices
    broker.update_prices(prices)

    # Pick top candidates; score_df is cached on picker._last_score_df by pick()
    picks = picker.pick(watchlist, price_data=price_data)
    score_df = picker._last_score_df
    scores = dict(zip(score_df["symbol"], score_df["composite_score"]))

    # Close positions that fell out of picks
    current_positions = broker.get_positions()
    for sym in list(current_positions.keys()):
        if sym not in picks:
            broker.place_order(sym, current_positions[sym]["shares"], "SELL", limit_price=prices.get(sym))

    # Open new long positions
    sizing = sizer.size_all(picks, scores, prices, equity)
    for order in sizing:
        sym = order["symbol"]
        if sym not in current_positions and order["shares"] > 0:
            broker.place_order(sym, order["shares"], "BUY", limit_price=prices.get(sym))

    logger.info(f"Cycle complete. Equity: ${equity:,.2f}")
    if hasattr(broker, "tracker"):
        summary = broker.tracker.summary()
        logger.info(
            "Portfolio P&L | equity=${:,.2f} cash=${:,.2f} total_pnl=${:,.2f}".format(
                summary["equity"],
                summary["cash"],
                summary["total_pnl"],
            )
        )


def mode_live(cfg: dict):
    setup_logging(cfg)
    loader = DataLoader(cfg)
    picker = StockPicker(cfg)
    sizer = PositionSizer(cfg)
    kill = KillSwitch(cfg)
    broker = IBKRStub(cfg)
    broker.connect()

    sched_cfg = cfg["scheduler"]
    open_offset = sched_cfg.get("market_open_offset_min", 5)
    close_offset = sched_cfg.get("market_close_offset_min", 15)
    market_tz = sched_cfg.get("timezone", "America/New_York")

    def _offset_time(base_hour: int, base_minute: int, offset_minutes: int) -> str:
        base = datetime(2000, 1, 1, base_hour, base_minute)
        shifted = base + timedelta(minutes=offset_minutes)
        return shifted.strftime("%H:%M")

    # NY market hours: 09:30 open, 16:00 close
    open_time = _offset_time(9, 30, open_offset)
    close_time = _offset_time(16, 0, -close_offset)

    logger.info(f"Scheduling signal cycles at {open_time} and {close_time} {market_tz}")
    schedule.every().day.at(open_time, market_tz).do(
        run_signal_cycle, cfg, loader, picker, sizer, kill, broker
    )
    schedule.every().day.at(close_time, market_tz).do(
        run_signal_cycle, cfg, loader, picker, sizer, kill, broker
    )
    schedule.every().day.at("09:31", market_tz).do(kill.reset)  # daily reset

    logger.info("edge-bot live loop started. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(10)


def mode_backtest(cfg: dict):
    setup_logging(cfg)
    from backtest import BacktestEngine
    symbols = cfg["universe"].get("watchlist", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    engine = BacktestEngine(cfg)
    result = engine.run(symbols)
    result.print_stats()


def mode_score(cfg: dict):
    setup_logging(cfg)
    loader = DataLoader(cfg)
    picker = StockPicker(cfg)

    watchlist = cfg["universe"].get("watchlist", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    logger.info(f"Scoring {len(watchlist)} symbols...")
    price_data = loader.fetch_many(watchlist)
    score_df = picker.score_live(watchlist, price_data=price_data)

    print("\n=== Signal Scores (today) ===")
    print(score_df.to_string(index=False))
    print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    cfg = load_config()

    if mode == "live":
        mode_live(cfg)
    elif mode == "backtest":
        mode_backtest(cfg)
    elif mode == "score":
        mode_score(cfg)
    else:
        print(f"Unknown mode: {mode}. Use: live | backtest | score")
        sys.exit(1)
