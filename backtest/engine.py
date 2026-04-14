"""
BacktestEngine — event-driven daily backtest harness using vectorbt.

Usage:
    from backtest import BacktestEngine
    engine = BacktestEngine(cfg)
    results = engine.run(symbols=["AAPL", "MSFT", "GOOGL"])
    results.print_stats()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from data import DataLoader
from picker import StockPicker
from risk import PositionSizer, KillSwitch
from portfolio import PortfolioTracker


class BacktestEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bt_cfg = cfg["backtest"]
        self.loader = DataLoader(cfg)
        self.picker = StockPicker(cfg)
        self.sizer = PositionSizer(cfg)
        self.kill = KillSwitch(cfg)

    def run(self, symbols: list[str]) -> "BacktestResult":
        start = self.bt_cfg["start"]
        end = self.bt_cfg["end"]
        initial_capital = self.bt_cfg["initial_capital"]

        logger.info(f"Backtest: {start} → {end} | symbols={symbols}")

        # Fetch data for all symbols + benchmark
        benchmark = self.bt_cfg.get("benchmark", "SPY")
        all_symbols = list(set(symbols + [benchmark]))
        price_data = self.loader.fetch_many(all_symbols, start=start, end=end, use_cache=False)

        # Build a shared date index
        closes = {
            sym: df["Close"]
            for sym, df in price_data.items()
            if not df.empty
        }
        if not closes:
            logger.error("No price data available for backtest.")
            return BacktestResult({})

        idx = sorted(set.intersection(*[set(s.index) for s in closes.values()]))
        if len(idx) < 2:
            logger.error("Insufficient overlapping dates.")
            return BacktestResult({})

        equity = initial_capital
        peak_equity = equity
        daily_equity: list[float] = []
        tracker = PortfolioTracker(
            initial_capital,
            cfg.get("portfolio", {}),
            mode="backtest",
        )
        portfolio: dict[str, dict] = {}  # {symbol: {shares, entry_price}}

        for i, date in enumerate(idx):
            # Compute daily P&L from open positions
            day_pnl = 0.0
            for sym, pos in list(portfolio.items()):
                price = closes[sym].get(date, None)
                if price is None:
                    continue
                day_pnl += (price - pos["entry_price"]) * pos["shares"]

                # Stop-loss check
                if not self.kill.check_position(sym, pos["entry_price"], float(price)):
                    tracker.record_fill(date, sym, "SELL", pos["shares"], float(price))
                    equity = tracker.summary()["cash"]
                    del portfolio[sym]
                    logger.debug(f"Stop-loss closed {sym} on {date}")

            # Portfolio-level kill switch
            peak_equity = max(peak_equity, equity)
            day_pnl_pct = day_pnl / equity if equity > 0 else 0.0
            if not self.kill.check_portfolio(peak_equity, equity, day_pnl_pct):
                logger.warning(f"Kill switch halted trading at {date}")
                break

            # Need at least lookback days of history to generate signals
            lookback = self.cfg["data"]["lookback_days"]
            if i < lookback:
                mtm_prices = {
                    sym: float(closes[sym].get(date, pos["entry_price"]))
                    for sym, pos in portfolio.items()
                    if sym in closes
                }
                snapshot = tracker.mark_to_market(date, mtm_prices)
                daily_equity.append(snapshot["equity"])
                equity = snapshot["cash"]
                self.kill.reset() if i == 0 else None
                continue

            # Slice history up to current date for signal computation
            history = {
                sym: price_data[sym].loc[:date]
                for sym in symbols
                if sym in price_data and not price_data[sym].empty
            }

            picks = self.picker.pick(symbols, price_data=history, use_external_data=False)
            prices_today = {sym: float(closes[sym].get(date, 0)) for sym in picks}
            score_df = self.picker.score_live(symbols, price_data=history, use_external_data=False)
            scores = dict(zip(score_df["symbol"], score_df["composite_score"]))

            # Close positions no longer in picks
            for sym in list(portfolio.keys()):
                if sym not in picks and sym in closes:
                    price = float(closes[sym].get(date, portfolio[sym]["entry_price"]))
                    tracker.record_fill(date, sym, "SELL", portfolio[sym]["shares"], price)
                    equity = tracker.summary()["cash"]
                    del portfolio[sym]

            # Open new positions
            sizing = self.sizer.size_all(picks, scores, prices_today, equity)
            for order in sizing:
                sym = order["symbol"]
                if sym in portfolio or order["shares"] <= 0:
                    continue
                cost = order["shares"] * prices_today.get(sym, 0)
                if cost > equity * 0.01:  # sanity check: at least 1%
                    tracker.record_fill(date, sym, "BUY", order["shares"], prices_today.get(sym, 0))
                    equity = tracker.summary()["cash"]
                    portfolio[sym] = {"shares": order["shares"], "entry_price": prices_today.get(sym, 0)}

            # Mark portfolio to market
            mtm_prices = {
                sym: float(closes[sym].get(date, pos["entry_price"]))
                for sym, pos in portfolio.items()
                if sym in closes
            }
            snapshot = tracker.mark_to_market(date, mtm_prices)
            daily_equity.append(snapshot["equity"])
            equity = snapshot["cash"]
            self.kill.reset()  # reset daily halt at end of day

        equity_series = pd.Series(daily_equity, index=idx[: len(daily_equity)])
        bm_series = closes.get(benchmark, pd.Series(dtype=float)).reindex(equity_series.index)

        logger.info(f"Backtest complete. Final equity: {equity_series.iloc[-1]:,.0f}")
        return BacktestResult(
            {
                "equity": equity_series,
                "benchmark": bm_series,
                "initial_capital": initial_capital,
                "cfg": self.bt_cfg,
                "tracker": tracker,
            }
        )


class BacktestResult:
    def __init__(self, data: dict):
        self.data = data

    def print_stats(self):
        if not self.data or "equity" not in self.data:
            print("No backtest data available.")
            return

        eq = self.data["equity"]
        initial = self.data["initial_capital"]
        final = eq.iloc[-1]
        n_days = len(eq)
        n_years = n_days / 252

        returns = eq.pct_change().dropna()
        cagr = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        max_dd = drawdown.min()

        print("\n=== Backtest Results ===")
        print(f"  Period       : {eq.index[0].date()} → {eq.index[-1].date()}")
        print(f"  Initial cap  : ${initial:,.0f}")
        print(f"  Final equity : ${final:,.0f}")
        print(f"  CAGR         : {cagr:.2%}")
        print(f"  Sharpe       : {sharpe:.2f}")
        print(f"  Max drawdown : {max_dd:.2%}")
        print("========================\n")
