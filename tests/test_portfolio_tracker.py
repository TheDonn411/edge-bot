from __future__ import annotations

import pandas as pd

from portfolio import PortfolioTracker


def test_portfolio_tracker_records_buy_sell_and_marks_equity():
    tracker = PortfolioTracker(
        10_000,
        {
            "enabled": False,
            "auto_save": False,
            "trades_file": "logs/test_trades.csv",
            "snapshots_file": "logs/test_snapshots.csv",
        },
        mode="test",
    )

    tracker.record_fill(pd.Timestamp("2024-01-02"), "AAPL", "BUY", 10, 100.0, commission=1.0)
    snap1 = tracker.mark_to_market(pd.Timestamp("2024-01-02"), {"AAPL": 100.0})
    assert snap1["equity"] == 9999.0

    tracker.record_fill(pd.Timestamp("2024-01-03"), "AAPL", "SELL", 5, 110.0, commission=1.0)
    snap2 = tracker.mark_to_market(pd.Timestamp("2024-01-03"), {"AAPL": 110.0})

    assert snap2["realized_pnl"] == 49.0
    assert snap2["unrealized_pnl"] == 50.0
    assert snap2["equity"] == 10098.0
    assert len(tracker.trades_df()) == 2
