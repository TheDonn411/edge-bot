"""
RiskManager — unified position sizing, hard-rule enforcement, and kill switch.

Composes with the existing KillSwitch and PositionSizer but adds:
  - Hard rule validation before every order
  - ATR-based stop-loss calculation
  - Per-asset-class leverage caps
  - Email alert on kill-switch trigger (optional, via smtplib)

Hard rules (in order of precedence):
  1. Max single-position weight ≤ max_portfolio_pct (10%)
  2. Max open positions ≤ max_open_positions (5)
  3. Max daily loss ≥ max_daily_loss_pct (5%) → kill switch
  4. Leverage ≤ max_leverage_stocks/cfds

Kelly sizing:
  position_size = (edge / odds) * capital * kelly_fraction
  where edge ≈ signal score mapped to win probability
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

import numpy as np
from loguru import logger

from .kill_switch import KillSwitch
from .position_sizer import PositionSizer


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.risk_cfg = cfg["risk"]
        self.alert_cfg = self.risk_cfg.get("alert", {})

        # Compose existing components
        self.kill_switch = KillSwitch(cfg)
        self.sizer = PositionSizer(cfg)

        # Hard-rule thresholds
        self.max_portfolio_pct: float  = float(self.risk_cfg.get("max_portfolio_pct", 0.10))
        self.max_open_positions: int   = int(self.risk_cfg.get("max_open_positions", 5))
        self.max_daily_loss_pct: float = float(self.risk_cfg.get("max_daily_loss_pct", 0.05))
        self.max_lev_stocks: float     = float(self.risk_cfg.get("max_leverage_stocks", 2.0))
        self.max_lev_cfds: float       = float(self.risk_cfg.get("max_leverage_cfds", 5.0))
        self.atr_multiplier: float     = float(self.risk_cfg.get("atr_stop_multiplier", 1.5))
        self.kelly_fraction: float     = float(self.risk_cfg.get("kelly_fraction", 0.25))

    # ── Stop-loss ─────────────────────────────────────────────────────────────

    def compute_stop_loss(self, entry_price: float, atr: float) -> float:
        """
        stop_loss = entry_price - (ATR_14 * atr_stop_multiplier)
        Returns the absolute stop-loss price.
        """
        stop = entry_price - atr * self.atr_multiplier
        return round(max(stop, entry_price * 0.50), 4)  # floor at 50% of entry (circuit breaker)

    # ── Kelly position sizing ─────────────────────────────────────────────────

    def kelly_size(
        self,
        symbol: str,
        score: float,
        price: float,
        equity: float,
        atr: float = 0.0,
    ) -> dict:
        """
        Full Kelly position with fractional multiplier and hard caps.

        score: normalised signal score ∈ [0, 1]
          - Mapped to edge: score > 0.5 → positive edge
          - odds = assumed 1:1 (can be refined with historical win/loss ratio)

        Returns:
            {
              "symbol":      str,
              "shares":      int,
              "dollar_value":float,
              "weight":      float,   # fraction of equity
              "stop_loss":   float,   # absolute stop price
            }
        """
        edge = (score - 0.5) * 2          # [-1, 1]; positive = bullish edge
        odds = 1.0                          # win payoff per unit risked
        kelly_raw = edge / odds if odds > 0 else 0.0
        kelly_frac = kelly_raw * self.kelly_fraction  # fractional Kelly
        weight = float(np.clip(kelly_frac, 0.0, self.max_portfolio_pct))

        dollar_value = equity * weight
        shares = int(dollar_value / price) if price > 0 else 0
        stop = self.compute_stop_loss(price, atr) if atr > 0 else price * (1 - self.risk_cfg.get("stop_loss_pct", 0.07))

        result = {
            "symbol": symbol,
            "shares": shares,
            "dollar_value": round(dollar_value, 2),
            "weight": round(weight, 4),
            "stop_loss": round(stop, 4),
        }
        logger.debug(f"RiskManager.kelly_size [{symbol}]: {result}")
        return result

    def size_all(
        self,
        picks: list[str],
        scores: dict[str, float],
        prices: dict[str, float],
        equity: float,
        atrs: dict[str, float] | None = None,
    ) -> list[dict]:
        """Size positions for all picks. Returns list of order dicts."""
        atrs = atrs or {}
        return [
            self.kelly_size(
                sym,
                scores.get(sym, 0.5),
                prices.get(sym, 1.0),
                equity,
                atrs.get(sym, 0.0),
            )
            for sym in picks
        ]

    # ── Hard-rule validation ──────────────────────────────────────────────────

    def validate_order(
        self,
        symbol: str,
        shares: int,
        price: float,
        side: str,
        account: dict,
        asset_class: str = "stock",
    ) -> tuple[bool, str]:
        """
        Validate an order against all hard rules.

        account: {"equity": float, "cash": float, "positions": {sym: {...}}}
        Returns (approved: bool, reason: str).
        """
        if self.kill_switch.is_halted:
            return False, "KillSwitch is active — trading halted"

        equity = float(account.get("equity", 0))
        positions = account.get("positions", {})
        open_count = len(positions)
        order_value = shares * price

        if side.upper() == "BUY":
            # Rule 1: max single position
            proposed_pct = order_value / equity if equity > 0 else 1.0
            if proposed_pct > self.max_portfolio_pct:
                return False, (
                    f"Position size {proposed_pct:.1%} exceeds max "
                    f"{self.max_portfolio_pct:.0%} for {symbol}"
                )

            # Rule 2: max open positions
            if symbol not in positions and open_count >= self.max_open_positions:
                return False, (
                    f"Max open positions reached ({open_count}/{self.max_open_positions})"
                )

            # Rule 4: leverage
            max_lev = self.max_lev_cfds if asset_class == "cfd" else self.max_lev_stocks
            cash = float(account.get("cash", 0))
            current_leverage = (equity - cash) / equity if equity > 0 else 0
            new_leverage = (equity - cash + order_value) / equity if equity > 0 else 0
            if new_leverage > max_lev:
                return False, (
                    f"Order would breach leverage limit "
                    f"({new_leverage:.1f}x > {max_lev:.1f}x)"
                )

        return True, "OK"

    # ── Kill-switch monitoring ────────────────────────────────────────────────

    def check_and_act(
        self,
        peak_equity: float,
        current_equity: float,
        daily_pnl_pct: float,
        positions: dict,
        prices: dict[str, float],
    ) -> dict:
        """
        Run all kill-switch checks; return a dict of required actions.

        Returns:
            {
              "halt":          bool,   # stop accepting new orders
              "close_all":     bool,   # liquidate all positions
              "close_symbols": list,   # specific positions to stop-out
            }
        """
        actions = {"halt": False, "close_all": False, "close_symbols": []}

        # Portfolio-level check (updates KillSwitch internal state)
        if not self.kill_switch.check_portfolio(peak_equity, current_equity, daily_pnl_pct):
            actions["halt"] = True
            actions["close_all"] = True
            self._send_alert(
                subject="edge-bot KILL SWITCH triggered",
                body=(
                    f"Kill switch activated.\n"
                    f"Peak equity: ${peak_equity:,.2f}\n"
                    f"Current equity: ${current_equity:,.2f}\n"
                    f"Daily P&L: {daily_pnl_pct:.2%}\n"
                ),
            )
            logger.critical("RiskManager: kill switch triggered — ALL POSITIONS CLOSED")
            return actions

        # Daily loss check against the tighter threshold
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            actions["halt"] = True
            actions["close_all"] = True
            self._send_alert(
                subject="edge-bot daily loss limit hit",
                body=f"Daily P&L {daily_pnl_pct:.2%} breached limit {-self.max_daily_loss_pct:.2%}.",
            )
            return actions

        # Per-position stop-loss checks
        for symbol, pos in positions.items():
            entry = float(pos.get("avg_cost", pos.get("entry_price", 0)))
            current = prices.get(symbol, entry)
            if not self.kill_switch.check_position(symbol, entry, current):
                actions["close_symbols"].append(symbol)

        return actions

    # ── Email alerting ────────────────────────────────────────────────────────

    def _send_alert(self, subject: str, body: str) -> None:
        logger.warning(f"ALERT: {subject}")
        if not self.alert_cfg.get("email_enabled", False):
            return

        smtp_host = self.alert_cfg.get("smtp_host", "smtp.gmail.com")
        smtp_port = int(self.alert_cfg.get("smtp_port", 587))
        smtp_user = self.alert_cfg.get("smtp_user", "")
        smtp_pass = self.alert_cfg.get("smtp_password", "")
        recipient = self.alert_cfg.get("alert_recipient", smtp_user)

        if not smtp_user or not smtp_pass:
            logger.warning("RiskManager: email alert skipped (smtp credentials not configured)")
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = recipient
            msg.set_content(body)

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            logger.info(f"RiskManager: alert email sent to {recipient}")
        except Exception as exc:
            logger.error(f"RiskManager: failed to send alert email — {exc}")

    def require_manual_reset(self) -> bool:
        """
        Kill switch requires a manual call to reset() before trading resumes.
        Returns True if currently halted.
        """
        return self.kill_switch.is_halted

    def manual_reset(self) -> None:
        """Operator must call this explicitly to re-enable trading after a halt."""
        logger.info("RiskManager: manual reset — trading re-enabled")
        self.kill_switch.reset()
