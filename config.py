"""
config.py — loads and validates config.yaml.

Usage:
    from config import load_config
    cfg = load_config()          # reads config.yaml from project root
    cfg = load_config("my.yaml") # override path
"""

from __future__ import annotations
import os
from pathlib import Path
import yaml


_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve env-var overrides (e.g. EDGE_BOT__EXECUTION__BROKER)
    cfg = _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict, prefix: str = "EDGE_BOT") -> dict:
    """
    Override any config leaf with an env var.
    e.g. EDGE_BOT__RISK__MAX_DRAWDOWN_PCT=0.20 overrides risk.max_drawdown_pct
    """
    for key, value in os.environ.items():
        if not key.startswith(prefix + "__"):
            continue
        parts = key[len(prefix) + 2:].lower().split("__")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Coerce type from existing value
        existing = node.get(parts[-1])
        try:
            if isinstance(existing, bool):
                node[parts[-1]] = value.lower() in ("1", "true", "yes")
            elif isinstance(existing, int):
                node[parts[-1]] = int(value)
            elif isinstance(existing, float):
                node[parts[-1]] = float(value)
            else:
                node[parts[-1]] = value
        except (ValueError, TypeError):
            node[parts[-1]] = value

    return cfg
