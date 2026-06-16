"""
config_loader.py - Load config.yaml and expose typed defaults.

Usage:
    from config_loader import load_config, get_bot_config
    cfg = load_config()
    bot_cfg = get_bot_config(cfg)
"""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """
    Load config.yaml and return the parsed dict.
    Falls back to an empty dict if the file is missing so callers
    can always use .get() with a default safely.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Drill into nested dict keys, returning default if any key is missing."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is None:
            return default
    return node


def get_bot_config(cfg: dict):
    """
    Build a BotConfig dataclass from the loaded YAML dict.
    Imports BotConfig here to avoid circular imports.
    """
    from bot import BotConfig

    s  = cfg.get("strategy", {})
    bt = cfg.get("backtest", {})

    tickers = bt.get("tickers") or ["SPY", "QQQ", "AMD"]
    start   = bt.get("start_date")
    end     = bt.get("end_date")

    return BotConfig(
        tickers              = tickers,
        monthly_budget_usd   = float(s.get("monthly_budget_usd",   50.0)),
        oversold_rsi         = float(s.get("oversold_rsi",          35.0)),
        rsi_period           = int(  s.get("rsi_period",            21)),
        sma_period           = int(  s.get("sma_period",            200)),
        below_sma_multiplier = float(s.get("below_sma_multiplier",  1.5)),
        oversold_multiplier  = float(s.get("oversold_multiplier",   2.0)),
        slippage_bps         = float(s.get("slippage_bps",          3.0)),
        clearing_fee_usd     = float(s.get("clearing_fee_usd",      0.005)),
        min_hold_days        = int(  s.get("min_hold_days",         15)),
        decimal_places       = int(  s.get("decimal_places",        6)),
        start_date           = str(start) if start else None,
        end_date             = str(end)   if end   else None,
    )


def get_risk_free_rate(cfg: dict) -> float:
    return float(_get(cfg, "metrics", "risk_free_rate", default=0.04))
