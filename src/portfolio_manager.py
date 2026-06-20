"""
portfolio_manager.py - Multiple portfolio presets and side-by-side comparison.

Defines three preset portfolio strategies and a runner that backtests all of
them so the dashboard can show an apples-to-apples comparison.

Presets:
  Conservative  — small budget, mild multipliers, no exits
  Balanced      — medium budget, standard multipliers, no exits  (matches config.yaml defaults)
  Aggressive    — large budget, high multipliers, take-profit exits enabled
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

@dataclass
class PortfolioPreset:
    name:                    str
    monthly_budget_usd:      float
    oversold_rsi:            float
    below_sma_multiplier:    float
    oversold_multiplier:     float
    slippage_bps:            float = 3.0
    clearing_fee_usd:        float = 0.005
    enable_exits:            bool  = False
    take_profit_pct:         float = 0.50
    stop_loss_pct:           float = 0.20
    enable_rebalance:        bool  = False
    rebalance_threshold_pct: float = 0.10
    # Six-strategy feature flags — each preset opts in explicitly
    enable_regime_filter:    bool  = False
    enable_kelly_sizing:     bool  = False
    enable_trailing_stop:    bool  = False
    enable_sector_rotation:  bool  = False
    trailing_atr_multiple:   float = 2.5
    min_hold_days:           int   = 15
    color:                   str   = "#9C27B0"
    description:             str   = ""


PRESETS: dict[str, PortfolioPreset] = {
    # All presets use the same $100/month budget so the equity curves compare
    # strategy quality, not how much cash was deposited.
    "Conservative": PortfolioPreset(
        name                 = "Conservative",
        monthly_budget_usd   = 100.0,
        oversold_rsi         = 30.0,
        below_sma_multiplier = 1.2,
        oversold_multiplier  = 1.5,
        enable_exits         = False,
        enable_regime_filter = True,    # skip buys in bear market
        enable_kelly_sizing  = False,
        enable_trailing_stop = False,
        enable_sector_rotation = False,
        color                = "#2196F3",
        description          = (
            "$100/month · RSI < 30 (deep crashes only) · 1.2x below SMA200 · "
            "Regime filter on · No trailing stop · Lowest risk"
        ),
    ),
    "Balanced": PortfolioPreset(
        name                 = "Balanced",
        monthly_budget_usd   = 100.0,
        oversold_rsi         = 35.0,
        below_sma_multiplier = 1.5,
        oversold_multiplier  = 2.0,
        enable_exits         = False,
        enable_regime_filter = True,
        enable_kelly_sizing  = True,    # Kelly position sizing
        enable_trailing_stop = False,
        enable_sector_rotation = False,
        color                = "#4CAF50",
        description          = (
            "$100/month · RSI < 35 · 1.5x below SMA200, 2x on RSI crash · "
            "Regime filter + Kelly sizing · Default strategy"
        ),
    ),
    "Aggressive": PortfolioPreset(
        name                 = "Aggressive",
        monthly_budget_usd   = 100.0,
        oversold_rsi         = 40.0,
        below_sma_multiplier = 2.0,
        oversold_multiplier  = 3.0,
        enable_exits         = True,
        take_profit_pct      = 0.40,
        stop_loss_pct        = 0.15,
        enable_rebalance     = True,
        rebalance_threshold_pct = 0.10,
        enable_regime_filter = True,
        enable_kelly_sizing  = True,
        enable_trailing_stop = True,    # ATR trailing stop
        trailing_atr_multiple = 2.5,
        enable_sector_rotation = True,  # rotate to XLU in bear regime
        min_hold_days        = 15,
        color                = "#FF5722",
        description          = (
            "$100/month · RSI < 40 · 2x below SMA200, 3x on RSI crash · "
            "Kelly + trailing stop + sector rotation · Highest risk"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    preset_names: list[str],
    tickers: list[str],
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict[str, dict]:
    """
    Run a backtest for each named preset and return results keyed by preset name.

    Returns:
        {
          "Conservative": {"trade_log": df, "summary": df, "equity": df, "preset": PortfolioPreset},
          "Balanced":     {...},
          "Aggressive":   {...},
        }
    """
    from bot import LongTermDCABot, BotConfig
    from pipeline import clean_all
    from ingestion import ingest_all

    # Download raw data first — comparison can be triggered independently
    # of the main backtest button, so we can't assume data already exists.
    ingest_all(tickers)
    clean_all(tickers)
    results = {}

    for name in preset_names:
        if name not in PRESETS:
            logger.warning("Unknown preset '%s' — skipping", name)
            continue

        p = PRESETS[name]
        cfg = BotConfig(
            tickers                 = tickers,
            monthly_budget_usd      = p.monthly_budget_usd,
            oversold_rsi            = p.oversold_rsi,
            below_sma_multiplier    = p.below_sma_multiplier,
            oversold_multiplier     = p.oversold_multiplier,
            slippage_bps            = p.slippage_bps,
            clearing_fee_usd        = p.clearing_fee_usd,
            enable_exits            = p.enable_exits,
            take_profit_pct         = p.take_profit_pct,
            stop_loss_pct           = p.stop_loss_pct,
            enable_rebalance        = p.enable_rebalance,
            rebalance_threshold_pct = p.rebalance_threshold_pct,
            enable_regime_filter    = p.enable_regime_filter,
            enable_kelly_sizing     = p.enable_kelly_sizing,
            enable_trailing_stop    = p.enable_trailing_stop,
            trailing_atr_multiple   = p.trailing_atr_multiple,
            enable_sector_rotation  = p.enable_sector_rotation,
            min_hold_days           = p.min_hold_days,
            start_date              = start_date,
            end_date                = end_date,
        )

        try:
            bot       = LongTermDCABot(cfg)
            trade_log = bot.run()
            summary   = bot.summary(trade_log)
            equity    = bot.equity_curve(trade_log)
            results[name] = {
                "trade_log": trade_log,
                "summary":   summary,
                "equity":    equity,
                "preset":    p,
                "cfg":       cfg,
            }
            logger.info("Comparison: %s — %d trades", name, len(trade_log))
        except Exception as exc:
            logger.error("Comparison failed for %s: %s", name, exc)
            results[name] = {"error": str(exc), "preset": p}

    return results


def comparison_equity_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Merge equity curves from all presets into a single DataFrame for charting.
    Columns: date, Conservative_usd, Balanced_usd, Aggressive_usd (whichever ran).
    """
    frames = {}
    for name, r in results.items():
        if "error" in r or "equity" not in r:
            continue
        eq = r["equity"].copy()
        eq.index = pd.to_datetime(eq.index, utc=True)
        if "total_portfolio_usd" in eq.columns:
            frames[name] = eq["total_portfolio_usd"].rename(f"{name}_usd")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1).sort_index()
    combined.index.name = "date"
    return combined


def _fmt_pct(val) -> str:
    """Format a decimal metric as a percentage string, or '—' if nan/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val * 100:.1f}%"


def _fmt_ratio(val) -> str:
    """Format a ratio metric as a 2-decimal string, or '—' if nan/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.2f}"


def comparison_metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a side-by-side metrics summary table for the comparison tab.
    """
    from metrics import compute_all

    rows = []
    for name, r in results.items():
        if "error" in r or "equity" not in r:
            rows.append({"Portfolio": name, "Error": r.get("error", "failed")})
            continue

        eq       = r["equity"]
        eq.index = pd.to_datetime(eq.index, utc=True)
        total_eq = eq.get("total_portfolio_usd") if "total_portfolio_usd" in eq.columns else None

        tlog = r.get("trade_log")
        m    = compute_all(total_eq, trade_log=tlog) if total_eq is not None else {}

        summary          = r["summary"]
        total_invested   = float(summary["total_invested_usd"].sum())
        total_value      = float(summary["market_value_usd"].sum())
        total_pnl        = total_value - total_invested
        total_pnl_pct    = total_pnl / total_invested * 100 if total_invested else 0

        p = r["preset"]
        rows.append({
            "Portfolio":       name,
            "Budget/mo":       f"${p.monthly_budget_usd:.0f}",
            "Total Invested":  f"${total_invested:,.0f}",
            "Portfolio Value": f"${total_value:,.0f}",
            "Total P&L":       f"${total_pnl:,.0f} ({total_pnl_pct:+.1f}%)",
            "CAGR":            _fmt_pct(m.get("cagr")),
            "Sharpe":          _fmt_ratio(m.get("sharpe")),
            "Max Drawdown":    _fmt_pct(m.get("max_drawdown")),
        })

    return pd.DataFrame(rows).set_index("Portfolio")
