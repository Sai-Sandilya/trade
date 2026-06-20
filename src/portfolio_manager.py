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
    color:                   str   = "#9C27B0"
    description:             str   = ""


PRESETS: dict[str, PortfolioPreset] = {
    "Conservative": PortfolioPreset(
        name                 = "Conservative",
        monthly_budget_usd   = 50.0,
        oversold_rsi         = 30.0,
        below_sma_multiplier = 1.2,
        oversold_multiplier  = 1.5,
        enable_exits         = False,
        color                = "#2196F3",
        description          = (
            "$50/month · Only buys extra at RSI < 30 (deep crashes only) · "
            "1.2x when below SMA200 · No automatic sells · Lowest risk"
        ),
    ),
    "Balanced": PortfolioPreset(
        name                 = "Balanced",
        monthly_budget_usd   = 100.0,
        oversold_rsi         = 35.0,
        below_sma_multiplier = 1.5,
        oversold_multiplier  = 2.0,
        enable_exits         = False,
        color                = "#4CAF50",
        description          = (
            "$100/month · Buys extra at RSI < 35 · "
            "1.5x when below SMA200, 2x on RSI crash · No automatic sells · Default strategy"
        ),
    ),
    "Aggressive": PortfolioPreset(
        name                 = "Aggressive",
        monthly_budget_usd   = 200.0,
        oversold_rsi         = 40.0,
        below_sma_multiplier = 2.0,
        oversold_multiplier  = 3.0,
        enable_exits         = True,
        take_profit_pct      = 0.40,
        stop_loss_pct        = 0.15,
        enable_rebalance     = True,
        rebalance_threshold_pct = 0.10,
        color                = "#FF5722",
        description          = (
            "$200/month · Buys extra at RSI < 40 (triggers often) · "
            "2x below SMA200, 3x on RSI crash · Take-profit at +40%, stop-loss at -15% · Highest risk"
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
            "CAGR":         f"{m.get('cagr',         float('nan')) * 100:.1f}%" if m else "—",
            "Sharpe":       f"{m.get('sharpe',       float('nan')):.2f}"        if m else "—",
            "Max Drawdown": f"{m.get('max_drawdown', float('nan')) * 100:.1f}%" if m else "—",
        })

    return pd.DataFrame(rows).set_index("Portfolio")
