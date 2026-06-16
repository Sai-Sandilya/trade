"""
performance_tracker.py - Persists portfolio snapshots after each backtest run.

Each time a backtest is run, a snapshot of the results is appended to
data/performance_history.parquet so you can track how the strategy would
have performed if you had run it at different points in time.

Also provides monthly breakdown helpers for the dashboard.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger    = logging.getLogger(__name__)
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
HIST_PATH = DATA_DIR / "performance_history.parquet"


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_snapshot(
    summary: pd.DataFrame,
    metrics: dict,
    equity: pd.DataFrame,
    cfg_label: str = "default",
) -> None:
    """
    Append a portfolio snapshot to the history file.

    summary   : bot.summary() DataFrame (one row per ticker)
    metrics   : compute_all() dict
    equity    : bot.equity_curve() DataFrame (used for monthly breakdown)
    cfg_label : human-readable label for this configuration (e.g. "Aggressive")
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(tz=timezone.utc).isoformat()

    rows = []

    # Per-ticker rows from summary
    for ticker, row in summary.iterrows():
        rows.append({
            "run_timestamp":      run_ts,
            "config_label":       cfg_label,
            "ticker":             ticker,
            "total_invested_usd": float(row.get("total_invested_usd", 0)),
            "market_value_usd":   float(row.get("market_value_usd", 0)),
            "unrealized_pnl_usd": float(row.get("unrealized_pnl_usd", 0)),
            "total_pnl_pct":      float(row.get("total_pnl_pct", 0)),
            "num_trades":         int(row.get("num_trades", 0)),
        })

    # Aggregate portfolio row
    total_invested = summary["total_invested_usd"].sum()
    total_value    = summary["market_value_usd"].sum()
    rows.append({
        "run_timestamp":      run_ts,
        "config_label":       cfg_label,
        "ticker":             "TOTAL",
        "total_invested_usd": round(float(total_invested), 2),
        "market_value_usd":   round(float(total_value), 2),
        "unrealized_pnl_usd": round(float(total_value - total_invested), 2),
        "total_pnl_pct":      round(float((total_value - total_invested) / total_invested * 100)
                                    if total_invested else 0, 2),
        "num_trades":         int(summary["num_trades"].sum()) if "num_trades" in summary.columns else 0,
        # Portfolio-level risk metrics
        "sharpe":             metrics.get("sharpe_ratio"),
        "sortino":            metrics.get("sortino_ratio"),
        "max_drawdown_pct":   metrics.get("max_drawdown_pct"),
        "cagr_pct":           metrics.get("cagr_pct"),
        "calmar":             metrics.get("calmar_ratio"),
        "win_rate_pct":       metrics.get("win_rate_pct"),
    })

    new_df = pd.DataFrame(rows)
    new_df["run_timestamp"] = pd.to_datetime(new_df["run_timestamp"], utc=True)

    if HIST_PATH.exists():
        existing = pd.read_parquet(HIST_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(HIST_PATH, index=False, compression="snappy")
    logger.info("Performance snapshot saved (%d total runs)", _count_runs(combined))


def load_history() -> pd.DataFrame:
    """Load all historical snapshots. Returns empty DataFrame if none yet or file is corrupt."""
    if not HIST_PATH.exists() or HIST_PATH.stat().st_size == 0:
        if HIST_PATH.exists():
            HIST_PATH.unlink()   # delete the 0-byte file so next save starts clean
        return pd.DataFrame()
    try:
        df = pd.read_parquet(HIST_PATH)
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], utc=True)
        return df.sort_values("run_timestamp")
    except Exception:
        HIST_PATH.unlink()       # delete corrupt file
        return pd.DataFrame()


def _count_runs(df: pd.DataFrame) -> int:
    return df["run_timestamp"].nunique() if not df.empty else 0


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------

def monthly_equity_breakdown(equity: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the total portfolio equity curve to monthly cadence.
    Returns columns: month, start_value, end_value, change_usd, change_pct.
    """
    if equity.empty or "total_portfolio_usd" not in equity.columns:
        return pd.DataFrame()

    eq = equity["total_portfolio_usd"].copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    eq = eq.sort_index()

    monthly = eq.resample("ME").last().dropna()
    if len(monthly) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(1, len(monthly)):
        start = monthly.iloc[i - 1]
        end   = monthly.iloc[i]
        rows.append({
            "month":      monthly.index[i].strftime("%Y-%m"),
            "start_value": round(float(start), 2),
            "end_value":   round(float(end), 2),
            "change_usd":  round(float(end - start), 2),
            "change_pct":  round(float((end - start) / start * 100) if start else 0, 2),
        })
    return pd.DataFrame(rows)


def portfolio_history_chart_data(history: pd.DataFrame) -> pd.DataFrame:
    """
    Extract TOTAL rows from history for charting portfolio value over time.
    Returns: run_timestamp, config_label, market_value_usd, total_pnl_pct
    """
    if history.empty:
        return pd.DataFrame()
    total = history[history["ticker"] == "TOTAL"].copy()
    return total[["run_timestamp", "config_label", "market_value_usd", "total_pnl_pct",
                  "sharpe", "cagr_pct"]].reset_index(drop=True)
