"""
metrics.py - Portfolio risk and performance metrics.

All metrics computed from a daily equity curve (pd.Series of portfolio values).
No external dependencies beyond numpy and pandas.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _to_nav(equity: pd.Series, trade_log: pd.DataFrame) -> pd.Series:
    """
    Convert a raw equity curve to a unitized NAV series (starts at 1.0).

    Problem: raw equity jumps on deposit days (monthly DCA buys), so
    pct_change() sees a +50% "return" simply because $50 was deposited into
    a $100 portfolio. This makes Sharpe, Sortino, volatility, CAGR, and
    drawdown all wrong.

    Fix (unitized NAV — same method used by mutual funds):
      - Start with nav_price = 1.0, total_units = 0
      - Each deposit issues new units at the CURRENT nav_price, so the
        nav_price itself is unchanged by the deposit
      - Between deposits, nav_price moves purely with market returns
      - Result: nav.pct_change() is deposit-free daily return

    If trade_log is None or empty, returns equity normalised to 1.0 (no-op).
    """
    if trade_log is None or trade_log.empty:
        eq = equity[equity > 0]
        return eq / eq.iloc[0] if not eq.empty else equity

    # Build deposit lookup: date -> total budget invested that day
    buys = trade_log[trade_log["budget_usd"] > 0].copy()
    buys["_date"] = pd.to_datetime(buys["date"], utc=True)
    deposits = buys.groupby("_date")["budget_usd"].sum()

    nav_price   = 1.0
    total_units = 0.0
    nav_values: list[float] = []

    for date, raw_value in equity.items():
        deposit_today = float(deposits.get(date, 0.0))

        # Value BEFORE today's deposit = raw equity minus today's fresh cash
        value_before = raw_value - deposit_today

        # Update nav_price from market movement (only when units exist)
        if total_units > 0 and value_before > 0:
            nav_price = value_before / total_units

        # Issue new units for today's deposit at current nav_price
        if deposit_today > 0 and nav_price > 0:
            total_units += deposit_today / nav_price

        nav_values.append(nav_price)

    return pd.Series(nav_values, index=equity.index, dtype=float)


def _daily_returns(equity: pd.Series) -> pd.Series:
    """Percentage daily returns, dropping leading zeros before first trade."""
    eq = equity[equity > 0]
    return eq.pct_change().dropna()


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sharpe ratio.
    risk_free_rate: annual rate (default 4% = current US T-bill yield).
    Returns NaN if fewer than 2 data points.
    """
    r = _daily_returns(equity)
    if len(r) < 2:
        return float("nan")
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess   = r - daily_rf
    if excess.std() == 0:
        return float("nan")
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sortino ratio (penalises only downside volatility).
    risk_free_rate: annual rate (default 4%).
    Returns NaN if no negative returns exist (all-up equity curve).
    """
    r = _daily_returns(equity)
    if len(r) < 2:
        return float("nan")
    daily_rf   = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess     = r - daily_rf
    downside   = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    # Use RMS of downside returns (Sortino denominator), NOT std().
    # All values in downside are strictly < 0, so RMS is guaranteed > 0.
    # Checking downside.std() == 0 was wrong: identical negative returns have
    # std=0 but RMS > 0, causing a false NaN.
    downside_std = np.sqrt((downside ** 2).mean())
    return float((excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative percentage.
    e.g. -0.34 means the portfolio fell 34% from its peak at worst.
    Returns 0.0 if equity never declines.
    """
    eq = equity[equity > 0]
    if eq.empty:
        return 0.0
    rolling_peak = eq.cummax()
    drawdown     = (eq - rolling_peak) / rolling_peak
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series) -> float:
    """
    Calmar ratio = annualised CAGR / abs(max drawdown).
    Higher is better. Returns NaN if max drawdown is 0.
    """
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("nan")
    cagr = annualised_return(equity)
    return float(cagr / abs(mdd))


def annualised_return(equity: pd.Series) -> float:
    """
    Compound Annual Growth Rate (CAGR) from first non-zero value to last.
    Returns as a decimal (0.15 = 15% per year).
    """
    eq = equity[equity > 0]
    if len(eq) < 2:
        return 0.0
    total_return = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def annualised_volatility(equity: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    r = _daily_returns(equity)
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def win_rate(equity: pd.Series) -> float:
    """
    Percentage of months where the portfolio value was higher than the prior month.
    Uses the equity curve (DCA never sells, so there is no per-trade win/loss).
    """
    monthly = equity[equity > 0].resample("ME").last().dropna()
    if len(monthly) < 2:
        return float("nan")
    up_months = (monthly.diff().dropna() > 0).sum()
    return float(up_months / (len(monthly) - 1))


def compute_all(
    equity: pd.Series,
    risk_free_rate: float = 0.04,
    trade_log: pd.DataFrame | None = None,
) -> dict:
    """
    Compute all metrics for one equity curve series.

    equity:          daily portfolio value (pd.Series with DatetimeIndex).
    trade_log:       bot.run() output DataFrame — used to strip out deposit
                     effects via unitized NAV conversion.  When None, raw
                     equity is used (correct only for lump-sum portfolios).
    risk_free_rate:  annual rate for Sharpe/Sortino (default 4%).

    Returns a dict of metric_name -> value.
    All percentage metrics are stored as decimals (0.15 = 15%).
    """
    # Convert to NAV series to remove deposit-day return distortions
    nav = _to_nav(equity, trade_log)

    # total_return: actual P&L vs cash invested (user-facing accounting)
    if trade_log is not None and not trade_log.empty:
        total_invested = float(trade_log[trade_log["budget_usd"] > 0]["budget_usd"].sum())
        ending_value   = float(equity.iloc[-1]) if not equity.empty else 0.0
        total_return   = (ending_value - total_invested) / total_invested if total_invested > 0 else 0.0
    else:
        total_return = float((equity.iloc[-1] / equity[equity > 0].iloc[0]) - 1) \
                       if (equity > 0).any() else 0.0

    return {
        "cagr":             annualised_return(nav),
        "annualised_vol":   annualised_volatility(nav),
        "sharpe":           sharpe_ratio(nav, risk_free_rate),
        "sortino":          sortino_ratio(nav, risk_free_rate),
        "max_drawdown":     max_drawdown(nav),
        "calmar":           calmar_ratio(nav),
        "win_rate_monthly": win_rate(nav),
        "total_return":     total_return,
    }


def format_metrics(metrics: dict) -> dict:
    """Return a human-readable version of the metrics dict."""
    def pct(v):
        return f"{v * 100:.2f}%" if not np.isnan(v) else "N/A"
    def ratio(v):
        return f"{v:.3f}" if not np.isnan(v) else "N/A"

    return {
        "Total Return":         pct(metrics["total_return"]),
        "CAGR":                 pct(metrics["cagr"]),
        "Annualised Volatility":pct(metrics["annualised_vol"]),
        "Sharpe Ratio":         ratio(metrics["sharpe"]),
        "Sortino Ratio":        ratio(metrics["sortino"]),
        "Max Drawdown":         pct(metrics["max_drawdown"]),
        "Calmar Ratio":         ratio(metrics["calmar"]),
        "Monthly Win Rate":     pct(metrics["win_rate_monthly"]),
    }


def buy_and_hold_equity(
    ticker: str,
    monthly_budget_usd: float,
    clean_dir,
    start_date=None,
    end_date=None,
    slippage_bps: float = 3.0,
    clearing_fee_usd: float = 0.005,
) -> pd.Series:
    """
    Simulate buying a fixed dollar amount on the first trading day of each month
    (pure buy-and-hold, no RSI or SMA triggers) and return a daily equity curve.

    This is the benchmark: same capital deployed, same fees as the DCA bot.
    slippage_bps and clearing_fee_usd must match the BotConfig values used in
    the backtest so the comparison is apples-to-apples.

    Used to answer: "did the DCA trigger system beat plain buy-and-hold?"
    """
    from pathlib import Path
    path = Path(clean_dir) / f"{ticker}_clean.parquet"
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)[["Close"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

    if df.empty:
        return pd.Series(dtype=float)

    # First trading day of each month
    months    = df.index.tz_localize(None).to_period("M").to_series(index=df.index)
    is_first  = months != months.shift(1)
    buy_dates = df[is_first].index

    cumulative_shares = 0.0
    shares_series     = pd.Series(0.0, index=df.index)
    for date in buy_dates:
        close        = df.loc[date, "Close"]
        slippage_cost= close * (slippage_bps / 10_000)
        fill_price   = close + slippage_cost
        net_budget   = monthly_budget_usd - clearing_fee_usd
        shares       = max(net_budget, 0.0) / fill_price
        cumulative_shares += shares
        shares_series.loc[date:] = cumulative_shares

    equity = shares_series * df["Close"]
    equity.name = f"{ticker}_bah"
    return equity
