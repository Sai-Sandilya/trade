"""
bot.py — Fractional long-term DCA + MA/RSI-trigger backtesting engine.

Strategy (minimum holding > 15 days):
  • Base cadence : Monthly DCA (last trading day of each month).
  • Trigger overlay:
      - RSI(21) < oversold_threshold  → 2× monthly budget (oversold dip-buy)
      - Price > 200-day SMA           → 1× budget (uptrend, normal pace)
      - Price < 200-day SMA           → 1.5× budget (mild accumulation in downtrend)
  • Optional exits (enable_exits=True):
      - Take-profit: sell entire position when price rises X% above avg cost
      - Stop-loss:   sell entire position when price falls X% below avg cost
  • Optional rebalancing (enable_rebalance=True):
      - Checked every rebalance_frequency_months months
      - Sells overweight tickers, buys underweight tickers back to target_weights
  • No intraday logic — all decisions use end-of-month closing prices.
  • Fractional shares computed to 6 decimal places.
  • Transaction costs: configurable slippage (bps) + flat clearing fee per trade.

Output: per-trade log (BUY / SELL / REBALANCE_BUY / REBALANCE_SELL),
        per-ticker P&L summary, portfolio equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 21) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[avg_loss == 0] = 100
    return rsi


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AMD"])
    monthly_budget_usd: float = 50.0
    oversold_rsi: float = 35.0
    rsi_period: int = 21
    sma_period: int = 200
    below_sma_multiplier: float = 1.5
    oversold_multiplier: float = 2.0
    slippage_bps: float = 3.0
    clearing_fee_usd: float = 0.005
    min_hold_days: int = 15
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    decimal_places: int = 6

    # ── Exit / sell settings (disabled by default) ────────────────────────────
    enable_exits: bool = False
    take_profit_pct: float = 0.50   # sell all when position up 50% from avg cost
    stop_loss_pct: float = 0.20     # sell all when position down 20% from avg cost

    # ── Rebalancing settings (disabled by default) ────────────────────────────
    enable_rebalance: bool = False
    rebalance_threshold_pct: float = 0.10   # rebalance when any ticker drifts >10%
    rebalance_frequency_months: int = 3      # check every 3 months (quarterly)
    target_weights: Optional[dict] = None    # None = equal weight across all tickers


# ── Engine ────────────────────────────────────────────────────────────────────

class LongTermDCABot:
    def __init__(self, config: BotConfig = BotConfig()):
        self.cfg = config
        self.trades: list[dict] = []
        self.holdings: dict[str, float] = {t: 0.0 for t in config.tickers}
        self.total_fees: float = 0.0

        # Cost basis tracking (total dollars invested, excluding fees, per ticker)
        # Used to compute avg cost and realized P&L on sells
        self._cost_basis: dict[str, float] = {t: 0.0 for t in config.tickers}
        self._realized_pnl: dict[str, float] = {t: 0.0 for t in config.tickers}

        # Last rebalance date — avoids rebalancing every month
        self._last_rebalance: Optional[pd.Timestamp] = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self, ticker: str, full: bool = False) -> pd.DataFrame:
        """
        Load cleaned data for *ticker*.
        Indicators computed on FULL history first to avoid warmup NaN bug,
        then sliced to start/end window.
        full=True: skip slice (used by summary/equity_curve for end_date valuation).
        """
        path = CLEAN_DIR / f"{ticker}_clean.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Cleaned file not found: {path}")
        df = pd.read_parquet(path, engine="pyarrow")[["Close", "Volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        df["RSI"]   = _rsi(df["Close"], self.cfg.rsi_period)
        df["SMA200"] = _sma(df["Close"], self.cfg.sma_period)

        if not full:
            if self.cfg.start_date:
                df = df[df.index >= pd.Timestamp(self.cfg.start_date, tz="UTC")]
            if self.cfg.end_date:
                df = df[df.index <= pd.Timestamp(self.cfg.end_date, tz="UTC")]

        return df

    # ── Budget multiplier logic ───────────────────────────────────────────────

    def _budget(self, rsi: float, close: float, sma200: float) -> tuple[float, str]:
        rsi_valid = not np.isnan(rsi)
        sma_valid = not np.isnan(sma200)

        if rsi_valid and rsi < self.cfg.oversold_rsi:
            return self.cfg.monthly_budget_usd * self.cfg.oversold_multiplier, "RSI_OVERSOLD_2X"
        if sma_valid and close < sma200:
            return self.cfg.monthly_budget_usd * self.cfg.below_sma_multiplier, "BELOW_SMA200_1.5X"
        return self.cfg.monthly_budget_usd, "DCA_NORMAL"

    # ── Average cost per share ────────────────────────────────────────────────

    def _avg_cost(self, ticker: str) -> float:
        shares = self.holdings[ticker]
        if shares <= 0:
            return 0.0
        return self._cost_basis[ticker] / shares

    # ── Trade execution — BUY ─────────────────────────────────────────────────

    def _execute_buy(
        self,
        ticker: str,
        date: pd.Timestamp,
        close: float,
        rsi: float,
        sma200: float,
        action: str = "BUY",
        forced_budget: Optional[float] = None,
        trigger_override: Optional[str] = None,
    ) -> dict:
        budget, trigger = self._budget(rsi, close, sma200)
        if forced_budget is not None:
            budget = forced_budget
        if trigger_override is not None:
            trigger = trigger_override

        slippage_cost = close * (self.cfg.slippage_bps / 10_000)
        fill_price    = round(close + slippage_cost, 4)
        net_budget    = budget - self.cfg.clearing_fee_usd
        shares        = round(max(net_budget, 0) / fill_price, self.cfg.decimal_places)

        self.holdings[ticker]    = round(self.holdings[ticker] + shares, self.cfg.decimal_places)
        self._cost_basis[ticker] = round(self._cost_basis[ticker] + budget, 4)
        fee = self.cfg.clearing_fee_usd + (shares * slippage_cost)
        self.total_fees += fee

        trade = {
            "date":               date.date(),
            "ticker":             ticker,
            "action":             action,
            "budget_usd":         round(budget, 4),
            "fill_price":         fill_price,
            "shares_transacted":  shares,
            "cumulative_shares":  self.holdings[ticker],
            "rsi":                round(rsi, 2) if not np.isnan(rsi) else None,
            "sma200":             round(sma200, 4) if not np.isnan(sma200) else None,
            "trigger":            trigger,
            "fee_usd":            round(fee, 6),
            "proceeds_usd":       None,
            "realized_pnl_usd":   None,
        }
        self.trades.append(trade)
        return trade

    # ── Trade execution — SELL ────────────────────────────────────────────────

    def _execute_sell(
        self,
        ticker: str,
        date: pd.Timestamp,
        close: float,
        shares_to_sell: float,
        trigger: str,
        rsi: float = float("nan"),
        sma200: float = float("nan"),
    ) -> dict:
        shares_to_sell = min(shares_to_sell, self.holdings[ticker])
        if shares_to_sell <= 0:
            return {}

        # Slippage on sells: price drops by slippage (selling at a slightly worse price)
        slippage_cost = close * (self.cfg.slippage_bps / 10_000)
        fill_price    = round(close - slippage_cost, 4)
        proceeds      = round(shares_to_sell * fill_price, 4)
        fee           = self.cfg.clearing_fee_usd + (shares_to_sell * slippage_cost)
        net_proceeds  = round(proceeds - fee, 4)

        # Realized P&L: proportional cost basis of shares sold
        sell_fraction  = shares_to_sell / self.holdings[ticker] if self.holdings[ticker] > 0 else 0
        cost_of_sold   = round(self._cost_basis[ticker] * sell_fraction, 4)
        realized_pnl   = round(net_proceeds - cost_of_sold, 4)

        # Update state
        self.holdings[ticker]     = round(self.holdings[ticker] - shares_to_sell, self.cfg.decimal_places)
        self._cost_basis[ticker]  = round(self._cost_basis[ticker] - cost_of_sold, 4)
        self._realized_pnl[ticker] = round(self._realized_pnl[ticker] + realized_pnl, 4)
        self.total_fees += fee

        trade = {
            "date":               date.date(),
            "ticker":             ticker,
            "action":             trigger,
            "budget_usd":         0.0,
            "fill_price":         fill_price,
            "shares_transacted":  -shares_to_sell,   # negative = sold
            "cumulative_shares":  self.holdings[ticker],
            "rsi":                round(rsi, 2) if not np.isnan(rsi) else None,
            "sma200":             round(sma200, 4) if not np.isnan(sma200) else None,
            "trigger":            trigger,
            "fee_usd":            round(fee, 6),
            "proceeds_usd":       net_proceeds,
            "realized_pnl_usd":   realized_pnl,
        }
        self.trades.append(trade)
        return trade

    # ── Exit checks (take-profit / stop-loss) ─────────────────────────────────

    def _check_exits(
        self,
        ticker: str,
        date: pd.Timestamp,
        close: float,
        rsi: float,
        sma200: float,
    ) -> bool:
        """
        Check take-profit and stop-loss conditions.
        Returns True if a sell was executed (so the monthly buy is skipped).
        """
        if not self.cfg.enable_exits:
            return False
        if self.holdings[ticker] <= 0:
            return False

        avg_cost = self._avg_cost(ticker)
        if avg_cost <= 0:
            return False

        gain_pct = (close - avg_cost) / avg_cost

        if gain_pct >= self.cfg.take_profit_pct:
            logger.info(
                "[%s] TAKE_PROFIT on %s: price $%.2f is +%.1f%% above avg cost $%.2f",
                date.date(), ticker, close, gain_pct * 100, avg_cost,
            )
            self._execute_sell(ticker, date, close, self.holdings[ticker],
                               "TAKE_PROFIT", rsi, sma200)
            return True

        if gain_pct <= -self.cfg.stop_loss_pct:
            logger.info(
                "[%s] STOP_LOSS on %s: price $%.2f is %.1f%% below avg cost $%.2f",
                date.date(), ticker, close, abs(gain_pct) * 100, avg_cost,
            )
            self._execute_sell(ticker, date, close, self.holdings[ticker],
                               "STOP_LOSS", rsi, sma200)
            return True

        return False

    # ── Rebalancing ───────────────────────────────────────────────────────────

    def _check_rebalance(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        rsi_map: dict[str, float],
        sma_map: dict[str, float],
    ) -> None:
        """
        Quarterly portfolio rebalancing.
        Sells overweight tickers and buys underweight tickers back to target weights.
        Only fires if:
          - enable_rebalance is True
          - At least rebalance_frequency_months have passed since last rebalance
          - At least one ticker drifts beyond rebalance_threshold_pct
        """
        if not self.cfg.enable_rebalance:
            return

        # Enforce cadence
        if self._last_rebalance is not None:
            months_since = (
                (date.year - self._last_rebalance.year) * 12
                + (date.month - self._last_rebalance.month)
            )
            if months_since < self.cfg.rebalance_frequency_months:
                return

        # Compute current portfolio values
        values = {
            t: self.holdings[t] * prices[t]
            for t in self.cfg.tickers
            if t in prices and self.holdings[t] > 0
        }
        total_value = sum(values.values())
        if total_value <= 0:
            return

        # Target weights — equal by default
        targets = self.cfg.target_weights or {t: 1 / len(self.cfg.tickers) for t in self.cfg.tickers}

        # Current weights
        current_weights = {t: values.get(t, 0) / total_value for t in self.cfg.tickers}

        # Check if any ticker exceeds threshold
        max_drift = max(
            abs(current_weights.get(t, 0) - targets.get(t, 0))
            for t in self.cfg.tickers
        )
        if max_drift < self.cfg.rebalance_threshold_pct:
            return

        logger.info(
            "[%s] REBALANCE triggered — max drift %.1f%% exceeds threshold %.1f%%",
            date.date(), max_drift * 100, self.cfg.rebalance_threshold_pct * 100,
        )

        # Execute sells for overweight tickers first, then buys for underweight
        target_values = {t: total_value * targets.get(t, 0) for t in self.cfg.tickers}

        for ticker in self.cfg.tickers:
            if ticker not in prices:
                continue
            excess_value = values.get(ticker, 0) - target_values.get(ticker, 0)
            if excess_value > 1.0:  # overweight — sell excess
                shares_to_sell = round(excess_value / prices[ticker], self.cfg.decimal_places)
                self._execute_sell(
                    ticker, date, prices[ticker], shares_to_sell,
                    "REBALANCE_SELL", rsi_map.get(ticker, float("nan")),
                    sma_map.get(ticker, float("nan")),
                )

        # Recompute total after sells (proceeds stay as cash to redeploy)
        total_after_sells = sum(
            self.holdings[t] * prices[t] for t in self.cfg.tickers if t in prices
        )

        for ticker in self.cfg.tickers:
            if ticker not in prices:
                continue
            current_value  = self.holdings[ticker] * prices[ticker]
            target_value   = total_value * targets.get(ticker, 0)
            deficit_value  = target_value - current_value
            if deficit_value > 1.0:  # underweight — buy to top up
                self._execute_buy(
                    ticker, date, prices[ticker],
                    rsi_map.get(ticker, float("nan")),
                    sma_map.get(ticker, float("nan")),
                    action="REBALANCE_BUY",
                    forced_budget=deficit_value,
                    trigger_override="REBALANCE_BUY",
                )

        self._last_rebalance = date

    # ── Backtest loop ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run monthly backtest across all tickers in chronological order.

        The loop is date-first (not ticker-first) so that exits and
        rebalancing fire at the correct point in time relative to buys —
        a ticker-first loop would execute all of one ticker before starting
        the next, making cross-ticker rebalancing impossible.
        """
        logger.info(
            "Starting Long-Term DCA backtest | tickers=%s | monthly_budget=$%.2f | "
            "min_hold=%d days | RSI_period=%d | SMA_period=%d | "
            "exits=%s | rebalance=%s",
            self.cfg.tickers, self.cfg.monthly_budget_usd,
            self.cfg.min_hold_days, self.cfg.rsi_period, self.cfg.sma_period,
            self.cfg.enable_exits, self.cfg.enable_rebalance,
        )

        # Load all ticker data upfront
        all_data: dict[str, pd.DataFrame] = {}
        for ticker in self.cfg.tickers:
            all_data[ticker] = self._load(ticker)

        # Build union of all last-trading-days across all tickers, sorted
        monthly_dates: set[pd.Timestamp] = set()
        for ticker, df in all_data.items():
            months = df.index.tz_localize(None).to_period("M").to_series(index=df.index)
            is_last = months != months.shift(-1)
            for ts in df[is_last].index:
                monthly_dates.add(ts)

        for date in sorted(monthly_dates):
            # Gather prices, RSI, SMA200 for all tickers on this date
            prices  = {}
            rsi_map = {}
            sma_map = {}
            for ticker, df in all_data.items():
                if date in df.index:
                    row = df.loc[date]
                    prices[ticker]  = row["Close"]
                    rsi_map[ticker] = row["RSI"]
                    sma_map[ticker] = row["SMA200"]

            # 1. Check exits before buying — don't buy into a position we just closed
            exited: set[str] = set()
            if self.cfg.enable_exits:
                for ticker in self.cfg.tickers:
                    if ticker in prices:
                        sold = self._check_exits(
                            ticker, date, prices[ticker],
                            rsi_map.get(ticker, float("nan")),
                            sma_map.get(ticker, float("nan")),
                        )
                        if sold:
                            exited.add(ticker)

            # 2. Check rebalancing (quarterly, cross-ticker)
            self._check_rebalance(date, prices, rsi_map, sma_map)

            # 3. Monthly DCA buy for each ticker that has data on this date
            for ticker in self.cfg.tickers:
                if ticker not in prices:
                    continue
                if ticker in exited:
                    continue   # skip buy this month for tickers just exited
                self._execute_buy(
                    ticker=ticker,
                    date=date,
                    close=prices[ticker],
                    rsi=rsi_map.get(ticker, float("nan")),
                    sma200=sma_map.get(ticker, float("nan")),
                )

        logger.info(
            "Backtest complete. Trades: %d | Total fees: $%.4f",
            len(self.trades), self.total_fees,
        )
        return pd.DataFrame(self.trades)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ticker in self.cfg.tickers:
            path = CLEAN_DIR / f"{ticker}_clean.parquet"
            if not path.exists():
                continue
            hist = self._load(ticker, full=True)
            if self.cfg.end_date:
                hist = hist[hist.index <= pd.Timestamp(self.cfg.end_date, tz="UTC")]
            last_close = hist["Close"].iloc[-1]

            t = trade_log[trade_log["ticker"] == ticker]

            total_invested  = t.loc[t["action"].isin(["BUY", "RSI_OVERSOLD_2X",
                                                       "BELOW_SMA200_1.5X", "DCA_NORMAL",
                                                       "REBALANCE_BUY"]), "budget_usd"].sum()
            total_proceeds  = t.loc[t["proceeds_usd"].notna(), "proceeds_usd"].sum()
            net_invested    = round(total_invested - total_proceeds, 4)
            realized_pnl    = round(self._realized_pnl.get(ticker, 0.0), 4)
            total_shares    = self.holdings[ticker]
            market_value    = round(total_shares * last_close, 4)
            unrealized_pnl  = round(market_value - self._cost_basis.get(ticker, 0.0), 4)
            total_pnl       = round(realized_pnl + unrealized_pnl, 4)
            total_pnl_pct   = round((total_pnl / net_invested) * 100, 2) if net_invested else 0.0
            avg_cost        = self._avg_cost(ticker)

            rows.append({
                "ticker":             ticker,
                "total_invested_usd": round(total_invested, 4),
                "total_proceeds_usd": round(total_proceeds, 4),
                "net_invested_usd":   net_invested,
                "avg_cost_per_share": round(avg_cost, 4),
                "total_shares":       total_shares,
                "last_close":         round(last_close, 4),
                "market_value_usd":   market_value,
                "realized_pnl_usd":   realized_pnl,
                "unrealized_pnl_usd": unrealized_pnl,
                "total_pnl_usd":      total_pnl,
                "total_pnl_pct":      total_pnl_pct,
                "num_trades":         len(t),
            })
        return pd.DataFrame(rows).set_index("ticker")

    def equity_curve(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Daily portfolio equity curve from cumulative share positions.
        Handles sells correctly because equity_curve reads cumulative_shares
        which is updated (decremented) by _execute_sell.
        """
        curves = []
        for ticker in self.cfg.tickers:
            path = CLEAN_DIR / f"{ticker}_clean.parquet"
            if not path.exists():
                continue
            hist = self._load(ticker, full=True)
            if self.cfg.end_date:
                hist = hist[hist.index <= pd.Timestamp(self.cfg.end_date, tz="UTC")]
            closes = hist[["Close"]].rename(columns={"Close": ticker})

            t = trade_log[trade_log["ticker"] == ticker].copy()
            t["date"] = pd.to_datetime(t["date"], utc=True)
            t = t.set_index("date")[["cumulative_shares"]].rename(
                columns={"cumulative_shares": "shares"}
            )

            combined = closes.join(t, how="left")
            combined["shares"] = combined["shares"].ffill().fillna(0)
            combined[f"{ticker}_value"] = combined[ticker] * combined["shares"]
            curves.append(combined[[f"{ticker}_value"]])

        if not curves:
            return pd.DataFrame()

        equity = pd.concat(curves, axis=1, join="outer").ffill().fillna(0)
        equity["total_portfolio_usd"] = equity.sum(axis=1)
        return equity


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot = LongTermDCABot()
    log = bot.run()
    print(log.tail(10).to_string())
    print("\n--- Summary ---")
    print(bot.summary(log).to_string())
