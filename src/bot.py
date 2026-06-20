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
    rebalance_threshold_pct: float = 0.10
    rebalance_frequency_months: int = 3
    target_weights: Optional[dict] = None

    # ── 1. Regime detection ────────────────────────────────────────────────────
    enable_regime_filter: bool = False
    regime_ticker: str = "SPY"

    # ── 2. RSI extreme tiers ───────────────────────────────────────────────────
    extreme_oversold_rsi: float = 25.0
    extreme_oversold_multiplier: float = 3.0
    crash_rsi: float = 15.0
    crash_multiplier: float = 5.0

    # ── 3. Earnings calendar ───────────────────────────────────────────────────
    enable_earnings_filter: bool = False
    earnings_tickers: list = field(default_factory=lambda: ["AMD"])
    earnings_pre_days: int = 14
    earnings_pre_multiplier: float = 0.5
    earnings_beat_multiplier: float = 2.0

    # ── 4. Kelly position sizing ───────────────────────────────────────────────
    enable_kelly_sizing: bool = False
    kelly_lookback_months: int = 24

    # ── 5. ATR trailing stop ───────────────────────────────────────────────────
    enable_trailing_stop: bool = False
    trailing_atr_multiple: float = 2.5

    # ── 6. Sector rotation ─────────────────────────────────────────────────────
    # During bear regime redirect normal DCA budgets to defensive_ticker.
    # defensive_ticker data must be present in data/cleaned/.
    enable_sector_rotation: bool = False
    defensive_ticker: str = "XLU"


# ── Engine ────────────────────────────────────────────────────────────────────

class LongTermDCABot:
    def __init__(self, config: BotConfig = BotConfig()):
        self.cfg = config

        # Sector rotation: defensive_ticker must be in tickers so that holdings,
        # cost_basis, and all_data are all initialized for it. Without this,
        # _execute_buy raises KeyError and capital silently disappears.
        if (
            config.enable_sector_rotation
            and config.defensive_ticker not in config.tickers
        ):
            config.tickers = list(config.tickers) + [config.defensive_ticker]

        all_tickers = config.tickers

        self.trades: list[dict] = []
        self.holdings: dict[str, float] = {t: 0.0 for t in all_tickers}
        self.total_fees: float = 0.0

        # Cost basis tracking (total dollars invested, excluding fees, per ticker)
        # Used to compute avg cost and realized P&L on sells
        self._cost_basis: dict[str, float] = {t: 0.0 for t in all_tickers}
        self._realized_pnl: dict[str, float] = {t: 0.0 for t in all_tickers}

        # Last rebalance date — avoids rebalancing every month
        self._last_rebalance: Optional[pd.Timestamp] = None

        # ATR trailing stop — tracks highest close since last buy per ticker
        self._highest_close: dict[str, float] = {t: 0.0 for t in all_tickers}

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
        raw  = pd.read_parquet(path, engine="pyarrow")
        cols = ["Close", "Volume"] + [c for c in ("High", "Low") if c in raw.columns]
        df   = raw[cols].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        df["RSI"]    = _rsi(df["Close"], self.cfg.rsi_period)
        df["SMA200"] = _sma(df["Close"], self.cfg.sma_period)

        # ATR14 — required by trailing stop strategy
        if "High" in df.columns and "Low" in df.columns:
            prev_close = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"]  - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["ATR14"] = tr.rolling(14, min_periods=14).mean()
        else:
            df["ATR14"] = float("nan")

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

        # RSI extreme tiers — checked in ascending order of severity
        if rsi_valid and rsi < self.cfg.crash_rsi:
            return self.cfg.monthly_budget_usd * self.cfg.crash_multiplier, "RSI_CRASH_5X"
        if rsi_valid and rsi < self.cfg.extreme_oversold_rsi:
            return self.cfg.monthly_budget_usd * self.cfg.extreme_oversold_multiplier, "RSI_EXTREME_3X"
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

    # ── Strategy 4: Kelly position sizing ─────────────────────────────────────

    def _kelly_multiplier(self, df: pd.DataFrame, date: pd.Timestamp) -> float:
        """Half-Kelly budget multiplier from prior kelly_lookback_months of monthly returns."""
        if not self.cfg.enable_kelly_sizing:
            return 1.0
        start = date - pd.DateOffset(months=self.cfg.kelly_lookback_months)
        hist  = df.loc[(df.index >= start) & (df.index < date), "Close"]
        if len(hist) < 50:
            return 1.0
        monthly = hist.resample("ME").last().pct_change().dropna()
        if len(monthly) < 6:
            return 1.0
        wins   = monthly[monthly > 0]
        losses = monthly[monthly < 0]
        if wins.empty or losses.empty:
            return 1.0
        win_rate  = len(wins) / len(monthly)
        avg_win   = float(wins.mean())
        avg_loss  = float(abs(losses.mean()))
        if avg_win <= 0:
            return 1.0
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return float(max(0.5, min(2.0, 1.0 + kelly_f / 2.0)))

    # ── Strategy 3: Earnings calendar ─────────────────────────────────────────

    def _fetch_earnings_dates(self) -> dict[str, object]:
        """Fetch historical earnings dates + EPS actuals for earnings_tickers."""
        result: dict[str, object] = {}
        if not self.cfg.enable_earnings_filter:
            return result
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — earnings filter disabled")
            return result
        for ticker in self.cfg.earnings_tickers:
            try:
                ed = yf.Ticker(ticker).earnings_dates
                if ed is not None and not ed.empty:
                    ed = ed.copy()
                    ed.index = pd.to_datetime(ed.index, utc=True)
                    result[ticker] = ed
            except Exception as exc:
                logger.warning("Could not fetch earnings for %s: %s", ticker, exc)
        return result

    def _earnings_budget_scale(
        self,
        ticker: str,
        date: pd.Timestamp,
        earnings_map: dict,
    ) -> tuple[float, str]:
        """Return (scale, trigger_suffix) based on proximity to earnings date."""
        if not self.cfg.enable_earnings_filter or ticker not in earnings_map:
            return 1.0, ""
        ed          = earnings_map[ticker]
        pre_window  = pd.Timedelta(days=self.cfg.earnings_pre_days)
        post_window = pd.Timedelta(days=30)
        for earn_date, row in ed.iterrows():
            if earn_date.tzinfo is None:
                earn_date = earn_date.tz_localize("UTC")
            if (earn_date - pre_window) <= date < earn_date:
                return self.cfg.earnings_pre_multiplier, "EARNINGS_PRE_REDUCE"
            if earn_date <= date <= (earn_date + post_window):
                try:
                    reported = float(row.get("Reported EPS", float("nan")))
                    estimate = float(row.get("EPS Estimate",  float("nan")))
                    if not (np.isnan(reported) or np.isnan(estimate)) and reported > estimate:
                        return self.cfg.earnings_beat_multiplier, "EARNINGS_BEAT_BUY"
                except (TypeError, ValueError):
                    pass
                return 1.0, ""
        return 1.0, ""

    # ── Strategy 5: ATR trailing stop ─────────────────────────────────────────

    def _check_trailing_stop(
        self,
        ticker: str,
        date: pd.Timestamp,
        close: float,
        rsi: float,
        sma200: float,
        atr14: float,
    ) -> bool:
        """ATR-based trailing stop. Returns True if position was closed."""
        if not self.cfg.enable_trailing_stop:
            return False
        if self.holdings[ticker] <= 0 or np.isnan(atr14) or atr14 <= 0:
            return False
        if close > self._highest_close[ticker]:
            self._highest_close[ticker] = close
        trailing_stop = self._highest_close[ticker] - self.cfg.trailing_atr_multiple * atr14
        if close < trailing_stop:
            logger.info(
                "[%s] TRAILING_STOP on %s: $%.2f < stop $%.2f (peak $%.2f − %.1f×ATR $%.2f)",
                date.date(), ticker, close, trailing_stop,
                self._highest_close[ticker], self.cfg.trailing_atr_multiple, atr14,
            )
            self._execute_sell(ticker, date, close, self.holdings[ticker],
                               "TRAILING_STOP", rsi, sma200)
            self._highest_close[ticker] = 0.0
            return True
        return False

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

        # Opening a fresh position — clear any ghost peak from a prior position.
        # Without this, a trailing stop from a previous trade cycle would fire
        # immediately on the new position because _highest_close was never reset
        # by take-profit, stop-loss, or rebalance sells.
        if self.cfg.enable_trailing_stop and self.holdings.get(ticker, 0.0) == 0.0:
            self._highest_close[ticker] = 0.0

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

        # Slippage on sells: fill_price is already worse than close by slippage.
        # Fee is clearing_fee only — slippage is already embedded in fill_price.
        slippage_cost = close * (self.cfg.slippage_bps / 10_000)
        fill_price    = round(close - slippage_cost, 4)
        proceeds      = round(shares_to_sell * fill_price, 4)
        fee           = self.cfg.clearing_fee_usd
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

        # Execute sells for overweight tickers first, then buys for underweight.
        # Capture net_proceeds from each sell into cash_pool so the cash is not lost.
        # (LongTermDCABot has no cash balance — proceeds must be tracked explicitly here.)
        target_values = {t: total_value * targets.get(t, 0) for t in self.cfg.tickers}
        cash_pool = 0.0

        for ticker in self.cfg.tickers:
            if ticker not in prices:
                continue
            excess_value = values.get(ticker, 0) - target_values.get(ticker, 0)
            if excess_value > 1.0:  # overweight — sell excess
                shares_to_sell = round(excess_value / prices[ticker], self.cfg.decimal_places)
                trade = self._execute_sell(
                    ticker, date, prices[ticker], shares_to_sell,
                    "REBALANCE_SELL", rsi_map.get(ticker, float("nan")),
                    sma_map.get(ticker, float("nan")),
                )
                # Accumulate actual cash received (after slippage + fees)
                cash_pool += trade.get("proceeds_usd", 0.0)

        # True portfolio value = remaining shares + cash from sells
        shares_value      = sum(self.holdings[t] * prices[t] for t in self.cfg.tickers if t in prices)
        total_after_sells = shares_value + cash_pool

        # Pass 1 — calculate raw deficits for all underweight tickers
        deficits: dict[str, float] = {}
        for ticker in self.cfg.tickers:
            if ticker not in prices:
                continue
            current_value = self.holdings[ticker] * prices[ticker]
            target_value  = total_after_sells * targets.get(ticker, 0)
            deficit_value = target_value - current_value
            if deficit_value > 1.0:
                deficits[ticker] = deficit_value

        # Pass 2 — distribute cash_pool proportionally so buys never exceed proceeds.
        # This makes the rebalance a true closed system: no fresh cash invented.
        total_deficit = sum(deficits.values())
        for ticker, deficit_value in deficits.items():
            if cash_pool <= 0:
                break
            buy_budget = (deficit_value / total_deficit) * cash_pool
            if buy_budget > 1.0:
                self._execute_buy(
                    ticker, date, prices[ticker],
                    rsi_map.get(ticker, float("nan")),
                    sma_map.get(ticker, float("nan")),
                    action="REBALANCE_BUY",
                    forced_budget=buy_budget,
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

        # Load all ticker data upfront.
        # Skip tickers whose parquet file doesn't exist yet (e.g. defensive_ticker
        # when sector rotation is enabled but XLU hasn't been downloaded).
        all_data: dict[str, pd.DataFrame] = {}
        for ticker in self.cfg.tickers:
            try:
                all_data[ticker] = self._load(ticker)
            except FileNotFoundError:
                logger.warning(
                    "No cleaned data for %s — ticker will be skipped in backtest", ticker
                )

        # Strategy 1 & 6 — load regime ticker if not already in all_data
        regime_data: Optional[pd.DataFrame] = None
        need_regime = self.cfg.enable_regime_filter or self.cfg.enable_sector_rotation
        if need_regime:
            if self.cfg.regime_ticker in all_data:
                regime_data = all_data[self.cfg.regime_ticker]
            else:
                regime_path = CLEAN_DIR / f"{self.cfg.regime_ticker}_clean.parquet"
                if regime_path.exists():
                    regime_data = self._load(self.cfg.regime_ticker)
                else:
                    logger.warning(
                        "Regime ticker %s not in cleaned data — regime filter disabled",
                        self.cfg.regime_ticker,
                    )

        # Strategy 3 — fetch earnings dates once before the loop (one network call)
        earnings_map = self._fetch_earnings_dates()

        # Build union of all last-trading-days across all tickers, sorted
        monthly_dates: set[pd.Timestamp] = set()
        for ticker, df in all_data.items():
            months  = df.index.tz_localize(None).to_period("M").to_series(index=df.index)
            is_last = months != months.shift(-1)
            for ts in df[is_last].index:
                monthly_dates.add(ts)

        for date in sorted(monthly_dates):
            # ── Regime detection (Strategies 1 & 6) ──────────────────────────
            bear_regime = False
            if need_regime and regime_data is not None:
                regime_row = regime_data[regime_data.index <= date]
                if not regime_row.empty:
                    r = regime_row.iloc[-1]
                    bear_regime = (not np.isnan(r["SMA200"])) and (r["Close"] < r["SMA200"])

            # ── Gather prices, RSI, SMA200, ATR14 ────────────────────────────
            prices  = {}
            rsi_map = {}
            sma_map = {}
            atr_map = {}
            for ticker, df in all_data.items():
                if date in df.index:
                    row = df.loc[date]
                    prices[ticker]  = float(row["Close"])
                    rsi_map[ticker] = float(row["RSI"])
                    sma_map[ticker] = float(row["SMA200"])
                    atr_map[ticker] = float(row["ATR14"]) if "ATR14" in df.columns else float("nan")

            # ── 1. ATR Trailing Stop (before other exits) ─────────────────────
            trailing_stopped: set[str] = set()
            for ticker in self.cfg.tickers:
                if ticker in prices:
                    stopped = self._check_trailing_stop(
                        ticker, date, prices[ticker],
                        rsi_map.get(ticker, float("nan")),
                        sma_map.get(ticker, float("nan")),
                        atr_map.get(ticker, float("nan")),
                    )
                    if stopped:
                        trailing_stopped.add(ticker)

            # ── 2. Take-profit / stop-loss exits ─────────────────────────────
            exited: set[str] = set()
            if self.cfg.enable_exits:
                for ticker in self.cfg.tickers:
                    if ticker in prices and ticker not in trailing_stopped:
                        sold = self._check_exits(
                            ticker, date, prices[ticker],
                            rsi_map.get(ticker, float("nan")),
                            sma_map.get(ticker, float("nan")),
                        )
                        if sold:
                            exited.add(ticker)

            # ── 3. Quarterly rebalancing ──────────────────────────────────────
            self._check_rebalance(date, prices, rsi_map, sma_map)

            # ── 4. Monthly DCA buys ───────────────────────────────────────────
            skipped_for_rotation: list[str] = []

            for ticker in self.cfg.tickers:
                if ticker not in prices:
                    continue
                if ticker in trailing_stopped or ticker in exited:
                    continue

                # ── Strategy 6: Sector rotation in bear regime ────────────────
                if bear_regime and self.cfg.enable_sector_rotation:
                    def_t = self.cfg.defensive_ticker
                    if def_t in all_data and def_t in prices and def_t != ticker:
                        skipped_for_rotation.append(ticker)
                        continue
                    # defensive_ticker itself — fall through to normal buy below

                # ── Strategy 1: Regime filter — skip buys in bear market ──────
                if bear_regime and self.cfg.enable_regime_filter and not self.cfg.enable_sector_rotation:
                    continue

                # ── Compute base budget + trigger ─────────────────────────────
                budget, trigger = self._budget(
                    rsi_map.get(ticker, float("nan")),
                    prices[ticker],
                    sma_map.get(ticker, float("nan")),
                )

                # ── Strategy 4: Kelly sizing multiplier ───────────────────────
                kelly = self._kelly_multiplier(all_data[ticker], date)

                # ── Strategy 3: Earnings calendar scale ───────────────────────
                earn_scale, earn_suffix = self._earnings_budget_scale(ticker, date, earnings_map)
                if earn_suffix:
                    trigger = earn_suffix

                final_budget = round(budget * kelly * earn_scale, 4)

                self._execute_buy(
                    ticker=ticker,
                    date=date,
                    close=prices[ticker],
                    rsi=rsi_map.get(ticker, float("nan")),
                    sma200=sma_map.get(ticker, float("nan")),
                    forced_budget=final_budget,
                    trigger_override=trigger,
                )

                # Update trailing stop peak after buy
                if self.cfg.enable_trailing_stop:
                    self._highest_close[ticker] = max(
                        self._highest_close.get(ticker, 0.0), prices[ticker]
                    )

            # ── Strategy 6: Execute rotation buys for defensive ticker ────────
            if skipped_for_rotation and self.cfg.enable_sector_rotation:
                def_t = self.cfg.defensive_ticker
                if def_t in all_data and def_t in prices:
                    for skipped in skipped_for_rotation:
                        budget, _ = self._budget(
                            rsi_map.get(skipped, float("nan")),
                            prices[skipped],
                            sma_map.get(skipped, float("nan")),
                        )
                        kelly = self._kelly_multiplier(all_data[def_t], date)
                        self._execute_buy(
                            ticker=def_t,
                            date=date,
                            close=prices[def_t],
                            rsi=rsi_map.get(def_t, float("nan")),
                            sma200=sma_map.get(def_t, float("nan")),
                            forced_budget=round(budget * kelly, 4),
                            trigger_override="ROTATION_BUY",
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

            total_invested  = t.loc[t["action"].isin([
                                "BUY", "DCA_NORMAL",
                                "RSI_OVERSOLD_2X", "RSI_EXTREME_3X", "RSI_CRASH_5X",
                                "BELOW_SMA200_1.5X",
                                "REBALANCE_BUY", "ROTATION_BUY",
                                "EARNINGS_BEAT_BUY", "EARNINGS_PRE_REDUCE",
                            ]), "budget_usd"].sum()
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
