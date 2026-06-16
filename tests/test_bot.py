r"""
tests/test_bot.py - Unit tests for src/bot.py logic.

Tests focus on the logic that had real bugs found during review:
  - Indicator warmup (RSI/SMA computed before date filter)
  - Budget multiplier regime selection
  - Fee calculation (no double-counting)
  - Monthly last-trading-day selection
  - RSI edge case (all-gains window → 100 not NaN)

Run with:
    .venv\Scripts\python -m pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot import BotConfig, LongTermDCABot, _rsi, _sma


# ---------------------------------------------------------------------------
# Helper: build a synthetic price series
# ---------------------------------------------------------------------------

def _price_series(n=500, start=100.0, trend=0.0005, seed=42) -> pd.Series:
    """
    Synthetic daily close prices with a small upward trend and noise.
    trend: daily drift (0.0005 = ~12.6% per year).
    """
    rng = np.random.default_rng(seed)
    returns = trend + rng.normal(0, 0.01, n)
    prices  = start * np.cumprod(1 + returns)
    idx     = pd.date_range("2010-01-04", periods=n, freq="B", tz="UTC")
    return pd.Series(prices, index=idx)


def _make_df(n=500, **kwargs) -> pd.DataFrame:
    close  = _price_series(n=n, **kwargs)
    volume = pd.Series(np.random.default_rng(0).integers(1_000_000, 5_000_000, n), index=close.index)
    return pd.DataFrame({"Close": close, "Volume": volume})


# ---------------------------------------------------------------------------
# _rsi helper
# ---------------------------------------------------------------------------

class TestRsiHelper:
    def test_rsi_range_0_100(self):
        s = _price_series(n=300)
        r = _rsi(s, 21)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains_returns_100(self):
        # Strictly increasing series → avg_loss = 0 → RSI must be 100, not NaN
        s = pd.Series(np.linspace(100, 200, 100))
        r = _rsi(s, 14)
        valid = r.dropna()
        assert (valid == 100).all()

    def test_rsi_all_losses_returns_0(self):
        # Strictly decreasing series → avg_gain = 0 → RSI should be 0
        s = pd.Series(np.linspace(200, 100, 100))
        r = _rsi(s, 14)
        valid = r.dropna()
        assert (valid <= 1).all()  # approaches 0

    def test_rsi_has_nans_before_warmup(self):
        s = _price_series(n=300)
        r = _rsi(s, 21)
        # First 21 values should be NaN (warmup)
        assert r.iloc[:21].isna().all()
        assert not r.iloc[22:].isna().any()

    def test_rsi_no_nans_after_warmup(self):
        s = _price_series(n=500)
        r = _rsi(s, 21)
        assert not r.iloc[21:].isna().any()


# ---------------------------------------------------------------------------
# _sma helper
# ---------------------------------------------------------------------------

class TestSmaHelper:
    def test_sma_nans_before_period(self):
        s = _price_series(n=300)
        sma = _sma(s, 200)
        assert sma.iloc[:199].isna().all()
        assert not sma.iloc[200:].isna().any()

    def test_sma_value_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = _sma(s, 3)
        assert abs(sma.iloc[2] - 2.0) < 1e-10
        assert abs(sma.iloc[3] - 3.0) < 1e-10
        assert abs(sma.iloc[4] - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# BotConfig defaults
# ---------------------------------------------------------------------------

class TestBotConfig:
    def test_default_values(self):
        cfg = BotConfig()
        assert cfg.monthly_budget_usd == 50.0
        assert cfg.oversold_rsi == 35.0
        assert cfg.rsi_period == 21
        assert cfg.sma_period == 200
        assert cfg.below_sma_multiplier == 1.5
        assert cfg.oversold_multiplier == 2.0
        assert cfg.slippage_bps == 3.0
        assert cfg.clearing_fee_usd == 0.005
        assert cfg.min_hold_days == 15
        assert cfg.decimal_places == 6

    def test_custom_values(self):
        cfg = BotConfig(monthly_budget_usd=200.0, oversold_rsi=30.0)
        assert cfg.monthly_budget_usd == 200.0
        assert cfg.oversold_rsi == 30.0


# ---------------------------------------------------------------------------
# LongTermDCABot._budget
# ---------------------------------------------------------------------------

class TestBudgetMultiplier:
    def setup_method(self):
        self.cfg = BotConfig(
            monthly_budget_usd=100.0,
            oversold_rsi=35.0,
            below_sma_multiplier=1.5,
            oversold_multiplier=2.0,
        )
        self.bot = LongTermDCABot(self.cfg)

    def test_normal_regime(self):
        budget, trigger = self.bot._budget(rsi=50.0, close=110.0, sma200=100.0)
        assert budget == 100.0
        assert trigger == "DCA_NORMAL"

    def test_below_sma200_regime(self):
        budget, trigger = self.bot._budget(rsi=50.0, close=90.0, sma200=100.0)
        assert budget == 150.0
        assert trigger == "BELOW_SMA200_1.5X"

    def test_rsi_oversold_regime(self):
        budget, trigger = self.bot._budget(rsi=30.0, close=90.0, sma200=100.0)
        assert budget == 200.0
        assert trigger == "RSI_OVERSOLD_2X"

    def test_rsi_oversold_takes_priority_over_below_sma(self):
        # Both conditions true: RSI crash wins
        budget, trigger = self.bot._budget(rsi=20.0, close=80.0, sma200=100.0)
        assert trigger == "RSI_OVERSOLD_2X"
        assert budget == 200.0

    def test_rsi_nan_falls_back_to_sma_check(self):
        # NaN RSI should not crash; SMA check still applies
        budget, trigger = self.bot._budget(rsi=float("nan"), close=80.0, sma200=100.0)
        assert trigger == "BELOW_SMA200_1.5X"

    def test_rsi_nan_and_above_sma_is_normal(self):
        budget, trigger = self.bot._budget(rsi=float("nan"), close=110.0, sma200=100.0)
        assert trigger == "DCA_NORMAL"

    def test_sma_nan_with_normal_rsi(self):
        # NaN SMA shouldn't crash; normal DCA
        budget, trigger = self.bot._budget(rsi=50.0, close=110.0, sma200=float("nan"))
        assert trigger == "DCA_NORMAL"

    def test_exact_rsi_threshold_is_not_oversold(self):
        # oversold_rsi=35 means RSI < 35, not <=
        budget, trigger = self.bot._budget(rsi=35.0, close=90.0, sma200=100.0)
        assert trigger == "BELOW_SMA200_1.5X"

    def test_exact_sma_boundary_not_below(self):
        # close == sma200: not strictly below, so normal
        budget, trigger = self.bot._budget(rsi=50.0, close=100.0, sma200=100.0)
        assert trigger == "DCA_NORMAL"


# ---------------------------------------------------------------------------
# LongTermDCABot._execute_buy — fee calculation
# ---------------------------------------------------------------------------

class TestExecuteBuy:
    def setup_method(self):
        self.cfg = BotConfig(
            tickers=["SPY"],
            monthly_budget_usd=100.0,
            oversold_rsi=35.0,
            slippage_bps=10.0,      # 10 bps = 0.10% for easy math
            clearing_fee_usd=0.0,   # isolate slippage in tests
        )
        self.bot = LongTermDCABot(self.cfg)

    def test_fill_price_includes_slippage(self):
        date  = pd.Timestamp("2020-01-31", tz="UTC")
        trade = self.bot._execute_buy("SPY", date, close=100.0, rsi=50.0, sma200=90.0)
        expected_fill = 100.0 + 100.0 * (10.0 / 10_000)  # 100.10
        assert abs(trade["fill_price"] - expected_fill) < 0.001

    def test_no_double_counting_in_fee(self):
        # Fee = shares × slippage_cost (not shares × fill_price × bps)
        # If double-counted: fee would be shares × (close + slippage_cost) × bps
        date  = pd.Timestamp("2020-01-31", tz="UTC")
        trade = self.bot._execute_buy("SPY", date, close=100.0, rsi=50.0, sma200=90.0)
        slippage_cost   = 100.0 * (10.0 / 10_000)       # 0.10
        expected_fee    = trade["shares_transacted"] * slippage_cost
        double_fee      = trade["shares_transacted"] * trade["fill_price"] * (10.0 / 10_000)
        assert abs(trade["fee_usd"] - expected_fee) < 1e-5   # allow for 6-dp rounding in shares
        assert abs(trade["fee_usd"] - double_fee) > 1e-4   # must NOT equal double-count

    def test_shares_computed_to_6_decimal_places(self):
        date  = pd.Timestamp("2020-01-31", tz="UTC")
        trade = self.bot._execute_buy("SPY", date, close=100.0, rsi=50.0, sma200=90.0)
        s = str(trade["shares_transacted"])
        decimal_part = s.split(".")[-1] if "." in s else ""
        assert len(decimal_part) <= 6

    def test_cumulative_shares_accumulate(self):
        cfg = BotConfig(tickers=["SPY"], monthly_budget_usd=100.0, slippage_bps=0.0, clearing_fee_usd=0.0)
        bot = LongTermDCABot(cfg)
        d1 = pd.Timestamp("2020-01-31", tz="UTC")
        d2 = pd.Timestamp("2020-02-28", tz="UTC")
        t1 = bot._execute_buy("SPY", d1, close=100.0, rsi=50.0, sma200=90.0)
        t2 = bot._execute_buy("SPY", d2, close=100.0, rsi=50.0, sma200=90.0)
        assert t2["cumulative_shares"] == round(t1["shares_transacted"] + t2["shares_transacted"], 6)

    def test_trade_action_is_buy(self):
        date  = pd.Timestamp("2020-01-31", tz="UTC")
        trade = self.bot._execute_buy("SPY", date, close=100.0, rsi=50.0, sma200=90.0)
        assert trade["action"] == "BUY"

    def test_trigger_recorded_in_trade(self):
        date  = pd.Timestamp("2020-01-31", tz="UTC")
        # RSI below threshold → RSI_OVERSOLD_2X
        trade = self.bot._execute_buy("SPY", date, close=100.0, rsi=20.0, sma200=200.0)
        assert trade["trigger"] == "RSI_OVERSOLD_2X"


# ---------------------------------------------------------------------------
# LongTermDCABot.run — last-trading-day selection
# ---------------------------------------------------------------------------

class TestLastTradingDay:
    """
    Verify that the monthly last-trading-day logic selects the actual
    last day with data in each month, not a calendar month-end date.
    We mock _load() to inject a controlled DataFrame.
    """

    def _make_mock_df(self) -> pd.DataFrame:
        """
        Build a DataFrame with 3 months of daily data where the last
        trading day of January is the 31st, February 28th, March 29th.
        """
        dates = pd.date_range("2020-01-02", "2020-03-31", freq="B", tz="UTC")
        df = pd.DataFrame(
            {"Close": 100.0, "Volume": 1_000_000},
            index=dates,
        )
        df["RSI"]   = 50.0
        df["SMA200"] = 90.0
        return df

    def test_run_produces_three_monthly_trades(self):
        cfg = BotConfig(
            tickers=["SPY"],
            monthly_budget_usd=100.0,
            slippage_bps=0.0,
            clearing_fee_usd=0.0,
        )
        bot = LongTermDCABot(cfg)
        mock_df = self._make_mock_df()

        with patch.object(bot, "_load", return_value=mock_df):
            trade_log = bot.run()

        assert len(trade_log) == 3

    def test_trade_dates_are_actual_last_trading_days(self):
        cfg = BotConfig(
            tickers=["SPY"],
            monthly_budget_usd=100.0,
            slippage_bps=0.0,
            clearing_fee_usd=0.0,
        )
        bot = LongTermDCABot(cfg)
        mock_df = self._make_mock_df()

        with patch.object(bot, "_load", return_value=mock_df):
            trade_log = bot.run()

        dates = pd.to_datetime(trade_log["date"]).dt.date.tolist()
        # Jan 31 2020 is Friday, Feb 28 is Friday, Mar 31 is Tuesday
        import datetime
        assert dates[0] == datetime.date(2020, 1, 31)
        assert dates[1] == datetime.date(2020, 2, 28)
        assert dates[2] == datetime.date(2020, 3, 31)


# ---------------------------------------------------------------------------
# LongTermDCABot — summary and equity_curve use end_date price
# ---------------------------------------------------------------------------

class TestSummaryUsesEndDate:
    """
    Verify that summary() values the portfolio at end_date, not today.
    We inject a two-period DataFrame: a low price at end_date and a high
    price after it. If end_date is respected, market_value uses the low price.
    """

    def _make_history(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-02", "2020-12-31", freq="B", tz="UTC")
        prices = np.where(idx <= pd.Timestamp("2020-06-30", tz="UTC"), 100.0, 500.0)
        df = pd.DataFrame({"Close": prices, "Volume": 1_000_000}, index=idx)
        df["RSI"]    = 50.0
        df["SMA200"] = 90.0
        return df

    def test_market_value_at_end_date_not_latest(self):
        cfg = BotConfig(
            tickers=["SPY"],
            monthly_budget_usd=100.0,
            slippage_bps=0.0,
            clearing_fee_usd=0.0,
            end_date="2020-06-30",
        )
        bot = LongTermDCABot(cfg)
        history = self._make_history()

        # run() uses _load(full=False) → sliced to end_date
        run_df    = history[history.index <= pd.Timestamp("2020-06-30", tz="UTC")]
        full_df   = history   # summary uses _load(full=True) then clips manually

        with patch.object(bot, "_load", side_effect=lambda t, full=False: full_df if full else run_df):
            trade_log = bot.run()
            summary   = bot.summary(trade_log)

        if not summary.empty and "SPY" in summary.index:
            last_close = summary.loc["SPY", "last_close"]
            # Should be 100.0 (end_date price), not 500.0 (post-end_date price)
            assert abs(last_close - 100.0) < 0.01, (
                f"Expected 100.0 (end_date price) but got {last_close} "
                f"— summary is using post-end_date prices (time-travel bug)"
            )
