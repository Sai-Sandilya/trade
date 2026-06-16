r"""
tests/test_metrics.py - Unit tests for src/metrics.py.

Run with:
    .venv\Scripts\python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    annualised_return,
    annualised_volatility,
    compute_all,
    format_metrics,
    TRADING_DAYS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _flat_equity(n=500, value=1000.0) -> pd.Series:
    """Equity curve that never moves — zero return, zero volatility."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(value, index=idx)


def _growing_equity(n=500, start=1000.0, end=2000.0) -> pd.Series:
    """Linearly growing equity curve from start to end over n trading days."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    vals = np.linspace(start, end, n)
    return pd.Series(vals, index=idx)


def _crashing_equity() -> pd.Series:
    """Equity that grows then falls 50% — tests max drawdown."""
    idx = pd.date_range("2015-01-01", periods=600, freq="B", tz="UTC")
    vals = np.concatenate([
        np.linspace(1000, 2000, 300),   # doubles
        np.linspace(2000, 1000, 300),   # halves back
    ])
    return pd.Series(vals, index=idx)


def _empty_equity() -> pd.Series:
    return pd.Series(dtype=float)


def _single_point_equity() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=1, freq="B", tz="UTC")
    return pd.Series([1000.0], index=idx)



# ---------------------------------------------------------------------------
# annualised_return
# ---------------------------------------------------------------------------

class TestAnnualisedReturn:
    def test_doubling_over_one_year(self):
        n = TRADING_DAYS_PER_YEAR
        eq = _growing_equity(n=n, start=1000.0, end=2000.0)
        cagr = annualised_return(eq)
        # 100% return over exactly 1 year
        assert abs(cagr - 1.0) < 0.01

    def test_flat_equity_returns_zero(self):
        cagr = annualised_return(_flat_equity())
        assert abs(cagr) < 1e-6

    def test_empty_series_returns_zero(self):
        assert annualised_return(_empty_equity()) == 0.0

    def test_single_point_returns_zero(self):
        assert annualised_return(_single_point_equity()) == 0.0

    def test_positive_for_growing_equity(self):
        assert annualised_return(_growing_equity()) > 0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_fifty_percent_drawdown(self):
        mdd = max_drawdown(_crashing_equity())
        # Peak = 2000, trough = 1000 → -50%
        assert abs(mdd - (-0.50)) < 0.02

    def test_flat_equity_zero_drawdown(self):
        mdd = max_drawdown(_flat_equity())
        assert mdd == 0.0

    def test_always_negative_or_zero(self):
        mdd = max_drawdown(_crashing_equity())
        assert mdd <= 0.0

    def test_growing_equity_near_zero_drawdown(self):
        # Linearly growing equity has tiny floating-point drawdowns but never >0
        mdd = max_drawdown(_growing_equity())
        assert mdd <= 0.0

    def test_empty_series_returns_zero(self):
        assert max_drawdown(_empty_equity()) == 0.0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_flat_equity_returns_nan(self):
        # Zero volatility → division by zero → NaN
        result = sharpe_ratio(_flat_equity())
        assert np.isnan(result)

    def test_growing_equity_positive_sharpe(self):
        # Consistently rising equity should produce positive Sharpe
        result = sharpe_ratio(_growing_equity())
        assert result > 0

    def test_empty_series_returns_nan(self):
        assert np.isnan(sharpe_ratio(_empty_equity()))

    def test_single_point_returns_nan(self):
        assert np.isnan(sharpe_ratio(_single_point_equity()))

    def test_risk_free_rate_effect(self):
        eq = _growing_equity()
        sharpe_low_rf  = sharpe_ratio(eq, risk_free_rate=0.0)
        sharpe_high_rf = sharpe_ratio(eq, risk_free_rate=0.10)
        # Higher risk-free rate reduces Sharpe
        assert sharpe_low_rf > sharpe_high_rf


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_growing_equity_positive_sortino(self):
        result = sortino_ratio(_growing_equity())
        assert result > 0 or np.isnan(result)  # pure growth may have no downside

    def test_crashing_equity_has_sortino(self):
        result = sortino_ratio(_crashing_equity())
        # Crashing equity has plenty of downside — should return a valid ratio
        assert not np.isnan(result)

    def test_empty_series_returns_nan(self):
        assert np.isnan(sortino_ratio(_empty_equity()))

    def test_identical_negative_returns_not_nan(self):
        # Regression for the downside.std()==0 false-NaN bug:
        # Two identical negative daily returns → std=0 but RMS > 0, so Sortino
        # must be calculable (not NaN).
        idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
        vals = np.ones(100) * 1000.0
        # Insert two identical -2% days
        vals[10] = 980.0
        vals[11:] = 980.0 * np.ones(89)
        vals[20] = 960.4
        vals[21:] = 960.4 * np.ones(79)
        eq = pd.Series(vals, index=idx)
        result = sortino_ratio(eq)
        # Should be a valid number, not NaN
        assert not np.isnan(result)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_positive_for_growing_equity(self):
        eq = _growing_equity()
        result = calmar_ratio(eq)
        # Growing equity with tiny drawdown → very high Calmar or NaN (0 drawdown)
        assert result > 0 or np.isnan(result)

    def test_crashing_equity_has_calmar(self):
        result = calmar_ratio(_crashing_equity())
        assert not np.isnan(result)

    def test_empty_series_returns_nan(self):
        assert np.isnan(calmar_ratio(_empty_equity()))


# ---------------------------------------------------------------------------
# annualised_volatility
# ---------------------------------------------------------------------------

class TestAnnualisedVolatility:
    def test_flat_equity_zero_volatility(self):
        # Flat line → all daily returns are 0 → std = 0
        result = annualised_volatility(_flat_equity())
        assert result == 0.0 or np.isnan(result)

    def test_volatile_equity_positive(self):
        eq = _crashing_equity()
        assert annualised_volatility(eq) > 0

    def test_empty_returns_nan(self):
        assert np.isnan(annualised_volatility(_empty_equity()))


# ---------------------------------------------------------------------------
# compute_all + format_metrics
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_all_keys(self):
        eq = _growing_equity()
        result = compute_all(eq)
        expected_keys = {
            "cagr", "annualised_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate_monthly", "total_return",
        }
        assert set(result.keys()) == expected_keys

    def test_total_return_correct(self):
        eq = _growing_equity(n=500, start=1000.0, end=2000.0)
        result = compute_all(eq)
        assert abs(result["total_return"] - 1.0) < 0.01  # 100% total return

    def test_format_metrics_produces_strings(self):
        eq = _crashing_equity()
        raw    = compute_all(eq)
        pretty = format_metrics(raw)
        for v in pretty.values():
            assert isinstance(v, str)

    def test_format_metrics_nan_shows_na(self):
        raw    = compute_all(_flat_equity())
        pretty = format_metrics(raw)
        # Flat equity → Sharpe is NaN → should show "N/A"
        assert pretty["Sharpe Ratio"] == "N/A"
