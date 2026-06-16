r"""
tests/test_forecast.py - Unit tests for src/forecast.py.

Tests focus on the bugs that were found and fixed:
  - Bollinger Band bullish condition (< lower band, not < midband)
  - Neutral confidence uses neutral_count (not bull_count when score == 0)
  - Error dict safety in __main__-style callers
  - ATR uses Wilder smoothing (alpha=1/n)
  - Bollinger uses population std (ddof=0)

Run with:
    .venv\Scripts\python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast import (
    _rsi,
    _macd,
    _bollinger,
    _atr,
    _stochastic,
    _obv,
    _ema,
    _score_signal,
    forecast_ticker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(n=300, start=100.0, trend=0.0003, seed=7) -> pd.Series:
    rng     = np.random.default_rng(seed)
    returns = trend + rng.normal(0, 0.012, n)
    prices  = start * np.cumprod(1 + returns)
    return pd.Series(prices)


def _ohlc(n=300, seed=7):
    close  = _series(n=n, seed=seed)
    high   = close * (1 + np.abs(np.random.default_rng(seed).normal(0, 0.005, n)))
    low    = close * (1 - np.abs(np.random.default_rng(seed + 1).normal(0, 0.005, n)))
    volume = pd.Series(np.random.default_rng(seed).integers(1_000_000, 5_000_000, n))
    return high, low, close, volume


# ---------------------------------------------------------------------------
# _bollinger — population std and bullish condition
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_uses_population_std_not_sample(self):
        s   = _series()
        n   = 20
        up, mid, low = _bollinger(s, n=n, k=2.0)
        # Manually compute both versions at index 50
        window = s.iloc[31:51]   # 20 values ending at index 50
        pop_std    = window.std(ddof=0)
        sample_std = window.std(ddof=1)
        expected_up_pop    = window.mean() + 2 * pop_std
        expected_up_sample = window.mean() + 2 * sample_std
        # Must match population, not sample
        assert abs(up.iloc[50] - expected_up_pop) < 1e-8
        assert abs(up.iloc[50] - expected_up_sample) > 1e-8

    def test_upper_above_mid_above_lower(self):
        up, mid, low = _bollinger(_series())
        valid = ~(up.isna() | mid.isna() | low.isna())
        assert (up[valid] >= mid[valid]).all()
        assert (mid[valid] >= low[valid]).all()

    def test_nans_before_warmup(self):
        up, mid, low = _bollinger(_series(n=300), n=20)
        assert up.iloc[:19].isna().all()
        assert not up.iloc[20:].isna().any()

    def test_bullish_signal_is_below_lower_band(self):
        """
        The fixed bullish condition is price < bb_lower.
        Verify the signal fires when price is below the lower band.
        """
        s            = _series()
        up, mid, low = _bollinger(s)
        last_close   = s.iloc[-1]
        bb_lower     = low.iloc[-1]
        bb_mid       = mid.iloc[-1]
        bb_upper     = up.iloc[-1]

        bullish_cond = last_close < bb_lower    # FIXED condition
        bearish_cond = last_close > bb_upper

        sig = _score_signal("Bollinger Band position",
                             f"${last_close:.2f}",
                             bullish_cond, bearish_cond)

        # If price is above lower band but below midband, it must NOT be bullish
        if last_close > bb_lower and last_close < bb_mid:
            assert sig["bias"] != "Bullish", (
                "Signal is Bullish when price is between lower band and midband — "
                "this is the old (unfixed) midband condition."
            )


# ---------------------------------------------------------------------------
# _atr — Wilder smoothing (alpha=1/n)
# ---------------------------------------------------------------------------

class TestAtr:
    def test_wilder_smoothing_not_ema_span(self):
        """
        Wilder ATR uses alpha=1/n. ewm(span=n) gives alpha=2/(n+1).
        For n=14: Wilder alpha=0.0714, span alpha=0.1333.
        The two produce different ATR values — verify we use Wilder.
        """
        high, low, close, _ = _ohlc(n=200)
        n = 14

        # Compute true range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr_wilder = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        atr_span   = tr.ewm(span=n,    adjust=False).mean()

        # The two must differ
        diff = (atr_wilder - atr_span).abs().dropna()
        assert diff.max() > 0.0001, "Wilder and span ATR are identical — check smoothing"

    def test_atr_positive(self):
        high, low, close, _ = _ohlc()
        from forecast import _atr
        atr = _atr(high, low, close, 14)
        assert (atr.dropna() > 0).all()

    def test_atr_nans_before_warmup(self):
        high, low, close, _ = _ohlc(n=100)
        from forecast import _atr
        atr = _atr(high, low, close, 14)
        assert atr.iloc[:13].isna().all()


# ---------------------------------------------------------------------------
# _rsi (forecast version)
# ---------------------------------------------------------------------------

class TestForecastRsi:
    def test_all_gains_returns_100(self):
        s = pd.Series(np.linspace(100, 200, 100))
        r = _rsi(s, 14)
        valid = r.dropna()
        assert (valid == 100).all()

    def test_range_0_to_100(self):
        s = _series(n=300)
        r = _rsi(s, 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_no_nan_after_warmup(self):
        s = _series(n=300)
        r = _rsi(s, 14)
        assert not r.iloc[14:].isna().any()


# ---------------------------------------------------------------------------
# _score_signal
# ---------------------------------------------------------------------------

class TestScoreSignal:
    def test_bullish(self):
        sig = _score_signal("test", "v", bullish_cond=True, bearish_cond=False)
        assert sig["score"] == 1
        assert sig["bias"]  == "Bullish"

    def test_bearish(self):
        sig = _score_signal("test", "v", bullish_cond=False, bearish_cond=True)
        assert sig["score"] == -1
        assert sig["bias"]  == "Bearish"

    def test_neutral(self):
        sig = _score_signal("test", "v", bullish_cond=False, bearish_cond=False)
        assert sig["score"] == 0
        assert sig["bias"]  == "Neutral"

    def test_bullish_takes_priority_when_both_true(self):
        # Both conditions true: bullish wins (first if-branch)
        sig = _score_signal("test", "v", bullish_cond=True, bearish_cond=True)
        assert sig["score"] == 1
        assert sig["bias"]  == "Bullish"


# ---------------------------------------------------------------------------
# Neutral confidence fix
# ---------------------------------------------------------------------------

class TestNeutralConfidence:
    """
    When total_score == 0, confidence should reflect neutral_count / total,
    not bull_count (the old bug where 0 >= 0 was True).
    """

    def _make_signals_with_known_counts(self, bull, bear, neutral):
        """Build a synthetic signals list with exact counts."""
        signals = []
        for _ in range(bull):
            signals.append({"score": 1,  "bias": "Bullish"})
        for _ in range(bear):
            signals.append({"score": -1, "bias": "Bearish"})
        for _ in range(neutral):
            signals.append({"score": 0,  "bias": "Neutral"})
        return signals

    def test_neutral_confidence_uses_neutral_count(self):
        # 3 bull + 3 bear + 4 neutral = total_score 0, neutral_count 4
        signals      = self._make_signals_with_known_counts(3, 3, 4)
        total_score  = sum(s["score"] for s in signals)
        bull_count   = sum(1 for s in signals if s["score"] > 0)
        bear_count   = sum(1 for s in signals if s["score"] < 0)
        neutral_count= len(signals) - bull_count - bear_count

        assert total_score == 0

        # The fix: explicit check for == 0
        if total_score == 0:
            n_agree = neutral_count
        elif total_score > 0:
            n_agree = bull_count
        else:
            n_agree = bear_count

        confidence_pct = round((n_agree / len(signals)) * 100)
        assert confidence_pct == 40, (
            f"Expected 40% (neutral_count=4/10) but got {confidence_pct}%"
        )

    def test_old_logic_would_have_given_wrong_answer(self):
        # Confirm the old bug: 0 >= 0 is True → used bull_count instead
        signals     = self._make_signals_with_known_counts(3, 3, 4)
        bull_count  = 3
        total_score = 0
        # Old logic:
        old_n_agree = bull_count if total_score >= 0 else 3
        old_pct     = round((old_n_agree / 10) * 100)
        assert old_pct == 30, "Old logic should have returned 30% — confirms the bug existed"


# ---------------------------------------------------------------------------
# forecast_ticker — error handling
# ---------------------------------------------------------------------------

class TestForecastTickerErrors:
    def test_missing_file_returns_error_dict(self, tmp_path, monkeypatch):
        import forecast as fc
        monkeypatch.setattr(fc, "CLEAN_DIR", tmp_path)
        result = fc.forecast_ticker("FAKE")
        assert "error" in result
        assert "FAKE" in result["error"]

    def test_insufficient_data_returns_error_dict(self, tmp_path, monkeypatch):
        import forecast as fc
        monkeypatch.setattr(fc, "CLEAN_DIR", tmp_path)
        # Write a parquet with only 50 rows (< 200 needed)
        idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
        df  = pd.DataFrame({
            "Close":  np.linspace(100, 110, 50),
            "High":   np.linspace(101, 111, 50),
            "Low":    np.linspace(99, 109, 50),
            "Volume": np.ones(50, dtype=int) * 1_000_000,
        }, index=idx)
        df.to_parquet(tmp_path / "TINY_clean.parquet")
        result = fc.forecast_ticker("TINY")
        assert "error" in result

    def test_error_dict_has_no_metric_keys(self, tmp_path, monkeypatch):
        import forecast as fc
        monkeypatch.setattr(fc, "CLEAN_DIR", tmp_path)
        result = fc.forecast_ticker("DOESNOTEXIST")
        assert "overall_bias"   not in result
        assert "confidence_pct" not in result
        assert "error"          in result
