"""
forecast.py - Technical analysis forecast for next trading day.

This is NOT a price prediction. It is a signal-based directional analysis
that produces a probable price RANGE and bias using classical TA indicators.

No model predicts stock prices accurately. This tool helps you understand
the current technical posture of each asset going into the next session.
"""

import numpy as np
import pandas as pd
from pathlib import Path

CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"

# Sentiment thresholds — deliberately conservative so only strong news moves the signal.
# Weak/mixed headlines (|score| < 0.15) stay Neutral and don't distort TA signals.
_SENT_BULL_THRESHOLD =  0.15   # composite VADER score above this → Bullish
_SENT_BEAR_THRESHOLD = -0.15   # composite VADER score below this → Bearish


# -- Indicators ----------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[l == 0] = 100
    return rsi


def _macd(s: pd.Series):
    fast = _ema(s, 12)
    slow = _ema(s, 26)
    macd = fast - slow
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid  = s.rolling(n).mean()
    std  = s.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()


def _stochastic(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# -- Signal scoring ------------------------------------------------------------

def _score_signal(name: str, value, bullish_cond: bool, bearish_cond: bool) -> dict:
    if bullish_cond:
        bias, score = "Bullish", +1
    elif bearish_cond:
        bias, score = "Bearish", -1
    else:
        bias, score = "Neutral", 0
    return {"signal": name, "value": value, "bias": bias, "score": score}


# -- Main forecast function ----------------------------------------------------

def forecast_ticker(
    ticker: str,
    sentiment_score: float | None = None,
    live_price: float | None = None,
) -> dict:
    """
    sentiment_score: optional VADER composite score in [-1, +1] from sentiment.py.
    live_price: optional real-time price from live_feed.py. When provided, overrides
                the parquet last_close so the forecast base is always current — no
                re-download needed to keep forecast and live feed in sync.
    """
    path = CLEAN_DIR / f"{ticker}_clean.parquet"
    if not path.exists():
        return {"error": f"No data for {ticker}"}

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Need at least 200 rows for SMA200
    if len(df) < 200:
        return {"error": f"Insufficient data for {ticker}"}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # Use live price as base when available — keeps forecast in sync with live feed
    # without requiring a manual re-download. Historical indicators (RSI, SMA, etc.)
    # still use the full parquet history for accuracy.
    last_close = live_price if live_price is not None else float(close.iloc[-1])
    prev_close = close.iloc[-2]
    last_date  = df.index[-1].date()

    # -- Compute indicators ----------------------------------------------------

    rsi_14       = _rsi(close, 14).iloc[-1]
    rsi_21       = _rsi(close, 21).iloc[-1]
    sma20        = close.rolling(20).mean().iloc[-1]
    sma50        = close.rolling(50).mean().iloc[-1]
    sma200       = close.rolling(200).mean().iloc[-1]
    ema12        = _ema(close, 12).iloc[-1]
    ema26        = _ema(close, 26).iloc[-1]
    macd_val, macd_sig, macd_hist = _macd(close)
    macd_v       = macd_val.iloc[-1]
    macd_s       = macd_sig.iloc[-1]
    macd_h       = macd_hist.iloc[-1]
    macd_h_prev  = macd_hist.iloc[-2]
    bb_up, bb_mid, bb_low = _bollinger(close, 20)
    bb_upper     = bb_up.iloc[-1]
    bb_lower     = bb_low.iloc[-1]
    bb_mid_v     = bb_mid.iloc[-1]
    atr_14       = _atr(high, low, close, 14).iloc[-1]
    stoch_k, stoch_d = _stochastic(high, low, close)
    sk           = stoch_k.iloc[-1]
    sd           = stoch_d.iloc[-1]
    obv_series   = _obv(close, volume)
    obv_sma      = obv_series.rolling(20).mean()
    obv_trend    = "rising" if obv_series.iloc[-1] > obv_sma.iloc[-1] else "falling"

    # 5-day momentum
    mom_5d       = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    mom_20d      = (close.iloc[-1] / close.iloc[-21] - 1) * 100

    # Volume trend (vs 20-day avg)
    vol_avg      = volume.rolling(20).mean().iloc[-1]
    vol_ratio    = volume.iloc[-1] / vol_avg if vol_avg else 1.0

    # -- Signal scoring --------------------------------------------------------

    signals = [
        _score_signal(
            "RSI(14)",
            f"{rsi_14:.1f}",
            bullish_cond=rsi_14 < 40,    # oversold = bullish setup
            bearish_cond=rsi_14 > 70,    # overbought = bearish setup
        ),
        _score_signal(
            "Price vs SMA200",
            f"${last_close:.2f} vs ${sma200:.2f}",
            bullish_cond=last_close > sma200,
            bearish_cond=last_close < sma200,
        ),
        _score_signal(
            "Price vs SMA50",
            f"${last_close:.2f} vs ${sma50:.2f}",
            bullish_cond=last_close > sma50,
            bearish_cond=last_close < sma50,
        ),
        _score_signal(
            "Price vs SMA20",
            f"${last_close:.2f} vs ${sma20:.2f}",
            bullish_cond=last_close > sma20,
            bearish_cond=last_close < sma20,
        ),
        _score_signal(
            "MACD vs Signal",
            f"{macd_v:.3f} vs {macd_s:.3f}",
            bullish_cond=macd_v > macd_s and macd_h > macd_h_prev,
            bearish_cond=macd_v < macd_s and macd_h < macd_h_prev,
        ),
        _score_signal(
            "Bollinger Band position",
            f"${last_close:.2f} in [${bb_lower:.2f}, ${bb_upper:.2f}]",
            bullish_cond=last_close < bb_lower,    # below lower band = oversold squeeze
            bearish_cond=last_close > bb_upper,    # above upper band = overbought stretch
        ),
        _score_signal(
            "Stochastic K/D",
            f"K={sk:.1f}, D={sd:.1f}",
            bullish_cond=sk < 25 and sk > sd,
            bearish_cond=sk > 80 and sk < sd,
        ),
        _score_signal(
            "5-day momentum",
            f"{mom_5d:+.2f}%",
            bullish_cond=mom_5d > 1.5,
            bearish_cond=mom_5d < -1.5,
        ),
        _score_signal(
            "20-day momentum",
            f"{mom_20d:+.2f}%",
            bullish_cond=mom_20d > 3.0,
            bearish_cond=mom_20d < -3.0,
        ),
        _score_signal(
            "OBV trend (volume flow)",
            obv_trend,
            bullish_cond=obv_trend == "rising",
            bearish_cond=obv_trend == "falling",
        ),
    ]

    # -- Sentiment signal (optional) -------------------------------------------
    # Added last so its absence doesn't change existing signal numbering.
    # Weight = 1 out of N total signals — intentionally low.
    sentiment_signal = None
    if sentiment_score is not None:
        label = (
            f"{sentiment_score:+.3f} (Positive)" if sentiment_score >= _SENT_BULL_THRESHOLD else
            f"{sentiment_score:+.3f} (Negative)" if sentiment_score <= _SENT_BEAR_THRESHOLD else
            f"{sentiment_score:+.3f} (Neutral)"
        )
        sentiment_signal = _score_signal(
            "News Sentiment",
            label,
            bullish_cond=sentiment_score >= _SENT_BULL_THRESHOLD,
            bearish_cond=sentiment_score <= _SENT_BEAR_THRESHOLD,
        )
        signals.append(sentiment_signal)

    total_score   = sum(s["score"] for s in signals)
    bull_count    = sum(1 for s in signals if s["score"] > 0)
    bear_count    = sum(1 for s in signals if s["score"] < 0)
    neutral_count = len(signals) - bull_count - bear_count

    # Overall directional bias
    if total_score >= 3:
        overall_bias = "Bullish"
    elif total_score <= -3:
        overall_bias = "Bearish"
    elif total_score > 0:
        overall_bias = "Mildly Bullish"
    elif total_score < 0:
        overall_bias = "Mildly Bearish"
    else:
        overall_bias = "Neutral"

    # Confidence: how many signals agree with the overall direction
    if total_score == 0:
        n_agree = neutral_count
    elif total_score > 0:
        n_agree = bull_count
    else:
        n_agree = bear_count
    confidence_pct = round((n_agree / len(signals)) * 100)

    # -- Expected range for NEXT trading session -------------------------------
    # Built from ATR (volatility measure) around last close.
    # NOT a prediction — a volatility-based range.
    # Historical intraday range is typically 0.8–1.2x ATR for a single session.

    atr_factor = 1.0
    expected_low  = round(last_close - atr_factor * atr_14, 2)
    expected_high = round(last_close + atr_factor * atr_14, 2)

    # Bias-adjusted midpoint (slight skew toward signal direction)
    skew = (total_score / len(signals)) * atr_14 * 0.5
    expected_mid = round(last_close + skew, 2)

    # Support / resistance
    support_1    = round(bb_lower, 2)
    support_2    = round(sma200, 2)
    resistance_1 = round(bb_upper, 2)
    resistance_2 = round(max(sma50, sma20), 2)

    return {
        "ticker":         ticker,
        "last_close":     round(last_close, 2),
        "last_date":      str(last_date),
        "overall_bias":   overall_bias,
        "total_score":    total_score,
        "confidence_pct": confidence_pct,
        "bull_signals":   bull_count,
        "bear_signals":   bear_count,
        "neutral_signals":neutral_count,
        "expected_low":   expected_low,
        "expected_high":  expected_high,
        "expected_mid":   expected_mid,
        "atr_14":         round(atr_14, 2),
        "rsi_14":         round(rsi_14, 1),
        "rsi_21":         round(rsi_21, 1),
        "sma20":          round(sma20, 2),
        "sma50":          round(sma50, 2),
        "sma200":         round(sma200, 2),
        "macd":           round(macd_v, 4),
        "macd_signal":    round(macd_s, 4),
        "bb_upper":       round(bb_upper, 2),
        "bb_lower":       round(bb_lower, 2),
        "stoch_k":        round(sk, 1),
        "stoch_d":        round(sd, 1),
        "vol_ratio":      round(vol_ratio, 2),
        "mom_5d":         round(mom_5d, 2),
        "mom_20d":        round(mom_20d, 2),
        "support_1":      support_1,
        "support_2":      support_2,
        "resistance_1":   resistance_1,
        "resistance_2":   resistance_2,
        "signals":         signals,
        "sentiment_score": round(sentiment_score, 4) if sentiment_score is not None else None,
        "disclaimer": (
            "This is technical signal analysis, NOT a price prediction. "
            "No model accurately predicts single-day closing prices. "
            "Use this as one of many inputs, never as a sole trading decision."
        ),
    }


def forecast_all(
    tickers=("SPY", "QQQ", "AMD"),
    use_sentiment: bool = True,
    live_prices: dict | None = None,
) -> dict:
    """
    Run forecast for all tickers.
    use_sentiment: fetch live VADER sentiment and include as signal #11.
    live_prices: dict of {ticker: price} from live_feed.fetch_all_live_prices().
                 When provided, each ticker's forecast uses the live price as its
                 base instead of the (potentially stale) parquet last close.
    """
    sentiment_scores: dict[str, float | None] = {}
    if use_sentiment:
        try:
            from sentiment import sentiment_all
            sent = sentiment_all(list(tickers))
            sentiment_scores = {
                t: sent[t]["composite_score"]
                for t in tickers
                if t in sent and "error" not in sent[t]
            }
        except Exception:
            pass   # degrade gracefully — TA signals still work

    return {
        t: forecast_ticker(
            t,
            sentiment_score=sentiment_scores.get(t),
            live_price=live_prices.get(t, {}).get("price") if live_prices else None,
        )
        for t in tickers
    }


# ---------------------------------------------------------------------------
# Weekly (5-trading-day) forecast
# ---------------------------------------------------------------------------

# US market holidays 2024-2030 (fixed + floating).
_US_MARKET_HOLIDAYS = {
    # New Year's Day
    "2024-01-01","2025-01-01","2026-01-01","2027-01-01","2028-01-01","2029-01-01","2030-01-01",
    # MLK Day (3rd Monday Jan)
    "2024-01-15","2025-01-20","2026-01-19","2027-01-18","2028-01-17","2029-01-15","2030-01-21",
    # Presidents' Day (3rd Monday Feb)
    "2024-02-19","2025-02-17","2026-02-16","2027-02-15","2028-02-21","2029-02-19","2030-02-18",
    # Good Friday
    "2024-03-29","2025-04-18","2026-04-03","2027-03-26","2028-04-14","2029-03-30","2030-04-19",
    # Memorial Day (last Monday May)
    "2024-05-27","2025-05-26","2026-05-25","2027-05-31","2028-05-29","2029-05-28","2030-05-27",
    # Juneteenth
    "2024-06-19","2025-06-19","2026-06-19","2027-06-18","2028-06-19","2029-06-19","2030-06-19",
    # Independence Day (observed when on weekend)
    "2024-07-04","2025-07-04","2026-07-03","2027-07-05","2028-07-04","2029-07-04","2030-07-04",
    # Labor Day (1st Monday Sep)
    "2024-09-02","2025-09-01","2026-09-07","2027-09-06","2028-09-04","2029-09-03","2030-09-02",
    # Thanksgiving (4th Thursday Nov)
    "2024-11-28","2025-11-27","2026-11-26","2027-11-25","2028-11-23","2029-11-22","2030-11-28",
    # Christmas (observed when on weekend)
    "2024-12-25","2025-12-25","2026-12-25","2027-12-24","2028-12-25","2029-12-25","2030-12-25",
}


def _next_trading_days(from_date, n: int = 5) -> list:
    """Return next n trading days after from_date, skipping weekends and US holidays."""
    import datetime
    days, current = [], from_date
    while len(days) < n:
        current += datetime.timedelta(days=1)
        if current.weekday() >= 5:
            continue
        if str(current) in _US_MARKET_HOLIDAYS:
            continue
        days.append(current)
    return days


def weekly_forecast(
    ticker: str,
    sentiment_score: float | None = None,
    live_price: float | None = None,
) -> dict:
    """
    Project a 5-trading-day outlook. Range widens by sqrt(t) each day
    (random-walk variance scaling). Bias direction held constant from
    the 11-signal consensus. Returns base_forecast fields plus 'daily' list.
    """
    import datetime
    import math

    base = forecast_ticker(ticker, sentiment_score=sentiment_score, live_price=live_price)
    if "error" in base:
        return base

    last_date    = datetime.date.fromisoformat(base["last_date"])
    trading_days = _next_trading_days(last_date, n=5)

    atr          = base["atr_14"]
    mid0         = base["last_close"]
    skew_per_day = (base["total_score"] / len(base["signals"])) * atr * 0.3

    daily = []
    for i, day in enumerate(trading_days, start=1):
        scale = math.sqrt(i)
        mid   = round(mid0 + skew_per_day * i, 2)
        daily.append({
            "day":   i,
            "date":  str(day),
            "label": day.strftime("%a %b %d"),
            "low":   round(mid - atr * scale, 2),
            "mid":   mid,
            "high":  round(mid + atr * scale, 2),
        })

    return {
        **base,
        "trading_days": [str(d) for d in trading_days],
        "daily":        daily,
        "weekly_note": (
            "Range widens by √t each day (random-walk scaling). "
            "Bias direction held constant from current signal consensus. "
            "This is NOT a day-by-day price prediction."
        ),
    }


def weekly_forecast_all(
    tickers=("SPY", "QQQ", "AMD"),
    use_sentiment: bool = True,
    live_prices: dict | None = None,
) -> dict:
    """Run weekly_forecast for all tickers with optional sentiment + live prices."""
    sentiment_scores: dict[str, float | None] = {}
    if use_sentiment:
        try:
            from sentiment import sentiment_all
            sent = sentiment_all(list(tickers))
            sentiment_scores = {
                t: sent[t]["composite_score"]
                for t in tickers
                if t in sent and "error" not in sent[t]
            }
        except Exception:
            pass

    return {
        t: weekly_forecast(
            t,
            sentiment_score=sentiment_scores.get(t),
            live_price=live_prices.get(t, {}).get("price") if live_prices else None,
        )
        for t in tickers
    }


if __name__ == "__main__":
    results = forecast_all()
    for t, r in results.items():
        print(f"\n{'='*50}")
        if "error" in r:
            print(f"  {t} — ERROR: {r['error']}")
            continue
        print(f"  {t} — {r['overall_bias']} ({r['confidence_pct']}% signals agree)")
        print(f"  Last close : ${r['last_close']} on {r['last_date']}")
        print(f"  Next session range : ${r['expected_low']} — ${r['expected_high']}")
        print(f"  Signals: {r['bull_signals']} bullish / {r['bear_signals']} bearish / {r['neutral_signals']} neutral")
