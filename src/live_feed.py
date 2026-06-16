"""
live_feed.py - Real-time (delayed) price and market status utilities.

Yahoo Finance free tier delivers data with a 15-20 minute delay and
enforces rate limits — this module is designed around those constraints:
  - fetch_live_price() pulls 1-minute bar history for "1d" to get the
    most recent available price (still delayed, but the freshest yfinance offers)
  - is_market_open() uses US Eastern Time to report session status
  - All functions return gracefully on API failure (return None / empty dict)

These are called from dashboard.py with @st.cache_data TTL caching so the
underlying yfinance API is only hit every 60 seconds for prices and every
5 minutes for news — not on every Streamlit rerun.
"""

import logging
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo  # Python 3.9+

import yfinance as yf

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Regular NYSE/NASDAQ session: 09:30 – 16:00 ET, Mon–Fri
_MARKET_OPEN  = dt_time(9, 30)
_MARKET_CLOSE = dt_time(16, 0)


def is_market_open() -> bool:
    """
    Return True if the US stock market is currently in its regular session
    (09:30–16:00 ET, Monday–Friday).

    Does NOT account for early-close days (day before Thanksgiving, etc.)
    or market holidays — treat the result as an approximation.
    """
    now_et = datetime.now(tz=_ET)
    if now_et.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= now_et.time() < _MARKET_CLOSE


def fetch_live_price(ticker: str) -> dict | None:
    """
    Fetch the most recent available price for *ticker* using yfinance
    1-minute bars over the last trading day.

    Returns a dict with keys:
      price        - last available close (float)
      prev_close   - previous bar's close, or previous day's close (float)
      change       - price - prev_close
      change_pct   - percentage change
      timestamp    - UTC datetime of the bar
      volume       - volume for the last bar

    Returns None if the API call fails or returns no data.

    Note: Yahoo Finance free tier is 15-20 minutes delayed.
    """
    try:
        tkr  = yf.Ticker(ticker)
        # Intraday for latest price; daily for true prev_close (yesterday's close)
        intra = tkr.history(period="1d", interval="1m")
        daily = tkr.history(period="5d", interval="1d")

        if daily.empty:
            return None

        daily = daily.sort_index()
        # prev_close = the most recent completed daily close
        # If market is open today, that's yesterday; if closed, it's the last session
        prev_close = float(daily["Close"].iloc[-2] if len(daily) > 1 else daily["Close"].iloc[-1])

        if not intra.empty:
            intra     = intra.sort_index()
            last_bar  = intra.iloc[-1]
            price     = float(last_bar["Close"])
        else:
            # Market closed — use latest daily bar as the price
            last_bar  = daily.iloc[-1]
            price     = float(last_bar["Close"])
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        ts = last_bar.name
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return {
            "price":      round(price, 4),
            "prev_close": round(prev_close, 4),
            "change":     round(change, 4),
            "change_pct": round(change_pct, 4),
            "timestamp":  ts,
            "volume":     int(last_bar.get("Volume", 0)),
        }

    except Exception as exc:
        logger.warning("fetch_live_price(%s) failed: %s", ticker, exc)
        return None


def fetch_all_live_prices(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch live prices for all tickers.
    Returns a dict keyed by ticker symbol; missing tickers are omitted.
    """
    results = {}
    for t in tickers:
        data = fetch_live_price(t)
        if data is not None:
            results[t] = data
    return results
