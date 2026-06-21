"""
ingestion.py — Download and persist raw OHLCV data with 0% loss guarantees.

Handles API rate limits via exponential backoff, verifies row-count integrity
before finalising each file, and saves to Parquet in data/raw/.
"""

import logging
import time
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TICKERS: list[str] = ["SPY", "QQQ", "AMD"]
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds; doubles each retry


def _download_with_backoff(ticker: str, retry: int = 0) -> Optional[pd.DataFrame]:
    """Fetch max-history daily OHLCV for *ticker* with exponential backoff."""
    try:
        logger.info("Fetching %s (attempt %d)", ticker, retry + 1)
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", interval="1d", auto_adjust=True, actions=False)
        if df.empty:
            raise ValueError(f"Empty response for {ticker}")
        return df
    except Exception as exc:
        if retry >= MAX_RETRIES:
            logger.error("All retries exhausted for %s: %s", ticker, exc)
            raise
        sleep_secs = BACKOFF_BASE ** retry
        logger.warning("Fetch failed for %s (%s). Retrying in %.1fs…", ticker, exc, sleep_secs)
        time.sleep(sleep_secs)
        return _download_with_backoff(ticker, retry + 1)


def _validate_and_save(df: pd.DataFrame, ticker: str) -> Path:
    """
    Persist *df* to Parquet and assert file row-count integrity.

    Raises AssertionError if the written file cannot be read back with the
    same number of rows — guaranteeing 0% silent data loss on disk.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{ticker}.parquet"

    # Normalise index name for consistency
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index, utc=True).normalize()

    expected_rows = len(df)
    df.to_parquet(dest, engine="pyarrow", compression="snappy")

    # Read back and verify
    written = pd.read_parquet(dest, engine="pyarrow")
    assert len(written) == expected_rows, (
        f"[{ticker}] Integrity check FAILED: wrote {expected_rows} rows, "
        f"read back {len(written)} rows from {dest}"
    )
    logger.info("[%s] Saved %d rows -> %s [OK]", ticker, expected_rows, dest)
    return dest


def _is_parquet_valid(path: Path) -> bool:
    """Return True if the file exists and can be read as a valid parquet file."""
    if not path.exists():
        return False
    try:
        pd.read_parquet(path, engine="pyarrow")
        return True
    except Exception:
        return False


def ingest_all(tickers: list[str] = TICKERS) -> dict[str, Path]:
    """Download and persist all *tickers*. Returns mapping of ticker → file path."""
    paths: dict[str, Path] = {}
    for ticker in tickers:
        raw_path = RAW_DIR / f"{ticker}.parquet"
        # Delete corrupt file before attempting download so the write-verify
        # loop starts clean rather than reading stale corrupt bytes.
        if raw_path.exists() and not _is_parquet_valid(raw_path):
            logger.warning("[%s] Corrupt parquet detected — deleting and re-downloading", ticker)
            raw_path.unlink()
        df = _download_with_backoff(ticker)
        path = _validate_and_save(df, ticker)
        paths[ticker] = path
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_all()
