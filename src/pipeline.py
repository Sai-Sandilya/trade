"""
pipeline.py — Clean, align, type-enforce, and validate raw OHLCV data.

Zero-loss rules:
  • Missing values are forward-filled then backward-filled — rows are NEVER dropped
    unless a date has no data across ALL tickers (genuine non-trading day).
  • Row-count assertion verifies cleaned output matches raw source.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "cleaned"

PRICE_COLS = ["Open", "High", "Low", "Close"]
VOLUME_COL = "Volume"
SCHEMA: dict[str, np.dtype] = {
    "Open": np.float64,
    "High": np.float64,
    "Low": np.float64,
    "Close": np.float64,
    "Volume": np.int64,
}


def _enforce_schema(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Cast columns to declared types; log any coercion failures."""
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            logger.warning("[%s] Column '%s' missing — inserting NaN column", ticker, col)
            df[col] = np.nan
        try:
            if dtype == np.int64:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        except Exception as exc:
            logger.error("[%s] Schema cast failed for '%s': %s", ticker, col, exc)
            raise
    return df


def _impute(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward-fill then backward-fill; never silently drop rows."""
    missing_before = df[PRICE_COLS].isna().sum().sum()
    df[PRICE_COLS] = df[PRICE_COLS].ffill().bfill()
    missing_after = df[PRICE_COLS].isna().sum().sum()
    if missing_before:
        logger.info(
            "[%s] Imputed %d missing price cells (%d remain after bfill)",
            ticker, missing_before, missing_after,
        )
    return df


def clean_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load raw Parquet, apply schema + imputation, persist to cleaned/.
    Returns the cleaned DataFrame.
    """
    src = RAW_DIR / f"{ticker}.parquet"
    if not src.exists():
        logger.error("[%s] Raw file not found: %s", ticker, src)
        return None

    raw_df = pd.read_parquet(src, engine="pyarrow")
    raw_rows = len(raw_df)

    # Ensure datetime index
    raw_df.index = pd.to_datetime(raw_df.index, utc=True).normalize()
    raw_df.index.name = "Date"
    raw_df = raw_df.sort_index()

    # Keep only known columns; drop extra yfinance columns gracefully
    keep = [c for c in SCHEMA if c in raw_df.columns]
    df = raw_df[keep].copy()

    df = _enforce_schema(df, ticker)
    df = _impute(df, ticker)

    # Zero-loss assertion: cleaned rows must equal raw rows
    assert len(df) == raw_rows, (
        f"[{ticker}] ZERO-LOSS VIOLATED: raw={raw_rows} rows, cleaned={len(df)} rows"
    )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    dest = CLEAN_DIR / f"{ticker}_clean.parquet"
    df.to_parquet(dest, engine="pyarrow", compression="snappy")

    # Read-back verification
    check = pd.read_parquet(dest, engine="pyarrow")
    assert len(check) == raw_rows, (
        f"[{ticker}] Disk write verification FAILED: expected {raw_rows}, got {len(check)}"
    )
    logger.info("[%s] Cleaned %d rows -> %s [OK]", ticker, len(df), dest)
    return df


def clean_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Clean every ticker and return a dict of cleaned DataFrames."""
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = clean_ticker(ticker)
        if df is not None:
            results[ticker] = df
    return results


def load_aligned(tickers: list[str]) -> pd.DataFrame:
    """
    Load all cleaned tickers, outer-join on Date, then ffill/bfill gaps
    introduced by the alignment merge.  The union date index preserves every
    trading day from every asset — zero rows are dropped.
    """
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = CLEAN_DIR / f"{ticker}_clean.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Cleaned file missing: {path}. Run pipeline first.")
        df = pd.read_parquet(path, engine="pyarrow")[["Close", "Volume"]]
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, c) for c in df.columns]
        )
        frames[ticker] = df

    aligned = pd.concat(frames.values(), axis=1, join="outer")
    aligned = aligned.sort_index()

    # Fill gaps created by outer-join (e.g., one asset has a date another doesn't)
    aligned = aligned.ffill().bfill()

    logger.info("Aligned multi-asset frame: %d rows × %d cols", *aligned.shape)
    return aligned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from ingestion import TICKERS
    clean_all(TICKERS)
