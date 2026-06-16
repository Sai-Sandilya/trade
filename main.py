"""
main.py - Orchestrator: ingest -> clean -> validate -> backtest.

Usage:
    python main.py                    # full pipeline
    python main.py --skip-ingest      # skip download, reuse raw data
    python main.py --start 2015-01-01 --end 2024-12-31
    python main.py --budget 100
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Force stdout to UTF-8 so log symbols don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ingestion import ingest_all, TICKERS
from pipeline import clean_all, load_aligned
from bot import LongTermDCABot, BotConfig

SEP = "=" * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US Stocks Long-Term DCA Pipeline")
    p.add_argument("--skip-ingest", action="store_true", help="Skip raw data download")
    p.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD")
    p.add_argument(
        "--budget", type=float, default=50.0, help="Monthly DCA budget per ticker (USD)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- Step 1: Ingestion -----------------------------------------------------
    if not args.skip_ingest:
        logger.info(SEP)
        logger.info("STEP 1 - Data Ingestion")
        logger.info(SEP)
        raw_paths = ingest_all(TICKERS)
        for ticker, path in raw_paths.items():
            logger.info("  [OK] %s -> %s", ticker, path)
    else:
        logger.info("STEP 1 - Skipping ingestion (--skip-ingest)")

    # -- Step 2: Cleaning & Validation -----------------------------------------
    logger.info(SEP)
    logger.info("STEP 2 - Cleaning & Zero-Loss Validation")
    logger.info(SEP)
    cleaned = clean_all(TICKERS)
    if len(cleaned) != len(TICKERS):
        missing = set(TICKERS) - set(cleaned.keys())
        logger.error("Cleaning failed for: %s. Aborting.", missing)
        sys.exit(1)
    for ticker, df in cleaned.items():
        logger.info(
            "  [OK] %s - %d rows, date range: %s -> %s",
            ticker, len(df), df.index.min().date(), df.index.max().date(),
        )

    # -- Step 3: Alignment check -----------------------------------------------
    logger.info(SEP)
    logger.info("STEP 3 - Multi-Asset Alignment")
    logger.info(SEP)
    aligned = load_aligned(TICKERS)
    logger.info("  Aligned frame shape: %s", aligned.shape)

    # -- Step 4: Backtest ------------------------------------------------------
    logger.info(SEP)
    logger.info("STEP 4 - Long-Term DCA Bot Backtest (monthly, min hold >15 days)")
    logger.info(SEP)
    cfg = BotConfig(
        tickers=TICKERS,
        monthly_budget_usd=args.budget,
        start_date=args.start,
        end_date=args.end,
    )
    bot = LongTermDCABot(cfg)
    trade_log = bot.run()

    # -- Step 5: Results -------------------------------------------------------
    logger.info(SEP)
    logger.info("STEP 5 - Results")
    logger.info(SEP)

    summary = bot.summary(trade_log)
    equity = bot.equity_curve(trade_log)

    print("\n" + SEP)
    print("  TRADE LOG (last 15 rows)")
    print(SEP)
    print(trade_log.tail(15).to_string(index=False))

    print("\n" + SEP)
    print("  PORTFOLIO SUMMARY")
    print(SEP)
    print(summary.to_string())

    print("\n" + SEP)
    print("  EQUITY CURVE (last 10 rows)")
    print(SEP)
    print(equity.tail(10).to_string())

    total_invested = summary["total_invested_usd"].sum()
    total_value = summary["market_value_usd"].sum()
    total_pnl = total_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

    print(f"\n  Total Invested   : ${total_invested:,.2f}")
    print(f"  Portfolio Value  : ${total_value:,.2f}")
    print(f"  Unrealised P&L   : ${total_pnl:,.2f}  ({total_pnl_pct:.2f}%)")
    print(f"  Total Fees Paid  : ${bot.total_fees:,.4f}")
    print(f"  Total Trades     : {len(trade_log)}")
    print(SEP + "\n")

    # Persist outputs
    out_dir = Path(__file__).parent / "data" / "cleaned"
    trade_log.to_csv(out_dir / "trade_log.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    equity.to_csv(out_dir / "equity_curve.csv")
    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
