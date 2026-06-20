"""
scheduler.py — Auto-runner for the monthly DCA pipeline.

Two modes:

  1. Run once (check today):
        python scheduler.py

  2. Run as a daily background loop (checks at 18:00 every day):
        python scheduler.py --loop

  3. Install as a Windows Task Scheduler task (runs daily at 18:00):
        python scheduler.py --install

The task fires on every calendar day at 18:00 but only executes the
pipeline on the last actual trading day of the month (Mon-Fri, not a
known US market holiday). All other days it exits in <1 second.

Why 18:00?
  US markets close at 16:00 ET. By 18:00, Yahoo Finance has published
  the final closing prices, so the data download is complete and accurate.
"""

import argparse
import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

HERE      = Path(__file__).resolve().parent
MAIN_PY   = HERE / "main.py"
PYTHON    = Path(sys.executable)
TASK_NAME = "USStonks_DCA_Monthly"

# Approximate US market holidays (month, day) for the current and next year.
# These are fixed-date holidays only. Floating holidays (Thanksgiving, MLK Day,
# etc.) would need a proper calendar library — this covers the majority of cases.
_FIXED_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (12, 25), # Christmas Day
    # Dec 26 is NOT a fixed holiday — it is only closed when Dec 25 falls on
    # Sunday (observed Monday). Hardcoding it would skip a normal trading day
    # 6 out of every 7 years. Handled dynamically in _is_likely_trading_day().
}


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5  # Mon=0 … Fri=4


def _is_likely_trading_day(d: date) -> bool:
    if not _is_weekday(d):
        return False
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return False
    # Christmas observed: Dec 26 is a market holiday only when Dec 25 is Sunday
    if d.month == 12 and d.day == 26:
        from datetime import date as _date
        if _date(d.year, 12, 25).weekday() == 6:  # 6 = Sunday
            return False
    return True


def _last_trading_day_of_month(year: int, month: int) -> date:
    """Return the last likely trading day of the given year/month."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    candidate = next_month_first - timedelta(days=1)
    while not _is_likely_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def is_last_trading_day_today() -> bool:
    """Return True if today is the last likely trading day of the current month."""
    today = date.today()
    last = _last_trading_day_of_month(today.year, today.month)
    return today == last


def run_pipeline() -> int:
    """
    Execute main.py --skip-ingest with the current Python interpreter.
    Returns the process exit code (0 = success).
    """
    cmd = [str(PYTHON), str(MAIN_PY), "--skip-ingest"]
    logger.info("Running pipeline: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(HERE))
    return result.returncode


def check_and_run() -> None:
    """Check if today is the last trading day; run pipeline if so."""
    today = date.today()
    last  = _last_trading_day_of_month(today.year, today.month)
    logger.info("Today: %s | Last trading day this month: %s", today, last)

    if today == last:
        logger.info("TODAY IS THE LAST TRADING DAY — running pipeline.")
        code = run_pipeline()
        if code == 0:
            logger.info("Pipeline completed successfully.")
        else:
            logger.error("Pipeline exited with code %d.", code)
    else:
        days_left = (last - today).days
        logger.info(
            "Not last trading day yet. %d day(s) until end-of-month run on %s.",
            days_left, last,
        )


def install_windows_task() -> None:
    """
    Create a Windows Task Scheduler task that runs this script daily at 18:00.
    Requires running PowerShell as Administrator (or the task creation will fail).
    """
    python_str  = str(PYTHON)
    script_str  = str(Path(__file__).resolve())
    working_dir = str(HERE)

    # Use schtasks.exe — available on all Windows versions without extra tools
    cmd = [
        "schtasks", "/Create",
        "/TN",  TASK_NAME,
        "/TR",  f'"{python_str}" "{script_str}" --check',
        "/SC",  "DAILY",
        "/ST",  "18:00",
        "/SD",  date.today().strftime("%m/%d/%Y"),
        "/RL",  "HIGHEST",
        "/F",                       # /F = overwrite if task already exists
    ]

    logger.info("Installing Windows task: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\nTask '{TASK_NAME}' installed successfully.")
        print("It will run daily at 18:00 and execute the pipeline on the last")
        print("trading day of each month.")
        print(f"\nTo view it: open Task Scheduler -> Task Scheduler Library -> {TASK_NAME}")
        print(f"To remove:  schtasks /Delete /TN {TASK_NAME} /F")
    else:
        print(f"\nFailed to install task (exit code {result.returncode}).")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("\nTip: run this command in an Administrator PowerShell window.")


def _loop_mode() -> None:
    """Run a daily loop using the schedule library (alternative to Task Scheduler)."""
    try:
        import schedule
        import time
    except ImportError:
        print("The 'schedule' library is not installed.")
        print("Install with:  .venv\\Scripts\\pip install schedule")
        sys.exit(1)

    logger.info("Starting daily loop — will check at 18:00 every day.")
    schedule.every().day.at("18:00").do(check_and_run)

    # Run once immediately on startup so you can verify it works
    check_and_run()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(HERE / "scheduler.log", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="DCA pipeline auto-runner")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Check today's date and run pipeline if it is month-end (default mode).",
    )
    group.add_argument(
        "--loop",
        action="store_true",
        help="Run as a persistent daily loop (checks at 18:00 every day).",
    )
    group.add_argument(
        "--install",
        action="store_true",
        help="Install as a Windows Task Scheduler task (run as Administrator).",
    )
    args = parser.parse_args()

    if args.install:
        install_windows_task()
    elif args.loop:
        _loop_mode()
    else:
        # Default: check once and exit
        check_and_run()
