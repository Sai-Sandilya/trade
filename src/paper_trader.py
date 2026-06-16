"""
paper_trader.py - Virtual DCA bot that runs against live prices without placing real orders.

Allows you to validate the strategy in real-time before committing real capital.
State is persisted across restarts so the paper portfolio accumulates over months.

Usage:
    from paper_trader import PaperTrader
    from bot import BotConfig

    cfg = BotConfig(tickers=["SPY", "QQQ", "AMD"], monthly_budget_usd=100)
    pt  = PaperTrader(cfg)
    pt.run_check()          # check today; record paper trade if triggered
    print(pt.summary())     # current virtual portfolio
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR       = Path(__file__).resolve().parents[1] / "data"
PAPER_LOG_PATH = DATA_DIR / "paper_trades.csv"
PAPER_STATE    = DATA_DIR / "paper_portfolio.json"
CLEAN_DIR      = DATA_DIR / "cleaned"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rsi(s: pd.Series, n: int = 21) -> float:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs))
    rsi[l == 0] = 100
    return float(rsi.iloc[-1])


def _sma(s: pd.Series, n: int = 200) -> float:
    return float(s.rolling(n).mean().iloc[-1])


def _is_last_trading_day_of_month() -> bool:
    """True if today is the last weekday of the current calendar month."""
    today = date.today()
    if today.weekday() >= 5:
        return False
    # Check if tomorrow (or any day before the 1st of next month) is also a weekday
    from datetime import timedelta
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day.month != today.month


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Simulates the DCA bot using live prices. No real orders are placed.
    On the last trading day of each month it decides whether to buy/sell
    and records the decision to paper_trades.csv.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._holdings: dict[str, float] = {}    # ticker -> virtual shares
        self._cost_basis: dict[str, float] = {}  # ticker -> total cost paid
        self._load_state()

    # -- Persistence -----------------------------------------------------------

    def _load_state(self):
        if PAPER_STATE.exists():
            try:
                state = json.loads(PAPER_STATE.read_text(encoding="utf-8"))
                self._holdings   = {k: float(v) for k, v in state.get("holdings", {}).items()}
                self._cost_basis = {k: float(v) for k, v in state.get("cost_basis", {}).items()}
            except Exception as exc:
                logger.warning("Could not load paper portfolio state: %s", exc)

    def _save_state(self):
        state = {
            "holdings":   self._holdings,
            "cost_basis": self._cost_basis,
            "updated":    datetime.now(tz=timezone.utc).isoformat(),
        }
        PAPER_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_log(self) -> pd.DataFrame:
        if PAPER_LOG_PATH.exists():
            return pd.read_csv(PAPER_LOG_PATH, parse_dates=["date"])
        cols = ["date", "ticker", "action", "trigger", "price", "shares",
                "budget_usd", "fee_usd", "cumulative_shares", "notes"]
        return pd.DataFrame(columns=cols)

    def _append_trade(self, trade: dict):
        log = self._load_log()
        log = pd.concat([log, pd.DataFrame([trade])], ignore_index=True)
        log.to_csv(PAPER_LOG_PATH, index=False)

    # -- Indicators from parquet history ---------------------------------------

    def _get_indicators(self, ticker: str) -> dict | None:
        path = CLEAN_DIR / f"{ticker}_clean.parquet"
        if not path.exists():
            logger.warning("No cleaned data for %s — cannot compute indicators", ticker)
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        if len(df) < 200:
            return None
        return {
            "rsi21":  _rsi(df["Close"], self.cfg.rsi_period),
            "sma200": _sma(df["Close"], self.cfg.sma_period),
        }

    # -- Budget logic (mirrors bot.py) -----------------------------------------

    def _budget(self, rsi: float, price: float, sma200: float) -> tuple[float, str]:
        if not np.isnan(rsi) and rsi < self.cfg.oversold_rsi:
            return self.cfg.monthly_budget_usd * self.cfg.oversold_multiplier, "RSI_OVERSOLD_2X"
        if not np.isnan(sma200) and price < sma200:
            return self.cfg.monthly_budget_usd * self.cfg.below_sma_multiplier, "BELOW_SMA200_1.5X"
        return self.cfg.monthly_budget_usd, "DCA_NORMAL"

    # -- Core check ------------------------------------------------------------

    def run_check(self, live_prices: dict | None = None, force: bool = False) -> list[dict]:
        """
        Check whether today should trigger paper trades.

        live_prices: {ticker: {"price": float, ...}} from live_feed.py.
                     If None, fetches automatically.
        force:       If True, execute even if today is not the last trading day.
                     Useful for testing or manual runs.

        Returns list of trade dicts recorded this run (empty if no trigger).
        """
        if not force and not _is_last_trading_day_of_month():
            logger.info("Paper trader: not last trading day — skipping")
            return []

        if live_prices is None:
            from live_feed import fetch_all_live_prices
            live_prices = fetch_all_live_prices(self.cfg.tickers)

        today     = date.today().isoformat()
        new_trades: list[dict] = []

        for ticker in self.cfg.tickers:
            if ticker not in live_prices:
                logger.warning("No live price for %s — skipping paper trade", ticker)
                continue

            price = live_prices[ticker]["price"]
            ind   = self._get_indicators(ticker)
            if ind is None:
                continue

            rsi21  = ind["rsi21"]
            sma200 = ind["sma200"]

            # -- Exit check (take-profit / stop-loss) --------------------------
            if self.cfg.enable_exits and self._holdings.get(ticker, 0) > 0:
                avg_cost = self._cost_basis.get(ticker, 0) / self._holdings[ticker]
                gain_pct = (price - avg_cost) / avg_cost if avg_cost else 0
                if gain_pct >= self.cfg.take_profit_pct:
                    trade = self._paper_sell(ticker, price, today, "TAKE_PROFIT")
                    new_trades.append(trade)
                    continue
                if gain_pct <= -self.cfg.stop_loss_pct:
                    trade = self._paper_sell(ticker, price, today, "STOP_LOSS")
                    new_trades.append(trade)
                    continue

            # -- Monthly buy ---------------------------------------------------
            budget, trigger = self._budget(rsi21, price, sma200)
            slippage        = price * (self.cfg.slippage_bps / 10_000)
            fill_price      = round(price + slippage, 4)
            fee             = round(self.cfg.clearing_fee_usd, 6)
            shares          = round((budget - fee) / fill_price, self.cfg.decimal_places)

            self._holdings[ticker]   = round(
                self._holdings.get(ticker, 0) + shares, self.cfg.decimal_places
            )
            self._cost_basis[ticker] = round(
                self._cost_basis.get(ticker, 0) + budget, 4
            )

            trade = {
                "date":               today,
                "ticker":             ticker,
                "action":             "PAPER_BUY",
                "trigger":            trigger,
                "price":              price,
                "shares":             shares,
                "budget_usd":         budget,
                "fee_usd":            fee,
                "cumulative_shares":  self._holdings[ticker],
                "notes":              f"RSI={rsi21:.1f} SMA200={sma200:.2f}",
            }
            self._append_trade(trade)
            new_trades.append(trade)
            logger.info("PAPER %s: %s %s shares @ $%.2f (trigger=%s)",
                        today, ticker, shares, fill_price, trigger)

        self._save_state()
        return new_trades

    def _paper_sell(self, ticker: str, price: float, today: str, reason: str) -> dict:
        shares     = self._holdings.get(ticker, 0)
        slippage   = price * (self.cfg.slippage_bps / 10_000)
        fill_price = round(price - slippage, 4)
        proceeds   = round(shares * fill_price, 4)
        cost       = self._cost_basis.get(ticker, 0)
        pnl        = round(proceeds - cost, 4)

        self._holdings[ticker]   = 0.0
        self._cost_basis[ticker] = 0.0

        trade = {
            "date":               today,
            "ticker":             ticker,
            "action":             f"PAPER_SELL",
            "trigger":            reason,
            "price":              price,
            "shares":             -shares,
            "budget_usd":         0,
            "fee_usd":            self.cfg.clearing_fee_usd,
            "cumulative_shares":  0,
            "notes":              f"proceeds=${proceeds:.2f} pnl=${pnl:+.2f}",
        }
        self._append_trade(trade)
        return trade

    # -- Portfolio snapshot ----------------------------------------------------

    def summary(self, live_prices: dict | None = None) -> pd.DataFrame:
        """
        Return a DataFrame summarising the current paper portfolio.
        live_prices: optional dict from live_feed; fetched automatically if None.
        """
        if live_prices is None:
            try:
                from live_feed import fetch_all_live_prices
                live_prices = fetch_all_live_prices(self.cfg.tickers)
            except Exception:
                live_prices = {}

        rows = []
        for ticker in self.cfg.tickers:
            shares    = self._holdings.get(ticker, 0)
            cost      = self._cost_basis.get(ticker, 0)
            price     = live_prices.get(ticker, {}).get("price", 0)
            mkt_val   = round(shares * price, 2)
            avg_cost  = round(cost / shares, 4) if shares > 0 else 0.0
            unreal_pnl = round(mkt_val - cost, 2)
            pnl_pct   = round(unreal_pnl / cost * 100, 2) if cost > 0 else 0.0
            rows.append({
                "ticker":             ticker,
                "shares":             shares,
                "avg_cost_per_share": avg_cost,
                "total_invested_usd": round(cost, 2),
                "live_price":         price,
                "market_value_usd":   mkt_val,
                "unrealized_pnl_usd": unreal_pnl,
                "unrealized_pnl_pct": pnl_pct,
            })
        return pd.DataFrame(rows).set_index("ticker")

    def trade_log(self) -> pd.DataFrame:
        return self._load_log()

    def reset(self):
        """Wipe all paper trades and reset virtual portfolio to zero."""
        self._holdings   = {}
        self._cost_basis = {}
        self._save_state()
        if PAPER_LOG_PATH.exists():
            PAPER_LOG_PATH.unlink()
        logger.info("Paper portfolio reset.")
