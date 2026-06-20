"""
paper_trader.py - Virtual DCA bot that runs against live prices without placing real orders.

Allows you to validate the strategy in real-time before committing real capital.
State is persisted across restarts so the paper portfolio accumulates over months.

Strategy parity: delegates all budget/strategy decisions to LongTermDCABot methods
so the paper trader always mirrors the backtest engine exactly — including RSI extreme
tiers, Kelly sizing, earnings filter, regime filter, and ATR trailing stop.

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
from typing import Optional

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

    All strategy decisions (RSI tiers, Kelly sizing, earnings filter,
    regime detection, ATR trailing stop) are delegated to LongTermDCABot
    methods so this always mirrors the backtest engine exactly.
    """

    def __init__(self, cfg):
        from bot import LongTermDCABot
        self.cfg  = cfg
        self._bot = LongTermDCABot(cfg)   # used for strategy method calls only

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._holdings:      dict[str, float]            = {}
        self._cost_basis:    dict[str, float]            = {}
        self._highest_close: dict[str, float]            = {}  # ATR trailing stop peak
        self._last_buy_date: dict[str, Optional[date]]   = {}  # min_hold_days enforcement
        self._load_state()

    # -- Persistence -----------------------------------------------------------

    def _load_state(self):
        if PAPER_STATE.exists():
            try:
                state = json.loads(PAPER_STATE.read_text(encoding="utf-8"))
                self._holdings      = {k: float(v) for k, v in state.get("holdings",       {}).items()}
                self._cost_basis    = {k: float(v) for k, v in state.get("cost_basis",     {}).items()}
                self._highest_close = {k: float(v) for k, v in state.get("highest_close",  {}).items()}
                raw_dates           = state.get("last_buy_date", {})
                self._last_buy_date = {
                    k: date.fromisoformat(v) if v else None
                    for k, v in raw_dates.items()
                }
            except Exception as exc:
                logger.warning("Could not load paper portfolio state: %s", exc)

    def _save_state(self):
        state = {
            "holdings":       self._holdings,
            "cost_basis":     self._cost_basis,
            "highest_close":  self._highest_close,
            "last_buy_date":  {k: v.isoformat() if v else None for k, v in self._last_buy_date.items()},
            "updated":        datetime.now(tz=timezone.utc).isoformat(),
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
        """Load cleaned parquet and return indicators + full DataFrame for strategy methods."""
        path = CLEAN_DIR / f"{ticker}_clean.parquet"
        if not path.exists():
            logger.warning("No cleaned data for %s — cannot compute indicators", ticker)
            return None
        raw = pd.read_parquet(path)
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.sort_index()
        if len(raw) < 200:
            return None

        close = raw["Close"]

        # ATR14 for trailing stop
        atr14 = float("nan")
        if "High" in raw.columns and "Low" in raw.columns:
            prev_c = close.shift(1)
            tr = pd.concat([
                raw["High"] - raw["Low"],
                (raw["High"] - prev_c).abs(),
                (raw["Low"]  - prev_c).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14, min_periods=14).mean()
            atr14 = float(atr_series.iloc[-1])

        return {
            "rsi21":  _rsi(close, self.cfg.rsi_period),
            "sma200": _sma(close, self.cfg.sma_period),
            "atr14":  atr14,
            "df":     raw,   # full DataFrame — passed to Kelly and regime checks
        }

    # -- Core check ------------------------------------------------------------

    def run_check(self, live_prices: dict | None = None, force: bool = False) -> list[dict]:
        """
        Check whether today should trigger paper trades.

        live_prices: {ticker: {"price": float, ...}} from live_feed.py.
                     If None, fetches automatically.
        force:       If True, execute even if today is not the last trading day.

        Returns list of trade dicts recorded this run (empty if no trigger).
        """
        if not force and not _is_last_trading_day_of_month():
            logger.info("Paper trader: not last trading day — skipping")
            return []

        if live_prices is None:
            from live_feed import fetch_all_live_prices
            live_prices = fetch_all_live_prices(self.cfg.tickers)

        today_str = date.today().isoformat()
        today_ts  = pd.Timestamp(date.today(), tz="UTC")
        new_trades: list[dict] = []

        # ── Strategy 1 & 6: Regime detection ─────────────────────────────────
        bear_regime  = False
        need_regime  = self.cfg.enable_regime_filter or self.cfg.enable_sector_rotation
        if need_regime:
            regime_ind = self._get_indicators(self.cfg.regime_ticker)
            if regime_ind is not None:
                regime_sma  = _sma(regime_ind["df"]["Close"], self.cfg.sma_period)
                regime_last = float(regime_ind["df"]["Close"].iloc[-1])
                bear_regime = not np.isnan(regime_sma) and regime_last < regime_sma

        # ── Strategy 3: Earnings calendar — fetch once per run ───────────────
        earnings_map = self._bot._fetch_earnings_dates()

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
            atr14  = ind["atr14"]

            # ── Strategy 5: ATR trailing stop ─────────────────────────────────
            if self.cfg.enable_trailing_stop and self._holdings.get(ticker, 0.0) > 0:
                if not np.isnan(atr14) and atr14 > 0:
                    if price > self._highest_close.get(ticker, 0.0):
                        self._highest_close[ticker] = price
                    last_buy   = self._last_buy_date.get(ticker)
                    hold_days  = (date.today() - last_buy).days if last_buy else self.cfg.min_hold_days
                    peak       = self._highest_close.get(ticker, price)
                    trail_stop = peak - self.cfg.trailing_atr_multiple * atr14
                    if hold_days >= self.cfg.min_hold_days and price < trail_stop:
                        logger.info("PAPER %s: TRAILING_STOP %s @ $%.2f (stop $%.2f)",
                                    today_str, ticker, price, trail_stop)
                        trade = self._paper_sell(ticker, price, today_str, "TRAILING_STOP")
                        new_trades.append(trade)
                        self._highest_close[ticker] = 0.0
                        continue

            # ── Take-profit / stop-loss ────────────────────────────────────────
            if self.cfg.enable_exits and self._holdings.get(ticker, 0.0) > 0:
                avg_cost = self._cost_basis.get(ticker, 0) / self._holdings[ticker]
                gain_pct = (price - avg_cost) / avg_cost if avg_cost else 0
                last_buy  = self._last_buy_date.get(ticker)
                hold_days = (date.today() - last_buy).days if last_buy else self.cfg.min_hold_days
                if hold_days >= self.cfg.min_hold_days:
                    if gain_pct >= self.cfg.take_profit_pct:
                        trade = self._paper_sell(ticker, price, today_str, "TAKE_PROFIT")
                        new_trades.append(trade)
                        continue
                    if gain_pct <= -self.cfg.stop_loss_pct:
                        trade = self._paper_sell(ticker, price, today_str, "STOP_LOSS")
                        new_trades.append(trade)
                        continue

            # ── Strategy 1: Regime filter — skip buys in bear market ──────────
            if bear_regime and self.cfg.enable_regime_filter and not self.cfg.enable_sector_rotation:
                logger.info("PAPER %s: bear regime — skipping buy for %s", today_str, ticker)
                continue

            # ── Strategy 2: Budget via bot._budget() (includes RSI extreme tiers)
            budget, trigger = self._bot._budget(rsi21, price, sma200)

            # ── Strategy 4: Kelly position sizing ─────────────────────────────
            kelly = self._bot._kelly_multiplier(ind["df"], today_ts)

            # ── Strategy 3: Earnings calendar ─────────────────────────────────
            earn_scale, earn_suffix = self._bot._earnings_budget_scale(ticker, today_ts, earnings_map)
            if earn_suffix:
                trigger = earn_suffix

            final_budget = round(budget * kelly * earn_scale, 4)

            # ── Execute paper buy ──────────────────────────────────────────────
            slippage   = price * (self.cfg.slippage_bps / 10_000)
            fill_price = round(price + slippage, 4)
            fee        = round(self.cfg.clearing_fee_usd, 6)
            shares     = round((final_budget - fee) / fill_price, self.cfg.decimal_places)

            # Clear ghost trailing stop peak when opening a fresh position
            if self.cfg.enable_trailing_stop and self._holdings.get(ticker, 0.0) == 0.0:
                self._highest_close[ticker] = 0.0

            self._holdings[ticker]      = round(self._holdings.get(ticker, 0.0) + shares, self.cfg.decimal_places)
            self._cost_basis[ticker]    = round(self._cost_basis.get(ticker, 0.0) + final_budget, 4)
            self._last_buy_date[ticker] = date.today()

            if self.cfg.enable_trailing_stop:
                self._highest_close[ticker] = max(self._highest_close.get(ticker, 0.0), price)

            trade = {
                "date":               today_str,
                "ticker":             ticker,
                "action":             "PAPER_BUY",
                "trigger":            trigger,
                "price":              price,
                "shares":             shares,
                "budget_usd":         final_budget,
                "fee_usd":            fee,
                "cumulative_shares":  self._holdings[ticker],
                "notes":              f"RSI={rsi21:.1f} SMA200={sma200:.2f} Kelly={kelly:.2f} earn={earn_scale:.1f}",
            }
            self._append_trade(trade)
            new_trades.append(trade)
            logger.info(
                "PAPER %s: BUY %s %.6f shares @ $%.2f (trigger=%s kelly=%.2f earn=%.1f)",
                today_str, ticker, shares, fill_price, trigger, kelly, earn_scale,
            )

        self._save_state()
        return new_trades

    def _paper_sell(self, ticker: str, price: float, today_str: str, reason: str) -> dict:
        shares     = self._holdings.get(ticker, 0.0)
        slippage   = price * (self.cfg.slippage_bps / 10_000)
        fill_price = round(price - slippage, 4)
        proceeds   = round(shares * fill_price, 4)
        cost       = self._cost_basis.get(ticker, 0.0)
        pnl        = round(proceeds - cost, 4)

        self._holdings[ticker]      = 0.0
        self._cost_basis[ticker]    = 0.0
        self._last_buy_date[ticker] = None
        self._highest_close[ticker] = 0.0

        trade = {
            "date":               today_str,
            "ticker":             ticker,
            "action":             "PAPER_SELL",
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
        """Return a DataFrame summarising the current paper portfolio."""
        if live_prices is None:
            try:
                from live_feed import fetch_all_live_prices
                live_prices = fetch_all_live_prices(self.cfg.tickers)
            except Exception:
                live_prices = {}

        rows = []
        for ticker in self.cfg.tickers:
            shares     = self._holdings.get(ticker, 0.0)
            cost       = self._cost_basis.get(ticker, 0.0)
            price      = live_prices.get(ticker, {}).get("price", 0)
            mkt_val    = round(shares * price, 2)
            avg_cost   = round(cost / shares, 4) if shares > 0 else 0.0
            unreal_pnl = round(mkt_val - cost, 2)
            pnl_pct    = round(unreal_pnl / cost * 100, 2) if cost > 0 else 0.0
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
        self._holdings      = {}
        self._cost_basis    = {}
        self._highest_close = {}
        self._last_buy_date = {}
        self._save_state()
        if PAPER_LOG_PATH.exists():
            PAPER_LOG_PATH.unlink()
        logger.info("Paper portfolio reset.")
