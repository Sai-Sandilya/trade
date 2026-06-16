# US Stocks DCA Pipeline — Project Overview

Long-term quantitative investment pipeline for SPY, QQQ, and AMD.
Implements Dollar-Cost Averaging with RSI(21) + SMA200 signal overlays,
live market feed, sentiment-augmented forecasting, paper trading, portfolio
comparison, performance history tracking, and Excel/PDF export.

---

## Architecture

```
F:\us_stocks\
├── main.py                      CLI entry point
├── dashboard.py                 Streamlit UI (16 sections)
├── scheduler.py                 Windows Task Scheduler auto-runner
├── audit.py                     Data integrity audit script
├── config.yaml                  All strategy/backtest/portfolio defaults
├── requirements.txt             Python dependencies
├── PROJECT_OVERVIEW.md          This file
│
├── src\
│   ├── ingestion.py             Download OHLCV from Yahoo Finance
│   ├── pipeline.py              Data cleaning and alignment
│   ├── bot.py                   DCA bot engine (backtesting)
│   ├── forecast.py              Technical analysis + sentiment forecast
│   ├── sentiment.py             VADER news sentiment scoring
│   ├── live_feed.py             Real-time price and market status
│   ├── metrics.py               Risk metrics (Sharpe, Sortino, CAGR, etc.)
│   ├── paper_trader.py          Virtual trading against live prices
│   ├── performance_tracker.py   Historical snapshot persistence
│   ├── portfolio_manager.py     Multi-preset comparison runner
│   ├── report_builder.py        Excel + PDF export
│   └── config_loader.py         Loads config.yaml into BotConfig
│
├── tests\
│   ├── test_bot.py              27 unit tests for bot logic
│   ├── test_forecast.py         20 unit tests for forecast signals
│   └── test_metrics.py          28 unit tests for risk metrics
│
└── data\
    ├── raw\                     Raw parquet files from yfinance
    ├── cleaned\                 Cleaned parquet + trade/equity CSVs
    ├── paper_trades.csv         Paper trading log
    ├── paper_portfolio.json     Paper portfolio state
    └── performance_history.parquet  Historical backtest snapshots
```

---

## Data Flow

```
Yahoo Finance API
      │
      ▼
ingestion.py → data/raw/{ticker}.parquet   (0% loss assertion)
      │
      ▼
pipeline.py  → data/cleaned/{ticker}_clean.parquet  (NaN-free)
      │
      ├──► bot.py → trade_log, equity_curve, summary
      │         │
      │         ├──► metrics.py            → Sharpe, Sortino, CAGR, drawdown
      │         ├──► performance_tracker.py → performance_history.parquet
      │         └──► report_builder.py     → .xlsx / .pdf
      │
      ├──► forecast.py ◄── sentiment.py ◄── Yahoo Finance News API
      │         ◄── live_feed.py  (live price override, no extra download)
      │
      ├──► paper_trader.py ◄── live_feed.py  (virtual trades, no real orders)
      │
      └──► portfolio_manager.py → 3x bot.py runs (Conservative/Balanced/Aggressive)
```

---

## src/ingestion.py

Downloads max-history OHLCV data from Yahoo Finance with exponential backoff.

- `_download_with_backoff(ticker, retry)` — retries up to 5 times with 2^n second delays
- `_validate_and_save(df, ticker)` — writes to `data/raw/{ticker}.parquet` then reads back;
  raises if written row count differs from expected (0% data loss guarantee)
- `ingest_all(tickers)` — downloads all tickers sequentially

---

## src/pipeline.py

Cleans raw OHLCV data and aligns multiple tickers.

- `clean_ticker(ticker)` — enforces float64 prices, int64 volume, drops duplicate indices,
  ffill+bfill imputation. Raises if any price column is still NaN after imputation.
- `clean_all(tickers)` — cleans all tickers
- `load_aligned(tickers)` — outer join on date index + ffill, returns single DataFrame

---

## src/bot.py

Core DCA backtesting engine.

**BotConfig fields:**

| Field | Default | Description |
|---|---|---|
| tickers | [SPY,QQQ,AMD] | Tickers to trade |
| monthly_budget_usd | 100.0 | Base monthly investment per ticker |
| oversold_rsi | 35.0 | RSI threshold for oversold multiplier |
| rsi_period | 21 | RSI lookback window |
| sma_period | 200 | SMA trend-filter window |
| below_sma_multiplier | 1.5 | Budget multiplier when price < SMA200 |
| oversold_multiplier | 2.0 | Budget multiplier when RSI oversold |
| slippage_bps | 3.0 | Transaction slippage in basis points |
| clearing_fee_usd | 0.005 | Flat fee per trade |
| min_hold_days | 15 | Min days between buys per ticker |
| decimal_places | 6 | Fractional share precision |
| enable_exits | False | Enable take-profit / stop-loss |
| take_profit_pct | 0.50 | Sell when up 50% from avg cost |
| stop_loss_pct | 0.20 | Sell when down 20% from avg cost |
| enable_rebalance | False | Enable quarterly rebalancing |
| rebalance_threshold_pct | 0.10 | Drift threshold for rebalance trigger |
| rebalance_frequency_months | 3 | Rebalance check interval |

**Key methods:**
- `run()` — date-first chronological loop, enabling cross-ticker rebalancing on same date
- `_budget(rsi, close, sma200)` — returns (budget_usd, trigger_name)
- `_execute_buy/sell(...)` — records trade with slippage, fees, and realized P&L
- `_check_exits(...)` — evaluates take-profit and stop-loss per ticker
- `_check_rebalance(...)` — quarterly rebalance with drift-threshold guard
- `summary(trade_log)` — per-ticker summary (invested, value, realized/unrealized P&L)
- `equity_curve(trade_log)` — daily portfolio value via cumulative shares x price

**Critical bug fixed:** `_check_rebalance()` was using `total_after_sells` (post-sell total)
to size buy orders, causing permanent capital loss each cycle. Fixed to use `total_value`
(pre-sell total) so proceeds are fully redeployed.

---

## src/forecast.py

Next-session technical analysis forecast with optional sentiment signal.

**11 signals scored (each +1 Bull / -1 Bear / 0 Neutral):**

| # | Signal | Bullish | Bearish |
|---|---|---|---|
| 1 | RSI(14) | < 40 | > 70 |
| 2 | Price vs SMA200 | above | below |
| 3 | Price vs SMA50 | above | below |
| 4 | Price vs SMA20 | above | below |
| 5 | MACD vs Signal + histogram | MACD > Signal and rising | MACD < Signal and falling |
| 6 | Bollinger Band | below lower band | above upper band |
| 7 | Stochastic K/D | K < 25 and K > D | K > 80 and K < D |
| 8 | 5-day momentum | > +1.5% | < -1.5% |
| 9 | 20-day momentum | > +3% | < -3% |
| 10 | OBV trend | rising | falling |
| 11 | News Sentiment (VADER) | composite >= +0.15 | composite <= -0.15 |

**Key functions:**
- `forecast_ticker(ticker, sentiment_score=None, live_price=None)` — runs all signals.
  `live_price` overrides parquet last_close so forecast base stays in sync with live feed
  without requiring a re-download.
- `forecast_all(tickers, use_sentiment=True, live_prices=None)` — fetches sentiment, passes
  live prices, returns dict of forecast results per ticker

**Output per ticker:** overall_bias, confidence_pct, expected_low/mid/high (ATR range),
support/resistance levels, all indicator values, per-signal scorecard, sentiment_score.

**Bugs fixed:**
- Bollinger bullish condition was `< bb_mid` -> fixed to `< bb_lower`
- ATR used `ewm(span=n)` -> fixed to `ewm(alpha=1/n)` (Wilder smoothing)
- Bollinger std used `ddof=1` -> fixed to `ddof=0` (population std)
- Neutral confidence used bull_count when score==0 -> fixed to neutral_count
- Error dict items were charted -> added `if "error" in r: continue` guard

---

## src/sentiment.py

News sentiment scoring using VADER + 80-term financial lexicon.

The lexicon patches VADER with domain-specific scores for terms like
`"earnings miss"` (-3.5), `"beats estimates"` (+3.5), `"bankruptcy"` (-4.0),
`"analyst upgrade"` (+2.5), `"guidance cut"` (-3.5) and 75 more.

- `fetch_and_score(ticker, max_articles=20)` — fetches Yahoo Finance news, handles both
  old and new API schema, scores title+summary per article
- `ticker_sentiment(ticker)` -> composite_score, label, positive/negative/neutral counts
- `sentiment_all(tickers)` -> dict keyed by ticker
- `sentiment_vs_price(ticker, clean_dir)` -> DataFrame aligning 30-day price with daily scores

---

## src/live_feed.py

Real-time (15-20 min delayed) price fetching and market status.

- `is_market_open()` — checks US Eastern Time via `ZoneInfo("America/New_York")`,
  returns True if 09:30-16:00 ET, Mon-Fri
- `fetch_live_price(ticker)` — fetches 1-minute intraday bars for latest price AND a
  separate 5-day daily fetch for true `prev_close` (avoids minute-to-minute change bug)
- `fetch_all_live_prices(tickers)` — batch fetch, skips failures gracefully

**Bug fixed:** Original used `hist.iloc[-2]` from the 1-minute bar as `prev_close`, showing
~$0.01 minute-to-minute change instead of the true daily change. Fixed with a separate
`period="5d", interval="1d"` fetch; `prev_close = daily["Close"].iloc[-2]`.

**Note:** Yahoo Finance free tier is 15-20 minute delayed. The dashboard labels this prominently.

---

## src/metrics.py

Portfolio risk and performance metrics.

| Function | Description |
|---|---|
| `sharpe_ratio(equity)` | Annualised excess return / std of daily returns |
| `sortino_ratio(equity)` | Uses RMS of downside returns, not std |
| `max_drawdown(equity)` | Peak-to-trough percentage (negative) |
| `calmar_ratio(equity)` | CAGR / abs(max_drawdown) |
| `annualised_return(equity)` | CAGR from first to last date |
| `annualised_volatility(equity)` | Annualised std of daily returns |
| `win_rate(equity)` | Percentage of months with positive returns |
| `compute_all(equity)` | All metrics in one dict |
| `buy_and_hold_equity(...)` | Benchmark: same monthly budget, first trading day, same fees |

**Bug fixed:** Sortino used `if downside.std() == 0` guard, returning NaN when all
downside returns were identical (std=0 but RMS>0). Removed the guard; RMS is always
positive when any negative returns exist.

---

## src/paper_trader.py

Virtual DCA bot that simulates trades against live prices without placing real orders.

**PaperTrader class:**
- State persisted in `data/paper_portfolio.json` (survives restarts)
- Trade log in `data/paper_trades.csv`
- Same budget/trigger/slippage logic as `LongTermDCABot`
- RSI and SMA computed from cleaned parquet history for signal accuracy

**Key methods:**
- `run_check(live_prices=None, force=False)` — executes virtual buy/sell if today is the
  last trading day of the month. `force=True` bypasses the calendar check (for testing).
- `_paper_sell(ticker, price, today, reason)` — records take-profit or stop-loss exit
- `summary(live_prices=None)` — current virtual portfolio value, avg cost, unrealised P&L
- `trade_log()` — all paper trades as DataFrame
- `reset()` — wipes all state and log

---

## src/performance_tracker.py

Persists portfolio snapshots after each backtest run for historical comparison.

- `save_snapshot(summary, metrics, equity, cfg_label)` — appends per-ticker rows + TOTAL
  aggregate row to `data/performance_history.parquet`
- `load_history()` — loads all snapshots sorted by timestamp
- `monthly_equity_breakdown(equity)` — resamples equity to monthly cadence:
  columns: month, start_value, end_value, change_usd, change_pct
- `portfolio_history_chart_data(history)` — extracts TOTAL rows for value-over-time chart

Auto-called: every successful backtest run in the dashboard saves a snapshot automatically.

---

## src/portfolio_manager.py

Defines preset strategies and runs side-by-side backtests.

**Three presets:**

| Preset | Budget | RSI threshold | Below-SMA mult | RSI-crash mult | Exits |
|---|---|---|---|---|---|
| Conservative | $50/mo | 30 | 1.2x | 1.5x | No |
| Balanced | $100/mo | 35 | 1.5x | 2.0x | No |
| Aggressive | $200/mo | 40 | 2.0x | 3.0x | Yes (TP=40%, SL=15%) |

- `run_comparison(preset_names, tickers, start_date, end_date)` — runs full backtest per
  preset, returns dict of results
- `comparison_equity_table(results)` — merges equity curves for overlay chart
- `comparison_metrics_table(results)` — side-by-side risk metrics table

---

## src/report_builder.py

Exports backtest results to Excel (openpyxl) and PDF (fpdf2).

**Excel (`build_excel_report`):** 5 sheets:
1. Summary — formatted with colour-coded P&L
2. Risk Metrics
3. Trade Log — BUY rows green, SELL rows red
4. Equity Curve — daily values
5. Monthly P&L — green/red by month sign

**PDF (`build_pdf_report`):**
- Title block with strategy label and UTC timestamp
- Per-ticker summary table with colour-coded P&L rows, TOTAL row
- 8 risk metric cards (2x4 grid), green if metric is favourable
- Disclaimer footer

Both return raw bytes for `st.download_button`.

---

## src/config_loader.py

Loads `config.yaml` into a `BotConfig` dataclass.

- `load_config(path)` — `yaml.safe_load`
- `get_bot_config(cfg)` — maps YAML strategy section to BotConfig fields
- `get_risk_free_rate(cfg)` — reads `metrics.risk_free_rate`

---

## dashboard.py

Streamlit web UI. All 16 sections:

1. **Live Market Feed** — 60s page rerun, prices TTL=15min, news TTL=5min.
   Market open/closed badge, ET clock, 15-20 MIN DELAYED warning. Live prices
   passed into `forecast_all()` so forecast base stays in sync with live feed.
2. **Portfolio Summary** — 5 KPI cards: invested, value, P&L, trades, fees.
3. **Risk & Performance Metrics** — 8 metric cards with "What do these mean?" expander.
4. **Strategy Settings** — expandable current-run config.
5. **Equity Curve** — DCA + buy-and-hold benchmark, combined and per-ticker tabs.
6. **Ticker Breakdown + Allocation** — summary table + donut chart.
   Auto-reruns backtest when selected tickers differ from cached data (fixes stale-ticker bug).
7. **Price History + Trade Entry Points** — price line with buy markers by trigger type.
8. **Trade Log** — filterable by ticker and trigger, downloadable.
9. **Trigger Breakdown + Monthly Deployment** — bar charts.
10. **Data Integrity Audit** — 0% data loss check.
11. **Next Session Technical Forecast** — 11-signal scorecard per ticker with sentiment callout.
12. **News Sentiment Analysis** — overview cards, bar chart, price vs sentiment, correlation.
13. **Paper Trading** — virtual portfolio, simulate/reset controls, paper trade log.
14. **Performance History** — portfolio value chart across all runs, monthly P&L table.
15. **Portfolio Comparison** — Conservative/Balanced/Aggressive overlaid equity + metrics.
16. **Export Report** — Excel and PDF download buttons.

---

## scheduler.py

Auto-runner for the monthly DCA pipeline.

- `is_last_trading_day_today()` — checks month-end, excluding weekends and US holidays
- `run_pipeline()` — subprocess call to `main.py --skip-ingest`
- `check_and_run()` — runs at 18:00 ET after markets close
- `install_windows_task()` — registers a Windows Task Scheduler task via `schtasks.exe`
- `_loop_mode()` — persistent daily loop using the `schedule` library

```
python scheduler.py              # check once and exit
python scheduler.py --loop       # persistent daily loop
python scheduler.py --install    # install Windows Task Scheduler task
```

---

## config.yaml

Central configuration. All dashboard sliders default to these values.

```yaml
strategy:          # DCA bot parameters (budget, RSI, SMA, slippage, exits, rebalance)
portfolios:        # Conservative / Balanced / Aggressive preset definitions
backtest:          # tickers, start_date, end_date
metrics:           # risk_free_rate (default 0.04 = 4% US T-bills)
dashboard:         # port (default 8501)
paper_trading:     # enabled flag, log_path, state_path
```

---

## tests/

75 unit tests across 3 files.

| File | Count | Notable coverage |
|---|---|---|
| test_bot.py | 27 | _rsi, _sma, BotConfig defaults, budget multiplier regimes, fee calc, last-trading-day selection, summary valuation |
| test_forecast.py | 20 | Bollinger ddof=0, bullish=below_lower, ATR Wilder smoothing, RSI all-gains=100, neutral confidence fix, error dict safety |
| test_metrics.py | 28 | All 8 metrics, edge cases (empty, flat, single-point, crashing), Sortino NaN regression on identical negative returns |

```
.venv\Scripts\python -m pytest tests\ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Date-first loop in `bot.run()` | Enables cross-ticker rebalancing on the same calendar date |
| RSI/SMA computed on full history before date filter | Prevents indicator warmup distortion at backtest start |
| Fractional shares to 6 decimal places | Matches fractional-share brokerage platforms |
| Slippage on both buy (price+) and sell (price-) | Apples-to-apples comparison with buy-and-hold benchmark |
| Sortino uses RMS not std of downside | std=0 when all downside returns are identical, causing false NaN |
| Live price overrides parquet last_close in forecast | Keeps forecast in sync with live feed without re-download |
| Sentiment weight = 1 of 11 signals | Conservative weighting prevents noisy news from dominating TA |
| VADER threshold +/-0.15 for sentiment signal | Only strong news moves the forecast; weak/mixed stays Neutral |
| Paper trades accumulate month-over-month | Realistic simulation — position builds just like real DCA |
| Performance snapshots auto-saved after every backtest | Enables tracking of how strategy performs over time |
| `_cached_tickers_match()` invalidates session cache | Prevents showing SPY data when QQQ is selected |
