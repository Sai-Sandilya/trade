# US Stocks DCA Pipeline

A quantitative Dollar-Cost Averaging (DCA) investment pipeline for **SPY**, **QQQ**, and **AMD** — with live market feed, sentiment-augmented forecasting, paper trading, portfolio comparison, and Excel/PDF reporting.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Past performance does not guarantee future results. Do not make real investment decisions based solely on this tool.

---

## Features

- **Smart DCA bot** — monthly buys with RSI(21) + SMA200 signal multipliers (1.5x–3x on crashes)
- **Live market feed** — 15-20 min delayed prices and 5-min refreshing news sentiment
- **11-signal technical forecast** — TA signals + VADER news sentiment, with ATR-based price range
- **Paper trading** — simulate the bot against live prices without placing real orders
- **Portfolio comparison** — run Conservative / Balanced / Aggressive presets side-by-side
- **Performance history** — snapshots saved after every backtest, with month-by-month P&L
- **Excel + PDF export** — styled 5-sheet workbook and metric-card PDF
- **75 unit tests** covering bot logic, forecast signals, and risk metrics

---

## Requirements

- Python 3.11 or later
- Windows, macOS, or Linux
- Internet connection (Yahoo Finance API — free, no key required)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sai-Sandilya/trade.git
cd trade
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download historical data

```bash
python main.py --download
```

This fetches max-history OHLCV data for SPY, QQQ, and AMD from Yahoo Finance and saves it to `data/raw/`.

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dashboard Walkthrough

The dashboard has 16 sections, all accessible from the sidebar:

| Section | Description |
|---|---|
| Live Market Feed | Delayed prices + news sentiment, auto-refreshes every 60s |
| Portfolio Summary | Total invested, current value, P&L, trade count, fees paid |
| Risk & Performance | Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Volatility, Win Rate |
| Strategy Settings | View the current backtest configuration |
| Equity Curve | DCA vs buy-and-hold benchmark, per-ticker breakdown |
| Ticker Breakdown | Summary table + portfolio allocation donut chart |
| Price History | OHLCV chart with buy entry points colour-coded by trigger |
| Trade Log | Filterable log of every trade, downloadable as CSV |
| Trigger Analysis | Bar charts of buy trigger types and monthly cash deployed |
| Data Integrity | 0% data loss audit across all tickers |
| Technical Forecast | 11-signal scorecard with next-session price range estimate |
| Sentiment Analysis | News sentiment scores, price vs sentiment chart, headline table |
| Paper Trading | Simulate trades against live prices, view virtual portfolio |
| Performance History | Portfolio value chart across all backtest runs |
| Portfolio Comparison | Conservative / Balanced / Aggressive side-by-side |
| Export Report | Download Excel workbook or PDF report |

---

## CLI Usage

### Run a full backtest (no data download)

```bash
python main.py
```

### Re-download data then run

```bash
python main.py --download
```

### Override tickers

```bash
python main.py --tickers SPY QQQ
```

### Override date range

```bash
python main.py --start 2018-01-01 --end 2024-12-31
```

### Override monthly budget

```bash
python main.py --budget 150
```

---

## Configuration

All strategy defaults live in [`config.yaml`](config.yaml). Edit this file to persist your preferred settings — the dashboard and CLI both read from it.

```yaml
strategy:
  monthly_budget_usd:     100.0   # Base investment per ticker per month
  oversold_rsi:           35.0    # RSI(21) below this triggers 2x buy
  below_sma_multiplier:   1.5     # Budget multiplier when price < SMA200
  oversold_multiplier:    2.0     # Budget multiplier when RSI oversold
  enable_exits:           false   # Take-profit / stop-loss
  take_profit_pct:        0.50    # Sell when up 50% from avg cost
  stop_loss_pct:          0.20    # Sell when down 20% from avg cost
  enable_rebalance:       false   # Quarterly rebalancing
```

---

## Portfolio Presets

Three built-in strategies available in the **Portfolio Comparison** section:

| Preset | Monthly Budget | RSI Trigger | Below-SMA Multiplier | RSI-Crash Multiplier | Exits |
|---|---|---|---|---|---|
| Conservative | $50 | < 30 | 1.2x | 1.5x | No |
| Balanced | $100 | < 35 | 1.5x | 2.0x | No |
| Aggressive | $200 | < 40 | 2.0x | 3.0x | Yes |

---

## Paper Trading

Paper trading simulates the bot logic against live (delayed) prices without touching real money.

- State is saved to `data/paper_portfolio.json` and persists between sessions
- Trade log is saved to `data/paper_trades.csv`
- Use the **Simulate Today's Trade** button in the dashboard to run a manual check
- Use **Reset Paper Portfolio** to start fresh

To enable automatic paper trades on the last trading day of each month, set `paper_trading.enabled: true` in `config.yaml`.

---

## Automated Scheduling (Windows)

The scheduler runs the pipeline automatically on the last trading day of each month at 18:00 ET (after markets close).

```powershell
# Install the Windows Task Scheduler task
python scheduler.py --install

# Or run in a persistent loop (checks daily at 18:00)
python scheduler.py --loop

# One-time check
python scheduler.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

75 tests across 3 files:

```
tests/test_bot.py        — 27 tests (bot logic, budget multipliers, fee calculation)
tests/test_forecast.py   — 20 tests (TA signals, ATR, Bollinger, RSI edge cases)
tests/test_metrics.py    — 28 tests (Sharpe, Sortino, CAGR, drawdown, edge cases)
```

---

## Project Structure

```
trade/
├── main.py                    CLI entry point
├── dashboard.py               Streamlit dashboard
├── scheduler.py               Monthly auto-runner
├── config.yaml                Strategy configuration
├── requirements.txt           Dependencies
│
├── src/
│   ├── ingestion.py           Download OHLCV data (with read-back assertion)
│   ├── pipeline.py            Clean and align data
│   ├── bot.py                 DCA backtesting engine
│   ├── forecast.py            11-signal technical forecast
│   ├── sentiment.py           VADER news sentiment scoring
│   ├── live_feed.py           Real-time delayed prices
│   ├── metrics.py             Risk metrics (Sharpe, Sortino, CAGR, etc.)
│   ├── paper_trader.py        Virtual trading simulation
│   ├── performance_tracker.py Backtest snapshot history
│   ├── portfolio_manager.py   Multi-preset comparison
│   ├── report_builder.py      Excel + PDF export
│   └── config_loader.py       YAML → BotConfig loader
│
├── tests/
│   ├── test_bot.py
│   ├── test_forecast.py
│   └── test_metrics.py
│
└── data/                      Created on first run (gitignored)
    ├── raw/                   Raw parquet files
    ├── cleaned/               Cleaned parquet + trade CSVs
    ├── paper_portfolio.json   Paper trading state
    ├── paper_trades.csv       Paper trade log
    └── performance_history.parquet
```

---

## How the DCA Strategy Works

1. **Monthly trigger** — on the last trading day of each month, the bot evaluates each ticker
2. **Base buy** — invests the configured monthly budget
3. **SMA200 multiplier** — if price is below SMA200 (downtrend), budget is multiplied (default 1.5x)
4. **RSI multiplier** — if RSI(21) is below the oversold threshold (default 35), budget is multiplied again (default 2x)
5. **Both conditions** — multipliers stack, so a crash below SMA200 with oversold RSI triggers the full 3x budget
6. **Exit signals** (optional) — take-profit at +50%, stop-loss at -20% from average cost
7. **Rebalancing** (optional) — quarterly check, rebalances when any ticker drifts more than 10% from target allocation

---

## Technical Forecast Signals

The forecast runs 11 signals on the most recent data:

| # | Signal | Bullish | Bearish |
|---|---|---|---|
| 1 | RSI(14) | < 40 | > 70 |
| 2 | Price vs SMA200 | Above | Below |
| 3 | Price vs SMA50 | Above | Below |
| 4 | Price vs SMA20 | Above | Below |
| 5 | MACD | MACD > Signal, rising | MACD < Signal, falling |
| 6 | Bollinger Bands | Below lower band | Above upper band |
| 7 | Stochastic K/D | K < 25 and K > D | K > 80 and K < D |
| 8 | 5-day momentum | > +1.5% | < -1.5% |
| 9 | 20-day momentum | > +3% | < -3% |
| 10 | OBV trend | Rising | Falling |
| 11 | News Sentiment | VADER score >= +0.15 | VADER score <= -0.15 |

Confidence = (signals agreeing with bias) / (total non-neutral signals). Price range is ATR-based.

---

## Data Notes

- **Source:** Yahoo Finance (free tier, no API key needed)
- **Delay:** Live prices are 15-20 minutes delayed
- **History:** Max available history per ticker (SPY since 1993, QQQ since 1999, AMD since 1983)
- **Storage:** Parquet format with Snappy compression via PyArrow
- **Data integrity:** Every write is followed by a read-back assertion — zero silent data loss

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Download OHLCV and news data from Yahoo Finance |
| `pandas` | Data manipulation |
| `pyarrow` | Parquet read/write |
| `numpy` | Numerical computing |
| `ta` | Technical analysis indicators |
| `vaderSentiment` | News sentiment scoring |
| `streamlit` | Web dashboard |
| `streamlit-autorefresh` | Non-blocking page auto-refresh |
| `openpyxl` | Excel export |
| `fpdf2` | PDF export |
| `pyyaml` | Config file parsing |
| `pytest` | Unit tests |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
