# US Stocks Long-Term DCA Pipeline — Project Overview

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Directory Structure](#2-directory-structure)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
   - [requirements.txt](#41-requirementstxt)
   - [src/ingestion.py](#42-srcingestionpy)
   - [src/pipeline.py](#43-srcpipelinepy)
   - [src/bot.py](#44-srcbotpy)
   - [src/forecast.py](#45-srcforecastpy)
   - [main.py](#46-mainpy)
   - [audit.py](#47-auditpy)
   - [dashboard.py](#48-dashboardpy)
5. [Data Folder](#5-data-folder)
6. [Zero Data Loss Guarantee](#6-zero-data-loss-guarantee)
7. [Strategy Logic](#7-strategy-logic)
8. [Technical Forecast Engine](#8-technical-forecast-engine)
9. [Bugs Found and Fixed](#9-bugs-found-and-fixed)
10. [How to Run](#10-how-to-run)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Limitations & Honest Disclaimers](#12-limitations--honest-disclaimers)

---

## 1. Project Summary

A **local, end-to-end quantitative investment pipeline** for three US assets:

| Ticker | Name | Type |
|--------|------|------|
| SPY | SPDR S&P 500 ETF | Broad market ETF |
| QQQ | Invesco NASDAQ-100 ETF | Tech-heavy ETF |
| AMD | Advanced Micro Devices | Individual stock |

The system does four things in sequence:

```
Download raw data  →  Clean & validate  →  Backtest DCA strategy  →  Visualise in dashboard
```

Key design principle: **0% data loss at every layer**, enforced by assertion checks and read-back verification at each step.

This is a **long-term investment simulator**, not an intraday or high-frequency trading system. The minimum holding period between any two buys is 28+ days (monthly cadence).

---

## 2. Directory Structure

```
F:\us_stocks\
│
├── data\
│   ├── raw\                    # Untouched OHLCV downloads from Yahoo Finance
│   │   ├── SPY.parquet
│   │   ├── QQQ.parquet
│   │   └── AMD.parquet
│   │
│   └── cleaned\                # Processed, schema-enforced, validated data
│       ├── SPY_clean.parquet
│       ├── QQQ_clean.parquet
│       ├── AMD_clean.parquet
│       ├── trade_log.csv       # Every buy trade executed in backtest
│       ├── summary.csv         # Per-ticker P&L summary
│       └── equity_curve.csv    # Daily portfolio value over time
│
├── src\
│   ├── __init__.py
│   ├── ingestion.py            # Download + persist raw data
│   ├── pipeline.py             # Clean, validate, align data
│   ├── bot.py                  # DCA strategy backtester
│   └── forecast.py             # Technical analysis signal engine
│
├── .venv\                      # Isolated Python virtual environment
├── main.py                     # CLI orchestrator (runs full pipeline)
├── audit.py                    # Standalone 0% data loss audit script
├── dashboard.py                # Streamlit web UI
├── requirements.txt            # Python dependencies
├── pipeline.log                # Runtime log file (auto-generated)
└── PROJECT_OVERVIEW.md         # This file
```

---

## 3. Architecture & Data Flow

```
Yahoo Finance API
      |
      | yfinance.Ticker.history(period="max")
      v
[ingestion.py]
  - Exponential backoff retry (up to 5 attempts)
  - Save to data/raw/{TICKER}.parquet
  - Read-back row count verification
      |
      v
[pipeline.py]
  - Load raw parquet
  - Enforce schema (float64 prices, int64 volume)
  - Forward-fill + backward-fill missing values (NEVER drop rows)
  - Assert: cleaned rows == raw rows
  - Save to data/cleaned/{TICKER}_clean.parquet
  - Read-back verification again
      |
      v
[bot.py]  ←── BotConfig (budget, RSI threshold, SMA multipliers, dates)
  - Load cleaned data, compute RSI(21) + SMA(200) on FULL history first
  - Apply date filter AFTER indicators (avoids warmup NaN bug)
  - Select actual last trading day of each month (not calendar month-end)
  - For each monthly date: decide buy amount based on signal regime
  - Record trade with fill price (close + slippage), fee, cumulative shares
  - summary(): per-ticker P&L using end_date price (not today's price)
  - equity_curve(): daily portfolio value from cumulative positions
      |
      v
[forecast.py]  (independent, uses cleaned data directly)
  - Compute 10 technical indicators on full price history
  - Score each signal as +1 (bullish), -1 (bearish), 0 (neutral)
  - Derive overall bias + confidence percentage
  - Build ATR-based price range for next trading session
      |
      v
[dashboard.py]  ←── Streamlit web UI on localhost:8501
  - Live price bar (latest close + 1-day change per ticker)
  - Sidebar: all strategy parameters with help text + effect explanations
  - Portfolio KPI cards, equity curve, allocation donut
  - Price chart with trade entry point markers
  - Filterable trade log + CSV download
  - Trigger breakdown charts
  - Data integrity audit panel
  - Next session technical forecast with signal scorecard
```

---

## 4. File-by-File Breakdown

### 4.1 `requirements.txt`

Lists all Python package dependencies. Install into the virtual environment with:

```
.venv\Scripts\pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| `yfinance` | >=0.2.40 | Download historical OHLCV data from Yahoo Finance |
| `pandas` | >=2.0.0 | All data manipulation, DataFrames, time series |
| `pyarrow` | >=14.0.0 | Parquet file read/write engine (fast, compressed) |
| `numpy` | >=1.26.0 | Numerical calculations, indicator math |
| `ta` | >=0.11.0 | Technical analysis library (imported but indicators hand-coded for control) |
| `requests` | >=2.31.0 | HTTP requests (used internally by yfinance) |
| `streamlit` | (installed separately) | Web UI framework |
| `plotly` | (installed separately) | Interactive charts in the dashboard |

---

### 4.2 `src/ingestion.py`

**Purpose:** Download maximum available historical daily OHLCV data for SPY, QQQ, and AMD from Yahoo Finance and persist it locally with integrity verification.

**Key functions:**

#### `_download_with_backoff(ticker, retry=0)`
- Calls `yfinance.Ticker(ticker).history(period="max", interval="1d")`
- If the API fails (network error, rate limit, empty response), waits `2^retry` seconds and retries
- Maximum 5 retries (waits: 1s, 2s, 4s, 8s, 16s)
- Raises after all retries exhausted
- Prevents silent partial downloads

#### `_validate_and_save(df, ticker)`
- Normalises the date index to UTC midnight
- Saves DataFrame to `data/raw/{TICKER}.parquet` using PyArrow + Snappy compression
- **Immediately reads the file back** and asserts row count matches
- If even 1 row is missing on disk, raises `AssertionError` and halts — no silent data loss

#### `ingest_all(tickers)`
- Loops over all tickers, calls the two functions above
- Returns a dict mapping ticker → file path
- Called by `main.py` and the dashboard's "Re-download Data" button

**Data integrity guarantee at this layer:**
```
rows_fetched_from_api == rows_written_to_parquet == rows_read_back_from_parquet
```

---

### 4.3 `src/pipeline.py`

**Purpose:** Load raw Parquet files, enforce schema, impute missing values without dropping rows, and save clean validated data.

**Key functions:**

#### `_enforce_schema(df, ticker)`
- Casts price columns (Open, High, Low, Close) to `float64`
- Casts Volume to `int64` (fills NaN volume with 0 before casting)
- Logs a warning and inserts a NaN column if any expected column is missing from the API response
- Uses `pd.to_numeric(errors="coerce")` to safely handle non-numeric values

#### `_impute(df, ticker)`
- Applies `ffill()` (forward-fill) then `bfill()` (backward-fill) on price columns
- **Rows are NEVER dropped** — gaps in data are filled from adjacent rows
- Logs how many cells were imputed before and after

#### `clean_ticker(ticker)`
- Full cleaning pipeline for one ticker
- Loads raw parquet → enforces schema → imputes → asserts row count unchanged → saves to cleaned parquet → reads back and asserts again
- Two separate assertion checkpoints ensure nothing is silently lost

#### `clean_all(tickers)`
- Runs `clean_ticker` for all tickers, returns dict of cleaned DataFrames

#### `load_aligned(tickers)`
- Loads all cleaned tickers and outer-joins them on the Date index
- Union of all dates means no trading day from any asset is dropped
- Gaps introduced by the outer join (one asset has dates another doesn't) are filled with `ffill().bfill()`
- Result: a single aligned DataFrame with `(ticker, column)` multi-level column names

**Zero-loss rule:**
```
len(cleaned_df) == len(raw_df)   # enforced by assertion before AND after disk write
```

---

### 4.4 `src/bot.py`

**Purpose:** Simulate a long-term fractional Dollar-Cost Averaging (DCA) strategy with intelligent buy triggers based on technical market regime signals.

**Strategy logic:**

Every month (last actual trading day of each calendar month), the bot evaluates the market regime and decides how much to invest:

| Condition | Monthly Buy Amount | Trigger Label |
|-----------|-------------------|---------------|
| RSI(21) < 35 (oversold crash) | Budget × 2.0 | `RSI_OVERSOLD_2X` |
| Price < SMA(200) (downtrend) | Budget × 1.5 | `BELOW_SMA200_1.5X` |
| Normal conditions | Budget × 1.0 | `DCA_NORMAL` |

**Key classes and functions:**

#### `BotConfig` (dataclass)
All strategy parameters in one place:
- `monthly_budget_usd`: base investment per ticker per month (default $50)
- `oversold_rsi`: RSI threshold below which crash-buying triggers (default 35)
- `rsi_period`: RSI calculation window (default 21 days)
- `sma_period`: SMA trend filter window (default 200 days)
- `below_sma_multiplier`: budget multiplier when below SMA200 (default 1.5)
- `oversold_multiplier`: budget multiplier on RSI crash (default 2.0)
- `slippage_bps`: realistic trading cost in basis points (default 3 bps)
- `clearing_fee_usd`: flat fee per trade (default $0.005)
- `min_hold_days`: minimum days between entry decisions (default 15)
- `start_date` / `end_date`: optional backtest window

#### `LongTermDCABot._load(ticker, full=False)`
Critical fix implemented here: indicators (RSI, SMA200) are computed on the **full history first**, then the date filter is applied. This prevents the warmup NaN problem where starting a backtest from 2015 would have invalid SMA200 values for the first 200 trading days.

- `full=False` (default for `run()`): computes indicators on all data, then filters to start/end window
- `full=True` (used by `summary()` and `equity_curve()`): returns full history, caller clips to end_date — ensures valuation uses the correct backtest endpoint price, not today's price

#### `LongTermDCABot._budget(rsi, close, sma200)`
Returns the adjusted budget and trigger label for a given market state. RSI oversold takes priority over SMA check.

#### `LongTermDCABot._execute_buy(ticker, date, close, rsi, sma200)`
- Calculates `fill_price = close + slippage` (realistic execution price)
- Calculates `shares = (budget - clearing_fee) / fill_price` to 6 decimal places (fractional shares)
- Fee calculation: `clearing_fee + (shares × raw_slippage_cost)` — uses pre-computed slippage cost, not fill_price, to avoid double-counting
- Appends trade record to `self.trades`

#### `LongTermDCABot.run()`
Main backtest loop. For each ticker:
1. Loads data via `_load()`
2. Identifies actual last trading day of each month using positional period comparison (avoids `resample("ME")` bug where calendar month-end may not be a trading day, causing equity curve join misses)
3. Calls `_execute_buy()` for each monthly date

#### `LongTermDCABot.summary(trade_log)`
Computes per-ticker P&L table. Uses `_load(full=True)` then clips to `end_date` — so the "last close" price used for valuation is the correct backtest end date price, not today's price.

#### `LongTermDCABot.equity_curve(trade_log)`
Builds a daily portfolio value time series by:
1. Loading full price history (clipped to end_date)
2. Left-joining cumulative share positions onto daily closes
3. Forward-filling share positions (shares accumulate and hold)
4. Multiplying shares × close price = daily value per ticker
5. Summing across tickers for total portfolio value

**Bugs found and fixed in this file:**

| Bug | Impact | Fix |
|-----|--------|-----|
| `resample("ME")` used calendar month-end dates, not actual trading days | Equity curve dropped trades on non-trading calendar end dates | Used positional period comparison `.to_series().shift(-1)` |
| `summary()` and `equity_curve()` read full parquet ignoring `end_date` | P&L showed today's price instead of backtest end date price | Both now use `_load(full=True)` with manual end_date clip |
| Indicators computed after date filter | First ~200 days of SMA200 were NaN after any `start_date` | Moved indicator computation before date filter |
| Fee used `fill_price × bps` instead of `slippage_cost` | Slippage was double-counted (charged on already-inflated price) | Fee now uses pre-computed `slippage_cost` variable |
| `self.cash_deployed` was initialised and updated but never read | Dead code | Removed entirely |
| RSI returned NaN when avg_loss = 0 (all-up market) | NaN indicators during strong uptrends | `rsi[avg_loss == 0] = 100` added |

---

### 4.5 `src/forecast.py`

**Purpose:** Compute a signal-based technical analysis report for the next trading session. This is NOT a machine learning model and NOT a price prediction — it is a structured scorecard of classical technical indicators.

**What it does:**
- Loads the full cleaned price history for each ticker
- Computes 10 technical indicators
- Scores each indicator as +1 (bullish), -1 (bearish), or 0 (neutral)
- Derives an overall directional bias and confidence percentage
- Builds an ATR-based price range (expected low / midpoint / high) for the next session
- Identifies key support and resistance levels

**Indicators computed:**

| Indicator | What it measures | Bullish condition | Bearish condition |
|-----------|-----------------|-------------------|-------------------|
| RSI(14) | Momentum / oversold-overbought | RSI < 40 (oversold) | RSI > 70 (overbought) |
| Price vs SMA200 | Long-term trend | Price above 200-day avg | Price below 200-day avg |
| Price vs SMA50 | Medium-term trend | Price above 50-day avg | Price below 50-day avg |
| Price vs SMA20 | Short-term trend | Price above 20-day avg | Price below 20-day avg |
| MACD vs Signal | Momentum crossover | MACD above signal and rising | MACD below signal and falling |
| Bollinger Band position | Volatility extremes | Price below lower band (oversold squeeze) | Price above upper band (overbought stretch) |
| Stochastic K/D | Overbought/oversold oscillator | K < 25 and K crossing above D | K > 80 and K crossing below D |
| 5-day momentum | Recent price direction | +1.5% or more over 5 days | -1.5% or more over 5 days |
| 20-day momentum | Monthly price direction | +3% or more over 20 days | -3% or more over 20 days |
| OBV trend | Volume-based money flow | OBV above its 20-day average | OBV below its 20-day average |

**Price range methodology:**
```
expected_low  = last_close - ATR(14)
expected_high = last_close + ATR(14)
expected_mid  = last_close + (signal_score / 10) * ATR(14) * 0.5
```
ATR (Average True Range) measures the average daily price movement over 14 days. The range represents the typical swing around last close — not a prediction of where price will close.

**Confidence percentage:**
Number of signals agreeing with the overall direction divided by total signals. A score of 50% means 5 out of 10 signals agree. Even the best technical setups rarely exceed 60-70% signal agreement.

**Key functions:**

- `_ema(s, n)`: Exponential Moving Average
- `_rsi(s, n)`: RSI with fix for all-gains edge case (returns 100 instead of NaN)
- `_macd(s)`: MACD line, signal line, and histogram
- `_bollinger(s, n, k)`: Bollinger Bands (upper, mid, lower)
- `_atr(high, low, close, n)`: Average True Range for volatility sizing
- `_stochastic(high, low, close, k, d)`: Stochastic oscillator %K and %D
- `_obv(close, volume)`: On-Balance Volume (money flow proxy)
- `_score_signal(name, value, bullish_cond, bearish_cond)`: Returns a scored signal dict
- `forecast_ticker(ticker)`: Full analysis for one ticker, returns a comprehensive dict
- `forecast_all(tickers)`: Runs forecast for all tickers, returns dict keyed by ticker

**Important disclaimer (embedded in every result):**
> No model predicts stock prices accurately. The best quantitative hedge funds achieve ~52-56% directional accuracy on daily moves — barely above a coin flip. This tool provides structured context, not predictions.

---

### 4.6 `main.py`

**Purpose:** CLI entry point that orchestrates the full pipeline in sequence. Designed to be run from the command line with configurable arguments.

**Usage:**
```bash
# Full pipeline (download + clean + backtest)
python main.py

# Skip re-download, use cached data
python main.py --skip-ingest

# Custom date window and budget
python main.py --skip-ingest --end 2024-12-31 --budget 100

# Full custom run
python main.py --start 2005-01-01 --end 2023-12-31 --budget 200
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip-ingest` | False | Skip Yahoo Finance download, use existing raw files |
| `--start` | None | Backtest start date (YYYY-MM-DD). None = each ticker's earliest date |
| `--end` | None | Backtest end date (YYYY-MM-DD). None = latest available |
| `--budget` | 50.0 | Monthly DCA budget per ticker in USD |

**Pipeline steps executed:**

1. **Step 1 - Ingestion**: Downloads max history for SPY, QQQ, AMD and saves to `data/raw/`
2. **Step 2 - Cleaning**: Validates, schema-enforces, imputes, and saves to `data/cleaned/`
3. **Step 3 - Alignment**: Outer-joins all tickers into a single aligned frame and logs shape
4. **Step 4 - Backtest**: Runs `LongTermDCABot` with configured parameters
5. **Step 5 - Results**: Prints trade log, summary table, and equity curve; saves CSV outputs

**Logging:** Writes to both stdout and `pipeline.log` with timestamps. File handler uses UTF-8 encoding to support all characters.

---

### 4.7 `audit.py`

**Purpose:** Standalone data integrity audit script. Run independently to verify that zero data loss has occurred at any point in the pipeline. Produces a clear PASS/FAIL report for each check.

**Run with:**
```bash
python audit.py
```

**Checks performed per ticker (8 checks × 3 tickers = 24 total checks):**

| Check | What it verifies |
|-------|-----------------|
| Row count match | `len(raw) == len(cleaned)` — zero rows silently dropped |
| NaN prices | No missing Open/High/Low/Close values after cleaning |
| NaN volume | No missing volume values after cleaning |
| Duplicate dates | No two rows share the same date index |
| Max calendar gap | No gap between consecutive trading days exceeds 10 calendar days |
| Close dtype | `Close` column is `float64` (not object or int) |
| Volume dtype | `Volume` column is `int64` (not float) |
| Chronological order | Date index is monotonically increasing |
| No negative prices | All OHLC values are positive |

**Exit codes:**
- Exit 0: all checks passed
- Exit 1: one or more checks failed (printed to stdout)

**Note:** The 7-day max gap seen in the output (around 2001-09-17) is the September 11 market closure — correct and expected behaviour, not a data loss event.

---

### 4.8 `dashboard.py`

**Purpose:** Interactive web dashboard built with Streamlit. Provides a full visual interface to run the pipeline, explore results, and view technical forecasts — no command line needed.

**Launch:**
```bash
.venv\Scripts\streamlit run dashboard.py --server.port 8501
```
Then open `http://localhost:8501` in your browser.

**Sidebar controls:**

| Control | Function |
|---------|----------|
| Select tickers | Choose any combination of SPY, QQQ, AMD |
| Start / End date | Backtest window (start blank = each ticker's earliest date) |
| Monthly budget slider | $10–$500 per ticker per month |
| RSI oversold threshold | 20–45; below this RSI the bot buys 2x |
| Below-SMA200 multiplier | 1.0–3.0; how much extra to buy in downtrends |
| RSI oversold multiplier | 1.5–4.0; how aggressively to buy on RSI crashes |
| Slippage slider | 1–20 bps; realistic transaction cost |
| Run Backtest button | Executes full clean + backtest with current settings |
| Re-download Data button | Fetches latest data from Yahoo Finance |

Each slider has a **"What does this do?"** expander below it explaining the indicator, why it matters, and what happens when you increase or decrease the value.

**Dashboard sections:**

1. **Live Market Prices** (always visible, top of page)
   - Latest closing price for each selected ticker
   - 1-day dollar and percentage change
   - Date of last data point
   - Visible even before running a backtest

2. **Portfolio KPI Cards**
   - Total Invested / Portfolio Value / Unrealised P&L (with % delta) / Total Trades / Fees Paid

3. **Current Strategy Settings** (collapsible)
   - Shows which exact parameters were used in the current backtest result

4. **Portfolio Equity Curve** (tabbed)
   - Combined: total portfolio value over time with filled area
   - Per Ticker: individual ticker value lines (colour-coded)

5. **Ticker Breakdown + Allocation Donut**
   - Table: invested, avg cost, total shares, last close, market value, P&L
   - Donut chart: portfolio allocation by current market value

6. **Price History + Trade Entry Points**
   - Full close price line for selected ticker
   - Trade markers overlaid on the chart:
     - Circle (blue) = normal monthly DCA
     - Triangle (orange) = below SMA200, 1.5x buy
     - Star (red) = RSI oversold, 2x buy

7. **Trade Log**
   - Filterable by ticker and trigger type
   - Shows: date, ticker, budget, fill price, shares bought, RSI, SMA200, trigger, fee
   - Download as CSV button

8. **Trigger Breakdown Charts**
   - Capital deployed by trigger type per ticker (grouped bar chart)
   - Monthly capital deployment over time (bar chart)

9. **Data Integrity Audit**
   - Live PASS/FAIL check for each ticker inline in the dashboard
   - Green success banner when all 24 checks pass

10. **Next Session Technical Forecast**
    - Shows last data date and explicitly states which trading session it is forecasting for
    - Warning disclaimer that this is not a price prediction
    - Per-ticker expandable sections containing:
      - 4 metric cards: Expected Low / Midpoint Bias / Expected High / ATR
      - Signal Scorecard bar chart (green/red per indicator)
      - Support and resistance levels
      - Full indicator values table (RSI, MACD, Stochastics, Bollinger, momentum, volume)

---

## 5. Data Folder

### `data/raw/`
Untouched data exactly as downloaded from Yahoo Finance. Never modified after initial write.

| File | Contents | Typical Size |
|------|----------|-------------|
| `SPY.parquet` | Daily OHLCV from 1993-01-29 to latest | ~8,400 rows |
| `QQQ.parquet` | Daily OHLCV from 1999-03-10 to latest | ~6,858 rows |
| `AMD.parquet` | Daily OHLCV from 1980-03-17 to latest | ~11,655 rows |

### `data/cleaned/`
Processed outputs.

| File | Contents |
|------|----------|
| `SPY_clean.parquet` | Schema-enforced, imputed, validated SPY data |
| `QQQ_clean.parquet` | Schema-enforced, imputed, validated QQQ data |
| `AMD_clean.parquet` | Schema-enforced, imputed, validated AMD data |
| `trade_log.csv` | Every BUY trade: date, ticker, budget, fill price, shares, RSI, SMA200, trigger, fee |
| `summary.csv` | Per-ticker: invested, avg cost, shares, last close, market value, P&L, trade count |
| `equity_curve.csv` | Daily portfolio value per ticker + total, from first trade to end date |

---

## 6. Zero Data Loss Guarantee

The pipeline enforces data integrity at **4 independent checkpoints**:

```
Checkpoint 1 (ingestion.py):
  rows fetched from API == rows in parquet file (read-back assertion)

Checkpoint 2 (pipeline.py, before save):
  len(cleaned_df) == len(raw_df)  [AssertionError if violated]

Checkpoint 3 (pipeline.py, after save):
  rows in cleaned parquet == rows in raw parquet (second read-back)

Checkpoint 4 (audit.py):
  8 independent checks per ticker including NaN scan, duplicate check,
  gap check, dtype check, sort check, negative price check
```

The only rows legitimately excluded are non-trading days that have no data at all across all tickers. These are excluded by the data source (Yahoo Finance) itself, not by our pipeline.

---

## 7. Strategy Logic

### Dollar-Cost Averaging (DCA)
DCA is the practice of investing a fixed amount at regular intervals regardless of price. Over long time horizons it reduces the impact of volatility by automatically buying more shares when prices are low and fewer when prices are high.

### How This Bot Enhances Basic DCA

Rather than a flat fixed amount every month, the bot uses **regime-triggered scaling**:

```
IF RSI(21) < 35 (crash / deep oversold):
    invest = monthly_budget × 2.0   ← aggressive dip buying

ELSE IF close < SMA(200) (long-term downtrend):
    invest = monthly_budget × 1.5   ← mild accumulation

ELSE (normal uptrend):
    invest = monthly_budget × 1.0   ← standard DCA
```

This means the bot automatically invests more during market crashes and downtrends — the times when stocks are statistically cheapest relative to their long-term average.

### Why Monthly Cadence?
- Minimum 28 days between buys (exceeds the 15-day minimum hold requirement)
- Reduces transaction fees vs weekly or daily DCA
- Reduces the psychological burden of daily market watching
- Consistent with how most retail investors receive income (monthly salary)

### Fractional Shares
All positions are computed to 6 decimal places:
```
shares = (budget - clearing_fee) / fill_price
```
This enables realistic simulation of fractional share investing (e.g., buying $50 worth of a $742 stock).

### Transaction Costs
- **Slippage**: `close × (bps / 10,000)` added to fill price (3 bps default)
- **Clearing fee**: flat $0.005 per trade (fractional-share platform model)
- Both costs reduce the effective budget and inflate the fill price realistically

---

## 8. Technical Forecast Engine

The forecast is **rule-based signal scoring**, not machine learning.

### What "Confidence %" Actually Means
If overall bias is Bullish and confidence is 50%, it means 5 out of 10 signals agree the direction is up. The other 5 are neutral or disagree.

### What "Expected Range" Actually Means
```
Expected Low  = last_close - ATR(14)
Expected High = last_close + ATR(14)
```
ATR is the average daily price swing over the last 14 days. This range says: "based on recent volatility, the next session is likely to stay within this band." It is NOT a prediction of where price will close — it is a volatility envelope.

### Why No ML Model?
- Daily stock price direction is close to a random walk
- Even the best quant funds achieve only ~52-56% directional accuracy on daily moves
- ML models (LSTM, Random Forest) on OHLCV data have been extensively studied and show marginal improvement over rule-based signals
- A Random Forest on TA features would be the most realistic upgrade (planned future enhancement)

---

## 9. Bugs Found and Fixed

During development, code reviews identified bugs across `bot.py` and `forecast.py`. All were verified against the actual code before fixing.

### Bugs fixed in `src/bot.py`

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `resample("ME")` sets calendar month-end dates; equity curve left-join on actual trading days silently drops trades | Equity curve was wrong | Replaced with positional period comparison |
| 2 | `summary()` and `equity_curve()` bypassed date filters by reading parquet directly | P&L always showed today's price regardless of `--end` | Both use `_load(full=True)` + manual end_date clip |
| 3 | RSI and SMA200 computed after date filter | First ~200 trading days after `--start` had NaN indicators; wrong signals for ~1 year | Moved indicator computation before date filter |
| 4 | Fee used `shares × fill_price × bps` instead of `shares × slippage_cost` | Slippage double-counted (charged on already-inflated price) | Use pre-computed `slippage_cost` variable |
| 5 | `self.cash_deployed` initialised and updated but never read | Dead code | Removed |
| 6 | RSI returned NaN when `avg_loss = 0` (all-up market for 21+ days) | NaN indicators during strong uptrends | `rsi[avg_loss == 0] = 100` |

### Bugs fixed in `src/forecast.py`

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 7 | `__main__` print loop accessed `r['overall_bias']` without checking for error dict | `KeyError` crash when any ticker has insufficient data or missing file | Added `if "error" in r: continue` guard |
| 8 | Neutral confidence used `bull_count` when `total_score == 0` (because `0 >= 0` is `True`) | Displayed bullish count as neutral confidence — e.g. 3 bullish / 3 bearish / 4 neutral reported "Neutral (30%)" instead of "Neutral (40%)" | Explicit `if total_score == 0: n_agree = neutral_count` branch |
| 9 | Bollinger bullish condition was `last_close < bb_mid_v` (below 20-day SMA) | Being below the moving average is bearish not bullish — permanently fired bullish signals during downtrends | Changed to `last_close < bb_lower` (symmetric with bearish `> bb_upper`) |
| 10 | `_bollinger()` used `std(ddof=1)` (sample standard deviation) | Diverges from the classic Bollinger Band formula which uses population std | Changed to `std(ddof=0)` |
| 11 | `_atr()` used `ewm(span=n)` giving alpha = 2/(n+1) ≈ 0.133 for n=14 | Wilder's ATR uses alpha = 1/n ≈ 0.071 — span=14 reacts ~2× faster than the classic 14-period ATR, inflating the expected price range | Changed to `ewm(alpha=1/n, min_periods=n)` |

2 additional bugs reported externally were **not real**:
- Empty DataFrame KeyError: `pd.DataFrame(list_of_dicts)` always preserves column names from dict keys even with 0 rows
- Missing index deduplication: verified 0 duplicates in all 3 raw files

---

## 10. How to Run

### First Time Setup
```powershell
# Navigate to project
cd F:\us_stocks

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install streamlit plotly
```

### Run Full Pipeline (CLI)
```powershell
cd F:\us_stocks

# Download data + clean + backtest with defaults
python main.py

# Skip download if data already exists
python main.py --skip-ingest

# Custom date range (from each ticker's earliest date to end of 2024, $100/month)
python main.py --skip-ingest --end 2024-12-31 --budget 100
```

### Run Data Audit
```powershell
cd F:\us_stocks
python audit.py
```

### Run Dashboard (UI)
```powershell
cd F:\us_stocks
.\.venv\Scripts\streamlit run dashboard.py --server.port 8501
```
Open browser at `http://localhost:8501`

### Refresh Data to Latest
Either click **"Re-download Data"** in the dashboard sidebar, or:
```powershell
python main.py  # runs full pipeline including fresh download
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Parquet format for storage | Columnar, compressed, typed — 5-10x smaller than CSV, faster reads |
| UTC timezone on all date indices | Avoids DST ambiguity; consistent across all operations |
| Monthly cadence (not weekly/daily) | Realistic for retail investors; reduces fees; exceeds 15-day hold requirement |
| Indicators computed before date filter | Prevents warmup NaN problem that would invalidate first year of any backtest |
| Actual last trading day (not calendar month-end) | Calendar month-end may be Saturday/Sunday; using it breaks equity curve joins |
| Absolute file paths via `Path(__file__).resolve()` | Dashboard works regardless of which directory Streamlit is launched from |
| `full=True` flag in `_load()` | Allows summary/equity_curve to use correct end-date valuation price |
| No ML model in forecast | Honest — ML on daily OHLCV data adds minimal edge; rule-based signals are interpretable |
| Forward-fill before backward-fill | Preserves most recent known value; bfill only used for leading NaNs at start of history |

---

## 12. Limitations & Honest Disclaimers

### On the Backtest
- **Survivorship bias**: SPY, QQQ, AMD were chosen with hindsight knowledge that they performed well. A real backtest should include assets that failed.
- **No exit strategy**: The bot only buys, never sells. Real portfolios require rebalancing and exit rules.
- **Historical returns do not guarantee future returns.** AMD's 7000%+ backtest return is real historical data — it does not mean AMD will repeat this performance.
- **Slippage is estimated**, not measured. Real execution costs depend on broker, order size, and market conditions.

### On the Forecast
- **No ML model is used.** The forecast is rule-based technical analysis.
- **Technical analysis is not predictive.** It describes what has happened, not what will happen.
- **Confidence % does not mean accuracy %.**  50% confidence means 5/10 signals agree, not that the forecast is right 50% of the time.
- **The expected price range is an ATR volatility band**, not a predicted closing price.
- **Never make trading decisions based solely on this tool.**

### On Real Trading
- This system has **not been tested in live trading**.
- Before connecting to any broker or exchange: run paper trading for minimum 4-8 weeks, validate all edge cases, implement a hard kill switch.
- US stocks (SPY, QQQ, AMD) are **not available on Binance** — Binance is a crypto exchange. A separate implementation would be needed for crypto assets.
