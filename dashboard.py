"""
dashboard.py - Streamlit UI for the US Stocks Long-Term DCA Pipeline.

Run with:
    .venv\Scripts\streamlit run dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import ingest_all, TICKERS
from pipeline import clean_all
from bot import LongTermDCABot, BotConfig
from forecast import forecast_all
from metrics import compute_all, format_metrics, buy_and_hold_equity
from sentiment import sentiment_all, sentiment_vs_price
from live_feed import fetch_all_live_prices, is_market_open

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="DCA Portfolio Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR  = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
RAW_DIR   = BASE_DIR / "data" / "raw"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKER_COLORS  = {"SPY": "#2196F3", "QQQ": "#4CAF50", "AMD": "#FF5722"}
TICKER_NAMES   = {
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ-100 ETF",
    "AMD": "Advanced Micro Devices",
}

# -- Parameter help text (shown as notes below each slider) --------------------

PARAM_HELP = {
    "monthly_budget": {
        "label": "Monthly budget per ticker ($)",
        "note": (
            "How much cash you invest in each ticker every month. "
            "Higher = faster wealth accumulation but requires more capital. "
            "Lower = safer for tight budgets. "
            "Tip: even $10/month compounded over 20+ years becomes significant."
        ),
        "effect": "Increase -> more shares per month -> higher long-term returns. "
                  "Decrease -> slower growth but lower monthly commitment.",
    },
    "oversold_rsi": {
        "label": "RSI oversold threshold",
        "note": (
            "RSI (Relative Strength Index) measures momentum on a 0-100 scale. "
            "Below 35 typically means the stock is oversold — it has dropped fast "
            "and may be due for a bounce. When RSI drops below this value, the bot "
            "doubles your monthly buy to accumulate more shares at a discount."
        ),
        "effect": "Lower value (e.g. 25) -> only buys extra at deep crashes, fewer triggers. "
                  "Higher value (e.g. 40) -> triggers more often, buys more frequently on dips.",
    },
    "below_sma_mult": {
        "label": "Below-SMA200 multiplier",
        "note": (
            "The 200-day Simple Moving Average (SMA200) is the average closing price "
            "over the last 200 trading days. When the current price is BELOW this average, "
            "the stock is in a long-term downtrend. The bot invests this multiplier x your "
            "base budget to accumulate more shares while prices are lower than the long-term average."
        ),
        "effect": "1.0 = no extra buying in downtrends. "
                  "1.5 = invest 50% more when below SMA200. "
                  "2.0 = double investment in downtrends (aggressive accumulation).",
    },
    "oversold_mult": {
        "label": "RSI oversold multiplier",
        "note": (
            "When RSI drops below the oversold threshold (a crash or sharp selloff), "
            "this multiplier controls how aggressively you buy extra. A multiplier of 2x "
            "means you invest double your monthly budget on those months — buying the dip. "
            "This is the most powerful lever for long-term outperformance."
        ),
        "effect": "2.0 = double buy on RSI crash (default). "
                  "3.0 = triple buy — very aggressive dip buying, great if you have spare cash. "
                  "1.5 = mild dip response, more conservative.",
    },
    "slippage_bps": {
        "label": "Slippage (basis points)",
        "note": (
            "Slippage is the small difference between the price you see and the price "
            "you actually pay when your order fills. 1 basis point = 0.01%. "
            "For large ETFs like SPY/QQQ, 2-5 bps is realistic. "
            "For AMD (individual stock), 5-10 bps is more accurate. "
            "This makes the backtest more realistic — real brokers always have some cost."
        ),
        "effect": "Higher slippage -> slightly lower real returns (costs more per trade). "
                  "Set to 0 for ideal-world backtest, 5-10 for realistic estimate.",
    },
}

# -- Helpers -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_live_prices():
    """Read the latest close price and 1-day change from cleaned parquet files."""
    prices = {}
    for t in TICKERS:
        p = CLEAN_DIR / f"{t}_clean.parquet"
        if not p.exists():
            p = RAW_DIR / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)[["Close"]]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            last  = df["Close"].iloc[-1]
            prev  = df["Close"].iloc[-2] if len(df) > 1 else last
            chg   = last - prev
            chg_p = (chg / prev * 100) if prev else 0
            date  = df.index[-1].date()
            prices[t] = {
                "price":  last,
                "change": chg,
                "change_pct": chg_p,
                "date":   date,
            }
    return prices


@st.cache_data(ttl=900, show_spinner=False)
def get_realtime_prices(tickers: tuple) -> dict:
    """Cached wrapper — actual yfinance API call at most once per 15 minutes."""
    return fetch_all_live_prices(list(tickers))


@st.cache_data(ttl=300, show_spinner=False)
def get_live_sentiments(tickers: tuple) -> dict:
    """Cached wrapper — actual yfinance news fetch at most once per 5 minutes."""
    return sentiment_all(list(tickers))


def run_audit():
    results = []
    for t in TICKERS:
        raw_p   = RAW_DIR   / f"{t}.parquet"
        clean_p = CLEAN_DIR / f"{t}_clean.parquet"
        if not raw_p.exists() or not clean_p.exists():
            results.append({
                "Ticker": t, "Status": "MISSING", "Raw Rows": "-",
                "Clean Rows": "-", "NaN Prices": "-",
                "Duplicates": "-", "Max Gap (days)": "-",
            })
            continue
        raw   = pd.read_parquet(raw_p)
        clean = pd.read_parquet(clean_p)
        idx     = pd.to_datetime(clean.index).sort_values()
        max_gap = int(idx.to_series().diff().dt.days.max())
        nan_p   = int(clean[["Open", "High", "Low", "Close"]].isna().sum().sum())
        dupes   = int(clean.index.duplicated().sum())
        match   = len(raw) == len(clean)
        results.append({
            "Ticker":         t,
            "Status":         "PASS" if (match and nan_p == 0 and dupes == 0) else "FAIL",
            "Raw Rows":       f"{len(raw):,}",
            "Clean Rows":     f"{len(clean):,}",
            "NaN Prices":     nan_p,
            "Duplicates":     dupes,
            "Max Gap (days)": max_gap,
        })
    return pd.DataFrame(results)


def sidebar_slider_with_note(key, min_val, max_val, default, step=None):
    """Render a slider + an expandable note explaining the parameter."""
    info = PARAM_HELP[key]
    val = st.sidebar.slider(
        info["label"],
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step if step else 1,
    )
    with st.sidebar.expander("What does this do?"):
        st.caption(info["note"])
        st.caption(f"**Effect of changing:** {info['effect']}")
    return val


# -- Live price ticker bar (always visible at top) ----------------------------

def render_price_bar(tickers):
    live = get_live_prices()
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        if t not in live:
            cols[i].metric(t, "No data")
            continue
        p   = live[t]
        arrow = "+" if p["change"] >= 0 else ""
        cols[i].metric(
            label=f"{t}  —  {TICKER_NAMES.get(t, '')}",
            value=f"${p['price']:,.2f}",
            delta=f"{arrow}{p['change']:.2f}  ({arrow}{p['change_pct']:.2f}%)  as of {p['date']}",
            delta_color="normal",
        )


# -- Sidebar -------------------------------------------------------------------

st.sidebar.title("Configuration")

st.sidebar.subheader("Tickers")
selected_tickers = st.sidebar.multiselect("Select tickers", TICKERS, default=TICKERS)

st.sidebar.subheader("Backtest Window")
start_date = st.sidebar.date_input(
    "Start date", value=None,
    help="Leave blank to use each ticker's earliest available date"
)
end_date = st.sidebar.date_input(
    "End date", value=pd.Timestamp("2024-12-31").date(),
    help="Last date to include in the backtest"
)

st.sidebar.subheader("Strategy Parameters")
st.sidebar.caption(
    "These parameters control HOW and WHEN the bot buys. "
    "Expand 'What does this do?' under each slider to learn more."
)

monthly_budget = sidebar_slider_with_note("monthly_budget", 10,  500, 100, step=10)
oversold_rsi   = sidebar_slider_with_note("oversold_rsi",   20,  45,  35,  step=1)
below_sma_mult = sidebar_slider_with_note("below_sma_mult", 1.0, 3.0, 1.5, step=0.1)
oversold_mult  = sidebar_slider_with_note("oversold_mult",  1.5, 4.0, 2.0, step=0.5)
slippage_bps   = sidebar_slider_with_note("slippage_bps",   1,   20,  3,   step=1)

st.sidebar.subheader("Exit Strategy")
enable_exits   = st.sidebar.toggle("Enable Take-Profit / Stop-Loss", value=False)
with st.sidebar.expander("What does this do?"):
    st.caption(
        "When enabled, the bot automatically SELLS a position when it hits "
        "either the take-profit target (price rose enough above avg cost) "
        "or the stop-loss floor (price fell too far below avg cost). "
        "After a sell, DCA buys resume the following month."
    )
take_profit_pct = st.sidebar.slider("Take-profit threshold (%)", 10, 200, 50, step=5,
                                    disabled=not enable_exits) / 100
stop_loss_pct   = st.sidebar.slider("Stop-loss threshold (%)",   5,  50,  20, step=5,
                                    disabled=not enable_exits) / 100

st.sidebar.subheader("Portfolio Rebalancing")
enable_rebalance = st.sidebar.toggle("Enable Quarterly Rebalancing", value=False)
with st.sidebar.expander("What does this do?"):
    st.caption(
        "Checks every 3 months whether any ticker has drifted beyond the "
        "threshold from its target equal-weight allocation. If so, it sells "
        "the overweight ticker and buys the underweight one — keeping the "
        "portfolio balanced without manual intervention."
    )
rebalance_threshold = st.sidebar.slider("Drift threshold (%)", 5, 30, 10, step=5,
                                        disabled=not enable_rebalance) / 100

st.sidebar.divider()
run_pipeline = st.sidebar.button("Run Backtest", type="primary", width='stretch')
run_ingest   = st.sidebar.button("Re-download Data", width='stretch')

# -- Re-download ---------------------------------------------------------------

if run_ingest:
    with st.spinner("Downloading latest data from Yahoo Finance..."):
        ingest_all(TICKERS)
        clean_all(TICKERS)
        st.cache_data.clear()
    st.success("Data refreshed — prices are now up to date.")

# -- Main content --------------------------------------------------------------

st.title("Long-Term DCA Portfolio Dashboard")

active_tickers = selected_tickers if selected_tickers else TICKERS

# -- Live Feed (auto-refresh every 5 seconds; actual API calls rate-limited by TTL cache) --

_refresh_ms  = 60_000  # page rerun every 60 s — keeps ET clock ticking, reads from cache
_count       = st_autorefresh(interval=_refresh_ms, key="live_feed_refresh")

_now_et      = __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("America/New_York"))
_market_open = is_market_open()
_status_icon = "🟢" if _market_open else "🔴"
_status_text = "MARKET OPEN" if _market_open else "MARKET CLOSED"

lf_head, lf_badge, lf_ts = st.columns([4, 1.5, 2])
lf_head.subheader("Live Market Feed")
lf_badge.markdown(
    f"""<div style="margin-top:6px; padding:6px 12px; border-radius:20px;
        background:{'#1b5e2033' if _market_open else '#b71c1c33'};
        border:1px solid {'#4CAF50' if _market_open else '#F44336'};
        text-align:center; font-weight:700; font-size:0.85rem;
        color:{'#4CAF50' if _market_open else '#F44336'}">
        {_status_icon} {_status_text}
    </div>""",
    unsafe_allow_html=True,
)
lf_ts.markdown(
    f"<div style='margin-top:10px; font-size:0.8rem; color:#888;'>"
    f"ET: {_now_et.strftime('%H:%M:%S')}  |  Page run #{_count}<br>"
    f"<span style='color:#FF9800'>⚠ DATA DELAYED 15-20 MIN · Prices refresh every 15 min · News every 5 min</span></div>",
    unsafe_allow_html=True,
)

# Cached price fetch (API call at most every 60 s)
_rt_prices = get_realtime_prices(tuple(active_tickers))

if _rt_prices:
    lf_cols = st.columns(len(active_tickers))
    for _i, _t in enumerate(active_tickers):
        if _t not in _rt_prices:
            lf_cols[_i].metric(_t, "Unavailable")
            continue
        _p    = _rt_prices[_t]
        _arr  = "+" if _p["change"] >= 0 else ""
        _ts_s = _p["timestamp"].astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).strftime("%H:%M ET")
        lf_cols[_i].metric(
            label=f"{_t}  —  {TICKER_NAMES.get(_t, '')}",
            value=f"${_p['price']:,.2f}",
            delta=f"{_arr}{_p['change']:.2f}  ({_arr}{_p['change_pct']:.2f}%)  as of {_ts_s}",
            delta_color="normal",
        )
    st.caption("Prices refresh every 15 min via yfinance (Yahoo data is 15-20 min delayed — more frequent calls return the same value). News refreshes every 5 min.")
else:
    st.caption("No live price data available — run 'Re-download Data' or check your internet connection.")
    render_price_bar(active_tickers)

# Cached news sentiment (API call at most every 5 min)
with st.expander("Live News Sentiment Feed (refreshes every 5 min)", expanded=False):
    with st.spinner("Loading news..."):
        _live_sents = get_live_sentiments(tuple(active_tickers))

    _sc = st.columns(len(active_tickers))
    _color_map_s = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
    for _i, _t in enumerate(active_tickers):
        _s = _live_sents.get(_t, {})
        _score = _s.get("composite_score", 0.0)
        _label = _s.get("label", "Neutral")
        _bg    = _color_map_s.get(_label, "#9E9E9E")
        _sc[_i].markdown(
            f"""<div style="background:{_bg}22; border-left:4px solid {_bg};
                    padding:10px 14px; border-radius:6px; margin-bottom:6px;">
                <div style="font-weight:700; color:{_bg}">{_t} — {_label}</div>
                <div style="font-size:1.6rem; font-weight:800">{_score:+.3f}</div>
                <div style="font-size:0.75rem; color:#888">
                    {_s.get('positive_count',0)}▲ {_s.get('negative_count',0)}▼ {_s.get('neutral_count',0)}–
                    &nbsp;({_s.get('total_articles',0)} articles)
                </div></div>""",
            unsafe_allow_html=True,
        )

    # Top 3 headlines per ticker
    for _t in active_tickers:
        _s = _live_sents.get(_t, {})
        _arts = _s.get("articles", [])[:3]
        if _arts:
            st.markdown(f"**{_t} — latest headlines**")
            for _a in _arts:
                _c = _a["compound"]
                _clr = "#4CAF50" if _c >= 0.05 else "#F44336" if _c <= -0.05 else "#9E9E9E"
                _link = f"[{_a['title'][:90]}]({_a['url']})" if _a.get("url") else _a['title'][:90]
                st.markdown(
                    f"<span style='color:{_clr};font-weight:600'>{_a['label']} {_c:+.2f}</span>  "
                    f"— {_link}  <span style='color:#555;font-size:0.75rem'>{_a['published']}</span>",
                    unsafe_allow_html=True,
                )

st.divider()

# -- Run backtest or load cached results ---------------------------------------

if run_pipeline or (CLEAN_DIR / "trade_log.csv").exists():

    cfg = BotConfig(
        tickers=active_tickers,
        monthly_budget_usd=monthly_budget,
        oversold_rsi=oversold_rsi,
        below_sma_multiplier=below_sma_mult,
        oversold_multiplier=oversold_mult,
        slippage_bps=float(slippage_bps),
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date)     if end_date   else None,
        enable_exits=enable_exits,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        enable_rebalance=enable_rebalance,
        rebalance_threshold_pct=rebalance_threshold,
    )

    def _cached_tickers_match() -> bool:
        """Return True if the session-state trade_log covers exactly active_tickers."""
        if "trade_log" not in st.session_state:
            return False
        cached_tickers = set(st.session_state["trade_log"]["ticker"].unique())
        return cached_tickers == set(active_tickers)

    needs_run = run_pipeline or not _cached_tickers_match()

    if needs_run:
        with st.spinner("Running backtest..."):
            clean_all(cfg.tickers)
            bot       = LongTermDCABot(cfg)
            trade_log = bot.run()
            summary   = bot.summary(trade_log)
            equity    = bot.equity_curve(trade_log)
            trade_log.to_csv(CLEAN_DIR / "trade_log.csv", index=False)
            summary.to_csv(CLEAN_DIR / "summary.csv")
            equity.to_csv(CLEAN_DIR / "equity_curve.csv")
            st.session_state["trade_log"] = trade_log
            st.session_state["summary"]   = summary
            st.session_state["equity"]    = equity
    else:
        if "trade_log" not in st.session_state:
            trade_log = pd.read_csv(CLEAN_DIR / "trade_log.csv", parse_dates=["date"])
            summary   = pd.read_csv(CLEAN_DIR / "summary.csv", index_col="ticker")
            equity    = pd.read_csv(CLEAN_DIR / "equity_curve.csv", index_col=0, parse_dates=True)
            st.session_state["trade_log"] = trade_log
            st.session_state["summary"]   = summary
            st.session_state["equity"]    = equity

    trade_log = st.session_state["trade_log"]
    summary   = st.session_state["summary"]
    equity    = st.session_state["equity"]

    # -- Portfolio KPIs --------------------------------------------------------

    st.subheader("Portfolio Summary")

    total_invested = summary["total_invested_usd"].sum()
    total_value    = summary["market_value_usd"].sum()
    total_pnl      = total_value - total_invested
    total_pnl_pct  = (total_pnl / total_invested * 100) if total_invested else 0
    total_trades   = len(trade_log)
    total_fees     = trade_log["fee_usd"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Invested",  f"${total_invested:,.0f}")
    k2.metric("Portfolio Value", f"${total_value:,.0f}")
    k3.metric("Unrealised P&L",  f"${total_pnl:,.0f}", f"{total_pnl_pct:+.2f}%")
    k4.metric("Total Trades",    f"{total_trades:,}")
    k5.metric("Total Fees Paid", f"${total_fees:,.4f}")

    # -- Risk Metrics ----------------------------------------------------------

    st.subheader("Risk & Performance Metrics")
    st.caption(
        "Computed from the daily equity curve. "
        "Risk-free rate: 4% per year (US T-bill). "
        "Sharpe/Sortino > 1.0 is considered good; > 2.0 is excellent."
    )

    total_eq = equity["total_portfolio_usd"] if "total_portfolio_usd" in equity.columns else None
    if total_eq is not None:
        total_eq.index = pd.to_datetime(total_eq.index, utc=True)
        raw_metrics    = compute_all(total_eq, risk_free_rate=0.04)
        pretty_metrics = format_metrics(raw_metrics)

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Total Return",  pretty_metrics["Total Return"])
        rm2.metric("CAGR",          pretty_metrics["CAGR"])
        rm3.metric("Sharpe Ratio",  pretty_metrics["Sharpe Ratio"])
        rm4.metric("Sortino Ratio", pretty_metrics["Sortino Ratio"])

        rm5, rm6, rm7, rm8 = st.columns(4)
        rm5.metric("Max Drawdown",        pretty_metrics["Max Drawdown"])
        rm6.metric("Calmar Ratio",        pretty_metrics["Calmar Ratio"])
        rm7.metric("Ann. Volatility",     pretty_metrics["Annualised Volatility"])
        rm8.metric("Monthly Win Rate",    pretty_metrics["Monthly Win Rate"])

        with st.expander("What do these metrics mean?"):
            st.markdown(
                "**Total Return**: overall gain from first trade to end date.  \n"
                "**CAGR**: Compound Annual Growth Rate — how much the portfolio grew per year on average.  \n"
                "**Sharpe Ratio**: return per unit of total risk. >1 = good, >2 = excellent.  \n"
                "**Sortino Ratio**: like Sharpe but only penalises downside volatility. Higher = better.  \n"
                "**Max Drawdown**: worst peak-to-trough loss (e.g. -34% means the portfolio fell 34% from its peak at worst).  \n"
                "**Calmar Ratio**: CAGR divided by max drawdown. Higher = better risk-adjusted return.  \n"
                "**Ann. Volatility**: standard deviation of daily returns, annualised. Lower = smoother ride.  \n"
                "**Monthly Win Rate**: percentage of months where the portfolio value was higher than the prior month."
            )

    st.divider()

    # -- Current strategy settings reminder ------------------------------------

    with st.expander("Current strategy settings in this backtest"):
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Monthly Budget", f"${monthly_budget}")
        s2.metric("RSI Oversold at", f"< {oversold_rsi}")
        s3.metric("Below SMA200 Buy", f"{below_sma_mult}x")
        s4.metric("RSI Crash Buy",    f"{oversold_mult}x")
        s5.metric("Slippage",         f"{slippage_bps} bps")
        st.caption(
            f"Period: {'earliest available' if not start_date else start_date}"
            f" to {end_date}. "
            "Change sliders in the sidebar and click **Run Backtest** to see how "
            "different parameters affect the result."
        )

    st.divider()

    # -- Equity Curve ----------------------------------------------------------

    st.subheader("Portfolio Equity Curve")

    equity.index = pd.to_datetime(equity.index, utc=True)
    value_cols   = [c for c in equity.columns if c != "total_portfolio_usd"]

    tab_combined, tab_individual = st.tabs(["Combined Portfolio", "Per Ticker"])

    with tab_combined:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity["total_portfolio_usd"],
            name="DCA Strategy (total)",
            line=dict(color="#9C27B0", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.08)",
        ))

        # Buy-and-hold benchmark: invest same monthly budget, no signal logic
        bah_total = None
        for _t in active_tickers:
            _bah = buy_and_hold_equity(
                ticker=_t,
                monthly_budget_usd=monthly_budget,
                clean_dir=CLEAN_DIR,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
                slippage_bps=float(slippage_bps),
                clearing_fee_usd=0.005,
            )
            if not _bah.empty:
                _bah.index = pd.to_datetime(_bah.index, utc=True)
                bah_total  = _bah if bah_total is None else bah_total.add(_bah, fill_value=0)

        if bah_total is not None:
            fig.add_trace(go.Scatter(
                x=bah_total.index,
                y=bah_total.values,
                name="Buy & Hold Benchmark",
                line=dict(color="#FF9800", width=1.8, dash="dash"),
            ))

        fig.update_layout(
            yaxis_title="Portfolio Value (USD)",
            xaxis_title="Date",
            hovermode="x unified",
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.caption(
            "**Purple (solid)** = DCA strategy with RSI + SMA200 triggers.  "
            "**Orange (dashed)** = Buy & Hold benchmark — same monthly budget, "
            "invested on the 1st trading day of each month, no signal logic."
        )
        st.plotly_chart(fig, width='stretch')

    with tab_individual:
        fig2 = go.Figure()
        for col in value_cols:
            ticker = col.replace("_value", "")
            fig2.add_trace(go.Scatter(
                x=equity.index,
                y=equity[col],
                name=ticker,
                line=dict(color=TICKER_COLORS.get(ticker, "#888"), width=2),
            ))
        fig2.update_layout(
            yaxis_title="Value (USD)",
            xaxis_title="Date",
            hovermode="x unified",
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, width='stretch')

    st.divider()

    # -- Ticker Summary + Allocation -------------------------------------------

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Ticker Breakdown")
        disp = summary.copy()
        for col in ["total_invested_usd", "total_proceeds_usd", "net_invested_usd",
                    "market_value_usd", "realized_pnl_usd",
                    "unrealized_pnl_usd", "total_pnl_usd"]:
            if col in disp.columns:
                disp[col] = disp[col].map("${:,.2f}".format)
        if "avg_cost_per_share" in disp.columns:
            disp["avg_cost_per_share"] = disp["avg_cost_per_share"].map("${:,.4f}".format)
        if "last_close" in disp.columns:
            disp["last_close"] = disp["last_close"].map("${:,.2f}".format)
        if "total_pnl_pct" in disp.columns:
            disp["total_pnl_pct"] = disp["total_pnl_pct"].map("{:+.2f}%".format)
        st.dataframe(disp, width='stretch')

    with col_right:
        st.subheader("Allocation by Market Value")
        alloc = summary["market_value_usd"].reset_index()
        alloc.columns = ["ticker", "value"]
        fig3 = px.pie(
            alloc, values="value", names="ticker",
            color="ticker", color_discrete_map=TICKER_COLORS, hole=0.45,
        )
        fig3.update_traces(textposition="outside", textinfo="percent+label")
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig3, width='stretch')

    st.divider()

    # -- Price History + Trade Entry Points ------------------------------------

    st.subheader("Price History + Trade Entry Points")
    st.caption(
        "Circle = normal monthly DCA  |  "
        "Triangle = below 200-day average (1.5x buy)  |  "
        "Star = RSI oversold crash (2x buy)"
    )

    price_ticker = st.selectbox("Select ticker to inspect", active_tickers)

    price_path = CLEAN_DIR / f"{price_ticker}_clean.parquet"
    if price_path.exists():
        price_df = pd.read_parquet(price_path)
        price_df.index = pd.to_datetime(price_df.index, utc=True)
        if end_date:
            price_df = price_df[price_df.index <= pd.Timestamp(str(end_date), tz="UTC")]
        if start_date:
            price_df = price_df[price_df.index >= pd.Timestamp(str(start_date), tz="UTC")]

        t_trades = trade_log[trade_log["ticker"] == price_ticker].copy()
        t_trades["date"] = pd.to_datetime(t_trades["date"], utc=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=price_df.index, y=price_df["Close"],
            name="Close Price",
            line=dict(color=TICKER_COLORS.get(price_ticker, "#888"), width=1.5),
        ))

        marker_styles = [
            ("DCA_NORMAL",        "circle",      "#2196F3", "Normal DCA (1x)"),
            ("BELOW_SMA200_1.5X", "triangle-up", "#FF9800", "Below SMA200 (1.5x)"),
            ("RSI_OVERSOLD_2X",   "star",        "#F44336", "RSI Oversold (2x)"),
        ]
        for trigger, symbol, color, label in marker_styles:
            sub = t_trades[t_trades["trigger"] == trigger]
            if not sub.empty:
                fig4.add_trace(go.Scatter(
                    x=sub["date"], y=sub["fill_price"],
                    mode="markers", name=label,
                    marker=dict(symbol=symbol, size=9, color=color,
                                line=dict(width=1, color="white")),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Fill Price: $%{y:.2f}<br>"
                        "Budget: $%{customdata:.0f}<extra></extra>"
                    ),
                    customdata=sub["budget_usd"].values,
                ))

        fig4.update_layout(
            yaxis_title="Price (USD)", xaxis_title="Date",
            hovermode="x unified", height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig4, width='stretch')

    st.divider()

    # -- Trade Log -------------------------------------------------------------

    st.subheader("Trade Log")

    c_tick, c_trig = st.columns(2)
    with c_tick:
        filter_ticker = st.multiselect(
            "Filter by ticker",
            options=trade_log["ticker"].unique().tolist(),
            default=trade_log["ticker"].unique().tolist(),
        )
    with c_trig:
        filter_trigger = st.multiselect(
            "Filter by trigger",
            options=trade_log["trigger"].unique().tolist(),
            default=trade_log["trigger"].unique().tolist(),
        )

    filtered = trade_log[
        trade_log["ticker"].isin(filter_ticker) &
        trade_log["trigger"].isin(filter_trigger)
    ].copy()
    filtered["budget_usd"] = filtered["budget_usd"].map("${:.2f}".format)
    filtered["fill_price"] = filtered["fill_price"].map("${:.4f}".format)
    filtered["fee_usd"]    = filtered["fee_usd"].map("${:.6f}".format)
    if "shares_transacted" in filtered.columns:
        filtered["shares_transacted"] = filtered["shares_transacted"].map("{:.6f}".format)

    st.dataframe(filtered, width='stretch', height=350)
    st.caption(f"Showing {len(filtered):,} of {len(trade_log):,} trades")
    st.download_button(
        "Download Trade Log CSV",
        trade_log.to_csv(index=False).encode("utf-8"),
        "trade_log.csv", "text/csv", width='stretch',
    )

    st.divider()

    # -- Trigger breakdown + monthly deployment --------------------------------

    st.subheader("Trade Trigger Breakdown")
    ch1, ch2 = st.columns(2)

    with ch1:
        trigger_counts = (
            trade_log.groupby(["ticker", "trigger"])["budget_usd"]
            .sum().reset_index()
        )
        fig5 = px.bar(
            trigger_counts, x="ticker", y="budget_usd",
            color="trigger", barmode="group",
            labels={"budget_usd": "Capital Deployed ($)", "ticker": "Ticker"},
            title="Capital Deployed by Trigger Type",
            color_discrete_sequence=["#2196F3", "#FF9800", "#F44336"],
        )
        fig5.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig5, width='stretch')

    with ch2:
        monthly_inv = (
            trade_log
            .assign(month=pd.to_datetime(trade_log["date"]).dt.to_period("M").astype(str))
            .groupby("month")["budget_usd"].sum().reset_index()
        )
        fig6 = px.bar(
            monthly_inv, x="month", y="budget_usd",
            labels={"budget_usd": "Capital Deployed ($)", "month": "Month"},
            title="Monthly Capital Deployment",
            color_discrete_sequence=["#9C27B0"],
        )
        fig6.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig6, width='stretch')

    st.divider()

    # -- Data Integrity Audit --------------------------------------------------

    st.subheader("Data Integrity Audit")
    audit_df = run_audit()
    for _, row in audit_df.iterrows():
        icon = "PASS" if row["Status"] == "PASS" else "FAIL"
        st.write(
            f"**[{icon}] {row['Ticker']}** — "
            f"{row['Raw Rows']} raw rows | {row['Clean Rows']} clean rows | "
            f"NaN prices: {row['NaN Prices']} | Duplicates: {row['Duplicates']} | "
            f"Max gap: {row['Max Gap (days)']} days"
        )
    if (audit_df["Status"] == "PASS").all():
        st.success("ALL CHECKS PASSED — 0% Data Loss Confirmed")
    else:
        st.error("One or more audit checks FAILED — review data before trading")

    st.divider()

    # -- Next Session Technical Forecast ---------------------------------------

    # Work out next trading day from last available data date
    import datetime
    _last_dates = []
    for _t in active_tickers:
        _p = CLEAN_DIR / f"{_t}_clean.parquet"
        if _p.exists():
            _df = pd.read_parquet(_p)
            _last_dates.append(pd.to_datetime(_df.index).max())
    _last_data_date = max(_last_dates).date() if _last_dates else datetime.date.today()

    # Next trading day = skip weekends
    _next_td = _last_data_date + datetime.timedelta(days=1)
    while _next_td.weekday() >= 5:   # 5=Sat, 6=Sun
        _next_td += datetime.timedelta(days=1)

    st.subheader("Next Session Technical Forecast")
    st.info(
        f"Data last updated: **{_last_data_date.strftime('%A, %B %d %Y')}**  |  "
        f"Forecasting for: **{_next_td.strftime('%A, %B %d %Y')}**  |  "
        f"Note: does not account for holidays — check market calendar if near a public holiday."
    )
    st.warning(
        "This is signal-based technical analysis — NOT a price prediction. "
        "No model can accurately predict a single day's closing price. "
        "Use this as one input among many, never as a sole trading decision."
    )

    forecasts = forecast_all(active_tickers, use_sentiment=True)

    for ticker, f in forecasts.items():
        if "error" in f:
            st.error(f"{ticker}: {f['error']}")
            continue

        bias_color = {
            "Bullish": "green", "Mildly Bullish": "green",
            "Bearish": "red",   "Mildly Bearish": "red",
            "Neutral": "gray",
        }.get(f["overall_bias"], "gray")

        with st.expander(
            f"{ticker}  |  Based on close ${f['last_close']:,.2f} ({f['last_date']})"
            f"  ->  Forecast for {_next_td.strftime('%a %b %d')}  "
            f"|  Bias: {f['overall_bias']}  |  Confidence: {f['confidence_pct']}%",
            expanded=True,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Low",  f"${f['expected_low']:,.2f}")
            c2.metric("Midpoint Bias", f"${f['expected_mid']:,.2f}")
            c3.metric("Expected High", f"${f['expected_high']:,.2f}")
            c4.metric("ATR (volatility)", f"${f['atr_14']:,.2f}")

            st.caption(
                f"Range built from 14-day Average True Range (ATR = ${f['atr_14']:.2f}). "
                f"The midpoint is skewed {f['overall_bias'].lower()} by signal consensus."
            )

            # -- Sentiment influence callout -----------------------------------
            _sent_score = f.get("sentiment_score")
            if _sent_score is not None:
                _sent_color = (
                    "#4CAF50" if _sent_score >= 0.15 else
                    "#F44336" if _sent_score <= -0.15 else
                    "#9E9E9E"
                )
                _sent_label = (
                    "Bullish" if _sent_score >= 0.15 else
                    "Bearish" if _sent_score <= -0.15 else
                    "Neutral — not moving the forecast"
                )
                st.markdown(
                    f"<div style='border-left:4px solid {_sent_color}; padding:8px 14px; "
                    f"background:{_sent_color}22; border-radius:4px; margin:8px 0'>"
                    f"<b style='color:{_sent_color}'>News Sentiment: {_sent_label}</b> "
                    f"&nbsp;·&nbsp; VADER composite score: <b>{_sent_score:+.3f}</b><br>"
                    f"<span style='font-size:0.8rem;color:#aaa'>"
                    f"Threshold: |score| ≥ 0.15 to influence forecast. "
                    f"Counts as 1 of {len(f['signals'])} total signals.</span></div>",
                    unsafe_allow_html=True,
                )

            # Signal score bar chart
            sig_df = pd.DataFrame(f["signals"])
            color_map = {"Bullish": "#4CAF50", "Bearish": "#F44336", "Neutral": "#9E9E9E"}
            sig_df["color"] = sig_df["bias"].map(color_map)

            fig_sig = go.Figure(go.Bar(
                x=sig_df["signal"],
                y=sig_df["score"],
                marker_color=sig_df["color"].tolist(),
                text=sig_df["value"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Value: %{text}<br>Score: %{y}<extra></extra>",
            ))
            fig_sig.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
            fig_sig.update_layout(
                title=f"{ticker} — Signal Scorecard "
                      f"({f['bull_signals']} bullish / {f['bear_signals']} bearish / "
                      f"{f['neutral_signals']} neutral)",
                yaxis=dict(tickvals=[-1, 0, 1],
                           ticktext=["Bearish", "Neutral", "Bullish"],
                           range=[-1.5, 1.5]),
                height=320,
                margin=dict(l=0, r=0, t=40, b=60),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_sig, width='stretch')

            # Key levels table
            kl1, kl2 = st.columns(2)
            with kl1:
                st.markdown("**Support Levels**")
                st.write(f"S1 (Bollinger Lower): ${f['support_1']:,.2f}")
                st.write(f"S2 (SMA200):          ${f['support_2']:,.2f}")
            with kl2:
                st.markdown("**Resistance Levels**")
                st.write(f"R1 (Bollinger Upper): ${f['resistance_1']:,.2f}")
                st.write(f"R2 (SMA20/50 max):    ${f['resistance_2']:,.2f}")

            # Indicator table
            ind_data = {
                "Indicator": ["RSI(14)", "RSI(21)", "SMA20", "SMA50", "SMA200",
                               "MACD", "MACD Signal", "Stoch K", "Stoch D",
                               "5d Momentum", "20d Momentum", "Volume vs Avg"],
                "Value": [
                    f"{f['rsi_14']}", f"{f['rsi_21']}",
                    f"${f['sma20']:,.2f}", f"${f['sma50']:,.2f}", f"${f['sma200']:,.2f}",
                    f"{f['macd']:.4f}", f"{f['macd_signal']:.4f}",
                    f"{f['stoch_k']}", f"{f['stoch_d']}",
                    f"{f['mom_5d']:+.2f}%", f"{f['mom_20d']:+.2f}%",
                    f"{f['vol_ratio']:.2f}x",
                ],
            }
            st.dataframe(pd.DataFrame(ind_data), width='stretch', hide_index=True)

    st.divider()

    # -- News Sentiment --------------------------------------------------------

    st.subheader("News Sentiment Analysis")
    st.caption(
        "Headlines fetched live from Yahoo Finance and scored with VADER "
        "(Valence Aware Dictionary and sEntiment Reasoner). "
        "Compound score: +1.0 = most positive, -1.0 = most negative, 0 = neutral. "
        "Positive >= +0.05, Negative <= -0.05."
    )

    with st.spinner("Fetching latest news..."):
        sentiments = sentiment_all(active_tickers)

    # -- Sentiment overview cards (one per ticker) ----------------------------

    sent_cols = st.columns(len(active_tickers))
    for i, ticker in enumerate(active_tickers):
        s = sentiments[ticker]
        score = s["composite_score"]
        label = s["label"]

        color_map = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}
        bg_color  = color_map.get(label, "#9E9E9E")

        sent_cols[i].markdown(
            f"""
            <div style="background:{bg_color}22; border-left:4px solid {bg_color};
                        padding:12px 16px; border-radius:6px;">
                <div style="font-size:1.1rem; font-weight:700; color:{bg_color}">
                    {ticker} — {label}
                </div>
                <div style="font-size:2rem; font-weight:800; margin:4px 0">
                    {score:+.3f}
                </div>
                <div style="font-size:0.8rem; color:#888;">
                    {s['positive_count']} positive &nbsp;·&nbsp;
                    {s['negative_count']} negative &nbsp;·&nbsp;
                    {s['neutral_count']} neutral
                    &nbsp;({s['total_articles']} articles)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Combined sentiment bar chart -----------------------------------------

    sent_chart_data = []
    for ticker in active_tickers:
        s = sentiments[ticker]
        for a in s["articles"]:
            sent_chart_data.append({
                "ticker":    ticker,
                "title":     a["title"][:60] + ("…" if len(a["title"]) > 60 else ""),
                "compound":  a["compound"],
                "label":     a["label"],
                "published": a["published"],
            })

    if sent_chart_data:
        chart_df = pd.DataFrame(sent_chart_data)
        fig_sent_bar = go.Figure()
        color_map_bar = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}

        for ticker in active_tickers:
            sub = chart_df[chart_df["ticker"] == ticker]
            if sub.empty:
                continue
            fig_sent_bar.add_trace(go.Bar(
                name=ticker,
                x=sub["published"],
                y=sub["compound"],
                marker_color=[color_map_bar.get(l, "#9E9E9E") for l in sub["label"]],
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Score: %{y:+.3f}<br>"
                    "Published: %{x}<extra></extra>"
                ),
                customdata=sub["title"],
                visible=True,
            ))

        fig_sent_bar.add_hline(y=0.05,  line_dash="dot", line_color="#4CAF50",
                               opacity=0.4, annotation_text="Positive threshold")
        fig_sent_bar.add_hline(y=-0.05, line_dash="dot", line_color="#F44336",
                               opacity=0.4, annotation_text="Negative threshold")
        fig_sent_bar.update_layout(
            title="Headline Sentiment Scores (hover for title)",
            yaxis_title="VADER Compound Score",
            xaxis_title="Published",
            height=380,
            barmode="group",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(range=[-1.1, 1.1]),
        )
        st.plotly_chart(fig_sent_bar, width='stretch')

    # -- Per-ticker sentiment detail + price impact ---------------------------

    st.markdown("#### Sentiment vs Price Movement (last 30 trading days)")
    st.caption(
        "Shows where news sentiment was recorded alongside the next-day price change. "
        "A positive sentiment day followed by a price rise suggests alignment; "
        "but correlation is not causation — many factors move prices."
    )

    sent_ticker_sel = st.selectbox(
        "Select ticker to inspect sentiment detail",
        active_tickers,
        key="sent_ticker_sel",
    )

    svp_df = sentiment_vs_price(sent_ticker_sel, CLEAN_DIR)

    if not svp_df.empty and "sentiment_score" in svp_df.columns:
        svp_df["Date"] = pd.to_datetime(svp_df["Date"])
        svp_df_plot    = svp_df.dropna(subset=["Close"])

        fig_svp = go.Figure()

        # Price line (left axis)
        fig_svp.add_trace(go.Scatter(
            x=svp_df_plot["Date"], y=svp_df_plot["Close"],
            name="Close Price",
            line=dict(color=TICKER_COLORS.get(sent_ticker_sel, "#888"), width=2),
            yaxis="y1",
        ))

        # Sentiment bars (right axis) — only days with news
        svp_with_sent = svp_df_plot.dropna(subset=["sentiment_score"])
        if not svp_with_sent.empty:
            bar_colors = [
                "#4CAF50" if v >= 0.05 else "#F44336" if v <= -0.05 else "#9E9E9E"
                for v in svp_with_sent["sentiment_score"]
            ]
            fig_svp.add_trace(go.Bar(
                x=svp_with_sent["Date"],
                y=svp_with_sent["sentiment_score"],
                name="Sentiment Score",
                marker_color=bar_colors,
                opacity=0.75,
                yaxis="y2",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Sentiment: %{y:+.3f}<extra></extra>",
            ))

        fig_svp.update_layout(
            title=f"{sent_ticker_sel} — Close Price vs Daily News Sentiment",
            xaxis_title="Date",
            yaxis=dict(title="Close Price (USD)", side="left"),
            yaxis2=dict(
                title="Sentiment Score",
                side="right",
                overlaying="y",
                range=[-1.2, 1.2],
                showgrid=False,
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            height=420,
            hovermode="x unified",
            margin=dict(l=0, r=60, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            barmode="overlay",
        )
        st.plotly_chart(fig_svp, width='stretch')

        # Correlation stat
        svp_corr = svp_with_sent.dropna(subset=["pct_change"])
        if len(svp_corr) >= 3:
            corr = svp_corr["sentiment_score"].corr(svp_corr["pct_change"])
            if pd.isna(corr):
                corr_label = "insufficient data"
            else:
                corr_label = (
                    "strong positive" if corr > 0.5 else
                    "moderate positive" if corr > 0.2 else
                    "weak positive" if corr > 0 else
                    "strong negative" if corr < -0.5 else
                    "moderate negative" if corr < -0.2 else
                    "weak negative"
                )
            st.info(
                f"**{sent_ticker_sel} sentiment vs same-day price change correlation: "
                f"{'N/A' if pd.isna(corr) else f'{corr:+.3f}'}** ({corr_label}).  \n"
                "Note: same-day correlation is typically weak because news and price "
                "both react to events simultaneously. Next-day predictive power is "
                "what matters — and even that rarely exceeds ±0.15 for large-cap stocks."
            )

    # -- Full headlines table per ticker --------------------------------------

    st.markdown("#### Recent Headlines")
    hl_ticker = st.selectbox(
        "Select ticker to read headlines",
        active_tickers,
        key="hl_ticker_sel",
    )

    s = sentiments[hl_ticker]
    if s.get("error"):
        st.warning(f"No news found for {hl_ticker}: {s['error']}")
    else:
        rows = []
        for a in s["articles"]:
            rows.append({
                "Published":  a["published"],
                "Sentiment":  f"{a['label']} ({a['compound']:+.2f})",
                "Publisher":  a["publisher"],
                "Headline":   a["title"],
                "URL":        a["url"],
            })
        hl_df = pd.DataFrame(rows)

        # Colour-code the Sentiment column using a style function
        def _colour_sentiment(val: str) -> str:
            if val.startswith("Positive"):
                return "color: #4CAF50; font-weight: 600"
            if val.startswith("Negative"):
                return "color: #F44336; font-weight: 600"
            return "color: #9E9E9E"

        styled = hl_df.drop(columns=["URL"]).style.map(
            _colour_sentiment, subset=["Sentiment"]
        )
        st.dataframe(styled, width='stretch', hide_index=True, height=380)

        # Clickable links separately
        with st.expander("Article links"):
            for a in s["articles"]:
                st.markdown(f"- [{a['title'][:80]}]({a['url']})")

    st.divider()
    st.caption(
        "Sentiment disclaimer: VADER scores are computed on headline + summary text only. "
        "Sentiment analysis is not a reliable standalone trading signal. "
        "A positive sentiment score does not mean the stock will rise."
    )

# -- Landing screen ------------------------------------------------------------

else:
    st.info(
        "Click **Run Backtest** in the sidebar to start, "
        "or **Re-download Data** to fetch fresh data from Yahoo Finance first.",
        icon="👈",
    )

    # Still show live prices on landing
    if any((RAW_DIR / f"{t}.parquet").exists() for t in TICKERS):
        st.subheader("Live Market Prices")
        render_price_bar(active_tickers)
        st.divider()

    raw_files = [RAW_DIR / f"{t}.parquet" for t in TICKERS]
    if any(p.exists() for p in raw_files):
        st.subheader("Cached Data Available")
        for t in TICKERS:
            p = RAW_DIR / f"{t}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                idx = pd.to_datetime(df.index)
                st.write(
                    f"**{t}** ({TICKER_NAMES.get(t, '')}) — "
                    f"{len(df):,} trading days "
                    f"({idx.min().date()} to {idx.max().date()})"
                )
    else:
        st.warning("No data found. Click **Re-download Data** to begin.")

    # Parameter guide on landing page
    st.divider()
    st.subheader("Strategy Parameter Guide")
    st.caption("Learn what each slider does before running your backtest.")
    for key, info in PARAM_HELP.items():
        with st.expander(info["label"]):
            st.write(info["note"])
            st.info(f"**Effect of changing:** {info['effect']}")
