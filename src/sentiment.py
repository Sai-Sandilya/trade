"""
sentiment.py - News sentiment analysis for individual tickers.

Fetches recent news headlines via yfinance, scores each headline using
VADER (Valence Aware Dictionary and sEntiment Reasoner) and returns
a structured sentiment report per ticker.

VADER is purpose-built for short, informal text (news headlines, social
media). It returns a compound score in [-1, +1]:
  >= +0.05  -> Positive
  <= -0.05  -> Negative
  between   -> Neutral

No API key required. All data comes from Yahoo Finance (already used
in ingestion.py) and runs fully offline once headlines are fetched.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial sentiment lexicon
#
# VADER's default lexicon was trained on social media text and misreads many
# financial terms. For example:
#   "AMD misses earnings but guidance raised"
#   → VADER scores "raised" as positive and "misses" as negative, but
#     "earnings miss" as a phrase has no entry, so the net score is near zero.
#
# This lexicon patches VADER with domain-specific scores for financial phrases.
# Scores are in VADER's [-4, +4] range (not compound — compound is derived).
# Sources: calibrated against known market reactions in financial research.
# ---------------------------------------------------------------------------
_FINANCIAL_LEXICON: dict[str, float] = {
    # ── Strong positive ──────────────────────────────────────────────────────
    "beats estimates":          3.5,
    "beats expectations":       3.5,
    "earnings beat":            3.5,
    "revenue beat":             3.2,
    "guidance raised":          3.2,
    "raises guidance":          3.2,
    "raises outlook":           3.2,
    "record revenue":           3.0,
    "record earnings":          3.0,
    "record profit":            3.0,
    "blowout quarter":          3.5,
    "strong demand":            2.5,
    "market share gains":       2.5,
    "dividend increase":        2.5,
    "dividend raised":          2.5,
    "share buyback":            2.0,
    "stock buyback":            2.0,
    "repurchase program":       2.0,
    "upgraded":                 2.5,
    "upgrade":                  2.0,
    "outperform":               2.5,
    "strong buy":               3.0,
    "price target raised":      2.8,
    "target raised":            2.5,
    "analyst upgrade":          2.5,
    "bullish":                  2.5,
    "all-time high":            2.5,
    "52-week high":             2.0,
    "market rally":             2.0,
    "surges":                   2.5,
    "soars":                    2.5,
    "jumps":                    2.0,
    "acquisition":              1.5,
    "partnership":              1.5,
    "expansion":                1.5,

    # ── Mild positive ────────────────────────────────────────────────────────
    "in line with estimates":   1.0,
    "meets expectations":       1.0,
    "resilient":                1.0,
    "recovery":                 1.0,
    "rebound":                  1.0,
    "solid results":            1.5,
    "steady growth":            1.5,

    # ── Strong negative ──────────────────────────────────────────────────────
    "misses estimates":        -3.5,
    "misses expectations":     -3.5,
    "earnings miss":           -3.5,
    "revenue miss":            -3.2,
    "guidance cut":            -3.5,
    "guidance lowered":        -3.2,
    "lowers guidance":         -3.2,
    "lowers outlook":          -3.2,
    "profit warning":          -3.5,
    "revenue warning":         -3.0,
    "mass layoffs":            -3.0,
    "layoffs":                 -2.5,
    "job cuts":                -2.5,
    "restructuring":           -2.0,
    "accounting irregularities": -4.0,
    "fraud":                   -4.0,
    "sec investigation":       -3.5,
    "class action":            -3.0,
    "bankruptcy":              -4.0,
    "chapter 11":              -4.0,
    "debt default":            -4.0,
    "downgraded":              -2.5,
    "downgrade":               -2.0,
    "underperform":            -2.5,
    "sell rating":             -2.5,
    "price target cut":        -2.8,
    "target cut":              -2.5,
    "analyst downgrade":       -2.5,
    "bearish":                 -2.5,
    "52-week low":             -2.0,
    "plunges":                 -2.5,
    "tumbles":                 -2.5,
    "crashes":                 -3.0,
    "market crash":            -3.0,
    "recession":               -2.5,
    "inflation":               -1.5,
    "rate hike":               -1.5,
    "interest rate hike":      -2.0,
    "supply chain issues":     -2.0,
    "margin compression":      -2.5,
    "slowing growth":          -2.0,
    "decelerating":            -1.5,
    "competition intensifies": -1.5,
    "market share loss":       -2.5,
    "inventory glut":          -2.0,
    "demand weakness":         -2.5,

    # ── Mild negative ────────────────────────────────────────────────────────
    "in line":                  0.0,
    "mixed results":           -0.5,
    "cautious outlook":        -1.0,
    "headwinds":               -1.5,
    "uncertainty":             -1.0,
    "volatility":              -0.5,
}

_analyzer = SentimentIntensityAnalyzer()
# Patch the default VADER lexicon with financial domain terms
_analyzer.lexicon.update(_FINANCIAL_LEXICON)

# Compound score thresholds (standard VADER guidance)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

# Labels
LABEL_POSITIVE = "Positive"
LABEL_NEGATIVE = "Negative"
LABEL_NEUTRAL  = "Neutral"


def _compound_to_label(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return LABEL_POSITIVE
    if compound <= NEGATIVE_THRESHOLD:
        return LABEL_NEGATIVE
    return LABEL_NEUTRAL


def _score_text(text: str) -> dict:
    """Run VADER on a string, return scores dict with label added."""
    scores = _analyzer.polarity_scores(text)
    scores["label"] = _compound_to_label(scores["compound"])
    return scores


def _parse_article(raw: dict) -> Optional[dict]:
    """
    Extract fields from a yfinance news item.
    yfinance returns nested content dicts; handle both old and new schema.
    Returns None if the article cannot be parsed.
    """
    try:
        content   = raw.get("content", raw)          # new schema wraps in "content"
        title     = content.get("title", "").strip()
        summary   = content.get("summary", "").strip()
        publisher = (
            content.get("provider", {}).get("displayName", "")
            or content.get("publisher", "")
        )
        url = (
            content.get("previewUrl", "")
            or content.get("canonicalUrl", {}).get("url", "")
            or content.get("link", "")
        )
        pub_date_str = content.get("pubDate") or content.get("providerPublishTime")

        if not title:
            return None

        # Parse publish date
        if isinstance(pub_date_str, (int, float)):
            pub_dt = datetime.fromtimestamp(pub_date_str, tz=timezone.utc)
        elif isinstance(pub_date_str, str):
            pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        else:
            pub_dt = datetime.now(tz=timezone.utc)

        # Score title + summary together for richer signal
        text_to_score = f"{title}. {summary}" if summary else title
        scores = _score_text(text_to_score)

        return {
            "title":     title,
            "summary":   summary,
            "publisher": publisher,
            "url":       url,
            "published": pub_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "published_dt": pub_dt,
            "compound":  round(scores["compound"], 4),
            "positive":  round(scores["pos"], 4),
            "negative":  round(scores["neg"], 4),
            "neutral":   round(scores["neu"], 4),
            "label":     scores["label"],
        }
    except Exception as exc:
        logger.debug("Could not parse article: %s", exc)
        return None


def fetch_and_score(ticker: str, max_articles: int = 20) -> list[dict]:
    """
    Fetch up to *max_articles* recent news items for *ticker* from Yahoo Finance
    and return them as a list of scored article dicts, newest first.
    Returns an empty list if no news is available or the API fails.
    """
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.warning("Failed to fetch news for %s: %s", ticker, exc)
        return []

    articles = []
    for raw in raw_news[:max_articles]:
        parsed = _parse_article(raw)
        if parsed:
            articles.append(parsed)

    # Sort newest first
    articles.sort(key=lambda a: a["published_dt"], reverse=True)
    return articles


def ticker_sentiment(ticker: str, max_articles: int = 20) -> dict:
    """
    Full sentiment report for one ticker.

    Returns a dict with:
      ticker          - the ticker symbol
      articles        - list of scored article dicts
      composite_score - average compound score across all articles (-1 to +1)
      label           - overall Positive / Neutral / Negative
      positive_count  - number of positive articles
      negative_count  - number of negative articles
      neutral_count   - number of neutral articles
      total_articles  - total articles found
      error           - set only if something went wrong
    """
    articles = fetch_and_score(ticker, max_articles)

    if not articles:
        return {
            "ticker":          ticker,
            "articles":        [],
            "composite_score": 0.0,
            "label":           LABEL_NEUTRAL,
            "positive_count":  0,
            "negative_count":  0,
            "neutral_count":   0,
            "total_articles":  0,
            "error":           "No news articles found",
        }

    compounds       = [a["compound"] for a in articles]
    composite_score = round(sum(compounds) / len(compounds), 4)

    positive_count = sum(1 for a in articles if a["label"] == LABEL_POSITIVE)
    negative_count = sum(1 for a in articles if a["label"] == LABEL_NEGATIVE)
    neutral_count  = sum(1 for a in articles if a["label"] == LABEL_NEUTRAL)

    return {
        "ticker":          ticker,
        "articles":        articles,
        "composite_score": composite_score,
        "label":           _compound_to_label(composite_score),
        "positive_count":  positive_count,
        "negative_count":  negative_count,
        "neutral_count":   neutral_count,
        "total_articles":  len(articles),
    }


def sentiment_all(tickers: list[str]) -> dict[str, dict]:
    """Run ticker_sentiment for all tickers. Returns dict keyed by ticker."""
    return {t: ticker_sentiment(t) for t in tickers}


def sentiment_vs_price(ticker: str, clean_dir) -> pd.DataFrame:
    """
    Build a DataFrame aligning daily close price with any same-day news
    sentiment score, for the last 30 trading days.

    Used in the dashboard to show whether sentiment correlates with
    next-day price movement.

    Returns columns: Date, Close, pct_change, sentiment_score, sentiment_label
    """
    from pathlib import Path
    path = Path(clean_dir) / f"{ticker}_clean.parquet"
    if not path.exists():
        return pd.DataFrame()

    price_df = pd.read_parquet(path)[["Close"]]
    price_df.index = pd.to_datetime(price_df.index, utc=True)
    price_df = price_df.sort_index().tail(30)
    price_df["pct_change"] = price_df["Close"].pct_change() * 100

    articles = fetch_and_score(ticker, max_articles=50)
    if not articles:
        price_df["sentiment_score"] = None
        price_df["sentiment_label"] = LABEL_NEUTRAL
        price_df = price_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
        return price_df

    # Group article compound scores by calendar date
    by_date: dict[str, list[float]] = {}
    for a in articles:
        day = a["published_dt"].strftime("%Y-%m-%d")
        by_date.setdefault(day, []).append(a["compound"])

    avg_by_date = {
        day: round(sum(scores) / len(scores), 4)
        for day, scores in by_date.items()
    }

    price_df["sentiment_score"] = price_df.index.map(
        lambda ts: avg_by_date.get(ts.strftime("%Y-%m-%d"))
    )
    price_df["sentiment_label"] = price_df["sentiment_score"].map(
        lambda v: _compound_to_label(v) if v is not None else LABEL_NEUTRAL
    )

    price_df = price_df.reset_index().rename(columns={"Date": "Date"})
    return price_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for t in ["SPY", "QQQ", "AMD"]:
        result = ticker_sentiment(t)
        print(f"\n{'='*55}")
        print(f"  {t}  |  {result['label']}  |  score: {result['composite_score']:+.3f}")
        print(f"  {result['positive_count']} positive / "
              f"{result['negative_count']} negative / "
              f"{result['neutral_count']} neutral  "
              f"({result['total_articles']} articles)")
        for a in result["articles"][:3]:
            print(f"  [{a['label']:8s} {a['compound']:+.2f}]  {a['title'][:70]}")
