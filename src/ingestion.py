"""
Data ingestion: financial news articles and SEC EDGAR earnings filings.

Public API
----------
fetch_news_articles(ticker, days, newsapi_key, av_key) -> list[dict]
fetch_sec_transcripts(ticker, days)                    -> list[dict]

Source priority:
  1. Alpha Vantage NEWS_SENTIMENT (genuine historical coverage, up to 1000 articles)
  2. NewsAPI (recent ~30 days, free tier)
  3. Yahoo Finance headlines (fallback, recent only)

Each returned dict has keys: date (datetime.date), title, content, source, url.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

# Common ticker → company name mapping for better NewsAPI search precision
_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia",
    "NFLX": "Netflix",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "INTC": "Intel",
    "AMD": "Advanced Micro Devices",
    "PYPL": "PayPal",
    "CRM": "Salesforce",
}

_SEC_HEADERS = {"User-Agent": "MarketSentimentExtractor research@example.com"}


# ── Public entry points ───────────────────────────────────────────────────────

def fetch_news_articles(
    ticker: str,
    days: int,
    newsapi_key: Optional[str] = None,
    av_key: Optional[str] = None,
) -> list[dict]:
    """
    Fetch recent news articles for *ticker*.

    Priority: Alpha Vantage (historical) → NewsAPI → Yahoo Finance fallback.
    Articles outside the requested *days* window are filtered out.
    """
    articles: list[dict] = []
    cutoff = datetime.now().date() - timedelta(days=days)

    if av_key:
        print(f"  Fetching from Alpha Vantage …")
        articles = _fetch_alpha_vantage(ticker, days, av_key)
        print(f"  → {len(articles)} articles")

    if not articles and newsapi_key:
        print(f"  Fetching from NewsAPI …")
        articles = _fetch_newsapi(ticker, days, newsapi_key)
        print(f"  → {len(articles)} articles")

    if not articles:
        print(f"  Fetching from Yahoo Finance …")
        articles = _fetch_yfinance_news(ticker)
        print(f"  → {len(articles)} articles")

    articles = [a for a in articles if a["date"] >= cutoff]
    return sorted(articles, key=lambda x: x["date"])


def fetch_sec_transcripts(ticker: str, days: int) -> list[dict]:
    """
    Fetch recent 8-K filings from SEC EDGAR for *ticker*.

    Returns parsed text snippets in the same dict format as fetch_news_articles.
    """
    print(f"  Resolving CIK for {ticker} …")
    cik = _resolve_cik(ticker)
    if not cik:
        print(f"  Could not find CIK — skipping SEC data.")
        return []

    print(f"  Fetching 8-K filings (CIK {cik}) …")
    filings = _get_8k_filings(cik, days)
    print(f"  → {len(filings)} filings")
    return filings


# ── Alpha Vantage News Sentiment API ─────────────────────────────────────────

def _fetch_alpha_vantage(ticker: str, days: int, api_key: str) -> list[dict]:
    """
    Fetch news via Alpha Vantage NEWS_SENTIMENT endpoint.

    Returns up to 1000 articles going back *days* calendar days.
    Free tier allows 25 requests/day; one call is enough for up to 1000 articles.
    """
    start_dt = datetime.now() - timedelta(days=days)
    time_from = start_dt.strftime("%Y%m%dT%H%M")  # e.g. 20240101T0000

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "limit": 1000,
        "sort": "EARLIEST",
        "apikey": api_key,
    }

    try:
        resp = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()

        if "Information" in data:
            print(f"  Alpha Vantage rate limit hit: {data['Information'][:120]}")
            return []

        articles = []
        for item in data.get("feed", []):
            raw_time = item.get("time_published", "")  # e.g. "20240101T120000"
            if not raw_time:
                continue
            try:
                art_date = datetime.strptime(raw_time[:8], "%Y%m%d").date()
            except ValueError:
                continue

            title = item.get("title", "")
            summary = item.get("summary", "")
            content = f"{title}. {summary}".strip(". ")

            if content:
                articles.append(
                    {
                        "date": art_date,
                        "title": title,
                        "content": content[:2000],
                        "source": item.get("source", "Alpha Vantage"),
                        "url": item.get("url", ""),
                    }
                )
        return articles

    except requests.HTTPError as e:
        print(f"  Alpha Vantage HTTP error {e.response.status_code}")
        return []
    except Exception as e:
        print(f"  Alpha Vantage error: {e}")
        return []


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def _fetch_newsapi(ticker: str, days: int, api_key: str) -> list[dict]:
    company = _COMPANY_NAMES.get(ticker.upper(), ticker)
    end_dt = datetime.now()
    # Free NewsAPI tier only supports 30-day lookback
    start_dt = end_dt - timedelta(days=min(days, 30))

    params = {
        "q": f'({company} OR "{ticker}") AND (stock OR earnings OR market OR shares)',
        "from": start_dt.strftime("%Y-%m-%d"),
        "to": end_dt.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("articles", []):
            published = item.get("publishedAt", "")
            if not published:
                continue
            art_date = datetime.strptime(published[:10], "%Y-%m-%d").date()
            content = " ".join(
                filter(
                    None,
                    [
                        item.get("title", ""),
                        item.get("description", ""),
                        item.get("content", ""),
                    ],
                )
            ).strip()
            if content:
                articles.append(
                    {
                        "date": art_date,
                        "title": item.get("title", ""),
                        "content": content[:2000],
                        "source": item.get("source", {}).get("name", "NewsAPI"),
                        "url": item.get("url", ""),
                    }
                )
        return articles

    except requests.HTTPError as e:
        print(f"  NewsAPI HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"  NewsAPI error: {e}")
        return []


# ── Yahoo Finance headlines (fallback) ───────────────────────────────────────

def _fetch_yfinance_news(ticker: str) -> list[dict]:
    try:
        news_items = yf.Ticker(ticker).news or []
        articles = []
        for item in news_items:
            ts = item.get("providerPublishTime", 0)
            art_date = datetime.fromtimestamp(ts).date() if ts else datetime.now().date()
            title = item.get("title", "")

            # yfinance ≥0.2.50 wraps body in a nested 'content' dict
            summary = ""
            raw_content = item.get("content")
            if isinstance(raw_content, dict):
                summary = raw_content.get("summary", "") or raw_content.get("body", "")
            else:
                summary = item.get("summary", "") or ""

            content = f"{title}. {summary}".strip(". ")
            if content:
                articles.append(
                    {
                        "date": art_date,
                        "title": title,
                        "content": content[:2000],
                        "source": item.get("publisher", "Yahoo Finance"),
                        "url": item.get("link", ""),
                    }
                )
        return articles
    except Exception as e:
        print(f"  Yahoo Finance news error: {e}")
        return []


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

def _resolve_cik(ticker: str) -> Optional[str]:
    """Map a ticker symbol to a zero-padded 10-digit CIK."""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_SEC_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        print(f"  CIK lookup failed: {e}")
    return None


def _get_8k_filings(cik: str, days: int) -> list[dict]:
    cutoff = datetime.now().date() - timedelta(days=days)
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_SEC_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for form, date_str, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form not in ("8-K", "8-K/A"):
                continue
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if filing_date < cutoff:
                break  # EDGAR returns newest-first

            time.sleep(0.15)  # respect EDGAR rate limit
            text = _fetch_filing_text(cik, acc, doc)
            if text:
                filings.append(
                    {
                        "date": filing_date,
                        "title": f"SEC 8-K Filing ({date_str})",
                        "content": text[:3000],
                        "source": "SEC EDGAR",
                        "url": (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{int(cik)}/{acc.replace('-', '')}/{doc}"
                        ),
                    }
                )
        return filings

    except Exception as e:
        print(f"  EDGAR submissions error: {e}")
        return []


def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str:
    acc_clean = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/{primary_doc}"
    )
    try:
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=10)
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""
