#!/usr/bin/env python3
"""
Market Sentiment Signal Extractor
==================================
Fetch financial news, score it with FinBERT, and correlate sentiment
against price movement.

Usage
-----
    python main.py --ticker AAPL --days 30
    python main.py --ticker TSLA --days 90 --include-sec
    python main.py --ticker MSFT --days 60 --no-dashboard
    python main.py --ticker NVDA --days 30 --av-key <KEY>
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

from src.ingestion import fetch_news_articles, fetch_sec_transcripts
from src.sentiment import aggregate_daily_sentiment, load_finbert, score_articles
from src.validation import (
    calculate_lead_lag_correlation,
    compute_summary_stats,
    fetch_price_data,
    merge_sentiment_price,
)
from src.visualization import create_dashboard, export_to_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Market Sentiment Signal Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ticker", "-t", required=True,
        help="Stock ticker symbol (e.g. AAPL, TSLA, MSFT)",
    )
    p.add_argument(
        "--days", "-d", type=int, default=30,
        help="Number of calendar days to analyse",
    )
    p.add_argument(
        "--newsapi-key",
        default=os.getenv("NEWSAPI_KEY"),
        help="NewsAPI key. Falls back to $NEWSAPI_KEY env var.",
    )
    p.add_argument(
        "--av-key",
        default=os.getenv("ALPHA_VANTAGE_KEY"),
        help="Alpha Vantage key. Falls back to $ALPHA_VANTAGE_KEY env var. "
             "Preferred over NewsAPI — provides genuine historical coverage.",
    )
    p.add_argument(
        "--include-sec", action="store_true",
        help="Also pull SEC EDGAR 8-K filings for the ticker",
    )
    p.add_argument(
        "--max-lag", type=int, default=5,
        help="Maximum ±lag days for lead-lag correlation analysis",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory for PNG dashboard and CSV exports",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip the matplotlib dashboard (faster, CSV only)",
    )
    p.add_argument(
        "--no-export", action="store_true",
        help="Skip CSV export",
    )
    return p


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    ticker = args.ticker.upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    _banner(ticker, start_date, end_date)

    # ── 1. Data ingestion ────────────────────────────────────────────────────
    print("\n[1/4]  Data Ingestion")
    print("─" * 50)
    articles = fetch_news_articles(ticker, args.days, args.newsapi_key, args.av_key)

    if args.include_sec:
        sec_docs = fetch_sec_transcripts(ticker, args.days)
        articles.extend(sec_docs)

    total = len(articles)
    if total == 0:
        print("  WARNING: no articles found — sentiment analysis will be skipped.")
    else:
        date_range = (
            f"{min(a['date'] for a in articles)}  →  "
            f"{max(a['date'] for a in articles)}"
        )
        print(f"  {total} documents collected  ({date_range})")

    # ── 2. Sentiment analysis ────────────────────────────────────────────────
    print("\n[2/4]  Sentiment Analysis  (FinBERT)")
    print("─" * 50)

    if articles:
        tokenizer, model, device = load_finbert()
        scored = score_articles(articles, tokenizer, model, device)
        daily_sentiment = aggregate_daily_sentiment(scored)
        print(f"  Daily index covers {len(daily_sentiment)} unique date(s).")
    else:
        import pandas as pd
        scored = []
        daily_sentiment = pd.DataFrame()

    # ── 3. Signal validation ─────────────────────────────────────────────────
    print("\n[3/4]  Signal Validation")
    print("─" * 50)

    try:
        price_df = fetch_price_data(ticker, start_date, end_date)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    merged_df = merge_sentiment_price(daily_sentiment, price_df)
    correlation_df = calculate_lead_lag_correlation(merged_df, args.max_lag)
    summary = compute_summary_stats(merged_df, correlation_df)

    _print_summary(summary)

    # ── 4. Output ────────────────────────────────────────────────────────────
    print("\n[4/4]  Output")
    print("─" * 50)

    if not args.no_dashboard:
        create_dashboard(
            daily_sentiment, price_df, correlation_df, summary, ticker, args.output_dir
        )

    if not args.no_export:
        export_to_csv(
            daily_sentiment, price_df, correlation_df, ticker, args.output_dir
        )

    _footer(ticker, args.output_dir)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _banner(ticker: str, start: datetime, end: datetime) -> None:
    w = 60
    print("=" * w)
    print("  Market Sentiment Signal Extractor".center(w))
    print(f"  {ticker}   {start.date()}  →  {end.date()}".center(w))
    print("=" * w)


def _print_summary(stats: dict) -> None:
    print("\n  ┌─ Key Statistics ─────────────────────────────┐")
    rows = []
    if "avg_sentiment" in stats:
        rows.append(("Avg sentiment score", f"{stats['avg_sentiment']:+.4f}"))
    if "bullish_days" in stats:
        rows.append(("Bullish / bearish days",
                     f"{stats['bullish_days']} / {stats.get('bearish_days', 0)}"))
    if "avg_daily_return" in stats:
        rows.append(("Avg daily return", f"{stats['avg_daily_return']:+.4f}"))
    if "annualised_vol" in stats:
        rows.append(("Annualised volatility", f"{stats['annualised_vol']:.2%}"))
    if "approx_sharpe" in stats:
        rows.append(("Approx. Sharpe (ann.)", f"{stats['approx_sharpe']:.2f}"))
    if "next_day_corr" in stats:
        sig = "  ✓ significant" if stats.get("next_day_sig") else ""
        rows.append(("Next-day correlation", f"{stats['next_day_corr']:+.4f}{sig}"))
    if "best_lag" in stats:
        sig = "  ✓" if stats.get("best_significant") else ""
        rows.append(("Peak lead-lag",
                     f"lag={stats['best_lag']}d  r={stats['best_correlation']:+.4f}{sig}"))

    for label, value in rows:
        print(f"  │  {label:<28}  {value}")
    print("  └──────────────────────────────────────────────┘")


def _footer(ticker: str, output_dir: str) -> None:
    w = 60
    print("\n" + "=" * w)
    print(f"  ✓  Analysis complete for {ticker}".center(w))
    print(f"  Output: {output_dir}/".center(w))
    print("=" * w + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(_build_parser().parse_args())
