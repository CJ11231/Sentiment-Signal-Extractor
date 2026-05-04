"""
Signal validation: price data fetching and lead-lag correlation analysis.

Public API
----------
fetch_price_data(ticker, start, end)                    -> pd.DataFrame
merge_sentiment_price(sentiment_df, price_df)           -> pd.DataFrame
calculate_lead_lag_correlation(merged_df, max_lag)      -> pd.DataFrame
compute_summary_stats(merged_df, correlation_df)        -> dict
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_price_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for *ticker* via yfinance.

    Extends start by 5 days so day-over-day returns are available for the
    full requested window. Raises ValueError if no data is returned.

    Columns: open, high, low, close, volume, return, log_return, next_return
    """
    print(f"  Downloading {ticker} price history …")
    extended_start = start_date - timedelta(days=7)

    stock = yf.Ticker(ticker)
    df = stock.history(start=extended_start, end=end_date + timedelta(days=2))

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}'. "
            "Check the ticker symbol and try again."
        )

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [c.lower() for c in df.columns]

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["next_return"] = df["return"].shift(-1)

    # Trim to the requested start
    df = df[df.index.date >= start_date.date()]
    print(f"  → {len(df)} trading days retrieved.")
    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_sentiment_price(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join price data with daily sentiment on *date*.

    Sentiment is forward-filled across non-publication days (weekends,
    holidays) so the merged frame aligns cleanly with the trading calendar.
    """
    price = price_df.copy()
    price.index.name = "date"
    price = price.reset_index()
    price["date"] = pd.to_datetime(price["date"])

    if sentiment_df.empty:
        return price.set_index("date").sort_index()

    sentiment = sentiment_df.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    merged = pd.merge(price, sentiment, on="date", how="left")
    merged = merged.set_index("date").sort_index()

    ffill_cols = ["composite", "positive", "negative", "neutral", "article_count"]
    for col in ffill_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    return merged


# ── Lead-lag correlation ──────────────────────────────────────────────────────

def calculate_lead_lag_correlation(
    merged_df: pd.DataFrame,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Pearson correlation between *composite* sentiment and *return* at each lag.

    lag > 0: sentiment at t predicts return at t + lag  (sentiment leads price)
    lag = 0: contemporaneous
    lag < 0: price at t + lag predicts sentiment at t   (price leads sentiment)

    Returns DataFrame with columns: lag, correlation, p_value, significant, n_obs
    """
    if (
        "composite" not in merged_df.columns
        or merged_df["composite"].isna().all()
        or "return" not in merged_df.columns
    ):
        return pd.DataFrame(
            columns=["lag", "correlation", "p_value", "significant", "n_obs"]
        )

    base = merged_df[["composite", "return"]].copy()

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        # shift(-lag) on return so that y[t] = return[t + lag]
        paired = pd.DataFrame(
            {
                "x": base["composite"],
                "y": base["return"].shift(-lag),
            }
        ).dropna()

        if len(paired) < 5:
            rows.append(
                {"lag": lag, "correlation": np.nan, "p_value": np.nan,
                 "significant": False, "n_obs": len(paired)}
            )
            continue

        corr, p_val = stats.pearsonr(paired["x"], paired["y"])
        rows.append(
            {
                "lag": lag,
                "correlation": corr,
                "p_value": p_val,
                "significant": bool(p_val < 0.05),
                "n_obs": len(paired),
            }
        )

    return pd.DataFrame(rows)


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary_stats(
    merged_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> dict:
    """Collect headline statistics for the dashboard annotation."""
    out: dict = {}

    if not merged_df.empty and "composite" in merged_df.columns:
        s = merged_df["composite"].dropna()
        if len(s):
            out["avg_sentiment"] = float(s.mean())
            out["sentiment_std"] = float(s.std())
            out["bullish_days"] = int((s > 0.05).sum())
            out["bearish_days"] = int((s < -0.05).sum())

    if not merged_df.empty and "return" in merged_df.columns:
        r = merged_df["return"].dropna()
        if len(r):
            out["avg_daily_return"] = float(r.mean())
            out["return_vol"] = float(r.std())
            out["annualised_vol"] = float(r.std() * np.sqrt(252))
            sharpe_denom = r.std() * np.sqrt(252)
            out["approx_sharpe"] = (
                float(r.mean() * 252 / sharpe_denom) if sharpe_denom > 0 else 0.0
            )

    if not correlation_df.empty and not correlation_df["correlation"].isna().all():
        valid = correlation_df.dropna(subset=["correlation"])
        best_row = valid.loc[valid["correlation"].abs().idxmax()]
        out["best_lag"] = int(best_row["lag"])
        out["best_correlation"] = float(best_row["correlation"])
        out["best_p_value"] = float(best_row["p_value"])
        out["best_significant"] = bool(best_row["significant"])

        next_day = correlation_df[correlation_df["lag"] == 1]
        if not next_day.empty:
            out["next_day_corr"] = float(next_day.iloc[0]["correlation"])
            out["next_day_p"] = float(next_day.iloc[0]["p_value"])
            out["next_day_sig"] = bool(next_day.iloc[0]["significant"])

    return out
