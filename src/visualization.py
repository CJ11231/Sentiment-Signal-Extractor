"""
Matplotlib dashboard and CSV export.

Public API
----------
create_dashboard(sentiment_df, price_df, correlation_df,
                 summary_stats, ticker, output_dir)  -> str (saved path)
export_to_csv(sentiment_df, price_df, correlation_df,
              ticker, output_dir)                    -> dict[str, str]
"""

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ── Colour palette (GitHub-dark inspired) ────────────────────────────────────
_C = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "text": "#c9d1d9",
    "grid": "#30363d",
    "positive": "#3fb950",
    "negative": "#f85149",
    "neutral": "#8b949e",
    "price": "#58a6ff",
    "volume_up": "#3fb950",
    "volume_dn": "#f85149",
    "sentiment_line": "#e3b341",
    "corr_bar": "#79c0ff",
    "corr_sig": "#ff7b72",
}


# ── Main entry point ──────────────────────────────────────────────────────────

def create_dashboard(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    summary_stats: dict,
    ticker: str,
    output_dir: str = "output",
) -> str:
    """
    Render a 4-panel dark-theme dashboard and save it as a PNG.

    Panels:
        1. Daily sentiment index (FinBERT composite score)
        2. Price + sentiment overlay (dual Y-axis)
        3. Trading volume (coloured by direction)
        4. Lead-lag correlation bar chart
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    merged = _prep_merged(sentiment_df, price_df)

    fig = plt.figure(figsize=(16, 22), facecolor=_C["bg"])
    fig.suptitle(
        f"{ticker}  —  Market Sentiment Signal Dashboard",
        fontsize=17,
        fontweight="bold",
        color=_C["text"],
        y=0.985,
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig, hspace=0.55, top=0.965, bottom=0.06, left=0.07, right=0.93
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    _panel_sentiment(ax1, sentiment_df, ticker)
    _panel_price_overlay(ax2, merged, ticker)
    _panel_volume(ax3, price_df, ticker)
    _panel_lead_lag(ax4, correlation_df, summary_stats)

    for ax in (ax1, ax2, ax3, ax4):
        _style_ax(ax)

    _add_stats_box(fig, summary_stats, ticker)

    out_path = os.path.join(output_dir, f"{ticker}_sentiment_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    print(f"  Dashboard → {out_path}")
    return out_path


# ── Panel 1: daily sentiment ──────────────────────────────────────────────────

def _panel_sentiment(ax: plt.Axes, df: pd.DataFrame, ticker: str) -> None:
    ax.set_title(
        "Daily Sentiment Index  (FinBERT composite = positive − negative)",
        color=_C["text"], fontsize=11, pad=8,
    )

    if df.empty:
        ax.text(0.5, 0.5, "No sentiment data available",
                transform=ax.transAxes, ha="center", color=_C["neutral"], fontsize=10)
        return

    dates = pd.to_datetime(df["date"])
    composite = df["composite"].values
    bar_colors = [_C["positive"] if v >= 0 else _C["negative"] for v in composite]

    ax.bar(dates, composite, color=bar_colors, alpha=0.85, width=0.8, zorder=3)
    ax.fill_between(dates, composite, 0,
                    where=(composite >= 0), color=_C["positive"], alpha=0.10, interpolate=True)
    ax.fill_between(dates, composite, 0,
                    where=(composite < 0), color=_C["negative"], alpha=0.10, interpolate=True)
    ax.axhline(0, color=_C["text"], linewidth=0.7, alpha=0.4, zorder=2)

    ax.set_ylabel("Composite Score", color=_C["text"], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Article count on secondary axis
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.plot(dates, df["article_count"], color="white", alpha=0.25,
             linewidth=1.2, linestyle=":", label="Article count")
    ax2.set_ylabel("Articles / day", color=_C["neutral"], fontsize=8, alpha=0.6)
    ax2.tick_params(colors=_C["neutral"], labelsize=7)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    _fmt_date_axis(ax, dates)


# ── Panel 2: price + sentiment overlay ───────────────────────────────────────

def _panel_price_overlay(ax: plt.Axes, merged: pd.DataFrame, ticker: str) -> None:
    ax.set_title(
        f"{ticker} Close Price  vs  Sentiment (right axis)",
        color=_C["text"], fontsize=11, pad=8,
    )

    dates = pd.to_datetime(merged["date"])
    ax.plot(dates, merged["close"], color=_C["price"], linewidth=2.0,
            label="Close Price", zorder=3)
    ax.set_ylabel("Price (USD)", color=_C["price"], fontsize=9)
    ax.tick_params(axis="y", labelcolor=_C["price"])

    ax_r = ax.twinx()
    ax_r.set_facecolor("none")

    if "composite" in merged.columns and not merged["composite"].isna().all():
        comp = merged["composite"].values
        ax_r.plot(dates, comp, color=_C["sentiment_line"], linewidth=1.4,
                  linestyle="--", alpha=0.85, label="Sentiment", zorder=4)
        ax_r.axhline(0, color=_C["text"], linewidth=0.5, alpha=0.2)
        ax_r.fill_between(dates, comp, 0,
                           where=(comp >= 0), color=_C["positive"], alpha=0.07, interpolate=True)
        ax_r.fill_between(dates, comp, 0,
                           where=(comp < 0), color=_C["negative"], alpha=0.07, interpolate=True)
        ax_r.set_ylabel("Sentiment Score", color=_C["sentiment_line"], fontsize=9)
        ax_r.tick_params(axis="y", labelcolor=_C["sentiment_line"])
    else:
        ax_r.set_yticks([])

    for sp in ax_r.spines.values():
        sp.set_color(_C["grid"])

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8,
              facecolor=_C["surface"], labelcolor=_C["text"], framealpha=0.8)

    _fmt_date_axis(ax, dates)


# ── Panel 3: volume ───────────────────────────────────────────────────────────

def _panel_volume(ax: plt.Axes, price_df: pd.DataFrame, ticker: str) -> None:
    ax.set_title("Trading Volume", color=_C["text"], fontsize=11, pad=8)

    p = price_df.copy()
    p.index = pd.to_datetime(p.index)
    ret = p["return"].fillna(0)
    colors = [_C["volume_up"] if r >= 0 else _C["volume_dn"] for r in ret]

    ax.bar(p.index, p["volume"] / 1e6, color=colors, alpha=0.75, width=0.8, zorder=3)
    ax.set_ylabel("Volume (M shares)", color=_C["text"], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    _fmt_date_axis(ax, p.index)


# ── Panel 4: lead-lag correlation ─────────────────────────────────────────────

def _panel_lead_lag(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    stats: dict,
) -> None:
    ax.set_title(
        "Lead-Lag Correlation  (positive lag → sentiment leads price)",
        color=_C["text"], fontsize=11, pad=8,
    )

    if corr_df.empty or corr_df["correlation"].isna().all():
        ax.text(0.5, 0.5, "Insufficient data for correlation analysis",
                transform=ax.transAxes, ha="center", color=_C["neutral"], fontsize=10)
        return

    lags = corr_df["lag"].values
    corrs = corr_df["correlation"].fillna(0).values
    sig = corr_df.get("significant", pd.Series([False] * len(corr_df))).values
    colors = [_C["corr_sig"] if s else _C["corr_bar"] for s in sig]

    ax.bar(lags, corrs, color=colors, alpha=0.85, width=0.65, zorder=3)
    ax.axhline(0, color=_C["text"], linewidth=0.7, alpha=0.4)
    ax.axvline(0, color=_C["text"], linewidth=0.7, alpha=0.25, linestyle="--")

    ax.set_xlabel(
        "Lag (days)   ·   lag > 0 means sentiment at t predicts return at t+lag",
        color=_C["text"], fontsize=8,
    )
    ax.set_ylabel("Pearson r", color=_C["text"], fontsize=9)
    ax.set_xticks(lags)

    # Annotate peak correlation
    best_lag = stats.get("best_lag")
    best_corr = stats.get("best_correlation")
    if best_lag is not None and best_corr is not None and not np.isnan(best_corr):
        offset_y = 0.04 if best_corr >= 0 else -0.06
        ax.annotate(
            f" lag={best_lag}, r={best_corr:+.3f}",
            xy=(best_lag, best_corr),
            xytext=(best_lag + 0.4, best_corr + offset_y),
            color=_C["text"],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_C["text"], lw=0.8),
        )

    legend_handles = [
        Patch(facecolor=_C["corr_sig"], alpha=0.85, label="p < 0.05"),
        Patch(facecolor=_C["corr_bar"], alpha=0.85, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              facecolor=_C["surface"], labelcolor=_C["text"], framealpha=0.8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_C["surface"])
    ax.tick_params(colors=_C["text"], labelsize=8)
    ax.xaxis.label.set_color(_C["text"])
    ax.yaxis.label.set_color(_C["text"])
    for sp in ax.spines.values():
        sp.set_color(_C["grid"])
    ax.grid(True, alpha=0.15, color=_C["grid"], linestyle="--", zorder=0)


def _fmt_date_axis(ax: plt.Axes, dates) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


def _prep_merged(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    price = price_df.copy()
    price.index.name = "date"
    price = price.reset_index()
    price["date"] = pd.to_datetime(price["date"])

    if sentiment_df.empty:
        return price

    s = sentiment_df[["date", "composite", "article_count"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    return pd.merge(price, s, on="date", how="left")


def _add_stats_box(fig: plt.Figure, stats: dict, ticker: str) -> None:
    lines = [f"  {ticker} Summary Statistics"]
    lines.append("  " + "─" * 36)

    def _row(label: str, value: str) -> str:
        return f"  {label:<28}{value}"

    if "avg_sentiment" in stats:
        lines.append(_row("Avg sentiment:", f"{stats['avg_sentiment']:+.4f}"))
    if "bullish_days" in stats:
        lines.append(_row("Bullish / bearish days:",
                          f"{stats['bullish_days']} / {stats.get('bearish_days', 0)}"))
    if "avg_daily_return" in stats:
        lines.append(_row("Avg daily return:", f"{stats['avg_daily_return']:+.4f}"))
    if "annualised_vol" in stats:
        lines.append(_row("Annualised vol:", f"{stats['annualised_vol']:.2%}"))
    if "approx_sharpe" in stats:
        lines.append(_row("Approx. Sharpe:", f"{stats['approx_sharpe']:.2f}"))
    if "next_day_corr" in stats:
        sig = " ✓ p<0.05" if stats.get("next_day_sig") else ""
        lines.append(_row("Next-day corr (lag=1):", f"{stats['next_day_corr']:+.4f}{sig}"))
    if "best_lag" in stats:
        sig = " ✓" if stats.get("best_significant") else ""
        lines.append(_row("Best lead lag:",
                          f"{stats['best_lag']}d  r={stats['best_correlation']:+.4f}{sig}"))

    fig.text(
        0.07, 0.015,
        "\n".join(lines),
        fontsize=8,
        color=_C["text"],
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=_C["surface"],
                  edgecolor=_C["grid"], alpha=0.9),
    )


# ── CSV export ────────────────────────────────────────────────────────────────

def export_to_csv(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    ticker: str,
    output_dir: str = "output",
) -> dict[str, str]:
    """Write sentiment, price, and correlation data to separate CSV files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if not sentiment_df.empty:
        p = os.path.join(output_dir, f"{ticker}_daily_sentiment.csv")
        sentiment_df.to_csv(p, index=False)
        paths["sentiment"] = p
        print(f"  Sentiment → {p}")

    if not price_df.empty:
        p = os.path.join(output_dir, f"{ticker}_price_data.csv")
        price_df.to_csv(p)
        paths["price"] = p
        print(f"  Price data → {p}")

    if not correlation_df.empty:
        p = os.path.join(output_dir, f"{ticker}_lead_lag_correlation.csv")
        correlation_df.to_csv(p, index=False)
        paths["correlation"] = p
        print(f"  Lead-lag  → {p}")

    return paths
