"""
FinBERT-based sentiment scoring for financial text.

Public API
----------
load_finbert()                                          -> (tokenizer, model, device)
score_articles(articles, tokenizer, model, device)      -> list[dict]
aggregate_daily_sentiment(scored_articles)              -> pd.DataFrame
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# torch and transformers are imported lazily inside load_finbert() so that
# the CLI --help and non-ML code paths work before the heavy deps are installed.
torch = None  # populated by _ensure_torch()
AutoTokenizer = None
AutoModelForSequenceClassification = None


def _ensure_torch() -> None:
    """Import ML deps on first use with a friendly error message."""
    global torch, AutoTokenizer, AutoModelForSequenceClassification
    if torch is not None:
        return
    try:
        import torch as _torch
        from transformers import (
            AutoModelForSequenceClassification as _AMSC,
            AutoTokenizer as _AT,
        )
        torch = _torch
        AutoTokenizer = _AT
        AutoModelForSequenceClassification = _AMSC
    except ImportError:
        raise ImportError(
            "FinBERT requires torch and transformers.\n"
            "Install them with:  pip install torch transformers"
        )

_FINBERT_MODEL = "ProsusAI/finbert"
# ProsusAI/finbert id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
_BATCH_SIZE = 8


# ── Model loading ─────────────────────────────────────────────────────────────

def load_finbert() -> tuple[Any, Any, str]:
    """
    Download (first run only, ~440 MB) and load ProsusAI/finbert.

    Returns (tokenizer, model, device).
    """
    _ensure_torch()
    print(f"  Loading {_FINBERT_MODEL} …")
    print("  (First run downloads ~440 MB; subsequent runs use the local cache.)")

    tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  Model ready on {device}.")
    return tokenizer, model, device


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_batch(texts: list[str], tokenizer, model, device: str) -> list[dict]:
    """Run FinBERT on a batch of texts; return one score dict per text."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    id2label: dict[int, str] = model.config.id2label
    results = []
    for row in probs:
        scores = {id2label[i].lower(): float(row[i]) for i in range(len(row))}
        label = max(scores, key=scores.get)
        results.append({**scores, "label": label, "confidence": scores[label]})
    return results


def score_articles(
    articles: list[dict],
    tokenizer: Any,
    model: Any,
    device: str,
) -> list[dict]:
    """
    Score every article in *articles* with FinBERT.

    Adds positive / negative / neutral probability fields plus *label* and
    *confidence* to each article dict.
    """
    n = len(articles)
    print(f"  Scoring {n} documents in batches of {_BATCH_SIZE} …")

    texts = [f"{a['title']}. {a['content']}"[:1500] for a in articles]
    all_scores: list[dict] = []

    for start in range(0, n, _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        all_scores.extend(_score_batch(batch, tokenizer, model, device))
        done = min(start + _BATCH_SIZE, n)
        print(f"  [{done}/{n}]", end="\r", flush=True)

    print()  # newline after progress
    return [{**article, **score} for article, score in zip(articles, all_scores)]


# ── Daily aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_sentiment(scored_articles: list[dict]) -> pd.DataFrame:
    """
    Roll article-level scores into a daily sentiment index.

    Composite score = mean(positive) − mean(negative), range [−1, +1].
    Each day's scores are confidence-weighted so high-certainty articles
    carry more weight.

    Returns a DataFrame with columns:
        date, composite, positive, negative, neutral, article_count, avg_confidence
    """
    if not scored_articles:
        return pd.DataFrame(
            columns=[
                "date", "composite", "positive", "negative",
                "neutral", "article_count", "avg_confidence",
            ]
        )

    df = pd.DataFrame(scored_articles)
    df["date"] = pd.to_datetime(df["date"])

    def _wavg(group: pd.DataFrame, col: str) -> float:
        weights = group["confidence"]
        if weights.sum() == 0:
            return group[col].mean()
        return float(np.average(group[col], weights=weights))

    rows = []
    for date, grp in df.groupby("date"):
        rows.append(
            {
                "date": date,
                "composite": _wavg(grp, "positive") - _wavg(grp, "negative"),
                "positive": _wavg(grp, "positive"),
                "negative": _wavg(grp, "negative"),
                "neutral": _wavg(grp, "neutral"),
                "article_count": len(grp),
                "avg_confidence": grp["confidence"].mean(),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return daily
