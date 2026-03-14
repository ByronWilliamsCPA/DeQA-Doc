"""Ordinal discrimination analysis: How well do models identify quality buckets?

Analyzes model accuracy at the categorical level (bad/poor/fair/good/excellent)
vs continuous precision, using existing DIQA-5000 checkpoint data.

Metrics:
  - Bucket accuracy (exact match)
  - Adjacent accuracy (correct or ±1 bucket)
  - Cohen's Kappa (chance-corrected agreement)
  - Per-bucket precision/recall/F1
  - Per-bucket within-class SRCC (ranking within a quality level)
  - Confusion matrix

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/analyze_ordinal.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

EVAL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"
DATA_DIR = EVAL_DIR / "data"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_DIR = EVAL_DIR / "results"

# Quality buckets matching DeQA convention
BUCKETS = [
    ("bad", 1.0, 1.8),
    ("poor", 1.8, 2.6),
    ("fair", 2.6, 3.4),
    ("good", 3.4, 4.0),
    ("excellent", 4.0, 5.01),
]

BUCKET_NAMES = [b[0] for b in BUCKETS]


def score_to_bucket(score: float) -> int:
    """Map a continuous score to bucket index (0-4)."""
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo <= score < hi:
            return i
    return 4  # scores exactly 5.0 → excellent


def load_ground_truth() -> dict[str, dict[str, float]]:
    """Load DIQA-5000 test set ground truth."""
    gt: dict[str, dict[str, float]] = {}
    with TEST_CSV.open() as f:
        for row in csv.DictReader(f):
            gt[row["res"]] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color_fidelity": float(row["color_fidelity"]),
            }
    return gt


def load_model_results(model_id: str) -> dict[str, dict]:
    """Load checkpoint results for a model."""
    safe = model_id.replace("/", "__")
    cp = CHECKPOINT_DIR / f"{safe}.jsonl"
    results: dict[str, dict] = {}
    if not cp.exists():
        return results
    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not item.get("error") and item.get("overall") is not None:
                results[item["image"]] = item
        except json.JSONDecodeError:
            continue
    return results


def cohens_kappa(true_buckets: list[int], pred_buckets: list[int], n_classes: int = 5) -> float:
    """Compute Cohen's Kappa for ordinal agreement."""
    n = len(true_buckets)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for t, p in zip(true_buckets, pred_buckets) if t == p) / n

    # Expected agreement (chance)
    pe = 0.0
    for k in range(n_classes):
        p_true = sum(1 for t in true_buckets if t == k) / n
        p_pred = sum(1 for p in pred_buckets if p == k) / n
        pe += p_true * p_pred

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def weighted_kappa(true_buckets: list[int], pred_buckets: list[int], n_classes: int = 5) -> float:
    """Compute quadratic weighted Kappa (penalizes larger disagreements more)."""
    n = len(true_buckets)
    if n == 0:
        return 0.0

    # Build confusion matrix
    conf = np.zeros((n_classes, n_classes))
    for t, p in zip(true_buckets, pred_buckets):
        conf[t][p] += 1

    # Weight matrix (quadratic)
    w = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w[i][j] = (i - j) ** 2 / (n_classes - 1) ** 2

    # Expected matrix
    row_sum = conf.sum(axis=1)
    col_sum = conf.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / n

    # Weighted kappa
    num = (w * conf).sum()
    den = (w * expected).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


def analyze_model(
    model_id: str,
    gt: dict[str, dict[str, float]],
    results: dict[str, dict],
) -> dict[str, Any]:
    """Run full ordinal analysis for a model."""
    analysis: dict[str, Any] = {"model_id": model_id}

    for dim in ("overall", "sharpness", "color_fidelity"):
        true_scores: list[float] = []
        pred_scores: list[float] = []
        true_buckets: list[int] = []
        pred_buckets: list[int] = []

        for img_id, r in results.items():
            if img_id not in gt:
                continue
            true_s = gt[img_id][dim]
            pred_s = r[dim]
            if pred_s is None:
                continue

            true_scores.append(true_s)
            pred_scores.append(pred_s)
            true_buckets.append(score_to_bucket(true_s))
            pred_buckets.append(score_to_bucket(pred_s))

        n = len(true_scores)
        if n < 30:
            analysis[dim] = {"n": n, "error": "insufficient data"}
            continue

        # Exact bucket accuracy
        exact = sum(1 for t, p in zip(true_buckets, pred_buckets) if t == p) / n

        # Adjacent accuracy (within ±1 bucket)
        adjacent = sum(
            1 for t, p in zip(true_buckets, pred_buckets) if abs(t - p) <= 1
        ) / n

        # Kappa
        kappa = cohens_kappa(true_buckets, pred_buckets)
        w_kappa = weighted_kappa(true_buckets, pred_buckets)

        # Per-bucket precision, recall, F1
        bucket_metrics: dict[str, dict] = {}
        for bi, bname in enumerate(BUCKET_NAMES):
            tp = sum(1 for t, p in zip(true_buckets, pred_buckets) if t == bi and p == bi)
            fp = sum(1 for t, p in zip(true_buckets, pred_buckets) if t != bi and p == bi)
            fn = sum(1 for t, p in zip(true_buckets, pred_buckets) if t == bi and p != bi)
            n_true = sum(1 for t in true_buckets if t == bi)
            n_pred = sum(1 for p in pred_buckets if p == bi)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Within-bucket SRCC (ranking accuracy within this quality level)
            within_pred = [
                pred_scores[i] for i in range(n) if true_buckets[i] == bi
            ]
            within_true = [
                true_scores[i] for i in range(n) if true_buckets[i] == bi
            ]
            if len(within_pred) >= 5:
                within_srcc = float(stats.spearmanr(within_pred, within_true).statistic)
            else:
                within_srcc = None

            bucket_metrics[bname] = {
                "n_true": n_true,
                "n_pred": n_pred,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "within_srcc": round(within_srcc, 4) if within_srcc is not None else None,
            }

        # Confusion matrix
        conf_matrix = np.zeros((5, 5), dtype=int)
        for t, p in zip(true_buckets, pred_buckets):
            conf_matrix[t][p] += 1

        # Most common error direction
        over_rate = sum(1 for t, p in zip(true_buckets, pred_buckets) if p > t)
        under_rate = sum(1 for t, p in zip(true_buckets, pred_buckets) if p < t)

        analysis[dim] = {
            "n": n,
            "exact_accuracy": round(exact, 4),
            "adjacent_accuracy": round(adjacent, 4),
            "cohens_kappa": round(kappa, 4),
            "weighted_kappa": round(w_kappa, 4),
            "over_rate_pct": round(over_rate / n * 100, 1),
            "under_rate_pct": round(under_rate / n * 100, 1),
            "per_bucket": bucket_metrics,
            "confusion_matrix": conf_matrix.tolist(),
        }

    return analysis


def print_analysis(analysis: dict[str, Any]) -> None:
    """Print formatted analysis for one model."""
    model = analysis["model_id"]
    print(f"\n{'=' * 80}")
    print(f"  {model}")
    print(f"{'=' * 80}")

    for dim in ("overall", "sharpness", "color_fidelity"):
        d = analysis.get(dim, {})
        if "error" in d:
            print(f"\n  {dim}: {d['error']}")
            continue

        print(f"\n  {dim.upper()} (n={d['n']})")
        print(f"    Exact bucket accuracy:    {d['exact_accuracy']:.1%}")
        print(f"    Adjacent accuracy (±1):   {d['adjacent_accuracy']:.1%}")
        print(f"    Cohen's Kappa:            {d['cohens_kappa']:.4f}")
        print(f"    Weighted Kappa:           {d['weighted_kappa']:.4f}")
        print(f"    Over-rating:              {d['over_rate_pct']:.1f}%")
        print(f"    Under-rating:             {d['under_rate_pct']:.1f}%")

        # Per-bucket table
        print(f"\n    {'Bucket':<12s} {'n_true':>7s} {'n_pred':>7s} "
              f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'inSRCC':>7s}")
        print(f"    {'-' * 55}")
        for bname, bm in d["per_bucket"].items():
            srcc_s = f"{bm['within_srcc']:.4f}" if bm["within_srcc"] is not None else "N/A"
            print(
                f"    {bname:<12s} {bm['n_true']:>7d} {bm['n_pred']:>7d} "
                f"{bm['precision']:>6.3f} {bm['recall']:>6.3f} "
                f"{bm['f1']:>6.3f} {srcc_s:>7s}"
            )

        # Confusion matrix
        conf = np.array(d["confusion_matrix"])
        print(f"\n    Confusion Matrix (rows=true, cols=pred):")
        print(f"    {'':>12s}", end="")
        for bname in BUCKET_NAMES:
            print(f" {bname[:4]:>5s}", end="")
        print()
        for i, bname in enumerate(BUCKET_NAMES):
            print(f"    {bname:<12s}", end="")
            for j in range(5):
                print(f" {conf[i][j]:>5d}", end="")
            print()


def main() -> None:
    """Run ordinal analysis on all completed models."""
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} images")

    # Find all completed checkpoints
    completed_models = []
    for cp_file in sorted(CHECKPOINT_DIR.glob("*.jsonl")):
        n_lines = sum(1 for _ in cp_file.open())
        if n_lines >= 1000:
            model_id = cp_file.stem.replace("__", "/")
            completed_models.append(model_id)

    if not completed_models:
        print("No completed models found in checkpoints/")
        sys.exit(1)

    print(f"Completed models: {', '.join(completed_models)}")

    all_analyses: dict[str, dict] = {}
    for model_id in completed_models:
        results = load_model_results(model_id)
        analysis = analyze_model(model_id, gt, results)
        all_analyses[model_id] = analysis
        print_analysis(analysis)

    # Cross-model comparison table
    print(f"\n\n{'=' * 80}")
    print("CROSS-MODEL ORDINAL COMPARISON (Overall Quality)")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<35s} {'Exact':>6s} {'Adj±1':>6s} {'Kappa':>7s} "
        f"{'wKappa':>7s} {'Over%':>6s}"
    )
    print("-" * 80)
    for model_id, a in all_analyses.items():
        d = a.get("overall", {})
        if "error" in d:
            continue
        print(
            f"{model_id:<35s} "
            f"{d['exact_accuracy']:>6.1%} "
            f"{d['adjacent_accuracy']:>6.1%} "
            f"{d['cohens_kappa']:>7.4f} "
            f"{d['weighted_kappa']:>7.4f} "
            f"{d['over_rate_pct']:>5.1f}%"
        )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ordinal_analysis.json"
    # Convert numpy arrays to lists for JSON serialization
    out.write_text(json.dumps(all_analyses, indent=2, default=str))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
