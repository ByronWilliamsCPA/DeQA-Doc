"""k-NN OOD detection sensitivity analysis over k parameter.

Sweeps k ∈ {1, 3, 5, 10, 20, 50, 100} and reports AUROC for each.
Identifies optimal k and generates a sensitivity plot.

Usage:
    cd DeQA-Score && .venv/bin/python ../research/ood_baselines/knn_sensitivity.py

    # With real OOD labels:
    cd DeQA-Score && .venv/bin/python ../research/ood_baselines/knn_sensitivity.py \
        --ood-labels ../research/ood_baselines/eval_id_ood.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # research/
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "papers"))
sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import apply_arxiv_style, save_figure

from research.ood_baselines.ood_methods import knn_scores
EMBED_DIR = REPO_ROOT / "results" / "siglip2_diqa5000" / "embeddings"
OOD_DETECTOR_PATH = REPO_ROOT / "results" / "siglip2_diqa5000" / "ood_detector_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "ood_baselines"
FIGURE_DIR = OUTPUT_DIR / "figures"

K_VALUES = [1, 3, 5, 10, 20, 50, 100]


def load_data(
    ood_labels_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reference + eval embeddings and labels.

    Args:
        ood_labels_path: Path to NPZ with eval embeddings and ground truth
            labels (keys: 'embeddings', 'labels'). If None, constructs
            proxy labels from pre-fitted Mahalanobis p99.

    Returns:
        (ref_emb, eval_emb, labels)
    """
    train = np.load(EMBED_DIR / "train.npz")
    val = np.load(EMBED_DIR / "val.npz")
    ref_emb = np.concatenate([train["embeddings"], val["embeddings"]], axis=0)

    if ood_labels_path is not None:
        data = np.load(ood_labels_path, allow_pickle=True)
        eval_emb = data["embeddings"]
        labels = data["labels"]
        n_id = int((labels == 0).sum())
        n_ood = int((labels == 1).sum())
        print(f"Loaded real labels: {eval_emb.shape} (ID={n_id}, OOD={n_ood})")
        return ref_emb, eval_emb, labels

    # Proxy labels from pre-fitted Mahalanobis p99
    test = np.load(EMBED_DIR / "test.npz")
    test_emb = test["embeddings"]
    data = np.load(OOD_DETECTOR_PATH)
    mean = data["mean"]
    precision = data["precision_matrix"]
    diffs = test_emb.astype(np.float64) - mean[np.newaxis, :]
    transformed = diffs @ precision
    distances = np.sqrt(np.sum(transformed * diffs, axis=1))
    threshold = float(np.percentile(data["calibration_distances"], 99))
    labels = (distances > threshold).astype(int)

    return ref_emb, test_emb, labels


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def main() -> None:
    """Run k-NN sensitivity sweep."""
    parser = argparse.ArgumentParser(description="k-NN OOD detection sensitivity")
    parser.add_argument(
        "--ood-labels",
        type=str,
        default=None,
        help="Path to NPZ with eval embeddings and labels "
        "(keys: 'embeddings', 'labels'). If not provided, uses proxy labels.",
    )
    args = parser.parse_args()

    ref_emb, test_emb, labels = load_data(args.ood_labels)
    n_ood = labels.sum()
    print(f"Reference: {ref_emb.shape}, Test: {test_emb.shape}, OOD: {n_ood}")

    results = {}
    for k in K_VALUES:
        print(f"  k={k:>3d}...", end=" ", flush=True)
        scores = knn_scores(ref_emb, test_emb, k=k)
        auroc_val = compute_auroc(labels, scores)
        results[f"k{k}"] = round(auroc_val, 4)
        print(f"AUROC={auroc_val:.4f}")

    # Find optimal k
    best_k_key = max(results, key=results.get)
    best_auroc = results[best_k_key]
    print(f"\nBest: {best_k_key} with AUROC={best_auroc:.4f}")

    # Save to results JSON (merge with existing if present)
    results_path = OUTPUT_DIR / "ood_baseline_results.json"
    if results_path.exists():
        with open(results_path) as f:
            full_results = json.load(f)
    else:
        full_results = {}

    full_results["knn_sensitivity"] = results
    full_results["knn_best_k"] = best_k_key

    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results updated in {results_path}")

    # Plot
    apply_arxiv_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    aurocs = [results[f"k{k}"] for k in K_VALUES]
    ax.plot(K_VALUES, aurocs, "o-", color="#D94701", linewidth=2, markersize=7)

    # Highlight best k
    best_k_int = int(best_k_key[1:])
    ax.axvline(best_k_int, color="#D94701", alpha=0.3, linestyle="--")
    ax.annotate(f"Best: k={best_k_int}\nAUROC={best_auroc:.4f}",
                xy=(best_k_int, best_auroc),
                xytext=(best_k_int + 15, best_auroc - 0.005),
                fontsize=9, ha="left",
                arrowprops={"arrowstyle": "->", "color": "#666"})

    ax.set_xlabel("k (number of nearest neighbors)")
    ax.set_ylabel("AUROC")
    ax.set_title("k-NN OOD Detection: Sensitivity to k")
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    save_figure(fig, FIGURE_DIR / "knn_k_sensitivity.png")


if __name__ == "__main__":
    main()
