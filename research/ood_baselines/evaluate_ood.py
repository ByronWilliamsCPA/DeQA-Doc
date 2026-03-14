"""Evaluate OOD detection baselines on SigLIP2 embeddings.

Compares four OOD scoring methods (Mahalanobis, k-NN, cosine, energy) on the
same pre-extracted embeddings. Computes AUROC, AUPRC, FPR@95TPR, FPR@99TPR
with bootstrap confidence intervals.

Usage:
    cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py

    # With real OOD embeddings (future):
    cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py \
        --ood-labels path/to/ood_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # research/
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "papers"))
sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import apply_arxiv_style, save_figure

from research.ood_baselines.ood_methods import (
    cosine_scores,
    energy_scores,
    knn_scores,
    mahalanobis_scores,
)

EMBED_DIR = REPO_ROOT / "results" / "siglip2_diqa5000" / "embeddings"
OOD_DETECTOR_PATH = REPO_ROOT / "results" / "siglip2_diqa5000" / "ood_detector_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "ood_baselines"
FIGURE_DIR = OUTPUT_DIR / "figures"

METHODS = {
    "mahalanobis": mahalanobis_scores,
    "knn_k10": lambda tr, te: knn_scores(tr, te, k=10),
    "cosine": cosine_scores,
    "energy": energy_scores,
}

METHOD_LABELS = {
    "mahalanobis": "Mahalanobis (Ledoit-Wolf)",
    "knn_k10": "k-NN (k=10)",
    "cosine": "Cosine distance",
    "energy": "Energy (neg. LogSumExp)",
}

METHOD_COLORS = {
    "mahalanobis": "#2171B5",
    "knn_k10": "#D94701",
    "cosine": "#238B45",
    "energy": "#6A3D9A",
}


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load train+val (reference) and test embeddings.

    Returns:
        (reference_emb, test_emb) — reference is train+val concatenated.
    """
    train = np.load(EMBED_DIR / "train.npz")
    val = np.load(EMBED_DIR / "val.npz")
    test = np.load(EMBED_DIR / "test.npz")

    ref_emb = np.concatenate([train["embeddings"], val["embeddings"]], axis=0)
    test_emb = test["embeddings"]
    test_names = test["image_names"]

    print(f"Reference (train+val): {ref_emb.shape}")
    print(f"Test: {test_emb.shape}")
    return ref_emb, test_emb, test_names


def construct_proxy_labels(test_emb: np.ndarray) -> np.ndarray:
    """Construct pseudo-OOD labels using pre-fitted Mahalanobis detector.

    Uses the pre-fitted detector's calibration to label test images:
    images with Mahalanobis distance > train+val p99 are labeled OOD.

    This is circular for Mahalanobis evaluation but valid for relative
    comparison between methods.

    Returns:
        Binary labels, shape (N_test,). 1 = OOD, 0 = ID.
    """
    data = np.load(OOD_DETECTOR_PATH)
    mean = data["mean"]
    precision = data["precision_matrix"]

    # Compute Mahalanobis distances using pre-fitted parameters
    diffs = test_emb.astype(np.float64) - mean[np.newaxis, :]
    transformed = diffs @ precision
    distances = np.sqrt(np.sum(transformed * diffs, axis=1))

    # Use train+val p99 as threshold for pseudo-labels
    cal_distances = data["calibration_distances"]
    threshold = float(np.percentile(cal_distances, 99))
    labels = (distances > threshold).astype(int)

    n_ood = labels.sum()
    print(f"Proxy labels: {n_ood}/{len(labels)} OOD "
          f"(threshold={threshold:.2f}, pre-fitted Mahalanobis p99)")
    return labels


def load_real_ood_labels(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load real OOD embeddings and labels from NPZ.

    Expected keys:
        - embeddings: (N, D) float32
        - labels: (N,) int — 1 = OOD, 0 = ID
        - categories: (N,) str (optional) — per-sample category names

    Returns:
        (eval_embeddings, labels, categories) where eval_embeddings
        combines test ID + OOD samples with ground truth labels.
    """
    data = np.load(path, allow_pickle=True)
    categories = data["categories"] if "categories" in data else None
    n_id = int((data["labels"] == 0).sum())
    n_ood = int((data["labels"] == 1).sum())
    print(f"Loaded eval set: {data['embeddings'].shape} "
          f"(ID={n_id}, OOD={n_ood}, total={len(data['labels'])})")
    return data["embeddings"], data["labels"], categories


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    """Compute FPR at a given TPR threshold.

    Args:
        labels: Binary ground truth (1 = OOD positive).
        scores: OOD scores (higher = more OOD).
        target_tpr: Target true positive rate (e.g., 0.95).

    Returns:
        False positive rate at the target TPR.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find the threshold where TPR >= target
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return fpr[-1]
    return float(fpr[idx])


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute AUROC with bootstrap 95% CI.

    Returns:
        (auroc, [ci_lower, ci_upper])
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_labels, b_scores = labels[idx], scores[idx]
        # Skip degenerate bootstrap samples
        if len(np.unique(b_labels)) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(b_labels, b_scores)
        boot_aurocs.append(auc(b_fpr, b_tpr))

    ci = [float(np.percentile(boot_aurocs, 2.5)),
          float(np.percentile(boot_aurocs, 97.5))]
    return float(auroc), ci


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute area under precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def evaluate_all_methods(
    ref_emb: np.ndarray,
    eval_emb: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Run all OOD scoring methods and compute metrics.

    Args:
        ref_emb: Reference (train+val) embeddings.
        eval_emb: Evaluation embeddings (test, or test+synthetic).
        labels: Binary OOD labels for eval_emb.

    Returns:
        Dict of method_name -> metrics dict.
    """
    results = {}
    roc_data = {}

    for name, method in METHODS.items():
        print(f"\nScoring: {METHOD_LABELS[name]}...")
        scores = method(ref_emb, eval_emb)

        auroc, auroc_ci = bootstrap_auroc(labels, scores)
        auprc = compute_auprc(labels, scores)
        fpr95 = fpr_at_tpr(labels, scores, 0.95)
        fpr99 = fpr_at_tpr(labels, scores, 0.99)

        results[name] = {
            "auroc": round(auroc, 4),
            "auroc_ci": [round(c, 4) for c in auroc_ci],
            "auprc": round(auprc, 4),
            "fpr_at_95tpr": round(fpr95, 4),
            "fpr_at_99tpr": round(fpr99, 4),
        }

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_data[name] = (fpr, tpr)

        print(f"  AUROC={auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]  "
              f"AUPRC={auprc:.4f}  FPR@95={fpr95:.4f}  FPR@99={fpr99:.4f}")

    return results, roc_data


def plot_roc_comparison(roc_data: dict, results: dict) -> None:
    """Plot overlaid ROC curves for all methods."""
    apply_arxiv_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for name in METHODS:
        fpr, tpr = roc_data[name]
        auroc = results[name]["auroc"]
        ax.plot(fpr, tpr, color=METHOD_COLORS[name],
                label=f"{METHOD_LABELS[name]} (AUROC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection: ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_figure(fig, FIGURE_DIR / "roc_comparison.png")


def plot_auroc_bars(results: dict) -> None:
    """Plot AUROC bar chart with confidence intervals."""
    apply_arxiv_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    names = list(METHODS.keys())
    aurocs = [results[n]["auroc"] for n in names]
    ci_lower = [results[n]["auroc"] - results[n]["auroc_ci"][0] for n in names]
    ci_upper = [results[n]["auroc_ci"][1] - results[n]["auroc"] for n in names]
    colors = [METHOD_COLORS[n] for n in names]
    labels = [METHOD_LABELS[n] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, color=colors, width=0.6,
                  yerr=[ci_lower, ci_upper], capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("OOD Detection Method Comparison")

    # Show values on bars
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Set y-axis to show differences clearly
    min_auroc = min(aurocs)
    ax.set_ylim(max(0, min_auroc - 0.05), 1.01)

    save_figure(fig, FIGURE_DIR / "auroc_bar_comparison.png")


def evaluate_per_category(
    ref_emb: np.ndarray,
    eval_emb: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute AUROC per OOD category for each method.

    For each OOD category, builds a subset of all ID samples + that category's
    OOD samples, then computes AUROC. Skips categories with < 5 OOD samples.

    Returns:
        Dict of {category: {method_name: auroc, ..., "n_ood": count}}.
    """
    unique_cats = sorted(set(categories[labels == 1]))
    id_mask = labels == 0

    per_cat: dict[str, dict[str, float | None]] = {}
    for cat in unique_cats:
        cat_ood_mask = (categories == cat) & (labels == 1)
        n_ood = int(cat_ood_mask.sum())
        if n_ood < 5:
            print(f"  Skipping {cat} (n_ood={n_ood} < 5)")
            continue

        subset_mask = id_mask | cat_ood_mask
        sub_emb = eval_emb[subset_mask]
        sub_labels = labels[subset_mask]

        cat_results: dict[str, float | None] = {"n_ood": n_ood}
        for name, method in METHODS.items():
            scores = method(ref_emb, sub_emb)
            try:
                fpr_arr, tpr_arr, _ = roc_curve(sub_labels, scores)
                auroc = float(auc(fpr_arr, tpr_arr))
            except ValueError:
                auroc = None
            cat_results[name] = round(auroc, 4) if auroc is not None else None

        per_cat[cat] = cat_results
        method_strs = "  ".join(
            f"{m}={cat_results[m]:.4f}" if cat_results[m] is not None else f"{m}=N/A"
            for m in METHODS
        )
        print(f"  {cat} (n={n_ood}): {method_strs}")

    return per_cat


def plot_per_category_heatmap(per_cat: dict[str, dict]) -> None:
    """Plot method x OOD category AUROC heatmap."""
    apply_arxiv_style()

    cats = list(per_cat.keys())
    methods = list(METHODS.keys())

    data = np.full((len(methods), len(cats)), np.nan)
    for j, cat in enumerate(cats):
        for i, method in enumerate(methods):
            val = per_cat[cat].get(method)
            if val is not None:
                data[i, j] = val

    # Short category names (strip ood_ prefix)
    short_cats = [c.replace("ood_", "") for c in cats]
    method_labels = [METHOD_LABELS[m] for m in methods]

    fig, ax = plt.subplots(
        figsize=(max(8, len(cats) * 0.8 + 2), 4),
        layout="constrained",
    )
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        xticklabels=short_cats,
        yticklabels=method_labels,
        linewidths=0.5,
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title("Per-Category OOD Detection AUROC")
    ax.set_xlabel("OOD Category")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    save_figure(fig, FIGURE_DIR / "per_category_heatmap.png")


def write_results_json(
    results: dict,
    label_source: str,
    per_category: dict | None = None,
    eval_set: dict | None = None,
) -> None:
    """Write machine-readable results JSON."""
    output = {
        "label_source": "ground_truth" if label_source == "real" else label_source,
        "methods": results,
        "note": (
            "Proxy labels derived from pre-fitted Mahalanobis p99 threshold. "
            "Mahalanobis AUROC is circular; compare other methods against it. "
            "Re-run with --ood-labels for proper evaluation."
            if label_source == "proxy"
            else "Ground truth OOD labels from synthetic dataset (DIQA test=ID, "
            "synthetic OOD categories=OOD)."
        ),
    }
    if eval_set is not None:
        output["eval_set"] = eval_set
    if per_category is not None:
        output["per_category"] = per_category
    out_path = OUTPUT_DIR / "ood_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


def write_results_md(
    results: dict,
    label_source: str,
    per_category: dict | None = None,
    eval_set: dict | None = None,
) -> None:
    """Write human-readable results summary."""
    lines = ["# OOD Detection Baseline Comparison", ""]

    if label_source == "real" and eval_set:
        lines.extend([
            "Comparison of four OOD scoring methods on SigLIP2 embeddings (768-dim) "
            f"with **ground truth** ID/OOD labels. Eval set: {eval_set['n_id']} ID + "
            f"{eval_set['n_ood']} OOD = {eval_set['n_total']} samples.",
            "",
            "- **ID**: DIQA-5000 test (1,000) + synthetic in-distribution (150)",
            "- **OOD**: Synthetic out-of-distribution (370) across 13 categories",
            "- **Reference**: train+val (4,000) used for fitting all methods",
            "",
        ])
    else:
        lines.extend([
            "Comparison of four OOD scoring methods on DIQA-5000 SigLIP2 embeddings "
            "(768-dim, train+val=4000, test=1000).",
            "",
        ])

    if label_source == "proxy":
        lines.extend([
            "> **Note**: OOD labels are proxy labels derived from the pre-fitted "
            "Mahalanobis detector (train+val p99 threshold). Mahalanobis AUROC is "
            "circular in this setup. Compare other methods relative to Mahalanobis.",
            "",
        ])

    # Results table
    lines.extend([
        "## Results",
        "",
        "| Method | AUROC | 95% CI | AUPRC | FPR@95TPR | FPR@99TPR |",
        "| ------ | ----- | ------ | ----- | --------- | --------- |",
    ])

    for name in METHODS:
        r = results[name]
        ci = f"[{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]"
        lines.append(
            f"| {METHOD_LABELS[name]} | {r['auroc']:.4f} | {ci} | "
            f"{r['auprc']:.4f} | {r['fpr_at_95tpr']:.4f} | {r['fpr_at_99tpr']:.4f} |"
        )

    # Per-category breakdown
    if per_category:
        lines.extend([
            "",
            "## Per-Category AUROC Breakdown",
            "",
        ])
        method_names = list(METHODS.keys())
        header = "| Category | n_OOD | " + " | ".join(METHOD_LABELS[m] for m in method_names) + " |"
        sep = "| ------ | ---: | " + " | ".join("----:" for _ in method_names) + " |"
        lines.extend([header, sep])

        for cat in sorted(per_category.keys()):
            cat_data = per_category[cat]
            n_ood = cat_data.get("n_ood", "?")
            vals = []
            for m in method_names:
                v = cat_data.get(m)
                vals.append(f"{v:.4f}" if v is not None else "N/A")
            short_cat = cat.replace("ood_", "")
            lines.append(f"| {short_cat} | {n_ood} | " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "## Figures",
        "",
        "- [ROC curves](figures/roc_comparison.png)",
        "- [AUROC bar chart](figures/auroc_bar_comparison.png)",
    ])
    if per_category:
        lines.append("- [Per-category heatmap](figures/per_category_heatmap.png)")

    lines.extend(["", "## Limitations", ""])

    if label_source == "proxy":
        lines.extend([
            "1. **Proxy OOD labels**: Ground truth OOD labels are derived from the "
            "pre-fitted Mahalanobis detector's train+val p99 threshold. This makes "
            "the Mahalanobis AUROC circular (evaluating the method against its own "
            "labels). The relative ranking of other methods is still informative.",
            "2. **No real OOD data**: Synthetic OOD embeddings were not available. "
            "Re-run with `--ood-labels` when SigLIP2 embeddings for the 520 "
            "synthetic OOD images are extracted.",
            "3. **Single embedding space**: All methods use the same SigLIP2 "
            "embeddings. Results may differ with other backbone features.",
        ])
    else:
        lines.extend([
            "1. **Single embedding space**: All methods use the same SigLIP2 "
            "embeddings. Results may differ with other backbone features.",
            "2. **Synthetic OOD**: OOD images are programmatically generated, "
            "not real-world OOD documents. Results indicate separability of the "
            "synthetic categories but may not generalize to all OOD types.",
        ])

    lines.extend(["", "## Recommendation", ""])

    if label_source == "real":
        # Find best method
        best = max(results, key=lambda m: results[m]["auroc"])
        best_auroc = results[best]["auroc"]
        lines.extend([
            f"**Best method**: {METHOD_LABELS[best]} (AUROC={best_auroc:.4f}).",
            "",
            "See per-category breakdown above for method-specific strengths. "
            "The optimal production threshold should be calibrated at 95% or 99% TPR "
            "using the best-performing method's FPR values.",
        ])
    else:
        lines.append("*To be filled after reviewing results with real OOD labels.*")

    lines.append("")

    out_path = OUTPUT_DIR / "RESULTS.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary written to {out_path}")


def main() -> None:
    """Run OOD baseline evaluation."""
    parser = argparse.ArgumentParser(description="OOD detection baseline comparison")
    parser.add_argument(
        "--ood-labels",
        type=str,
        default=None,
        help="Path to NPZ with real OOD embeddings and labels "
        "(keys: 'embeddings', 'labels'). If not provided, uses proxy labels.",
    )
    args = parser.parse_args()

    ref_emb, test_emb, test_names = load_embeddings()

    categories = None
    if args.ood_labels:
        print(f"\nLoading real OOD labels from {args.ood_labels}")
        eval_emb, labels, categories = load_real_ood_labels(args.ood_labels)
        label_source = "real"
    else:
        print("\nNo --ood-labels provided, using proxy labels (Mahalanobis p99)")
        labels = construct_proxy_labels(test_emb)
        eval_emb = test_emb
        label_source = "proxy"

    results, roc_data = evaluate_all_methods(ref_emb, eval_emb, labels)

    # Per-category breakdown (only with real labels + categories)
    per_category = None
    if categories is not None:
        print("\n--- Per-category evaluation ---")
        per_category = evaluate_per_category(ref_emb, eval_emb, labels, categories)

    print("\n--- Generating figures ---")
    plot_roc_comparison(roc_data, results)
    plot_auroc_bars(results)
    if per_category:
        plot_per_category_heatmap(per_category)

    # Build eval_set summary
    eval_set = None
    if label_source == "real":
        n_id = int((labels == 0).sum())
        n_ood = int((labels == 1).sum())
        eval_set = {"n_id": n_id, "n_ood": n_ood, "n_total": n_id + n_ood}

    print("\n--- Writing results ---")
    write_results_json(results, label_source, per_category, eval_set)
    write_results_md(results, label_source, per_category, eval_set)


if __name__ == "__main__":
    main()
