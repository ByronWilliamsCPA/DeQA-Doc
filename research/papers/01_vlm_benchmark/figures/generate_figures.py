"""Generate all figures for Paper 1: VLM Benchmark for Document Image Quality Assessment.

Produces 5 figures from raw JSONL checkpoint data:
  1. wSRCC bar chart with 95% bootstrap CIs
  2. Per-dimension SRCC heatmap (models x dimensions)
  3. Predicted vs human MOS scatter (top 2 models)
  4. Ordinal confusion matrix (best model: Gemini 3 Flash)
  5. Systematic bias bar chart (mean bias per model per dimension)

Usage:
    python generate_figures.py           # Generate all figures
    python generate_figures.py --show    # Show instead of saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add shared infrastructure to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/papers/
from shared.plot_style import (
    DIM_COLORS,
    MODEL_COLORS,
    apply_arxiv_style,
    save_figure,
)
from shared.constants import (
    DIMENSIONS,
    DIMENSION_LABELS,
    PRIMARY_MODELS,
    PROJECT_ROOT,
    VLM_EVAL_DIR,
    MODEL_NAMES,
    WSRCC_WEIGHTS,
)
from shared.data_loader import (
    load_ground_truth,
    load_vlm_checkpoints,
    merge_predictions_with_gt,
    compute_metrics,
    load_json,
)

FIGURES_DIR = Path(__file__).resolve().parent

# DeQA-Doc-3Specialists baseline for reference lines
BASELINE_WSRCC = 0.716
BASELINE_SRCC = {"overall": 0.733, "sharpness": 0.681, "color_fidelity": 0.716}


def load_all_data() -> dict[str, dict]:
    """Load and compute metrics for all primary models."""
    gt = load_ground_truth()
    checkpoints = load_vlm_checkpoints(PRIMARY_MODELS)

    model_metrics: dict[str, dict] = {}
    model_merged: dict[str, list] = {}

    for model_id in PRIMARY_MODELS:
        if model_id not in checkpoints:
            print(f"  Skipping {model_id}: no checkpoint found")
            continue
        merged = merge_predictions_with_gt(checkpoints[model_id], gt)
        if len(merged) < 100:
            print(f"  Skipping {model_id}: only {len(merged)} matched records")
            continue
        metrics = compute_metrics(merged, n_bootstrap=1000)
        model_metrics[model_id] = metrics
        model_merged[model_id] = merged

    return model_metrics, model_merged


def figure_1_wsrcc_bar(model_metrics: dict[str, dict]) -> None:
    """Figure 1: wSRCC bar chart with 95% bootstrap CIs for all models."""
    apply_arxiv_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    models = [m for m in PRIMARY_MODELS if m in model_metrics]
    labels = [MODEL_NAMES.get(m, m) for m in models]
    wsrccs = [model_metrics[m]["wsrcc"] for m in models]

    # Compute wSRCC CIs from dimension CIs using the weights
    ci_lows = []
    ci_highs = []
    for m in models:
        met = model_metrics[m]
        dims = DIMENSIONS
        w = WSRCC_WEIGHTS
        low = sum(w[i] * met[dims[i]]["srcc_ci"][0] for i in range(3))
        high = sum(w[i] * met[dims[i]]["srcc_ci"][1] for i in range(3))
        ci_lows.append(low)
        ci_highs.append(high)

    errors_low = [wsrccs[i] - ci_lows[i] for i in range(len(models))]
    errors_high = [ci_highs[i] - wsrccs[i] for i in range(len(models))]

    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    bars = ax.barh(
        range(len(models)), wsrccs, color=colors, edgecolor="white", linewidth=0.5,
        xerr=[errors_low, errors_high], capsize=3, error_kw={"linewidth": 1, "color": "#333333"},
    )

    # Reference line for DeQA-Doc-3Specialists
    ax.axvline(BASELINE_WSRCC, color="#888888", linestyle="--", linewidth=1.2, label=f"DeQA-Doc-3Spec. ({BASELINE_WSRCC})")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weighted SRCC (wSRCC)")
    ax.set_title("Figure 1: VLM Benchmark on DIQA-5000 (n=1,000)")
    ax.set_xlim(0.3, 0.78)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    # Value labels
    for i, (v, lo, hi) in enumerate(zip(wsrccs, ci_lows, ci_highs)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")

    save_figure(fig, FIGURES_DIR / "fig1_wsrcc_bar.png")
    save_figure(fig, FIGURES_DIR / "fig1_wsrcc_bar.pdf")


def figure_2_dimension_heatmap(model_metrics: dict[str, dict]) -> None:
    """Figure 2: Per-dimension SRCC heatmap (models x dimensions)."""
    apply_arxiv_style()

    models = [m for m in PRIMARY_MODELS if m in model_metrics]
    labels = [MODEL_NAMES.get(m, m) for m in models]
    dim_labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]

    data = np.zeros((len(models), len(DIMENSIONS)))
    for i, m in enumerate(models):
        for j, d in enumerate(DIMENSIONS):
            data[i, j] = model_metrics[m][d]["srcc"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=0.75)

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels(dim_labels, rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(DIMENSIONS)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_title("Figure 2: SRCC by Model and Dimension")
    fig.colorbar(im, ax=ax, label="SRCC", shrink=0.8)

    save_figure(fig, FIGURES_DIR / "fig2_dimension_heatmap.png")
    save_figure(fig, FIGURES_DIR / "fig2_dimension_heatmap.pdf")


def figure_3_scatter(model_merged: dict[str, list]) -> None:
    """Figure 3: Predicted vs human MOS scatter for top 2 models."""
    apply_arxiv_style()

    top_models = [
        "google__gemini-3-flash-preview",
        "openai__gpt-4.1",
    ]
    top_models = [m for m in top_models if m in model_merged]

    fig, axes = plt.subplots(1, len(top_models), figsize=(10, 4.5), sharey=True)
    if len(top_models) == 1:
        axes = [axes]

    for idx, model_id in enumerate(top_models):
        ax = axes[idx]
        merged = model_merged[model_id]

        pred = np.array([r["pred_overall"] for r in merged])
        gt = np.array([r["gt_overall"] for r in merged])

        ax.scatter(gt, pred, alpha=0.25, s=12, color=MODEL_COLORS.get(model_id, "#888888"), edgecolors="none")

        # Regression line
        z = np.polyfit(gt, pred, 1)
        p = np.poly1d(z)
        x_range = np.linspace(gt.min(), gt.max(), 100)
        ax.plot(x_range, p(x_range), "k--", linewidth=1.2, label=f"r={np.corrcoef(gt, pred)[0, 1]:.3f}")

        # Perfect agreement line
        ax.plot([1, 5], [1, 5], ":", color="#999999", linewidth=0.8, label="y = x")

        from scipy import stats
        srcc, _ = stats.spearmanr(pred, gt)
        plcc, _ = stats.pearsonr(pred, gt)

        ax.set_xlabel("Human MOS")
        if idx == 0:
            ax.set_ylabel("VLM Predicted Score")
        ax.set_title(f"{MODEL_NAMES.get(model_id, model_id)}\nSRCC={srcc:.3f}, PLCC={plcc:.3f}")
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle("Figure 3: Predicted vs Human MOS (Overall Quality)", fontsize=12, y=1.02)

    save_figure(fig, FIGURES_DIR / "fig3_scatter_top2.png")
    save_figure(fig, FIGURES_DIR / "fig3_scatter_top2.pdf")


def figure_4_confusion_matrix() -> None:
    """Figure 4: Ordinal confusion matrix for best model (Gemini 3 Flash, overall)."""
    apply_arxiv_style()

    ordinal_data = load_json(VLM_EVAL_DIR / "results" / "ordinal_analysis.json")
    gemini_key = "google/gemini-3-flash-preview"
    if gemini_key not in ordinal_data:
        print(f"  Warning: {gemini_key} not in ordinal_analysis.json, skipping figure 4")
        return

    cm = np.array(ordinal_data[gemini_key]["overall"]["confusion_matrix"])
    labels = ["Bad", "Poor", "Fair", "Good", "Excellent"]

    # Normalize by row (true label)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.where(row_sums > 0, row_sums, 1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Quality Level")
    ax.set_ylabel("True Quality Level (Human MOS)")

    # Annotate with raw counts and percentages
    for i in range(5):
        for j in range(5):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if pct > 20 else "normal")

    ax.set_title("Figure 4: Ordinal Confusion Matrix\nGemini 3 Flash (Overall Quality, n=1,000)")
    fig.colorbar(im, ax=ax, label="Row-Normalized Proportion", shrink=0.8)

    save_figure(fig, FIGURES_DIR / "fig4_confusion_matrix.png")
    save_figure(fig, FIGURES_DIR / "fig4_confusion_matrix.pdf")


def figure_5_bias_chart(model_metrics: dict[str, dict]) -> None:
    """Figure 5: Systematic bias (mean pred - GT) per model per dimension."""
    apply_arxiv_style()

    models = [m for m in PRIMARY_MODELS if m in model_metrics]
    labels = [MODEL_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    n_dims = len(DIMENSIONS)
    bar_width = 0.22
    x = np.arange(len(models))

    for j, dim in enumerate(DIMENSIONS):
        biases = [model_metrics[m][dim]["bias"] for m in models]
        offset = (j - (n_dims - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, biases, bar_width,
            color=DIM_COLORS[dim], label=DIMENSION_LABELS[dim],
            edgecolor="white", linewidth=0.5,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Bias (Predicted - Human MOS)")
    ax.set_title("Figure 5: Systematic Bias by Model and Dimension")
    ax.legend(fontsize=8)

    save_figure(fig, FIGURES_DIR / "fig5_bias_chart.png")
    save_figure(fig, FIGURES_DIR / "fig5_bias_chart.pdf")


def main() -> None:
    """Generate all figures for Paper 1."""
    parser = argparse.ArgumentParser(description="Generate Paper 1 figures")
    parser.add_argument("--show", action="store_true", help="Show figures instead of saving")
    args = parser.parse_args()

    if args.show:
        plt.ion()

    print("Loading data and computing metrics...")
    model_metrics, model_merged = load_all_data()

    print(f"\nLoaded {len(model_metrics)} models:")
    for m, met in model_metrics.items():
        print(f"  {MODEL_NAMES.get(m, m)}: wSRCC={met['wsrcc']:.3f}, n={met['n']}")

    print("\n--- Figure 1: wSRCC Bar Chart ---")
    figure_1_wsrcc_bar(model_metrics)

    print("\n--- Figure 2: Dimension Heatmap ---")
    figure_2_dimension_heatmap(model_metrics)

    print("\n--- Figure 3: Scatter Plots ---")
    figure_3_scatter(model_merged)

    print("\n--- Figure 4: Confusion Matrix ---")
    figure_4_confusion_matrix()

    print("\n--- Figure 5: Bias Chart ---")
    figure_5_bias_chart(model_metrics)

    print("\nAll figures generated successfully.")

    if args.show:
        plt.show()
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
