"""Generate all figures for Paper 9: Training HyperIQA++ for Document IQA.

Produces:
  1. fig1_id_vs_ood_performance.pdf  -- Grouped bar: HyperIQA++ vs SigLIP2 vs VLMs on ID and OOD
  2. fig2_architecture.pdf           -- Architecture diagram: ResNet-50 + HyperNet + spatial attention + 3 heads
  3. fig3_finetuning_gain.pdf        -- Bar chart: pretrained vs fine-tuned performance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.plot_style import apply_arxiv_style, save_figure, DIM_COLORS
from shared.constants import PROJECT_ROOT, RESULTS_DIR, DIMENSIONS, DIMENSION_LABELS, WSRCC_WEIGHTS, LICENSE
from shared.data_loader import load_json

FIGURES_DIR = Path(__file__).resolve().parent


# ── Data ─────────────────────────────────────────────────────────────────────

# MainScore values from the paper (source: diqa_5_hyperiqa_training.md)
MODEL_SCORES: dict[str, dict[str, float]] = {
    "HyperIQA++ (ours)": {"id": 0.856, "ood": 0.694},
    "SigLIP2-IQA-Base": {"id": 0.886, "ood": 0.659},
    "DeQA-Doc-3Spec.": {"id": 0.716, "ood": 0.746},
    "Gemini 3 Flash": {"id": 0.743, "ood": 0.782},
    "HyperIQA (off-the-shelf)": {"id": 0.437, "ood": 0.723},
}

# ID/OOD gap for annotating
ID_OOD_GAP: dict[str, float] = {
    "HyperIQA++ (ours)": -0.165,
    "SigLIP2-IQA-Base": +0.004,
    "DeQA-Doc-3Spec.": -0.096,
    "Gemini 3 Flash": -0.042,
    "HyperIQA (off-the-shelf)": +0.286,
}

# Pretrained vs fine-tuned (from baseline_summary.json and paper)
PRETRAINED_VS_FINETUNED: dict[str, dict[str, float]] = {
    "MainScore": {"pretrained": 0.437, "finetuned": 0.856},
    "SRCC (Overall)": {"pretrained": 0.420, "finetuned": 0.856},
    "PLCC (Overall)": {"pretrained": 0.437, "finetuned": 0.886},
}

# Per-dimension OOD metrics (SRCC and PLCC)
OOD_DIM_METRICS: dict[str, dict[str, float]] = {
    "overall": {"srcc": 0.589, "plcc": 0.780},
    "sharpness": {"srcc": 0.623, "plcc": 0.797},
    "color_fidelity": {"srcc": 0.606, "plcc": 0.790},
}


# ── Figure 1: ID vs OOD Performance ─────────────────────────────────────────

def fig1_id_vs_ood_performance() -> None:
    """Grouped bar chart comparing models on both ID and OOD test sets."""
    apply_arxiv_style()

    # Order by ID score descending
    model_order = [
        "SigLIP2-IQA-Base",
        "HyperIQA++ (ours)",
        "Gemini 3 Flash",
        "DeQA-Doc-3Spec.",
        "HyperIQA (off-the-shelf)",
    ]

    id_scores = [MODEL_SCORES[m]["id"] for m in model_order]
    ood_scores = [MODEL_SCORES[m]["ood"] for m in model_order]

    x = np.arange(len(model_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars_id = ax.bar(
        x - width / 2, id_scores, width,
        label="DIQA-5000 (ID, n=1,000)",
        color="#2171B5", edgecolor="white", linewidth=0.5,
    )
    bars_ood = ax.bar(
        x + width / 2, ood_scores, width,
        label="Synthetic OOD (n=520)",
        color="#D94701", edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("MainScore")
    ax.set_title("In-Distribution vs Out-of-Distribution Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    # Value labels
    for bar in bars_id:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
        )
    for bar in bars_ood:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

    # Delta annotations
    for i, name in enumerate(model_order):
        delta = ID_OOD_GAP[name]
        color = "#2ca25f" if delta >= 0 else "#d7191c"
        sign = "+" if delta >= 0 else ""
        y_pos = max(id_scores[i], ood_scores[i]) + 0.04
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(x[i], y_pos),
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=color,
        )

    save_figure(fig, FIGURES_DIR / "fig1_id_vs_ood_performance.pdf")
    save_figure(fig, FIGURES_DIR / "fig1_id_vs_ood_performance.png", dpi=150)


# ── Figure 2: Architecture Diagram ──────────────────────────────────────────

def fig2_architecture() -> None:
    """Architecture diagram: ResNet-50 + HyperNet + spatial attention + 3 distribution heads."""
    apply_arxiv_style()

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("HyperIQA++ Architecture", fontsize=13, fontweight="bold", pad=12)

    box_style = dict(boxstyle="round,pad=0.3", edgecolor="#333333", linewidth=1.2)

    # Input
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 2.0), 1.2, 1.0, **box_style, facecolor="#E8F4FD",
    ))
    ax.text(0.8, 2.5, "Document\nImage\n1600x1600", ha="center", va="center", fontsize=7.5)

    # ResNet-50 backbone
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.9, 1.8), 1.4, 1.4, **box_style, facecolor="#C6DBEF",
    ))
    ax.text(2.6, 2.5, "ResNet-50\nBackbone\n(ImageNet +\nKonIQ-10k)", ha="center", va="center", fontsize=7)

    # Spatial attention
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.8, 1.8), 1.4, 1.4, **box_style, facecolor="#A1D99B",
    ))
    ax.text(4.5, 2.5, "Spatial\nAttention\n(DocIQ)", ha="center", va="center", fontsize=7.5)

    # HyperNet
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.7, 1.8), 1.3, 1.4, **box_style, facecolor="#FDAE6B",
    ))
    ax.text(6.35, 2.5, "HyperNet\n(Content-\nAdaptive\nWeights)", ha="center", va="center", fontsize=7)

    # Three distribution heads
    head_colors = ["#2171B5", "#238B45", "#D94701"]
    head_labels = ["Overall\n10-bin dist.", "Sharpness\n10-bin dist.", "Color\n10-bin dist."]
    head_y_positions = [3.6, 2.0, 0.4]

    for i, (label, y, color) in enumerate(zip(head_labels, head_y_positions, head_colors)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (7.6, y), 1.6, 0.9, **box_style, facecolor=color, alpha=0.3,
        ))
        ax.text(8.4, y + 0.45, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

    # Arrows (main flow)
    arrow_props = dict(arrowstyle="-|>", color="#333333", linewidth=1.5)
    ax.annotate("", xy=(1.9, 2.5), xytext=(1.4, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 2.5), xytext=(3.3, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(5.7, 2.5), xytext=(5.2, 2.5), arrowprops=arrow_props)

    # Fan-out arrows to three heads
    for y in head_y_positions:
        ax.annotate("", xy=(7.6, y + 0.45), xytext=(7.0, 2.5), arrowprops=arrow_props)

    # Parameter count annotation
    ax.text(5.0, 0.3, "~138M parameters total", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#666666")

    save_figure(fig, FIGURES_DIR / "fig2_architecture.pdf")
    save_figure(fig, FIGURES_DIR / "fig2_architecture.png", dpi=150)


# ── Figure 3: Before/After Fine-tuning ──────────────────────────────────────

def fig3_finetuning_gain() -> None:
    """Bar chart showing pretrained vs fine-tuned performance gain."""
    apply_arxiv_style()

    metrics = list(PRETRAINED_VS_FINETUNED.keys())
    pretrained = [PRETRAINED_VS_FINETUNED[m]["pretrained"] for m in metrics]
    finetuned = [PRETRAINED_VS_FINETUNED[m]["finetuned"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars1 = ax.bar(
        x - width / 2, pretrained, width,
        label="HyperIQA (off-the-shelf)",
        color="#6baed6", edgecolor="white", linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2, finetuned, width,
        label="HyperIQA++ (fine-tuned)",
        color="#2171b5", edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("Score")
    ax.set_title("Fine-Tuning Impact: HyperIQA to HyperIQA++")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
        )

    # Improvement annotations
    for i in range(len(metrics)):
        if pretrained[i] > 0:
            pct = ((finetuned[i] - pretrained[i]) / pretrained[i]) * 100
            y_pos = max(pretrained[i], finetuned[i]) + 0.04
            ax.annotate(
                f"+{pct:.0f}%",
                xy=(x[i], y_pos),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d7191c",
            )

    save_figure(fig, FIGURES_DIR / "fig3_finetuning_gain.pdf")
    save_figure(fig, FIGURES_DIR / "fig3_finetuning_gain.png", dpi=150)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all Paper 9 figures."""
    print("Paper 9: Training HyperIQA++ -- Figure Generation")
    print("=" * 55)

    print("\nFigure 1: ID vs OOD Performance")
    fig1_id_vs_ood_performance()

    print("\nFigure 2: Architecture Diagram")
    fig2_architecture()

    print("\nFigure 3: Fine-tuning Gain")
    fig3_finetuning_gain()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
