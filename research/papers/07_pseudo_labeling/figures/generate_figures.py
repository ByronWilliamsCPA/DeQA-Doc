"""Generate all figures for Paper 7: Iterative Pseudo-Labeling Pipeline.

Produces 3 figures from raw JSON data:
  1. Pipeline flow diagram (conceptual block diagram)
  2. Calibration methods comparison (pre vs post MAE bar chart)
  3. OOD gating decision flowchart (threshold-annotated decision tree)

Usage:
    python generate_figures.py           # Generate all figures
    python generate_figures.py --show    # Show instead of saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Add shared infrastructure to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/papers/
from shared.plot_style import apply_arxiv_style, save_figure, DIM_COLORS
from shared.constants import PROJECT_ROOT, RESULTS_DIR
from shared.data_loader import load_json

FIGURES_DIR = Path(__file__).resolve().parent

# Paths to data files
CALIBRATION_RESULTS = RESULTS_DIR / "siglip2_diqa5000" / "calibration_results.json"
SUMMARY_JSON = RESULTS_DIR / "siglip2_diqa5000" / "summary.json"
SWEEP_RESULTS = RESULTS_DIR / "threshold_sensitivity" / "sweep_results.json"


def load_calibration_data() -> dict:
    """Load calibration results JSON."""
    return load_json(CALIBRATION_RESULTS)


def load_sweep_data() -> dict:
    """Load threshold sensitivity sweep results."""
    with open(SWEEP_RESULTS) as f:
        import json
        content = f.read()
        # Handle Infinity values in JSON
        content = content.replace("Infinity", "1e9")
        return json.loads(content)


def figure_1_pipeline_flow(show: bool = False) -> None:
    """Figure 1: End-to-end pipeline flow diagram."""
    apply_arxiv_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_aspect("equal")

    # Colors
    input_color = "#E3F2FD"
    process_color = "#E8F5E9"
    gate_color = "#FFF3E0"
    output_color = "#F3E5F5"
    reject_color = "#FFEBEE"
    border_color = "#424242"

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=border_color, linewidth=1.2
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, wrap=True)

    def draw_diamond(x, y, size, text, color, fontsize=8):
        diamond = plt.Polygon(
            [(x, y + size), (x + size * 1.3, y), (x, y - size), (x - size * 1.3, y)],
            facecolor=color, edgecolor=border_color, linewidth=1.2
        )
        ax.add_patch(diamond)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label="", label_offset=(0, 0.15)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=border_color, lw=1.5)
        )
        if label:
            mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                    fontstyle="italic", color="#616161")

    # Row 1: Input
    draw_box(1.5, 5.0, 2.2, 0.7, "Document\nImage", input_color, fontsize=10, bold=True)

    # Row 2: SigLIP2 embedding
    draw_box(1.5, 3.8, 2.2, 0.7, "SigLIP2 Backbone\n(768-dim embedding)", process_color)
    draw_arrow(1.5, 4.65, 1.5, 4.15)

    # Row 3: OOD gate (diamond)
    draw_diamond(1.5, 2.5, 0.55, "OOD\nGate", gate_color)
    draw_arrow(1.5, 3.45, 1.5, 3.05)

    # Branch: Hard reject
    draw_box(4.5, 1.2, 1.8, 0.6, "Hard Reject\n(d > 58.6)", reject_color, fontsize=8)
    draw_arrow(2.3, 2.0, 3.6, 1.2, label="d > 58.6", label_offset=(0.1, 0.2))

    # Branch: Soft flag (Tier 2)
    draw_box(4.5, 2.5, 2.0, 0.7, "Tier 2: VLM\nCross-Validation\n(Qwen3-VL-8B)", gate_color)
    draw_arrow(2.8, 2.5, 3.5, 2.5, label="46.0 < d < 58.6", label_offset=(0, 0.25))

    # Branch: Accept
    draw_box(4.5, 4.0, 2.0, 0.7, "VLM Teacher\n(Gemini 3 Flash)\nwSRCC = 0.708", process_color)
    draw_arrow(2.3, 3.0, 3.5, 3.7, label="d < 46.0", label_offset=(-0.1, 0.2))

    # Calibration
    draw_box(7.5, 4.0, 2.0, 0.7, "Isotonic\nCalibration\n14x MAE reduction", process_color)
    draw_arrow(5.5, 4.0, 6.5, 4.0)

    # Uncertainty filter
    draw_diamond(7.5, 2.5, 0.55, "sigma^2\n< 0.64?", gate_color)
    draw_arrow(7.5, 3.3, 7.5, 3.05)

    # Auto-accept path
    draw_box(9.3, 3.5, 1.2, 0.6, "Auto-\nAccept\n(w=1.0)", output_color, fontsize=8)
    draw_arrow(8.3, 3.0, 8.7, 3.3, label="Yes", label_offset=(0, 0.15))

    # Low-weight path
    draw_box(9.3, 1.8, 1.2, 0.6, "Low\nWeight\n(w<1.0)", gate_color, fontsize=8)
    draw_arrow(8.3, 2.0, 8.7, 1.9, label="No", label_offset=(0, 0.15))

    # Student training
    draw_box(5.0, 5.2, 2.6, 0.6, "Student: SigLIP2-IQA-Base (86M)\nRetrain on expanded data", output_color,
             fontsize=9, bold=True)

    # Feedback loop arrow
    ax.annotate(
        "", xy=(1.5, 5.35), xytext=(3.7, 5.35),
        arrowprops=dict(
            arrowstyle="-|>", color="#1565C0", lw=1.8,
            connectionstyle="arc3,rad=-0.3"
        )
    )
    ax.text(2.6, 5.7, "Re-fit OOD detector\n& iterate", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#1565C0")

    ax.set_title("Figure 1: Iterative Pseudo-Labeling Pipeline for Document IQA Domain Expansion",
                 fontsize=11, fontweight="bold", pad=10)

    if show:
        plt.show()
    else:
        save_figure(fig, FIGURES_DIR / "fig1_pipeline_flow.png", dpi=300)


def figure_2_calibration_comparison(show: bool = False) -> None:
    """Figure 2: Calibration methods comparison - pre vs post MAE."""
    apply_arxiv_style()

    data = load_calibration_data()
    per_dim = data["per_dimension"]

    methods = ["raw", "linear", "4PL", "isotonic"]
    method_labels = ["Raw\n(uncalibrated)", "Linear", "4-Parameter\nLogistic", "Isotonic"]
    dimensions = ["overall", "sharpness", "color"]
    dim_labels = ["Overall", "Sharpness", "Color Fidelity"]
    dim_colors = [DIM_COLORS["overall"], DIM_COLORS["sharpness"], DIM_COLORS["color_fidelity"]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)

    for i, (dim, dim_label) in enumerate(zip(dimensions, dim_labels)):
        ax = axes[i]
        mae_values = []
        ci_lows = []
        ci_highs = []

        for method in methods:
            m = per_dim[dim][method]["MAE"]
            mae_values.append(m["point"])
            ci_lows.append(m["point"] - m["ci_lower"])
            ci_highs.append(m["ci_upper"] - m["point"])

        x = np.arange(len(methods))
        bars = ax.bar(
            x, mae_values,
            yerr=[ci_lows, ci_highs],
            color=[
                "#D32F2F" if j == 0 else dim_colors[i]
                for j in range(len(methods))
            ],
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=7)
        ax.set_title(dim_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE (MOS scale)" if i == 0 else "")

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, mae_values)):
            label_y = bar.get_height() + ci_highs[j] + 0.02
            if val > 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color="#D32F2F")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        # Break axis indicator for raw value
        if mae_values[0] > 1.0:
            ax.set_ylim(0, max(mae_values) * 1.15)

    # Add reduction annotation
    raw_mae_avg = np.mean([per_dim[d]["raw"]["MAE"]["point"] for d in dimensions])
    iso_mae_avg = np.mean([per_dim[d]["isotonic"]["MAE"]["point"] for d in dimensions])
    reduction = raw_mae_avg / iso_mae_avg

    fig.suptitle(
        f"Figure 2: Calibration Methods Comparison ({reduction:.0f}x MAE Reduction: "
        f"Raw {raw_mae_avg:.2f} to Isotonic {iso_mae_avg:.3f})",
        fontsize=10, fontweight="bold", y=1.02
    )

    if show:
        plt.show()
    else:
        save_figure(fig, FIGURES_DIR / "fig2_calibration_comparison.png", dpi=300)


def figure_3_ood_gating_flowchart(show: bool = False) -> None:
    """Figure 3: OOD gating decision flowchart with threshold annotations."""
    apply_arxiv_style()

    # Load sweep data for the "current" config
    sweep_data = load_sweep_data()
    current_test = sweep_data["tier1_sweep"]["results"]["current"]["test"]["overall"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    border_color = "#424242"

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=border_color, linewidth=1.3
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight)

    def draw_diamond(x, y, size, text, color, fontsize=8):
        diamond = plt.Polygon(
            [(x, y + size), (x + size * 1.5, y), (x, y - size), (x - size * 1.5, y)],
            facecolor=color, edgecolor=border_color, linewidth=1.3
        )
        ax.add_patch(diamond)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label="", label_offset=(0, 0.12), color_line=None):
        c = color_line or border_color
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5)
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
                    fontstyle="italic", color="#616161",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.8))

    # Input
    draw_box(5, 6.3, 2.2, 0.6, "Input Document Image\n+ SigLIP2 Embedding", "#E3F2FD",
             fontsize=9, bold=True)

    # Step 1: Mahalanobis distance
    draw_box(5, 5.2, 2.8, 0.6, "Compute Mahalanobis Distance\nd = sqrt((x-mu)^T * Sigma^-1 * (x-mu))",
             "#E8F5E9", fontsize=8)
    draw_arrow(5, 6.0, 5, 5.5)

    # Decision 1: Hard reject?
    draw_diamond(5, 4.0, 0.5, "d > 58.6?", "#FFF3E0")
    draw_arrow(5, 4.9, 5, 4.5)

    # Hard reject outcome
    draw_box(8.5, 4.0, 1.8, 0.6,
             f"HARD REJECT\nn={current_test['n_hard_reject']} ({current_test['pct_hard_reject']:.1f}%)",
             "#FFCDD2", fontsize=8, bold=True)
    draw_arrow(6.5, 4.0, 7.6, 4.0, label="Yes (p99)", label_offset=(0, 0.2))

    # Decision 2: OOD flag?
    draw_diamond(5, 2.7, 0.5, "d > 46.0?", "#FFF3E0")
    draw_arrow(5, 3.5, 5, 3.2, label="No", label_offset=(-0.3, 0))

    # Tier 2 trigger
    draw_box(8.5, 2.7, 1.8, 0.6,
             f"TIER 2 TRIGGER\nn={current_test['n_tier2_trigger']} ({current_test['pct_tier2_trigger']:.1f}%)",
             "#FFE0B2", fontsize=8, bold=True)
    draw_arrow(6.5, 2.7, 7.6, 2.7, label="Yes (p95-p99)", label_offset=(0, 0.2))

    # Decision 3: Uncertainty check
    draw_diamond(5, 1.4, 0.5, "sigma^2\n< 0.64?", "#FFF3E0")
    draw_arrow(5, 2.2, 5, 1.9, label="No", label_offset=(-0.3, 0))

    # Auto-accept
    draw_box(1.5, 1.4, 1.8, 0.6,
             f"AUTO-ACCEPT\nn={current_test['n_auto_accept']} ({current_test['pct_auto_accept']:.1f}%)\nw = 1.0",
             "#C8E6C9", fontsize=8, bold=True)
    draw_arrow(3.5, 1.4, 2.4, 1.4, label="Yes", label_offset=(0, 0.2))

    # Low weight
    draw_box(8.5, 1.4, 1.8, 0.6,
             f"LOW WEIGHT\nn={current_test['n_low_weight']} ({current_test['pct_low_weight']:.1f}%)\nw = sigma-based",
             "#FFF9C4", fontsize=7.5)
    draw_arrow(6.5, 1.4, 7.6, 1.4, label="No", label_offset=(0, 0.2))

    # Summary stats box
    total = current_test["n_total"]
    summary = (
        f"Test Set Summary (n={total})\n"
        f"Auto-accept: {current_test['pct_auto_accept']:.1f}%  |  "
        f"Tier 2: {current_test['pct_tier2_trigger']:.1f}%  |  "
        f"Reject: {current_test['pct_hard_reject']:.1f}%\n"
        f"Effective weight: {current_test['mean_weight']:.3f}"
    )
    ax.text(5, 0.4, summary, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=1),
            family="monospace")

    ax.set_title("Figure 3: OOD Gating Decision Tree (Current Configuration)",
                 fontsize=11, fontweight="bold", pad=10)

    if show:
        plt.show()
    else:
        save_figure(fig, FIGURES_DIR / "fig3_ood_gating_flowchart.png", dpi=300)


def main() -> None:
    """Generate all figures."""
    parser = argparse.ArgumentParser(description="Generate Paper 7 figures")
    parser.add_argument("--show", action="store_true", help="Show figures instead of saving")
    args = parser.parse_args()

    print("Paper 7: Iterative Pseudo-Labeling Pipeline")
    print("=" * 50)

    print("\nFigure 1: Pipeline flow diagram...")
    figure_1_pipeline_flow(show=args.show)

    print("\nFigure 2: Calibration methods comparison...")
    figure_2_calibration_comparison(show=args.show)

    print("\nFigure 3: OOD gating decision flowchart...")
    figure_3_ood_gating_flowchart(show=args.show)

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
