"""Generate all figures for Paper 3: Prompt Engineering for VLM-Based Quality Assessment.

Produces:
    fig1_seven_arm_comparison.pdf   -- 7-arm prompt variant bar chart with regression-to-mean overlay
    fig2_ab_test_per_dimension.pdf  -- Grouped bar chart: 1-prompt vs 3-prompt per dimension
    fig3_regression_to_mean.pdf     -- Scatter: n=23 wSRCC vs n=1000 wSRCC with diagonal

Usage:
    python figures/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.constants import DIMENSIONS, DIMENSION_LABELS, PROJECT_ROOT, VLM_EVAL_DIR
from shared.data_loader import load_json, load_jsonl, load_ground_truth
from shared.plot_style import DIM_COLORS, apply_arxiv_style, save_figure

FIGURES_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_optimization_results() -> dict:
    """Load the 7-arm prompt optimization results for Gemini 3 Flash."""
    return load_json(VLM_EVAL_DIR / "prompt_optimization" / "optimization_results.json")


def load_qwen_optimization_results() -> dict:
    """Load the 7-arm prompt optimization results for Qwen 3.5 Flash."""
    return load_json(
        VLM_EVAL_DIR / "prompt_optimization" / "qwen__qwen3.5-flash-02-23" / "optimization_results.json"
    )


# Arm labels for the 7-arm experiment
ARM_LABELS: dict[str, str] = {
    "1_single_all3": "Single\n(baseline)",
    "2_separate_3": "Separate\n(3 calls)",
    "3_hybrid": "Hybrid",
    "4_few_shot": "Few-shot\n(3 examples)",
    "5_multi_sample": "Multi-sample\n(3x median)",
    "6_res_2048": "Resize\n2048px",
    "7_no_resize": "No resize\n(native)",
}

ARM_ORDER = list(ARM_LABELS.keys())


# ---------------------------------------------------------------------------
# Figure 1: Seven-arm comparison with regression-to-mean annotation
# ---------------------------------------------------------------------------

def fig1_seven_arm_comparison() -> None:
    """Bar chart comparing wSRCC across 7 prompt variants (n=23).

    Includes a horizontal reference line for the full-scale baseline (n=1000)
    and annotation showing that the n=23 winner (no-resize) did not replicate.
    """
    gemini = load_optimization_results()
    qwen = load_qwen_optimization_results()

    arms = ARM_ORDER
    gemini_wsrcc = [gemini[arm]["wsrcc"] for arm in arms]
    qwen_wsrcc = [qwen[arm]["wsrcc"] for arm in arms]
    labels = [ARM_LABELS[arm] for arm in arms]

    x = np.arange(len(arms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_g = ax.bar(x - width / 2, gemini_wsrcc, width, label="Gemini 3 Flash (n=23)",
                    color="#4285F4", edgecolor="white", linewidth=0.5, zorder=3)
    bars_q = ax.bar(x + width / 2, qwen_wsrcc, width, label="Qwen 3.5 Flash (n=23)",
                    color="#FF6B35", edgecolor="white", linewidth=0.5, zorder=3)

    # Full-scale baseline (Gemini at 1024px, n=1000)
    full_scale_wsrcc = 0.708
    ax.axhline(y=full_scale_wsrcc, color="#333333", linestyle="--", linewidth=1.0,
               label=f"Gemini full-scale (n=1000): {full_scale_wsrcc:.3f}", zorder=2)

    # Full-scale no-resize result
    noresize_full = 0.699
    ax.axhline(y=noresize_full, color="#999999", linestyle=":", linewidth=1.0,
               label=f"Gemini no-resize (n=1000): {noresize_full:.3f}", zorder=2)

    # Add value labels on bars
    for bar_set in [bars_g, bars_q]:
        for bar in bar_set:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.003,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=7)

    # Annotation arrow for regression to mean
    ax.annotate(
        "n=23 winner\ndoes not replicate\nat n=1000",
        xy=(6 - width / 2, gemini_wsrcc[6]),
        xytext=(5.0, 0.96),
        fontsize=8,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#D32F2F", "lw": 1.2},
        color="#D32F2F",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF3F3", "edgecolor": "#D32F2F", "alpha": 0.9},
    )

    ax.set_ylabel("Weighted SRCC (wSRCC)")
    ax.set_xlabel("Prompt Variant")
    ax.set_title("Seven-Arm Prompt Comparison (n=23 pilot)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.78, 0.98)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    save_figure(fig, FIGURES_DIR / "fig1_seven_arm_comparison.pdf")
    save_figure(fig, FIGURES_DIR / "fig1_seven_arm_comparison.png", dpi=150)


# ---------------------------------------------------------------------------
# Figure 2: A/B test per-dimension analysis
# ---------------------------------------------------------------------------

def fig2_ab_test_per_dimension() -> None:
    """Grouped bar chart: 1-prompt vs 3-prompt SRCC per dimension, two models."""
    # Data from computed A/B test results
    # 1-prompt (from full checkpoint, filtered to A/B images)
    one_prompt = {
        "Gemini 3 Flash": {"overall": 0.8950, "sharpness": 0.8599, "color_fidelity": 0.8452},
        "GPT-4.1": {"overall": 0.9140, "sharpness": 0.9055, "color_fidelity": 0.8293},
    }
    # 3-prompt (from A/B test files)
    three_prompt = {
        "Gemini 3 Flash": {"overall": 0.8780, "sharpness": 0.8961, "color_fidelity": 0.8597},
        "GPT-4.1": {"overall": 0.8628, "sharpness": 0.9241, "color_fidelity": 0.8661},
    }

    dims = ["overall", "sharpness", "color_fidelity"]
    dim_labels = [DIMENSION_LABELS[d] for d in dims]
    models = ["Gemini 3 Flash", "GPT-4.1"]
    model_colors_1p = {"Gemini 3 Flash": "#4285F4", "GPT-4.1": "#10A37F"}
    model_colors_3p = {"Gemini 3 Flash": "#7BAAF7", "GPT-4.1": "#66D4B5"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for ax, model in zip(axes, models):
        x = np.arange(len(dims))
        width = 0.35

        vals_1p = [one_prompt[model][d] for d in dims]
        vals_3p = [three_prompt[model][d] for d in dims]

        bars1 = ax.bar(x - width / 2, vals_1p, width, label="1-prompt (all dims)",
                       color=model_colors_1p[model], edgecolor="white", linewidth=0.5, zorder=3)
        bars3 = ax.bar(x + width / 2, vals_3p, width, label="3-prompt (per dim)",
                       color=model_colors_3p[model], edgecolor="white", linewidth=0.5, zorder=3)

        # Value labels
        for bars in [bars1, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.003,
                        f"{height:.3f}", ha="center", va="bottom", fontsize=7.5)

        # Delta annotations
        for i, dim in enumerate(dims):
            delta = vals_3p[i] - vals_1p[i]
            sign = "+" if delta > 0 else ""
            color = "#2E7D32" if delta > 0 else "#C62828"
            ax.text(x[i], min(vals_1p[i], vals_3p[i]) - 0.018,
                    f"{sign}{delta:.3f}", ha="center", va="top", fontsize=7,
                    color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0.80, 0.96)
        ax.legend(fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("SRCC (n=44)")

    fig.suptitle("A/B Test: Single-Prompt vs Per-Dimension Prompting", fontsize=12, y=1.02)

    save_figure(fig, FIGURES_DIR / "fig2_ab_test_per_dimension.pdf")
    save_figure(fig, FIGURES_DIR / "fig2_ab_test_per_dimension.png", dpi=150)


# ---------------------------------------------------------------------------
# Figure 3: Regression-to-mean scatter
# ---------------------------------------------------------------------------

def fig3_regression_to_mean() -> None:
    """Scatter: x = n=23 wSRCC, y = n=1000 wSRCC, with identity line.

    Shows how small-sample estimates regress toward the mean at full scale.
    Uses the 7-arm data for Gemini (n=23 pilot) and full-scale results.
    """
    gemini_pilot = load_optimization_results()

    # Full-scale equivalents (from VLM_TEACHER_EVALUATION.md and computed metrics)
    # Only arms 1 (baseline/single) and 7 (no-resize) were run at full scale
    # Other arms: use pilot data only, mark as "not validated"
    pilot_wsrcc = {arm: gemini_pilot[arm]["wsrcc"] for arm in ARM_ORDER}

    # Full-scale data points we have
    full_scale = {
        "1_single_all3": 0.708,  # Standard single prompt, 1024px
        "7_no_resize": 0.699,    # No resize, native resolution
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    # Identity line
    ax.plot([0.80, 0.96], [0.80, 0.96], "--", color="#999999", linewidth=1.0,
            label="Identity (no regression)", zorder=1)

    # Mean regression line (visual guide)
    pilot_vals = list(pilot_wsrcc.values())
    grand_mean_pilot = np.mean(pilot_vals)
    ax.axhline(y=np.mean(list(full_scale.values())), color="#CCCCCC", linestyle=":",
               linewidth=0.8, zorder=1)
    ax.axvline(x=grand_mean_pilot, color="#CCCCCC", linestyle=":",
               linewidth=0.8, zorder=1)

    # Plot validated arms (large markers)
    for arm, full_val in full_scale.items():
        pilot_val = pilot_wsrcc[arm]
        ax.scatter(pilot_val, full_val, s=120, c="#D32F2F", edgecolor="white",
                   linewidth=1.5, zorder=5)
        label = ARM_LABELS[arm].replace("\n", " ")
        offset_x = -0.015 if arm == "7_no_resize" else 0.008
        offset_y = 0.005
        ax.annotate(label, (pilot_val, full_val),
                    xytext=(pilot_val + offset_x, full_val + offset_y),
                    fontsize=8, ha="center")

    # Plot unvalidated arms (hollow markers)
    for arm in ARM_ORDER:
        if arm not in full_scale:
            pilot_val = pilot_wsrcc[arm]
            ax.scatter(pilot_val, pilot_val, s=60, c="none", edgecolor="#4285F4",
                       linewidth=1.5, zorder=4, marker="o")

    # Custom legend
    ax.scatter([], [], s=120, c="#D32F2F", edgecolor="white", linewidth=1.5,
               label="Validated at n=1000")
    ax.scatter([], [], s=60, c="none", edgecolor="#4285F4", linewidth=1.5,
               label="Pilot only (n=23, on identity)")

    # Arrow showing the regression effect
    ax.annotate(
        "",
        xy=(0.951, 0.699),
        xytext=(0.951, 0.951),
        arrowprops={"arrowstyle": "->", "color": "#D32F2F", "lw": 1.5,
                    "connectionstyle": "arc3,rad=0.1"},
    )
    ax.text(0.957, 0.825, "Regression\nto mean\n(-0.252)",
            fontsize=8, color="#D32F2F", ha="left", va="center")

    ax.set_xlabel("wSRCC at n=23 (pilot)")
    ax.set_ylabel("wSRCC at n=1000 (full scale)")
    ax.set_title("Small-Sample Regression to the Mean")
    ax.set_xlim(0.82, 0.97)
    ax.set_ylim(0.68, 0.97)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")

    save_figure(fig, FIGURES_DIR / "fig3_regression_to_mean.pdf")
    save_figure(fig, FIGURES_DIR / "fig3_regression_to_mean.png", dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all figures for Paper 3."""
    apply_arxiv_style()
    print("Generating Paper 3 figures...")
    print()

    print("Figure 1: Seven-arm prompt comparison")
    fig1_seven_arm_comparison()

    print("Figure 2: A/B test per-dimension analysis")
    fig2_ab_test_per_dimension()

    print("Figure 3: Regression-to-mean scatter")
    fig3_regression_to_mean()

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
