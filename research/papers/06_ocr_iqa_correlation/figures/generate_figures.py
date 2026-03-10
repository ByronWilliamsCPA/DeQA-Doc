"""Generate all figures for Paper 6: OCR-IQA Correlation Study.

Produces four figures from the OCR-IQA correlation dataset:
  1. CER vs MOS scatter (4-panel, one per OCR engine)
  2. CER boxplots by quality tier, grouped by engine
  3. Engine x tier mean CER heatmap
  4. Monotonicity line plot (mean CER per tier per engine)

Usage:
    python research/papers/06_ocr_iqa_correlation/figures/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # research/papers/
from shared.plot_style import (
    apply_arxiv_style,
    save_figure,
    TIER_COLORS,
    ENGINE_COLORS,
)
from shared.constants import (
    OCR_IQA_DIR,
    OCR_ENGINES,
    OCR_ENGINE_LABELS,
    QUALITY_TIERS,
)
from shared.data_loader import load_jsonl, load_json

FIGURES_DIR = Path(__file__).resolve().parent
DATASET_PATH = OCR_IQA_DIR / "data" / "dataset.jsonl"
REPORT_PATH = OCR_IQA_DIR / "outputs" / "correlation_report.json"


def load_data() -> tuple[list[dict], dict]:
    """Load dataset records and correlation report."""
    records = load_jsonl(DATASET_PATH)
    report = load_json(REPORT_PATH)
    return records, report


def extract_arrays(
    records: list[dict],
) -> dict[str, dict[str, np.ndarray]]:
    """Extract per-engine arrays of MOS, CER, and tier labels.

    Returns:
        Dict mapping engine -> {mos, cer, tier} arrays.
    """
    data: dict[str, dict[str, list]] = {
        engine: {"mos": [], "cer": [], "tier": []}
        for engine in OCR_ENGINES
    }
    for rec in records:
        mos = rec.get("deqa_mos")
        tier = rec.get("tier")
        ocr = rec.get("ocr", {})
        if mos is None or tier is None:
            continue
        for engine in OCR_ENGINES:
            eng_data = ocr.get(engine, {})
            cer = eng_data.get("cer")
            if cer is not None:
                data[engine]["mos"].append(mos)
                data[engine]["cer"].append(cer)
                data[engine]["tier"].append(tier)

    return {
        engine: {
            "mos": np.array(vals["mos"]),
            "cer": np.array(vals["cer"]),
            "tier": np.array(vals["tier"]),
        }
        for engine, vals in data.items()
    }


def fig1_cer_vs_mos_scatter(
    engine_data: dict[str, dict[str, np.ndarray]],
    report: dict,
) -> None:
    """Figure 1: 4-panel CER vs MOS scatter, colored by quality tier."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, engine in enumerate(OCR_ENGINES):
        ax = axes_flat[idx]
        d = engine_data[engine]

        for tier in QUALITY_TIERS:
            mask = d["tier"] == tier
            if not mask.any():
                continue
            ax.scatter(
                d["mos"][mask],
                d["cer"][mask],
                c=TIER_COLORS[tier],
                label=tier.capitalize(),
                alpha=0.45,
                s=12,
                edgecolors="none",
            )

        # Annotate with SRCC
        corr = report["engine_correlations"][engine]
        srcc = corr["srcc"]
        ax.text(
            0.97, 0.97,
            f"SRCC = {srcc:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("DeQA MOS")
        ax.set_ylabel("CER")
        ax.set_title(OCR_ENGINE_LABELS[engine])
        ax.set_ylim(-0.05, 1.15)

    # Single shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("CER vs. DeQA MOS by OCR Engine", fontsize=13, y=1.02)
    save_figure(fig, FIGURES_DIR / "fig1_cer_vs_mos_scatter.png")


def fig2_cer_boxplots(
    engine_data: dict[str, dict[str, np.ndarray]],
) -> None:
    """Figure 2: CER boxplots by quality tier, grouped by engine."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)

    for idx, engine in enumerate(OCR_ENGINES):
        ax = axes[idx]
        d = engine_data[engine]
        tier_data = []
        tier_labels = []
        colors = []

        for tier in QUALITY_TIERS:
            mask = d["tier"] == tier
            if mask.any():
                tier_data.append(d["cer"][mask])
                tier_labels.append(tier[:4].capitalize())
                colors.append(TIER_COLORS[tier])

        bp = ax.boxplot(
            tier_data,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(tier_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(OCR_ENGINE_LABELS[engine])
        if idx == 0:
            ax.set_ylabel("CER")

    fig.suptitle("CER Distribution by Quality Tier", fontsize=13)
    save_figure(fig, FIGURES_DIR / "fig2_cer_boxplots_by_tier.png")


def fig3_engine_tier_heatmap(report: dict) -> None:
    """Figure 3: Mean CER heatmap (engine x tier)."""
    tier_stats = report["tier_stats"]

    matrix = np.zeros((len(OCR_ENGINES), len(QUALITY_TIERS)))
    for i, engine in enumerate(OCR_ENGINES):
        for j, tier in enumerate(QUALITY_TIERS):
            matrix[i, j] = tier_stats[tier][engine]["mean_cer"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.2, vmax=0.9)

    ax.set_xticks(range(len(QUALITY_TIERS)))
    ax.set_xticklabels([t.capitalize() for t in QUALITY_TIERS], fontsize=9)
    ax.set_yticks(range(len(OCR_ENGINES)))
    ax.set_yticklabels([OCR_ENGINE_LABELS[e] for e in OCR_ENGINES], fontsize=10)

    # Annotate cells
    for i in range(len(OCR_ENGINES)):
        for j in range(len(QUALITY_TIERS)):
            val = matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean CER")
    ax.set_title("Mean CER by Engine and Quality Tier", fontsize=13)
    save_figure(fig, FIGURES_DIR / "fig3_engine_tier_heatmap.png")


def fig4_monotonicity_line(report: dict) -> None:
    """Figure 4: Mean CER per tier for each engine (monotonicity check)."""
    tier_stats = report["tier_stats"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(QUALITY_TIERS))

    for engine in OCR_ENGINES:
        means = [tier_stats[tier][engine]["mean_cer"] for tier in QUALITY_TIERS]
        stds = [tier_stats[tier][engine]["std_cer"] for tier in QUALITY_TIERS]
        ax.errorbar(
            x, means,
            yerr=np.array(stds) / np.sqrt(200),  # standard error
            label=OCR_ENGINE_LABELS[engine],
            color=ENGINE_COLORS[engine],
            marker="o",
            markersize=6,
            capsize=3,
            linewidth=1.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in QUALITY_TIERS], fontsize=9)
    ax.set_xlabel("Quality Tier")
    ax.set_ylabel("Mean CER")
    ax.set_title("Mean CER Across Quality Tiers by Engine", fontsize=13)
    ax.legend(loc="upper left", framealpha=0.9)

    # Add MOS on secondary axis
    ax2 = ax.twinx()
    mos_means = [tier_stats[tier][OCR_ENGINES[0]]["mean_mos"] for tier in QUALITY_TIERS]
    ax2.plot(
        x, mos_means,
        color="gray", linestyle="--", marker="s",
        markersize=5, alpha=0.6, label="DeQA MOS",
    )
    ax2.set_ylabel("DeQA MOS", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(loc="upper right", framealpha=0.9)

    save_figure(fig, FIGURES_DIR / "fig4_monotonicity_line.png")


def main() -> None:
    """Generate all figures for Paper 6."""
    apply_arxiv_style()
    print("Loading data...")
    records, report = load_data()
    engine_data = extract_arrays(records)
    print(f"  Loaded {len(records)} records across {len(engine_data)} engines")

    print("\nGenerating figures...")
    fig1_cer_vs_mos_scatter(engine_data, report)
    fig2_cer_boxplots(engine_data)
    fig3_engine_tier_heatmap(report)
    fig4_monotonicity_line(report)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
