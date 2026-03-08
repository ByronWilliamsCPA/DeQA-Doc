"""Visualization for OCR-IQA correlation analysis.

Generates scatter plots, box plots, and heatmaps for the correlation
study results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.ocr_iqa_correlation.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Tier color scheme
TIER_COLORS = {
    "ORIGINAL": "#2ca02c",
    "PRISTINE": "#1f77b4",
    "HIGH": "#ff7f0e",
    "MEDIUM": "#d62728",
    "LOW": "#9467bd",
    "DEGRADED": "#8c564b",
}

TIER_ORDER = ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]


def plot_cer_vs_mos_scatter(
    dataset_records: list[dict],
    engine: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Scatter plot of CER vs MOS for a single engine, colored by tier.

    Args:
        dataset_records: Master dataset records.
        engine: Engine name to plot.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    for tier in TIER_ORDER:
        tier_records = [r for r in dataset_records if r.get("tier") == tier]
        cers = []
        moss = []
        for r in tier_records:
            ocr = r.get("ocr", {}).get(engine, {})
            mos = r.get("deqa_mos")
            cer = ocr.get("cer")
            if cer is not None and mos is not None:
                cers.append(cer)
                moss.append(mos)

        if cers:
            ax.scatter(
                moss,
                cers,
                c=TIER_COLORS.get(tier, "#333333"),
                label=tier,
                alpha=0.6,
                s=30,
            )

    ax.set_xlabel("DeQA MOS Score")
    ax.set_ylabel("Character Error Rate (CER)")
    ax.set_title(f"CER vs MOS — {engine}")
    ax.legend(title="Quality Tier")
    ax.grid(alpha=0.3)

    output_path = output_dir / f"cer_vs_mos_{engine}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved scatter plot: %s", output_path)
    return output_path


def plot_cer_boxplots(
    dataset_records: list[dict],
    engines: list[str],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Box plots of CER distribution per tier, one subplot per engine.

    Args:
        dataset_records: Master dataset records.
        engines: List of engine names.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_engines = len(engines)
    fig, axes = plt.subplots(1, n_engines, figsize=(5 * n_engines, 6), sharey=True)
    if n_engines == 1:
        axes = [axes]

    for ax, engine in zip(axes, engines):
        data_by_tier = []
        tier_labels = []

        for tier in TIER_ORDER:
            cers = []
            for r in dataset_records:
                if r.get("tier") != tier:
                    continue
                ocr = r.get("ocr", {}).get(engine, {})
                cer = ocr.get("cer")
                if cer is not None:
                    cers.append(cer)

            if cers:
                data_by_tier.append(cers)
                tier_labels.append(tier)

        if data_by_tier:
            bp = ax.boxplot(
                data_by_tier,
                labels=tier_labels,
                patch_artist=True,
            )
            for patch, tier in zip(bp["boxes"], tier_labels):
                patch.set_facecolor(TIER_COLORS.get(tier, "#cccccc"))
                patch.set_alpha(0.7)

        ax.set_title(engine)
        ax.set_xlabel("Quality Tier")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Character Error Rate (CER)")
    fig.suptitle("CER Distribution by Quality Tier", y=1.02)
    fig.tight_layout()

    output_path = output_dir / "cer_boxplots_by_tier.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved box plots: %s", output_path)
    return output_path


def plot_engine_tier_heatmap(
    tier_stats: dict[str, dict[str, dict[str, float]]],
    engines: list[str],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Heatmap of mean CER by engine x tier.

    Args:
        tier_stats: Output from compute_per_tier_stats().
        engines: List of engine names.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = [t for t in TIER_ORDER if t in tier_stats]
    matrix = np.zeros((len(engines), len(tiers)))

    for i, engine in enumerate(engines):
        for j, tier in enumerate(tiers):
            stats = tier_stats.get(tier, {}).get(engine, {})
            matrix[i, j] = stats.get("mean_cer", 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers, rotation=45)
    ax.set_yticks(range(len(engines)))
    ax.set_yticklabels(engines)

    # Add value annotations
    for i in range(len(engines)):
        for j in range(len(tiers)):
            ax.text(
                j, i, f"{matrix[i, j]:.3f}",
                ha="center", va="center", fontsize=9,
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
            )

    fig.colorbar(im, label="Mean CER")
    ax.set_title("Mean CER: Engine × Quality Tier")
    fig.tight_layout()

    output_path = output_dir / "engine_tier_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved heatmap: %s", output_path)
    return output_path


def plot_paired_deltas(
    deltas: dict[str, list[dict[str, float]]],
    engines: list[str],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Scatter plot of delta CER vs delta MOS for paired analysis.

    Args:
        deltas: Output from compute_paired_deltas().
        engines: List of engine names.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_engines = len(engines)
    fig, axes = plt.subplots(1, n_engines, figsize=(5 * n_engines, 5))
    if n_engines == 1:
        axes = [axes]

    for ax, engine in zip(axes, engines):
        engine_deltas = deltas.get(engine, [])
        if not engine_deltas:
            ax.set_title(f"{engine} (no data)")
            continue

        for tier in TIER_ORDER:
            tier_deltas = [d for d in engine_deltas if d.get("tier") == tier]
            if tier_deltas:
                d_cer = [d["delta_cer"] for d in tier_deltas]
                d_mos = [d["delta_mos"] for d in tier_deltas]
                ax.scatter(
                    d_mos, d_cer,
                    c=TIER_COLORS.get(tier, "#333333"),
                    label=tier, alpha=0.5, s=20,
                )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("ΔMOS (distorted − original)")
        ax.set_ylabel("ΔCER (distorted − original)")
        ax.set_title(engine)
        ax.legend(fontsize=7)

    fig.suptitle("Paired Analysis: ΔCER vs ΔMOS", y=1.02)
    fig.tight_layout()

    output_path = output_dir / "paired_delta_scatter.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved paired delta scatter: %s", output_path)
    return output_path
