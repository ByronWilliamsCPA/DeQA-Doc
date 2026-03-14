"""Generate all figures for Paper 2: Cross-Domain Generalization of VLM Quality Assessors.

Produces 4 figures from raw data:
1. ID vs OOD comparison (paired bar chart)
2. Per-category heatmap (models x OOD categories, SRCC)
3. Model strengths radar chart
4. Fine-tuned vs VLM bar comparison

Usage:
    python research/papers/02_cross_domain/figures/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.constants import (
    MODEL_NAMES,
    OOD_CATEGORIES,
    PRIMARY_MODELS,
    PROJECT_ROOT,
    VLM_EVAL_DIR,
)
from shared.data_loader import load_json
from shared.plot_style import MODEL_COLORS, apply_arxiv_style, save_figure

FIGURES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

UNIFIED_PATH = VLM_EVAL_DIR / "results" / "synthetic_eval_metrics_unified.json"
FINETUNED_PATH = VLM_EVAL_DIR / "results" / "finetuned_synthetic_eval_metrics.json"
BASELINE_PATH = PROJECT_ROOT / "results" / "iqa_baselines" / "baseline_summary.json"

# DIQA-5000 wSRCC values from VLM_TEACHER_EVALUATION.md Table 1
DIQA_WSRCC: dict[str, float] = {
    "google__gemini-3-flash-preview": 0.708,
    "openai__gpt-4.1": 0.669,
    "google__gemini-2.5-pro": 0.612,
    "qwen__qwen3.5-flash-02-23": 0.593,
    "anthropic__claude-haiku-4.5": 0.579,
    "qwen__qwen3-vl-8b-instruct": 0.481,
    "qwen__qwen3-vl-8b-thinking": 0.409,
}

# Fine-tuned model DIQA-5000 wSRCC / MainScore
FINETUNED_DIQA: dict[str, float] = {
    "deqa_doc_3specialists": 0.716,
    "hyperiqa_plus_plus": 0.840,  # HyperIQA++ MainScore on DIQA
    "siglip2_iqa_base_86m": 0.886,  # SigLIP2 MainScore on DIQA
}

# Display names for fine-tuned + baseline models
EXTRA_NAMES: dict[str, str] = {
    "deqa_doc_3specialists": "DeQA-Doc-3Spec",
    "hyperiqa_plus_plus": "HyperIQA++",
    "siglip2_iqa_base_86m": "SigLIP2-IQA",
    "TReS_synthetic": "TReS",
    "HyperIQA_synthetic": "HyperIQA",
    "RichIQA_synthetic": "RichIQA",
    "DBCNN_synthetic": "DBCNN",
    "MUSIQ_synthetic": "MUSIQ",
}

# OOD category display labels (cleaner names)
OOD_LABELS: dict[str, str] = {
    "ood_heavily_degraded": "Heavily Degraded",
    "ood_adversarial_nastaliq": "Nastaliq",
    "ood_very_low_dpi": "Very Low DPI",
    "ood_multiscript": "Multiscript",
    "ood_script_tibetan": "Tibetan",
    "ood_script_ethiopic": "Ethiopic",
    "ood_form_layout": "Form Layout",
    "ood_adversarial_fraktur": "Fraktur",
    "ood_pristine": "Pristine",
    "ood_very_high_dpi": "Very High DPI",
    "ood_binarized": "Binarized",
    "ood_script_myanmar": "Myanmar",
    "ood_cjk_vertical": "CJK Vertical",
}


def _short_name(model_id: str) -> str:
    """Get short display name for a model."""
    return MODEL_NAMES.get(model_id, EXTRA_NAMES.get(model_id, model_id))


# ---------------------------------------------------------------------------
# Figure 1: ID vs OOD wSRCC comparison (paired bars)
# ---------------------------------------------------------------------------


def fig1_id_vs_ood(unified: dict) -> None:
    """Paired bar chart: each model's wSRCC on DIQA-5000 (ID) vs synthetic (OOD)."""
    # Models to include (VLMs only for clean comparison)
    models = PRIMARY_MODELS
    labels = [_short_name(m) for m in models]

    # Map from OpenRouter ID to unified JSON key
    key_map = {
        "google__gemini-3-flash-preview": "google/gemini-3-flash-preview",
        "openai__gpt-4.1": "openai/gpt-4.1",
        "google__gemini-2.5-pro": "google/gemini-2.5-pro",
        "qwen__qwen3.5-flash-02-23": "qwen/qwen3.5-flash-02-23",
        "anthropic__claude-haiku-4.5": "anthropic/claude-haiku-4.5",
        "qwen__qwen3-vl-8b-instruct": "qwen/qwen3-vl-8b-instruct",
        "qwen__qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    }

    id_vals = [DIQA_WSRCC[m] for m in models]
    ood_vals = []
    for m in models:
        ukey = key_map[m]
        ood_vals.append(unified[ukey]["ood"]["wsrcc"])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_id = ax.bar(x - width / 2, id_vals, width, label="DIQA-5000 (ID)", color="#2171B5", alpha=0.85)
    bars_ood = ax.bar(x + width / 2, ood_vals, width, label="Synthetic OOD", color="#D94701", alpha=0.85)

    ax.set_ylabel("wSRCC")
    ax.set_title("In-Distribution vs Out-of-Distribution Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.85)

    # Add value labels on bars
    for bar in bars_id:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for bar in bars_ood:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    save_figure(fig, FIGURES_DIR / "fig1_id_vs_ood.png")
    save_figure(fig, FIGURES_DIR / "fig1_id_vs_ood.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Per-category heatmap (models x OOD categories)
# ---------------------------------------------------------------------------


def fig2_category_heatmap(unified: dict) -> None:
    """Heatmap of SRCC by model and OOD category."""
    # Select top 5 models + DeQA-Doc for comparison
    model_keys = [
        ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
        ("openai/gpt-4.1", "GPT-4.1"),
        ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
        ("qwen/qwen3.5-flash-02-23", "Qwen 3.5 Flash"),
        ("deqa_doc_3specialists", "DeQA-Doc-3Spec"),
        ("hyperiqa_plus_plus", "HyperIQA++"),
        ("siglip2_iqa_base_86m", "SigLIP2-IQA"),
    ]

    categories = OOD_CATEGORIES
    cat_labels = [OOD_LABELS[c] for c in categories]

    # Build matrix
    matrix = np.full((len(model_keys), len(categories)), np.nan)
    for i, (mkey, _) in enumerate(model_keys):
        if mkey not in unified:
            continue
        per_cat = unified[mkey].get("per_category", {})
        for j, cat in enumerate(categories):
            entry = per_cat.get(cat, {})
            srcc = entry.get("srcc")
            if srcc is not None:
                matrix[i, j] = srcc

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.85)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(model_keys)))
    ax.set_yticklabels([mk[1] for mk in model_keys], fontsize=9)

    # Annotate cells
    for i in range(len(model_keys)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="gray")
            else:
                color = "white" if val < -0.1 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Per-Category Overall SRCC Across Models (Synthetic OOD)")
    fig.colorbar(im, ax=ax, label="SRCC", shrink=0.8)

    save_figure(fig, FIGURES_DIR / "fig2_category_heatmap.png")
    save_figure(fig, FIGURES_DIR / "fig2_category_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Radar chart for model strengths
# ---------------------------------------------------------------------------


def fig3_model_radar(unified: dict) -> None:
    """Radar/spider chart showing relative model strengths across OOD category groups."""
    # Group OOD categories into thematic clusters
    category_groups = {
        "Non-Latin\nScripts": ["ood_script_tibetan", "ood_script_ethiopic", "ood_script_myanmar"],
        "Adversarial\nScripts": ["ood_adversarial_fraktur", "ood_adversarial_nastaliq"],
        "Layout\nVariants": ["ood_cjk_vertical", "ood_form_layout"],
        "Multi-\nscript": ["ood_multiscript"],
        "Extreme\nDegradation": ["ood_binarized", "ood_heavily_degraded"],
        "DPI\nExtremes": ["ood_very_low_dpi", "ood_very_high_dpi"],
    }

    models_to_plot = [
        ("google/gemini-3-flash-preview", "Gemini 3 Flash", "#4285F4"),
        ("openai/gpt-4.1", "GPT-4.1", "#10A37F"),
        ("deqa_doc_3specialists", "DeQA-Doc-3Spec", "#9B59B6"),
        ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "#D4A574"),
    ]

    group_names = list(category_groups.keys())
    n_groups = len(group_names)
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})

    for mkey, mname, mcolor in models_to_plot:
        if mkey not in unified:
            continue
        per_cat = unified[mkey].get("per_category", {})
        values = []
        for group_cats in category_groups.values():
            srcc_vals = []
            for cat in group_cats:
                entry = per_cat.get(cat, {})
                srcc = entry.get("srcc")
                if srcc is not None:
                    srcc_vals.append(max(srcc, 0))  # clip negative for radar
            values.append(np.mean(srcc_vals) if srcc_vals else 0)

        values += values[:1]
        ax.plot(angles, values, "o-", label=mname, color=mcolor, linewidth=1.5, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=mcolor)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(group_names, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7)
    ax.set_title("Model Strengths by OOD Category Group (mean SRCC)", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=8)

    save_figure(fig, FIGURES_DIR / "fig3_model_radar.png")
    save_figure(fig, FIGURES_DIR / "fig3_model_radar.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Fine-tuned vs VLM comparison on synthetic data
# ---------------------------------------------------------------------------


def fig4_finetuned_vs_vlm(unified: dict, baselines: dict) -> None:
    """Grouped bar chart comparing fine-tuned models, VLMs, and NR-IQA baselines on synthetic."""
    # Collect all models with synthetic MainScore
    entries: list[tuple[str, float, str, str]] = []  # (name, mainscore, type, color)

    # VLMs from unified
    vlm_models = [
        ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
        ("openai/gpt-4.1", "GPT-4.1"),
        ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
        ("qwen/qwen3.5-flash-02-23", "Qwen 3.5 Flash"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ]
    for mkey, mname in vlm_models:
        if mkey in unified:
            entries.append((mname, unified[mkey]["all"]["mainscore"], "VLM", "#4285F4"))

    # Fine-tuned from unified
    ft_models = [
        ("deqa_doc_3specialists", "DeQA-Doc-3Spec"),
        ("hyperiqa_plus_plus", "HyperIQA++"),
        ("siglip2_iqa_base_86m", "SigLIP2-IQA"),
    ]
    for mkey, mname in ft_models:
        if mkey in unified:
            entries.append((mname, unified[mkey]["all"]["mainscore"], "Fine-tuned", "#E63946"))

    # NR-IQA baselines on synthetic
    nriqa_models = [
        ("TReS_synthetic", "TReS"),
        ("HyperIQA_synthetic", "HyperIQA"),
        ("RichIQA_synthetic", "RichIQA"),
        ("DBCNN_synthetic", "DBCNN"),
        ("MUSIQ_synthetic", "MUSIQ"),
    ]
    for bkey, bname in nriqa_models:
        if bkey in baselines:
            entries.append((bname, baselines[bkey]["metrics"]["main_score"], "NR-IQA Baseline", "#238B45"))

    # Sort by mainscore descending
    entries.sort(key=lambda x: x[1], reverse=True)

    names = [e[0] for e in entries]
    scores = [e[1] for e in entries]
    colors = [e[3] for e in entries]
    types = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.85, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Synthetic MainScore")
    ax.set_title("Fine-Tuned Models vs VLMs vs NR-IQA Baselines (Synthetic Dataset)")
    ax.set_xlim(0, 0.85)
    ax.invert_yaxis()

    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4285F4", alpha=0.85, label="VLM (zero-shot)"),
        Patch(facecolor="#E63946", alpha=0.85, label="Fine-tuned"),
        Patch(facecolor="#238B45", alpha=0.85, label="NR-IQA Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    save_figure(fig, FIGURES_DIR / "fig4_finetuned_vs_vlm.png")
    save_figure(fig, FIGURES_DIR / "fig4_finetuned_vs_vlm.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all figures for Paper 2."""
    apply_arxiv_style()

    print("Loading data...")
    unified = load_json(UNIFIED_PATH)
    baselines = load_json(BASELINE_PATH)

    print("\n--- Figure 1: ID vs OOD ---")
    fig1_id_vs_ood(unified)

    print("\n--- Figure 2: Category Heatmap ---")
    fig2_category_heatmap(unified)

    print("\n--- Figure 3: Model Radar ---")
    fig3_model_radar(unified)

    print("\n--- Figure 4: Fine-tuned vs VLM ---")
    fig4_finetuned_vs_vlm(unified, baselines)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
