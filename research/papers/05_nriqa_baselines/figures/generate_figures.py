"""Generate all figures for Paper 5: Off-the-Shelf NR-IQA Models on Document Images.

Produces:
  1. fig1_pretrained_vs_finetuned.pdf  -- Grouped bar: off-the-shelf vs fine-tuned SRCC
  2. fig2_domain_gap_bar.pdf           -- Bar: DIQA vs synthetic MainScore per model
  3. fig3_unified_ranking.pdf          -- Horizontal bar: all models ranked by wSRCC/MainScore
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Shared infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.plot_style import apply_arxiv_style, save_figure, DIM_COLORS
from shared.constants import PROJECT_ROOT, RESULTS_DIR, DIMENSIONS, DIMENSION_LABELS, WSRCC_WEIGHTS, LICENSE
from shared.data_loader import load_json

FIGURES_DIR = Path(__file__).resolve().parent

# ── Data loading ──────────────────────────────────────────────────────────────

def load_baseline_data() -> dict:
    """Load baseline_summary.json and extract per-model metrics."""
    raw = load_json(RESULTS_DIR / "iqa_baselines" / "baseline_summary.json")

    models: dict[str, dict] = {}
    for key, entry in raw.items():
        if "error" in entry:
            continue
        model = entry["model"]
        dataset = entry["dataset"]
        metrics = entry["metrics"]

        if model not in models:
            models[model] = {}

        # Compute MainScore = 0.5*(SRCC_O+PLCC_O)/2 + 0.25*(SRCC_S+PLCC_S)/2 + 0.25*(SRCC_C+PLCC_C)/2
        overall_score = (metrics["overall"]["srcc"] + metrics["overall"]["plcc"]) / 2
        sharpness_score = (metrics["sharpness"]["srcc"] + metrics["sharpness"]["plcc"]) / 2
        color_score = (metrics["color"]["srcc"] + metrics["color"]["plcc"]) / 2
        main_score = 0.5 * overall_score + 0.25 * sharpness_score + 0.25 * color_score

        models[model][dataset] = {
            "srcc_overall": metrics["overall"]["srcc"],
            "plcc_overall": metrics["overall"]["plcc"],
            "srcc_sharpness": metrics["sharpness"]["srcc"],
            "plcc_sharpness": metrics["sharpness"]["plcc"],
            "srcc_color": metrics["color"]["srcc"],
            "plcc_color": metrics["color"]["plcc"],
            "main_score": main_score,
            "n": metrics.get("n_valid", entry.get("n_images", 0)),
        }

    return models


# Reported fine-tuned scores from VQualA 2025 competition (Table in diqa_3.md)
FINETUNED_SCORES: dict[str, float] = {
    "RichIQA": 0.866,
    "TReS": 0.863,
    "MUSIQ": 0.859,
    "HyperIQA": 0.844,
    "DBCNN": 0.587,
}

# VLM zero-shot MainScore on DIQA-5000 (from diqa_3.md Table)
VLM_DIQA_SCORES: dict[str, float] = {
    "Gemini 3 Flash": 0.743,
    "GPT-4.1": 0.715,
    "Gemini 2.5 Pro": 0.655,
    "Qwen 3.5 Flash": 0.626,
    "Claude Haiku 4.5": 0.601,
    "Qwen3-VL-8B": 0.505,
    "Qwen3-VL-8B Think": 0.439,
}

# Fine-tuned specialist models on DIQA-5000 (from diqa_3.md)
FINETUNED_DIQA: dict[str, float] = {
    "SigLIP2-IQA-Base": 0.886,
    "DeQA-Doc-3Spec.": 0.716,
}


# ── Figure 1: Pretrained vs Fine-tuned ────────────────────────────────────────

def fig1_pretrained_vs_finetuned(models: dict) -> None:
    """Grouped bar comparing off-the-shelf MainScore vs reported fine-tuned score."""
    apply_arxiv_style()

    # Sort by fine-tuned score descending
    model_names = sorted(FINETUNED_SCORES.keys(), key=lambda m: FINETUNED_SCORES[m], reverse=True)

    pretrained_scores = []
    finetuned_scores = []
    for name in model_names:
        diqa = models.get(name, {}).get("diqa", {})
        pretrained_scores.append(diqa.get("main_score", 0.0))
        finetuned_scores.append(FINETUNED_SCORES[name])

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars1 = ax.bar(x - width / 2, pretrained_scores, width, label="Off-the-shelf (pretrained)", color="#6baed6", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, finetuned_scores, width, label="Fine-tuned on DIQA-5000", color="#2171b5", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("MainScore")
    ax.set_title("Off-the-Shelf vs Fine-Tuned NR-IQA on DIQA-5000")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Add improvement multiplier annotations
    for i, name in enumerate(model_names):
        pre = pretrained_scores[i]
        fine = finetuned_scores[i]
        if pre > 0:
            multiplier = fine / pre
            ax.annotate(
                f"{multiplier:.1f}x",
                xy=(x[i], max(pre, fine) + 0.04),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#d7191c",
            )

    save_figure(fig, FIGURES_DIR / "fig1_pretrained_vs_finetuned.pdf")
    save_figure(fig, FIGURES_DIR / "fig1_pretrained_vs_finetuned.png", dpi=150)


# ── Figure 2: Domain gap bar (DIQA vs synthetic) ─────────────────────────────

def fig2_domain_gap(models: dict) -> None:
    """Bar chart showing MainScore on DIQA-5000 vs synthetic for each model."""
    apply_arxiv_style()

    # Only models with both datasets
    model_names = [m for m in models if "diqa" in models[m] and "synthetic" in models[m]]
    model_names.sort(key=lambda m: models[m]["synthetic"]["main_score"], reverse=True)

    diqa_scores = [models[m]["diqa"]["main_score"] for m in model_names]
    synth_scores = [models[m]["synthetic"]["main_score"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars1 = ax.bar(x - width / 2, diqa_scores, width, label="DIQA-5000 (n=1,000)", color=DIM_COLORS["overall"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, synth_scores, width, label="Synthetic OOD (n=520)", color=DIM_COLORS["sharpness"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("MainScore")
    ax.set_title("NR-IQA Performance: DIQA-5000 vs Synthetic OOD")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right")

    # Delta annotations
    for i, name in enumerate(model_names):
        delta = synth_scores[i] - diqa_scores[i]
        color = "#2ca25f" if delta > 0 else "#d7191c"
        ax.annotate(
            f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}",
            xy=(x[i], max(diqa_scores[i], synth_scores[i]) + 0.01),
            ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold",
        )

    save_figure(fig, FIGURES_DIR / "fig2_domain_gap_bar.pdf")
    save_figure(fig, FIGURES_DIR / "fig2_domain_gap_bar.png", dpi=150)


# ── Figure 3: Unified ranking (all model types) ──────────────────────────────

def fig3_unified_ranking(models: dict) -> None:
    """Horizontal bar chart ranking all models (NR-IQA + VLM + fine-tuned) by DIQA MainScore."""
    apply_arxiv_style()

    entries: list[tuple[str, float, str]] = []

    # NR-IQA pretrained (from baseline data)
    for name in models:
        if "diqa" in models[name]:
            entries.append((name, models[name]["diqa"]["main_score"], "NR-IQA (pretrained)"))

    # VLM zero-shot
    for name, score in VLM_DIQA_SCORES.items():
        entries.append((name, score, "VLM (zero-shot)"))

    # Fine-tuned
    for name, score in FINETUNED_DIQA.items():
        entries.append((name, score, "Fine-tuned"))

    # Sort ascending (bottom to top in horizontal bar)
    entries.sort(key=lambda e: e[1])

    names = [e[0] for e in entries]
    scores = [e[1] for e in entries]
    categories = [e[2] for e in entries]

    cat_colors = {
        "NR-IQA (pretrained)": "#6baed6",
        "VLM (zero-shot)": "#fd8d3c",
        "Fine-tuned": "#2171b5",
    }
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("MainScore on DIQA-5000 (n=1,000)")
    ax.set_title("Unified DIQA-5000 Leaderboard: All Model Types")
    ax.set_xlim(0, 1.0)

    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Vertical reference lines
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    save_figure(fig, FIGURES_DIR / "fig3_unified_ranking.pdf")
    save_figure(fig, FIGURES_DIR / "fig3_unified_ranking.png", dpi=150)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all Paper 5 figures."""
    print("Paper 5: NR-IQA Baselines — Figure Generation")
    print("=" * 55)

    models = load_baseline_data()
    print(f"Loaded {len(models)} NR-IQA models from baseline_summary.json")
    for name, data in models.items():
        for ds, m in data.items():
            print(f"  {name} [{ds}]: MainScore = {m['main_score']:.3f} (n={m['n']})")

    print("\nFigure 1: Pretrained vs Fine-tuned")
    fig1_pretrained_vs_finetuned(models)

    print("\nFigure 2: Domain Gap (DIQA vs Synthetic)")
    fig2_domain_gap(models)

    print("\nFigure 3: Unified Ranking")
    fig3_unified_ranking(models)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
