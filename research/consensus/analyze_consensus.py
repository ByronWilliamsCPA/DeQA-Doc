"""Multi-model consensus scoring analysis with calibration.

Compares single-model baselines against ensemble configurations on DIQA-5000
test (ID) and synthetic OOD splits. Includes per-model calibration to correct
systematic overscoring bias.

Addresses peer review items in Papers 1, 2, and 7.

Usage:
    cd DeQA-Score
    .venv/bin/python ../research/consensus/analyze_consensus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.papers.shared.constants import (
    DIMENSIONS,
    MODEL_NAMES,
    PRIMARY_MODELS,
    VLM_EVAL_DIR,
    WSRCC_WEIGHTS,
)
from research.papers.shared.data_loader import (
    compute_metrics,
    extract_scores,
    load_ground_truth,
    load_vlm_checkpoints,
    merge_predictions_with_gt,
)
from research.papers.shared.plot_style import (
    MODEL_COLORS,
    apply_arxiv_style,
    save_figure,
)

OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
SYNTHETIC_META = Path("/tmp/ood_poc_test/metadata.jsonl")

# Named ensemble configurations: (label, model_ids)
PAIRWISE_ENSEMBLES: list[tuple[str, list[str]]] = [
    ("Gemini3Flash + GPT-4.1", [
        "google__gemini-3-flash-preview", "openai__gpt-4.1",
    ]),
    ("Gemini3Flash + Gemini2.5Pro", [
        "google__gemini-3-flash-preview", "google__gemini-2.5-pro",
    ]),
    ("GPT-4.1 + Gemini2.5Pro", [
        "openai__gpt-4.1", "google__gemini-2.5-pro",
    ]),
]

N_FOLDS = 5
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_image_scores(
    checkpoints: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build {model_id: {image: {dim: score}}} from raw checkpoints.

    Filters records with errors or missing scores.
    """
    result: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, records in checkpoints.items():
        model_scores: dict[str, dict[str, float]] = {}
        for rec in records:
            if rec.get("error", ""):
                continue
            scores = extract_scores(rec)
            if any(v is None for v in scores.values()):
                continue
            image = Path(rec["image"]).name
            model_scores[image] = {d: float(scores[d]) for d in DIMENSIONS}  # type: ignore[arg-type]
        result[model_id] = model_scores
    return result


def load_ood_ground_truth() -> dict[str, dict[str, float]] | None:
    """Load synthetic OOD ground truth from metadata.jsonl.

    Returns:
        Dict keyed by image basename with {overall, sharpness, color_fidelity}
        scores, or None if the metadata file is missing.
    """
    if not SYNTHETIC_META.exists():
        print(f"  WARNING: OOD metadata not found at {SYNTHETIC_META}, skipping OOD")
        return None

    gt: dict[str, dict[str, float]] = {}
    with SYNTHETIC_META.open() as f:
        for line in f:
            d = json.loads(line)
            scores = d["synthetic_scores"]
            image_key = Path(d["image_path"]).name
            gt[image_key] = {
                "overall": scores["overall"],
                "sharpness": scores["sharpness"],
                "color_fidelity": scores["color"],
            }
    return gt


def common_images(
    image_scores: dict[str, dict[str, dict[str, float]]],
    model_ids: list[str],
) -> list[str]:
    """Find images present in all specified models."""
    sets = [set(image_scores[m].keys()) for m in model_ids if m in image_scores]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def compute_bias(
    image_scores: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict[str, float]],
    model_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-model, per-dimension mean bias on matched images.

    Returns:
        {model_id: {dim: mean_bias}} where bias = pred - gt.
    """
    bias_params: dict[str, dict[str, float]] = {}
    for model_id in model_ids:
        model_bias: dict[str, float] = {}
        preds = image_scores.get(model_id, {})
        for dim in DIMENSIONS:
            diffs = []
            for img, scores in preds.items():
                if img in gt:
                    diffs.append(scores[dim] - gt[img][dim])
            model_bias[dim] = float(np.mean(diffs)) if diffs else 0.0
        bias_params[model_id] = model_bias
    return bias_params


def apply_bias_calibration(
    image_scores: dict[str, dict[str, dict[str, float]]],
    bias_params: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Subtract per-model per-dimension bias from all predictions."""
    calibrated: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, preds in image_scores.items():
        bias = bias_params.get(model_id, {d: 0.0 for d in DIMENSIONS})
        calibrated[model_id] = {
            img: {d: s - bias[d] for d, s in scores.items()}
            for img, scores in preds.items()
        }
    return calibrated


def cv_linear_calibration(
    image_scores: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict[str, float]],
    model_ids: list[str],
    n_folds: int = N_FOLDS,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float]]]]:
    """5-fold cross-validated linear calibration on ID test set.

    Returns:
        (calibrated_image_scores, linear_params) where linear_params has
        {model_id: {dim: {"a": slope, "b": intercept}}} fitted on full data
        (for applying to OOD).
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Deep copy: each inner dict must be independent to avoid mutating originals
    calibrated = {
        m: {img: dict(scores) for img, scores in image_scores[m].items()}
        for m in model_ids if m in image_scores
    }
    full_params: dict[str, dict[str, dict[str, float]]] = {}

    for model_id in model_ids:
        if model_id not in image_scores:
            continue
        preds = image_scores[model_id]
        matched_images = sorted(img for img in preds if img in gt)
        n = len(matched_images)
        if n < n_folds * 2:
            continue

        indices = rng.permutation(n)
        folds = np.array_split(indices, n_folds)
        model_params: dict[str, dict[str, float]] = {}

        for dim in DIMENSIONS:
            pred_arr = np.array([preds[matched_images[i]][dim] for i in range(n)])
            gt_arr = np.array([gt[matched_images[i]][dim] for i in range(n)])
            calibrated_arr = np.empty(n)

            for fold_idx in range(n_folds):
                test_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
                reg = LinearRegression()
                reg.fit(pred_arr[train_idx].reshape(-1, 1), gt_arr[train_idx])
                calibrated_arr[test_idx] = reg.predict(pred_arr[test_idx].reshape(-1, 1))

            for i, img in enumerate(matched_images):
                calibrated[model_id][img][dim] = float(calibrated_arr[i])

            # Fit on full data for OOD application
            reg_full = LinearRegression()
            reg_full.fit(pred_arr.reshape(-1, 1), gt_arr)
            model_params[dim] = {
                "a": float(reg_full.coef_[0]),
                "b": float(reg_full.intercept_),
            }

        full_params[model_id] = model_params

    return calibrated, full_params


def apply_linear_calibration_ood(
    image_scores: dict[str, dict[str, dict[str, float]]],
    linear_params: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Apply ID-fitted linear calibration to OOD predictions."""
    calibrated: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, preds in image_scores.items():
        params = linear_params.get(model_id)
        if params is None:
            calibrated[model_id] = dict(preds)
            continue
        cal_preds: dict[str, dict[str, float]] = {}
        for img, scores in preds.items():
            cal_preds[img] = {
                d: float(params[d]["a"] * scores[d] + params[d]["b"])
                for d in DIMENSIONS
            }
        calibrated[model_id] = cal_preds
    return calibrated


# ---------------------------------------------------------------------------
# Ensemble computation
# ---------------------------------------------------------------------------


def ensemble_predictions(
    image_scores: dict[str, dict[str, dict[str, float]]],
    model_ids: list[str],
    method: str = "mean",
) -> list[dict[str, Any]]:
    """Average (or median) predictions across models for common images.

    Returns list of prediction records compatible with merge_predictions_with_gt.
    """
    images = common_images(image_scores, model_ids)
    results: list[dict[str, Any]] = []
    for img in images:
        scores_per_model = [image_scores[m][img] for m in model_ids if m in image_scores]
        record: dict[str, Any] = {"image": img}
        for dim in DIMENSIONS:
            vals = [s[dim] for s in scores_per_model]
            if method == "median":
                record[dim] = float(np.median(vals))
            else:
                record[dim] = float(np.mean(vals))
        results.append(record)
    return results


def single_model_predictions(
    image_scores: dict[str, dict[str, dict[str, float]]],
    model_id: str,
) -> list[dict[str, Any]]:
    """Convert single-model scores to prediction records."""
    results: list[dict[str, Any]] = []
    for img, scores in image_scores[model_id].items():
        record: dict[str, Any] = {"image": img}
        for dim in DIMENSIONS:
            record[dim] = scores[dim]
        results.append(record)
    return results


def weighted_ensemble_predictions(
    image_scores: dict[str, dict[str, dict[str, float]]],
    model_ids: list[str],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Weighted average predictions across models."""
    images = common_images(image_scores, model_ids)
    total_w = sum(weights[m] for m in model_ids)
    results: list[dict[str, Any]] = []
    for img in images:
        record: dict[str, Any] = {"image": img}
        for dim in DIMENSIONS:
            val = sum(weights[m] * image_scores[m][img][dim] for m in model_ids) / total_w
            record[dim] = float(val)
        results.append(record)
    return results


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_config(
    predictions: list[dict[str, Any]],
    gt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Merge predictions with GT and compute metrics."""
    merged = merge_predictions_with_gt(predictions, gt)
    if len(merged) < 10:
        return {"n": len(merged), "wsrcc": float("nan"), "error": "too few samples"}
    return compute_metrics(merged)


def run_all_configs(
    image_scores: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict[str, float]],
    available_models: list[str],
    top3_override: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Run all ensemble configurations and return metrics.

    Args:
        top3_override: If provided, use these model IDs for top-3 ensembles
            instead of computing from single-model results (for cross-split
            consistency).

    Returns:
        (config_results, top3_ids) where top3_ids can be passed as
        top3_override to other splits.
    """
    results: dict[str, dict[str, Any]] = {}

    # 1. Single-model baselines
    for model_id in available_models:
        if model_id not in image_scores:
            continue
        label = MODEL_NAMES.get(model_id, model_id)
        preds = single_model_predictions(image_scores, model_id)
        results[label] = evaluate_config(preds, gt)

    # 2. Pairwise mean ensembles
    for label, model_ids in PAIRWISE_ENSEMBLES:
        if all(m in image_scores for m in model_ids):
            preds = ensemble_predictions(image_scores, model_ids, "mean")
            results[f"Pair: {label}"] = evaluate_config(preds, gt)

    # 3. Top-3 by wSRCC (mean + median)
    if top3_override:
        top3_ids = [m for m in top3_override if m in image_scores]
    else:
        single_wsrcc = []
        for model_id in available_models:
            if model_id not in image_scores:
                continue
            label = MODEL_NAMES.get(model_id, model_id)
            if label in results and "wsrcc" in results[label]:
                single_wsrcc.append((model_id, results[label]["wsrcc"]))
        single_wsrcc.sort(key=lambda x: x[1], reverse=True)
        top3_ids = [m for m, _ in single_wsrcc[:3]]

    if len(top3_ids) >= 3:
        top3_names = [MODEL_NAMES.get(m, m) for m in top3_ids]
        for method in ("mean", "median"):
            preds = ensemble_predictions(image_scores, top3_ids, method)
            label = f"Top-3 {method} ({', '.join(top3_names)})"
            results[label] = evaluate_config(preds, gt)

    # 4. All-model mean + median
    all_ids = [m for m in available_models if m in image_scores]
    if len(all_ids) >= 2:
        for method in ("mean", "median"):
            preds = ensemble_predictions(image_scores, all_ids, method)
            results[f"All-{len(all_ids)} {method}"] = evaluate_config(preds, gt)

    return results, top3_ids


def run_weighted_configs(
    image_scores: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict[str, float]],
    available_models: list[str],
    single_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run weighted ensemble configurations.

    Args:
        single_results: Single-model metrics to derive weights from.
    """
    results: dict[str, dict[str, Any]] = {}
    all_ids = [m for m in available_models if m in image_scores]
    if len(all_ids) < 2:
        return results

    # Inverse-MAE weighting
    inv_mae_weights: dict[str, float] = {}
    wsrcc_weights_map: dict[str, float] = {}
    for model_id in all_ids:
        label = MODEL_NAMES.get(model_id, model_id)
        if label not in single_results:
            continue
        metrics = single_results[label]
        maes = [metrics.get(d, {}).get("mae", 1.0) for d in DIMENSIONS]
        avg_mae = float(np.mean(maes))
        inv_mae_weights[model_id] = 1.0 / max(avg_mae, 0.01)
        wsrcc_weights_map[model_id] = max(metrics.get("wsrcc", 0.0), 0.01)

    if inv_mae_weights:
        preds = weighted_ensemble_predictions(image_scores, all_ids, inv_mae_weights)
        results[f"All-{len(all_ids)} inv-MAE weighted"] = evaluate_config(preds, gt)

    if wsrcc_weights_map:
        preds = weighted_ensemble_predictions(image_scores, all_ids, wsrcc_weights_map)
        results[f"All-{len(all_ids)} wSRCC weighted"] = evaluate_config(preds, gt)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_wsrcc_comparison(
    raw_id: dict[str, dict[str, Any]],
    raw_ood: dict[str, dict[str, Any]] | None,
    cal_id: dict[str, dict[str, Any]],
    cal_ood: dict[str, dict[str, Any]] | None,
) -> None:
    """Bar chart comparing wSRCC across configurations."""
    import matplotlib.pyplot as plt

    apply_arxiv_style()

    # Use configs from raw_id as canonical list, sorted by wSRCC
    configs = sorted(raw_id.keys(), key=lambda c: raw_id[c].get("wsrcc", 0), reverse=True)
    x = np.arange(len(configs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    raw_id_vals = [raw_id[c].get("wsrcc", 0) for c in configs]
    cal_id_vals = [cal_id.get(c, {}).get("wsrcc", 0) for c in configs]
    ax.bar(x - 1.5 * width, raw_id_vals, width, label="Raw (ID)", color="#4285F4", alpha=0.8)
    ax.bar(x - 0.5 * width, cal_id_vals, width, label="Calibrated (ID)", color="#0F9D58", alpha=0.8)

    if raw_ood and cal_ood:
        raw_ood_vals = [raw_ood.get(c, {}).get("wsrcc", 0) for c in configs]
        cal_ood_vals = [cal_ood.get(c, {}).get("wsrcc", 0) for c in configs]
        ax.bar(x + 0.5 * width, raw_ood_vals, width, label="Raw (OOD)", color="#4285F4", alpha=0.4)
        ax.bar(x + 1.5 * width, cal_ood_vals, width, label="Calibrated (OOD)", color="#0F9D58", alpha=0.4)

    ax.set_ylabel("wSRCC")
    ax.set_title("Consensus Scoring: wSRCC Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="lower left")
    ax.set_ylim(bottom=0)

    save_figure(fig, FIGURES_DIR / "consensus_wsrcc_comparison.png")


def plot_improvement_heatmap(
    image_scores: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict[str, float]],
    single_results: dict[str, dict[str, Any]],
) -> None:
    """Heatmap of ALL pairwise ensemble wSRCC gain over best component."""
    import matplotlib.pyplot as plt

    apply_arxiv_style()

    available = [m for m in PRIMARY_MODELS if m in image_scores]
    model_labels = [MODEL_NAMES.get(m, m) for m in available]
    n = len(available)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = available[i], available[j]
            l1 = MODEL_NAMES.get(m1, m1)
            l2 = MODEL_NAMES.get(m2, m2)
            preds = ensemble_predictions(image_scores, [m1, m2], "mean")
            metrics = evaluate_config(preds, gt)
            pair_wsrcc = metrics.get("wsrcc", 0)
            best_single = max(
                single_results.get(l1, {}).get("wsrcc", 0),
                single_results.get(l2, {}).get("wsrcc", 0),
            )
            delta = pair_wsrcc - best_single
            matrix[i, j] = delta
            matrix[j, i] = delta

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isnan(matrix)
    masked = np.ma.array(matrix, mask=mask)

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.05)
    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title("Pairwise Ensemble wSRCC Gain over Best Component")

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Δ wSRCC")
    save_figure(fig, FIGURES_DIR / "consensus_improvement_heatmap.png")


def plot_calibration_impact(
    raw_results: dict[str, dict[str, Any]],
    bias_results: dict[str, dict[str, Any]],
    cv_results: dict[str, dict[str, Any]],
) -> None:
    """Per-model MAE before/after calibration, per dimension."""
    import matplotlib.pyplot as plt

    apply_arxiv_style()

    model_labels = []
    for m in PRIMARY_MODELS:
        label = MODEL_NAMES.get(m, m)
        if label in raw_results:
            model_labels.append(label)

    if not model_labels:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, dim in zip(axes, DIMENSIONS):
        x = np.arange(len(model_labels))
        width = 0.25

        raw_mae = [raw_results[m].get(dim, {}).get("mae", 0) for m in model_labels]
        bias_mae = [bias_results.get(m, {}).get(dim, {}).get("mae", 0) for m in model_labels]
        cv_mae = [cv_results.get(m, {}).get(dim, {}).get("mae", 0) for m in model_labels]

        ax.bar(x - width, raw_mae, width, label="Raw", color="#E63946", alpha=0.8)
        ax.bar(x, bias_mae, width, label="Bias-corrected", color="#457B9D", alpha=0.8)
        ax.bar(x + width, cv_mae, width, label="CV-linear", color="#2A9D8F", alpha=0.8)

        ax.set_title(dim.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("MAE" if dim == DIMENSIONS[0] else "")

    axes[0].legend(fontsize=8)
    fig.suptitle("Calibration Impact on MAE (ID Test Set)", fontsize=13)
    save_figure(fig, FIGURES_DIR / "calibration_impact.png")


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def write_results_json(all_results: dict[str, Any]) -> None:
    """Write full results to JSON."""
    path = OUTPUT_DIR / "consensus_results.json"
    with path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {path}")


def write_results_md(
    raw_id: dict[str, dict[str, Any]],
    raw_ood: dict[str, dict[str, Any]] | None,
    cal_id: dict[str, dict[str, Any]],
    cal_ood: dict[str, dict[str, Any]] | None,
    bias_params: dict[str, dict[str, float]],
) -> None:
    """Write summary markdown with ranked tables."""
    path = OUTPUT_DIR / "RESULTS.md"
    lines: list[str] = []
    lines.append("# Multi-Model Consensus Scoring Results\n")

    def _table(
        results: dict[str, dict[str, Any]],
        ood_results: dict[str, dict[str, Any]] | None,
        title: str,
    ) -> None:
        lines.append(f"\n## {title}\n")
        configs = sorted(results.keys(), key=lambda c: results[c].get("wsrcc", 0), reverse=True)
        header = "| Rank | Configuration | n | wSRCC | SRCC_O | SRCC_S | SRCC_C | MAE_O | MAE_S | MAE_C |"
        if ood_results:
            header += " wSRCC_OOD |"
        lines.append(header)
        sep = "|------|---------------|---|-------|--------|--------|--------|-------|-------|-------|"
        if ood_results:
            sep += "-----------|"
        lines.append(sep)

        best_wsrcc = max(results[c].get("wsrcc", 0) for c in configs)
        for rank, cfg in enumerate(configs, 1):
            m = results[cfg]
            marker = " **" if m.get("wsrcc", 0) == best_wsrcc else ""
            end_marker = "**" if marker else ""
            row = (
                f"| {rank} | {marker}{cfg}{end_marker} | {m.get('n', '?')} "
                f"| {m.get('wsrcc', 0):.4f} "
            )
            for dim in DIMENSIONS:
                dm = m.get(dim, {})
                row += f"| {dm.get('srcc', 0):.4f} "
            for dim in DIMENSIONS:
                dm = m.get(dim, {})
                row += f"| {dm.get('mae', 0):.4f} "
            row += "|"
            if ood_results:
                ood_m = ood_results.get(cfg, {})
                row += f" {ood_m.get('wsrcc', 0):.4f} |"
            lines.append(row)

    _table(raw_id, raw_ood, "Raw Predictions (ID Test Set)")
    _table(cal_id, cal_ood, "Calibrated Predictions — 5-Fold CV Linear (ID Test Set)")

    # Calibration impact summary
    lines.append("\n## Calibration Impact (Mean Bias per Model)\n")
    lines.append("| Model | Bias_O | Bias_S | Bias_C |")
    lines.append("|-------|--------|--------|--------|")
    for model_id in PRIMARY_MODELS:
        label = MODEL_NAMES.get(model_id, model_id)
        bias = bias_params.get(model_id, {})
        row = f"| {label} "
        for dim in DIMENSIONS:
            row += f"| {bias.get(dim, 0):+.3f} "
        row += "|"
        lines.append(row)

    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def print_recommendation(
    raw_id: dict[str, dict[str, Any]],
    cal_id: dict[str, dict[str, Any]],
) -> None:
    """Print final recommendation to stdout."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR PAPER 7")
    print("=" * 70)

    # Best raw single model
    single_models = [MODEL_NAMES.get(m, m) for m in PRIMARY_MODELS]
    raw_singles = {k: v for k, v in raw_id.items() if k in single_models}
    raw_ensembles = {k: v for k, v in raw_id.items() if k not in single_models}

    if raw_singles:
        best_single = max(raw_singles, key=lambda c: raw_singles[c].get("wsrcc", 0))
        best_single_wsrcc = raw_singles[best_single]["wsrcc"]
        print(f"\nBest single model (raw): {best_single} — wSRCC = {best_single_wsrcc:.4f}")

    if raw_ensembles:
        best_ensemble = max(raw_ensembles, key=lambda c: raw_ensembles[c].get("wsrcc", 0))
        best_ens_wsrcc = raw_ensembles[best_ensemble]["wsrcc"]
        print(f"Best ensemble (raw):     {best_ensemble} — wSRCC = {best_ens_wsrcc:.4f}")
        if raw_singles:
            delta = best_ens_wsrcc - best_single_wsrcc
            print(f"Ensemble advantage:      {delta:+.4f}")

    # Best calibrated
    cal_singles = {k: v for k, v in cal_id.items() if k in single_models}
    cal_ensembles = {k: v for k, v in cal_id.items() if k not in single_models}

    if cal_singles:
        best_cal_single = max(cal_singles, key=lambda c: cal_singles[c].get("wsrcc", 0))
        print(f"\nBest single (calibrated): {best_cal_single} — wSRCC = {cal_singles[best_cal_single]['wsrcc']:.4f}")

    if cal_ensembles:
        best_cal_ens = max(cal_ensembles, key=lambda c: cal_ensembles[c].get("wsrcc", 0))
        print(f"Best ensemble (calibrated): {best_cal_ens} — wSRCC = {cal_ensembles[best_cal_ens]['wsrcc']:.4f}")

    # Overall best
    best_overall = max(cal_id, key=lambda c: cal_id[c].get("wsrcc", 0))
    print(f"\nOverall best config: {best_overall} — wSRCC = {cal_id[best_overall]['wsrcc']:.4f}")

    # CI overlap check
    best_metrics = cal_id[best_overall]
    print("\nPer-dimension breakdown (calibrated best):")
    for dim in DIMENSIONS:
        dm = best_metrics.get(dim, {})
        ci = dm.get("srcc_ci", [0, 0])
        print(f"  {dim:18s}: SRCC={dm.get('srcc', 0):.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  MAE={dm.get('mae', 0):.4f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full consensus scoring analysis."""
    print("=" * 60)
    print("Multi-Model Consensus Scoring Analysis")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    id_gt = load_ground_truth()
    print(f"  ID ground truth: {len(id_gt)} images")

    id_checkpoints = load_vlm_checkpoints(PRIMARY_MODELS, split="real")
    id_scores = build_image_scores(id_checkpoints)
    for m, s in id_scores.items():
        print(f"  {MODEL_NAMES.get(m, m):30s}: {len(s)} images")

    ood_gt = load_ood_ground_truth()
    ood_scores: dict[str, dict[str, dict[str, float]]] | None = None
    if ood_gt:
        ood_checkpoints = load_vlm_checkpoints(PRIMARY_MODELS, split="synthetic")
        ood_scores = build_image_scores(ood_checkpoints)
        print(f"  OOD ground truth: {len(ood_gt)} images")

    # --- Calibration ---
    print("\n[2/5] Calibrating...")
    available = [m for m in PRIMARY_MODELS if m in id_scores]

    # A. Bias subtraction
    bias_params = compute_bias(id_scores, id_gt, available)
    id_scores_bias = apply_bias_calibration(id_scores, bias_params)
    for m in available:
        bias = bias_params[m]
        print(f"  {MODEL_NAMES.get(m, m):30s}: bias O={bias['overall']:+.3f}  S={bias['sharpness']:+.3f}  C={bias['color_fidelity']:+.3f}")

    # B. 5-fold CV linear
    id_scores_cv, linear_params = cv_linear_calibration(id_scores, id_gt, available)
    print("  CV linear calibration fitted")

    # OOD calibration (apply ID-fitted params)
    ood_scores_bias: dict[str, dict[str, dict[str, float]]] | None = None
    ood_scores_cv: dict[str, dict[str, dict[str, float]]] | None = None
    if ood_scores:
        ood_scores_bias = apply_bias_calibration(ood_scores, bias_params)
        ood_scores_cv = apply_linear_calibration_ood(ood_scores, linear_params)

    # --- Run all configurations ---
    print("\n[3/5] Computing ensemble configurations...")

    single_models = {MODEL_NAMES.get(m, m) for m in PRIMARY_MODELS}

    # Raw
    raw_id, raw_top3 = run_all_configs(id_scores, id_gt, available)
    raw_id_singles = {k: v for k, v in raw_id.items() if k in single_models}
    raw_id.update(run_weighted_configs(id_scores, id_gt, available, raw_id_singles))

    # Bias-calibrated
    bias_id, _ = run_all_configs(id_scores_bias, id_gt, available, top3_override=raw_top3)
    bias_id.update(run_weighted_configs(id_scores_bias, id_gt, available, bias_id))

    # CV-calibrated
    cv_id, _ = run_all_configs(id_scores_cv, id_gt, available, top3_override=raw_top3)
    cv_id.update(run_weighted_configs(id_scores_cv, id_gt, available, cv_id))

    print(f"  {len(raw_id)} configurations evaluated (raw)")
    print(f"  {len(cv_id)} configurations evaluated (calibrated)")

    # OOD — use ID top-3 and ID-derived weights for consistency
    raw_ood: dict[str, dict[str, Any]] | None = None
    bias_ood: dict[str, dict[str, Any]] | None = None
    cv_ood: dict[str, dict[str, Any]] | None = None
    if ood_scores and ood_gt:
        ood_available = [m for m in available if m in ood_scores]
        raw_ood, _ = run_all_configs(ood_scores, ood_gt, ood_available, top3_override=raw_top3)
        raw_ood.update(run_weighted_configs(ood_scores, ood_gt, ood_available, raw_id_singles))
        if ood_scores_bias:
            bias_ood, _ = run_all_configs(ood_scores_bias, ood_gt, ood_available, top3_override=raw_top3)
            bias_singles = {k: v for k, v in bias_id.items() if k in single_models}
            bias_ood.update(run_weighted_configs(ood_scores_bias, ood_gt, ood_available, bias_singles))
        if ood_scores_cv:
            cv_ood, _ = run_all_configs(ood_scores_cv, ood_gt, ood_available, top3_override=raw_top3)
            cv_singles = {k: v for k, v in cv_id.items() if k in single_models}
            cv_ood.update(run_weighted_configs(ood_scores_cv, ood_gt, ood_available, cv_singles))

    # --- Outputs ---
    print("\n[4/5] Writing outputs...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "raw": {"id": raw_id, "ood": raw_ood},
        "calibrated_bias": {"id": bias_id, "ood": bias_ood},
        "calibrated_cv": {"id": cv_id, "ood": cv_ood},
        "calibration_params": {
            "bias": bias_params,
            "cv_linear": linear_params,
        },
    }
    write_results_json(all_results)
    write_results_md(raw_id, raw_ood, cv_id, cv_ood, bias_params)

    # --- Figures ---
    print("\n[5/5] Generating figures...")
    plot_wsrcc_comparison(raw_id, raw_ood, cv_id, cv_ood)
    plot_improvement_heatmap(id_scores, id_gt, raw_id_singles)
    plot_calibration_impact(raw_id, bias_id, cv_id)

    # --- Recommendation ---
    print_recommendation(raw_id, cv_id)


if __name__ == "__main__":
    main()
