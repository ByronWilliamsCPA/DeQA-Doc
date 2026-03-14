"""Data loading utilities for the DeQA-Doc paper series."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .constants import DIMENSIONS, PROJECT_ROOT, VLM_EVAL_DIR, WSRCC_WEIGHTS


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_ground_truth() -> dict[str, dict[str, float]]:
    """Load DIQA-5000 test set ground truth from test.csv.

    Returns dict mapping image name -> {overall, sharpness, color_fidelity}.
    """
    csv_path = VLM_EVAL_DIR / "data" / "test.csv"
    gt: dict[str, dict[str, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_key = row.get("res", row.get("image", ""))
            gt[image_key] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color_fidelity": float(row["color_fidelity"]),
            }
    return gt


def load_vlm_checkpoints(
    model_ids: list[str],
    split: str = "real",
) -> dict[str, list[dict[str, Any]]]:
    """Load VLM checkpoint JSONL files for given model IDs.

    Args:
        model_ids: List of model ID strings (e.g., 'google__gemini-3-flash-preview')
        split: 'real' for DIQA-5000 checkpoints, 'synthetic' for OOD checkpoints

    Returns:
        Dict mapping model_id -> list of prediction records.
    """
    if split == "real":
        base_dir = VLM_EVAL_DIR / "checkpoints"
    else:
        base_dir = VLM_EVAL_DIR / "checkpoints_synthetic"

    results: dict[str, list[dict[str, Any]]] = {}
    for model_id in model_ids:
        path = base_dir / f"{model_id}.jsonl"
        if path.exists():
            results[model_id] = load_jsonl(path)
        else:
            print(f"  Warning: checkpoint not found: {path}")
    return results


def extract_scores(
    record: dict[str, Any],
) -> dict[str, float | None]:
    """Extract dimension scores from a VLM prediction record.

    Handles both flat format (overall, sharpness, color_fidelity as top-level keys)
    and nested format.
    """
    scores: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        val = record.get(dim)
        if isinstance(val, (int, float)):
            scores[dim] = float(val)
        elif isinstance(val, dict):
            scores[dim] = val.get("score")
        else:
            scores[dim] = None
    return scores


def merge_predictions_with_gt(
    predictions: list[dict[str, Any]],
    gt: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Merge VLM predictions with ground truth, matching on image name.

    Returns list of dicts with pred_overall, gt_overall, etc.
    """
    merged = []
    for pred in predictions:
        image = pred.get("image", pred.get("res", ""))
        # Try matching with and without path prefix
        image_key = Path(image).name if "/" in image else image
        gt_entry = gt.get(image_key) or gt.get(image)
        if gt_entry is None:
            continue

        scores = extract_scores(pred)
        if any(v is None for v in scores.values()):
            continue

        row = {"image": image_key}
        for dim in DIMENSIONS:
            row[f"pred_{dim}"] = scores[dim]
            row[f"gt_{dim}"] = gt_entry[dim]
        merged.append(row)
    return merged


def compute_srcc(pred: list[float], gt: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient and p-value."""
    corr, pval = stats.spearmanr(pred, gt)
    return float(corr), float(pval)


def compute_plcc(pred: list[float], gt: list[float]) -> tuple[float, float]:
    """Compute Pearson linear correlation coefficient and p-value."""
    corr, pval = stats.pearsonr(pred, gt)
    return float(corr), float(pval)


def compute_metrics(
    merged: list[dict[str, Any]],
    dimensions: list[str] | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute SRCC, PLCC, MAE, RMSE, wSRCC with bootstrap 95% CIs.

    Args:
        merged: List of dicts with pred_{dim} and gt_{dim} keys.
        dimensions: Dimensions to compute metrics for (default: all 3).
        n_bootstrap: Number of bootstrap iterations for CI estimation.

    Returns:
        Dict with per-dimension metrics and weighted aggregate.
    """
    if dimensions is None:
        dimensions = DIMENSIONS

    results: dict[str, Any] = {"n": len(merged)}
    srcc_values = []

    for dim in dimensions:
        pred = [r[f"pred_{dim}"] for r in merged]
        gt = [r[f"gt_{dim}"] for r in merged]
        pred_arr = np.array(pred)
        gt_arr = np.array(gt)

        srcc, srcc_p = compute_srcc(pred, gt)
        plcc, plcc_p = compute_plcc(pred, gt)
        mae = float(np.mean(np.abs(pred_arr - gt_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
        bias = float(np.mean(pred_arr - gt_arr))

        # Bootstrap CI for SRCC
        rng = np.random.default_rng(42)
        boot_srcc = []
        n = len(pred)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_corr, _ = stats.spearmanr(pred_arr[idx], gt_arr[idx])
            boot_srcc.append(boot_corr)
        ci_low, ci_high = float(np.percentile(boot_srcc, 2.5)), float(np.percentile(boot_srcc, 97.5))

        results[dim] = {
            "srcc": srcc,
            "srcc_p": srcc_p,
            "srcc_ci": [ci_low, ci_high],
            "plcc": plcc,
            "plcc_p": plcc_p,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        }
        srcc_values.append(srcc)

    # Weighted SRCC (VQualA-style)
    weights = WSRCC_WEIGHTS[: len(dimensions)]
    results["wsrcc"] = float(np.average(srcc_values, weights=weights))

    return results


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load NPZ file containing embeddings and metadata."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
