"""Fit per-model calibration curves on DIQA-5000 training predictions.

For each model x dimension (2 models x 3 dimensions = 6 combos), fits three
calibration methods: linear regression, 4-parameter logistic, and isotonic
regression. Saves fitted calibrators as pickle for use by evaluate_calibration.py.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../research/vlm_calibration/fit_calibration.py

    # Single model:
    ... fit_calibration.py --model openai/gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

# --- Paths ---

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CALIBRATOR_DIR = SCRIPT_DIR / "calibrators"

TRAIN_GT_DIR = REPO_ROOT / "DeQA-Score" / "Data-DeQA-Score" / "DIQA" / "metas"
TRAIN_GT_FILES = {
    "overall": TRAIN_GT_DIR / "train_diqa_overall.json",
    "sharpness": TRAIN_GT_DIR / "train_diqa_sharpness.json",
    "color": TRAIN_GT_DIR / "train_diqa_color.json",
}

# Dimension config: (vlm_checkpoint_field, gt_dimension_key)
DIMENSIONS = {
    "overall": ("overall", "overall"),
    "sharpness": ("sharpness", "sharpness"),
    "color": ("color_fidelity", "color"),
}

MODELS: list[str] = [
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-122b-a10b",
]


# --- Data Loading ---


def _image_key(path_or_name: str) -> str:
    """Canonical image key: basename without path prefix."""
    return Path(path_or_name).name


def load_vlm_train_predictions(model_id: str) -> dict[str, dict[str, Any]]:
    """Load VLM training predictions from checkpoint JSONL.

    Returns:
        Dict mapping image basename to prediction dict.
    """
    safe_name = model_id.replace("/", "__")
    jsonl_path = CHECKPOINT_DIR / f"{safe_name}__train.jsonl"

    if not jsonl_path.exists():
        msg = f"Checkpoint not found: {jsonl_path}"
        raise FileNotFoundError(msg)

    preds: dict[str, dict[str, Any]] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("error") and item.get("overall") is not None:
            preds[item["image"]] = item

    return preds


def load_train_gt() -> dict[str, dict[str, float]]:
    """Load training ground truth from 3 dimension-specific JSON files.

    Returns:
        Dict mapping image basename to {overall, sharpness, color} gt_score.
    """
    merged: dict[str, dict[str, float]] = {}

    for dim, gt_path in TRAIN_GT_FILES.items():
        data = json.loads(gt_path.read_text())
        for item in data:
            key = _image_key(item["image"])
            if key not in merged:
                merged[key] = {}
            merged[key][dim] = item["gt_score"]

    return merged


def align_predictions_and_gt(
    preds: dict[str, dict[str, Any]],
    gt: dict[str, dict[str, float]],
    pred_field: str,
    gt_dim: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inner join predictions and GT by image key for a single dimension.

    Returns:
        Tuple of (pred_array, gt_array, matched_keys).
    """
    pred_vals: list[float] = []
    gt_vals: list[float] = []
    keys: list[str] = []

    for key in sorted(preds.keys() & gt.keys()):
        if gt_dim not in gt[key]:
            continue
        pred_val = preds[key].get(pred_field)
        if pred_val is None:
            continue
        pred_vals.append(float(pred_val))
        gt_vals.append(gt[key][gt_dim])
        keys.append(key)

    return np.array(pred_vals), np.array(gt_vals), keys


# --- Calibration Methods ---


def _logistic_4pl(
    x: np.ndarray, b1: float, b2: float, b3: float, b4: float
) -> np.ndarray:
    """4-parameter logistic: y = b1 + (b2-b1) / (1 + exp(b3*(x-b4)))."""
    return b1 + (b2 - b1) / (1.0 + np.exp(b3 * (x - b4)))


def fit_calibrators(
    x_train: np.ndarray, y_train: np.ndarray
) -> dict[str, Any]:
    """Fit 3 calibration methods on training data.

    Returns:
        Dict mapping method name to serializable calibrator object.
        Use ``apply_calibrator(calibrators[name], x)`` to transform.
    """
    calibrators: dict[str, Any] = {}

    # 1. Linear regression
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)
    calibrators["linear"] = lr

    # 2. 4-parameter logistic (4PL)
    try:
        x_min, x_max = float(x_train.min()), float(x_train.max())
        y_min, y_max = float(y_train.min()), float(y_train.max())
        p0 = [y_min, y_max, -10.0, (x_min + x_max) / 2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            popt, _ = optimize.curve_fit(
                _logistic_4pl, x_train, y_train, p0=p0, maxfev=10000
            )
        calibrators["4PL"] = popt  # numpy array of 4 params
    except (RuntimeError, ValueError) as exc:
        print(f"    WARNING: 4PL fit failed ({exc}), skipping")

    # 3. Isotonic regression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(x_train, y_train)
    calibrators["isotonic"] = iso

    return calibrators


def apply_calibrator(calibrator: Any, x: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to transform predictions.

    Args:
        calibrator: Fitted model object (LinearRegression, ndarray for 4PL,
            or IsotonicRegression).
        x: Raw prediction array.

    Returns:
        Calibrated prediction array.
    """
    if isinstance(calibrator, LinearRegression):
        return calibrator.predict(x.reshape(-1, 1))
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.transform(x)
    if isinstance(calibrator, np.ndarray):
        # 4PL parameters
        return _logistic_4pl(x, *calibrator)
    msg = f"Unknown calibrator type: {type(calibrator)}"
    raise TypeError(msg)


# --- Main ---


def fit_model_calibrators(
    model_id: str,
) -> dict[str, dict[str, Any]]:
    """Fit calibrators for all dimensions of a single model.

    Returns:
        Dict of {dim: {method: transform_fn}}.
    """
    preds = load_vlm_train_predictions(model_id)
    gt = load_train_gt()

    print(f"  Predictions: {len(preds)}, GT images: {len(gt)}")

    all_calibrators: dict[str, dict[str, Any]] = {}
    fit_summary: dict[str, dict[str, Any]] = {}

    for dim, (pred_field, gt_dim) in DIMENSIONS.items():
        print(f"\n  Dimension: {dim}")

        x_train, y_train, keys = align_predictions_and_gt(
            preds, gt, pred_field, gt_dim
        )
        print(f"    Aligned: {len(keys)} samples")

        if len(keys) < 30:
            print(f"    WARNING: Too few samples ({len(keys)}), skipping")
            continue

        # Print raw stats
        raw_mae = float(np.mean(np.abs(x_train - y_train)))
        raw_bias = float(np.mean(x_train - y_train))
        print(f"    Raw MAE: {raw_mae:.4f}, bias: {raw_bias:+.4f}")
        print(
            f"    Pred range: [{x_train.min():.2f}, {x_train.max():.2f}], "
            f"GT range: [{y_train.min():.2f}, {y_train.max():.2f}]"
        )

        calibrators = fit_calibrators(x_train, y_train)
        all_calibrators[dim] = calibrators

        # Print fit quality
        dim_summary: dict[str, Any] = {
            "n_samples": len(keys),
            "raw_mae": round(raw_mae, 4),
            "raw_bias": round(raw_bias, 4),
        }

        for method_name, cal_obj in calibrators.items():
            x_cal = apply_calibrator(cal_obj, x_train)
            cal_mae = float(np.mean(np.abs(x_cal - y_train)))
            cal_bias = float(np.mean(x_cal - y_train))
            reduction_pct = (1 - cal_mae / raw_mae) * 100 if raw_mae > 0 else 0
            print(
                f"    {method_name}: MAE={cal_mae:.4f} (Δ{reduction_pct:+.1f}%), "
                f"bias={cal_bias:+.4f}"
            )
            dim_summary[method_name] = {
                "train_mae": round(cal_mae, 4),
                "train_bias": round(cal_bias, 4),
                "mae_reduction_pct": round(reduction_pct, 1),
            }

        fit_summary[dim] = dim_summary

    return all_calibrators


def save_calibrators(
    calibrators: dict[str, dict[str, Any]],
    model_id: str,
) -> Path:
    """Save fitted calibrators as pickle."""
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_id.replace("/", "__")
    out_path = CALIBRATOR_DIR / f"{safe_name}_calibrators.pkl"
    with out_path.open("wb") as f:
        pickle.dump(calibrators, f)
    return out_path


def main() -> None:
    """Fit calibration curves for all models."""
    parser = argparse.ArgumentParser(
        description="Fit VLM calibration curves on DIQA-5000 training split"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Fit only this model (e.g. openai/gpt-4.1)",
    )
    args = parser.parse_args()

    models = MODELS
    if args.model:
        models = [m for m in MODELS if m == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not in MODELS list")
            raise SystemExit(1)

    for model_id in models:
        print(f"\n{'=' * 70}")
        print(f"Fitting calibrators: {model_id}")
        print(f"{'=' * 70}")

        calibrators = fit_model_calibrators(model_id)

        if calibrators:
            out_path = save_calibrators(calibrators, model_id)
            print(f"\n  Saved calibrators to: {out_path}")
        else:
            print("\n  WARNING: No calibrators fitted (insufficient data)")


if __name__ == "__main__":
    main()
