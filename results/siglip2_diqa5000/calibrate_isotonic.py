"""Isotonic regression calibration experiment for SigLIP2 on DIQA-5000.

Fits three calibration methods (linear, 4PL logistic, isotonic) per quality
dimension on the train split (3,500 samples) and evaluates on the test split
(1,000 samples). Reports pre/post calibration SRCC, PLCC, MAE, RMSE with
bootstrapped 95% CIs.

Usage:
    cd DeQA-Score
    .venv/bin/python ../results/siglip2_diqa5000/calibrate_isotonic.py

    # Save results JSON:
    .venv/bin/python ../results/siglip2_diqa5000/calibrate_isotonic.py --output-json
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

# --- Paths ---

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SIGLIP2_TRAIN = SCRIPT_DIR / "siglip2_diqa5000_train.jsonl"
SIGLIP2_TEST = SCRIPT_DIR / "siglip2_diqa5000_test.jsonl"

TRAIN_GT_DIR = REPO_ROOT / "DeQA-Score" / "Data-DeQA-Score" / "DIQA" / "metas"
TRAIN_GT_FILES = {
    "overall": TRAIN_GT_DIR / "train_diqa_overall.json",
    "sharpness": TRAIN_GT_DIR / "train_diqa_sharpness.json",
    "color": TRAIN_GT_DIR / "train_diqa_color.json",
}

TEST_GT_CSV = (
    REPO_ROOT
    / "results"
    / "vlm_teacher_eval"
    / "full_eval"
    / "data"
    / "test.csv"
)

# Dimension config: (siglip2_field, test_csv_column)
DIMENSIONS = {
    "overall": ("iqa_overall_mu", "overall"),
    "sharpness": ("iqa_sharpness_mu", "sharpness"),
    "color": ("iqa_color_mu", "color_fidelity"),
}

# Bootstrap parameters (match run_full_diqa_eval.py)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# --- Data Loading ---


def _image_key(path_or_name: str) -> str:
    """Canonical image key: basename without path prefix."""
    return Path(path_or_name).name


def load_siglip2_predictions(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load SigLIP2 predictions from JSONL, keyed by image basename."""
    preds: dict[str, dict[str, Any]] = {}
    with jsonl_path.open() as f:
        for line in f:
            item = json.loads(line)
            key = _image_key(item["image"])
            preds[key] = item
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


def load_test_gt() -> dict[str, dict[str, float]]:
    """Load test ground truth from CSV.

    Returns:
        Dict mapping image basename to {overall, sharpness, color} MOS.
    """
    gt: dict[str, dict[str, float]] = {}
    with TEST_GT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _image_key(row["res"])
            gt[key] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color": float(row["color_fidelity"]),
            }
    return gt


def align_predictions_and_gt(
    preds: dict[str, dict[str, Any]],
    gt: dict[str, dict[str, float]],
    dim: str,
    pred_field: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inner join predictions and GT by image key for a single dimension.

    Returns:
        Tuple of (pred_array, gt_array, matched_keys).
    """
    pred_vals: list[float] = []
    gt_vals: list[float] = []
    keys: list[str] = []

    for key in sorted(preds.keys() & gt.keys()):
        if dim not in gt[key]:
            continue
        pred_vals.append(preds[key][pred_field])
        gt_vals.append(gt[key][dim])
        keys.append(key)

    return np.array(pred_vals), np.array(gt_vals), keys


# --- Calibration Methods ---


def _logistic_4pl(x: np.ndarray, b1: float, b2: float, b3: float, b4: float) -> np.ndarray:
    """4-parameter logistic function: y = b1 + (b2-b1) / (1 + exp(b3*(x-b4)))."""
    return b1 + (b2 - b1) / (1.0 + np.exp(b3 * (x - b4)))


def fit_calibrators(
    x_train: np.ndarray, y_train: np.ndarray
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Fit 3 calibration methods on training data.

    Returns:
        Dict mapping method name to transform function.
    """
    calibrators: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    # 1. Linear regression
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)
    calibrators["linear"] = lambda x, _lr=lr: _lr.predict(x.reshape(-1, 1))

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
        calibrators["4PL"] = lambda x, _p=popt: _logistic_4pl(x, *_p)
    except (RuntimeError, ValueError) as exc:
        print(f"  WARNING: 4PL fit failed ({exc}), skipping")

    # 3. Isotonic regression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(x_train, y_train)
    calibrators["isotonic"] = lambda x, _iso=iso: _iso.transform(x)

    return calibrators


# --- Metrics ---


def srcc_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(stats.spearmanr(pred, true).statistic)


def plcc_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson linear correlation."""
    return float(stats.pearsonr(pred, true).statistic)


def mae_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(pred - true)))


def rmse_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute metric with bootstrapped 95% CI.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(pred)
    point = float(metric_fn(pred, true))

    boot_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            val = float(metric_fn(pred[idx], true[idx]))
            if not np.isnan(val):
                boot_vals.append(val)
        except (ValueError, FloatingPointError):
            continue

    if len(boot_vals) < 30:
        return point, float("nan"), float("nan")

    ci_lower = float(np.percentile(boot_vals, 2.5))
    ci_upper = float(np.percentile(boot_vals, 97.5))
    return point, ci_lower, ci_upper


def compute_dim_metrics(
    pred: np.ndarray, true: np.ndarray
) -> dict[str, tuple[float, float, float]]:
    """Compute all metrics with CIs for a single dimension.

    Returns:
        Dict mapping metric name to (point, ci_lo, ci_hi).
    """
    return {
        "SRCC": bootstrap_ci(pred, true, srcc_fn),
        "PLCC": bootstrap_ci(pred, true, plcc_fn),
        "MAE": bootstrap_ci(pred, true, mae_fn),
        "RMSE": bootstrap_ci(pred, true, rmse_fn),
    }


# --- Main ---


def run_calibration(output_json: bool = False) -> dict[str, Any]:
    """Run the full calibration experiment."""
    # Load data
    print("Loading data...")
    train_preds = load_siglip2_predictions(SIGLIP2_TRAIN)
    test_preds = load_siglip2_predictions(SIGLIP2_TEST)
    train_gt = load_train_gt()
    test_gt = load_test_gt()

    print(f"  Train predictions: {len(train_preds)}")
    print(f"  Test predictions:  {len(test_preds)}")
    print(f"  Train GT images:   {len(train_gt)}")
    print(f"  Test GT images:    {len(test_gt)}")

    # Results storage
    all_results: dict[str, Any] = {}
    method_names = ["raw", "linear", "4PL", "isotonic"]

    # Per-dimension SRCC for wSRCC computation
    wsrcc_parts: dict[str, dict[str, float]] = {m: {} for m in method_names}
    wmae_parts: dict[str, dict[str, float]] = {m: {} for m in method_names}

    for dim, (pred_field, _csv_col) in DIMENSIONS.items():
        print(f"\n{'=' * 60}")
        print(f"Dimension: {dim}")
        print(f"{'=' * 60}")

        # Align train
        x_train, y_train, train_keys = align_predictions_and_gt(
            train_preds, train_gt, dim, pred_field
        )
        print(f"  Train aligned: {len(train_keys)} samples")

        # Align test
        x_test, y_test, test_keys = align_predictions_and_gt(
            test_preds, test_gt, dim, pred_field
        )
        print(f"  Test aligned:  {len(test_keys)} samples")

        # Fit calibrators on train
        print("  Fitting calibrators...")
        calibrators = fit_calibrators(x_train, y_train)

        # Evaluate each method on test
        dim_results: dict[str, dict[str, tuple[float, float, float]]] = {}

        # Raw (uncalibrated)
        print("  Evaluating: raw")
        dim_results["raw"] = compute_dim_metrics(x_test, y_test)
        wsrcc_parts["raw"][dim] = dim_results["raw"]["SRCC"][0]
        wmae_parts["raw"][dim] = dim_results["raw"]["MAE"][0]

        for method_name, transform in calibrators.items():
            print(f"  Evaluating: {method_name}")
            x_calibrated = transform(x_test)
            dim_results[method_name] = compute_dim_metrics(x_calibrated, y_test)
            wsrcc_parts[method_name][dim] = dim_results[method_name]["SRCC"][0]
            wmae_parts[method_name][dim] = dim_results[method_name]["MAE"][0]

            # SRCC invariance check
            raw_srcc = dim_results["raw"]["SRCC"][0]
            cal_srcc = dim_results[method_name]["SRCC"][0]
            delta = abs(raw_srcc - cal_srcc)
            if delta > 1e-3:
                print(
                    f"  WARNING: {method_name} SRCC delta = {delta:.6f} "
                    f"(raw={raw_srcc:.4f}, cal={cal_srcc:.4f})"
                )

        all_results[dim] = dim_results

    # Compute weighted metrics
    print(f"\n{'=' * 60}")
    print("Weighted metrics (wSRCC = 0.5*O + 0.25*S + 0.25*C)")
    print(f"{'=' * 60}")

    weighted: dict[str, dict[str, float]] = {}
    for method in method_names:
        if method not in wsrcc_parts or not wsrcc_parts[method]:
            continue
        parts = wsrcc_parts[method]
        mae_p = wmae_parts[method]
        wsrcc = (
            0.5 * parts.get("overall", 0)
            + 0.25 * parts.get("sharpness", 0)
            + 0.25 * parts.get("color", 0)
        )
        wmae = (
            0.5 * mae_p.get("overall", 0)
            + 0.25 * mae_p.get("sharpness", 0)
            + 0.25 * mae_p.get("color", 0)
        )
        weighted[method] = {"wSRCC": wsrcc, "wMAE": wmae}

    # Print results table
    print(f"\n{'=' * 60}")
    print("RESULTS: SigLIP2 Calibration on DIQA-5000 (train->test)")
    print(f"{'=' * 60}\n")

    # Per-dimension table
    print(
        f"| {'Method':<10s} | {'Dim':<10s} | {'SRCC':>7s} | {'PLCC':>7s} "
        f"| {'MAE':>7s} | {'RMSE':>7s} |"
    )
    print(f"|{'-' * 12}|{'-' * 12}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")

    for dim in DIMENSIONS:
        for method in method_names:
            if method not in all_results.get(dim, {}):
                continue
            m = all_results[dim][method]
            print(
                f"| {method:<10s} | {dim:<10s} "
                f"| {m['SRCC'][0]:>7.4f} | {m['PLCC'][0]:>7.4f} "
                f"| {m['MAE'][0]:>7.4f} | {m['RMSE'][0]:>7.4f} |"
            )
        print(f"|{'-' * 12}|{'-' * 12}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")

    # Weighted table
    print(f"\n| {'Method':<10s} | {'wSRCC':>7s} | {'wMAE':>7s} |")
    print(f"|{'-' * 12}|{'-' * 9}|{'-' * 9}|")
    for method in method_names:
        if method not in weighted:
            continue
        w = weighted[method]
        print(f"| {method:<10s} | {w['wSRCC']:>7.4f} | {w['wMAE']:>7.4f} |")

    # Delta table (vs raw)
    if "raw" in weighted:
        print(f"\n| {'Method':<10s} | {'dSRCC':>7s} | {'dMAE':>8s} |")
        print(f"|{'-' * 12}|{'-' * 9}|{'-' * 10}|")
        raw_w = weighted["raw"]
        for method in method_names:
            if method == "raw" or method not in weighted:
                continue
            w = weighted[method]
            d_srcc = w["wSRCC"] - raw_w["wSRCC"]
            d_mae = w["wMAE"] - raw_w["wMAE"]
            print(
                f"| {method:<10s} | {d_srcc:>+7.4f} | {d_mae:>+8.4f} |"
            )

    # Build output dict
    output = {
        "experiment": "isotonic_calibration",
        "model": "SigLIP2-IQA-Base-86M",
        "train_samples": len(train_gt),
        "test_samples": len(test_gt),
        "methods": method_names,
        "per_dimension": {},
        "weighted": weighted,
    }

    for dim in DIMENSIONS:
        output["per_dimension"][dim] = {}
        for method in method_names:
            if method not in all_results.get(dim, {}):
                continue
            m = all_results[dim][method]
            output["per_dimension"][dim][method] = {
                metric: {"point": vals[0], "ci_lower": vals[1], "ci_upper": vals[2]}
                for metric, vals in m.items()
            }

    if output_json:
        out_path = SCRIPT_DIR / "calibration_results.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to: {out_path}")

    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Isotonic regression calibration for SigLIP2 on DIQA-5000"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Save results to calibration_results.json",
    )
    args = parser.parse_args()

    run_calibration(output_json=args.output_json)


if __name__ == "__main__":
    main()
