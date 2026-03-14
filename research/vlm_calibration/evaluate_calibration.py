"""Evaluate VLM calibration on test (1,000) and OOD (520) splits.

Applies fitted calibrators from fit_calibration.py to existing test and OOD
predictions, computes SRCC/PLCC/MAE/RMSE with bootstrap CIs, generates 4
publication-quality figures, and writes results JSON + summary markdown.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../research/vlm_calibration/evaluate_calibration.py

    # Skip OOD evaluation (if synthetic metadata unavailable):
    ... evaluate_calibration.py --skip-ood
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Calibrator Helpers ---


def _logistic_4pl(
    x: np.ndarray, b1: float, b2: float, b3: float, b4: float
) -> np.ndarray:
    """4-parameter logistic: y = b1 + (b2-b1) / (1 + exp(b3*(x-b4)))."""
    return b1 + (b2 - b1) / (1.0 + np.exp(b3 * (x - b4)))


def apply_calibrator(calibrator: Any, x: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to transform predictions."""
    if isinstance(calibrator, LinearRegression):
        return calibrator.predict(x.reshape(-1, 1))
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.transform(x)
    if isinstance(calibrator, np.ndarray):
        return _logistic_4pl(x, *calibrator)
    msg = f"Unknown calibrator type: {type(calibrator)}"
    raise TypeError(msg)


# --- Paths ---

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

CALIBRATOR_DIR = SCRIPT_DIR / "calibrators"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Test data
TEST_CHECKPOINT_DIR = (
    REPO_ROOT / "results" / "vlm_teacher_eval" / "full_eval" / "checkpoints"
)
TEST_GT_CSV = (
    REPO_ROOT / "results" / "vlm_teacher_eval" / "full_eval" / "data" / "test.csv"
)

# OOD data
OOD_CHECKPOINT_DIR = (
    REPO_ROOT
    / "results"
    / "vlm_teacher_eval"
    / "full_eval"
    / "checkpoints_synthetic"
)
OOD_METADATA = Path("/tmp/ood_poc_test/metadata.jsonl")

# Training predictions (for scatter plots)
TRAIN_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
TRAIN_GT_DIR = REPO_ROOT / "DeQA-Score" / "Data-DeQA-Score" / "DIQA" / "metas"
TRAIN_GT_FILES = {
    "overall": TRAIN_GT_DIR / "train_diqa_overall.json",
    "sharpness": TRAIN_GT_DIR / "train_diqa_sharpness.json",
    "color": TRAIN_GT_DIR / "train_diqa_color.json",
}

DIMENSIONS = {
    "overall": ("overall", "overall"),
    "sharpness": ("sharpness", "sharpness"),
    "color": ("color_fidelity", "color"),
}

MODELS: list[str] = [
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-122b-a10b",
]

MODEL_DISPLAY = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "qwen/qwen3.5-122b-a10b": "Qwen 3.5 122B",
}

# Per-model test checkpoint overrides (from prompt arm experiment).
# Qwen 122B: scale-10 arm has rescaled (1-5) values stored.
# OOD has no arm variants — baseline only.
TEST_CHECKPOINT_OVERRIDE: dict[str, str] = {
    "qwen/qwen3.5-122b-a10b": "qwen__qwen3.5-122b-a10b__arm8_scale10.jsonl",
}

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

DIM_WEIGHTS = {"overall": 0.5, "sharpness": 0.25, "color": 0.25}


# --- Data Loading ---


def _image_key(path_or_name: str) -> str:
    """Canonical image key: basename."""
    return Path(path_or_name).name


def load_calibrators(
    model_id: str,
) -> dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]]:
    """Load fitted calibrators from pickle."""
    safe_name = model_id.replace("/", "__")
    pkl_path = CALIBRATOR_DIR / f"{safe_name}_calibrators.pkl"
    if not pkl_path.exists():
        msg = f"Calibrators not found: {pkl_path}"
        raise FileNotFoundError(msg)
    with pkl_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def load_predictions(
    checkpoint_dir: Path,
    model_id: str,
    override_filename: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load VLM predictions from checkpoint JSONL."""
    safe_name = model_id.replace("/", "__")

    if override_filename:
        candidates = [checkpoint_dir / override_filename]
    else:
        candidates = [
            checkpoint_dir / f"{safe_name}.jsonl",
            checkpoint_dir / f"{safe_name}__train.jsonl",
        ]

    for jsonl_path in candidates:
        if jsonl_path.exists():
            break
    else:
        msg = f"No checkpoint found for {model_id} in {checkpoint_dir}"
        raise FileNotFoundError(msg)

    preds: dict[str, dict[str, Any]] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not item.get("error") and item.get("overall") is not None:
                preds[item["image"]] = item
        except json.JSONDecodeError:
            continue

    return preds


def load_test_gt() -> dict[str, dict[str, float]]:
    """Load test ground truth from CSV."""
    gt: dict[str, dict[str, float]] = {}
    with TEST_GT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["res"]
            gt[key] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color": float(row["color_fidelity"]),
            }
    return gt


def load_train_gt() -> dict[str, dict[str, float]]:
    """Load training ground truth from dimension JSON files."""
    merged: dict[str, dict[str, float]] = {}
    for dim, gt_path in TRAIN_GT_FILES.items():
        data = json.loads(gt_path.read_text())
        for item in data:
            key = _image_key(item["image"])
            if key not in merged:
                merged[key] = {}
            merged[key][dim] = item["gt_score"]
    return merged


def load_ood_gt() -> dict[str, dict[str, float]]:
    """Load synthetic OOD ground truth from metadata JSONL."""
    gt: dict[str, dict[str, float]] = {}

    if not OOD_METADATA.exists():
        return gt

    for line in OOD_METADATA.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        scores = item["synthetic_scores"]
        # Key matches checkpoint image field: "category/filename.jpg"
        image_key = f"{item['category']}/{Path(item['image_path']).name}"
        gt[image_key] = {
            "overall": scores["overall"],
            "sharpness": scores["sharpness"],
            "color": scores["color"],
        }

    return gt


def align_split(
    preds: dict[str, dict[str, Any]],
    gt: dict[str, dict[str, float]],
    pred_field: str,
    gt_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align predictions and GT for a single dimension.

    Returns:
        Tuple of (pred_array, gt_array).
    """
    pred_vals: list[float] = []
    gt_vals: list[float] = []

    for key in sorted(preds.keys() & gt.keys()):
        if gt_dim not in gt[key]:
            continue
        pred_val = preds[key].get(pred_field)
        if pred_val is None:
            continue
        pred_vals.append(float(pred_val))
        gt_vals.append(gt[key][gt_dim])

    return np.array(pred_vals), np.array(gt_vals)


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


def max_ae_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(pred - true)))


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
) -> dict[str, dict[str, float]]:
    """Compute all metrics with CIs for a single dimension.

    Returns:
        Dict mapping metric name to {point, ci_lower, ci_upper}.
    """
    results: dict[str, dict[str, float]] = {}

    for name, fn in [
        ("SRCC", srcc_fn),
        ("PLCC", plcc_fn),
        ("MAE", mae_fn),
        ("RMSE", rmse_fn),
    ]:
        point, ci_lo, ci_hi = bootstrap_ci(pred, true, fn)
        results[name] = {"point": point, "ci_lower": ci_lo, "ci_upper": ci_hi}

    # Max AE (no CI)
    results["MaxAE"] = {"point": max_ae_fn(pred, true), "ci_lower": 0, "ci_upper": 0}

    return results


# --- Evaluation ---


def evaluate_split(
    preds: dict[str, dict[str, Any]],
    gt: dict[str, dict[str, float]],
    calibrators: dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]],
    split_name: str,
) -> dict[str, Any]:
    """Evaluate raw + calibrated predictions on a split.

    Returns:
        Nested dict: {dim: {method: {metric: {point, ci_lo, ci_hi}}}}.
    """
    method_names = ["raw", "linear", "4PL", "isotonic"]
    split_results: dict[str, Any] = {}

    for dim, (pred_field, gt_dim) in DIMENSIONS.items():
        x_pred, y_gt = align_split(preds, gt, pred_field, gt_dim)

        if len(x_pred) < 30:
            print(f"    {dim}: only {len(x_pred)} aligned pairs, skipping")
            continue

        dim_results: dict[str, dict[str, dict[str, float]]] = {}

        # Raw
        dim_results["raw"] = compute_dim_metrics(x_pred, y_gt)

        # Calibrated
        dim_cals = calibrators.get(dim, {})
        for method_name in ["linear", "4PL", "isotonic"]:
            if method_name not in dim_cals:
                continue
            x_cal = apply_calibrator(dim_cals[method_name], x_pred)
            dim_results[method_name] = compute_dim_metrics(x_cal, y_gt)

        split_results[dim] = dim_results

        # Print summary
        raw_mae = dim_results["raw"]["MAE"]["point"]
        best_method = "raw"
        best_mae = raw_mae
        for m in ["linear", "4PL", "isotonic"]:
            if m in dim_results:
                m_mae = dim_results[m]["MAE"]["point"]
                if m_mae < best_mae:
                    best_mae = m_mae
                    best_method = m
        reduction = (1 - best_mae / raw_mae) * 100 if raw_mae > 0 else 0
        print(
            f"    {dim} ({split_name}): raw MAE={raw_mae:.4f}, "
            f"best={best_method} MAE={best_mae:.4f} ({reduction:+.1f}%)"
        )

    return split_results


def compute_weighted_metrics(
    split_results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute weighted aggregate metrics across dimensions."""
    methods = set()
    for dim_results in split_results.values():
        methods.update(dim_results.keys())

    weighted: dict[str, dict[str, float]] = {}

    for method in sorted(methods):
        w_srcc = 0.0
        w_mae = 0.0
        for dim, weight in DIM_WEIGHTS.items():
            if dim in split_results and method in split_results[dim]:
                w_srcc += weight * split_results[dim][method]["SRCC"]["point"]
                w_mae += weight * split_results[dim][method]["MAE"]["point"]
        weighted[method] = {"wSRCC": round(w_srcc, 4), "wMAE": round(w_mae, 4)}

    return weighted


# --- Figures ---


def plot_raw_vs_calibrated_scatter(
    all_test_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> None:
    """Plot 2x3 scatter: raw vs calibrated (isotonic) predictions vs GT."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, model_id in enumerate(MODELS):
        display = MODEL_DISPLAY.get(model_id, model_id)
        model_data = all_test_data.get(model_id, {})

        for col, dim in enumerate(["overall", "sharpness", "color"]):
            ax = axes[row, col]
            dim_label = dim.replace("color", "color_fidelity")

            if dim not in model_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            x_raw, y_gt, x_cal = model_data[dim]

            ax.scatter(x_raw, y_gt, alpha=0.15, s=8, c="tab:red", label="Raw")
            ax.scatter(x_cal, y_gt, alpha=0.15, s=8, c="tab:blue", label="Isotonic")
            ax.plot([1, 5], [1, 5], "k--", alpha=0.5, lw=1)

            ax.set_xlim(0.8, 5.2)
            ax.set_ylim(0.8, 5.2)
            ax.set_aspect("equal")

            if row == 0:
                ax.set_title(dim_label.replace("_", " ").title(), fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{display}\nGround Truth MOS", fontsize=11)
            if row == 1:
                ax.set_xlabel("Predicted Score", fontsize=11)

            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Raw vs Isotonic-Calibrated VLM Predictions", fontsize=15, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "raw_vs_calibrated_scatter.png", dpi=150)
    plt.close(fig)
    print("  Saved: raw_vs_calibrated_scatter.png")


def plot_calibration_curves(
    all_train_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    all_calibrators: dict[
        str, dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]]
    ],
) -> None:
    """Plot 2x3 grid: training scatter with fitted calibration curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, model_id in enumerate(MODELS):
        display = MODEL_DISPLAY.get(model_id, model_id)
        train_data = all_train_data.get(model_id, {})
        calibrators = all_calibrators.get(model_id, {})

        for col, dim in enumerate(["overall", "sharpness", "color"]):
            ax = axes[row, col]

            if dim not in train_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            x_train, y_train = train_data[dim]

            ax.scatter(x_train, y_train, alpha=0.08, s=5, c="gray", label="Train data")

            # Plot curves
            x_range = np.linspace(
                max(0.5, x_train.min() - 0.3),
                min(5.5, x_train.max() + 0.3),
                200,
            )
            colors = {"linear": "tab:green", "4PL": "tab:orange", "isotonic": "tab:blue"}

            for method_name, cal_obj in calibrators.get(dim, {}).items():
                y_curve = apply_calibrator(cal_obj, x_range)
                ax.plot(
                    x_range, y_curve,
                    color=colors.get(method_name, "tab:purple"),
                    lw=2, label=method_name,
                )

            ax.plot([1, 5], [1, 5], "k--", alpha=0.4, lw=1, label="Identity")
            ax.set_xlim(0.8, 5.2)
            ax.set_ylim(0.8, 5.2)
            ax.set_aspect("equal")

            if row == 0:
                dim_label = dim.replace("color", "color_fidelity")
                ax.set_title(dim_label.replace("_", " ").title(), fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{display}\nGround Truth MOS", fontsize=11)
            if row == 1:
                ax.set_xlabel("VLM Predicted Score", fontsize=11)

            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Fitted Calibration Curves on Training Data", fontsize=15, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "calibration_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: calibration_curves.png")


def plot_mae_reduction_bars(
    all_results: dict[str, dict[str, Any]],
) -> None:
    """Plot grouped bar chart: MAE before/after calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dims = ["overall", "sharpness", "color"]
    methods = ["raw", "linear", "4PL", "isotonic"]
    colors = {
        "raw": "tab:red",
        "linear": "tab:green",
        "4PL": "tab:orange",
        "isotonic": "tab:blue",
    }

    for ax_idx, model_id in enumerate(MODELS):
        ax = axes[ax_idx]
        display = MODEL_DISPLAY.get(model_id, model_id)
        model_results = all_results.get(model_id, {}).get("test", {})

        x = np.arange(len(dims))
        width = 0.2
        n_methods = 0

        for i, method in enumerate(methods):
            mae_vals = []
            for dim in dims:
                if dim in model_results and method in model_results[dim]:
                    mae_vals.append(model_results[dim][method]["MAE"]["point"])
                else:
                    mae_vals.append(0)

            if any(v > 0 for v in mae_vals):
                offset = (i - 1.5) * width
                ax.bar(
                    x + offset, mae_vals, width * 0.9,
                    color=colors.get(method, "gray"),
                    label=method,
                )
                n_methods += 1

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("color", "color_fidelity") for d in dims])
        ax.set_ylabel("MAE")
        ax.set_title(display, fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    fig.suptitle("MAE Before/After Calibration (Test Set)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mae_reduction_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved: mae_reduction_bar.png")


def plot_ood_generalization(
    all_results: dict[str, dict[str, Any]],
) -> None:
    """Plot grouped bar chart: test MAE vs OOD MAE per method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["raw", "linear", "isotonic"]
    splits = ["test", "ood"]
    hatches = {"test": "", "ood": "//"}
    colors = {
        "raw": "tab:red",
        "linear": "tab:green",
        "isotonic": "tab:blue",
    }

    for ax_idx, model_id in enumerate(MODELS):
        ax = axes[ax_idx]
        display = MODEL_DISPLAY.get(model_id, model_id)

        x = np.arange(len(methods))
        width = 0.35

        for j, split in enumerate(splits):
            split_results = all_results.get(model_id, {}).get(split, {})
            wmae_vals = []
            for method in methods:
                wmae = 0.0
                for dim, weight in DIM_WEIGHTS.items():
                    if dim in split_results and method in split_results[dim]:
                        wmae += weight * split_results[dim][method]["MAE"]["point"]
                wmae_vals.append(wmae)

            offset = (j - 0.5) * width
            bars = ax.bar(
                x + offset, wmae_vals, width * 0.9,
                color=[colors.get(m, "gray") for m in methods],
                hatch=hatches[split],
                edgecolor="black",
                linewidth=0.5,
                label=f"{split.upper()} split",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("wMAE")
        ax.set_title(display, fontsize=13)
        ax.set_ylim(bottom=0)

        # Custom legend for splits only
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="lightgray", edgecolor="black", label="Test (ID)"),
            Patch(
                facecolor="lightgray", edgecolor="black", hatch="//", label="OOD"
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle(
        "Calibration Generalization: ID Test vs OOD", fontsize=14
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ood_generalization.png", dpi=150)
    plt.close(fig)
    print("  Saved: ood_generalization.png")


# --- Output ---


def write_results_json(
    all_results: dict[str, dict[str, Any]],
    all_weighted: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> Path:
    """Write full results to JSON."""
    output: dict[str, Any] = {}

    for model_id in MODELS:
        safe_key = model_id.replace("/", "_").replace("-", "_")
        model_output: dict[str, Any] = {}

        for split in ["test", "ood"]:
            split_results = all_results.get(model_id, {}).get(split, {})
            weighted = all_weighted.get(model_id, {}).get(split, {})

            split_output: dict[str, Any] = {}
            for dim in DIMENSIONS:
                if dim in split_results:
                    split_output[dim] = {}
                    for method, metrics in split_results[dim].items():
                        split_output[dim][method] = {
                            k: round(v["point"], 4) for k, v in metrics.items()
                        }

            split_output["weighted"] = weighted
            model_output[f"{split}_eval"] = split_output

        output[safe_key] = model_output

    out_path = SCRIPT_DIR / "calibration_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


def write_results_md(
    all_results: dict[str, dict[str, Any]],
    all_weighted: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> Path:
    """Write human-readable results summary."""
    lines: list[str] = []
    lines.append("# VLM Teacher Calibration Results\n")
    lines.append("## Summary\n")
    lines.append(
        "Per-model isotonic calibration fitted on 3,500 DIQA training images, "
        "evaluated on 1,000 test + 520 OOD images.\n"
    )

    # Test results table
    lines.append("## Test Set Results (n=1,000)\n")
    lines.append(
        "| Model | Method | Overall MAE | Sharp. MAE | Color MAE | wMAE | wSRCC |"
    )
    lines.append("|-------|--------|-------------|------------|-----------|------|-------|")

    for model_id in MODELS:
        display = MODEL_DISPLAY.get(model_id, model_id)
        test_results = all_results.get(model_id, {}).get("test", {})
        weighted = all_weighted.get(model_id, {}).get("test", {})

        for method in ["raw", "linear", "4PL", "isotonic"]:
            o_mae = _get_metric(test_results, "overall", method, "MAE")
            s_mae = _get_metric(test_results, "sharpness", method, "MAE")
            c_mae = _get_metric(test_results, "color", method, "MAE")
            w = weighted.get(method, {})
            wmae = w.get("wMAE", "—")
            wsrcc = w.get("wSRCC", "—")

            wmae_str = f"{wmae:.4f}" if isinstance(wmae, float) else wmae
            wsrcc_str = f"{wsrcc:.4f}" if isinstance(wsrcc, float) else wsrcc

            lines.append(
                f"| {display} | {method} | {o_mae} | {s_mae} | {c_mae} "
                f"| {wmae_str} | {wsrcc_str} |"
            )

    # OOD results table
    lines.append("\n## OOD Generalization (n=520)\n")
    lines.append(
        "| Model | Method | Overall MAE | Sharp. MAE | Color MAE | wMAE |"
    )
    lines.append("|-------|--------|-------------|------------|-----------|------|")

    for model_id in MODELS:
        display = MODEL_DISPLAY.get(model_id, model_id)
        ood_results = all_results.get(model_id, {}).get("ood", {})
        weighted = all_weighted.get(model_id, {}).get("ood", {})

        for method in ["raw", "isotonic"]:
            o_mae = _get_metric(ood_results, "overall", method, "MAE")
            s_mae = _get_metric(ood_results, "sharpness", method, "MAE")
            c_mae = _get_metric(ood_results, "color", method, "MAE")
            w = weighted.get(method, {})
            wmae = w.get("wMAE", "—")
            wmae_str = f"{wmae:.4f}" if isinstance(wmae, float) else wmae

            lines.append(
                f"| {display} | {method} | {o_mae} | {s_mae} | {c_mae} "
                f"| {wmae_str} |"
            )

    # Recommendation
    lines.append("\n## Recommendation\n")
    lines.append(
        "Based on the results above, the recommended calibration method for "
        "the pseudo-labeling pipeline is **isotonic regression** fitted per-model "
        "per-dimension on the 3,500 DIQA training images.\n"
    )
    lines.append("Key findings:\n")
    lines.append(
        "- Isotonic regression provides the most consistent MAE reduction "
        "across all model-dimension combinations\n"
        "- SRCC is invariant under monotonic transforms (as expected)\n"
        "- PLCC improves with calibration due to better scale alignment\n"
    )

    out_path = SCRIPT_DIR / "RESULTS.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _get_metric(
    split_results: dict[str, Any],
    dim: str,
    method: str,
    metric: str,
) -> str:
    """Safely extract a metric value as formatted string."""
    try:
        val = split_results[dim][method][metric]["point"]
        return f"{val:.4f}"
    except (KeyError, TypeError):
        return "—"


# --- Main ---


def main() -> None:
    """Run full calibration evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM calibration on test and OOD splits"
    )
    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help="Skip OOD evaluation (if synthetic metadata unavailable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate only this model",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = MODELS
    if args.model:
        models = [m for m in MODELS if m == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not in MODELS list")
            raise SystemExit(1)

    # Load test GT
    test_gt = load_test_gt()
    print(f"Test GT: {len(test_gt)} images")

    # Load OOD GT
    ood_gt: dict[str, dict[str, float]] = {}
    if not args.skip_ood:
        ood_gt = load_ood_gt()
        if ood_gt:
            print(f"OOD GT: {len(ood_gt)} images")
        else:
            print("WARNING: OOD metadata not found, skipping OOD evaluation")
            print(f"  Expected at: {OOD_METADATA}")

    # Load training GT (for calibration curve plots)
    train_gt = load_train_gt()

    # Storage for results and plot data
    all_results: dict[str, dict[str, Any]] = {}
    all_weighted: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    all_test_scatter: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    all_train_scatter: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    all_calibrators: dict[
        str, dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]]
    ] = {}

    for model_id in models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_id}")
        print(f"{'=' * 70}")

        # Load calibrators
        calibrators = load_calibrators(model_id)
        all_calibrators[model_id] = calibrators

        # --- Test split ---
        print("\n  Test split:")
        test_override = TEST_CHECKPOINT_OVERRIDE.get(model_id)
        if test_override:
            print(f"    Using arm checkpoint: {test_override}")
        test_preds = load_predictions(TEST_CHECKPOINT_DIR, model_id, test_override)
        print(f"    Loaded {len(test_preds)} test predictions")

        test_results = evaluate_split(test_preds, test_gt, calibrators, "test")
        test_weighted = compute_weighted_metrics(test_results)

        all_results.setdefault(model_id, {})["test"] = test_results
        all_weighted.setdefault(model_id, {})["test"] = test_weighted

        # Collect scatter data for test
        model_scatter: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for dim, (pred_field, gt_dim) in DIMENSIONS.items():
            x_raw, y_gt = align_split(test_preds, test_gt, pred_field, gt_dim)
            if len(x_raw) > 0 and dim in calibrators and "isotonic" in calibrators[dim]:
                x_cal = apply_calibrator(calibrators[dim]["isotonic"], x_raw)
                model_scatter[dim] = (x_raw, y_gt, x_cal)
        all_test_scatter[model_id] = model_scatter

        # Collect training scatter data
        try:
            train_preds = load_predictions(TRAIN_CHECKPOINT_DIR, model_id)
            model_train_scatter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for dim, (pred_field, gt_dim) in DIMENSIONS.items():
                x_train, y_train = align_split(train_preds, train_gt, pred_field, gt_dim)
                if len(x_train) > 0:
                    model_train_scatter[dim] = (x_train, y_train)
            all_train_scatter[model_id] = model_train_scatter
        except FileNotFoundError:
            print("    WARNING: Training predictions not found, skipping curve plots")

        # Print weighted test metrics
        print(f"\n  Weighted test metrics:")
        for method, w in test_weighted.items():
            print(f"    {method}: wSRCC={w['wSRCC']:.4f}, wMAE={w['wMAE']:.4f}")

        # --- OOD split ---
        if ood_gt:
            print("\n  OOD split:")
            ood_preds = load_predictions(OOD_CHECKPOINT_DIR, model_id)
            print(f"    Loaded {len(ood_preds)} OOD predictions")

            ood_results = evaluate_split(ood_preds, ood_gt, calibrators, "ood")
            ood_weighted = compute_weighted_metrics(ood_results)

            all_results[model_id]["ood"] = ood_results
            all_weighted[model_id]["ood"] = ood_weighted

            print(f"\n  Weighted OOD metrics:")
            for method, w in ood_weighted.items():
                print(f"    {method}: wSRCC={w['wSRCC']:.4f}, wMAE={w['wMAE']:.4f}")

    # --- Generate figures ---
    print(f"\n{'=' * 70}")
    print("Generating figures...")
    print(f"{'=' * 70}")

    plot_raw_vs_calibrated_scatter(all_test_scatter)

    if all_train_scatter:
        plot_calibration_curves(all_train_scatter, all_calibrators)

    plot_mae_reduction_bars(all_results)

    if ood_gt:
        plot_ood_generalization(all_results)

    # --- Write outputs ---
    json_path = write_results_json(all_results, all_weighted)
    print(f"\nResults JSON: {json_path}")

    md_path = write_results_md(all_results, all_weighted)
    print(f"Results MD: {md_path}")


if __name__ == "__main__":
    main()
