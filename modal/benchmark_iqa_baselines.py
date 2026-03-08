"""Benchmark VQualA 2025 baseline IQA models on DIQA-5000 test and synthetic datasets.

Evaluates 6 baseline NR-IQA models used by competition organizers:
  DBCNN (0.587), HyperIQA (0.844), StairIQA (0.850),
  MUSIQ (0.859), TReS (0.863), RichIQA (0.866)

These models output single scalar quality scores (not per-dimension).
We compare each model's output against all 3 DIQA dimensions (overall,
sharpness, color_fidelity) to compute SRCC, PLCC, and the VQualA MainScore.

NOTE: StairIQA and RichIQA may not be available in pyiqa. The script
tries alternative model names and skips unavailable models gracefully.
TOPIQ-NR is used as the closest available substitute for RichIQA.
Run with --discover to see all available pyiqa models.

Setup (one-time):
    uv run modal run modal/benchmark_iqa_baselines.py --setup

Usage:
    # Discover available pyiqa NR-IQA models
    uv run modal run modal/benchmark_iqa_baselines.py --discover

    # Run all baselines on both datasets
    uv run modal run modal/benchmark_iqa_baselines.py

    # Single dataset
    uv run modal run modal/benchmark_iqa_baselines.py --dataset diqa
    uv run modal run modal/benchmark_iqa_baselines.py --dataset synthetic

    # Single model
    uv run modal run modal/benchmark_iqa_baselines.py --model DBCNN

    # Metrics only (from existing checkpoints)
    uv run modal run modal/benchmark_iqa_baselines.py --metrics-only

    # Download results locally
    modal volume get iqa-baseline-results / results/iqa_baselines/
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("iqa-baseline-benchmark")

diqa_volume = modal.Volume.from_name("diqa-test-data", create_if_missing=True)
synthetic_volume = modal.Volume.from_name(
    "synthetic-ood-data", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "iqa-baseline-results", create_if_missing=True
)

# ============================================================================
# Modal Image
# ============================================================================

baseline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pyiqa>=0.1.12",
        "timm>=1.0.0",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "tqdm",
        "opencv-python-headless",
    )
)

# ============================================================================
# Constants
# ============================================================================

DIQA_DATA_DIR = "/diqa"
SYNTHETIC_DATA_DIR = "/synthetic/ood_poc_test"
RESULTS_DIR = "/results"

# VQualA 2025 MainScore weights
W_OVERALL = 0.5
W_SHARPNESS = 0.25
W_COLOR = 0.25

# Baseline models from VQualA 2025 competition evaluation.
# pyiqa_name: primary model identifier in the pyiqa library.
# alternatives: fallback names to try if primary fails.
# reported_main_score: score reported by competition organizers on DIQA-5000 test set.
BASELINE_MODELS = [
    {
        "name": "DBCNN",
        "pyiqa_name": "dbcnn",
        "alternatives": [],
        "reported_main_score": 0.587,
    },
    {
        "name": "HyperIQA",
        "pyiqa_name": "hyperiqa",
        "alternatives": [],
        "reported_main_score": 0.844,
    },
    {
        "name": "StairIQA",
        "pyiqa_name": "stariqa",
        "alternatives": ["stairiqa", "stair_iqa"],
        "reported_main_score": 0.850,
    },
    {
        "name": "MUSIQ",
        "pyiqa_name": "musiq",
        "alternatives": ["musiq-koniq", "musiq-spaq"],
        "reported_main_score": 0.859,
    },
    {
        "name": "TReS",
        "pyiqa_name": "tres",
        "alternatives": ["tres-koniq", "tres-flive"],
        "reported_main_score": 0.863,
    },
    {
        "name": "RichIQA",
        "pyiqa_name": "topiq_nr",
        "alternatives": ["topiq_nr-koniq", "richiqa"],
        "reported_main_score": 0.866,
        "note": "TOPIQ-NR as closest pyiqa substitute for RichIQA",
    },
]


# ============================================================================
# Data Loading
# ============================================================================


def load_diqa_dataset(data_dir: str = DIQA_DATA_DIR) -> list[dict]:
    """Load DIQA-5000 test images and ground truth from volume.

    Returns list of dicts with image_id, image_path, and ground_truth
    (overall, sharpness, color on 1-5 MOS scale).
    """
    import csv
    from pathlib import Path

    csv_path = Path(data_dir) / "test.csv"
    if not csv_path.exists():
        msg = f"DIQA GT not found at {csv_path}. Run with --setup first."
        raise FileNotFoundError(msg)

    images = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            images.append(
                {
                    "image_id": row["res"],
                    "image_path": str(Path(data_dir) / "images" / row["res"]),
                    "ground_truth": {
                        "overall": float(row["overall"]),
                        "sharpness": float(row["sharpness"]),
                        "color": float(row["color_fidelity"]),
                    },
                }
            )
    return images


def load_synthetic_dataset(data_dir: str = SYNTHETIC_DATA_DIR) -> list[dict]:
    """Load 520-image synthetic OOD dataset from volume.

    Returns list of dicts with image_id, image_path, category, is_ood,
    and ground_truth (overall, sharpness, color).
    """
    import json
    from pathlib import Path

    meta_path = Path(data_dir) / "metadata.jsonl"
    if not meta_path.exists():
        msg = f"Synthetic metadata not found at {meta_path}."
        raise FileNotFoundError(msg)

    images = []
    with meta_path.open() as f:
        for line in f:
            d = json.loads(line)
            images.append(
                {
                    "image_id": d["image_id"],
                    "image_path": str(Path(data_dir) / d["image_id"]),
                    "category": d.get("category", "unknown"),
                    "is_ood": d.get("is_ood", False),
                    "ground_truth": d["synthetic_scores"],
                }
            )
    return images


def load_checkpoint(checkpoint_path: str) -> dict[str, dict]:
    """Load existing JSONL checkpoint for resume support."""
    import json
    from pathlib import Path

    results: dict[str, dict] = {}
    p = Path(checkpoint_path)
    if not p.exists():
        return results
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            results[item["image_id"]] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def append_checkpoint(checkpoint_path: str, result: dict) -> None:
    """Append a single result to JSONL checkpoint."""
    import json
    from pathlib import Path

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result) + "\n")


# ============================================================================
# Scoring
# ============================================================================


def score_image(metric, img_path: str, device: str = "cuda") -> float:
    """Score a single image with a pyiqa metric.

    Tries path-based inference first (pyiqa handles preprocessing internally).
    Falls back to manual tensor loading if path-based inference fails.
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    try:
        score = metric(img_path)
    except Exception:
        img = Image.open(img_path).convert("RGB")
        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        score = metric(tensor)

    if isinstance(score, torch.Tensor):
        return score.item()
    return float(score)


# ============================================================================
# Metrics Computation
# ============================================================================


def compute_srcc_plcc(
    predicted: list[float], ground_truth: list[float]
) -> dict[str, float]:
    """Compute SRCC and PLCC with 4-parameter logistic curve fitting.

    Matches VQualA 2025 evaluation protocol which applies nonlinear
    regression before computing Pearson correlation.
    """
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit

    pred = np.array(predicted, dtype=np.float64)
    gt = np.array(ground_truth, dtype=np.float64)

    # Remove NaN/Inf
    mask = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[mask], gt[mask]

    if len(pred) < 5:
        return {"srcc": float("nan"), "plcc": float("nan"), "n": int(len(pred))}

    # SRCC (Spearman rank-order correlation)
    srcc, srcc_p = stats.spearmanr(pred, gt)

    # PLCC with 4-parameter logistic nonlinear regression
    def logistic4(x, a, b, c, d):
        return a / (1.0 + np.exp(-c * (x - d))) + b

    try:
        p0 = [
            float(np.max(gt) - np.min(gt)),
            float(np.min(gt)),
            1.0,
            float(np.median(pred)),
        ]
        popt, _ = curve_fit(logistic4, pred, gt, p0=p0, maxfev=20000)
        pred_fitted = logistic4(pred, *popt)
        plcc, plcc_p = stats.pearsonr(pred_fitted, gt)
    except (RuntimeError, ValueError):
        # Fall back to linear Pearson if curve fitting fails
        plcc, plcc_p = stats.pearsonr(pred, gt)

    return {
        "srcc": float(srcc),
        "plcc": float(plcc),
        "srcc_p": float(srcc_p),
        "plcc_p": float(plcc_p),
        "n": int(len(pred)),
    }


def compute_all_metrics(results: list[dict]) -> dict:
    """Compute per-dimension SRCC/PLCC and VQualA MainScore.

    Since baseline models output a single scalar (not per-dimension),
    we compare the same predicted score against each dimension's GT.
    """
    import math

    import numpy as np

    valid = [r for r in results if r.get("predicted_score") is not None]
    if not valid:
        return {"error": "No valid predictions"}

    predicted = [r["predicted_score"] for r in valid]

    # Score statistics for diagnostics
    pred_arr = np.array(predicted)
    score_stats = {
        "mean": float(np.mean(pred_arr)),
        "std": float(np.std(pred_arr)),
        "min": float(np.min(pred_arr)),
        "max": float(np.max(pred_arr)),
    }

    dims = {}
    for dim in ["overall", "sharpness", "color"]:
        gt = [r["ground_truth"][dim] for r in valid]
        dims[dim] = compute_srcc_plcc(predicted, gt)

    # VQualA MainScore = 0.5 * Score_overall + 0.25 * Score_sharpness + 0.25 * Score_color
    # where Score_dim = 0.5 * (|PLCC_dim| + |SRCC_dim|)
    def dim_score(m: dict) -> float:
        s = abs(m["srcc"]) if not math.isnan(m["srcc"]) else 0.0
        p = abs(m["plcc"]) if not math.isnan(m["plcc"]) else 0.0
        return 0.5 * (s + p)

    main_score = (
        W_OVERALL * dim_score(dims["overall"])
        + W_SHARPNESS * dim_score(dims["sharpness"])
        + W_COLOR * dim_score(dims["color"])
    )

    return {
        "overall": dims["overall"],
        "sharpness": dims["sharpness"],
        "color": dims["color"],
        "main_score": float(main_score),
        "score_stats": score_stats,
        "n_valid": len(valid),
    }


# ============================================================================
# Summary Output
# ============================================================================


def print_summary_table(summary: dict) -> None:
    """Print formatted comparison table of all model results."""
    print("\n" + "=" * 90)
    print("BASELINE IQA MODEL COMPARISON — VQualA 2025 Evaluation Protocol")
    print("=" * 90)

    # Group by dataset
    datasets: set[str] = set()
    for val in summary.values():
        if isinstance(val, dict) and "dataset" in val:
            datasets.add(val["dataset"])

    for ds in sorted(datasets):
        print(f"\nDataset: {ds.upper()}")
        header = (
            f"{'Model':<12} {'SRCC_o':>8} {'PLCC_o':>8} {'SRCC_s':>8} "
            f"{'PLCC_s':>8} {'SRCC_c':>8} {'PLCC_c':>8} "
            f"{'Main':>8} {'Report':>8}"
        )
        print(header)
        print("-" * 90)

        for val in summary.values():
            if not isinstance(val, dict) or val.get("dataset") != ds:
                continue
            name = val.get("model", "?")

            if "error" in val:
                print(f"{name:<12} {'UNAVAILABLE — ' + val['error']}")
                continue

            m = val.get("metrics", {})
            if "error" in m:
                print(f"{name:<12} {'NO VALID PREDICTIONS'}")
                continue

            reported = val.get("reported_main_score", 0)
            print(
                f"{name:<12} "
                f"{m['overall']['srcc']:>8.4f} {m['overall']['plcc']:>8.4f} "
                f"{m['sharpness']['srcc']:>8.4f} {m['sharpness']['plcc']:>8.4f} "
                f"{m['color']['srcc']:>8.4f} {m['color']['plcc']:>8.4f} "
                f"{m['main_score']:>8.4f} "
                f"{reported:>8.3f}"
            )

    print()


# ============================================================================
# Modal Images for setup (bundles local data into container)
# ============================================================================

setup_image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(
        local_path="results/vlm_teacher_eval/full_eval/data",
        remote_path="/local_data",
    )
)

# ============================================================================
# Modal Functions
# ============================================================================


@app.function(
    image=setup_image,
    volumes={"/diqa": diqa_volume},
    timeout=600,
)
def setup_diqa_volume() -> int:
    """Upload DIQA-5000 test images and GT to Modal volume (one-time).

    Bundles 1000 test images and test.csv into the container image,
    then copies them to the persistent diqa-test-data Modal volume.
    """
    import shutil
    from pathlib import Path

    src = Path("/local_data")
    dst = Path("/diqa")

    # Copy images
    img_dst = dst / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    src_images = sorted((src / "res").glob("*.jpg"))
    print(f"Uploading {len(src_images)} DIQA test images...")

    for i, img in enumerate(src_images):
        shutil.copy2(img, img_dst / img.name)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(src_images)} copied")

    # Copy GT CSV
    shutil.copy2(src / "test.csv", dst / "test.csv")

    diqa_volume.commit()
    print(f"Done. {len(src_images)} images + test.csv uploaded.")
    return len(src_images)


@app.function(image=baseline_image, timeout=300)
def discover_available_models() -> dict:
    """List available pyiqa models and check which baselines are available."""
    import pyiqa

    all_models = sorted(pyiqa.list_models())

    # Check which of our target model names exist
    target_names: set[str] = set()
    for m in BASELINE_MODELS:
        target_names.add(m["pyiqa_name"])
        target_names.update(m.get("alternatives", []))

    available = [n for n in sorted(target_names) if n in all_models]
    missing = [n for n in sorted(target_names) if n not in all_models]

    return {
        "total_pyiqa_models": len(all_models),
        "all_models": all_models,
        "target_available": available,
        "target_missing": missing,
    }


@app.function(
    image=baseline_image,
    gpu="L4",
    timeout=3600,
    volumes={
        "/diqa": diqa_volume,
        "/synthetic": synthetic_volume,
        "/results": results_volume,
    },
)
def run_evaluations(
    model_names: list[str] | None = None,
    dataset_names: list[str] | None = None,
    metrics_only: bool = False,
) -> dict:
    """Evaluate baseline IQA models on specified datasets.

    Iterates through models sequentially on a single GPU. Each model is
    loaded, run on all requested datasets, then freed before the next.

    Args:
        model_names: List of model names to evaluate (None = all).
        dataset_names: List of dataset names ["diqa", "synthetic"] (None = both).
        metrics_only: If True, skip inference and compute metrics from checkpoints.

    Returns:
        Summary dict keyed by "{model}_{dataset}" with metrics and metadata.
    """
    import gc
    import json
    import time
    from pathlib import Path

    import pyiqa
    import torch
    from tqdm import tqdm

    if model_names is None:
        model_names = [m["name"] for m in BASELINE_MODELS]
    if dataset_names is None:
        dataset_names = ["diqa", "synthetic"]

    # Filter to requested models
    models = [m for m in BASELINE_MODELS if m["name"] in model_names]
    if not models:
        print(f"No matching models for: {model_names}")
        print(f"Available: {[m['name'] for m in BASELINE_MODELS]}")
        return {}

    all_summary: dict[str, dict] = {}

    for model_config in models:
        model_name = model_config["name"]
        pyiqa_name = model_config["pyiqa_name"]
        note = model_config.get("note", "")

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} (pyiqa: {pyiqa_name})")
        if note:
            print(f"  Note: {note}")
        print(f"{'=' * 60}")

        # --- Load model ---
        metric = None
        lower_better = False
        actual_pyiqa_name = pyiqa_name

        if not metrics_only:
            # Try primary name, then alternatives
            names_to_try = [pyiqa_name, *model_config.get("alternatives", [])]
            for try_name in names_to_try:
                try:
                    t0 = time.time()
                    metric = pyiqa.create_metric(try_name, device="cuda")
                    lower_better = getattr(metric, "lower_better", False)
                    actual_pyiqa_name = try_name
                    print(
                        f"  Loaded '{try_name}' in {time.time() - t0:.1f}s "
                        f"(lower_better={lower_better})"
                    )
                    break
                except Exception as exc:
                    print(f"  '{try_name}' not available: {exc}")

            if metric is None:
                msg = f"No available pyiqa model. Tried: {names_to_try}"
                print(f"  SKIP: {msg}")
                for ds in dataset_names:
                    all_summary[f"{model_name}_{ds}"] = {
                        "model": model_name,
                        "dataset": ds,
                        "reported_main_score": model_config["reported_main_score"],
                        "error": msg,
                    }
                continue

        # --- Evaluate on each dataset ---
        for dataset_name in dataset_names:
            print(f"\n  Dataset: {dataset_name}")

            try:
                if dataset_name == "diqa":
                    dataset = load_diqa_dataset()
                else:
                    dataset = load_synthetic_dataset()
            except FileNotFoundError as exc:
                print(f"    SKIP: {exc}")
                all_summary[f"{model_name}_{dataset_name}"] = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "reported_main_score": model_config["reported_main_score"],
                    "error": str(exc),
                }
                continue

            # Checkpoint path
            safe_name = actual_pyiqa_name.replace("-", "_").replace("/", "_")
            ckpt_path = (
                f"{RESULTS_DIR}/checkpoints/{safe_name}_{dataset_name}.jsonl"
            )
            existing = load_checkpoint(ckpt_path)
            print(f"    {len(dataset)} images, {len(existing)} cached")

            # --- Inference ---
            if not metrics_only and metric is not None:
                n_new = 0
                n_err = 0
                for item in tqdm(
                    dataset, desc=f"    {model_name}/{dataset_name}"
                ):
                    img_id = item["image_id"]
                    if img_id in existing:
                        continue

                    try:
                        raw_score = score_image(metric, item["image_path"])
                        result = {
                            "image_id": img_id,
                            "model": model_name,
                            "pyiqa_name": actual_pyiqa_name,
                            "predicted_score": float(raw_score),
                            "lower_better": lower_better,
                            "ground_truth": item["ground_truth"],
                        }
                        n_new += 1
                    except Exception as exc:
                        result = {
                            "image_id": img_id,
                            "model": model_name,
                            "pyiqa_name": actual_pyiqa_name,
                            "predicted_score": None,
                            "error": str(exc),
                            "ground_truth": item["ground_truth"],
                        }
                        n_err += 1

                    existing[img_id] = result
                    append_checkpoint(ckpt_path, result)

                results_volume.commit()
                print(f"    Inference: {n_new} new, {n_err} errors")

            # --- Compute metrics ---
            results_list = list(existing.values())

            # For lower_better models, negate predicted scores so that
            # higher = better quality, aligning with MOS direction
            if lower_better:
                for r in results_list:
                    if r.get("predicted_score") is not None:
                        r["predicted_score"] = -r["predicted_score"]

            metrics = compute_all_metrics(results_list)

            key = f"{model_name}_{dataset_name}"
            all_summary[key] = {
                "model": model_name,
                "pyiqa_name": actual_pyiqa_name,
                "dataset": dataset_name,
                "reported_main_score": model_config["reported_main_score"],
                "n_images": len(results_list),
                "metrics": metrics,
            }

            # Print per-dimension results
            if "error" not in metrics:
                m = metrics
                stats = m.get("score_stats", {})
                print(
                    f"    Score range: [{stats.get('min', '?'):.3f}, "
                    f"{stats.get('max', '?'):.3f}] "
                    f"mean={stats.get('mean', '?'):.3f}"
                )
                print(
                    f"    Overall:   SRCC={m['overall']['srcc']:.4f}  "
                    f"PLCC={m['overall']['plcc']:.4f}"
                )
                print(
                    f"    Sharpness: SRCC={m['sharpness']['srcc']:.4f}  "
                    f"PLCC={m['sharpness']['plcc']:.4f}"
                )
                print(
                    f"    Color:     SRCC={m['color']['srcc']:.4f}  "
                    f"PLCC={m['color']['plcc']:.4f}"
                )
                print(
                    f"    MainScore: {m['main_score']:.4f} "
                    f"(reported: {model_config['reported_main_score']:.3f})"
                )

        # --- Free GPU memory ---
        if not metrics_only and metric is not None:
            del metric
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU memory freed for {model_name}")

    # --- Save summary ---
    summary_path = f"{RESULTS_DIR}/metrics/baseline_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    results_volume.commit()

    print_summary_table(all_summary)
    return all_summary


# ============================================================================
# Local Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    setup: bool = False,
    discover: bool = False,
    dataset: str = "both",
    model: str = "all",
    metrics_only: bool = False,
):
    """Orchestrate baseline IQA evaluation.

    Args:
        setup: Upload DIQA test data to Modal volume (run once).
        discover: List available pyiqa models and exit.
        dataset: Which dataset(s) to evaluate: "diqa", "synthetic", or "both".
        model: Which model to evaluate (e.g. "DBCNN") or "all".
        metrics_only: Compute metrics from existing checkpoints only.
    """
    if setup:
        print("Uploading DIQA test data to Modal volume...")
        n = setup_diqa_volume.remote()
        print(f"Done! {n} images uploaded. Run without --setup to evaluate.")
        return

    if discover:
        print("Discovering available pyiqa models...\n")
        info = discover_available_models.remote()
        print(f"Total pyiqa models: {info['total_pyiqa_models']}")
        print(f"\nTarget models AVAILABLE ({len(info['target_available'])}):")
        for m in info["target_available"]:
            print(f"  + {m}")
        print(f"\nTarget models MISSING ({len(info['target_missing'])}):")
        for m in info["target_missing"]:
            print(f"  - {m}")
        print(f"\nAll pyiqa models ({info['total_pyiqa_models']}):")
        for m in info["all_models"]:
            print(f"  {m}")
        return

    # Determine datasets
    if dataset == "both":
        dataset_names = ["diqa", "synthetic"]
    else:
        dataset_names = [dataset]

    # Determine models
    model_names = None if model == "all" else [model]

    print("Running baseline IQA evaluation:")
    print(f"  Models: {model_names or 'all (' + str(len(BASELINE_MODELS)) + ')'}")
    print(f"  Datasets: {dataset_names}")
    print(f"  Metrics only: {metrics_only}")
    print()

    summary = run_evaluations.remote(
        model_names=model_names,
        dataset_names=dataset_names,
        metrics_only=metrics_only,
    )

    # Save summary locally
    import json
    from pathlib import Path

    local_dir = Path("results/iqa_baselines")
    local_dir.mkdir(parents=True, exist_ok=True)
    summary_path = local_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
