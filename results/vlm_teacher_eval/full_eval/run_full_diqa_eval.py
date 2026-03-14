"""Full DIQA-5000 test set evaluation (n=1000) across selected VLM models.

Downloads test images from GCS, evaluates each model on all 1000 images,
computes SRCC/PLCC/MAE with bootstrapped 95% CIs, and outputs results
compatible with the diqa5000_benchmark_results.csv schema.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/run_full_diqa_eval.py

    # Single model:
    ... run_full_diqa_eval.py --model openai/gpt-4.1

    # Resume interrupted run:
    ... run_full_diqa_eval.py  # auto-resumes from checkpoint

    # Dry run (first 10 images):
    ... run_full_diqa_eval.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt
from results.vlm_teacher_eval.response_parser import parse_iqa_response

# --- Configuration ---

GCS_BUCKET = (
    "gs://image_detection_b/image-preprocessing-detector/"
    "datasets/diqa-5000/diqa-5000/test"
)

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
IMAGES_DIR = DATA_DIR / "res"
TEST_CSV = DATA_DIR / "test.csv"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"
RESULTS_DIR = EVAL_DIR / "results"

# Models selected for full evaluation
MODELS: list[tuple[str, str]] = [
    ("openai/gpt-4.1", "Strong"),
    ("google/gemini-3-flash-preview", "Value"),
    ("google/gemini-2.5-pro", "Frontier"),
    ("anthropic/claude-haiku-4.5", "Value"),
    ("qwen/qwen3-vl-8b-instruct", "VL"),
    ("qwen/qwen3-vl-8b-thinking", "VL"),
    ("qwen/qwen3.5-flash-02-23", "VL"),
    # --- New models (batch 2) ---
    ("qwen/qwen3.5-plus-02-15", "Strong"),
    ("qwen/qwen3.5-122b-a10b", "Strong"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "Strong"),
    ("google/gemini-3.1-flash-lite-preview", "Value"),
    ("bytedance-seed/seed-1.6", "Strong"),
    ("x-ai/grok-4.1-fast", "Strong"),
    # z-ai/glm-5 and z-ai/glm-4.7-flash removed: no image input on OpenRouter
    ("bytedance-seed/seed-1.6-flash", "Value"),
    ("nvidia/nemotron-nano-12b-v2-vl", "VL"),
    ("qwen/qwen3-vl-30b-a3b-thinking", "VL"),
    ("qwen/qwen3-vl-235b-a22b-thinking", "VL"),
    ("mistralai/mistral-small-3.1-24b-instruct", "Value"),
    ("google/gemma-3-4b-it", "VL"),
    ("google/gemma-3-12b-it", "VL"),
    ("google/gemma-3-27b-it", "VL"),
]

# Rate limiting per model (seconds between calls)
RATE_LIMIT_S = 0.3

# Bootstrap parameters for confidence intervals
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ImageResult:
    """Result from a single model x image evaluation."""

    model_id: str
    image: str
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning: str
    raw_response: str
    latency_ms: int
    error: str


@dataclass(frozen=True)
class GroundTruth:
    """DIQA-5000 ground truth for a single image."""

    res_file: str
    overall: float
    sharpness: float
    color_fidelity: float


# --- GCS Download ---


def download_test_data() -> None:
    """Download DIQA-5000 test images and CSV from GCS."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_CSV.exists() and IMAGES_DIR.exists():
        n_images = len(list(IMAGES_DIR.glob("*.jpg")))
        if n_images >= 1000:
            print(f"Test data already downloaded ({n_images} images)")
            return

    print("Downloading DIQA-5000 test data from GCS...")

    # Download test.csv
    if not TEST_CSV.exists():
        print("  Downloading test.csv...")
        subprocess.run(
            ["gsutil", "cp", f"{GCS_BUCKET}/test.csv", str(TEST_CSV)],
            check=True,
        )

    # Download res/ images
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading test images (res/)...")
    subprocess.run(
        [
            "gsutil",
            "-m",
            "cp",
            "-n",  # no-clobber: skip existing
            f"{GCS_BUCKET}/res/*.jpg",
            str(IMAGES_DIR) + "/",
        ],
        check=True,
    )

    n_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"  Downloaded {n_images} test images")


def load_ground_truth() -> list[GroundTruth]:
    """Load ground truth from test.csv."""
    gt: list[GroundTruth] = []
    with TEST_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.append(
                GroundTruth(
                    res_file=row["res"],
                    overall=float(row["overall"]),
                    sharpness=float(row["sharpness"]),
                    color_fidelity=float(row["color_fidelity"]),
                )
            )
    return gt


# --- API Calls ---


def load_env() -> None:
    """Load .env file from repo root."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def rate_image_openrouter(
    model_id: str,
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 3,
    force_temp: float | None = None,
) -> tuple[str, int, str]:
    """Rate an image via OpenRouter with retry logic.

    Args:
        force_temp: If set, override temperature for all models (including
            thinking models that normally skip temperature).

    Returns:
        Tuple of (raw_text, latency_ms, error).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    is_thinking = "thinking" in model_id

    for attempt in range(max_retries):
        start = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model_id,
                "max_tokens": 2048 if is_thinking else 1024,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
            }
            if force_temp is not None:
                kwargs["temperature"] = force_temp
            elif not is_thinking:
                kwargs["temperature"] = 0.0

            response = client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)
            raw_text = response.choices[0].message.content or ""
            return raw_text, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            err_str = str(exc)

            # Retry on rate limits and server errors
            if attempt < max_retries - 1 and (
                "429" in err_str
                or "500" in err_str
                or "502" in err_str
                or "503" in err_str
                or "timeout" in err_str.lower()
            ):
                wait = 2 ** (attempt + 1)
                print(f" RETRY({attempt + 1}) in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue

            return "", latency_ms, err_str

    return "", 0, "max retries exceeded"


# --- Checkpoint Management ---


def checkpoint_path(model_id: str, suffix: str = "") -> Path:
    """Get checkpoint file path for a model."""
    safe_name = model_id.replace("/", "__")
    if suffix:
        return CHECKPOINT_DIR / f"{safe_name}__{suffix}.jsonl"
    return CHECKPOINT_DIR / f"{safe_name}.jsonl"


def load_checkpoint(model_id: str, suffix: str = "") -> dict[str, dict[str, Any]]:
    """Load existing checkpoint results for a model.

    Returns:
        Dict mapping image filename to result dict.
    """
    cp = checkpoint_path(model_id, suffix)
    results: dict[str, dict[str, Any]] = {}
    if not cp.exists():
        return results

    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not item.get("error"):
                results[item["image"]] = item
        except json.JSONDecodeError:
            continue

    return results


def append_checkpoint(model_id: str, result: ImageResult, suffix: str = "") -> None:
    """Append a single result to the model's checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp = checkpoint_path(model_id, suffix)
    with cp.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# --- Metrics Computation ---


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    metric_fn: Any,
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

    boot_vals = []
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


def compute_all_metrics(
    results: list[ImageResult],
    gt_lookup: dict[str, GroundTruth],
) -> dict[str, Any]:
    """Compute full benchmark metrics for a model's results.

    Returns:
        Dict with all metrics matching the benchmark CSV schema.
    """
    # Pair predictions with ground truth (successful results only)
    pairs: list[tuple[ImageResult, GroundTruth]] = []
    for r in results:
        if r.error or r.overall is None:
            continue
        gt = gt_lookup.get(r.image)
        if gt:
            pairs.append((r, gt))

    n = len(pairs)
    success_rate = n / len(results) if results else 0.0

    metrics: dict[str, Any] = {
        "num_samples": n,
        "success_rate": round(success_rate, 4),
    }

    if n < 30:
        print(f"  WARNING: Only {n} valid pairs, skipping metrics")
        return metrics

    for dim in ("overall", "sharpness", "color_fidelity"):
        pred = np.array([getattr(r, dim) for r, _ in pairs])
        true = np.array([getattr(gt, dim) for _, gt in pairs])

        dim_prefix = dim.split("_")[0] if dim != "color_fidelity" else "color"

        srcc, srcc_lo, srcc_hi = bootstrap_ci(pred, true, srcc_fn)
        plcc, plcc_lo, plcc_hi = bootstrap_ci(pred, true, plcc_fn)
        mae = mae_fn(pred, true)
        rmse = rmse_fn(pred, true)

        metrics[f"{dim_prefix}_srcc"] = round(srcc, 4)
        metrics[f"{dim_prefix}_srcc_ci_lower"] = round(srcc_lo, 4)
        metrics[f"{dim_prefix}_srcc_ci_upper"] = round(srcc_hi, 4)
        metrics[f"{dim_prefix}_plcc"] = round(plcc, 4)
        metrics[f"{dim_prefix}_plcc_ci_lower"] = round(plcc_lo, 4)
        metrics[f"{dim_prefix}_plcc_ci_upper"] = round(plcc_hi, 4)
        metrics[f"{dim_prefix}_mae"] = round(mae, 4)
        metrics[f"{dim_prefix}_rmse"] = round(rmse, 4)

    # VQualA weighted SRCC
    if "overall_srcc" in metrics:
        wsrcc = (
            0.5 * metrics["overall_srcc"]
            + 0.25 * metrics.get("sharpness_srcc", 0)
            + 0.25 * metrics.get("color_srcc", 0)
        )
        metrics["wsrcc"] = round(wsrcc, 4)

    # Latency stats
    latencies = [r.latency_ms for r in results if not r.error]
    if latencies:
        metrics["inference_mean_ms"] = round(np.mean(latencies), 1)
        metrics["inference_p50_ms"] = round(np.median(latencies), 1)
        metrics["inference_p95_ms"] = round(np.percentile(latencies, 95), 1)

    return metrics


# --- Main ---


def evaluate_model(
    model_id: str,
    ground_truth: list[GroundTruth],
    api_key: str,
    system_prompt: str,
    limit: int | None = None,
    max_pixels: int = 1024,
    checkpoint_suffix: str = "",
    force_temp: float | None = None,
) -> list[ImageResult]:
    """Evaluate a single model on all test images with resume support."""
    existing = load_checkpoint(model_id, checkpoint_suffix)
    images_to_eval = ground_truth[:limit] if limit else ground_truth
    total = len(images_to_eval)

    results: list[ImageResult] = []
    skipped = 0

    for idx, gt in enumerate(images_to_eval):
        # Resume from checkpoint
        if gt.res_file in existing:
            r = ImageResult(**existing[gt.res_file])
            results.append(r)
            skipped += 1
            continue

        if skipped and idx == skipped:
            print(f"  Resumed from checkpoint ({skipped} cached)")

        img_path = str(IMAGES_DIR / gt.res_file)
        if not Path(img_path).exists():
            print(f"  [{idx + 1}/{total}] MISSING: {gt.res_file}")
            continue

        print(
            f"  [{idx + 1}/{total}] {gt.res_file}",
            end=" ",
            flush=True,
        )

        img_b64, media_type = encode_image_base64(img_path, max_pixels=max_pixels)

        raw_text, latency_ms, error = rate_image_openrouter(
            model_id=model_id,
            image_b64=img_b64,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=USER_PROMPT,
            api_key=api_key,
            force_temp=force_temp,
        )

        if error:
            r = ImageResult(
                model_id=model_id,
                image=gt.res_file,
                overall=None,
                sharpness=None,
                color_fidelity=None,
                reasoning="",
                raw_response="",
                latency_ms=latency_ms,
                error=error,
            )
            print(f"ERROR ({latency_ms}ms): {error[:80]}")
        else:
            try:
                rating = parse_iqa_response(raw_text)
                r = ImageResult(
                    model_id=model_id,
                    image=gt.res_file,
                    overall=rating.overall,
                    sharpness=rating.sharpness,
                    color_fidelity=rating.color_fidelity,
                    reasoning=rating.reasoning,
                    raw_response=raw_text,
                    latency_ms=latency_ms,
                    error="",
                )
                print(
                    f"O={rating.overall:.1f} S={rating.sharpness:.1f} "
                    f"C={rating.color_fidelity:.1f} ({latency_ms}ms)"
                )
            except ValueError as exc:
                r = ImageResult(
                    model_id=model_id,
                    image=gt.res_file,
                    overall=None,
                    sharpness=None,
                    color_fidelity=None,
                    reasoning="",
                    raw_response=raw_text,
                    latency_ms=latency_ms,
                    error=f"Parse error: {exc}",
                )
                print(f"PARSE ({latency_ms}ms): {exc!s:.60s}")

        results.append(r)
        append_checkpoint(model_id, r, checkpoint_suffix)
        time.sleep(RATE_LIMIT_S)

    if skipped == total:
        print(f"  All {total} images cached from checkpoint")

    return results


def write_benchmark_csv(
    all_metrics: dict[str, dict[str, Any]],
) -> Path:
    """Write results in diqa5000_benchmark_results.csv format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "vlm_benchmark_results.csv"

    fieldnames = [
        "model_id",
        "model_type",
        "benchmark_date",
        "num_samples",
        "success_rate",
        "overall_plcc",
        "overall_plcc_ci_lower",
        "overall_plcc_ci_upper",
        "overall_srcc",
        "overall_srcc_ci_lower",
        "overall_srcc_ci_upper",
        "overall_mae",
        "overall_rmse",
        "sharpness_plcc",
        "sharpness_plcc_ci_lower",
        "sharpness_plcc_ci_upper",
        "sharpness_srcc",
        "sharpness_srcc_ci_lower",
        "sharpness_srcc_ci_upper",
        "sharpness_mae",
        "sharpness_rmse",
        "color_plcc",
        "color_plcc_ci_lower",
        "color_plcc_ci_upper",
        "color_srcc",
        "color_srcc_ci_lower",
        "color_srcc_ci_upper",
        "color_mae",
        "color_rmse",
        "wsrcc",
        "inference_mean_ms",
        "inference_p50_ms",
        "inference_p95_ms",
        "gpu_type",
        "notes",
    ]

    today = time.strftime("%Y-%m-%d")
    model_tier = dict(MODELS)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for model_id, metrics in all_metrics.items():
            row = {
                "model_id": model_id,
                "model_type": "vlm",
                "benchmark_date": today,
                "gpu_type": "API",
                "notes": f"OpenRouter API, tier={model_tier.get(model_id, '?')}",
            }
            row.update(metrics)
            writer.writerow(row)

    return out_path


def main() -> None:
    """Run full DIQA-5000 evaluation."""
    parser = argparse.ArgumentParser(description="Full DIQA-5000 VLM evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate only this model (e.g. openai/gpt-4.1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N images (for dry run)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics from existing checkpoints without running API calls",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1024,
        help="Max longest-edge pixels for resize (0 = no resize). Default: 1024",
    )
    parser.add_argument(
        "--force-temp0",
        action="store_true",
        help="Force temperature=0.0 for all models, including thinking models "
        "that normally skip the temperature parameter",
    )
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.metrics_only:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    # Download test data
    if not args.metrics_only:
        download_test_data()

    # Load ground truth
    ground_truth = load_ground_truth()
    gt_lookup = {gt.res_file: gt for gt in ground_truth}
    print(f"Loaded {len(ground_truth)} ground truth entries")

    system_prompt = build_system_prompt()

    # Determine checkpoint suffix for non-default max_pixels
    max_pixels = args.max_pixels
    checkpoint_suffix = ""
    if max_pixels == 0:
        checkpoint_suffix = "no_resize"
        print("Image resize: DISABLED (sending original resolution)")
    elif max_pixels != 1024:
        checkpoint_suffix = f"px{max_pixels}"
        print(f"Image resize: max {max_pixels}px longest edge")
    else:
        print(f"Image resize: max {max_pixels}px longest edge (default)")

    # Handle --force-temp0 flag
    force_temp: float | None = None
    if args.force_temp0:
        force_temp = 0.0
        # Append temp0 to checkpoint suffix so results go to a separate file
        checkpoint_suffix = f"{checkpoint_suffix}__temp0" if checkpoint_suffix else "temp0"
        print("Temperature: FORCED to 0.0 (overriding thinking-model default)")

    # Select models
    models = MODELS
    if args.model:
        models = [(m, t) for m, t in MODELS if m == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not in MODELS list")
            sys.exit(1)

    all_metrics: dict[str, dict[str, Any]] = {}

    for model_id, tier in models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_id} (tier={tier})")
        print(f"{'=' * 80}")

        if args.metrics_only:
            # Load from checkpoint
            existing = load_checkpoint(model_id, checkpoint_suffix)
            if not existing:
                print(f"  No checkpoint found, skipping")
                continue
            results = [ImageResult(**v) for v in existing.values()]
            print(f"  Loaded {len(results)} results from checkpoint")
        else:
            results = evaluate_model(
                model_id=model_id,
                ground_truth=ground_truth,
                api_key=api_key,
                system_prompt=system_prompt,
                limit=args.limit,
                max_pixels=max_pixels,
                checkpoint_suffix=checkpoint_suffix,
                force_temp=force_temp,
            )

        # Compute metrics
        ok = sum(1 for r in results if not r.error)
        err = sum(1 for r in results if r.error)
        print(f"\n  Results: {ok} success, {err} errors")

        metrics = compute_all_metrics(results, gt_lookup)
        all_metrics[model_id] = metrics

        # Print summary
        if "overall_srcc" in metrics:
            print(f"  Overall SRCC: {metrics['overall_srcc']:.4f} "
                  f"[{metrics['overall_srcc_ci_lower']:.4f}, "
                  f"{metrics['overall_srcc_ci_upper']:.4f}]")
            print(f"  Sharpness SRCC: {metrics.get('sharpness_srcc', 'N/A')}")
            print(f"  Color SRCC: {metrics.get('color_srcc', 'N/A')}")
            print(f"  wSRCC: {metrics.get('wsrcc', 'N/A')}")
            print(f"  Overall MAE: {metrics.get('overall_mae', 'N/A')}")

    # Write benchmark CSV
    csv_path = write_benchmark_csv(all_metrics)
    print(f"\nBenchmark CSV: {csv_path}")

    # Write metrics JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "vlm_benchmark_metrics.json"
    json_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Metrics JSON: {json_path}")

    # Print leaderboard
    print(f"\n{'=' * 80}")
    print("LEADERBOARD (by wSRCC)")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<40s} {'wSRCC':>7s} {'SRCC_O':>7s} {'SRCC_S':>7s} "
        f"{'SRCC_C':>7s} {'MAE_O':>7s} {'n':>5s}"
    )
    print("-" * 80)

    ranked = sorted(
        all_metrics.items(),
        key=lambda x: x[1].get("wsrcc", -1),
        reverse=True,
    )
    for model_id, m in ranked:
        print(
            f"{model_id:<40s} "
            f"{m.get('wsrcc', 0):>7.4f} "
            f"{m.get('overall_srcc', 0):>7.4f} "
            f"{m.get('sharpness_srcc', 0):>7.4f} "
            f"{m.get('color_srcc', 0):>7.4f} "
            f"{m.get('overall_mae', 0):>7.4f} "
            f"{m.get('num_samples', 0):>5d}"
        )

    # Reference line
    print("-" * 80)
    print(f"{'DeQA-Doc-3Specialists (ref)':<40s} {'0.7160':>7s} "
          f"{'0.7330':>7s} {'0.6810':>7s} {'0.7160':>7s} {'—':>7s} {'1000':>5s}")


if __name__ == "__main__":
    main()
