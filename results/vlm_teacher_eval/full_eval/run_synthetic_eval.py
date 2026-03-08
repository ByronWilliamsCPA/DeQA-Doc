"""Evaluate VLM models on the 520-image synthetic OOD PoC dataset.

Tests cross-dataset generalization by evaluating models on synthetically
generated document images with known quality parameters, including OOD
categories (non-Latin scripts, extreme DPI, degraded, pristine, etc.).

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/run_synthetic_eval.py

    # Single model:
    ... run_synthetic_eval.py --model openai/gpt-4.1

    # Metrics only (from checkpoints):
    ... run_synthetic_eval.py --metrics-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt
from results.vlm_teacher_eval.response_parser import parse_iqa_response

EVAL_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = Path("/tmp/ood_poc_test")
SYNTHETIC_META = SYNTHETIC_DIR / "metadata.jsonl"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints_synthetic"
RESULTS_DIR = EVAL_DIR / "results"

# Models to evaluate (all 7 base models from DIQA-5000 evaluation)
MODELS: list[tuple[str, str]] = [
    ("google/gemini-3-flash-preview", "Value"),
    ("openai/gpt-4.1", "Strong"),
    ("anthropic/claude-haiku-4.5", "Value"),
    ("google/gemini-2.5-pro", "Frontier"),
    ("qwen/qwen3.5-flash-02-23", "VL"),
    ("qwen/qwen3-vl-8b-instruct", "Open"),
    ("qwen/qwen3-vl-8b-thinking", "Open"),
]

RATE_LIMIT_S = 0.3
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class SyntheticImage:
    """Synthetic dataset image with ground truth."""

    image_id: str
    image_path: str
    category: str
    is_ood: bool
    gt_overall: float
    gt_sharpness: float
    gt_color: float


@dataclass(frozen=True)
class ImageResult:
    """Result from a single model x image evaluation."""

    model_id: str
    image: str
    category: str
    is_ood: bool
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning: str
    raw_response: str
    latency_ms: int
    error: str


def load_synthetic_dataset() -> list[SyntheticImage]:
    """Load synthetic dataset metadata."""
    images: list[SyntheticImage] = []
    with SYNTHETIC_META.open() as f:
        for line in f:
            d = json.loads(line)
            scores = d["synthetic_scores"]
            images.append(SyntheticImage(
                image_id=d["image_id"],
                image_path=d["image_path"],
                category=d["category"],
                is_ood=d["is_ood"],
                gt_overall=scores["overall"],
                gt_sharpness=scores["sharpness"],
                gt_color=scores["color"],
            ))
    return images


def load_env() -> None:
    """Load .env from repo root."""
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
) -> tuple[str, int, str]:
    """Rate an image via OpenRouter with retry. Returns (text, latency_ms, error)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
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
                                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
            }
            # Thinking models don't accept temperature parameter
            if not is_thinking:
                kwargs["temperature"] = 0.0

            response = client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)
            return response.choices[0].message.content or "", latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            err = str(exc)
            if attempt < max_retries - 1 and any(
                k in err for k in ("429", "500", "502", "503", "timeout")
            ):
                time.sleep(2 ** (attempt + 1))
                continue
            return "", latency_ms, err

    return "", 0, "max retries exceeded"


def checkpoint_path(model_id: str) -> Path:
    """Get checkpoint file path for a model."""
    safe = model_id.replace("/", "__")
    return CHECKPOINT_DIR / f"{safe}.jsonl"


def load_checkpoint(model_id: str) -> dict[str, dict[str, Any]]:
    """Load checkpoint results for a model."""
    cp = checkpoint_path(model_id)
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


def append_checkpoint(model_id: str, result: ImageResult) -> None:
    """Append result to checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with checkpoint_path(model_id).open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def srcc_fn(p: np.ndarray, t: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(stats.spearmanr(p, t).statistic)


def bootstrap_ci(
    pred: np.ndarray, true: np.ndarray, metric_fn: Any,
    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute metric with bootstrapped 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(pred)
    point = float(metric_fn(pred, true))
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            v = float(metric_fn(pred[idx], true[idx]))
            if not np.isnan(v):
                vals.append(v)
        except (ValueError, FloatingPointError):
            continue
    if len(vals) < 30:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_model(
    model_id: str,
    dataset: list[SyntheticImage],
    api_key: str,
    system_prompt: str,
) -> list[ImageResult]:
    """Evaluate a model on synthetic dataset with resume support."""
    existing = load_checkpoint(model_id)
    total = len(dataset)
    results: list[ImageResult] = []
    skipped = 0

    for idx, img in enumerate(dataset):
        if img.image_id in existing:
            results.append(ImageResult(**existing[img.image_id]))
            skipped += 1
            continue

        if skipped and idx == skipped:
            print(f"  Resumed from checkpoint ({skipped} cached)")

        if not Path(img.image_path).exists():
            print(f"  [{idx+1}/{total}] MISSING: {img.image_id}")
            continue

        print(f"  [{idx+1}/{total}] {img.image_id}", end=" ", flush=True)
        img_b64, mt = encode_image_base64(img.image_path)

        raw, latency, error = rate_image_openrouter(
            model_id, img_b64, mt, system_prompt, USER_PROMPT, api_key,
        )

        if error:
            r = ImageResult(
                model_id=model_id, image=img.image_id,
                category=img.category, is_ood=img.is_ood,
                overall=None, sharpness=None, color_fidelity=None,
                reasoning="", raw_response="",
                latency_ms=latency, error=error,
            )
            print(f"ERROR ({latency}ms): {error[:60]}")
        else:
            try:
                rating = parse_iqa_response(raw)
                r = ImageResult(
                    model_id=model_id, image=img.image_id,
                    category=img.category, is_ood=img.is_ood,
                    overall=rating.overall, sharpness=rating.sharpness,
                    color_fidelity=rating.color_fidelity,
                    reasoning=rating.reasoning, raw_response=raw,
                    latency_ms=latency, error="",
                )
                print(
                    f"O={rating.overall:.1f} S={rating.sharpness:.1f} "
                    f"C={rating.color_fidelity:.1f} ({latency}ms) "
                    f"[{img.category}]"
                )
            except ValueError as exc:
                r = ImageResult(
                    model_id=model_id, image=img.image_id,
                    category=img.category, is_ood=img.is_ood,
                    overall=None, sharpness=None, color_fidelity=None,
                    reasoning="", raw_response=raw,
                    latency_ms=latency, error=f"Parse: {exc}",
                )
                print(f"PARSE ({latency}ms): {exc!s:.50s}")

        results.append(r)
        append_checkpoint(model_id, r)
        time.sleep(RATE_LIMIT_S)

    if skipped == total:
        print(f"  All {total} images cached from checkpoint")
    return results


def compute_metrics(
    results: list[ImageResult],
    dataset: list[SyntheticImage],
    subset_name: str = "all",
) -> dict[str, Any]:
    """Compute metrics for a subset of results."""
    gt_lookup = {img.image_id: img for img in dataset}
    valid = [r for r in results if not r.error and r.overall is not None]

    # Filter to subset
    if subset_name == "id":
        valid = [r for r in valid if not r.is_ood]
    elif subset_name == "ood":
        valid = [r for r in valid if r.is_ood]

    n = len(valid)
    if n < 10:
        return {"n": n, "subset": subset_name}

    metrics: dict[str, Any] = {"n": n, "subset": subset_name}

    dim_map = {
        "overall": ("overall", "gt_overall"),
        "sharpness": ("sharpness", "gt_sharpness"),
        "color": ("color_fidelity", "gt_color"),
    }

    for dim_name, (pred_attr, gt_attr) in dim_map.items():
        pred = np.array([getattr(r, pred_attr) for r in valid])
        true = np.array([getattr(gt_lookup[r.image], gt_attr) for r in valid])

        srcc, srcc_lo, srcc_hi = bootstrap_ci(pred, true, srcc_fn)
        mae = float(np.mean(np.abs(pred - true)))
        bias = float(np.mean(pred - true))

        metrics[f"{dim_name}_srcc"] = round(srcc, 4)
        metrics[f"{dim_name}_srcc_ci"] = f"[{srcc_lo:.4f}, {srcc_hi:.4f}]"
        metrics[f"{dim_name}_mae"] = round(mae, 4)
        metrics[f"{dim_name}_bias"] = round(bias, 4)

    # wSRCC
    if "overall_srcc" in metrics:
        metrics["wsrcc"] = round(
            0.5 * metrics["overall_srcc"]
            + 0.25 * metrics.get("sharpness_srcc", 0)
            + 0.25 * metrics.get("color_srcc", 0),
            4,
        )

    return metrics


def compute_per_category(
    results: list[ImageResult],
    dataset: list[SyntheticImage],
) -> dict[str, dict[str, Any]]:
    """Compute per-category SRCC for overall dimension."""
    gt_lookup = {img.image_id: img for img in dataset}
    valid = [r for r in results if not r.error and r.overall is not None]

    categories: dict[str, list[tuple[float, float]]] = {}
    for r in valid:
        gt = gt_lookup[r.image]
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((r.overall, gt.gt_overall))

    cat_metrics: dict[str, dict[str, Any]] = {}
    for cat, pairs in sorted(categories.items()):
        n = len(pairs)
        pred = np.array([p for p, _ in pairs])
        true = np.array([t for _, t in pairs])
        mae = float(np.mean(np.abs(pred - true)))
        bias = float(np.mean(pred - true))

        if n >= 5:
            srcc = float(stats.spearmanr(pred, true).statistic)
        else:
            srcc = float("nan")

        cat_metrics[cat] = {
            "n": n,
            "is_ood": "ood" in cat,
            "srcc_overall": round(srcc, 4) if not np.isnan(srcc) else None,
            "mae_overall": round(mae, 4),
            "bias_overall": round(bias, 4),
        }

    return cat_metrics


def main() -> None:
    """Run synthetic evaluation."""
    parser = argparse.ArgumentParser(description="Synthetic OOD dataset VLM eval")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.metrics_only:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if not SYNTHETIC_META.exists():
        print(f"ERROR: Synthetic dataset not found at {SYNTHETIC_META}")
        sys.exit(1)

    dataset = load_synthetic_dataset()
    print(f"Loaded {len(dataset)} synthetic images")

    models = MODELS
    if args.model:
        models = [(m, t) for m, t in MODELS if m == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not in MODELS list")
            sys.exit(1)

    all_metrics: dict[str, dict] = {}

    for model_id, tier in models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_id}")
        print(f"{'=' * 70}")

        if args.metrics_only:
            existing = load_checkpoint(model_id)
            if not existing:
                print("  No checkpoint, skipping")
                continue
            results = [ImageResult(**v) for v in existing.values()]
            print(f"  Loaded {len(results)} from checkpoint")
        else:
            results = evaluate_model(model_id, dataset, api_key, build_system_prompt())

        ok = sum(1 for r in results if not r.error)
        err = sum(1 for r in results if r.error)
        print(f"\n  Results: {ok} success, {err} errors")

        # Overall metrics
        m_all = compute_metrics(results, dataset, "all")
        m_id = compute_metrics(results, dataset, "id")
        m_ood = compute_metrics(results, dataset, "ood")

        all_metrics[model_id] = {
            "all": m_all,
            "in_distribution": m_id,
            "out_of_distribution": m_ood,
        }

        # Print summary
        print(f"\n  {'Subset':<20s} {'wSRCC':>7s} {'SRCC_O':>7s} {'SRCC_S':>7s} "
              f"{'SRCC_C':>7s} {'MAE_O':>7s} {'Bias_O':>7s}  n")
        print("  " + "-" * 75)
        for label, m in [("All", m_all), ("In-Distribution", m_id), ("OOD", m_ood)]:
            if m.get("overall_srcc") is not None:
                print(
                    f"  {label:<20s} {m.get('wsrcc', 0):>7.4f} "
                    f"{m.get('overall_srcc', 0):>7.4f} "
                    f"{m.get('sharpness_srcc', 0):>7.4f} "
                    f"{m.get('color_srcc', 0):>7.4f} "
                    f"{m.get('overall_mae', 0):>7.4f} "
                    f"{m.get('overall_bias', 0):>+7.4f}  "
                    f"{m['n']}"
                )

        # Per-category breakdown
        cat_metrics = compute_per_category(results, dataset)
        print(f"\n  Per-Category (Overall SRCC):")
        print(f"  {'Category':<30s} {'SRCC':>7s} {'MAE':>7s} {'Bias':>7s}  n")
        print("  " + "-" * 55)
        for cat, cm in cat_metrics.items():
            srcc_s = f"{cm['srcc_overall']:.4f}" if cm['srcc_overall'] is not None else "N/A"
            ood_tag = " *" if cm["is_ood"] else ""
            print(
                f"  {cat:<30s} {srcc_s:>7s} {cm['mae_overall']:>7.4f} "
                f"{cm['bias_overall']:>+7.4f}  {cm['n']}{ood_tag}"
            )

        # Add to model metrics
        all_metrics[model_id]["per_category"] = cat_metrics

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "synthetic_eval_metrics.json"
    json_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved to: {json_path}")

    # Cross-model comparison
    print(f"\n{'=' * 70}")
    print("CROSS-MODEL COMPARISON (synthetic dataset)")
    print(f"{'=' * 70}")
    print(f"{'Model':<35s} {'All':>7s} {'ID':>7s} {'OOD':>7s} {'MAE':>7s}")
    print("-" * 70)
    for model_id, m in all_metrics.items():
        print(
            f"{model_id:<35s} "
            f"{m['all'].get('wsrcc', 0):>7.4f} "
            f"{m['in_distribution'].get('wsrcc', 0):>7.4f} "
            f"{m['out_of_distribution'].get('wsrcc', 0):>7.4f} "
            f"{m['all'].get('overall_mae', 0):>7.4f}"
        )


if __name__ == "__main__":
    main()
