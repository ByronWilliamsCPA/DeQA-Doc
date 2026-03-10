"""Full-scale prompt arm evaluation (n=1000) for VLM models via OpenRouter.

Runs all 7 prompt arms from the Paper 3 optimization experiment at full scale
on the DIQA-5000 test set. Reuses arm implementations from the pilot study
(run_prompt_optimization.py) with the full evaluation checkpoint infrastructure
(run_full_diqa_eval.py).

Arms:
  1. single_all3    - Single prompt, all 3 dimensions (baseline, reuses existing checkpoint)
  2. separate_3     - 3 separate prompts, one per dimension
  3. hybrid         - 1 prompt for overall, separate for sharpness & color
  4. few_shot       - Single prompt + 3 example images as calibration anchors
  5. multi_sample   - 3 calls with temp=0.3, take median
  6. res_2048       - Single prompt, resize to 2048px instead of 1024
  7. no_resize      - Single prompt, no image resizing (full resolution)
  8. scale_10       - Single prompt, 1-10 scale (rescaled to 1-5 for comparison)
  9. half_step      - Single prompt, 1-5 scale with 0.5 increments

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/run_full_prompt_arms.py \
        --model qwen/qwen3.5-flash-02-23 --arm 7

    # Run all non-baseline arms:
    ... run_full_prompt_arms.py --model qwen/qwen3.5-flash-02-23 --all

    # Compute metrics from existing checkpoints:
    ... run_full_prompt_arms.py --model qwen/qwen3.5-flash-02-23 --metrics-only

    # Dry run (first 5 images):
    ... run_full_prompt_arms.py --model qwen/qwen3.5-flash-02-23 --arm 7 --limit 5
"""

from __future__ import annotations

import argparse
import csv
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
from results.vlm_teacher_eval.prompts import (
    USER_PROMPT,
    IQAPromptConfig,
    build_system_prompt,
)
from results.vlm_teacher_eval.response_parser import parse_iqa_response

# --- Configuration ---

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
IMAGES_DIR = DATA_DIR / "res"
TEST_CSV = DATA_DIR / "test.csv"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"

RATE_LIMIT_S = 0.3
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# Few-shot example images (same as pilot)
FEW_SHOT_EXAMPLES = [
    {
        "file": "test_res_00558.jpg",
        "overall": 1.5, "sharpness": 1.6, "color_fidelity": 1.6,
        "label": "bad quality",
    },
    {
        "file": "test_res_00188.jpg",
        "overall": 3.0, "sharpness": 2.9, "color_fidelity": 3.0,
        "label": "fair quality",
    },
    {
        "file": "test_res_00354.jpg",
        "overall": 4.1, "sharpness": 4.0, "color_fidelity": 4.1,
        "label": "excellent quality",
    },
]

# Arm definitions
ARM_INFO = {
    2: ("arm2_separate", "3 separate dimension-specific prompts"),
    3: ("arm3_hybrid", "Overall combined, sub-dimensions separate"),
    4: ("arm4_few_shot", "Single prompt + 3 calibration examples"),
    5: ("arm5_multi_sample", "3 calls at temp=0.3, take median"),
    6: ("arm6_res2048", "Single prompt, 2048px resize"),
    7: ("arm7_no_resize", "Single prompt, native resolution"),
    8: ("arm8_scale10", "Single prompt, 1-10 scale (rescaled to 1-5)"),
    9: ("arm9_half_step", "Single prompt, 1-5 scale with 0.5 increments"),
}

# Scale configurations for arms 8 and 9
SCALE_10_CONFIG = IQAPromptConfig(scale_min=1.0, scale_max=10.0, scale_step=0.1)
HALF_STEP_CONFIG = IQAPromptConfig(scale_min=1.0, scale_max=5.0, scale_step=0.5)

# User prompt for 1-10 scale (adjusted range in JSON template)
USER_PROMPT_SCALE10 = """\
Rate the quality of this document image.

Respond with exactly this JSON structure:
{{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}}

Scores should be on a 1.0-10.0 scale. \
The reasoning field should be 1-2 sentences explaining the key quality \
factors you observed. Keep it concise.\
"""

# User prompt for 0.5 increments (explicit constraint)
USER_PROMPT_HALF_STEP = """\
Rate the quality of this document image.

Respond with exactly this JSON structure:
{{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}}

Use only 0.5 increments: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, or 5.0. \
The reasoning field should be 1-2 sentences explaining the key quality \
factors you observed. Keep it concise.\
"""


def rescale_10_to_5(score: float) -> float:
    """Linearly rescale a 1-10 score to 1-5."""
    return (score - 1.0) / 9.0 * 4.0 + 1.0


# Dimension-specific prompts for arms 2 and 3
SINGLE_DIM_SYSTEM = """\
You are an expert document image quality assessor. You evaluate scanned or \
photographed document images for visual quality as perceived by a human reader.

You will rate documents on a single specific quality dimension using a 1.0-5.0 \
scale with 0.1 increments.

Scale anchors:
- 1.0: Completely unusable / severe degradation
- 2.0: Poor - significant issues
- 3.0: Fair - acceptable but with noticeable problems
- 4.0: Good - minor issues, generally readable
- 5.0: Excellent - crisp, clean, high-quality reproduction

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.\
"""

DIM_PROMPTS = {
    "overall": (
        'Rate the OVERALL QUALITY of this document image.\n\n'
        'Consider holistic readability and usability.\n\n'
        'Respond with exactly: {{"score": X.X, "reasoning": "..."}}'
    ),
    "sharpness": (
        'Rate the SHARPNESS of this document image.\n\n'
        'Focus on text edge clarity, blur level, and resolution. '
        'Ignore color issues.\n\n'
        'Respond with exactly: {{"score": X.X, "reasoning": "..."}}'
    ),
    "color_fidelity": (
        'Rate the COLOR FIDELITY of this document image.\n\n'
        'Focus on color accuracy, contrast, white balance. '
        'Ignore blur.\n\n'
        'Respond with exactly: {{"score": X.X, "reasoning": "..."}}'
    ),
}


# --- Data classes ---


@dataclass(frozen=True)
class ImageResult:
    """Result from a single model x arm x image evaluation."""

    model_id: str
    arm: str
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


# --- Data loading ---


def load_ground_truth() -> list[GroundTruth]:
    """Load ground truth from test.csv."""
    gt: list[GroundTruth] = []
    with TEST_CSV.open() as f:
        for row in csv.DictReader(f):
            gt.append(GroundTruth(
                res_file=row["res"],
                overall=float(row["overall"]),
                sharpness=float(row["sharpness"]),
                color_fidelity=float(row["color_fidelity"]),
            ))
    return gt


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


# --- Checkpoint management ---


def checkpoint_path(model_id: str, arm_suffix: str) -> Path:
    """Get checkpoint file path for a model+arm combination."""
    safe_name = model_id.replace("/", "__")
    return CHECKPOINT_DIR / f"{safe_name}__{arm_suffix}.jsonl"


def load_checkpoint(model_id: str, arm_suffix: str) -> dict[str, dict[str, Any]]:
    """Load checkpoint, keeping ALL records (including errors) for paired analysis."""
    cp = checkpoint_path(model_id, arm_suffix)
    results: dict[str, dict[str, Any]] = {}
    if not cp.exists():
        return results
    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            results[item["image"]] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def append_checkpoint(model_id: str, arm_suffix: str, result: ImageResult) -> None:
    """Append a single result to checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp = checkpoint_path(model_id, arm_suffix)
    with cp.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# --- API calls ---


def api_call(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_id: str,
    temperature: float = 0.0,
    extra_images: list[tuple[str, str, str]] | None = None,
    max_retries: int = 3,
) -> tuple[str, int, str]:
    """Make an OpenRouter API call with retry logic.

    Returns:
        Tuple of (raw_text, latency_ms, error).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    user_content: list[dict[str, Any]] = []

    # Add few-shot examples if provided
    if extra_images:
        for ex_b64, ex_mt, ex_label in extra_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{ex_mt};base64,{ex_b64}"},
            })
            user_content.append({"type": "text", "text": ex_label})

    # Add the target image
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
    })
    user_content.append({"type": "text", "text": user_prompt})

    for attempt in range(max_retries):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                temperature=temperature,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            latency = int((time.time() - start) * 1000)
            return resp.choices[0].message.content or "", latency, ""
        except Exception as exc:
            latency = int((time.time() - start) * 1000)
            err_str = str(exc)
            if attempt < max_retries - 1 and (
                "429" in err_str or "500" in err_str or "502" in err_str
                or "503" in err_str or "timeout" in err_str.lower()
            ):
                wait = 2 ** (attempt + 1)
                print(f" RETRY({attempt + 1}) in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            return "", latency, err_str

    return "", 0, "max retries exceeded"


def parse_single_score(raw: str) -> tuple[float | None, str]:
    """Parse a single-dimension JSON response. Returns (score, error)."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")
    try:
        d = json.loads(cleaned)
        v = float(d["score"])
        if 1.0 <= v <= 5.0:
            return v, ""
        return None, f"score {v} out of range"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None, str(e)


# --- Arm implementations ---


def run_arm2_separate(
    img_b64: str, mt: str, api_key: str, model_id: str,
) -> tuple[dict[str, float | None], int, str]:
    """Arm 2: 3 separate dimension-specific prompts."""
    scores: dict[str, float | None] = {}
    total_lat = 0
    errors: list[str] = []
    for dim, prompt in DIM_PROMPTS.items():
        raw, lat, err = api_call(img_b64, mt, SINGLE_DIM_SYSTEM, prompt, api_key, model_id)
        total_lat += lat
        if err:
            scores[dim] = None
            errors.append(f"{dim}: {err}")
        else:
            score, parse_err = parse_single_score(raw)
            scores[dim] = score
            if parse_err:
                errors.append(f"{dim}: {parse_err}")
        time.sleep(0.2)
    return scores, total_lat, "; ".join(errors)


def run_arm3_hybrid(
    img_b64: str, mt: str, api_key: str, sys_prompt: str, model_id: str,
) -> tuple[dict[str, float | None], int, str]:
    """Arm 3: Combined for overall, separate for sharpness & color."""
    scores: dict[str, float | None] = {}
    errors: list[str] = []

    # Overall via combined prompt
    raw, lat1, err = api_call(img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id)
    if err:
        scores["overall"] = None
        errors.append(f"overall: {err}")
    else:
        try:
            rating = parse_iqa_response(raw)
            scores["overall"] = rating.overall
        except ValueError as e:
            scores["overall"] = None
            errors.append(f"overall: {e}")

    # Sharpness separate
    time.sleep(0.2)
    raw, lat2, err = api_call(img_b64, mt, SINGLE_DIM_SYSTEM, DIM_PROMPTS["sharpness"], api_key, model_id)
    if err:
        scores["sharpness"] = None
        errors.append(f"sharpness: {err}")
    else:
        score, parse_err = parse_single_score(raw)
        scores["sharpness"] = score
        if parse_err:
            errors.append(f"sharpness: {parse_err}")

    # Color separate
    time.sleep(0.2)
    raw, lat3, err = api_call(img_b64, mt, SINGLE_DIM_SYSTEM, DIM_PROMPTS["color_fidelity"], api_key, model_id)
    if err:
        scores["color_fidelity"] = None
        errors.append(f"color: {err}")
    else:
        score, parse_err = parse_single_score(raw)
        scores["color_fidelity"] = score
        if parse_err:
            errors.append(f"color: {parse_err}")

    return scores, lat1 + lat2 + lat3, "; ".join(errors)


def run_arm4_few_shot(
    img_b64: str, mt: str, api_key: str, sys_prompt: str, model_id: str,
    example_data: list[tuple[str, str, str]],
) -> tuple[dict[str, float | None], int, str, str]:
    """Arm 4: Single prompt + few-shot examples. Returns (scores, lat, error, raw)."""
    raw, lat, err = api_call(
        img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id,
        extra_images=example_data,
    )
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat, err, raw
    try:
        rating = parse_iqa_response(raw)
        return {
            "overall": rating.overall,
            "sharpness": rating.sharpness,
            "color_fidelity": rating.color_fidelity,
        }, lat, "", raw
    except ValueError as e:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat, str(e), raw


def run_arm5_multi_sample(
    img_b64: str, mt: str, api_key: str, sys_prompt: str, model_id: str,
) -> tuple[dict[str, float | None], int, str, str]:
    """Arm 5: 3 calls at temp=0.3, take median. Returns (scores, lat, error, raw)."""
    all_scores: dict[str, list[float]] = {"overall": [], "sharpness": [], "color_fidelity": []}
    total_lat = 0
    last_raw = ""

    for _ in range(3):
        raw, lat, err = api_call(
            img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id,
            temperature=0.3,
        )
        total_lat += lat
        last_raw = raw
        if not err:
            try:
                rating = parse_iqa_response(raw)
                all_scores["overall"].append(rating.overall)
                all_scores["sharpness"].append(rating.sharpness)
                all_scores["color_fidelity"].append(rating.color_fidelity)
            except ValueError:
                pass
        time.sleep(0.2)

    result: dict[str, float | None] = {}
    for dim, vals in all_scores.items():
        result[dim] = float(np.median(vals)) if vals else None

    has_all = all(result[d] is not None for d in result)
    error = "" if has_all else "incomplete multi-sample"
    return result, total_lat, error, last_raw


def run_single_prompt_arm(
    img_path: str, api_key: str, sys_prompt: str, model_id: str,
    max_pixels: int,
) -> tuple[dict[str, float | None], int, str, str]:
    """Shared implementation for arms 1/6/7 (single prompt, different resolutions).

    Returns:
        (scores_dict, latency_ms, error_str, raw_response)
    """
    b64, mt = encode_image_base64(img_path, max_pixels=max_pixels)
    raw, lat, err = api_call(b64, mt, sys_prompt, USER_PROMPT, api_key, model_id)
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat, err, raw
    try:
        rating = parse_iqa_response(raw)
        return {
            "overall": rating.overall,
            "sharpness": rating.sharpness,
            "color_fidelity": rating.color_fidelity,
        }, lat, "", raw
    except ValueError as e:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat, str(e), raw


# --- Metrics ---


def srcc_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(stats.spearmanr(pred, true).statistic)


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    metric_fn: Any,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute metric with bootstrapped 95% CI."""
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


def compute_arm_metrics(
    checkpoint: dict[str, dict[str, Any]],
    gt_lookup: dict[str, GroundTruth],
) -> dict[str, Any]:
    """Compute full metrics for an arm's checkpoint data."""
    # Build paired arrays (valid results only)
    pairs: list[tuple[dict[str, Any], GroundTruth]] = []
    for img, record in checkpoint.items():
        if record.get("error"):
            continue
        if record.get("overall") is None:
            continue
        gt = gt_lookup.get(img)
        if gt:
            pairs.append((record, gt))

    n_total = len(checkpoint)
    n_valid = len(pairs)
    success_rate = n_valid / n_total if n_total > 0 else 0.0

    metrics: dict[str, Any] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "success_rate": round(success_rate, 4),
    }

    if n_valid < 30:
        print(f"  WARNING: Only {n_valid} valid pairs, skipping metrics")
        return metrics

    dims = ["overall", "sharpness", "color_fidelity"]
    srcc_values = []

    for dim in dims:
        pred = np.array([r[dim] for r, _ in pairs])
        true_vals = np.array([getattr(gt, dim) for _, gt in pairs])

        srcc, srcc_lo, srcc_hi = bootstrap_ci(pred, true_vals, srcc_fn)
        mae = float(np.mean(np.abs(pred - true_vals)))
        bias = float(np.mean(pred - true_vals))

        dim_key = dim.split("_")[0] if dim != "color_fidelity" else "color"
        metrics[f"{dim_key}_srcc"] = round(srcc, 4)
        metrics[f"{dim_key}_srcc_ci"] = [round(srcc_lo, 4), round(srcc_hi, 4)]
        metrics[f"{dim_key}_mae"] = round(mae, 4)
        metrics[f"{dim_key}_bias"] = round(bias, 4)
        srcc_values.append(srcc)

    # Weighted SRCC
    weights = [0.5, 0.25, 0.25]
    metrics["wsrcc"] = round(float(np.average(srcc_values, weights=weights)), 4)

    # Latency
    latencies = [r.get("latency_ms", 0) for r in checkpoint.values() if not r.get("error")]
    if latencies:
        metrics["avg_latency_ms"] = int(np.mean(latencies))

    return metrics


# --- Main execution ---


def run_arm(
    arm_num: int,
    model_id: str,
    gt_list: list[GroundTruth],
    api_key: str,
    sys_prompt: str,
    example_data: list[tuple[str, str, str]] | None,
    limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run a single arm at full scale with checkpoint/resume."""
    arm_suffix, arm_desc = ARM_INFO[arm_num]
    checkpoint = load_checkpoint(model_id, arm_suffix)

    images = gt_list[:limit] if limit else gt_list
    done = len(checkpoint)
    remaining = [gt for gt in images if gt.res_file not in checkpoint]

    print(f"\n{'=' * 70}")
    print(f"ARM {arm_num}: {arm_desc}")
    print(f"Model: {model_id}")
    print(f"Images: {done} done, {len(remaining)} remaining, {len(images)} total")
    print(f"{'=' * 70}")

    if not remaining:
        print("  All images already complete.")
        return checkpoint

    for idx, gt in enumerate(remaining):
        img_path = str(IMAGES_DIR / gt.res_file)
        print(f"  [{done + idx + 1}/{len(images)}] {gt.res_file}", end=" ", flush=True)

        # Encode at 1024 (default) for arms 2-5, 8, 9
        if arm_num in (2, 3, 4, 5, 8, 9):
            b64, mt = encode_image_base64(img_path, max_pixels=1024)

        scores: dict[str, float | None]
        raw = ""
        error = ""

        if arm_num == 2:
            scores, lat, error = run_arm2_separate(b64, mt, api_key, model_id)
        elif arm_num == 3:
            scores, lat, error = run_arm3_hybrid(b64, mt, api_key, sys_prompt, model_id)
        elif arm_num == 4:
            scores, lat, error, raw = run_arm4_few_shot(
                b64, mt, api_key, sys_prompt, model_id, example_data or [],
            )
        elif arm_num == 5:
            scores, lat, error, raw = run_arm5_multi_sample(
                b64, mt, api_key, sys_prompt, model_id,
            )
        elif arm_num == 6:
            scores, lat, error, raw = run_single_prompt_arm(
                img_path, api_key, sys_prompt, model_id, max_pixels=2048,
            )
        elif arm_num == 7:
            scores, lat, error, raw = run_single_prompt_arm(
                img_path, api_key, sys_prompt, model_id, max_pixels=0,
            )
        elif arm_num == 8:
            # Arm 8: 1-10 scale, rescale to 1-5
            sys_10 = build_system_prompt(SCALE_10_CONFIG)
            raw, lat, err = api_call(
                b64, mt, sys_10, USER_PROMPT_SCALE10, api_key, model_id,
            )
            if err:
                scores = {"overall": None, "sharpness": None, "color_fidelity": None}
                error = err
            else:
                try:
                    rating = parse_iqa_response(raw, scale_min=1.0, scale_max=10.0)
                    scores = {
                        "overall": round(rescale_10_to_5(rating.overall), 2),
                        "sharpness": round(rescale_10_to_5(rating.sharpness), 2),
                        "color_fidelity": round(rescale_10_to_5(rating.color_fidelity), 2),
                    }
                except ValueError as e:
                    scores = {"overall": None, "sharpness": None, "color_fidelity": None}
                    error = str(e)
        elif arm_num == 9:
            # Arm 9: 1-5 scale with 0.5 increments
            sys_half = build_system_prompt(HALF_STEP_CONFIG)
            raw, lat, err = api_call(
                b64, mt, sys_half, USER_PROMPT_HALF_STEP, api_key, model_id,
            )
            if err:
                scores = {"overall": None, "sharpness": None, "color_fidelity": None}
                error = err
            else:
                try:
                    rating = parse_iqa_response(raw, scale_min=1.0, scale_max=5.0)
                    scores = {
                        "overall": rating.overall,
                        "sharpness": rating.sharpness,
                        "color_fidelity": rating.color_fidelity,
                    }
                except ValueError as e:
                    scores = {"overall": None, "sharpness": None, "color_fidelity": None}
                    error = str(e)
        else:
            continue

        result = ImageResult(
            model_id=model_id,
            arm=arm_suffix,
            image=gt.res_file,
            overall=scores.get("overall"),
            sharpness=scores.get("sharpness"),
            color_fidelity=scores.get("color_fidelity"),
            reasoning="",
            raw_response=raw[:500] if raw else "",
            latency_ms=lat,
            error=error,
        )

        append_checkpoint(model_id, arm_suffix, result)
        checkpoint[gt.res_file] = asdict(result)

        if scores.get("overall") is not None:
            print(
                f"O={scores['overall']:.1f} S={scores.get('sharpness', 0):.1f} "
                f"C={scores.get('color_fidelity', 0):.1f} ({lat}ms)"
                + (f" ERR: {error[:40]}" if error else "")
            )
        else:
            print(f"FAIL ({lat}ms) {error[:60]}")

        time.sleep(RATE_LIMIT_S)

    return checkpoint


def print_metrics_table(
    all_metrics: dict[str, dict[str, Any]],
    model_id: str,
) -> None:
    """Print a formatted comparison table for all arms."""
    print(f"\n{'=' * 90}")
    print(f"FULL-SCALE PROMPT ARM RESULTS — {model_id}")
    print(f"{'=' * 90}")
    print(
        f"{'Arm':<25s} {'wSRCC':>7s} {'O_SRCC':>7s} {'S_SRCC':>7s} "
        f"{'C_SRCC':>7s} {'O_MAE':>7s} {'O_Bias':>7s} {'Lat':>6s}  "
        f"{'n':>4s} {'rate':>5s}"
    )
    print("-" * 95)

    for arm_key, m in sorted(all_metrics.items()):
        if m.get("n_valid", 0) < 30:
            print(f"{arm_key:<25s} insufficient data (n={m.get('n_valid', 0)})")
            continue
        print(
            f"{arm_key:<25s} "
            f"{m.get('wsrcc', 0):>7.4f} "
            f"{m.get('overall_srcc', 0):>7.4f} "
            f"{m.get('sharpness_srcc', 0):>7.4f} "
            f"{m.get('color_srcc', 0):>7.4f} "
            f"{m.get('overall_mae', 0):>7.4f} "
            f"{m.get('overall_bias', 0):>+7.4f} "
            f"{m.get('avg_latency_ms', 0):>5d}ms "
            f"{m.get('n_valid', 0):>4d} "
            f"{m.get('success_rate', 0):>5.1%}"
        )

    # Ranking
    ranked = sorted(
        [(k, v) for k, v in all_metrics.items() if v.get("wsrcc")],
        key=lambda x: x[1]["wsrcc"],
        reverse=True,
    )
    print(f"\nRANKED BY wSRCC:")
    for i, (arm_key, m) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i + 1}. {arm_key}: wSRCC={m['wsrcc']:.4f}{marker}")


def main() -> None:
    """Run prompt arms at full scale."""
    parser = argparse.ArgumentParser(
        description="Full-scale prompt arm evaluation (n=1000)"
    )
    parser.add_argument(
        "--model", required=True,
        help="OpenRouter model ID (e.g., qwen/qwen3.5-flash-02-23)",
    )
    parser.add_argument(
        "--arm", type=int, action="append", default=None,
        help="Arm number(s) to run (2-7). Can be repeated: --arm 2 --arm 4",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all non-baseline arms (2-7)",
    )
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Compute metrics from existing checkpoints without making API calls",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N images (for dry runs)",
    )
    args = parser.parse_args()

    model_id: str = args.model

    # Determine which arms to run
    if args.all:
        arms_to_run = [2, 3, 4, 5, 6, 7, 8, 9]
    elif args.arm:
        arms_to_run = args.arm
    elif args.metrics_only:
        arms_to_run = []  # Will compute metrics for all available checkpoints
    else:
        parser.error("Specify --arm N, --all, or --metrics-only")
        return

    # Load ground truth
    gt_list = load_ground_truth()
    gt_lookup = {gt.res_file: gt for gt in gt_list}
    print(f"Loaded {len(gt_list)} ground truth entries")

    if not args.metrics_only:
        # Load API key
        load_env()
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)

        sys_prompt = build_system_prompt()

        # Pre-encode few-shot examples
        example_data: list[tuple[str, str, str]] = []
        if 4 in arms_to_run:
            print("Encoding few-shot examples...")
            for ex in FEW_SHOT_EXAMPLES:
                ex_path = str(IMAGES_DIR / ex["file"])
                ex_b64, ex_mt = encode_image_base64(ex_path, max_pixels=1024)
                label = (
                    f'This {ex["label"]} document scores: '
                    f'{{"overall": {ex["overall"]}, "sharpness": {ex["sharpness"]}, '
                    f'"color_fidelity": {ex["color_fidelity"]}}}'
                )
                example_data.append((ex_b64, ex_mt, label))

        # Run arms
        for arm_num in arms_to_run:
            if arm_num not in ARM_INFO:
                print(f"Unknown arm: {arm_num}, skipping")
                continue
            run_arm(
                arm_num, model_id, gt_list, api_key, sys_prompt,
                example_data if arm_num == 4 else None,
                limit=args.limit,
            )

    # --- Compute and display metrics ---
    print(f"\n\nComputing metrics...")

    all_metrics: dict[str, dict[str, Any]] = {}

    # Arm 1 baseline (existing checkpoint, no arm suffix)
    baseline_cp_path = CHECKPOINT_DIR / f"{model_id.replace('/', '__')}.jsonl"
    if baseline_cp_path.exists():
        baseline_data: dict[str, dict[str, Any]] = {}
        for line in baseline_cp_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                baseline_data[item["image"]] = item
            except (json.JSONDecodeError, KeyError):
                continue
        if baseline_data:
            all_metrics["arm1_baseline"] = compute_arm_metrics(baseline_data, gt_lookup)

    # Arms 2-7
    for arm_num, (arm_suffix, _) in ARM_INFO.items():
        cp = load_checkpoint(model_id, arm_suffix)
        if cp:
            all_metrics[arm_suffix] = compute_arm_metrics(cp, gt_lookup)

    if all_metrics:
        print_metrics_table(all_metrics, model_id)

        # Save metrics
        results_path = EVAL_DIR / "results" / f"prompt_arms_{model_id.replace('/', '__')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(all_metrics, indent=2))
        print(f"\nMetrics saved to: {results_path}")
    else:
        print("No checkpoint data found for any arm.")


if __name__ == "__main__":
    main()
