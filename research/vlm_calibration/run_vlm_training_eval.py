"""Run VLM teachers on DIQA-5000 training split (3,500 images).

Generates per-image predictions for Gemini 3 Flash and Qwen 3.5 122B using
optimal per-model settings from the prompt arm experiment:
  - Gemini 3 Flash: 1024px, scale 1-5 (baseline — no benefit from higher res)
  - Qwen 3.5 122B: native resolution, scale 1-10 (rescaled to 1-5 for storage)

Outputs are checkpoint JSONL files for subsequent calibration fitting.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../research/vlm_calibration/run_vlm_training_eval.py

    # Single model:
    ... run_vlm_training_eval.py --model qwen/qwen3.5-122b-a10b

    # Dry run (first 10 images):
    ... run_vlm_training_eval.py --limit 10

    # Metrics only (from checkpoints):
    ... run_vlm_training_eval.py --metrics-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import (
    USER_PROMPT,
    IQAPromptConfig,
    build_system_prompt,
)
from results.vlm_teacher_eval.response_parser import parse_iqa_response

# --- Configuration ---

GCS_BUCKET = (
    "gs://image_detection_b/image-preprocessing-detector/"
    "datasets/diqa-5000/diqa-5000/train"
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
IMAGES_DIR = DATA_DIR / "res"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"

TRAIN_GT_DIR = REPO_ROOT / "DeQA-Score" / "Data-DeQA-Score" / "DIQA" / "metas"
TRAIN_GT_FILES = {
    "overall": TRAIN_GT_DIR / "train_diqa_overall.json",
    "sharpness": TRAIN_GT_DIR / "train_diqa_sharpness.json",
    "color": TRAIN_GT_DIR / "train_diqa_color.json",
}

RATE_LIMIT_S = 0.3

# Scale-10 prompt (matches run_full_prompt_arms.py arm 8)
SCALE_10_CONFIG = IQAPromptConfig(scale_min=1.0, scale_max=10.0, scale_step=0.1)
USER_PROMPT_SCALE10 = """\
Rate the quality of this document image.

Respond with exactly this JSON structure:
{{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}}

Scores should be on a 1.0-10.0 scale. \
The reasoning field should be 1-2 sentences explaining the key quality \
factors you observed. Keep it concise.\
"""


@dataclass(frozen=True)
class ModelConfig:
    """Per-model optimal settings from prompt arm experiment."""

    model_id: str
    tier: str
    max_pixels: int  # 0 = no resize (native resolution)
    use_scale_10: bool  # True = 1-10 scale, rescaled to 1-5


# Optimal settings per model (from prompt arm experiment)
MODEL_CONFIGS: list[ModelConfig] = [
    # Gemini: 1024px ≈ 2048px (no benefit), scale 1-5 baseline
    ModelConfig("google/gemini-3-flash-preview", "Value", max_pixels=1024, use_scale_10=False),
    # Qwen 122B: native > 2048px > 1024px, scale-10 reduces bias by 12%
    ModelConfig("qwen/qwen3.5-122b-a10b", "Strong", max_pixels=0, use_scale_10=True),
]


def rescale_10_to_5(score: float) -> float:
    """Linearly rescale a 1-10 score to 1-5 (matches run_full_prompt_arms.py)."""
    return (score - 1.0) / 9.0 * 4.0 + 1.0


@dataclass(frozen=True)
class TrainImage:
    """Training image with ground truth from all dimensions."""

    res_file: str  # e.g. "train_res_00001.jpg"
    gt_overall: float
    gt_sharpness: float
    gt_color: float


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


# --- Data Loading ---


def download_train_data() -> None:
    """Download DIQA-5000 training images from GCS."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if IMAGES_DIR.exists():
        n_images = len(list(IMAGES_DIR.glob("*.jpg")))
        if n_images >= 3500:
            print(f"Train data already downloaded ({n_images} images)")
            return

    print("Downloading DIQA-5000 training images from GCS...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"  Downloaded {n_images} training images")


def load_train_images() -> list[TrainImage]:
    """Load training ground truth from all 3 dimension JSON files.

    Merges the per-dimension files by image basename and returns a sorted
    list of TrainImage objects.
    """
    merged: dict[str, dict[str, float]] = {}

    for dim, gt_path in TRAIN_GT_FILES.items():
        data = json.loads(gt_path.read_text())
        for item in data:
            key = Path(item["image"]).name  # train_res_NNNNN.jpg
            if key not in merged:
                merged[key] = {}
            merged[key][dim] = item["gt_score"]

    images: list[TrainImage] = []
    for res_file in sorted(merged.keys()):
        scores = merged[res_file]
        if len(scores) == 3:
            images.append(
                TrainImage(
                    res_file=res_file,
                    gt_overall=scores["overall"],
                    gt_sharpness=scores["sharpness"],
                    gt_color=scores["color"],
                )
            )

    return images


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
) -> tuple[str, int, str]:
    """Rate an image via OpenRouter with retry logic.

    Returns:
        Tuple of (raw_text, latency_ms, error).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,  # 2-minute timeout to avoid hung connections
    )

    for attempt in range(max_retries):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_id,
                temperature=0.0,
                max_tokens=1024,
                messages=[
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
            )
            latency_ms = int((time.time() - start) * 1000)
            raw_text = response.choices[0].message.content or ""
            return raw_text, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            err_str = str(exc)

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


def checkpoint_path(model_id: str) -> Path:
    """Get checkpoint file path for a model."""
    safe_name = model_id.replace("/", "__")
    return CHECKPOINT_DIR / f"{safe_name}__train.jsonl"


def load_checkpoint(model_id: str) -> dict[str, dict[str, Any]]:
    """Load existing checkpoint results for a model.

    Returns:
        Dict mapping image filename to result dict.
    """
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
    """Append a single result to the model's checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp = checkpoint_path(model_id)
    with cp.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# --- Main ---


def evaluate_model(
    config: ModelConfig,
    train_images: list[TrainImage],
    api_key: str,
    limit: int | None = None,
) -> list[ImageResult]:
    """Evaluate a single model on training images with resume support."""
    model_id = config.model_id
    existing = load_checkpoint(model_id)
    images_to_eval = train_images[:limit] if limit else train_images
    total = len(images_to_eval)

    # Build prompts based on model config
    if config.use_scale_10:
        system_prompt = build_system_prompt(SCALE_10_CONFIG)
        user_prompt = USER_PROMPT_SCALE10
        parse_scale_max = 10.0
        print(f"  Config: max_pixels={config.max_pixels}, scale=1-10 (rescaled to 1-5)")
    else:
        system_prompt = build_system_prompt()
        user_prompt = USER_PROMPT
        parse_scale_max = 5.0
        print(f"  Config: max_pixels={config.max_pixels}, scale=1-5")

    results: list[ImageResult] = []
    skipped = 0

    for idx, img in enumerate(images_to_eval):
        # Resume from checkpoint
        if img.res_file in existing:
            r = ImageResult(**existing[img.res_file])
            results.append(r)
            skipped += 1
            continue

        if skipped and idx == skipped:
            print(f"  Resumed from checkpoint ({skipped} cached)")

        img_path = str(IMAGES_DIR / img.res_file)
        if not Path(img_path).exists():
            print(f"  [{idx + 1}/{total}] MISSING: {img.res_file}")
            continue

        print(
            f"  [{idx + 1}/{total}] {img.res_file}",
            end=" ",
            flush=True,
        )

        img_b64, media_type = encode_image_base64(
            img_path, max_pixels=config.max_pixels,
        )

        raw_text, latency_ms, error = rate_image_openrouter(
            model_id=model_id,
            image_b64=img_b64,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
        )

        if error:
            r = ImageResult(
                model_id=model_id,
                image=img.res_file,
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
                rating = parse_iqa_response(
                    raw_text, scale_max=parse_scale_max,
                )

                # Rescale 1-10 → 1-5 if needed (stored values always on 1-5)
                if config.use_scale_10:
                    overall = round(rescale_10_to_5(rating.overall), 2)
                    sharpness = round(rescale_10_to_5(rating.sharpness), 2)
                    color_fidelity = round(rescale_10_to_5(rating.color_fidelity), 2)
                else:
                    overall = rating.overall
                    sharpness = rating.sharpness
                    color_fidelity = rating.color_fidelity

                r = ImageResult(
                    model_id=model_id,
                    image=img.res_file,
                    overall=overall,
                    sharpness=sharpness,
                    color_fidelity=color_fidelity,
                    reasoning=rating.reasoning,
                    raw_response=raw_text,
                    latency_ms=latency_ms,
                    error="",
                )
                print(
                    f"O={overall:.1f} S={sharpness:.1f} "
                    f"C={color_fidelity:.1f} ({latency_ms}ms)"
                )
            except ValueError as exc:
                r = ImageResult(
                    model_id=model_id,
                    image=img.res_file,
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
        append_checkpoint(model_id, r)
        time.sleep(RATE_LIMIT_S)

    if skipped == total:
        print(f"  All {total} images cached from checkpoint")

    return results


def print_summary(results: list[ImageResult], train_images: list[TrainImage]) -> None:
    """Print quick summary statistics for completed evaluation."""
    ok = [r for r in results if not r.error and r.overall is not None]
    err = [r for r in results if r.error]

    print(f"\n  Results: {len(ok)} success, {len(err)} errors")

    if len(ok) < 30:
        return

    # Build GT lookup
    gt_lookup = {img.res_file: img for img in train_images}

    for dim_name, pred_attr, gt_attr in [
        ("Overall", "overall", "gt_overall"),
        ("Sharpness", "sharpness", "gt_sharpness"),
        ("Color", "color_fidelity", "gt_color"),
    ]:
        preds = []
        gts = []
        for r in ok:
            gt = gt_lookup.get(r.image)
            if gt:
                preds.append(getattr(r, pred_attr))
                gts.append(getattr(gt, gt_attr))

        pred_arr = np.array(preds)
        gt_arr = np.array(gts)
        mae = float(np.mean(np.abs(pred_arr - gt_arr)))
        bias = float(np.mean(pred_arr - gt_arr))
        print(f"  {dim_name}: MAE={mae:.3f}, bias={bias:+.3f}, "
              f"pred range=[{pred_arr.min():.1f}, {pred_arr.max():.1f}]")


def main() -> None:
    """Run VLM evaluation on DIQA-5000 training split."""
    parser = argparse.ArgumentParser(
        description="VLM teacher evaluation on DIQA-5000 training split"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate only this model (e.g. qwen/qwen3.5-122b-a10b)",
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
        help="Print summary from existing checkpoints without API calls",
    )
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.metrics_only:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    # Download training images
    if not args.metrics_only:
        download_train_data()

    # Load training ground truth
    train_images = load_train_images()
    print(f"Loaded {len(train_images)} training images with GT")

    # Select models
    configs = MODEL_CONFIGS
    if args.model:
        configs = [c for c in MODEL_CONFIGS if c.model_id == args.model]
        if not configs:
            print(f"ERROR: Model '{args.model}' not in MODEL_CONFIGS")
            sys.exit(1)

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {config.model_id} (tier={config.tier})")
        print(f"{'=' * 80}")

        if args.metrics_only:
            existing = load_checkpoint(config.model_id)
            if not existing:
                print("  No checkpoint found, skipping")
                continue
            results = [ImageResult(**v) for v in existing.values()]
            print(f"  Loaded {len(results)} results from checkpoint")
        else:
            results = evaluate_model(
                config=config,
                train_images=train_images,
                api_key=api_key,
                limit=args.limit,
            )

        print_summary(results, train_images)


if __name__ == "__main__":
    main()
