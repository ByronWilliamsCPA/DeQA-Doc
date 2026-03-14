"""A/B test: 1-prompt (all 3 scores) vs 3-prompt (one score each).

Tests whether asking for each quality dimension separately improves
correlation with human MOS, especially for color_fidelity where models
tend to anchor high when rating all dimensions together.

Design:
  - 50 stratified images (10 per quality bucket)
  - 2 models: gemini-3-flash-preview (best overall), gpt-4.1 (best smoke)
  - Condition A: Current 1-prompt approach (baseline, reuse checkpoint data)
  - Condition B: 3 separate prompts, one per dimension

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/run_prompt_ab_test.py
"""

from __future__ import annotations

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

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
IMAGES_DIR = DATA_DIR / "res"
TEST_CSV = DATA_DIR / "test.csv"
AB_DIR = EVAL_DIR / "ab_test"

# Models to compare
AB_MODELS = [
    "google/gemini-3-flash-preview",
    "openai/gpt-4.1",
]

# Number of images per quality bucket (5 buckets × 10 = 50 images)
IMAGES_PER_BUCKET = 10

# --- Dimension-Specific Prompts ---

SINGLE_DIM_SYSTEM = """\
You are an expert document image quality assessor. You evaluate scanned or \
photographed document images for visual quality as perceived by a human reader.

You will rate documents on a single specific quality dimension using a 1.0-5.0 \
scale with 0.1 increments.

Scale anchors:
- 1.0: Completely unusable / severe degradation
- 2.0: Poor — significant issues
- 3.0: Fair — acceptable but with noticeable problems
- 4.0: Good — minor issues, generally readable
- 5.0: Excellent — crisp, clean, high-quality reproduction

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.\
"""

PROMPT_OVERALL = """\
Rate the OVERALL QUALITY of this document image.

Consider holistic readability and usability — could a human comfortably read \
this document? Factor in text clarity, layout preservation, contrast, and \
any artifacts or degradation.

Respond with exactly this JSON structure:
{{"score": X.X, "reasoning": "..."}}

The reasoning field should be 1-2 sentences about the key quality factors.\
"""

PROMPT_SHARPNESS = """\
Rate the SHARPNESS of this document image.

Focus specifically on text edge clarity, blur level, and resolution adequacy. \
Are characters crisp and well-defined, or soft and blurred? Ignore color \
issues — only assess sharpness/blur/resolution.

Respond with exactly this JSON structure:
{{"score": X.X, "reasoning": "..."}}

The reasoning field should be 1-2 sentences about sharpness specifically.\
"""

PROMPT_COLOR = """\
Rate the COLOR FIDELITY of this document image.

Focus specifically on color accuracy, contrast, white balance, and tonal \
reproduction. Are colors natural? Is the page background clean white or \
yellowed/tinted? Is there color fringing or chromatic aberration? Ignore \
blur — only assess color and tone.

Respond with exactly this JSON structure:
{{"score": X.X, "reasoning": "..."}}

The reasoning field should be 1-2 sentences about color fidelity specifically.\
"""

DIMENSIONS = [
    ("overall", PROMPT_OVERALL),
    ("sharpness", PROMPT_SHARPNESS),
    ("color_fidelity", PROMPT_COLOR),
]


@dataclass(frozen=True)
class GroundTruth:
    """DIQA-5000 ground truth."""

    res_file: str
    overall: float
    sharpness: float
    color_fidelity: float


@dataclass(frozen=True)
class ABResult:
    """Result from a single A/B evaluation."""

    model_id: str
    image: str
    condition: str  # "1prompt" or "3prompt"
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning_overall: str
    reasoning_sharpness: str
    reasoning_color: str
    total_latency_ms: int
    error: str


# --- Sample Selection ---


def select_stratified_sample(n_per_bucket: int = IMAGES_PER_BUCKET) -> list[GroundTruth]:
    """Select stratified sample: n images per quality bucket.

    Buckets based on overall MOS:
      bad [1.0, 1.8), poor [1.8, 2.6), fair [2.6, 3.4),
      good [3.4, 4.0), excellent [4.0, 5.0]
    """
    all_gt: list[GroundTruth] = []
    with TEST_CSV.open() as f:
        for row in csv.DictReader(f):
            all_gt.append(GroundTruth(
                res_file=row["res"],
                overall=float(row["overall"]),
                sharpness=float(row["sharpness"]),
                color_fidelity=float(row["color_fidelity"]),
            ))

    buckets = {
        "bad": (1.0, 1.8),
        "poor": (1.8, 2.6),
        "fair": (2.6, 3.4),
        "good": (3.4, 4.0),
        "excellent": (4.0, 5.01),
    }

    rng = np.random.RandomState(42)
    selected: list[GroundTruth] = []

    for bucket_name, (lo, hi) in buckets.items():
        candidates = [g for g in all_gt if lo <= g.overall < hi]
        if len(candidates) < n_per_bucket:
            print(f"  WARNING: {bucket_name} has only {len(candidates)} images")
            chosen = candidates
        else:
            idx = rng.choice(len(candidates), size=n_per_bucket, replace=False)
            chosen = [candidates[i] for i in idx]
        selected.extend(chosen)
        print(f"  {bucket_name}: {len(chosen)} images (from {len(candidates)})")

    return selected


# --- API Calls ---


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


def call_openrouter(
    model_id: str,
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> tuple[str, int, str]:
    """Single OpenRouter API call. Returns (text, latency_ms, error)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            temperature=0.0,
            max_tokens=512,
            messages=[
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
        )
        latency = int((time.time() - start) * 1000)
        return response.choices[0].message.content or "", latency, ""
    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        return "", latency, str(exc)


def parse_single_score(raw: str) -> tuple[float | None, str]:
    """Parse a single-dimension JSON response. Returns (score, reasoning)."""
    import json as _json

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    try:
        d = _json.loads(cleaned)
        score = float(d["score"])
        if not (1.0 <= score <= 5.0):
            return None, f"out of range: {score}"
        return score, d.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return None, f"parse error: {exc}"


def evaluate_3prompt(
    model_id: str,
    image_b64: str,
    media_type: str,
    api_key: str,
) -> tuple[dict[str, float | None], dict[str, str], int, str]:
    """Run 3 separate prompts for one image.

    Returns:
        (scores_dict, reasoning_dict, total_latency_ms, error)
    """
    scores: dict[str, float | None] = {}
    reasoning: dict[str, str] = {}
    total_latency = 0

    for dim_name, dim_prompt in DIMENSIONS:
        raw, latency, error = call_openrouter(
            model_id, image_b64, media_type,
            SINGLE_DIM_SYSTEM, dim_prompt, api_key,
        )
        total_latency += latency

        if error:
            return {}, {}, total_latency, f"{dim_name}: {error}"

        score, reason = parse_single_score(raw)
        if score is None:
            return {}, {}, total_latency, f"{dim_name}: {reason}"

        scores[dim_name] = score
        reasoning[dim_name] = reason
        time.sleep(0.2)

    return scores, reasoning, total_latency, ""


# --- Metrics ---


def compute_dim_metrics(
    pred: list[float], true: list[float],
) -> dict[str, float]:
    """Compute SRCC, PLCC, MAE for a dimension."""
    p, t = np.array(pred), np.array(true)
    return {
        "srcc": round(float(stats.spearmanr(p, t).statistic), 4),
        "plcc": round(float(stats.pearsonr(p, t).statistic), 4),
        "mae": round(float(np.mean(np.abs(p - t))), 4),
        "bias": round(float(np.mean(p - t)), 4),
    }


# --- Main ---


def main() -> None:
    """Run the A/B test."""
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    AB_DIR.mkdir(parents=True, exist_ok=True)

    print("Selecting stratified sample...")
    sample = select_stratified_sample()
    print(f"Selected {len(sample)} images\n")

    # Load 1-prompt results from full eval checkpoints
    checkpoint_dir = EVAL_DIR / "checkpoints"

    for model_id in AB_MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_id}")
        print(f"{'=' * 70}")

        # --- Condition A: 1-prompt (from checkpoint) ---
        safe_name = model_id.replace("/", "__")
        cp_path = checkpoint_dir / f"{safe_name}.jsonl"

        one_prompt: dict[str, dict] = {}
        if cp_path.exists():
            for line in cp_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not item.get("error"):
                        one_prompt[item["image"]] = item
                except json.JSONDecodeError:
                    continue

        print(f"\n  Condition A (1-prompt): {len(one_prompt)} cached results")

        # --- Condition B: 3-prompt (run now) ---
        ab_checkpoint = AB_DIR / f"{safe_name}_3prompt.jsonl"
        existing_3p: dict[str, dict] = {}
        if ab_checkpoint.exists():
            for line in ab_checkpoint.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not item.get("error"):
                        existing_3p[item["image"]] = item
                except json.JSONDecodeError:
                    continue

        print(f"  Condition B (3-prompt): {len(existing_3p)} cached results")
        print(f"  Running 3-prompt evaluations...")

        three_prompt_results: dict[str, dict] = dict(existing_3p)

        for idx, gt in enumerate(sample):
            if gt.res_file in three_prompt_results:
                continue

            img_path = IMAGES_DIR / gt.res_file
            if not img_path.exists():
                print(f"    [{idx+1}/{len(sample)}] MISSING: {gt.res_file}")
                continue

            print(f"    [{idx+1}/{len(sample)}] {gt.res_file}", end=" ", flush=True)
            img_b64, media_type = encode_image_base64(str(img_path))

            scores, reasoning, latency, error = evaluate_3prompt(
                model_id, img_b64, media_type, api_key,
            )

            if error:
                print(f"ERROR: {error[:60]}")
                result = {
                    "image": gt.res_file,
                    "overall": None, "sharpness": None, "color_fidelity": None,
                    "reasoning_overall": "", "reasoning_sharpness": "",
                    "reasoning_color": "",
                    "latency_ms": latency, "error": error,
                }
            else:
                result = {
                    "image": gt.res_file,
                    "overall": scores["overall"],
                    "sharpness": scores["sharpness"],
                    "color_fidelity": scores["color_fidelity"],
                    "reasoning_overall": reasoning.get("overall", ""),
                    "reasoning_sharpness": reasoning.get("sharpness", ""),
                    "reasoning_color": reasoning.get("color_fidelity", ""),
                    "latency_ms": latency, "error": "",
                }
                print(
                    f"O={scores['overall']:.1f} S={scores['sharpness']:.1f} "
                    f"C={scores['color_fidelity']:.1f} ({latency}ms)"
                )

            three_prompt_results[gt.res_file] = result

            # Checkpoint
            with ab_checkpoint.open("a") as f:
                f.write(json.dumps(result) + "\n")

            time.sleep(0.3)

        # --- Compare A vs B ---
        print(f"\n  --- Comparison: {model_id} ---")
        print(f"  {'Dim':<18s} {'Cond':<10s} {'SRCC':>7s} {'PLCC':>7s} "
              f"{'MAE':>7s} {'Bias':>7s}  n")
        print("  " + "-" * 65)

        for dim in ("overall", "sharpness", "color_fidelity"):
            gt_key = dim

            # 1-prompt
            pred_1p, true_1p = [], []
            for gt in sample:
                if gt.res_file in one_prompt:
                    r = one_prompt[gt.res_file]
                    if r.get(dim) is not None:
                        pred_1p.append(r[dim])
                        true_1p.append(getattr(gt, gt_key))

            # 3-prompt
            pred_3p, true_3p = [], []
            for gt in sample:
                if gt.res_file in three_prompt_results:
                    r = three_prompt_results[gt.res_file]
                    if r.get(dim) is not None:
                        pred_3p.append(r[dim])
                        true_3p.append(getattr(gt, gt_key))

            if len(pred_1p) >= 10:
                m1 = compute_dim_metrics(pred_1p, true_1p)
                print(
                    f"  {dim:<18s} {'1-prompt':<10s} {m1['srcc']:>7.4f} "
                    f"{m1['plcc']:>7.4f} {m1['mae']:>7.4f} {m1['bias']:>+7.4f}  "
                    f"{len(pred_1p)}"
                )

            if len(pred_3p) >= 10:
                m3 = compute_dim_metrics(pred_3p, true_3p)
                print(
                    f"  {dim:<18s} {'3-prompt':<10s} {m3['srcc']:>7.4f} "
                    f"{m3['plcc']:>7.4f} {m3['mae']:>7.4f} {m3['bias']:>+7.4f}  "
                    f"{len(pred_3p)}"
                )

                # Delta
                if len(pred_1p) >= 10:
                    d_srcc = m3["srcc"] - m1["srcc"]
                    d_mae = m3["mae"] - m1["mae"]
                    better = "3p" if d_srcc > 0 else "1p"
                    print(
                        f"  {dim:<18s} {'delta':<10s} {d_srcc:>+7.4f} "
                        f"{'':>7s} {d_mae:>+7.4f} {'':>7s}  "
                        f"winner={better}"
                    )
            print()

    # Save summary
    print(f"\nResults saved to: {AB_DIR}/")


if __name__ == "__main__":
    main()
