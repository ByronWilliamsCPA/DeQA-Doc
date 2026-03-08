"""7-arm prompt optimization trial for VLM models via OpenRouter.

Tests different prompting strategies on 25 stratified images to find
the best approach for pseudo-label generation.

Arms:
  1. single_all3    - Single prompt, all 3 dimensions (current baseline)
  2. separate_3     - 3 separate prompts, one per dimension
  3. hybrid         - 1 prompt for overall, separate for sharpness & color
  4. few_shot       - Single prompt + 3 example images as calibration anchors
  5. multi_sample   - 3 calls with temp=0.3, take median
  6. res_2048       - Single prompt, resize to 2048px instead of 1024
  7. no_resize      - Single prompt, no image resizing (full resolution)

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/run_prompt_optimization.py \
        --model qwen/qwen3.5-flash-02-23
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
IMAGES_DIR = DATA_DIR / "res"
TEST_CSV = DATA_DIR / "test.csv"
OPT_DIR = EVAL_DIR / "prompt_optimization"

DEFAULT_MODEL = "google/gemini-3-flash-preview"
IMAGES_PER_BUCKET = 5  # 5 buckets × 5 = 25 images

# Few-shot example images (bad=1.5, fair=3.0, excellent=4.05)
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

# --- Image encoding ---


def encode_image(path: str, max_pixels: int = 1024) -> tuple[str, str]:
    """Encode image to base64 with resizing."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max_pixels and max(w, h) > max_pixels:
        scale = max_pixels / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def encode_image_no_resize(path: str) -> tuple[str, str]:
    """Encode image at full resolution."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


# --- Sample selection ---


def select_sample() -> list[dict]:
    """Select 25 stratified images (5 per quality bucket)."""
    all_rows: list[dict] = []
    with TEST_CSV.open() as f:
        for row in csv.DictReader(f):
            all_rows.append({
                "file": row["res"],
                "gt_overall": float(row["overall"]),
                "gt_sharpness": float(row["sharpness"]),
                "gt_color": float(row["color_fidelity"]),
            })

    buckets = [
        ("bad", 1.0, 1.8),
        ("poor", 1.8, 2.6),
        ("fair", 2.6, 3.4),
        ("good", 3.4, 4.0),
        ("excellent", 4.0, 5.01),
    ]

    # Use different seed from A/B test to get different images
    rng = np.random.RandomState(123)
    selected: list[dict] = []

    for bname, lo, hi in buckets:
        candidates = [r for r in all_rows if lo <= r["gt_overall"] < hi]
        # Exclude few-shot examples
        candidates = [r for r in candidates if r["file"] not in
                      {e["file"] for e in FEW_SHOT_EXAMPLES}]
        n = min(IMAGES_PER_BUCKET, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        for i in idx:
            selected.append(candidates[i])

    return selected


# --- API calls ---


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


def api_call(
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    extra_images: list[tuple[str, str, str]] | None = None,
) -> tuple[str, int, str]:
    """Make an OpenRouter API call. Returns (text, latency_ms, error)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Build user content
    user_content: list[dict] = []

    # Add few-shot example images if provided
    if extra_images:
        for ex_b64, ex_mt, ex_label in extra_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{ex_mt};base64,{ex_b64}"},
            })
            user_content.append({
                "type": "text",
                "text": ex_label,
            })

    # Add the target image
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
    })
    user_content.append({"type": "text", "text": user_prompt})

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
        return "", latency, str(exc)


def parse_scores(raw: str) -> dict[str, float | None]:
    """Parse JSON response to extract scores."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    try:
        d = json.loads(cleaned)
        result: dict[str, float | None] = {}
        for key in ("overall", "sharpness", "color_fidelity"):
            v = d.get(key)
            if v is not None:
                v = float(v)
                if 1.0 <= v <= 5.0:
                    result[key] = v
                else:
                    result[key] = None
            else:
                result[key] = None
        return result
    except (json.JSONDecodeError, ValueError):
        return {"overall": None, "sharpness": None, "color_fidelity": None}


def parse_single_score(raw: str) -> float | None:
    """Parse a single-dimension JSON response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    try:
        d = json.loads(cleaned)
        v = float(d["score"])
        return v if 1.0 <= v <= 5.0 else None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# --- Arm implementations ---

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


def run_arm1_single_all3(
    img_b64: str, mt: str, api_key: str, sys_prompt: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 1: Single prompt, all 3 dimensions."""
    raw, lat, err = api_call(img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id=model_id)
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat
    return parse_scores(raw), lat


def run_arm2_separate(
    img_b64: str, mt: str, api_key: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 2: 3 separate prompts."""
    scores: dict[str, float | None] = {}
    total_lat = 0
    for dim, prompt in DIM_PROMPTS.items():
        raw, lat, err = api_call(img_b64, mt, SINGLE_DIM_SYSTEM, prompt, api_key, model_id=model_id)
        total_lat += lat
        if err:
            scores[dim] = None
        else:
            scores[dim] = parse_single_score(raw)
        time.sleep(0.2)
    return scores, total_lat


def run_arm3_hybrid(
    img_b64: str, mt: str, api_key: str, sys_prompt: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 3: Combined for overall, separate for sharpness & color."""
    # Overall via combined prompt
    raw, lat1, err = api_call(img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id=model_id)
    scores: dict[str, float | None] = {}
    if err:
        scores["overall"] = None
    else:
        parsed = parse_scores(raw)
        scores["overall"] = parsed.get("overall")

    # Sharpness separate
    time.sleep(0.2)
    raw, lat2, err = api_call(
        img_b64, mt, SINGLE_DIM_SYSTEM, DIM_PROMPTS["sharpness"], api_key, model_id=model_id,
    )
    scores["sharpness"] = parse_single_score(raw) if not err else None

    # Color separate
    time.sleep(0.2)
    raw, lat3, err = api_call(
        img_b64, mt, SINGLE_DIM_SYSTEM, DIM_PROMPTS["color_fidelity"], api_key, model_id=model_id,
    )
    scores["color_fidelity"] = parse_single_score(raw) if not err else None

    return scores, lat1 + lat2 + lat3


def run_arm4_few_shot(
    img_b64: str, mt: str, api_key: str, sys_prompt: str,
    example_data: list[tuple[str, str, str]],
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 4: Single prompt + 3 few-shot examples."""
    raw, lat, err = api_call(
        img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id=model_id,
        extra_images=example_data,
    )
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat
    return parse_scores(raw), lat


def run_arm5_multi_sample(
    img_b64: str, mt: str, api_key: str, sys_prompt: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 5: 3 calls with temp=0.3, take median."""
    all_scores: dict[str, list[float]] = {"overall": [], "sharpness": [], "color_fidelity": []}
    total_lat = 0

    for _ in range(3):
        raw, lat, err = api_call(
            img_b64, mt, sys_prompt, USER_PROMPT, api_key, model_id=model_id, temperature=0.3,
        )
        total_lat += lat
        if not err:
            parsed = parse_scores(raw)
            for dim in all_scores:
                v = parsed.get(dim)
                if v is not None:
                    all_scores[dim].append(v)
        time.sleep(0.2)

    result: dict[str, float | None] = {}
    for dim, vals in all_scores.items():
        result[dim] = float(np.median(vals)) if vals else None
    return result, total_lat


def run_arm6_res2048(
    img_path: str, mt: str, api_key: str, sys_prompt: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 6: Single prompt, 2048px resize."""
    b64, mt2 = encode_image(img_path, max_pixels=2048)
    raw, lat, err = api_call(b64, mt2, sys_prompt, USER_PROMPT, api_key, model_id=model_id)
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat
    return parse_scores(raw), lat


def run_arm7_no_resize(
    img_path: str, mt: str, api_key: str, sys_prompt: str,
    model_id: str = DEFAULT_MODEL,
) -> tuple[dict[str, float | None], int]:
    """Arm 7: Single prompt, no resizing."""
    b64, mt2 = encode_image_no_resize(img_path)
    raw, lat, err = api_call(b64, mt2, sys_prompt, USER_PROMPT, api_key, model_id=model_id)
    if err:
        return {"overall": None, "sharpness": None, "color_fidelity": None}, lat
    return parse_scores(raw), lat


# --- Metrics ---


def compute_metrics(
    pred: list[float], true: list[float],
) -> dict[str, float]:
    """Compute SRCC, MAE, bias."""
    p, t = np.array(pred), np.array(true)
    return {
        "srcc": round(float(stats.spearmanr(p, t).statistic), 4),
        "mae": round(float(np.mean(np.abs(p - t))), 4),
        "bias": round(float(np.mean(p - t)), 4),
    }


# --- Main ---


def main() -> None:
    """Run all 7 arms."""
    parser = argparse.ArgumentParser(description="7-arm prompt optimization trial")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    model_id: str = args.model

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Model-specific output directory (slashes replaced with dunders)
    model_slug = model_id.replace("/", "__")
    opt_dir = OPT_DIR / model_slug
    opt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {model_id}")
    print(f"Output: {opt_dir}")

    # Select sample
    print("Selecting 25 stratified images...")
    sample = select_sample()
    print(f"Selected {len(sample)} images\n")

    sys_prompt = build_system_prompt()

    # Pre-encode few-shot examples at 1024px
    print("Encoding few-shot examples...")
    example_data: list[tuple[str, str, str]] = []
    for ex in FEW_SHOT_EXAMPLES:
        ex_path = str(IMAGES_DIR / ex["file"])
        ex_b64, ex_mt = encode_image(ex_path, max_pixels=1024)
        label = (
            f'This {ex["label"]} document scores: '
            f'{{"overall": {ex["overall"]}, "sharpness": {ex["sharpness"]}, '
            f'"color_fidelity": {ex["color_fidelity"]}}}'
        )
        example_data.append((ex_b64, ex_mt, label))

    # Define arms
    arms = [
        "1_single_all3",
        "2_separate_3",
        "3_hybrid",
        "4_few_shot",
        "5_multi_sample",
        "6_res_2048",
        "7_no_resize",
    ]

    # Checkpoint
    cp_path = opt_dir / "checkpoint.json"
    checkpoint: dict[str, dict[str, dict]] = {}
    if cp_path.exists():
        checkpoint = json.loads(cp_path.read_text())
        print(f"Loaded checkpoint ({sum(len(v) for v in checkpoint.values())} entries)")

    for arm in arms:
        if arm not in checkpoint:
            checkpoint[arm] = {}

        print(f"\n{'=' * 60}")
        print(f"ARM: {arm}")
        print(f"{'=' * 60}")

        for idx, img_info in enumerate(sample):
            img_file = img_info["file"]

            # Skip if cached
            if img_file in checkpoint[arm]:
                continue

            img_path = str(IMAGES_DIR / img_file)
            print(f"  [{idx+1}/{len(sample)}] {img_file}", end=" ", flush=True)

            # Pre-encode at 1024 (default) for most arms
            b64_1024, mt = encode_image(img_path, max_pixels=1024)

            if arm == "1_single_all3":
                scores, lat = run_arm1_single_all3(b64_1024, mt, api_key, sys_prompt, model_id)
            elif arm == "2_separate_3":
                scores, lat = run_arm2_separate(b64_1024, mt, api_key, model_id)
            elif arm == "3_hybrid":
                scores, lat = run_arm3_hybrid(b64_1024, mt, api_key, sys_prompt, model_id)
            elif arm == "4_few_shot":
                scores, lat = run_arm4_few_shot(
                    b64_1024, mt, api_key, sys_prompt, example_data, model_id,
                )
            elif arm == "5_multi_sample":
                scores, lat = run_arm5_multi_sample(b64_1024, mt, api_key, sys_prompt, model_id)
            elif arm == "6_res_2048":
                scores, lat = run_arm6_res2048(img_path, mt, api_key, sys_prompt, model_id)
            elif arm == "7_no_resize":
                scores, lat = run_arm7_no_resize(img_path, mt, api_key, sys_prompt, model_id)
            else:
                continue

            checkpoint[arm][img_file] = {
                "scores": scores,
                "latency_ms": lat,
                "gt": img_info,
            }

            s = scores
            if s.get("overall") is not None:
                print(
                    f"O={s['overall']:.1f} S={s.get('sharpness', 0):.1f} "
                    f"C={s.get('color_fidelity', 0):.1f} ({lat}ms)"
                )
            else:
                print(f"FAIL ({lat}ms)")

            # Save checkpoint
            cp_path.write_text(json.dumps(checkpoint, indent=2))
            time.sleep(0.3)

        cached = sum(1 for f in sample if f["file"] in checkpoint[arm])
        print(f"  {cached}/{len(sample)} complete")

    # --- Results comparison ---
    print(f"\n\n{'=' * 80}")
    print(f"PROMPT OPTIMIZATION RESULTS — {model_id} (n=25)")
    print(f"{'=' * 80}")
    print(
        f"{'Arm':<22s} {'O_SRCC':>7s} {'S_SRCC':>7s} {'C_SRCC':>7s} "
        f"{'wSRCC':>7s} {'O_MAE':>7s} {'O_Bias':>7s} {'Lat':>7s}  n"
    )
    print("-" * 85)

    arm_results: dict[str, dict] = {}

    for arm in arms:
        data = checkpoint.get(arm, {})
        dims_data: dict[str, tuple[list[float], list[float]]] = {
            "overall": ([], []),
            "sharpness": ([], []),
            "color_fidelity": ([], []),
        }
        latencies: list[int] = []

        for img_file, entry in data.items():
            s = entry["scores"]
            gt = entry["gt"]
            for dim in dims_data:
                gt_key = f"gt_{dim}" if dim != "color_fidelity" else "gt_color"
                if s.get(dim) is not None:
                    dims_data[dim][0].append(s[dim])
                    dims_data[dim][1].append(gt[gt_key])
            latencies.append(entry["latency_ms"])

        n_valid = len(dims_data["overall"][0])
        if n_valid < 5:
            print(f"{arm:<22s} insufficient data (n={n_valid})")
            continue

        metrics: dict[str, dict] = {}
        for dim, (pred, true) in dims_data.items():
            if len(pred) >= 5:
                metrics[dim] = compute_metrics(pred, true)

        o = metrics.get("overall", {})
        s = metrics.get("sharpness", {})
        c = metrics.get("color_fidelity", {})

        wsrcc = (
            0.5 * o.get("srcc", 0)
            + 0.25 * s.get("srcc", 0)
            + 0.25 * c.get("srcc", 0)
        )

        avg_lat = int(np.mean(latencies)) if latencies else 0

        print(
            f"{arm:<22s} "
            f"{o.get('srcc', 0):>7.4f} {s.get('srcc', 0):>7.4f} "
            f"{c.get('srcc', 0):>7.4f} {wsrcc:>7.4f} "
            f"{o.get('mae', 0):>7.4f} {o.get('bias', 0):>+7.4f} "
            f"{avg_lat:>6d}ms  {n_valid}"
        )

        arm_results[arm] = {
            "metrics": metrics,
            "wsrcc": round(wsrcc, 4),
            "n": n_valid,
            "avg_latency_ms": avg_lat,
        }

    # Save results
    results_path = opt_dir / "optimization_results.json"
    results_path.write_text(json.dumps(arm_results, indent=2))
    print(f"\nResults saved to: {results_path}")

    # Rank
    print(f"\nRANKED BY wSRCC:")
    ranked = sorted(arm_results.items(), key=lambda x: x[1]["wsrcc"], reverse=True)
    for i, (arm, r) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {arm}: wSRCC={r['wsrcc']:.4f}{marker}")


if __name__ == "__main__":
    main()
