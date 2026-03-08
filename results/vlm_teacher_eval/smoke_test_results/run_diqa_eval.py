"""Multi-model IQA evaluation on DIQA-5000 sample images.

Evaluates 26 VLM models (including Anthropic Opus/Sonnet/Haiku) on 7 DIQA
test images spanning all 15 quality categories (5 levels × 3 dimensions).

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/smoke_test_results/run_diqa_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt
from results.vlm_teacher_eval.response_parser import parse_iqa_response


@dataclass(frozen=True)
class ImageResult:
    """Result from a single model × image evaluation."""

    model_id: str
    image: str
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning: str
    raw_response: str
    latency_ms: int
    error: str


# DIQA-5000 test images with human MOS ground truth
# Selected to cover all 15 categories (5 quality levels × 3 dimensions)
SAMPLE_IMAGES: list[dict[str, Any]] = [
    {
        "file": "test_res_00354.jpg",
        "mos_overall": 4.053, "mos_sharpness": 4.027, "mos_color": 4.067,
        "category": "excellent/excellent/excellent",
    },
    {
        "file": "test_res_00001.jpg",
        "mos_overall": 3.760, "mos_sharpness": 3.653, "mos_color": 3.707,
        "category": "good/good/good",
    },
    {
        "file": "test_res_00008.jpg",
        "mos_overall": 2.807, "mos_sharpness": 2.847, "mos_color": 2.927,
        "category": "fair/fair/fair",
    },
    {
        "file": "test_res_00756.jpg",
        "mos_overall": 2.700, "mos_sharpness": 2.407, "mos_color": 3.020,
        "category": "fair/poor/fair — sharpness disagrees",
    },
    {
        "file": "test_res_00052.jpg",
        "mos_overall": 2.293, "mos_sharpness": 2.507, "mos_color": 2.360,
        "category": "poor/poor/poor",
    },
    {
        "file": "test_res_00316.jpg",
        "mos_overall": 1.840, "mos_sharpness": 1.500, "mos_color": 2.080,
        "category": "poor/bad/poor — mixed low",
    },
    {
        "file": "test_res_00312.jpg",
        "mos_overall": 1.700, "mos_sharpness": 1.653, "mos_color": 1.667,
        "category": "bad/bad/bad",
    },
]

# All models to evaluate via OpenRouter
MODELS: list[tuple[str, str]] = [
    # Anthropic (via OpenRouter)
    ("anthropic/claude-opus-4.6", "Anthropic"),
    ("anthropic/claude-sonnet-4.6", "Anthropic"),
    ("anthropic/claude-haiku-4.5", "Anthropic"),
    # Frontier
    ("google/gemini-3.1-pro-preview", "Frontier"),
    ("openai/gpt-5.2", "Frontier"),
    ("openai/gpt-5.1", "Frontier"),
    ("google/gemini-2.5-pro", "Frontier"),
    ("openai/gpt-5", "Frontier"),
    # Strong mid-range
    ("openai/gpt-4.1", "Strong"),
    ("mistralai/pixtral-large-2411", "Strong"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "Strong"),
    ("qwen/qwen2.5-vl-72b-instruct", "Strong"),
    # Value
    ("google/gemini-2.5-flash", "Value"),
    ("google/gemini-3-flash-preview", "Value"),
    ("mistralai/mistral-large-2512", "Value"),
    ("openai/gpt-4.1-mini", "Value"),
    ("qwen/qwen3-vl-32b-instruct", "Value"),
    ("x-ai/grok-4-fast", "Value"),
    ("openai/gpt-5-mini", "Value"),
    ("meta-llama/llama-4-maverick", "Value"),
    # VL-specific
    ("qwen/qwen3-vl-8b-instruct", "VL"),
    ("qwen/qwen-2.5-vl-7b-instruct", "VL"),
    ("qwen/qwen2.5-vl-32b-instruct", "VL"),
    ("qwen/qwen-vl-max", "VL"),
    ("nvidia/nemotron-nano-12b-v2-vl", "VL"),
    ("baidu/ernie-4.5-vl-424b-a47b", "VL"),
]

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"


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
) -> tuple[str, int, str]:
    """Rate an image via OpenRouter. Returns (raw_text, latency_ms, error)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    start = time.time()
    try:
        is_reasoning = model_id.startswith("openai/o")
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": 1024,
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
        if not is_reasoning:
            kwargs["temperature"] = 0.0

        response = client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - start) * 1000)
        raw_text = response.choices[0].message.content or ""
        return raw_text, latency_ms, ""
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return "", latency_ms, str(exc)


def main() -> None:
    """Run evaluation across all models × all images."""
    load_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    system_prompt = build_system_prompt()

    # Pre-encode all images
    print("Encoding sample images...")
    encoded_images: list[tuple[str, str, str]] = []
    for img_info in SAMPLE_IMAGES:
        img_path = str(SAMPLE_DIR / img_info["file"])
        b64, mt = encode_image_base64(img_path)
        encoded_images.append((img_info["file"], b64, mt))
        print(f"  {img_info['file']} ({img_info['category']})")

    # Output file
    output_dir = Path(__file__).resolve().parent
    results_json = output_dir / "diqa_eval_raw_results.json"

    # Load existing results for resume
    existing: dict[str, dict] = {}
    if results_json.exists():
        try:
            for item in json.loads(results_json.read_text()):
                key = f"{item['model_id']}:{item['image']}"
                existing[key] = item
            print(f"Loaded {len(existing)} existing results (resume mode)")
        except (json.JSONDecodeError, KeyError):
            pass

    results: list[ImageResult] = []
    total_calls = len(MODELS) * len(SAMPLE_IMAGES)
    call_idx = 0

    for model_id, tier in MODELS:
        for img_name, img_b64, img_mt in encoded_images:
            call_idx += 1
            result_key = f"{model_id}:{img_name}"

            # Skip if already have successful result
            if result_key in existing and not existing[result_key].get("error"):
                r = ImageResult(**existing[result_key])
                results.append(r)
                continue

            print(
                f"[{call_idx:3d}/{total_calls}] {model_id:<45s} {img_name} ...",
                end=" ",
                flush=True,
            )

            raw_text, latency_ms, error = rate_image_openrouter(
                model_id=model_id,
                image_b64=img_b64,
                media_type=img_mt,
                system_prompt=system_prompt,
                user_prompt=USER_PROMPT,
                api_key=api_key,
            )

            if error:
                r = ImageResult(
                    model_id=model_id, image=img_name,
                    overall=None, sharpness=None, color_fidelity=None,
                    reasoning="", raw_response="",
                    latency_ms=latency_ms, error=error,
                )
                print(f"ERROR ({latency_ms}ms): {error[:80]}")
            else:
                try:
                    rating = parse_iqa_response(raw_text)
                    r = ImageResult(
                        model_id=model_id, image=img_name,
                        overall=rating.overall, sharpness=rating.sharpness,
                        color_fidelity=rating.color_fidelity,
                        reasoning=rating.reasoning, raw_response=raw_text,
                        latency_ms=latency_ms, error="",
                    )
                    print(
                        f"O={rating.overall:.1f} S={rating.sharpness:.1f} "
                        f"C={rating.color_fidelity:.1f} ({latency_ms}ms)"
                    )
                except ValueError as exc:
                    r = ImageResult(
                        model_id=model_id, image=img_name,
                        overall=None, sharpness=None, color_fidelity=None,
                        reasoning="", raw_response=raw_text,
                        latency_ms=latency_ms, error=f"Parse error: {exc}",
                    )
                    print(f"PARSE ERROR ({latency_ms}ms): {exc!s:.80s}")

            results.append(r)

            # Save after each call for resume
            results_json.write_text(
                json.dumps([asdict(x) for x in results], indent=2)
            )

            # Rate limit
            time.sleep(0.5)

    # Print per-model summary
    print("\n" + "=" * 120)
    print("PER-MODEL AVERAGE SCORES (successful images only)")
    print(f"{'Model':<45s} {'Avg_O':>7s} {'Avg_S':>7s} {'Avg_C':>7s} {'OK':>4s} {'Err':>4s}")
    print("-" * 120)

    model_tier = dict(MODELS)
    for model_id, tier in MODELS:
        model_results = [r for r in results if r.model_id == model_id and not r.error]
        errors = sum(1 for r in results if r.model_id == model_id and r.error)
        if model_results:
            avg_o = sum(r.overall for r in model_results) / len(model_results)
            avg_s = sum(r.sharpness for r in model_results) / len(model_results)
            avg_c = sum(r.color_fidelity for r in model_results) / len(model_results)
            print(
                f"{model_id:<45s} {avg_o:>7.2f} {avg_s:>7.2f} {avg_c:>7.2f} "
                f"{len(model_results):>4d} {errors:>4d}"
            )
        else:
            print(f"{model_id:<45s}     {'—':>5s}     {'—':>5s}     {'—':>5s} {0:>4d} {errors:>4d}")

    # Print per-image summary
    print("\n\nPER-IMAGE SCORES vs HUMAN MOS")
    mos_lookup = {img["file"]: img for img in SAMPLE_IMAGES}
    for img_info in SAMPLE_IMAGES:
        img = img_info["file"]
        print(f"\n  {img} — MOS: O={img_info['mos_overall']:.3f} "
              f"S={img_info['mos_sharpness']:.3f} C={img_info['mos_color']:.3f} "
              f"({img_info['category']})")
        img_results = [r for r in results if r.image == img and not r.error]
        for r in img_results:
            delta_o = r.overall - img_info["mos_overall"]
            delta_s = r.sharpness - img_info["mos_sharpness"]
            delta_c = r.color_fidelity - img_info["mos_color"]
            print(
                f"    {r.model_id:<45s} O={r.overall:>4.1f}({delta_o:>+5.2f}) "
                f"S={r.sharpness:>4.1f}({delta_s:>+5.2f}) "
                f"C={r.color_fidelity:>4.1f}({delta_c:>+5.2f})"
            )

    ok = sum(1 for r in results if not r.error)
    print(f"\n{ok}/{total_calls} evaluations succeeded")
    print(f"Raw results saved to: {results_json}")


if __name__ == "__main__":
    main()
