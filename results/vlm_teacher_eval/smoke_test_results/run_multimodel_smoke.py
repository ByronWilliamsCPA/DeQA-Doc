"""Multi-model IQA smoke test via OpenRouter.

Evaluates all candidate VLM teachers on a single test image
(singapore_flyer.jpg) to compare scoring calibration and reasoning quality.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/smoke_test_results/run_multimodel_smoke.py
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
class ModelResult:
    """Result from a single model evaluation."""

    model_id: str
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning: str
    raw_response: str
    latency_ms: int
    error: str


# All models to evaluate, ordered by expected quality (best first)
MODELS: list[tuple[str, str]] = [
    # Tier 1 — Frontier
    ("google/gemini-3.1-pro-preview", "Frontier"),
    ("openai/gpt-5.2", "Frontier"),
    ("openai/gpt-5.1", "Frontier"),
    ("google/gemini-2.5-pro", "Frontier"),
    ("openai/gpt-5", "Frontier"),
    # Tier 2 — Strong mid-range
    ("openai/gpt-4.1", "Strong"),
    ("mistralai/pixtral-large-2411", "Strong"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "Strong"),
    ("qwen/qwen2.5-vl-72b-instruct", "Strong"),
    # Tier 3 — Value
    ("google/gemini-2.5-flash", "Value"),
    ("google/gemini-3-flash-preview", "Value"),
    ("mistralai/mistral-large-2512", "Value"),
    ("openai/gpt-4.1-mini", "Value"),
    ("qwen/qwen3-vl-32b-instruct", "Value"),
    ("x-ai/grok-4-fast", "Value"),
    ("openai/gpt-5-mini", "Value"),
    ("meta-llama/llama-4-maverick", "Value"),
    # VL-specific models
    ("qwen/qwen3-vl-8b-instruct", "VL"),
    ("qwen/qwen-2.5-vl-7b-instruct", "VL"),
    ("qwen/qwen2.5-vl-32b-instruct", "VL"),
    ("qwen/qwen-vl-max", "VL"),
    ("nvidia/nemotron-nano-12b-v2-vl", "VL"),
    ("baidu/ernie-4.5-vl-424b-a47b", "VL"),
]


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
) -> ModelResult:
    """Rate an image using a model via OpenRouter."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    start = time.time()
    try:
        # Reasoning models (o3, o4-mini) don't support temperature
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
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return ModelResult(
            model_id=model_id,
            overall=None,
            sharpness=None,
            color_fidelity=None,
            reasoning="",
            raw_response="",
            latency_ms=latency_ms,
            error=str(exc),
        )

    try:
        rating = parse_iqa_response(raw_text)
        return ModelResult(
            model_id=model_id,
            overall=rating.overall,
            sharpness=rating.sharpness,
            color_fidelity=rating.color_fidelity,
            reasoning=rating.reasoning,
            raw_response=raw_text,
            latency_ms=latency_ms,
            error="",
        )
    except ValueError as exc:
        return ModelResult(
            model_id=model_id,
            overall=None,
            sharpness=None,
            color_fidelity=None,
            reasoning="",
            raw_response=raw_text,
            latency_ms=latency_ms,
            error=f"Parse error: {exc}",
        )


def main() -> None:
    """Run smoke test across all models."""
    load_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    # Encode test image
    image_path = str(REPO_ROOT / "DeQA-Score" / "fig" / "singapore_flyer.jpg")
    print(f"Encoding image: {image_path}")
    image_b64, media_type = encode_image_base64(image_path)
    system_prompt = build_system_prompt()

    # Output file for raw JSON results
    output_dir = Path(__file__).resolve().parent
    results_json = output_dir / "multimodel_raw_results.json"

    # Load existing results for resume support
    existing: dict[str, dict] = {}
    if results_json.exists():
        try:
            for item in json.loads(results_json.read_text()):
                existing[item["model_id"]] = item
            print(f"Loaded {len(existing)} existing results (resume mode)")
        except (json.JSONDecodeError, KeyError):
            pass

    results: list[ModelResult] = []
    total = len(MODELS)

    for idx, (model_id, tier) in enumerate(MODELS, 1):
        # Skip if we already have a successful result
        if model_id in existing and not existing[model_id].get("error"):
            print(f"[{idx:2d}/{total}] SKIP {model_id} (already have result)")
            result = ModelResult(**existing[model_id])
            results.append(result)
            continue

        print(f"[{idx:2d}/{total}] {tier:<10s} {model_id} ...", end=" ", flush=True)

        result = rate_image_openrouter(
            model_id=model_id,
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=USER_PROMPT,
            api_key=api_key,
        )
        results.append(result)

        if result.error:
            print(f"ERROR ({result.latency_ms}ms): {result.error[:100]}")
        else:
            print(
                f"O={result.overall:.1f} S={result.sharpness:.1f} "
                f"C={result.color_fidelity:.1f} ({result.latency_ms}ms)"
            )

        # Save after each model for resume support
        results_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )

        # Rate limit: small pause between calls
        if idx < total:
            time.sleep(1)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':<50s} {'Overall':>8s} {'Sharp':>8s} {'Color':>8s} {'ms':>8s} {'Status':>10s}")
    print("-" * 100)

    model_tier = dict(MODELS)
    for r in results:
        tier = model_tier.get(r.model_id, "?")
        if r.error:
            print(f"{r.model_id:<50s} {'—':>8s} {'—':>8s} {'—':>8s} {r.latency_ms:>8d} {'ERROR':>10s}")
        else:
            print(
                f"{r.model_id:<50s} {r.overall:>8.1f} {r.sharpness:>8.1f} "
                f"{r.color_fidelity:>8.1f} {r.latency_ms:>8d} {tier:>10s}"
            )

    # Count successes
    ok = sum(1 for r in results if not r.error)
    print(f"\n{ok}/{total} models returned valid scores")

    print(f"\nRaw results saved to: {results_json}")


if __name__ == "__main__":
    main()
