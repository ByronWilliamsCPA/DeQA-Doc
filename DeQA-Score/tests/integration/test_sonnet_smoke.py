"""Two-way Sonnet 4.6 IQA smoke test.

Compares quality ratings from:
1. Direct Anthropic API (ANTHROPIC_API_KEY)
2. OpenRouter API via openai SDK (OPENROUTER_API_KEY)

Both use identical system/user prompts and the same test image.

Usage:
    cd DeQA-Score

    # Dry run (skips API tests)
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python -m pytest \
        tests/integration/test_sonnet_smoke.py -v

    # Live run (hits real APIs, ~$0.01 cost)
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python -m pytest \
        tests/integration/test_sonnet_smoke.py -v -s --run-api-tests

Requires:
    - ANTHROPIC_API_KEY in .env or environment
    - OPENROUTER_API_KEY in .env or environment
    - fig/singapore_flyer.jpg (ships with repo)
    - uv sync --extra api --extra dev
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt
from results.vlm_teacher_eval.response_parser import IQARating, parse_iqa_response
from results.vlm_teacher_eval.vlm_client import VLMResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # DeQA-Score/
_REPO_ROOT = _PROJECT_ROOT.parent  # DeQA-Doc/
SAMPLE_IMAGE = str(_PROJECT_ROOT / "fig" / "singapore_flyer.jpg")


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
def _load_env() -> None:
    """Load .env from repo root into os.environ (no python-dotenv needed)."""
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ProviderResult:
    """Result from a single provider."""

    provider: str
    rating: IQARating | None
    raw_response: str
    latency_ms: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------
def _rate_via_anthropic(image_path: str) -> ProviderResult:
    """Rate image via direct Anthropic API."""
    from results.vlm_teacher_eval.vlm_client import AnthropicClient

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ProviderResult("anthropic", None, "", 0, "ANTHROPIC_API_KEY not set")

    client = AnthropicClient(api_key=api_key)
    system_prompt = build_system_prompt()
    b64_data, media_type = encode_image_base64(image_path)

    response: VLMResponse = client.rate_image(
        image_b64=b64_data,
        media_type=media_type,
        system_prompt=system_prompt,
        user_prompt=USER_PROMPT,
        temperature=0.0,
    )

    try:
        rating = parse_iqa_response(response.text)
        return ProviderResult("anthropic", rating, response.text, response.latency_ms)
    except ValueError as exc:
        return ProviderResult(
            "anthropic", None, response.text, response.latency_ms, str(exc)
        )


def _rate_via_openrouter(image_path: str) -> ProviderResult:
    """Rate image via OpenRouter API (openai SDK)."""
    from results.vlm_teacher_eval.vlm_client import OpenRouterClient

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ProviderResult("openrouter", None, "", 0, "OPENROUTER_API_KEY not set")

    client = OpenRouterClient(api_key=api_key)
    system_prompt = build_system_prompt()
    b64_data, media_type = encode_image_base64(image_path)

    response: VLMResponse = client.rate_image(
        image_b64=b64_data,
        media_type=media_type,
        system_prompt=system_prompt,
        user_prompt=USER_PROMPT,
        temperature=0.0,
    )

    try:
        rating = parse_iqa_response(response.text)
        return ProviderResult("openrouter", rating, response.text, response.latency_ms)
    except ValueError as exc:
        return ProviderResult(
            "openrouter", None, response.text, response.latency_ms, str(exc)
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.api
def test_anthropic_direct_rates_image():
    """Sonnet 4.6 via direct Anthropic API returns valid IQA scores."""
    result = _rate_via_anthropic(SAMPLE_IMAGE)
    assert result.error is None, f"Anthropic API error: {result.error}"
    assert result.rating is not None
    assert 1.0 <= result.rating.overall <= 5.0
    assert 1.0 <= result.rating.sharpness <= 5.0
    assert 1.0 <= result.rating.color_fidelity <= 5.0
    print(f"\n  Anthropic: overall={result.rating.overall}, "
          f"sharpness={result.rating.sharpness}, "
          f"color={result.rating.color_fidelity}, "
          f"latency={result.latency_ms}ms")
    print(f"  Reasoning: {result.rating.reasoning}")


@pytest.mark.api
def test_openrouter_rates_image():
    """Sonnet 4.6 via OpenRouter returns valid IQA scores."""
    result = _rate_via_openrouter(SAMPLE_IMAGE)
    assert result.error is None, f"OpenRouter API error: {result.error}"
    assert result.rating is not None
    assert 1.0 <= result.rating.overall <= 5.0
    assert 1.0 <= result.rating.sharpness <= 5.0
    assert 1.0 <= result.rating.color_fidelity <= 5.0
    print(f"\n  OpenRouter: overall={result.rating.overall}, "
          f"sharpness={result.rating.sharpness}, "
          f"color={result.rating.color_fidelity}, "
          f"latency={result.latency_ms}ms")
    print(f"  Reasoning: {result.rating.reasoning}")


@pytest.mark.api
def test_two_way_comparison():
    """Compare Anthropic direct vs OpenRouter scores on the same image.

    Both should return valid scores. At temp=0 with the same model,
    scores should be close (within 0.5 tolerance).
    """
    anthropic_result = _rate_via_anthropic(SAMPLE_IMAGE)
    openrouter_result = _rate_via_openrouter(SAMPLE_IMAGE)

    assert anthropic_result.rating is not None, (
        f"Anthropic failed: {anthropic_result.error}"
    )
    assert openrouter_result.rating is not None, (
        f"OpenRouter failed: {openrouter_result.error}"
    )

    # Print comparison table
    print(f"\n  {'Provider':<14} {'Overall':>8} {'Sharpness':>10} "
          f"{'Color':>8} {'Latency':>10}")
    print(f"  {'-' * 54}")

    for name, r in [("anthropic", anthropic_result), ("openrouter", openrouter_result)]:
        print(
            f"  {name:<14} {r.rating.overall:>8.1f} {r.rating.sharpness:>10.1f} "
            f"{r.rating.color_fidelity:>8.1f} {r.latency_ms:>8}ms"
        )

    # Score comparison
    tolerance = 0.5
    dims_to_check = [
        ("overall", anthropic_result.rating.overall, openrouter_result.rating.overall),
        ("sharpness", anthropic_result.rating.sharpness, openrouter_result.rating.sharpness),
        ("color_fidelity", anthropic_result.rating.color_fidelity, openrouter_result.rating.color_fidelity),
    ]

    print(f"\n  Score differences (tolerance={tolerance}):")
    for dim, a_score, o_score in dims_to_check:
        diff = abs(a_score - o_score)
        status = "OK" if diff <= tolerance else "WARN"
        print(f"    {dim}: diff={diff:.1f} [{status}]")

    # Print reasoning from both
    print(f"\n  Anthropic reasoning: {anthropic_result.rating.reasoning}")
    print(f"  OpenRouter reasoning: {openrouter_result.rating.reasoning}")
