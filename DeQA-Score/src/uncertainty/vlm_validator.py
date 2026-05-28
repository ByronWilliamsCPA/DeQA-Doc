"""Tier-2 VLM validator using Qwen3-VL-8B via OpenRouter.

Veto-only: never generates labels, only rejects bad pseudo-labels.
If the VLM disagrees by >= 1.5 levels from SigLIP2's prediction,
the sample is rejected.

Capped at <= 10% of total pool to control API cost.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

# Quality level mapping (matches experiment_config.json)
QUALITY_LEVEL_MAP: dict[str, float] = {
    "excellent": 5.0,
    "good": 4.0,
    "fair": 3.0,
    "poor": 2.0,
    "bad": 1.0,
}

# Prompt templates from experiment_config.json
PROMPTS: dict[str, str] = {
    "overall": (
        "Rate the overall quality of this document image. "
        "Consider readability, clarity, and general visual quality. "
        "Choose exactly one: excellent, good, fair, poor, or bad. "
        "Respond with only one word."
    ),
    "sharpness": (
        "Rate the sharpness quality of this document image. "
        "Consider text clarity, edge definition, and focus. "
        "Choose exactly one: excellent, good, fair, poor, or bad. "
        "Respond with only one word."
    ),
    "color": (
        "Rate the color fidelity of this document image. "
        "Consider color accuracy, saturation, and consistency. "
        "Choose exactly one: excellent, good, fair, poor, or bad. "
        "Respond with only one word."
    ),
}


@dataclass(frozen=True)
class VLMVetoResult:
    """Result of VLM veto validation for one image+dimension."""

    image_id: str
    dimension: str
    vlm_label: str | None  # "excellent", "good", etc. or None if parse failed
    vlm_score: float | None  # Numeric score (5.0, 4.0, etc.)
    siglip2_mu: float
    level_disagreement: float  # |vlm_score - siglip2_mu|
    is_vetoed: bool  # True if disagreement >= veto_threshold
    latency_ms: float
    parse_success: bool


@dataclass
class VLMBudgetTracker:
    """Tracks API calls and estimated cost."""

    total_calls: int = 0
    total_cost_usd: float = 0.0
    vetoed_count: int = 0
    parse_failures: int = 0
    # Qwen3-VL-8B pricing: $0.08/$0.50 per 1M tokens (input/output)
    input_cost_per_1m: float = 0.08
    output_cost_per_1m: float = 0.50
    est_input_tokens_per_call: int = 1500  # image + prompt
    est_output_tokens_per_call: int = 5  # single word

    def record_call(self, vetoed: bool, parse_success: bool) -> None:
        """Record a single API call."""
        self.total_calls += 1
        input_cost = self.est_input_tokens_per_call / 1e6 * self.input_cost_per_1m
        output_cost = self.est_output_tokens_per_call / 1e6 * self.output_cost_per_1m
        self.total_cost_usd += input_cost + output_cost
        if vetoed:
            self.vetoed_count += 1
        if not parse_success:
            self.parse_failures += 1

    def summary(self) -> dict:
        """Return budget summary."""
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "vetoed_count": self.vetoed_count,
            "parse_failures": self.parse_failures,
            "veto_rate": (
                self.vetoed_count / self.total_calls if self.total_calls > 0 else 0.0
            ),
        }


def _parse_vlm_response(response_text: str) -> str | None:
    """Parse VLM response to extract quality level.

    Returns:
        Quality level string or None if parsing failed.
    """
    text = response_text.strip().lower().rstrip(".")
    # Require word boundary to prevent "badly"→"bad", "goodness"→"good" false positives
    import re

    for level in QUALITY_LEVEL_MAP:
        if text == level or re.match(rf"^{level}\b", text):
            return level
    return None


class VLMValidator:
    """Tier-2 VLM validator using OpenRouter API.

    Veto-only: rejects samples where VLM disagrees by >= veto_threshold
    levels from SigLIP2's prediction.

    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var).
        model: OpenRouter model ID.
        veto_threshold: Minimum level disagreement to trigger veto.
        max_pool_fraction: Maximum fraction of pool to send to VLM.
        max_tokens: Maximum output tokens per call.
        temperature: Sampling temperature (0.0 for deterministic).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen/qwen3-vl-8b-instruct",
        veto_threshold: float = 1.5,
        max_pool_fraction: float = 0.10,
        max_tokens: int = 32,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.veto_threshold = veto_threshold
        self.max_pool_fraction = max_pool_fraction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.budget = VLMBudgetTracker()

        if not self.api_key:
            logger.warning("No OpenRouter API key set. VLM validation will fail.")

    def _call_api(self, image_path: str, dimension: str) -> tuple[str, float]:
        """Make a single API call to OpenRouter.

        Args:
            image_path: Path to image file.
            dimension: Quality dimension for prompt selection.

        Returns:
            Tuple of (response_text, latency_ms).

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: If API call fails.
        """
        try:
            import httpx
        except ImportError as exc:
            msg = "httpx is required for VLM validation: pip install httpx"
            raise ImportError(msg) from exc

        prompt = PROMPTS.get(dimension, PROMPTS["overall"])

        # Read and encode image
        image_data = Path(image_path).read_bytes()
        b64_image = base64.b64encode(image_data).decode("utf-8")
        # Detect MIME type
        suffix = Path(image_path).suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
            suffix.lstrip("."), "image/jpeg"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        start = time.monotonic()
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code != 200:
            msg = f"OpenRouter API error {response.status_code}: {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return text, latency_ms

    def validate_single(
        self,
        image_id: str,
        image_path: str,
        dimension: str,
        siglip2_mu: float,
    ) -> VLMVetoResult:
        """Validate a single image with VLM.

        Args:
            image_id: Image identifier.
            image_path: Path to image file on disk.
            dimension: Quality dimension.
            siglip2_mu: SigLIP2's predicted MOS for this dimension.

        Returns:
            VLMVetoResult with veto decision.
        """
        try:
            response_text, latency_ms = self._call_api(image_path, dimension)
        except Exception:
            logger.exception("VLM API call failed for %s", image_id)
            self.budget.record_call(vetoed=False, parse_success=False)
            return VLMVetoResult(
                image_id=image_id,
                dimension=dimension,
                vlm_label=None,
                vlm_score=None,
                siglip2_mu=siglip2_mu,
                level_disagreement=0.0,
                is_vetoed=False,  # Don't veto on API failure
                latency_ms=0.0,
                parse_success=False,
            )

        vlm_label = _parse_vlm_response(response_text)
        parse_success = vlm_label is not None
        vlm_score = QUALITY_LEVEL_MAP.get(vlm_label, 0.0) if vlm_label else None

        if vlm_score is not None:
            disagreement = abs(vlm_score - siglip2_mu)
        else:
            disagreement = 0.0  # Don't veto on parse failure

        is_vetoed = vlm_score is not None and disagreement >= self.veto_threshold
        self.budget.record_call(vetoed=is_vetoed, parse_success=parse_success)

        return VLMVetoResult(
            image_id=image_id,
            dimension=dimension,
            vlm_label=vlm_label,
            vlm_score=vlm_score,
            siglip2_mu=siglip2_mu,
            level_disagreement=disagreement,
            is_vetoed=is_vetoed,
            latency_ms=latency_ms,
            parse_success=parse_success,
        )

    def select_tier2_queue(
        self,
        candidates: list[dict],
        total_pool_size: int,
    ) -> list[dict]:
        """Select which tier-2 candidates to actually send to VLM.

        Caps at max_pool_fraction of total pool, prioritizing by highest JSD.

        Args:
            candidates: List of dicts with at least 'jsd' and 'image_id' keys.
            total_pool_size: Total number of images in the pool.

        Returns:
            Selected candidates, sorted by descending JSD.
        """
        max_count = int(total_pool_size * self.max_pool_fraction)
        sorted_candidates = sorted(candidates, key=lambda x: x["jsd"], reverse=True)
        selected = sorted_candidates[:max_count]

        if len(candidates) > max_count:
            logger.info(
                "Tier-2 queue capped: %d/%d candidates selected (%.1f%% of pool)",
                len(selected),
                len(candidates),
                len(selected) / total_pool_size * 100,
            )

        return selected
