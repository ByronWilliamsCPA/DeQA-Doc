"""Parse and validate VLM IQA rating responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IQARating:
    """Parsed quality rating from a VLM."""

    overall: float
    sharpness: float
    color_fidelity: float
    reasoning: str = ""


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model response text.

    Handles markdown code fences (``json ... ``) and leading/trailing
    whitespace.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    # Fix double braces — some models echo the template format
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        msg = f"Cannot parse JSON from response: {exc}"
        raise ValueError(msg) from exc


def parse_iqa_response(
    raw_text: str,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> IQARating:
    """Parse and validate a VLM IQA rating response.

    Args:
        raw_text: Raw model response text (may include markdown fencing).
        scale_min: Minimum valid score.
        scale_max: Maximum valid score.

    Returns:
        Validated IQARating.

    Raises:
        ValueError: If response cannot be parsed or scores are out of range.
    """
    parsed = extract_json(raw_text)

    scores: dict[str, float] = {}
    for key in ("overall", "sharpness", "color_fidelity"):
        if key not in parsed:
            msg = f"Missing required key '{key}' in response"
            raise ValueError(msg)

        score = float(parsed[key])
        if not (scale_min <= score <= scale_max):
            msg = f"{key} score {score} outside [{scale_min}, {scale_max}]"
            raise ValueError(msg)
        scores[key] = score

    return IQARating(
        overall=scores["overall"],
        sharpness=scores["sharpness"],
        color_fidelity=scores["color_fidelity"],
        reasoning=parsed.get("reasoning", ""),
    )
