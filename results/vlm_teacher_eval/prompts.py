"""IQA rating prompt templates for VLM teacher evaluation.

These prompts elicit structured quality ratings from VLMs on three
dimensions (overall, sharpness, color fidelity) using a configurable
numeric scale. The default 1-5 scale aligns with DIQA-5000 MOS for
direct correlation comparison.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IQAPromptConfig:
    """Configuration for IQA rating prompts."""

    scale_min: float = 1.0
    scale_max: float = 5.0
    scale_step: float = 0.1


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert document image quality assessor. You evaluate scanned or \
photographed document images for visual quality as perceived by a human reader.

You rate documents on three dimensions using a {scale_min}-{scale_max} scale \
with {scale_step} increments:

1. **Overall Quality**: Holistic readability and usability of the document. \
   Consider all factors together — could a human comfortably read this?
2. **Sharpness**: Text edge clarity, blur level, and resolution adequacy. \
   Are characters crisp and well-defined, or soft and blurred?
3. **Color Fidelity**: Color accuracy, contrast, white balance, and \
   tonal reproduction. Are colors natural and the page background clean?

Scale anchors:
- {scale_min}: Completely unusable / illegible / severe degradation
- {mid_low}: Poor — significant issues affecting readability
- {mid}: Fair — acceptable but with noticeable problems
- {mid_high}: Good — minor issues, generally readable
- {scale_max}: Excellent — crisp, clean, high-quality reproduction

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.\
"""

USER_PROMPT = """\
Rate the quality of this document image.

Respond with exactly this JSON structure:
{{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}}

The reasoning field should be 1-2 sentences explaining the key quality \
factors you observed. Keep it concise.\
"""


def build_system_prompt(config: IQAPromptConfig | None = None) -> str:
    """Build system prompt with scale parameters.

    Args:
        config: Prompt configuration. Uses defaults (1-5 scale) if None.

    Returns:
        Formatted system prompt string.
    """
    if config is None:
        config = IQAPromptConfig()

    scale_range = config.scale_max - config.scale_min
    return SYSTEM_PROMPT_TEMPLATE.format(
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        scale_step=config.scale_step,
        mid_low=config.scale_min + scale_range * 0.25,
        mid=config.scale_min + scale_range * 0.50,
        mid_high=config.scale_min + scale_range * 0.75,
    )
