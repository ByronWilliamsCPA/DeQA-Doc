"""Integration contract tests for image_detection bridge compatibility.

Validates that the stable API surface documented in STABILITY.md exists
with expected signatures and values. Does NOT load models or require GPU.

Run:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python -m pytest \
        tests/integration/test_contract.py -v --tb=short
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_constants_importable():
    """Verify constants module exports expected values."""
    from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    assert DEFAULT_IMAGE_TOKEN == "<|image|>"
    assert IMAGE_TOKEN_INDEX == -200


def test_conversation_template_exists():
    """Verify mplug_owl2 conversation template is registered."""
    from src.conversation import conv_templates

    assert "mplug_owl2" in conv_templates
    conv = conv_templates["mplug_owl2"]
    assert hasattr(conv, "copy"), "conv_templates['mplug_owl2'] must have .copy()"
    assert hasattr(conv, "append_message"), "must have .append_message()"
    assert hasattr(conv, "get_prompt"), "must have .get_prompt()"
    assert hasattr(conv, "roles"), "must have .roles"
    assert len(conv.roles) >= 2, "must have at least 2 roles"


def test_conversation_template_roundtrip():
    """Verify conversation template can build a prompt."""
    from src.conversation import conv_templates

    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], "test user message")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_mm_utils_get_model_name_from_path():
    """Verify get_model_name_from_path works without GPU.

    This function is pure string manipulation — no torch needed at call time,
    but the module imports torch at the top level.
    """
    torch = importlib.util.find_spec("torch")
    if torch is None:
        import pytest

        pytest.skip("torch not installed")

    from src.mm_utils import get_model_name_from_path

    assert get_model_name_from_path("/path/to/deqa-doc-overall") == "deqa-doc-overall"
    result = get_model_name_from_path("/path/to/model/checkpoint-100")
    assert "checkpoint" in result


def test_mm_utils_functions_exist():
    """Verify mm_utils exports expected functions."""
    torch = importlib.util.find_spec("torch")
    if torch is None:
        import pytest

        pytest.skip("torch not installed")

    from src.mm_utils import (
        expand2square,
        get_model_name_from_path,
        tokenizer_image_token,
    )

    assert callable(get_model_name_from_path)
    assert callable(tokenizer_image_token)
    assert callable(expand2square)


def test_builder_module_findable():
    """Verify src/model/builder.py exists on disk.

    Cannot use importlib.util.find_spec because importing the builder
    triggers the full transformers/bitsandbytes/CUDA chain.
    """
    builder_path = Path(__file__).parent.parent.parent / "src" / "model" / "builder.py"
    assert builder_path.exists(), f"src/model/builder.py not found at {builder_path}"


def test_scorer_module_findable():
    """Verify src/evaluate/scorer.py exists on disk."""
    scorer_path = Path(__file__).parent.parent.parent / "src" / "evaluate" / "scorer.py"
    assert scorer_path.exists(), f"src/evaluate/scorer.py not found at {scorer_path}"
