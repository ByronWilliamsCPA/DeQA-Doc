"""Conftest for integration tests.

Patches src module to avoid CUDA imports (same pattern as
tests/uncertainty/conftest.py) and adds --run-api-tests flag for
tests that hit real VLM APIs.
"""

import sys
import types
from pathlib import Path

import pytest

# Add DeQA-Score project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add repo root (parent of DeQA-Score) for results.vlm_teacher_eval imports
repo_root = project_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Create a lightweight src module that doesn't trigger CUDA model imports
if "src" not in sys.modules:
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(project_root / "src")]
    sys.modules["src"] = src_module


def pytest_addoption(parser):
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys (Anthropic, OpenRouter)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "api: marks tests as requiring API keys"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-api-tests"):
        skip = pytest.mark.skip(reason="need --run-api-tests to run")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip)
