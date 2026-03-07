"""Conftest to allow importing src.uncertainty without loading the full model.

The root src/__init__.py imports MPLUGOwl2LlamaForCausalLM which requires
CUDA and heavy dependencies. We patch the src module to avoid this.
"""

import sys
import types
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create a lightweight src module that doesn't import the model
if "src" not in sys.modules:
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(project_root / "src")]
    sys.modules["src"] = src_module
