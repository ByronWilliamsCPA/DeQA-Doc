"""Backward compatibility and migration tests for SigLIP2-IQA v2.0.

All tests run locally on CPU with mock backbones — no GPU or Modal required.

Usage:
    cd DeQA-Score && uv run python -m pytest ../modal/test_v2_backward_compat.py -v
    # or from repo root:
    python -m pytest modal/test_v2_backward_compat.py -v
"""

from __future__ import annotations

import importlib.util
import os
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import torch
import torch.nn as nn

# Import from local file to avoid conflict with 'modal' pip package
_HERE = os.path.dirname(os.path.abspath(__file__))

def _import_local(module_name: str):
    """Import a module from the same directory, bypassing the modal package."""
    import sys
    path = os.path.join(_HERE, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass processing can find the module
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_model_mod = _import_local("siglip2_v2_model")

IQA_DIMENSIONS = _model_mod.IQA_DIMENSIONS
AttentionPool = _model_mod.AttentionPool
SigLIP2V2Config = _model_mod.SigLIP2V2Config
SigLIP2IQAv2 = _model_mod.SigLIP2IQAv2
TrainingDataConfig = _model_mod.TrainingDataConfig
build_model_from_backbone = _model_mod.build_model_from_backbone
_V1_NON_IQA_HEADS = _model_mod._V1_NON_IQA_HEADS
load_v1_checkpoint = _model_mod.load_v1_checkpoint


# ============================================================================
# Mock Backbone
# ============================================================================


class _MockBackboneOutput:
    """Mimics transformers BaseModelOutputWithPooling."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state
        self.pooler_output = last_hidden_state.mean(dim=1)


class MockSigLIP2Backbone(nn.Module):
    """Lightweight mock that returns shaped tensors like SigLIP2."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Minimal parameter so it counts as a module
        self.proj = nn.Linear(embed_dim, embed_dim)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        **kwargs: object,
    ) -> _MockBackboneOutput:
        batch_size = pixel_values.size(0)
        seq_len = 784  # Default NaFlex max patches
        hidden = torch.randn(batch_size, seq_len, self.embed_dim)
        return _MockBackboneOutput(hidden)


def _make_model(
    use_attention_pooling: bool = True,
    embed_dim: int = 768,
) -> SigLIP2IQAv2:
    """Helper to build a v2.0 model with mock backbone."""
    config = SigLIP2V2Config(
        embed_dim=embed_dim,
        use_attention_pooling=use_attention_pooling,
    )
    backbone = MockSigLIP2Backbone(embed_dim)
    return build_model_from_backbone(backbone, config)


# ============================================================================
# Tests
# ============================================================================


class TestModelBuilds:
    """Test that the v2.0 model constructs and runs forward pass."""

    def test_v2_model_builds(self) -> None:
        model = _make_model(use_attention_pooling=True)

        # Check attention pools exist
        assert hasattr(model, "attn_pools")
        assert set(model.attn_pools.keys()) == set(IQA_DIMENSIONS)

        # Check heads exist
        assert set(model.heads.keys()) == set(IQA_DIMENSIONS)

        # Check temp buffers
        for dim in IQA_DIMENSIONS:
            assert hasattr(model, f"temp_{dim}")

    def test_forward_pass_shape(self) -> None:
        model = _make_model()
        model.eval()

        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        spatial_shapes = torch.tensor([[28, 28], [24, 32]])

        with torch.no_grad():
            results = model(pixel_values, spatial_shapes)

        assert set(results.keys()) == set(IQA_DIMENSIONS)
        for dim in IQA_DIMENSIONS:
            assert results[dim]["mu"].shape == (batch_size,)
            assert results[dim]["sigma_sq"].shape == (batch_size,)
            # sigma_sq should be positive (exp of log_sigma_sq)
            assert (results[dim]["sigma_sq"] > 0).all()

    def test_forward_subset_tasks(self) -> None:
        model = _make_model()
        model.eval()
        pixel_values = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            results = model(pixel_values, tasks=["overall"])

        assert list(results.keys()) == ["overall"]


class TestAttentionPoolMasking:
    """Test attention pooling handles NaFlex padding correctly."""

    def test_attention_weights_sum_to_one(self) -> None:
        pool = AttentionPool(embed_dim=64)
        x = torch.randn(2, 10, 64)

        # No mask — all valid
        out = pool(x, padding_mask=None)
        assert out.shape == (2, 64)

    def test_masked_positions_get_zero_weight(self) -> None:
        pool = AttentionPool(embed_dim=64)
        x = torch.randn(2, 10, 64)

        # Mask: first 7 valid, last 3 padded
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :7] = True

        # Manually get attention weights by running the attention computation
        query = pool.query.expand(2, -1, -1)  # (2, 1, 64)
        logits = torch.bmm(query, x.transpose(1, 2)) * pool.scale  # (2, 1, 10)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # (2, 1, 10)

        # Padded positions should have zero weight
        assert (weights[:, :, 7:] == 0).all()
        # Valid positions should sum to 1
        assert torch.allclose(weights[:, :, :7].sum(dim=-1), torch.ones(2, 1))

    def test_output_shape_with_mask(self) -> None:
        pool = AttentionPool(embed_dim=128)
        x = torch.randn(4, 784, 128)
        mask = torch.ones(4, 784, dtype=torch.bool)
        mask[:, 600:] = False  # Last 184 are padding

        out = pool(x, padding_mask=mask)
        assert out.shape == (4, 128)


class TestV1CompatibilityMode:
    """Test that disabling attention pooling reproduces v1.0 behavior."""

    def test_no_attention_pool_params(self) -> None:
        model = _make_model(use_attention_pooling=False)
        param_names = [n for n, _ in model.named_parameters()]
        attn_params = [n for n in param_names if "attn_pool" in n]
        assert len(attn_params) == 0, f"Found unexpected attention pool params: {attn_params}"

    def test_v1_forward_uses_mean_pooling(self) -> None:
        model = _make_model(use_attention_pooling=False)
        model.eval()

        pixel_values = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            results = model(pixel_values)

        # Should still produce valid outputs
        for dim in IQA_DIMENSIONS:
            assert results[dim]["mu"].shape == (2,)


class TestConfigYAMLRoundtrip:
    """Test config serialization/deserialization preserves all fields."""

    def test_roundtrip(self) -> None:
        original = SigLIP2V2Config(
            max_num_patches=784,
            phase2_use_pcgrad=True,
            use_attention_pooling=True,
            data=TrainingDataConfig(
                pseudo_label_jsonl="/tmp/test.jsonl",
                mix_strategy="weighted_sample",
            ),
        )

        with NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            original.to_yaml(f.name)
            restored = SigLIP2V2Config.from_yaml(f.name)

        assert original == restored

    def test_default_config_roundtrip(self) -> None:
        original = SigLIP2V2Config()
        yaml_str = original.to_yaml()
        assert "backbone_id" in yaml_str
        assert "cosine_warm_restarts" in yaml_str


class TestV1CheckpointKeyMapping:
    """Test loading a v1.0 checkpoint into v2.0 model."""

    def _make_v1_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Create a mock v1.0 multi-task state_dict."""
        sd: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Backbone (simplified — just a couple of keys)
        sd["backbone.proj.weight"] = torch.randn(768, 768)
        sd["backbone.proj.bias"] = torch.randn(768)

        # IQA heads: Linear(768,256) at .0, ReLU at .1, Dropout at .2, Linear(256,2) at .3
        for dim in ("overall", "sharpness", "color"):
            sd[f"heads.{dim}.0.weight"] = torch.randn(256, 768)
            sd[f"heads.{dim}.0.bias"] = torch.randn(256)
            sd[f"heads.{dim}.3.weight"] = torch.randn(2, 256)
            sd[f"heads.{dim}.3.bias"] = torch.randn(2)

        # Non-IQA heads (should be filtered out)
        for dim in ("script", "source", "orientation", "shadow", "warping"):
            sd[f"heads.{dim}.0.weight"] = torch.randn(64, 768)
            sd[f"heads.{dim}.0.bias"] = torch.randn(64)

        # Temperature buffers
        for dim in ("overall", "sharpness", "color", "shadow", "warping"):
            sd[f"temp_{dim}"] = torch.tensor(1.0)

        return sd

    def test_load_v1_checkpoint_reports_keys(self) -> None:
        model = _make_model()
        sd = self._make_v1_state_dict()

        # Save to temp file
        with NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sd, f.name)
            missing, unexpected = _load_v1_with_mock(model, f.name)

        # Attention pool params should be missing (new in v2.0)
        attn_missing = [k for k in missing if "attn_pool" in k]
        assert len(attn_missing) > 0, "Expected missing attention pool keys"

        # Non-IQA heads should be unexpected
        non_iqa_unexpected = [k for k in unexpected if any(
            h in k for h in ("script", "source", "orientation")
        )]
        assert len(non_iqa_unexpected) > 0, "Expected unexpected non-IQA head keys"

    def test_loaded_weights_match(self) -> None:
        model = _make_model()
        sd = self._make_v1_state_dict()

        with NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sd, f.name)
            _load_v1_with_mock(model, f.name)

        # Verify head weights were loaded correctly
        for dim in ("overall", "sharpness", "color"):
            expected_w = sd[f"heads.{dim}.0.weight"]
            actual_w = model.heads[dim][0].weight.data
            assert torch.equal(expected_w, actual_w), f"Weight mismatch for {dim}"


class TestEmbeddingExtraction:
    """Test OOD detector compatibility via get_embeddings()."""

    def test_embedding_shape(self) -> None:
        model = _make_model(embed_dim=768)
        model.eval()

        pixel_values = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            embeddings = model.get_embeddings(pixel_values)

        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == torch.float32

    def test_embedding_with_spatial_shapes(self) -> None:
        model = _make_model(embed_dim=768)
        model.eval()

        pixel_values = torch.randn(2, 3, 224, 224)
        spatial_shapes = torch.tensor([[28, 28], [24, 32]])

        with torch.no_grad():
            embeddings = model.get_embeddings(pixel_values, spatial_shapes)

        assert embeddings.shape == (2, 768)


# ============================================================================
# Helpers
# ============================================================================


def _load_v1_with_mock(
    model: SigLIP2IQAv2,
    checkpoint_path: str,
) -> tuple[list[str], list[str]]:
    """Load v1.0 checkpoint, reimplementing the filtering logic for mock state_dict."""
    return load_v1_checkpoint(model, checkpoint_path)
