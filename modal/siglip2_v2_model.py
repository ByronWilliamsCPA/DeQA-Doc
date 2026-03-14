"""SigLIP2-IQA v2.0 model definition and configuration.

Standalone module with no Modal dependencies — can be imported locally for
testing or reused in inference scripts.

Architecture changes from v1.0:
- AttentionPool: per-dimension learnable attention over patch sequence
- 784 max patches (up from 576) for higher effective resolution
- Config-driven with YAML serialization

Reference: v1.0 model inlined at modal/benchmark_synthetic_ood.py:181-261
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


# ============================================================================
# Configuration
# ============================================================================

IQA_DIMENSIONS = ("overall", "sharpness", "color")

# v1.0 checkpoint keys for non-IQA heads (expected as unexpected when loading)
_V1_NON_IQA_HEADS = ("script", "source", "orientation", "shadow", "warping")


@dataclass(frozen=True)
class TrainingDataConfig:
    """Configuration for training data sources."""

    # Required: DIQA-5000 ground truth
    diqa_train_dir: str = "Data-DeQA-Score/DIQA/train/res/"
    diqa_meta_dir: str = "Data-DeQA-Score/DIQA/metas/"
    diqa_val_dir: str = "Data-DeQA-Score/DIQA/val/res/"

    # Optional: VLM pseudo-labeled expansion data
    pseudo_label_jsonl: str | None = None
    pseudo_label_weight: float = 0.5

    # Optional: Public dataset samples
    public_data_jsonl: str | None = None
    public_data_weight: float = 0.5

    # Mixing strategy: "interleave" | "epoch_alternate" | "weighted_sample"
    mix_strategy: str = "interleave"


@dataclass(frozen=True)
class SigLIP2V2Config:
    """Complete training configuration for SigLIP2-IQA v2.0."""

    # Architecture
    backbone_id: str = "google/siglip2-base-patch16-naflex"
    embed_dim: int = 768
    max_num_patches: int = 784
    head_hidden: int = 256
    head_dropout: float = 0.3
    use_attention_pooling: bool = True

    # Phase 1: Head warmup
    phase1_epochs: int = 10
    phase1_lr: float = 1e-3
    phase1_freeze_backbone: bool = True

    # Phase 2: Full fine-tuning
    phase2_epochs: int = 40
    phase2_lr: float = 1e-5
    phase2_scheduler: str = "cosine_warm_restarts"
    phase2_t0: int = 10
    phase2_t_mult: int = 2
    phase2_eta_min: float = 1e-7
    phase2_use_pcgrad: bool = True

    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # Loss
    loss_lambda_gnll: float = 0.5
    label_normalization: str = "linear_01"

    # Data
    data: TrainingDataConfig = field(default_factory=TrainingDataConfig)

    # Hardware
    gpu_type: str = "A10"
    seed: int = 42

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize config to YAML string, optionally writing to file."""
        d = asdict(self)
        yaml_str = yaml.safe_dump(d, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(yaml_str)
        return yaml_str

    @classmethod
    def from_yaml(cls, path: str | Path) -> SigLIP2V2Config:
        """Deserialize config from YAML file."""
        d = yaml.safe_load(Path(path).read_text())
        # Reconstruct nested TrainingDataConfig
        if "data" in d and isinstance(d["data"], dict):
            d["data"] = TrainingDataConfig(**d["data"])
        return cls(**d)


# ============================================================================
# Attention Pooling
# ============================================================================


class AttentionPool(nn.Module):
    """Learnable single-query attention pooling over patch sequence.

    Each IQA dimension gets its own AttentionPool so it can learn which
    spatial positions (text edges vs background) matter for its task.

    Args:
        embed_dim: Dimensionality of input patch embeddings.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale = embed_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention-weighted pooling over patch sequence.

        Args:
            x: Patch embeddings, shape ``(B, S, D)``.
            padding_mask: Boolean mask, shape ``(B, S)``.
                ``True`` = valid position, ``False`` = padding.
                If ``None``, all positions are treated as valid.

        Returns:
            Pooled features, shape ``(B, D)``.
        """
        # (B, 1, D) @ (B, D, S) -> (B, 1, S)
        attn_logits = torch.bmm(
            self.query.expand(x.size(0), -1, -1),
            x.transpose(1, 2),
        ) * self.scale

        if padding_mask is not None:
            # Mask padded positions to -inf so they get zero attention weight
            # padding_mask: (B, S) -> (B, 1, S)
            attn_logits = attn_logits.masked_fill(
                ~padding_mask.unsqueeze(1), float("-inf")
            )

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, 1, S)
        pooled = torch.bmm(attn_weights, x)  # (B, 1, D)
        return pooled.squeeze(1)  # (B, D)


# ============================================================================
# Model
# ============================================================================


class SigLIP2IQAv2(nn.Module):
    """SigLIP2-IQA v2.0 multi-task model.

    Architecture: SigLIP2 ViT backbone -> per-dimension attention pooling
    -> per-dimension regression heads outputting (mu, sigma_sq).

    Args:
        backbone: Pretrained SigLIP2 ViT model.
        config: Training configuration.
    """

    def __init__(self, backbone: nn.Module, config: SigLIP2V2Config) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

        d = config.embed_dim

        # Per-dimension attention pools
        if config.use_attention_pooling:
            self.attn_pools = nn.ModuleDict(
                {dim: AttentionPool(d) for dim in IQA_DIMENSIONS}
            )

        # Regression heads: Linear -> ReLU -> Dropout -> Linear
        # Matches v1.0 structure for checkpoint compatibility
        self.heads = nn.ModuleDict()
        for dim in IQA_DIMENSIONS:
            layers: list[nn.Module] = [
                nn.Linear(d, config.head_hidden),
                nn.ReLU(),
            ]
            if config.head_dropout > 0:
                layers.append(nn.Dropout(config.head_dropout))
            layers.append(nn.Linear(config.head_hidden, 2))
            self.heads[dim] = nn.Sequential(*layers)

        # Temperature buffers (per v1.0 convention)
        for dim in IQA_DIMENSIONS:
            self.register_buffer(f"temp_{dim}", torch.tensor(1.0))

    def _compute_padding_mask(
        self,
        seq_len: int,
        spatial_shapes: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build boolean padding mask from NaFlex spatial_shapes.

        Args:
            seq_len: Sequence length (S) of the hidden states.
            spatial_shapes: ``(B, 2)`` tensor of ``(h_patches, w_patches)``
                from the SigLIP2 processor.

        Returns:
            Boolean mask ``(B, S)`` where ``True`` = valid, or ``None``.
        """
        if spatial_shapes is None:
            return None
        actual_len = spatial_shapes[:, 0] * spatial_shapes[:, 1]  # (B,)
        seq_indices = torch.arange(seq_len, device=spatial_shapes.device)
        return seq_indices.unsqueeze(0) < actual_len.unsqueeze(1)  # (B, S)

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass producing per-dimension (mu, sigma_sq) predictions.

        Args:
            pixel_values: Preprocessed image tensor from SigLIP2 processor.
            spatial_shapes: NaFlex spatial shapes ``(B, 2)``.
            tasks: Subset of dimensions to compute. Defaults to all.

        Returns:
            Dict mapping dimension name to ``{"mu": (B,), "sigma_sq": (B,)}``.
        """
        active = list(IQA_DIMENSIONS) if tasks is None else tasks

        # Get backbone features
        backbone_out = self.backbone.get_image_features(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
        )

        # Extract hidden states for attention pooling or pooler_output for mean
        if self.config.use_attention_pooling:
            if hasattr(backbone_out, "last_hidden_state"):
                hidden = backbone_out.last_hidden_state  # (B, S, D)
            elif isinstance(backbone_out, torch.Tensor) and backbone_out.ndim == 3:
                hidden = backbone_out
            else:
                raise ValueError(
                    "Cannot extract last_hidden_state from backbone output. "
                    "Ensure transformers>=4.51.0 and SigLIP2 model is used."
                )
            padding_mask = self._compute_padding_mask(
                hidden.size(1), spatial_shapes
            )
        else:
            # v1.0 fallback: use pooler_output or mean pooling
            if hasattr(backbone_out, "pooler_output") and backbone_out.pooler_output is not None:
                features = backbone_out.pooler_output  # (B, D)
            elif hasattr(backbone_out, "last_hidden_state"):
                hidden = backbone_out.last_hidden_state
                mask = self._compute_padding_mask(hidden.size(1), spatial_shapes)
                if mask is not None:
                    # Masked mean pooling
                    mask_f = mask.unsqueeze(-1).float()  # (B, S, 1)
                    features = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                else:
                    features = hidden.mean(dim=1)
            elif isinstance(backbone_out, torch.Tensor):
                features = backbone_out if backbone_out.ndim == 2 else backbone_out.mean(dim=1)
            else:
                raise ValueError("Cannot extract features from backbone output.")

        results: dict[str, dict[str, torch.Tensor]] = {}
        for dim in active:
            if dim not in self.heads:
                continue

            if self.config.use_attention_pooling:
                feat = self.attn_pools[dim](hidden, padding_mask)
            else:
                feat = features

            out = self.heads[dim](feat)  # (B, 2)
            mu = out[:, 0]
            log_sigma_sq = out[:, 1]
            sigma_sq = torch.exp(log_sigma_sq)
            temp = getattr(self, f"temp_{dim}")
            results[dim] = {"mu": mu, "sigma_sq": temp * sigma_sq}

        return results

    def get_embeddings(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract 768-dim embeddings for OOD detector compatibility.

        Always uses masked mean pooling regardless of attention pooling setting
        to produce a single global embedding vector.

        Returns:
            Embeddings, shape ``(B, 768)``.
        """
        backbone_out = self.backbone.get_image_features(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
        )
        if hasattr(backbone_out, "last_hidden_state"):
            hidden = backbone_out.last_hidden_state
            mask = self._compute_padding_mask(hidden.size(1), spatial_shapes)
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            return hidden.mean(dim=1)
        if hasattr(backbone_out, "pooler_output") and backbone_out.pooler_output is not None:
            return backbone_out.pooler_output
        if isinstance(backbone_out, torch.Tensor):
            return backbone_out if backbone_out.ndim == 2 else backbone_out.mean(dim=1)
        raise ValueError("Cannot extract embeddings from backbone output.")


# ============================================================================
# Factory & Checkpoint Loading
# ============================================================================


def build_model(config: SigLIP2V2Config) -> SigLIP2IQAv2:
    """Build SigLIP2-IQA v2.0 model from config.

    Loads the pretrained backbone from HuggingFace and wraps it with
    attention pools and regression heads.

    Args:
        config: Model and training configuration.

    Returns:
        Initialized model (weights random for heads/attention pools).
    """
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(config.backbone_id)
    return SigLIP2IQAv2(backbone, config)


def build_model_from_backbone(
    backbone: nn.Module,
    config: SigLIP2V2Config,
) -> SigLIP2IQAv2:
    """Build model with an already-loaded backbone (for testing).

    Args:
        backbone: Pretrained or mock backbone module.
        config: Model and training configuration.

    Returns:
        Initialized model.
    """
    return SigLIP2IQAv2(backbone, config)


def load_v1_checkpoint(
    model: SigLIP2IQAv2,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """Load v1.0 checkpoint into v2.0 model.

    Handles key mismatches:
    - Missing keys: attention pool params (new in v2.0)
    - Unexpected keys: non-IQA heads (script, source, orientation, etc.)

    Args:
        model: Target v2.0 model.
        checkpoint_path: Path to v1.0 ``.pt`` checkpoint.
        device: Device to load weights onto.

    Returns:
        Tuple of (missing_keys, unexpected_keys) for logging.
    """
    state_dict: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    # Filter out non-IQA head keys
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    unexpected: list[str] = []
    for key, value in state_dict.items():
        is_non_iqa = any(f"heads.{h}" in key for h in _V1_NON_IQA_HEADS)
        if is_non_iqa:
            unexpected.append(key)
        else:
            filtered[key] = value

    result = model.load_state_dict(filtered, strict=False)
    missing = list(result.missing_keys)
    unexpected.extend(result.unexpected_keys)

    return missing, unexpected
