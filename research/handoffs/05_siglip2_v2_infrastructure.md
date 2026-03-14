# Handoff: SigLIP2-IQA v2.0 Training Infrastructure

**Priority**: High | **Effort**: Medium | **Est. compute cost**: ~$5-10 (validation runs only)
**Addresses**: Paper 4 Section 6.2 (Planned v2.0 Improvements)
**Depends on**: None (infrastructure only, training data decisions deferred)
**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

---

## Objective

Build the training infrastructure for SigLIP2-IQA v2.0 — implementing all Tier 1 architectural improvements and making the training pipeline configurable for expanded datasets. **Do NOT train the final model.** The team is evaluating training set expansion options (VLM pseudo-labels, public datasets) before committing to a training run. The deliverable is a ready-to-launch training script with validated components.

## Why Infrastructure First

1. **Training data decision pending**: VLM calibration (Handoff 03) and dataset expansion evaluation are in progress. The final training set may include DIQA-5000 + calibrated VLM pseudo-labels + public dataset samples.
2. **Architecture changes need validation**: Each Tier 1 improvement should be validated independently (forward pass, loss computation, gradient flow) before combining them in a full training run.
3. **Cost control**: A full 50-epoch training run costs ~$15-20 on Modal. We want confidence in the infrastructure before burning compute.

## Current v1.0 Architecture (Baseline)

Reference: [Paper 4](../diqa_4_siglip2_training.md), [benchmark_synthetic_ood.py](../../modal/benchmark_synthetic_ood.py)

| Component | v1.0 Specification |
|-----------|-------------------|
| **Backbone** | `google/siglip2-base-patch16-naflex` (86M params) |
| **Max patches** | 576 (training), 784 (inference) |
| **Pooling** | Global average pooling / pooler_output |
| **Heads** | `Linear(768->256)->ReLU->Dropout(0.3)->Linear(256->2)` per dimension |
| **Loss** | `L_NormInNorm + lambda * L_GaussianNLL` |
| **Optimizer** | AdamW |
| **Schedule** | Phase 1 (10ep, backbone frozen) + Phase 2 (40ep, all unfrozen, cosine annealing) |
| **Batch size** | Effective 16 (4 per GPU x 4 gradient accumulation) |
| **Hardware** | Modal A10 (24GB VRAM) |
| **Training time** | ~4 hours total |
| **Results** | VQualA 0.886 (Overall 0.896, Sharpness 0.869, Color 0.885) |

### v1.0 Training Script Location

The original training script lives **outside this repo**:
- Training: `image_detection/modal/train_siglip2_iqa_v2.py`
- Production model: `image_detection/src/image_preprocessing_detector/detection/siglip2_multitask.py`

The benchmark/inference code is inlined in this repo at `modal/benchmark_synthetic_ood.py` (lines 181-261).

## v2.0 Changes (Tier 1 Only)

### Change 1: Increase max_num_patches 576 -> 784

**Rationale**: Sharpness is the weakest dimension (0.869 SRCC, -0.031 gap to 0.90 target). Sharpness assessment requires fine-grained text edge analysis, limited by the 576-patch resolution ceiling (~384x384 effective). At 784 patches, effective resolution increases to ~448x448, preserving more text detail.

**Implementation**:
```python
# v1.0
inputs = processor(images=img, return_tensors="pt", max_num_patches=576, padding="max_length")

# v2.0
inputs = processor(images=img, return_tensors="pt", max_num_patches=784, padding="max_length")
```

**VRAM impact**: ~30% increase in sequence length. Verify that batch_size=4 still fits in A10 24GB. If not, reduce to batch_size=2 and increase gradient accumulation to 8 (keeping effective batch=16).

**Validation**: Run a single forward+backward pass on one batch with 784 patches. Confirm VRAM usage and gradient shapes.

### Change 2: CosineAnnealingWarmRestarts Scheduler

**Rationale**: The current cosine annealing schedule decays LR monotonically to near-zero by epoch 40. CosineAnnealingWarmRestarts periodically resets the LR, allowing the model to escape local minima and continue improving past the point where standard cosine annealing plateaus.

**Implementation**:
```python
# v1.0 (Phase 2)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)

# v2.0 (Phase 2)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First restart at epoch 10 of Phase 2
    T_mult=2,      # Second cycle is 20 epochs (10, 30 = restart points)
    eta_min=1e-7,
)
```

**Validation**: Plot the LR schedule for 40 epochs and confirm restart points at epochs 10 and 30 of Phase 2.

### Change 3: Attention Pooling Per Dimension

**Rationale**: Global average pooling treats all spatial positions equally. But sharpness depends on text regions (high-frequency patches), while color fidelity depends on background/image regions. Dimension-specific attention pooling lets each head learn which patches matter most for its task.

**Implementation**:
```python
class AttentionPool(nn.Module):
    """Learnable attention pooling over patch sequence."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) — patch embeddings from backbone.

        Returns:
            (batch, embed_dim) — attention-weighted pooled features.
        """
        # x: (B, S, D), query: (1, 1, D)
        attn_weights = torch.bmm(
            self.query.expand(x.size(0), -1, -1),  # (B, 1, D)
            x.transpose(1, 2),                       # (B, D, S)
        ) * self.scale                                # (B, 1, S)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        pooled = torch.bmm(attn_weights, x)          # (B, 1, D)
        return pooled.squeeze(1)                      # (B, D)
```

Each dimension gets its own `AttentionPool`:
```python
# v1.0 head
# features = backbone_output.pooler_output  # single global pooling

# v2.0 head — per-dimension attention before the linear layers
self.attn_pool_overall = AttentionPool(768)
self.attn_pool_sharpness = AttentionPool(768)
self.attn_pool_color = AttentionPool(768)

# In forward():
# Need last_hidden_state (B, S, 768) instead of pooler_output (B, 768)
hidden = backbone_output.last_hidden_state  # (B, S, 768)
feat_overall = self.attn_pool_overall(hidden)
feat_sharpness = self.attn_pool_sharpness(hidden)
feat_color = self.attn_pool_color(hidden)

# Then pass each to its respective head
overall_out = self.head_overall(feat_overall)
sharpness_out = self.head_sharpness(feat_sharpness)
color_out = self.head_color(feat_color)
```

**Parameter overhead**: 3 x (768 query params) = 2,304 params (~0.003% increase). Negligible.

**Validation**: Verify attention weights sum to 1 over sequence dimension. Visualize attention maps on a few sample documents to confirm sharpness head attends to text regions.

**Important**: This changes the forward pass interface — the model now needs `last_hidden_state` (full sequence) rather than `pooler_output` (single vector). Verify this works with NaFlex's variable-length sequences and padding.

### Change 4: Gradient Accumulation for PCGrad

**Rationale**: PCGrad (Project Conflicting Gradients) mitigates negative transfer in multi-task learning by projecting each task's gradient to remove components that conflict with other tasks. It requires per-task gradient computation, which needs explicit gradient accumulation control.

**Implementation**:
```python
# PCGrad requires computing gradients per task separately
# then projecting before the optimizer step

def pcgrad_step(model, optimizer, task_losses: dict[str, torch.Tensor]):
    """PCGrad update: project conflicting gradients between tasks."""
    task_grads = {}
    for task_name, loss in task_losses.items():
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        task_grads[task_name] = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]

    # Project gradients
    task_names = list(task_grads.keys())
    for i, name_i in enumerate(task_names):
        for j, name_j in enumerate(task_names):
            if i == j:
                continue
            grads_i = task_grads[name_i]
            grads_j = task_grads[name_j]
            for gi, gj in zip(grads_i, grads_j):
                dot = (gi * gj).sum()
                if dot < 0:
                    gi -= (dot / (gj.norm() ** 2 + 1e-8)) * gj

    # Apply projected gradients
    optimizer.zero_grad()
    for params, *grad_lists in zip(model.parameters(), *task_grads.values()):
        if params.requires_grad:
            params.grad = sum(grad_lists) / len(grad_lists)
    optimizer.step()
```

**VRAM impact**: Requires `retain_graph=True` and storing per-task gradients — roughly 3x gradient memory (3 IQA tasks). With 784 patches + attention pooling, verify this fits in A10 24GB. May need to reduce batch_size to 2 with gradient_accumulation=8.

**Validation**: Run 10 training steps with PCGrad on a small data subset. Verify that per-task SRCC doesn't degrade vs standard gradient averaging. Log gradient norms and conflict counts.

**Phase gating**: PCGrad should only activate in Phase 2 (full fine-tuning). Phase 1 (head warmup) uses standard gradient averaging since the backbone is frozen and there's no cross-task gradient conflict in the shared representation.

## Training Data Interface

The training script must support **pluggable data sources** since the final training set is TBD:

```python
@dataclass(frozen=True)
class TrainingDataConfig:
    """Configuration for training data sources."""

    # Required: DIQA-5000 ground truth
    diqa_train_dir: str = "Data-DeQA-Score/DIQA/train/res/"
    diqa_meta_dir: str = "Data-DeQA-Score/DIQA/metas/"
    diqa_val_dir: str = "Data-DeQA-Score/DIQA/val/res/"

    # Optional: VLM pseudo-labeled expansion data
    pseudo_label_jsonl: str | None = None  # Path to calibrated pseudo-labels
    pseudo_label_weight: float = 0.5       # Loss weight for pseudo-labeled samples

    # Optional: Public dataset samples
    public_data_jsonl: str | None = None   # Path to public dataset labels
    public_data_weight: float = 0.5        # Loss weight for public samples

    # Mixing strategy
    mix_strategy: str = "interleave"  # "interleave" | "epoch_alternate" | "weighted_sample"
```

**Data format for expansion sources** (JSONL):
```json
{
  "image_path": "/path/to/image.jpg",
  "overall": 3.45,
  "sharpness": 3.12,
  "color_fidelity": 3.67,
  "source": "vlm_pseudo_label",
  "confidence": 0.85,
  "labeler": "gemini-3-flash-calibrated"
}
```

The training loop should:
1. Always include DIQA-5000 at full weight (human ground truth)
2. Optionally mix in pseudo-labeled data with configurable weight
3. Support confidence-weighted loss for pseudo-labels (higher confidence = higher weight)

## Model Configuration

Create a config dataclass that captures all v2.0 hyperparameters:

```python
@dataclass(frozen=True)
class SigLIP2V2Config:
    """Complete training configuration for SigLIP2-IQA v2.0."""

    # Architecture
    backbone_id: str = "google/siglip2-base-patch16-naflex"
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
    phase2_scheduler: str = "cosine_warm_restarts"  # "cosine" | "cosine_warm_restarts"
    phase2_t0: int = 10          # CosineAnnealingWarmRestarts T_0
    phase2_t_mult: int = 2       # CosineAnnealingWarmRestarts T_mult
    phase2_eta_min: float = 1e-7
    phase2_use_pcgrad: bool = True

    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    batch_size: int = 4           # Per-GPU
    gradient_accumulation: int = 4  # Effective batch = batch_size * gradient_accumulation
    max_grad_norm: float = 1.0

    # Loss
    loss_lambda_gnll: float = 0.5  # Weight for GaussianNLLLoss
    label_normalization: str = "linear_01"  # (MOS - 1) / 4 -> [0, 1]

    # Data
    data: TrainingDataConfig = field(default_factory=TrainingDataConfig)

    # Hardware
    gpu_type: str = "A10"  # Modal GPU type
    seed: int = 42
```

Save configs as YAML alongside checkpoints for reproducibility.

## Deliverables

### 1. Training script: `modal/train_siglip2_iqa_v2.py`

Modal-native training script implementing:
- [x] v2.0 model architecture (attention pooling, 784 patches)
- [x] Two-phase training with CosineAnnealingWarmRestarts
- [x] PCGrad multi-task optimization (Phase 2 only)
- [x] Pluggable data sources (DIQA-5000 + optional expansion data)
- [x] Config-driven (SigLIP2V2Config dataclass, YAML serialization)
- [x] Checkpoint saving (best val wSRCC + periodic)
- [x] WandB logging (loss, per-dimension SRCC, LR, gradient norms)
- [x] Resume from checkpoint support

### 2. Model module: `modal/siglip2_v2_model.py`

Clean model definition with:
- `AttentionPool` module
- `SigLIP2IQAv2` model class (backbone + attention pools + heads)
- `build_model(config: SigLIP2V2Config)` factory function
- Forward pass returning per-task outputs with mu/sigma_sq

### 3. Data module: `modal/siglip2_v2_data.py`

Dataset and dataloader with:
- DIQA-5000 dataset (existing format)
- Pseudo-label dataset (JSONL format)
- Mixed dataset with configurable weighting
- Proper train/val split handling

### 4. PCGrad module: `modal/pcgrad.py`

Standalone PCGrad implementation:
- `pcgrad_step()` function
- Gradient conflict logging (count and magnitude of projections)
- Unit test: verify gradient projection correctness on synthetic 2-task loss

### 5. Validation script: `modal/validate_siglip2_v2.py`

Script that validates all infrastructure components WITHOUT full training:

```bash
# Validate forward pass + VRAM with 784 patches
uv run modal run modal/validate_siglip2_v2.py --check forward

# Validate attention pooling outputs
uv run modal run modal/validate_siglip2_v2.py --check attention

# Validate PCGrad on 10 steps
uv run modal run modal/validate_siglip2_v2.py --check pcgrad

# Validate scheduler produces expected LR curve
uv run modal run modal/validate_siglip2_v2.py --check scheduler

# Validate data loading with expansion sources
uv run modal run modal/validate_siglip2_v2.py --check data --pseudo-labels /path/to/test.jsonl

# Run all checks
uv run modal run modal/validate_siglip2_v2.py --check all
```

### 6. Config files: `modal/configs/`

```
modal/configs/
  siglip2_v2_diqa_only.yaml       # DIQA-5000 only (baseline comparison)
  siglip2_v2_expanded.yaml         # DIQA-5000 + pseudo-labels (template, data TBD)
```

### 7. Migration test: `modal/test_v2_backward_compat.py`

Verify that:
- v1.0 checkpoint can be loaded into v2.0 model (with expected missing keys for attention pools)
- v2.0 model with attention pooling disabled produces identical outputs to v1.0
- Embedding extraction produces compatible 768-dim vectors for OOD detector

## Technical Notes

### VRAM Budget (A10, 24GB)

Estimate VRAM with 784 patches:

| Component | v1.0 (576 patches) | v2.0 (784 patches) |
|-----------|-------------------|-------------------|
| Backbone params | ~344 MB (fp32) | ~344 MB (same) |
| Backbone activations (batch=4) | ~1.5 GB | ~2.0 GB (+33%) |
| Head params | ~1.2 MB | ~1.2 MB |
| Attention pool params | 0 | ~9 KB |
| Gradients | ~344 MB | ~344 MB |
| Optimizer state (AdamW) | ~688 MB | ~688 MB |
| PCGrad gradient storage (3x) | 0 | ~1.0 GB |
| **Total estimate** | ~3.0 GB | ~4.4 GB |

Fits comfortably in 24GB even with PCGrad. AMP (fp16 forward, fp32 backward) can reduce further if needed.

If batch_size=4 with 784 patches causes OOM, fall back to batch_size=2 with gradient_accumulation=8 (same effective batch of 16).

### transformers Version

SigLIP2 NaFlex requires `transformers>=4.51.0`. The existing Modal image in `benchmark_synthetic_ood.py` already uses this. Pin to `transformers>=4.51.0,<5.0.0` for stability.

### Checkpoint Compatibility

The v1.0 checkpoint (`siglip2_iqa_best.pt`) has this state_dict structure:
```
backbone.*                    # ViT weights
heads.overall.0.weight        # Linear(768, 256)
heads.overall.0.bias
heads.overall.2.weight        # Linear(256, 2) (index 2 because ReLU at index 1 has no params)
heads.overall.2.bias          # ... but Dropout at index 2 also has no params,
                              # so the second Linear is at index 3 if Dropout is included
heads.sharpness.*
heads.color.*
heads.script.*                # Non-IQA heads (22 missing keys in IQA-only mode)
heads.source.*
heads.orientation.*
temp_overall                  # Temperature buffer
temp_sharpness
temp_color
```

The v2.0 model adds `attn_pool_{overall,sharpness,color}` — these will be `unexpected` when loading v1.0 checkpoint. Initialize them randomly and let Phase 1 train them from scratch.

### NaFlex and Attention Pooling Interaction

With NaFlex, the sequence length varies per image (padded to `max_num_patches`). The attention pooling must handle padding correctly:
- Padded positions should be masked (set attention weight to -inf before softmax)
- The `spatial_shapes` tensor from the processor indicates actual vs padded positions
- Verify this works correctly by comparing attention-pooled output against manually-masked mean pooling

### WandB Logging

Log per training step:
- Total loss, NormInNorm loss, GaussianNLL loss
- Per-dimension loss breakdown
- Learning rate
- Gradient norm (global and per-task)
- PCGrad conflict count and projection magnitude (Phase 2 only)

Log per validation epoch:
- Per-dimension SRCC, PLCC, MAE on val set
- wSRCC (VQualA MainScore formula)
- Mean sigma_sq per dimension (uncertainty calibration)

## Non-Goals (Explicitly Deferred)

| Item | Reason | When |
|------|--------|------|
| **Train the v2.0 model** | Training data decision pending | After training set expansion evaluation |
| **Tier 2 improvements** (dropout reduction, wider heads, LLRD, MarginRankingLoss) | Lower priority, smaller expected impact | After Tier 1 validation |
| **ONNX export** | Production deployment concern | After v2.0 training validates improvement |
| **Multi-GPU training** | A10 single-GPU is sufficient for 3.5-10K samples | If dataset grows past ~50K |

## Definition of Done

- [ ] v2.0 model builds and runs forward pass with 784 patches on Modal A10
- [ ] Attention pooling produces per-dimension features from patch sequence
- [ ] Attention pooling correctly handles NaFlex padding
- [ ] CosineAnnealingWarmRestarts LR schedule validated (plot saved)
- [ ] PCGrad gradient projection verified on 10 training steps
- [ ] v1.0 checkpoint loads into v2.0 model (with expected missing keys)
- [ ] Data pipeline accepts DIQA-5000 + optional pseudo-label JSONL
- [ ] Config YAML serialization/deserialization works
- [ ] Validation script passes all checks (`--check all`)
- [ ] VRAM usage documented for 784 patches + PCGrad on A10
- [ ] No training run executed (infrastructure only)
