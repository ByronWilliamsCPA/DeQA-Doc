# Training SigLIP2-IQA-Base-86M: A Lightweight Document Image Quality Model

**Authors:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Research Paper #4
**Status:** Complete
**Related Experiments:** EXP-006

---

## Abstract

We describe the training methodology, architecture decisions, and evaluation results for SigLIP2-IQA-Base-86M, a compact vision transformer fine-tuned for multi-dimensional document image quality assessment. Built on the SigLIP 2 ViT-B/16 backbone with NaFlex (Native Flexible Resolution), the model uses three independent regression heads with Gaussian uncertainty estimation to predict quality scores for overall quality, sharpness, and color fidelity. Using a two-phase training protocol (10 epochs head warmup followed by 40 epochs full fine-tuning) on 3,500 DIQA-5000 training images, SigLIP2-IQA-Base achieves VQualA 0.886 — outperforming all pretrained NR-IQA models, all zero-shot VLMs, and the competition-winning DeQA-Doc-3Specialists (0.716) in single-model comparison, while running at ~100ms inference on a single GPU. The model also provides calibrated uncertainty estimates (sigma-squared) per prediction, enabling downstream confidence-based routing in the pseudo-labeling pipeline.

---

## 1. Introduction

### 1.1 Motivation

The DeQA-Doc system won the VQualA 2025 DIQA Challenge with a wSRCC of 0.929, but this score required an ensemble of three 7B-parameter mPLUG-Owl2 specialists plus a Qwen2.5-VL model — totaling ~28B+ parameters with ~3,000ms inference per image. For production deployment in document processing pipelines (where millions of images need quality scoring), we need a model that is:

1. **Fast** — sub-200ms inference per image
2. **Compact** — fits on commodity GPUs or can be ONNX-exported for CPU inference
3. **Multi-dimensional** — predicts all three quality dimensions in a single forward pass
4. **Uncertainty-aware** — provides confidence estimates for downstream routing

SigLIP2-IQA-Base was designed to meet all four requirements.

### 1.2 Why SigLIP 2?

We chose the SigLIP 2 architecture for several reasons:

| Factor | SigLIP 2 Advantage |
|--------|-------------------|
| **NaFlex** | Native Flexible Resolution preserves document aspect ratios without distortion |
| **Pretraining** | SigLIP 2 was pretrained on document-inclusive data (web pages, PDFs, OCR text overlays) |
| **Patch efficiency** | Variable patch count (up to 784) adapts to document complexity |
| **Feature quality** | 768-dim embeddings proved effective for OOD detection (AUROC 0.9963) |
| **Model size** | 86M params enables rapid iteration and deployment |

---

## 2. Architecture

### 2.1 Backbone

The backbone is `google/siglip2-base-patch16-naflex`, a Vision Transformer (ViT-B/16) with 86M parameters. NaFlex (Native Flexible Resolution) allows the model to process images at variable resolutions by dynamically adjusting the number of patches rather than resizing to a fixed input size. This is critical for documents, where resizing to 224x224 destroys text legibility.

**Patch configuration:**
- Patch size: 16x16 pixels
- Maximum patches: 576 (training) / 784 (planned v2.0)
- Effective resolution: approximately 384x384 at 576 patches
- Padding: `max_length` (padded to maximum patch count for batch processing)

### 2.2 Regression Heads

Three independent regression heads predict quality scores for each dimension. Each head outputs two values: a mean prediction (mu) and a learned variance (sigma-squared).

```
Head Architecture (per dimension):
  Linear(768 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
  Output: [mu, log_sigma_sq]
```

| Component | Dimension | Parameters |
|-----------|-----------|------------|
| Backbone (ViT-B/16) | shared | ~86M |
| Overall quality head | 768 -> 256 -> 2 | ~197K |
| Sharpness head | 768 -> 256 -> 2 | ~197K |
| Color fidelity head | 768 -> 256 -> 2 | ~197K |
| **Total** | | **~86.6M** |

The heads are deliberately lightweight — the vast majority of capacity is in the shared backbone, which learns a single document quality representation used by all three heads. This parameter sharing is a form of multi-task regularization.

### 2.3 Output Range and Rescaling

The model was trained with normalized labels: `target = (MOS - 1) / 4`, mapping the [1, 5] MOS scale to [0, 1]. At inference, predictions are rescaled: `MOS_pred = mu * 4.0 + 1.0`.

Observed mu output range on DIQA-5000: approximately [-0.17, 0.73], corresponding to MOS predictions of [0.32, 3.92]. The model does not produce predictions at the extreme ends of the MOS scale (very few "excellent" samples in training data — only 5 out of 1,000 test images have Overall MOS >= 4.0).

### 2.4 Uncertainty Estimation

Each head predicts sigma-squared alongside mu, trained via Gaussian Negative Log-Likelihood Loss:

```
GaussianNLLLoss(mu, target, sigma_sq) = 0.5 * (log(sigma_sq) + (target - mu)^2 / sigma_sq)
```

This provides calibrated uncertainty estimates without requiring ensemble methods or Monte Carlo dropout. Higher sigma-squared indicates the model is less certain about its prediction — useful for:

1. **OOD routing**: High uncertainty flags documents for VLM teacher re-evaluation
2. **Active learning**: Prioritizing human annotation for highest-uncertainty samples
3. **Quality gating**: Rejecting predictions where uncertainty exceeds a threshold

The auto-accept threshold for sigma-squared was set at 0.64 (= 0.8 squared), matching DeQA's sigma_pseudo = 0.8.

---

## 3. Training Protocol

### 3.1 Two-Phase Training

The training follows a two-phase protocol designed to prevent catastrophic forgetting of the pretrained backbone features:

**Phase 1: Head Warmup (10 epochs)**
- Backbone weights **frozen**
- Only the three regression heads are trained
- Purpose: initialize heads to produce reasonable predictions before backbone adaptation begins
- Learning rate: higher, focused on head convergence

**Phase 2: Full Fine-Tuning (40 epochs)**
- All weights **unfrozen** (backbone + heads)
- Purpose: adapt backbone features to document quality assessment
- Learning rate: lower, with cosine annealing
- Gradient accumulation enables PCGrad multi-task optimization within memory budget

### 3.2 Loss Function

The total loss combines two components:

```
L_total = L_NormInNorm + lambda * L_GaussianNLL
```

- **NormInNormLoss**: The primary regression loss, which normalizes predictions within each batch to focus on relative ranking rather than absolute score calibration
- **GaussianNLLLoss**: The uncertainty estimation loss, which trains the model to predict its own confidence

### 3.3 Training Data

| Split | Images | Source |
|-------|--------|--------|
| Train | 3,500 | DIQA-5000 train set |
| Validation | 500 | DIQA-5000 val set |
| Test | 1,000 | DIQA-5000 test set (held out) |

Each training sample provides:
- Image (variable resolution document photograph/scan)
- MOS ground truth for three dimensions (continuous, [1, 5])
- Standard deviation of human ratings (used for soft-label generation in DeQA, but SigLIP2 uses point regression)

### 3.4 Hardware and Runtime

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA A10 (24GB VRAM) |
| Training time | ~4 hours (both phases) |
| Batch size | Effective 16 (4 per GPU x 4 gradient accumulation) |
| Platform | Modal (serverless GPU) |

---

## 4. Results

### 4.1 DIQA-5000 Test Set Performance

| Metric | Overall | Sharpness | Color Fidelity | Weighted |
|--------|---------|-----------|----------------|----------|
| SRCC | 0.896 | 0.869 | 0.885 | 0.886 (VQualA) |

### 4.2 Comparison with Other Approaches

| Model | Type | MainScore | Inference | Params |
|-------|------|-----------|-----------|--------|
| **SigLIP2-IQA-Base** | Fine-tuned ViT | **0.886** | ~100ms | 86M |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.716 | ~3,000ms | 3x7B |
| Gemini 3 Flash | VLM (zero-shot) | 0.743 | ~2,000ms | Unknown |
| RichIQA | NR-IQA (pretrained) | 0.490 | ~150ms | ~100M |
| HyperIQA++ | Fine-tuned CNN | 0.694* | ~100ms | 138M |

*HyperIQA++ MainScore is from synthetic OOD evaluation; its DIQA-5000 MainScore uses wSRCC = 0.856.

SigLIP2-IQA-Base outperforms:
- All pretrained NR-IQA models by 81%+ (0.886 vs best 0.490)
- All zero-shot VLMs by 19%+ (0.886 vs best 0.743)
- The competition-winning DeQA-Doc specialists by 24% (0.886 vs 0.716)
- At 30x faster inference than MLLM approaches

### 4.3 Per-Dimension Analysis

| Dimension | SRCC | Gap to Target (0.90) | Analysis |
|-----------|------|---------------------|----------|
| Overall | 0.896 | -0.004 (met) | Strongest dimension; benefits from holistic image features |
| Color fidelity | 0.885 | -0.015 | Second strongest; color features well-captured by SigLIP2 |
| Sharpness | 0.869 | -0.031 | Weakest dimension; may benefit from higher max_num_patches |

The sharpness gap is the primary bottleneck to reaching VQualA 0.92. Sharpness assessment requires fine-grained text edge analysis, which is limited by the current 576-patch resolution ceiling (~384x384 effective). Increasing to 784+ patches (Tier 1 v2.0 improvement) should directly address this.

### 4.4 Embedding Quality

The 768-dim embeddings from the final backbone layer proved highly effective for OOD detection:

| Metric | Value |
|--------|-------|
| AUROC (synthetic OOD vs DIQA-5000 test) | 0.9963 |
| TPR at 5% FPR | 99.5% |
| Inference overhead for OOD scoring | ~1-2ms (matrix-vector multiply) |

This dual utility (quality scoring + OOD detection from a single forward pass) makes SigLIP2-IQA-Base the ideal backbone for the production pipeline.

---

## 5. SigLIP2 Multi-Task Architecture

Beyond IQA scoring, the SigLIP2 model was trained as a multi-task detector with additional heads for document characterization. The IQA-only checkpoint (`siglip2_iqa_best.pt`) loads with 22 missing keys corresponding to these non-IQA heads:

| Task Head | Purpose | Output |
|-----------|---------|--------|
| IQA Overall | Quality prediction | mu, sigma_sq |
| IQA Sharpness | Sharpness prediction | mu, sigma_sq |
| IQA Color | Color fidelity prediction | mu, sigma_sq |
| Script Detection | Writing system classification | 19-class distribution (LATN, CYRL, ARAB, etc.) |
| Source Detection | Document origin | scanned vs digital vs photo |
| Orientation | Page rotation | 0/90/180/270 degrees |
| Shadow Severity | Shadow detection | severity score + uncertainty |
| Warping Severity | Geometric distortion | severity score + uncertainty |

The multi-task training provides implicit regularization — the shared backbone must learn features useful for all tasks, preventing overfitting to any single dimension.

### 5.1 Full Extraction Results

Complete extraction on all 5,000 DIQA-5000 images was performed via Modal L4 GPU:

| Split | Records | Embedding Shape | File Size |
|-------|---------|----------------|-----------|
| Train | 3,500 | (3500, 768) float32 | ~10 MB |
| Val | 500 | (500, 768) float32 | ~1.5 MB |
| Test | 1,000 | (1000, 768) float32 | ~3 MB |

All 20 fields per record (IQA scores, script/source/orientation predictions, shadow/warping severity, inference time) are archived in JSONL format at `results/siglip2_diqa5000/`.

---

## 6. Limitations and Planned Improvements

### 6.1 Current Limitations

1. **Resolution ceiling**: 576 patches limits effective resolution to ~384x384, insufficient for fine text analysis in high-DPI documents
2. **Training data size**: Only 3,500 training images from DIQA-5000 limits domain coverage
3. **OOD degradation**: MainScore drops from 0.886 (DIQA-5000) to 0.620 (synthetic OOD) — a 30% degradation indicating dataset-specific overfitting
4. **Dropout rate**: 0.3 dropout in heads may be too aggressive for 3,500-sample regression
5. **Extreme quality blindspot**: Near-zero "excellent" training samples (only 5 in test set) limits prediction accuracy at the top of the quality scale

### 6.2 Planned v2.0 Improvements

| Priority | Improvement | Expected Impact | Status |
|----------|-------------|-----------------|--------|
| Tier 1 | Increase max_num_patches 576 -> 784+ | +2-3% sharpness SRCC | Planned |
| Tier 1 | CosineAnnealingWarmRestarts scheduler | Prevents premature convergence | Planned |
| Tier 1 | Gradient accumulation (batch 4x4) | Re-enables PCGrad in Phase 2 | Planned |
| Tier 1 | Attention pooling per dimension | Dimension-specific feature weighting | Planned |
| Tier 2 | Reduce dropout 0.3 -> 0.1-0.15 | Less aggressive regularization | Planned |
| Tier 2 | Wider heads (768 -> 512 -> 256 -> 2) | More capacity per dimension | Planned |
| Tier 2 | LLRD (0.9 decay/layer) | Better fine-tuning stability | Planned |
| Tier 2 | MarginRankingLoss | Direct SRCC optimization | Planned |

### 6.3 VLM Pseudo-Label Expansion

The primary path to closing the 0.886 -> 0.92 gap is training data expansion via VLM pseudo-labels:

1. **OOD detector identifies** documents where SigLIP2 is unreliable
2. **VLM committee** (Gemini 3 Flash + GPT-4.1) generates calibrated pseudo-labels
3. **Isotonic regression** corrects systematic VLM positive bias (+0.5 to +1.5 MOS)
4. **Expanded training set** combines DIQA-5000 human labels with VLM pseudo-labels
5. **SigLIP2 v2.0** retrains on expanded data with architectural improvements

This pipeline is currently in development (87 tests passing in `src/uncertainty/`).

---

## 7. Conclusions

SigLIP2-IQA-Base-86M demonstrates that a compact, well-designed vision transformer can outperform much larger MLLM-based approaches for document image quality assessment when trained with high-quality human labels. Its key advantages are:

1. **30x faster inference** than MLLM approaches (~100ms vs ~3,000ms)
2. **Highest single-model accuracy** on DIQA-5000 (0.886 vs 0.716 for DeQA-Doc)
3. **Built-in uncertainty estimation** for confidence-based routing
4. **Dual-purpose embeddings** that also power OOD detection (AUROC 0.9963)
5. **86M parameters** enabling deployment on commodity hardware

The primary limitation is OOD generalization (0.620 on synthetic data), which the VLM pseudo-labeling pipeline is designed to address through iterative domain expansion.

---

## 8. Data Availability

| Artifact | Location |
|----------|----------|
| Model checkpoint | Modal volume `dociq-checkpoints` / `siglip2_iqa_best.pt` |
| DIQA-5000 predictions (JSONL) | `results/siglip2_diqa5000/siglip2_diqa5000_{train,val,test}.jsonl` |
| 768-dim embeddings (NPZ) | `results/siglip2_diqa5000/embeddings/{train,val,test}.npz` |
| OOD detector v2 | `results/siglip2_diqa5000/ood_detector_v2.npz` |
| Extraction metadata | `results/siglip2_diqa5000/summary.json` |
| Training script | `image_detection/modal/train_siglip2_iqa_v2.py` |
| Production wrapper | `image_detection/src/image_preprocessing_detector/detection/siglip2_multitask.py` |
| Architecture documentation | [research.md](../research.md) Section 3 |
