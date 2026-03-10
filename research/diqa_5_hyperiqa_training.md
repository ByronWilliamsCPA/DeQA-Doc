# Training HyperIQA++: Document-Specific Fine-Tuning of a CNN-Based IQA Model

**Authors:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Research Paper #5
**Status:** Complete
**Related Experiments:** EXP-003, EXP-007

---

## Abstract

We describe HyperIQA++, a document-specific fine-tuning of the HyperIQA architecture for multi-dimensional document image quality assessment. Starting from HyperIQA's ResNet-50 + HyperNet backbone (pretrained on KonIQ-10k natural images), we add spatial attention and 10-bin soft-label distribution heads for overall quality, sharpness, and color fidelity. After fine-tuning on 3,500 DIQA-5000 training images, HyperIQA++ achieves wSRCC 0.856 on DIQA-5000 (up from 0.437 MainScore off-the-shelf) — a 96% improvement. However, evaluation on the 520-image synthetic OOD dataset reveals a significant generalization gap: MainScore drops to 0.694, with the ID/OOD delta of -0.165 being the largest among fine-tuned models. We analyze the training methodology, characterize the model's strengths and weaknesses relative to VLM and ViT-based approaches, and discuss the implications for CNN-based document quality assessment.

---

## 1. Introduction

### 1.1 Background: HyperIQA

HyperIQA (Su et al., 2020) is a no-reference image quality assessment model that uses a HyperNetwork to generate content-adaptive quality prediction weights. Unlike standard CNN regressors that apply fixed learned weights to all images, HyperIQA:

1. Extracts multi-scale features via a ResNet-50 backbone
2. Generates image-specific prediction weights through a HyperNetwork conditioned on the content
3. Applies these adaptive weights to produce a quality score

This content-adaptive mechanism is particularly relevant for documents, where quality perception depends heavily on content type (dense text vs. sparse forms, photographs vs. line drawings).

### 1.2 Motivation for HyperIQA++

Off-the-shelf HyperIQA achieves only MainScore 0.437 on DIQA-5000 — below the zero-shot VLM baseline (0.743 for Gemini 3 Flash). However, reported competition results show fine-tuned HyperIQA reaching 0.844, suggesting the architecture has substantial untapped capacity for document quality assessment.

HyperIQA++ extends the base architecture with:
1. **Spatial attention** for layout-aware quality weighting
2. **Multi-dimensional output** (3 quality dimensions instead of 1)
3. **Soft-label distribution learning** (10-bin distributions per dimension)
4. **High-resolution input** (1600x1600) for fine-grained text analysis

---

## 2. Architecture

### 2.1 Base Architecture

| Component | Specification |
|-----------|--------------|
| Backbone | ResNet-50 (pretrained on ImageNet, further trained on KonIQ-10k) |
| HyperNet | Content-adaptive weight generator |
| Input resolution | 1600 x 1600 x 3 (critical — much higher than standard IQA) |
| Normalization | ImageNet standard: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Total parameters | ~138M |

### 2.2 Spatial Attention (DocIQ-Simplified)

The key architectural addition is a spatial attention mechanism inspired by document layout analysis. Documents have spatially varying quality importance — text regions matter more for sharpness assessment than margins, while color backgrounds matter more for color fidelity assessment.

The spatial attention module learns to weight different image regions based on their content, providing a document-specific inductive bias that the original HyperIQA lacks.

### 2.3 Soft-Label Distribution Heads

Instead of predicting a single scalar quality score, HyperIQA++ predicts a 10-bin probability distribution per dimension:

```
Output per dimension: softmax(Linear(features) -> 10)
```

The 10 bins span the [1, 5] MOS range with uniform spacing (bin width = 0.4). The expected value of the distribution gives the predicted MOS:

```
MOS_pred = sum(prob_i * center_i for i in range(10))
```

This soft-label approach captures the inherent ambiguity in quality perception — a document that 8 of 15 raters call "fair" and 7 call "good" has a different quality profile than one that all 15 call "fair", even if both have MOS ~3.0.

---

## 3. Training Protocol

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training data | DIQA-5000 train (3,500 images) |
| Input size | 1600 x 1600 (with aspect ratio preservation) |
| Optimizer | AdamW |
| Training method | Full fine-tuning (all layers) |
| Loss function | Cross-entropy over 10-bin soft-label distributions |
| Platform | Modal (serverless GPU) |
| GPU | NVIDIA A10 (24GB) / L4 (24GB) |

### 3.2 Label Generation

For each training image, the continuous MOS ground truth is converted to a 10-bin soft-label distribution. Given MOS = mu and rater standard deviation = sigma:

1. Define 10 bin centers uniformly in [1.0, 5.0]
2. Compute Gaussian PDF at each bin center: `p_i = N(center_i; mu, sigma^2)`
3. Normalize to sum to 1: `prob_i = p_i / sum(p_j)`

This preserves both the mean quality and the uncertainty of human ratings in the training signal.

### 3.3 Multi-Dimensional Training

HyperIQA++ trains three separate heads simultaneously. Each head receives the same backbone features but produces an independent 10-bin distribution. The total loss is:

```
L_total = L_CE(overall) + L_CE(sharpness) + L_CE(color)
```

All three losses are weighted equally. No pairwise ranking loss is used (unlike DeQA-Score).

---

## 4. Results

### 4.1 DIQA-5000 Performance

| Metric | Value | Comparison |
|--------|-------|------------|
| wSRCC | 0.856 | +96% vs off-the-shelf HyperIQA (0.437) |
| SRCC Overall | — | — |
| PLCC Overall | 0.886 | Strong linear correlation after logistic fitting |
| MAE | 2.225 | **Suspiciously high** — see Section 4.3 |

### 4.2 Comparison with Other Models on DIQA-5000

| Model | Type | MainScore/wSRCC | Params | Inference |
|-------|------|----------------|--------|-----------|
| SigLIP2-IQA-Base | Fine-tuned ViT | **0.886** | 86M | ~100ms |
| HyperIQA++ | Fine-tuned CNN | 0.856 | 138M | ~100ms |
| Gemini 3 Flash | VLM (zero-shot) | 0.743 | Unknown | ~2,000ms |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.716 | 3x7B | ~3,000ms |
| HyperIQA (off-the-shelf) | NR-IQA baseline | 0.437 | ~25M | ~100ms |

HyperIQA++ substantially outperforms all zero-shot VLMs and the DeQA-Doc specialists, but falls 3 points below SigLIP2-IQA-Base despite having 60% more parameters. The SigLIP2 advantage likely comes from:
- NaFlex resolution handling (adaptive patches vs fixed 1600x1600 resize)
- ViT attention mechanisms (global context vs CNN local receptive fields)
- SigLIP 2's document-inclusive pretraining

### 4.3 The MAE Anomaly

HyperIQA++ shows MAE = 2.225 on DIQA-5000, which is very high for a model with PLCC = 0.886. For context:

| Model | PLCC | MAE |
|-------|------|-----|
| HyperIQA++ | 0.886 | **2.225** |
| Gemini 3 Flash | 0.792 | 0.91 |
| Claude Haiku 4.5 | 0.650 | **0.68** |

This indicates a **systematic scale offset** — the model's output distribution is shifted relative to the true MOS scale. The high PLCC and SRCC show the model preserves ranking and linear relationships, but the absolute predicted values are miscalibrated.

**Root cause hypothesis:** The 10-bin soft-label expected value may not be properly calibrated to the [1, 5] MOS range. The bin center definitions or the output rescaling may introduce a constant offset. This was flagged in the synthetic OOD handoff document as a verification requirement.

**Impact:** For correlation-based metrics (SRCC, PLCC), this is irrelevant — the model ranks documents correctly. For absolute quality scoring (MAE, direct MOS prediction), post-hoc calibration (isotonic regression or simple linear rescaling) would be required.

---

## 5. Synthetic OOD Performance (n=520)

### 5.1 Overall Results

| Subset | MainScore | SRCC_O | PLCC_O | SRCC_S | SRCC_C |
|--------|-----------|--------|--------|--------|--------|
| All (n=520) | 0.694 | 0.589 | 0.780 | 0.623 | 0.606 |
| In-distribution (n=150) | 0.840 | — | — | — | — |
| Out-of-distribution (n=370) | 0.675 | — | — | — | — |

### 5.2 ID/OOD Gap Analysis

| Model | Type | MainScore (ID) | MainScore (OOD) | Delta |
|-------|------|---------------|----------------|-------|
| HyperIQA++ | Fine-tuned CNN | 0.840 | 0.675 | **-0.165** |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.842 | 0.746 | -0.096 |
| SigLIP2-IQA-Base | Fine-tuned ViT | 0.659 | 0.663 | +0.004 |
| Gemini 3 Flash | VLM (zero-shot) | 0.824 | 0.782 | -0.042 |

HyperIQA++ shows the **largest ID/OOD gap** (-0.165) among fine-tuned models. This is expected for a CNN architecture:

1. **Fixed receptive field**: ResNet-50's fixed receptive fields learn DIQA-5000-specific spatial patterns that don't transfer to OOD document types
2. **No semantic understanding**: Unlike MLLMs, the CNN has no concept of text legibility or document layout
3. **Resolution bias**: The 1600x1600 fixed input may not handle extreme DPI variations well

### 5.3 The Off-the-Shelf Paradox

Surprisingly, off-the-shelf HyperIQA (pretrained on KonIQ-10k only, no DIQA fine-tuning) achieves MainScore 0.723 on synthetic OOD — **higher** than the fine-tuned HyperIQA++ (0.694):

| Model | DIQA MainScore | OOD MainScore | Direction |
|-------|---------------|---------------|-----------|
| HyperIQA (off-the-shelf) | 0.437 | 0.723 | Better OOD |
| HyperIQA++ (fine-tuned) | 0.856 | 0.694 | Worse OOD |

Fine-tuning on DIQA-5000 improved DIQA-5000 performance by 96% but **degraded** OOD performance by 4%. This is a textbook case of catastrophic forgetting — the model specialized to DIQA-5000's distortion types and document characteristics at the expense of general quality assessment capability.

This finding has direct implications for the pseudo-labeling strategy: expanding training data beyond DIQA-5000 is not optional — it is necessary to prevent overfitting that hurts real-world generalization.

### 5.4 PLCC vs SRCC Pattern

HyperIQA++ shows a consistent pattern where PLCC substantially exceeds SRCC across all dimensions:

| Dimension | SRCC | PLCC | Gap |
|-----------|------|------|-----|
| Overall | 0.589 | 0.780 | +0.191 |
| Sharpness | 0.623 | 0.797 | +0.174 |
| Color | 0.606 | 0.790 | +0.184 |

The large SRCC-PLCC gap indicates that the 4-parameter logistic curve fitting is compensating for nonlinear prediction-to-MOS relationships. The model's raw outputs preserve relative quality information but don't monotonically track MOS — there are regions where the model's score function has inversions that PLCC's nonlinear fitting corrects.

---

## 6. Comparison with Competition Results

The VQualA 2025 competition provides context for HyperIQA++'s performance:

| Competition Result | Method | Score |
|-------------------|--------|-------|
| HyperIQA (team-reported fine-tuned) | HyperIQA fine-tuned | 0.844 |
| MUSIQ (team-reported fine-tuned) | MUSIQ fine-tuned | 0.859 |
| TReS (team-reported fine-tuned) | TReS fine-tuned | 0.863 |
| RichIQA (team-reported fine-tuned) | TOPIQ-NR fine-tuned | 0.866 |
| **HyperIQA++ (ours)** | HyperIQA + SpatialAttn + soft labels | **0.856** |
| ThinkSmart AI team | MUSIQ + TReS ensemble | 0.828 |
| VisionBlend-IQA team | MUSIQ + TReS + RichIQA ensemble | 0.805 |

HyperIQA++ (0.856) exceeds the team-reported HyperIQA baseline (0.844) by 1.2 points, confirming that the spatial attention and soft-label heads provide meaningful improvement. However, it still falls short of single-model MUSIQ (0.859), TReS (0.863), and RichIQA (0.866) — models that may have used more extensive hyperparameter tuning or augmentation strategies.

The critical insight is that all CNN-based approaches plateau around 0.85-0.87, while MLLM approaches reach 0.92-0.93. The CNN ceiling appears to be a fundamental limitation of architectures that lack language-grounded quality understanding.

---

## 7. Strengths and Weaknesses

### 7.1 Strengths

1. **Fast inference**: ~100ms per image on a single GPU, comparable to SigLIP2 and much faster than MLLMs
2. **Significant fine-tuning improvement**: 96% gain from off-the-shelf to fine-tuned demonstrates the architecture's capacity
3. **Content-adaptive predictions**: HyperNet generates image-specific weights, providing a form of test-time adaptation
4. **High PLCC**: Strong linear correlation with human MOS after logistic fitting (0.780-0.797 on OOD)
5. **Soft-label distribution output**: Provides richer uncertainty information than point predictions

### 7.2 Weaknesses

1. **Largest OOD gap**: -0.165 MainScore from ID to OOD — the worst generalization among fine-tuned models
2. **Scale miscalibration**: MAE of 2.225 indicates systematic offset requiring post-hoc correction
3. **Fixed resolution**: 1600x1600 input doesn't adapt to document aspect ratios or DPI variations
4. **No semantic understanding**: Cannot reason about text legibility, layout coherence, or document type
5. **CNN ceiling**: Performance plateaus at ~0.86, below the MLLM tier (~0.92+)
6. **Catastrophic forgetting**: Fine-tuning hurts OOD generalization compared to off-the-shelf

### 7.3 Role in the Pipeline

Despite its limitations, HyperIQA++ serves a useful role as a **complementary signal** in the pseudo-labeling pipeline:

- **Ensemble diversity**: As a CNN-based model, its failure modes differ from ViT-based (SigLIP2) and MLLM-based (DeQA-Doc) approaches
- **Agreement signal**: When HyperIQA++, SigLIP2, and VLM teachers agree on a quality score, confidence is very high
- **Disagreement signal**: When HyperIQA++ disagrees with other models, it flags images where model-specific biases may affect predictions

---

## 8. Conclusions

1. **Fine-tuning transforms HyperIQA from inadequate to competitive**: MainScore improves from 0.437 to 0.856 (+96%) with 3,500 training images, demonstrating that the document IQA domain gap is primarily a data problem, not an architecture problem.

2. **CNN architectures hit a ceiling**: At ~0.86, HyperIQA++ plateaus below MLLM approaches (~0.93) and ViT approaches (0.886), suggesting that CNN inductive biases are suboptimal for document quality assessment.

3. **Fine-tuning hurts generalization**: Off-the-shelf HyperIQA outperforms HyperIQA++ on OOD data (0.723 vs 0.694), confirming that domain-specific fine-tuning without sufficient data diversity leads to overfitting.

4. **Scale calibration is a separate problem**: High correlation metrics (SRCC 0.856, PLCC 0.886) coexist with poor absolute accuracy (MAE 2.225), requiring post-hoc calibration for production use.

5. **Complementary value in ensembles**: HyperIQA++'s CNN-based feature extraction provides diversity when combined with ViT and MLLM approaches for consensus scoring.

---

## 9. Data Availability

| Artifact | Location |
|----------|----------|
| Model checkpoint | Modal volume `dociq-checkpoints` / `hyperiqa_plus_plus_best.pt` |
| Training script | `image_detection/modal/train_hyperiqa_plus_plus.py` (check git history) |
| Model card | `image_detection/docs/model-cards/production/hyperiqa_plus_plus_diqa5000.md` |
| Synthetic OOD predictions | `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/hyperiqa_plus_plus.jsonl` |
| Fine-tuned OOD metrics | `results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json` |
| NR-IQA baseline comparison | `results/iqa_baselines/baseline_summary.json` |
| Synthetic eval handoff | `results/vlm_teacher_eval/full_eval/MODAL_SYNTHETIC_EVAL_HANDOFF.md` |
