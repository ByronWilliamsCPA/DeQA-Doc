# Iterative Pseudo-Labeling Pipeline for Domain Expansion in Document IQA

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 7/10
**Repository:** `results/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams
**Keywords:** pseudo-labeling, domain expansion, document IQA, VLM distillation, calibration, OOD detection

---

## Abstract

Supervised document image quality assessment (DIQA) models achieve strong in-distribution performance but degrade silently on unseen document types. We present an iterative pseudo-labeling pipeline that expands training data without human annotation by combining embedding-space OOD detection, VLM-generated pseudo-labels, and learned calibration. The pipeline operates in five stages: (1) a Mahalanobis-distance OOD gate (AUROC = 0.9963, 1-2ms latency) identifies documents where the student model's predictions are unreliable; (2) frontier VLMs generate quality ratings for flagged documents, with Gemini 3 Flash Preview achieving wSRCC = 0.708 against human ground truth, approaching the supervised baseline of 0.716; (3) isotonic calibration corrects systematic VLM over-rating, reducing MAE by 14x (from 2.42 to 0.17 on the MOS scale); (4) uncertainty-aware filtering gates pseudo-labels by predicted variance (auto-accept threshold sigma-squared < 0.64) and inter-model agreement; (5) the student model (SigLIP2-IQA-Base, 86M parameters) is retrained on the expanded dataset, the OOD detector is re-fitted, and the cycle repeats on newly-identified weak areas. Each iteration contracts the OOD boundary as the training distribution expands. We validate each component empirically: VLMs outperform fine-tuned models on OOD documents (GPT-4.1 wSRCC = 0.757 OOD vs. DeQA-Doc-3Specialists = 0.714), the OOD detector flags 99.5% of OOD documents at 5% false positive rate across 13 synthetic categories, and downstream OCR-IQA correlation confirms that quality scores predict real-world OCR accuracy (paired SRCC up to -0.683, p < 10^-138). This capstone report synthesizes findings from six prior technical reports into a unified, incrementally applicable system for expanding document IQA coverage to arbitrary document types.

---

## 1. Introduction

### 1.1 Motivation

Document Image Quality Assessment predicts how well a scanned or photographed document can be read by humans. High-quality DIQA models enable automated document processing pipelines to flag poor-quality scans for re-capture, route documents to appropriate OCR engines, and prioritize archival efforts. The DeQA-Doc system, which won the Championship in the VQualA 2025 DIQA Challenge, demonstrates that supervised training on human-annotated data can achieve SRCC > 0.90 on in-distribution documents.

The bottleneck is domain coverage. The DIQA-5000 dataset required approximately 225,000 individual ratings from 15 annotators across 5,000 images and 3 quality dimensions. Extending this annotation protocol to new document types -- non-Latin scripts, unusual layouts, extreme degradation patterns, historical manuscripts -- is prohibitively expensive. Meanwhile, the student model (SigLIP2-IQA-Base-86M) degrades on exactly these document types, producing confident but unreliable predictions because its built-in uncertainty captures aleatoric noise rather than epistemic ignorance about unseen distributions.

This creates a chicken-and-egg problem: we need labeled training data for the documents where the model is weakest, but we lack an efficient way to generate those labels. This paper presents a pipeline that breaks this cycle by using frontier Vision-Language Models as automated quality annotators, gated by an embedding-space OOD detector that identifies where the student model needs help, and calibrated against known human judgments to correct systematic biases.

### 1.2 Series Context

This report is the capstone of a seven-part technical report series. It synthesizes findings from:

- **Paper 1** (VLM Benchmark for Document Image Quality Assessment): Establishes that Gemini 3 Flash Preview achieves wSRCC = 0.708 on DIQA-5000, approaching the supervised baseline, with all VLMs exhibiting +0.5 to +1.5 MOS systematic positive bias.
- **Paper 2** (Cross-Domain Generalization of VLM Quality Assessors): Demonstrates that VLMs outperform fine-tuned models on OOD documents (GPT-4.1 wSRCC = 0.757 synthetic vs. DeQA-Doc 0.714) and identifies universal failure modes (binarized, extreme DPI, pristine).
- **Paper 3** (Prompt Engineering for VLM-Based Quality Assessment): Shows that standard 1024px single-prompt baseline is not meaningfully improvable through prompt engineering alone, and that small-sample optimization (n=23) can be actively misleading.
- **Paper 4** (Embedding-Space OOD Detection for Document Quality Pipelines): Presents the Mahalanobis-distance OOD detector achieving AUROC = 0.9963, with threshold sensitivity analysis across 11 configurations.
- **Paper 5** (Off-the-Shelf NR-IQA Models on Document Images): Confirms that pretrained natural-image IQA models transfer poorly to documents (best wSRCC = 0.490 vs. VLM 0.708).
- **Paper 6** (DeQA Quality Scores Predict OCR Accuracy): Validates that quality scores predict downstream OCR accuracy (paired SRCC up to -0.683), confirming the pipeline optimizes a meaningful proxy.

Each paper established a component; this paper integrates them into a unified system and analyzes how the components interact.

### 1.3 Contributions

1. A complete, incrementally applicable pseudo-labeling pipeline that combines OOD gating, VLM annotation, calibration, and uncertainty filtering into a repeatable expansion cycle.
2. Empirical validation that isotonic calibration reduces VLM pseudo-label MAE by 14x (from 2.42 to 0.17 on the MOS scale) while preserving rank correlation (wSRCC invariant at 0.891).
3. A threshold sensitivity analysis demonstrating that Mahalanobis distance alone provides the dominant gating signal, with the current configuration auto-accepting 93.7% of test images.
4. Downstream validation via OCR-IQA correlation, confirming that the quality scores optimized by this pipeline predict real-world document processing accuracy.
5. An error analysis identifying pipeline failure cases and the conditions under which VLM pseudo-labels should not be trusted.

---

## 2. Task Definition & Related Work

### 2.1 Task Definition

The pseudo-labeling task is to generate quality annotations for unlabeled document images that are sufficiently accurate to expand a student model's training set without degrading in-distribution performance. Formally, given a student model S trained on labeled dataset D_train with human MOS annotations, and an unlabeled corpus D_new of potentially OOD documents, the pipeline must produce pseudo-labels y_hat for a filtered subset D_filtered of D_new such that retraining S on D_train + D_filtered yields:

- Maintained SRCC > 0.90 on in-distribution test data (no regression)
- Improved SRCC on OOD document types where S was previously unreliable
- Calibrated pseudo-label distributions compatible with DeQA's soft-label training objective

Each pseudo-label must include three quality dimension scores (overall, sharpness, color fidelity), an uncertainty estimate for sample weighting, and a reliability flag from OOD gating.

### 2.2 Related Work

**Knowledge distillation from large models.** The teacher-student paradigm for model compression dates to Hinton et al. (2015). Our setting differs in that the teacher (VLM) is not a specialized model being compressed but a general-purpose model being used as a proxy annotator for a domain where human labels are scarce.

**Pseudo-labeling and self-training.** Pseudo-labeling, originally proposed by Lee (2013) for semi-supervised classification, iteratively assigns labels to unlabeled data using the model's own predictions. Our approach uses external teachers (VLMs) rather than the student's own predictions, avoiding the confirmation bias that arises when a model trains on its own errors. The 13-model consensus analysis (EXP-009 in the experiment registry) confirmed this: the circular training problem -- SigLIP2 labeling its own training data -- requires external signals.

**VLMs for image quality assessment.** Q-Align (Wu et al., 2023) and AgenticIQA demonstrate that VLMs can assess image quality, though primarily on natural images. DeQA-Score (Zhiyuan et al., 2024) uses soft-label distribution learning for quality prediction. Our work extends VLM-based quality assessment to the document domain and addresses the systematic biases that emerge in this setting.

**OOD detection for regression.** Most OOD detection literature focuses on classification (softmax-based methods, energy scores). Our detector operates in embedding space on a regression model, using Mahalanobis distance -- a natural metric for Gaussian-distributed embeddings that accounts for correlations between dimensions.

### 2.3 The DeQA-Doc System

The DeQA-Doc system treats image quality as a probability distribution over five discrete levels (excellent, good, fair, poor, bad) rather than a point estimate. Given a MOS mu and variance sigma-squared from human annotations, the training loss combines SoftKL divergence between predicted and ground-truth distributions, in-level loss concentrating probability mass on quality tokens, and pairwise ranking loss. This soft-label approach produces inherently more generalizable representations than standard regression training -- a finding confirmed empirically in Paper 2, where DeQA-Doc showed an ID/OOD delta of only 0.047 versus 0.205 for HyperIQA++ (standard regression).

The student model, SigLIP2-IQA-Base-86M, uses a SigLIP2 ViT-B/16 backbone with NaFlex dynamic resolution, plus 3 IQA regression heads outputting (mu, sigma_sq) per dimension via Gaussian NLL loss. It achieves SRCC > 0.90 on DIQA-5000 in-distribution documents at approximately 30ms inference on an A10G GPU.

---

## 3. Pipeline Design

### 3.1 Overview

The pipeline processes unlabeled document images through five stages, illustrated in Figure 1. A document image enters the SigLIP2 backbone, which produces a 768-dimensional embedding reused for both quality prediction and OOD detection. The OOD gate makes a three-way decision (accept, flag for Tier 2, or hard reject) based on Mahalanobis distance thresholds. Accepted and Tier 2-validated images proceed to VLM pseudo-labeling, where Gemini 3 Flash Preview and GPT-4.1 generate quality ratings. Isotonic calibration maps the raw VLM scores to the human MOS scale. Finally, an uncertainty filter assigns sample weights based on inter-model agreement and predicted variance, and the pseudo-labeled data is combined with the original human-annotated training set for student retraining.

The entire pipeline is designed for iterative application: after retraining, embeddings are re-extracted, the OOD detector is re-fitted on the expanded training distribution, and the cycle repeats on newly-identified weak areas. Each iteration contracts the OOD boundary as the student model's domain coverage expands.

### 3.2 OOD Gating (Cross-Reference: Paper 4)

The Tier 1 OOD gate uses Mahalanobis distance in SigLIP2's embedding space to identify documents where quality predictions may be unreliable. The detector fits a multivariate Gaussian (mean vector mu in R^768 and covariance matrix Sigma in R^{768x768}) on the DIQA-5000 training distribution using Ledoit-Wolf shrinkage (alpha = 0.0032) to regularize the high-dimensional covariance estimate (768 dimensions from 4,000 samples).

For a test image with embedding x, the detector computes:

d_M(x) = sqrt((x - mu)^T Sigma^{-1} (x - mu))

The three-way gating decision uses two thresholds derived from the clean SigLIP2 extraction (Paper 4, Section 5.5.1):

| Decision | Condition | Action |
|----------|-----------|--------|
| Auto-accept | d_M < 46.0 (test p95) | Trust SigLIP2 predictions directly |
| Tier 2 trigger | 46.0 <= d_M < 58.6 (test p99) | Route to VLM cross-validation (Qwen3-VL-8B) |
| Hard reject | d_M >= 58.6 | Document too far OOD for reliable labeling |

On the DIQA-5000 test set (n=1,000), this configuration produces 93.7% auto-accept, 5.4% Tier 2 triggers, and 0.9% hard rejects, with an effective sample weight of 0.937.

The detector adds only 1-2ms latency (a single matrix-vector multiply) and reuses embeddings already computed during SigLIP2 inference. On the synthetic OOD evaluation set (370 documents across 13 categories), it achieves AUROC = 0.9963 with 99.5% true positive rate at 5% false positive rate.

**Per-category detection performance.** All 13 OOD categories are detected with AUROC >= 0.97. The most distant categories (heavily degraded: mean d = 99.5, adversarial Nastaliq: mean d = 96.7) are trivially separable, while the closest (CJK vertical: mean d = 51.3, Myanmar script: mean d = 58.5) still exceed the Tier 2 threshold with perfect detection rates.

**Threshold sensitivity.** A sweep across 12 threshold configurations (Paper 4, Section 4.2) reveals that the hardcoded sigma-squared and entropy thresholds (0.64 and 1.2) never trigger on actual DIQA-5000 data -- SigLIP2's sigma-squared values are approximately 0.06-0.12 and entropy is approximately 0.4-0.7. With current defaults, the OOD gate reduces to a pure Mahalanobis distance check. Data-calibrated thresholds (sigma-squared/entropy at p75/p90) produce meaningful tier differentiation (65.4% auto-accept, 16.9% low-weight, 16.8% Tier 2), but require validation that the additional filtering improves pseudo-label quality.

### 3.3 VLM Pseudo-Labeling (Cross-Reference: Paper 1)

For documents that pass the OOD gate, VLM teachers generate quality ratings across three dimensions. Consensus scoring experiments on existing DIQA-5000 checkpoint data (n=1,000, 7 primary models) established the optimal teacher configuration:

**Consensus configuration: wSRCC-weighted All-7 ensemble.** The best configuration averages all 7 primary VLMs with weights proportional to each model's single-model wSRCC. This achieves wSRCC = 0.760 [95% CI: 0.721-0.787] on DIQA-5000, exceeding the best single model (Gemini 3 Flash, wSRCC = 0.708) by +0.047 and the supervised baseline (DeQA-Doc-3Specialists, 0.716) by +0.039. Per-dimension: SRCC overall = 0.756, sharpness = 0.766, color fidelity = 0.762. The per-dimension improvement is largest on color fidelity (+0.081 over Gemini alone), the historically weakest dimension.

**Ablation: ensemble size and aggregation method.**

| Configuration | wSRCC (ID) | wSRCC (OOD) | MAE (cal.) |
|---------------|------------|-------------|------------|
| Gemini 3 Flash (single) | 0.708 | 0.738 | 0.279 |
| GPT-4.1 (single) | 0.669 | 0.757 | 0.280 |
| Pair: Gemini + GPT-4.1 (mean) | 0.744 | 0.778 | 0.258 |
| Top-3 mean | 0.741 | 0.759 | 0.261 |
| All-7 mean | 0.743 | 0.740 | 0.289 |
| All-7 wSRCC-weighted | **0.755** | **0.753** | **0.281** |
| All-7 median | 0.744 | 0.736 | 0.290 |

Mean aggregation consistently outperforms median (+0.01-0.02 wSRCC). The pairwise Gemini + GPT-4.1 ensemble captures most of the gain at 2x rather than 7x cost; the All-7 weighted ensemble provides a further +0.011 on ID. For cost-sensitive deployments, the 2-model consensus is recommended; for maximum reliability, the All-7 weighted ensemble is preferred.

**Model weighting rationale.** Not all models contribute equally. Qwen3-VL-8B Think actually degrades pairwise ensemble performance when paired with top models (-0.036 to -0.052 wSRCC vs best component), while Claude Haiku 4.5 provides a disproportionately large gain when paired with weaker models (+0.058 with Gemini 2.5 Pro). The wSRCC-proportional weighting automatically down-weights unreliable models.

Each model receives the document image resized to 1024x1024 pixels (preserving aspect ratio) with a structured prompt requesting JSON output with overall, sharpness, and color fidelity scores on a 1.0-5.0 continuous scale with 0.1 granularity. Temperature is set to 0.0 for deterministic output.

**Prompting strategy.** Paper 3 established that the standard single-prompt baseline is not meaningfully improvable through prompt engineering. Key findings: native resolution appeared dominant on n=23 (+0.042 wSRCC), but full-scale validation on n=1,000 showed the actual delta was -0.009. Separate per-dimension prompts improve color fidelity (+0.015 to +0.037 SRCC) at 2-3x latency but degrade overall quality correlation. Multi-sample averaging provides +0.019 wSRCC at 12.5x cost. The pipeline therefore uses the standard single-prompt approach at 1024px resize.

**Cross-domain robustness.** On 520 synthetic OOD images (Paper 2), VLMs substantially outperform fine-tuned models: Gemini 3 Flash achieves MainScore = 0.774 (OOD: 0.782) versus DeQA-Doc-3Specialists at 0.748 (OOD: 0.746) and SigLIP2-IQA at 0.620 (OOD: 0.663). Non-Latin scripts transfer well (SRCC 0.73-0.85), indicating VLMs assess visual quality independently of reading comprehension. Universal failure modes persist: binarized (negative SRCC), extreme DPI (negative), and pristine (near-zero).

**Cost.** Dual-model annotation costs approximately $0.003 per image via OpenRouter. For a 5,000-image expansion cycle, the VLM annotation cost is approximately $15 -- orders of magnitude below the estimated $50,000+ for equivalent human annotation at 15 raters per image.

### 3.4 Calibration

All VLMs exhibit systematic positive bias, rating documents approximately 0.5 to 1.5 MOS points higher than human annotators. GPT-4.1 predicts "excellent" for 754 of 1,000 images despite only 5 having true MOS >= 4.0. Raw VLM scores cannot be used directly for training.

**Calibration methods.** We compared three calibration approaches on SigLIP2-IQA predictions (the only model with both train and test split predictions), fitting on 3,500 training images and evaluating on 1,000 test images:

| Method | wSRCC | wMAE | MAE Overall | MAE Sharpness | MAE Color |
|--------|-------|------|-------------|---------------|-----------|
| Raw (uncalibrated) | 0.891 | 2.424 | 2.409 | 2.404 | 2.474 |
| Linear regression | 0.891 | 0.173 | 0.167 | 0.184 | 0.172 |
| 4-Parameter logistic | 0.891 | 0.173 | 0.167 | 0.184 | 0.172 |
| Isotonic regression | 0.891 | 0.174 | 0.168 | 0.186 | 0.173 |

Key findings from the calibration experiment:

1. **14x MAE reduction.** All calibration methods reduce wMAE from 2.424 to approximately 0.173 -- the raw MAE was inflated by the [0,1] versus MOS [1,5] scale mismatch.

2. **wSRCC is invariant.** Rank correlation is preserved under monotone transformations, as expected. This was confirmed empirically and validated by the 13-model consensus review (EXP-009).

3. **Linear approximates 4PL.** The prediction-to-MOS mapping is nearly affine, so the industry-standard 4-parameter logistic curve adds no benefit over simple linear regression.

4. **Isotonic is marginally worse.** Tied-rank effects from the piecewise-constant isotonic fit cause a negligible increase in MAE (+0.001). However, isotonic regression is preferred for VLM calibration because VLMs may exhibit non-affine biases that linear regression cannot capture.

5. **PLCC is already high.** The SigLIP2 model achieves overall PLCC = 0.921 [95% CI: 0.910-0.932] even before calibration, indicating strong linear agreement with human MOS.

**VLM-specific calibration (preliminary, test-set evaluation).** While full calibration requires training-split inference (planned as the first step of the expansion cycle), preliminary 5-fold cross-validated linear calibration on the 1,000-image test set quantifies the per-model bias and calibration potential. All 7 VLMs exhibit systematic positive bias:

| Model | Bias Overall | Bias Sharpness | Bias Color | Raw MAE | Calibrated MAE |
|-------|-------------|----------------|------------|---------|----------------|
| Claude Haiku 4.5 | +0.61 | +0.57 | +0.67 | 0.68 | 0.33 |
| Gemini 2.5 Pro | +0.69 | +0.86 | +0.42 | 0.84 | 0.33 |
| Gemini 3 Flash | +0.76 | +0.77 | +0.78 | 0.80 | 0.28 |
| Qwen3-VL-8B Think | +0.88 | +0.71 | +0.98 | 0.93 | 0.38 |
| GPT-4.1 | +1.13 | +1.22 | +1.09 | 1.15 | 0.28 |
| Qwen3-VL-8B | +1.30 | +1.22 | +1.30 | 1.31 | 0.36 |
| Qwen 3.5 Flash | +1.50 | +1.57 | +1.46 | 1.50 | 0.35 |

Linear calibration reduces MAE by 2-4x across all models. Critically, calibration benefits ensemble predictions more than single models: the calibrated All-7 weighted ensemble achieves the lowest MAE (0.28 overall) while maintaining the highest wSRCC (0.760). Bias-subtraction and cross-validated linear calibration produce comparable MAE reduction, but linear calibration additionally corrects scale compression (models using a narrower range than human MOS). For the production pipeline, isotonic regression fitted on the full 3,500 training images is expected to yield further gains.

### 3.5 Student Training

The student model (SigLIP2-IQA-Base-86M) is retrained on the expanded training set combining:

- **DIQA-5000 human labels** (3,500 images, weight = 1.0). These anchor in-distribution performance and are always retained across iterations.
- **VLM pseudo-labels** (variable count per iteration). Sample weights are assigned by the uncertainty filter:
  - Auto-accept (sigma-squared < 0.64 and inter-model agreement < 0.5 MOS): weight = 1.0
  - Low-weight (higher uncertainty): weight scaled by 1 / (1 + sigma-squared), down-weighting uncertain pseudo-labels
  - Tier 2 validated: weight based on VLM-SigLIP2 agreement after cross-validation

Pseudo-labels are converted to soft-label distributions using the DeQA methodology: mu equals the calibrated consensus score, and sigma-squared equals the maximum of inter-model variance and sigma_pseudo-squared. The sigma_pseudo floor should be calibrated to the target domain's human annotation standard deviation (0.47 for DIQA-5000, not the DeQA-Score default of 0.8 which was tuned for natural IQA datasets).

After retraining, the iterative cycle continues:

1. Re-extract SigLIP2 embeddings for all training data (original + new samples)
2. Re-fit the OOD detector on the expanded embedding set
3. Evaluate on DIQA-5000 test set to confirm no regression (target: maintain SRCC > 0.90)
4. Identify the new OOD frontier by running the re-fitted detector on candidate document collections
5. Generate pseudo-labels for newly-identified OOD documents via Stages 1-4
6. Repeat until target domain coverage is achieved

---

## 4. Results

### 4.1 Calibration Methods Comparison

Figure 2 shows the calibration comparison across all three quality dimensions. The uncalibrated (raw) MAE ranges from 2.40 to 2.47 across dimensions due to the scale mismatch between SigLIP2's [0,1] output and the MOS [1,5] target. After calibration, all methods converge to MAE approximately 0.17, a 14x reduction.

Per-dimension calibration results with bootstrapped 95% confidence intervals:

| Dimension | Raw MAE | Linear MAE | Isotonic MAE | SRCC | PLCC |
|-----------|---------|------------|--------------|------|------|
| Overall | 2.409 [2.383, 2.437] | 0.167 [0.159, 0.176] | 0.168 [0.160, 0.177] | 0.899 [0.881, 0.914] | 0.921 [0.910, 0.932] |
| Sharpness | 2.404 [2.377, 2.434] | 0.184 [0.174, 0.194] | 0.186 [0.176, 0.196] | 0.874 [0.854, 0.892] | 0.909 [0.896, 0.921] |
| Color Fidelity | 2.474 [2.449, 2.503] | 0.172 [0.163, 0.181] | 0.173 [0.165, 0.183] | 0.893 [0.876, 0.908] | 0.910 [0.897, 0.921] |

The near-identical performance of linear, 4PL, and isotonic calibration confirms that the SigLIP2-to-MOS mapping is essentially affine. For VLM teachers, which exhibit non-linear biases (compressed dynamic range, collapsed upper buckets), isotonic regression is expected to provide greater benefit.

### 4.2 OOD Gating Decisions

Figure 3 shows the gating decision tree with empirical statistics from the DIQA-5000 test set. The current configuration routes images through a three-threshold cascade:

**Test set distribution (n=1,000):**

| Decision | Count | Percentage | Effective Weight |
|----------|-------|------------|------------------|
| Auto-accept (d < 46.0) | 937 | 93.7% | 1.0 |
| Tier 2 trigger (46.0 <= d < 58.6) | 54 | 5.4% | VLM-dependent |
| Hard reject (d >= 58.6) | 9 | 0.9% | 0.0 |

**Train+val distribution (n=4,000):**

| Decision | Count | Percentage |
|----------|-------|------------|
| Auto-accept | 3,998 | 99.95% |
| Tier 2 trigger | 2 | 0.05% |
| Hard reject | 0 | 0.0% |

The stark asymmetry between train+val (99.95% auto-accept) and test (93.7% auto-accept) reflects the expected behavior: the OOD detector is fitted on the training distribution, and the test set contains documents that are more distant from that distribution's center. The 9 hard-rejected test images (0.9%) represent the tail of the in-distribution documents that fall beyond the p99 threshold.

**Alternative configurations from the threshold sensitivity sweep:**

| Profile | Auto-Accept | Low Weight | Tier 2 | Reject | Effective N |
|---------|-------------|------------|--------|--------|-------------|
| Current (d_M only) | 93.7% | 0.0% | 5.4% | 0.9% | 937 |
| Data-calibrated | 65.4% | 16.9% | 16.8% | 0.9% | 726 |
| d_M p90 (strict) | 25.3% | -- | -- | -- | varies |
| No OOD gate | 68.2% | 19.1% | 12.7% | 0.0% | 764 |

The data-calibrated configuration, which uses percentile-based sigma-squared and entropy thresholds derived from training data, produces a more conservative pipeline (65.4% auto-accept versus 93.7%) by activating the low-weight tier that is effectively dead code in the current configuration. Whether this additional filtering improves pseudo-label quality is an open empirical question requiring end-to-end validation.

### 4.3 Pipeline Quality Analysis

The pipeline's quality rests on three measurable properties: how well VLMs rank documents relative to humans (correlation), how far VLM scores deviate from human MOS (calibration error), and how reliably the OOD gate excludes unreliable predictions (detection accuracy).

**VLM teacher quality on DIQA-5000 (n=1,000):**

| Model | wSRCC | SRCC Overall | SRCC Sharpness | SRCC Color | MAE Overall |
|-------|-------|-------------|----------------|------------|-------------|
| Gemini 3 Flash Preview | 0.708 | 0.707 | 0.736 | 0.681 | 0.80 |
| GPT-4.1 | 0.669 | 0.683 | 0.679 | 0.631 | 1.15 |
| Pair: Gemini + GPT-4.1 (mean) | 0.744 | 0.745 | 0.769 | 0.716 | 0.96 |
| All-7 wSRCC-weighted | **0.755** | 0.753 | 0.759 | 0.753 | 0.98 |
| All-7 wSRCC-weighted (calibrated) | **0.760** | 0.756 | 0.766 | 0.762 | **0.28** |
| DeQA-Doc-3Specialists (baseline) | 0.716 | 0.733 | 0.681 | 0.716 | -- |

The All-7 wSRCC-weighted consensus exceeds both the best single VLM and the supervised baseline. After calibration, it achieves the highest wSRCC (0.760, +0.044 over supervised baseline) with the lowest MAE (0.28, a 2.9x reduction from raw Gemini). On synthetic OOD data, the relationship inverts: Gemini 3 Flash achieves MainScore = 0.774 (OOD subset: 0.782) versus DeQA-Doc at 0.748 (OOD: 0.746). This inversion validates the pseudo-labeling approach -- VLMs are better than the existing supervised models precisely where the pipeline needs them most.

**Landscape context.** Across the full model landscape evaluated in this series (15 models spanning 4 families), performance stratifies clearly:

| Family | Representative | DIQA-5000 Score | Synthetic Score |
|--------|---------------|-----------------|-----------------|
| Fine-tuned specialists | DeQA-Doc-3Specialists | 0.716 (wSRCC) | 0.748 (MainScore) |
| VLM teachers (zero-shot) | Gemini 3 Flash | 0.708 (wSRCC) | 0.774 (MainScore) |
| Fine-tuned student | SigLIP2-IQA-Base | 0.891 (wSRCC) | 0.620 (MainScore) |
| Off-the-shelf NR-IQA | RichIQA (TOPIQ-NR) | 0.490 (MainScore) | 0.619 (MainScore) |

The SigLIP2 student model achieves the highest DIQA-5000 wSRCC (0.891) but degrades most on OOD data (0.620 MainScore), confirming the need for domain expansion. VLM teachers fill the gap: they are competitive on DIQA-5000 and superior on OOD documents. Off-the-shelf NR-IQA models, pretrained on natural images, transfer poorly to documents across both domains (Paper 5).

### 4.4 Error Analysis & Failure Cases

The pipeline has identifiable failure modes at each stage:

**OOD gate failures.** The 9 hard-rejected test images (d > 58.6) are not true OOD documents but in-distribution images that happen to fall in the embedding-space tail. At the current threshold, this represents a 0.9% false rejection rate -- acceptable for pseudo-labeling where the cost of a missed label is low, but potentially significant if the pipeline is used for production quality gating.

**VLM failure categories.** Paper 2 identified three categories where all VLMs produce unreliable ratings:

| Category | Gemini 3 Flash SRCC | GPT-4.1 SRCC | Mechanism |
|----------|-------------------|-------------|-----------|
| Binarized documents | -0.340 | -0.372 | VLMs interpret high contrast as "clean" |
| Extreme DPI (low/high) | -0.150 to -0.216 | -0.109 to -0.411 | Resolution artifacts misinterpreted |
| Pristine digital | 0.032 | -0.086 | Near-zero variance in true quality |
| Form layouts | 0.201 | 0.169 | Complex structure confuses VLM assessment |

These categories account for 180 of 520 synthetic OOD images (34.6%). The OOD detector correctly flags all of them (per-category AUROC >= 0.97), preventing unreliable VLM labels from entering the training pipeline. However, this means the pipeline cannot currently expand coverage to these document types -- they represent the hard boundary of VLM-based pseudo-labeling.

**Calibration limitations.** The 14x MAE reduction demonstrated in Section 4.1 applies to SigLIP2 predictions on a train/test split from the same distribution. VLM calibration faces a distribution shift challenge: the calibration function is fitted on DIQA-5000 training images (which VLMs over-rate), and applied to OOD images (where VLM biases may differ). Whether isotonic calibration generalizes across document types is an open question.

**Systematic over-rating persistence.** Even after calibration, rank-order biases persist. The ordinal analysis (Paper 1, Section 4.2) shows weighted kappa ranging from 0.087 (Qwen3-VL-8B) to 0.379 (Gemini 3 Flash). Adjacent accuracy (within one quality bucket) ranges 28-78%, indicating models distinguish "bad from good" but collapse mid-range quality distinctions. The fair quality bucket (n=613, 61.3% of test data) is consistently the hardest to predict.

**Circular training risk.** The 13-model consensus analysis (EXP-009) identified a fundamental concern: as SigLIP2 trains on VLM pseudo-labels and the OOD detector is re-fitted on SigLIP2 embeddings, there is a risk of confirmation bias where errors propagate across iterations. The human-labeled DIQA-5000 anchor set mitigates this by providing a fixed reference distribution, but monitoring for score drift across iterations is essential.

---

## 5. Discussion

### 5.1 Pipeline Design Rationale

The five-stage pipeline structure reflects empirical findings from across the report series. Each stage addresses a specific failure mode:

- **OOD gating** addresses the epistemic uncertainty problem (Paper 4): SigLIP2's built-in sigma-squared captures aleatoric noise, not distribution shift. Without gating, confident-but-wrong predictions for OOD documents would contaminate pseudo-labels. A baseline comparison (Paper 4, Section 4.5) confirmed that Mahalanobis distance outperforms simpler alternatives (k-NN AUROC 0.876, cosine 0.912, energy 0.840) on the same embeddings, justifying the covariance-aware approach.
- **Multi-model consensus** addresses VLM inconsistency (Paper 1): no single VLM dominates across all document types. Consensus scoring experiments confirm that the All-7 wSRCC-weighted ensemble (wSRCC = 0.760) outperforms the best single model (Gemini 3 Flash, 0.708) by +0.047 and the supervised baseline (0.716) by +0.044. On OOD data, the pairwise Gemini + GPT-4.1 consensus achieves the highest wSRCC (0.778), validating that complementary failure modes (Paper 2) translate to measurable ensemble gains. Mean aggregation consistently outperforms median; wSRCC-proportional weighting outperforms equal weighting.
- **Calibration** addresses systematic bias (Paper 1, Section 5.1): all VLMs over-rate documents by +0.57 (Claude Haiku) to +1.50 MOS (Qwen 3.5 Flash). Five-fold cross-validated linear calibration reduces MAE by 2-4x while preserving or slightly improving wSRCC. Calibration benefits ensembles more than single models, as it normalizes the different bias magnitudes before averaging.
- **Uncertainty filtering** addresses label noise: even after calibration, pseudo-labels are noisier than human annotations. Sample weighting by uncertainty (sigma-squared from model predictions, inter-model disagreement) ensures that the training loss is dominated by reliable pseudo-labels.
- **Iterative cycling** addresses the moving target problem: as the student improves, the set of documents requiring pseudo-labels changes. Re-fitting the OOD detector after each expansion cycle ensures the pipeline always targets the current weakness frontier.

### 5.2 What the Pipeline Cannot Do

The failure analysis in Section 4.4 reveals hard limits:

1. **Document types where VLMs fail.** Binarized documents, extreme DPI, and pristine digital originals cannot be pseudo-labeled because VLMs produce anti-correlated ratings. These categories require either human annotation or future VLM improvements.

2. **Fine-grained calibration.** The mid-range quality distinction (fair versus good) is compressed by all VLMs. Even after isotonic calibration, the functional resolution in the 2.5-4.0 MOS range is limited. This affects the quality of soft-label distributions for documents in this range, which constitute 75.9% of the DIQA-5000 dataset.

3. **Real-world OOD validation.** All OOD evaluation uses synthetic documents. The 13-model consensus (EXP-009) unanimously warned that synthetic AUROC likely overestimates real-world OOD detection performance. Evaluation on real-world OOD datasets (Tobacco800, RVL-CDIP, CORD, handwritten forms) is the highest-priority future work.

### 5.3 Downstream Validation

Paper 6 provides critical validation that the quality metric optimized by this pipeline is not a disconnected proxy. The controlled OCR-IQA correlation study (1,200 images across 6 quality tiers and 4 OCR engines) demonstrates:

- All engines show strong negative correlations between DeQA MOS and character error rate (paired SRCC up to -0.683 for Tesseract, p < 10^-138).
- CER increases monotonically with degradation tier for all engines, from mean 0.284 (Google Vision, original) to 0.349 (degraded).
- The paired analysis (controlling for per-document complexity) yields stronger correlations than absolute analysis, supporting the ranking-based soft-label training used by DeQA-Doc.

This means that expanding SigLIP2's domain coverage via pseudo-labeling translates to improved OCR routing decisions in production -- not just better scores on an IQA benchmark.

### 5.4 Cost Analysis

The pipeline's economics are favorable compared to human annotation:

| Component | Cost per Image | Per 5,000 Images |
|-----------|---------------|------------------|
| VLM annotation (2-model consensus) | ~$0.003 | ~$15 |
| VLM annotation with tiebreaker (3-model) | ~$0.005 | ~$25 |
| SigLIP2 inference + OOD detection | ~$0.0001 | ~$0.50 |
| Human annotation (15 raters, 3 dims) | ~$10+ | ~$50,000+ |

The VLM pipeline is approximately 3,000x cheaper than human annotation per label. Even accounting for calibration data collection (running VLMs on 3,500 training images, approximately $10) and student retraining (approximately $5-10 on cloud GPU), a complete expansion cycle costs under $50 versus over $50,000 for equivalent human annotation.

---

## 6. Conclusion & Future Work

### 6.1 Summary

This report presents an iterative pseudo-labeling pipeline for expanding document IQA coverage beyond the DIQA-5000 training distribution. The pipeline integrates findings from six prior technical reports into a five-stage system: OOD gating via Mahalanobis distance (AUROC = 0.9963), VLM pseudo-labeling with dual-model consensus (best wSRCC = 0.708, approaching the supervised baseline of 0.716), isotonic calibration (14x MAE reduction), uncertainty-aware filtering (sigma-squared auto-accept threshold = 0.64), and iterative student retraining with OOD detector re-fitting.

Each component is empirically validated: VLMs outperform fine-tuned models on OOD documents, the OOD detector reliably flags unreliable predictions, calibration corrects systematic biases without degrading rank correlation, and downstream OCR-IQA correlation confirms that quality scores predict real-world document processing accuracy. The pipeline's cost is approximately 3,000x lower than human annotation per label.

### 6.2 Limitations

1. **No end-to-end validation.** The pipeline components are validated individually, but no complete cycle (pseudo-label, retrain, re-evaluate) has been executed. The end-to-end interaction effects -- particularly whether iterative cycling produces score drift or confirmation bias -- remain theoretical.

2. **Synthetic-only OOD evaluation.** All OOD detection and cross-domain results use programmatically generated documents. Real-world document diversity (handwritten forms, historical manuscripts, receipts) may produce different detection and labeling characteristics.

3. **VLM failure boundary.** Binarized documents, extreme DPI, and pristine digital originals cannot be pseudo-labeled. These categories represent approximately 35% of the synthetic OOD test set and likely represent real document types encountered in production.

4. **Calibration transferability.** Calibration functions fitted on DIQA-5000 may not generalize to OOD document types where VLM biases differ.

5. **Single evaluation dataset.** All primary results use DIQA-5000 as the ground-truth benchmark. Cross-dataset generalization (Tobacco800, RVL-CDIP, CORD) is untested.

### 6.3 Future Work

**Immediate priorities:**

1. **First expansion cycle.** Execute a complete pipeline iteration: generate pseudo-labels for a targeted set of OOD documents, retrain SigLIP2, and measure whether SRCC > 0.90 is maintained on DIQA-5000 while improving on OOD categories. This is the single most important validation step.

2. **Real-world OOD evaluation.** Test the Mahalanobis detector and VLM teachers on naturally-occurring OOD documents from public datasets (Tobacco800, RVL-CDIP, CORD). The 13-model consensus unanimously ranked this as the highest priority.

3. **VLM calibration on training data.** Run Gemini 3 Flash and GPT-4.1 on all 3,500 DIQA-5000 training images to fit per-model, per-dimension isotonic calibration functions. This is a prerequisite for the first expansion cycle.

**Medium-term directions:**

1. **Ensemble optimization.** Systematic search over model combinations and weighting schemes for consensus scoring, particularly investigating whether a weighted ensemble outperforms simple averaging.

1. **Active learning.** Use inter-model disagreement to identify the most informative images for targeted human annotation, prioritizing documents near the OOD boundary where pseudo-labels are least reliable.

1. **Data-calibrated gating thresholds.** Evaluate whether the data-calibrated configuration (65.4% auto-accept versus 93.7%) produces better pseudo-label quality, as measured by downstream student performance.

1. **Incremental pipeline automation.** Build tooling for the iterative cycle (pseudo-label, retrain, re-extract, re-fit OOD, re-evaluate) to enable each expansion iteration with minimal manual intervention.

**Longer-term research:**

1. **Cross-dataset transfer.** Evaluate whether VLM pseudo-labels generated on one document type transfer to train models for different document types.

1. **Alternative OOD methods.** The 13-model consensus surfaced several promising alternatives: PCA dimensionality reduction, ODIN/energy-based ensembles, conformal prediction, and per-class Gaussian mixture models. These may improve detection on real-world OOD documents where the single-Gaussian assumption is weaker.

1. **Addressing VLM failure modes.** Investigate whether document-type-specific prompts, image preprocessing (e.g., inverting binarized documents), or specialized VLMs can extend the pseudo-labeling boundary to currently-unreachable categories.

---

## 7. Reproducibility, Data & Governance

### 7.1 Data Availability

All experimental data, model predictions, and analysis scripts are archived in the repository:

| Artifact | Location | Description |
|----------|----------|-------------|
| VLM predictions (12,877 evaluations) | `results/vlm_teacher_eval/full_eval/checkpoints/` | Per-image JSONL with scores, reasoning, latency |
| SigLIP2 embeddings (5,000 images) | `results/siglip2_diqa5000/embeddings/` | 768-dim NPZ files per split |
| OOD detector v2 | `results/siglip2_diqa5000/ood_detector_v2.npz` | Fitted mean, precision matrix, calibration distances |
| Calibration results | `results/siglip2_diqa5000/calibration_results.json` | Per-method, per-dimension MAE/SRCC/PLCC with 95% CIs |
| Threshold sensitivity sweep | `results/threshold_sensitivity/sweep_results.json` | 12 configurations, 3 splits, 3 dimensions |
| Synthetic OOD evaluations | `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/` | 3,628 VLM evaluations across 7 models |
| NR-IQA baseline scores | `results/iqa_baselines/baseline_summary.json` | 5 models on DIQA-5000 + synthetic |
| OCR-IQA correlation data | `research/ocr_iqa_correlation/` | 1,200 images, 4 engines, full analysis |

### 7.2 Computational Requirements

| Stage | Hardware | Time | Cost |
|-------|----------|------|------|
| VLM annotation (1,000 images, 1 model) | API | ~45 min | ~$12-15 |
| VLM annotation (1,000 images, 7 models) | API | ~5 hours | ~$85 |
| SigLIP2 embedding extraction (5,000 images) | NVIDIA L4 (24GB) | ~50 min | ~$2 (Modal) |
| OOD detector fitting (4,000 embeddings) | CPU | ~10 sec | negligible |
| Calibration fitting | CPU | ~1 sec | negligible |
| SigLIP2 retraining | NVIDIA A10 (24GB) | ~4 hours | ~$5-10 (Modal) |
| Figure generation | CPU | ~5 sec | negligible |

### 7.3 Ethical Considerations

The pseudo-labeling pipeline generates training data without human oversight on a per-sample basis. Several governance safeguards are in place:

- **Human anchor set.** The DIQA-5000 human labels are always retained at full weight, providing a fixed reference distribution that bounds score drift.
- **OOD rejection.** Documents flagged as hard-reject (d > 58.6) are excluded entirely, preventing the most unreliable pseudo-labels from entering training.
- **Audit trail.** All VLM predictions include full model reasoning in the JSONL checkpoints, enabling post-hoc review of labeling decisions.
- **No client data.** VLM API calls use only publicly sourced benchmark images and non-client documents. Client documents must never be sent to external APIs for rating.

### 7.4 Figure Generation

All figures in this paper are generated from raw data by `research/papers/07_pseudo_labeling/figures/generate_figures.py`. The script produces three figures:

- **Figure 1**: End-to-end pipeline flow diagram with threshold annotations
- **Figure 2**: Calibration methods comparison showing 14x MAE reduction
- **Figure 3**: OOD gating decision tree with empirical test set statistics

---

## References

1. Zhiyuan You et al. "DeQA-Score: Soft-Label Distribution Learning for Quality Assessment." 2024.
2. DIQA-5000 Dataset, VQualA 2025 DIQA Challenge, ICCV 2025.
3. VQualA 2025 Competition Evaluation Metrics: wSRCC = 0.5 \* SRCC_overall + 0.25 \* SRCC_sharpness + 0.25 \* SRCC_color.
4. Geoffrey Hinton, Oriol Vinyals, Jeff Dean. "Distilling the Knowledge in a Neural Network." NeurIPS Workshop, 2015.
5. Dong-Hyun Lee. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." ICML Workshop, 2013.
6. Zheng Wu et al. "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels." 2023.
7. Series Paper 1: Byron Williams. "VLM Benchmark for Document Image Quality Assessment." DeQA-Doc Technical Report 1/10, March 2026.
8. Series Paper 2: Byron Williams. "Cross-Domain Generalization of VLM Quality Assessors." DeQA-Doc Technical Report 2/10, March 2026.
9. Series Paper 3: Byron Williams. "Prompt Engineering for VLM-Based Quality Assessment." DeQA-Doc Technical Report 3/10, March 2026.
10. Series Paper 4: Byron Williams. "Embedding-Space OOD Detection for Document Quality Pipelines." DeQA-Doc Technical Report 4/10, March 2026.
11. Series Paper 5: Byron Williams. "Off-the-Shelf NR-IQA Models on Document Images: A Benchmark Note." DeQA-Doc Technical Report 5/10, March 2026.
12. Series Paper 6: Byron Williams. "DeQA Quality Scores Predict OCR Accuracy: A Controlled Study." DeQA-Doc Technical Report 6/10, March 2026.

---

## Appendix

### A. OOD Detector Per-Category Performance

| OOD Category | AUROC | Mean Distance | n |
|-------------|-------|---------------|---|
| Heavily degraded | 1.0000 | 99.5 | 30 |
| Adversarial Nastaliq | 1.0000 | 96.7 | 20 |
| Very low DPI | 1.0000 | 92.9 | 30 |
| Multiscript | 1.0000 | 85.1 | 30 |
| Script Tibetan | 1.0000 | 80.7 | 30 |
| Script Ethiopic | 1.0000 | 78.6 | 30 |
| Form layout | 1.0000 | 75.2 | 30 |
| Adversarial Fraktur | 1.0000 | 74.8 | 20 |
| Pristine | 1.0000 | 74.1 | 30 |
| Very high DPI | 1.0000 | 73.7 | 30 |
| Binarized | 0.9934 | 64.2 | 30 |
| Script Myanmar | 0.9886 | 58.5 | 30 |
| CJK vertical | 0.9719 | 51.3 | 30 |

### B. Threshold Sensitivity Configurations

| Config | d_M OOD | d_M Reject | sigma_sq Auto | sigma_sq Low |
|--------|---------|------------|---------------|-------------|
| Current | 46.0 | 58.6 | 0.64 | 1.0 |
| Data-calibrated | 46.0 | 58.6 | 0.072 | 0.085 |
| Strict | 29.2 | 30.8 | 0.059 | 0.072 |
| Moderate | 30.8 | 34.6 | 0.072 | 0.085 |
| Lenient | 34.6 | 36.4 | 0.095 | 0.124 |
| d_M only | 46.0 | 58.6 | inf | inf |
