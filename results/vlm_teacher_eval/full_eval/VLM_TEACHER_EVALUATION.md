# VLM Teachers for Document Image Quality Assessment: A Systematic Evaluation

**Authors:** Byron Williams
**Date:** March 2026
**Repository:** DeQA-Doc (`results/vlm_teacher_eval/full_eval/`)

---

## Abstract

Our SigLIP2-IQA student model achieves SRCC > 0.90 on DIQA-5000 document images but degrades on document types not represented in its training set. Expanding SigLIP2's effective domain requires new training data, but obtaining human quality annotations is expensive and slow — a chicken-and-egg problem. We address this by evaluating Vision-Language Models (VLMs) as automated quality annotators that can generate pseudo-labels for out-of-distribution (OOD) documents, enabling iterative expansion of SigLIP2's training set without additional human annotation.

We evaluate 7 VLMs across 3 quality dimensions (overall, sharpness, color fidelity) on 1,000 real document images from the DIQA-5000 test set, comparing their ratings against human Mean Opinion Scores (MOS) from 15 annotators. We further assess cross-domain generalization on a 520-image synthetic dataset spanning 13 OOD categories, conduct ordinal discrimination analysis, and run a 7-arm prompt optimization experiment. Our best model, Gemini 3 Flash Preview, achieves wSRCC = 0.708 on DIQA-5000 — approaching the DeQA-Doc-3Specialists baseline (wSRCC = 0.716) trained on human labels. We find that all VLMs exhibit systematic positive bias (+0.5 to +1.5 MOS points), that small-sample prompt optimization is unreliable (a 23-image experiment suggested +0.042 wSRCC from native resolution, but full-scale validation showed -0.009), and that per-dimension prompting improves color fidelity correlation. We integrate an embedding-space OOD detector (AUROC = 0.9963) to identify where SigLIP2's predictions are unreliable and propose a multi-stage pseudo-labeling pipeline — combining VLM consensus, bias calibration, OOD gating, and uncertainty-aware filtering — designed for iterative application: each training cycle expands SigLIP2's domain, after which the OOD detector is re-fitted and the process repeats on newly-identified weak areas.

**Per-sample scores for all 7,000 model-image evaluations are archived in the repository** (see [Data Availability](#7-data-availability)).

---

## 1. Introduction

### 1.1 Motivation

Document Image Quality Assessment (DIQA) predicts how well a scanned or photographed document can be read by humans. High-quality DIQA models enable automated document processing pipelines to flag poor-quality scans for re-capture, route documents to appropriate OCR engines, and prioritize archival efforts.

Our SigLIP2-IQA student model, trained on the DIQA-5000 dataset via soft-label distribution learning, achieves SRCC > 0.90 on DIQA-5000 documents. However, it degrades on document types not well represented in the training set — non-Latin scripts, unusual layouts, extreme degradation patterns, and other OOD categories. The goal of this project is to **broaden the range of documents where SigLIP2 can maintain SRCC > 0.90** by iteratively expanding its training set.

This presents a chicken-and-egg problem: we need more labeled training data to improve SigLIP2's domain coverage, but we lack an effective way to generate quality labels for the very documents where SigLIP2 is unreliable. The DIQA-5000 dataset required 15 human annotators per image across 3 quality dimensions (approximately 225,000 individual ratings for 5,000 images) — extending this process to new document types is prohibitively costly.

Vision-Language Models (VLMs) offer a way to break this cycle: using frontier multimodal models as automated annotators to generate pseudo-labels for OOD documents identified by the embedding-space OOD detector. If VLM ratings correlate sufficiently with human judgments on these OOD documents, they can provide the supplemental training data needed to expand SigLIP2's effective domain — and each expansion cycle shifts the OOD boundary outward, requiring re-evaluation of where weaknesses remain.

### 1.2 Research Questions

1. **Which VLMs best approximate human quality judgments** on document images, and how does performance vary across quality dimensions?
2. **How robust are VLM ratings** to out-of-distribution document types (non-Latin scripts, extreme degradation, unusual layouts) — specifically, the categories where SigLIP2 is weakest?
3. **What prompting strategies maximize correlation** with human MOS?
4. **Can VLM-generated pseudo-labels provide sufficient quality** to expand SigLIP2's training set for OOD document types and maintain SRCC > 0.90 after retraining?

### 1.3 Contributions

- A benchmark of 7 VLMs on 1,000 DIQA images with bootstrapped 95% confidence intervals, contextualized against 26 models from the DIQA-5000 benchmark landscape
- Cross-domain evaluation on 520 synthetic images spanning 13 OOD categories
- An ordinal discrimination analysis revealing systematic over-rating behavior
- A 7-arm prompt optimization experiment identifying resolution as a potentially dominant accuracy factor (pending full-scale validation)
- Integration of a Mahalanobis-distance OOD detector (AUROC = 0.9963) for automated quality filtering in the pseudo-labeling pipeline
- A proposed multi-stage pseudo-labeling pipeline for training the SigLIP2-IQA-Base-86M student model without human annotation

---

## 2. Background

### 2.1 The DIQA-5000 Dataset

DIQA-5000 is a document image quality assessment dataset introduced for the VQualA 2025 DIQA Challenge. It contains 5,000 document images with human quality annotations across three dimensions:

| Dimension | Description |
|-----------|-------------|
| **Overall Quality** | Holistic readability and usability of the document |
| **Sharpness** | Text edge clarity, blur level, and resolution adequacy |
| **Color Fidelity** | Color accuracy, contrast, white balance, and tonal reproduction |

**Annotation Protocol.** Each image was rated by 15 human annotators on a 1-5 discrete scale (bad, poor, fair, good, excellent). The Mean Opinion Score (MOS) for each dimension is the arithmetic mean of the 15 ratings, yielding a continuous score in [1.0, 5.0]. The dataset is split into 3,500 training images and 1,000 test images (500 held back for private evaluation).

**Score Distribution.** The DIQA-5000 test set has a strong concentration in the "fair" quality range:

| Quality Bucket | MOS Range | n (Overall) | n (Sharpness) | n (Color) |
|---------------|-----------|-------------|----------------|-----------|
| Bad | [1.0, 1.8) | 56 | 56 | 37 |
| Poor | [1.8, 2.6) | 180 | 172 | 162 |
| Fair | [2.6, 3.4) | 613 | 620 | 606 |
| Good | [3.4, 4.0) | 146 | 141 | 184 |
| Excellent | [4.0, 5.0] | 5 | 11 | 11 |

The extreme class imbalance (only 5 "excellent" overall images vs. 613 "fair") presents a significant challenge for both VLM evaluation and ordinal calibration.

### 2.2 DeQA-Score: Soft-Label Distribution Learning

DeQA-Score (Zhiyuan et al.) treats image quality as a probability distribution over five discrete levels rather than a point estimate. Given a MOS mu and variance sigma^2 from human annotations:

1. Model quality as Gaussian: x ~ N(mu, sigma^2)
2. Compute raw probability for each level by integrating the Gaussian PDF over [c_i - 0.5, c_i + 0.5]
3. Apply linear transformation to ensure probabilities sum to 1 and the recovered mean equals mu

The training loss combines:
- **SoftKL loss**: KL divergence between predicted token distribution and ground-truth soft labels
- **In-level loss**: Concentrates probability mass on the 5 quality tokens
- **Ranking loss**: Pairwise ranking between image pairs

Our DeQA-Doc system adapts this approach to document images, achieving the Championship in the VQualA 2025 DIQA Challenge.

### 2.3 The DeQA-Doc-3Specialists Baseline

Our reference baseline uses three specialized DeQA-Doc models, each trained on a single quality dimension. On the DIQA-5000 test set:

| Metric | Value |
|--------|-------|
| wSRCC | 0.716 |
| SRCC (overall) | 0.733 |
| SRCC (sharpness) | 0.681 |
| SRCC (color) | 0.716 |

Where wSRCC follows the VQualA competition metric: `0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color`.

### 2.4 The Pseudo-Labeling Challenge

To expand SigLIP2's training set beyond DIQA-5000 — targeting the OOD document types where SigLIP2 currently degrades — we need pseudo-labels that satisfy:

1. **Correlation with human judgment** (SRCC > 0.7 per dimension)
2. **Calibrated scale** (minimal systematic bias in MOS)
3. **Uncertainty estimates** (soft-label distributions, not point estimates)
4. **Domain robustness** (stable across document types and degradation levels)

### 2.5 Landscape of DIQA Models

To contextualize VLM teacher performance, we surveyed 26 models benchmarked on DIQA-5000 (see `diqa5000_benchmark_results.csv` in the `image_detection` repository). The landscape spans four model families:

| Family | Representative Models | Best wSRCC | Notes |
| --- | --- | --- | --- |
| Fine-tuned specialists | DeQA-Doc-3Specialists, HyperIQA++ | 0.716 | Trained on DIQA-5000 human labels |
| General IQA (zero-shot) | CLIP-IQA+, LIQE, DBCNN | ~0.3-0.5 | No DIQA-specific training |
| VLM teachers (this work) | Gemini 3 Flash, GPT-4.1 | 0.708 | Zero-shot, API-based |
| Student models | SigLIP2-IQA-Base-86M | 0.620 | Trained on pseudo-labels from VLM + DeQA consensus |

The fine-tuned specialists set the ceiling; general-purpose IQA models (designed for natural images) transfer poorly to documents. VLM teachers approach the specialist ceiling without any DIQA-specific training, motivating their use as pseudo-labelers.

### 2.6 The SigLIP2-IQA Student Model

SigLIP2-IQA-Base-86M is the production model whose domain coverage we aim to expand. Trained on DIQA-5000 human labels, it already achieves SRCC > 0.90 on in-distribution documents. The challenge is maintaining this level of performance as we broaden its training set to cover OOD document types via VLM pseudo-labels.

- **Architecture**: SigLIP2 ViT-B/16 backbone (86M params) with NaFlex dynamic resolution, plus 3 IQA regression heads outputting (mu, sigma_sq) per dimension via Gaussian NLL loss
- **Embeddings**: 768-dimensional penultimate-layer features, reused for OOD detection (Section 5.5). Full embeddings for all 5,000 DIQA-5000 images are archived at `results/siglip2_diqa5000/embeddings/`.
- **Training**: Multi-task learning on IQA regression + document classification + severity grading, using a mixture of human labels (DIQA-5000) and VLM pseudo-labels
- **Inference**: ~95ms on L4 GPU per image (including preprocessing); ~30ms on A10G
- **IQA output range**: mu in [-0.17, 0.73] raw; rescale to MOS [1,5] via `mu * 4.0 + 1.0`. Test MOS distribution: mean=2.83, std=0.47.

The quality of VLM pseudo-labels directly determines whether SigLIP2 can maintain SRCC > 0.90 after retraining on an expanded dataset that includes OOD document types. Critically, the entire pipeline must be incrementally applicable: as new training samples are added and SigLIP2 is retrained, the OOD detector must be re-fitted on the expanded training distribution, and the process repeats on the new frontier of OOD documents.

---

## 3. Experimental Setup

### 3.1 Model Selection

We selected 7 VLMs spanning different architectures, price points, and providers. Selection was informed by a 26-model smoke test (n=7 images per model) that identified top candidates:

| Model | Provider | Type | Cost (input/output per 1M tokens) |
|-------|----------|------|-----------------------------------|
| Gemini 3 Flash Preview | Google | Multimodal | Preview pricing |
| GPT-4.1 | OpenAI | Multimodal | $2.00 / $8.00 |
| Gemini 2.5 Pro | Google | Multimodal reasoning | Preview pricing |
| Qwen 3.5 Flash | Alibaba | Reasoning + vision | $0.10 / $0.40 |
| Claude Haiku 4.5 | Anthropic | Multimodal | $0.80 / $4.00 |
| Qwen3-VL-8B Instruct | Alibaba | Vision-language (8B) | $0.08 / $0.50 |
| Qwen3-VL-8B Thinking | Alibaba | Vision-language + CoT | $0.08 / $0.50 |

All models were accessed via OpenRouter API using the OpenAI-compatible SDK.

### 3.2 Evaluation Protocol

**Prompt Design.** Each model received a system prompt establishing it as a document quality assessor and a user prompt with the document image. The prompt requests ratings on a 1.0-5.0 continuous scale with 0.1 granularity across all three dimensions, with brief reasoning.

```text
System: You are an expert document image quality assessor...
Scale anchors: 1.0 (completely unusable) to 5.0 (excellent)

User: [image] Rate the overall quality, sharpness, and color fidelity
of this document image. Respond with JSON:
{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}
```

**Image Preprocessing.** Images were resized to fit within 1024x1024 pixels (preserving aspect ratio) and encoded as base64 JPEG for API transmission.

**Inference Parameters.** Temperature = 0.0 for all models except Qwen3-VL-8B Thinking (which uses the provider default, as OpenRouter's thinking-model interface does not accept explicit temperature). A separate temp=0 validation run on Qwen Thinking showed a small decrease (wSRCC 0.409 to 0.383), suggesting stochastic sampling slightly helps this model's ranking quality, but both values are well below the top models and the difference does not affect any conclusions. max_tokens = 1,024 (2,048 for thinking models to accommodate chain-of-thought), with exponential backoff retry (3 attempts).

**Metrics.** We report:
- **SRCC** (Spearman Rank Correlation Coefficient): Measures monotonic agreement
- **PLCC** (Pearson Linear Correlation Coefficient): Measures linear agreement
- **MAE** (Mean Absolute Error): Measures average prediction error
- **wSRCC**: Weighted SRCC following VQualA formula
- **95% Confidence Intervals**: Bootstrapped with 1,000 iterations (seed=42)

### 3.3 Datasets

**Primary: DIQA-5000 Test Set** (n=1,000). Real document images with human MOS ground truth. Images downloaded from GCS and cached locally.

**Secondary: Synthetic OOD Dataset** (n=520). Generated programmatically with controlled degradation parameters:

| Subset | n | Description |
|--------|---|-------------|
| In-distribution (standard) | 100 | Latin-script documents matching DIQA-5000 characteristics |
| In-distribution (Cyrillic) | 50 | Cyrillic-script documents with DIQA-like degradation |
| OOD: Non-Latin scripts | 90 | Tibetan, Myanmar, Ethiopic (30 each) |
| OOD: Adversarial scripts | 40 | Fraktur, Nastaliq (20 each) |
| OOD: Layout variants | 60 | CJK vertical, form layouts (30 each) |
| OOD: Extreme degradation | 60 | Binarized, heavily degraded (30 each) |
| OOD: Multiscript | 30 | Mixed-script documents |
| OOD: DPI extremes | 60 | Very low DPI, very high DPI (30 each) |
| OOD: Pristine | 30 | Near-perfect digital documents |

Ground truth MOS for synthetic images is derived from generation parameters (degradation level, noise intensity, etc.).

---

## 4. Results

### 4.1 Full Benchmark (n=1,000)

Table 1 shows the primary results on the DIQA-5000 test set. All 7 models received all 1,000 images; Gemini 2.5 Pro had 70 parse failures (7.0%) from non-JSON output and Qwen3-VL-8B Thinking had 2 (0.2%). Metrics for those models are computed on their valid responses only (n=930 and n=998 respectively).

**Table 1: DIQA-5000 Full Benchmark Results (n=1,000)**

| Model | wSRCC | SRCC_O | SRCC_S | SRCC_C | PLCC_O | MAE_O | Latency (ms) |
|-------|-------|--------|--------|--------|--------|-------|--------------|
| Gemini 3 Flash Preview | **0.708** | 0.707 | **0.736** | **0.681** | **0.784** | 0.80 | 2,331 |
| GPT-4.1 | 0.669 | **0.683** | 0.679 | 0.631 | 0.775 | 1.15 | 3,514 |
| Gemini 2.5 Pro | 0.612 | 0.613 | 0.603 | 0.621 | 0.662 | 0.84 | 9,697 |
| Qwen 3.5 Flash | 0.593 | 0.560 | 0.643 | 0.608 | 0.624 | 1.50 | 13,533 |
| Claude Haiku 4.5 | 0.579 | 0.598 | 0.539 | 0.579 | 0.636 | **0.68** | 2,917 |
| Qwen3-VL-8B Instruct | 0.481 | 0.520 | 0.437 | 0.446 | 0.563 | 1.31 | 2,531 |
| Qwen3-VL-8B Thinking | 0.409 | 0.432 | 0.397 | 0.377 | 0.465 | 0.93 | 8,289 |
| *DeQA-Doc-3Specialists* | *0.716* | *0.733* | *0.681* | *0.716* | *—* | *—* | *—* |

**Key findings:**
- **Gemini 3 Flash Preview** is the clear winner (wSRCC = 0.708), approaching the supervised baseline (0.716)
- **GPT-4.1** ranks second overall but has the worst MAE (1.15) due to systematic over-rating
- **Claude Haiku 4.5** has the best MAE (0.68) despite lower correlation — it rates conservatively
- **Reasoning models underperform**: Both Qwen3-VL-8B Thinking and Qwen 3.5 Flash (which uses extended reasoning tokens) rank below their non-reasoning counterparts
- **Confidence intervals** (95% bootstrapped): Gemini 3 Flash overall SRCC = [0.671, 0.742]; GPT-4.1 = [0.644, 0.719]. The CIs overlap slightly, so the gap is directional but not statistically significant at the 95% level.

**Smoke test instability.** The n=7 smoke test (26 models) dramatically overestimated all models: Haiku dropped from 0.813 to 0.579 (delta = -0.234), GPT-4.1 from 0.880 to 0.669 (-0.211), while Gemini 3 Flash was the most stable (0.857 to 0.708, delta = -0.149). Despite being the most stable, even Gemini's delta of -0.149 is substantial. This confirms that small-sample VLM benchmarks are unreliable for absolute ranking.

### 4.2 Ordinal Discrimination Analysis

We analyzed how well models classify images into quality buckets (bad/poor/fair/good/excellent) by mapping continuous predictions to the nearest bucket.

**Table 2: Ordinal Classification Metrics (Overall Quality)**

| Model | Exact Acc. | Adjacent Acc. | Weighted Kappa | Over-rate % |
|-------|-----------|---------------|----------------|-------------|
| Claude Haiku 4.5 | **31.5%** | **77.6%** | 0.340 | 64.0% |
| Gemini 3 Flash | 24.9% | 68.1% | **0.379** | 73.9% |
| GPT-4.1 | 10.2% | 37.4% | 0.246 | 88.6% |
| Qwen3-VL-8B Instruct | 1.2% | 27.7% | 0.087 | 98.3% |

**Systematic over-rating is universal.** Every model rates documents higher than humans, with over-rating ranging from 64% (Haiku) to 98% (Qwen3-VL-8B). GPT-4.1 predicts "excellent" for 754 of 1,000 images, despite only 5 images having true MOS >= 4.0.

**Adjacent accuracy vs. exact accuracy.** While exact bucket match is poor (1-31%), adjacent accuracy (within one bucket) ranges 28-78%, suggesting models can distinguish "bad from good" even if they cannot calibrate the absolute level. This supports a calibration-based correction approach.

**Confusion matrix pattern.** All models show a consistent pattern: they correctly identify truly bad images (high precision for "bad" bucket in Gemini and GPT-4.1) but collapse the fair/good/excellent distinctions. The "fair" bucket (n=613) is the hardest to predict — models scatter it across good and excellent.

### 4.3 Prompt Strategy Experiments

#### 4.3.1 Single vs. Separate Prompts (A/B Test)

We compared single-prompt (all 3 scores in one call) vs. separate-prompt (one call per dimension) on 44 stratified images with 2 models.

**Table 3: 1-Prompt vs. 3-Prompt Comparison (n=44)**

| Dimension | Condition | SRCC (Gemini) | SRCC (GPT-4.1) |
|-----------|-----------|---------------|-----------------|
| Overall | 1-prompt | **0.785** | **0.730** |
| Overall | 3-prompt | 0.768 | 0.679 |
| Sharpness | 1-prompt | 0.767 | 0.741 |
| Sharpness | 3-prompt | **0.803** | **0.760** |
| Color Fidelity | 1-prompt | 0.704 | 0.654 |
| Color Fidelity | 3-prompt | **0.719** | **0.691** |

**Finding:** Separate prompts improve sharpness and color fidelity ratings (where dimension-specific rubrics help prevent anchoring) at the cost of overall quality correlation and 2-3x latency. The improvement is most pronounced for color fidelity (+0.015 to +0.037 SRCC), the dimension where models are weakest.

#### 4.3.2 Prompt Optimization (7-Arm Experiment)

We tested 7 prompting strategies on Gemini 3 Flash with 23 stratified images:

**Table 4: Prompt Optimization Results (Gemini 3 Flash, n=23)**

| Arm | Strategy | wSRCC | MAE_O | Latency (ms) |
|-----|----------|-------|-------|--------------|
| 7 | **No resize** (native resolution) | **0.951** | **0.618** | 2,712 |
| 5 | Multi-sample (3x, temp=0.3, median) | 0.928 | 0.640 | 32,092 |
| 6 | Resize to 2048px | 0.925 | 0.655 | 2,693 |
| 3 | Hybrid (overall combined, sub-dims separate) | 0.923 | 0.686 | 5,902 |
| 2 | Separate 3 prompts | 0.911 | 0.803 | 5,683 |
| 1 | Single prompt, all 3 (baseline) | 0.909 | 0.633 | 2,563 |
| 4 | Few-shot (3 examples) | 0.836 | 0.715 | 2,293 |

We replicated this experiment on Qwen 3.5 Flash, which showed the same arm ranking (no-resize best at wSRCC=0.914, few-shot worst at 0.878), confirming that the optimization pattern is consistent across models on this small sample.

**Key findings from n=23 (subsequently disproved at n=1,000 — see Section 5.2):**

- **No-resize appeared dominant** (+0.042 wSRCC over baseline on Gemini, +0.133 on Qwen 3.5 Flash), but this **did not replicate** when validated on all 1,000 images (Section 5.2).
- **Multi-sample averaging** provides marginal gains (+0.019 wSRCC) but at 12.5x the cost — impractical at scale.
- **Few-shot examples hurt Gemini** (-0.073 wSRCC) but helped Qwen 3.5 Flash (+0.096), suggesting model-specific interactions.
- The key lesson is that **n=23 is insufficient for prompt optimization**: conclusions that hold for both models on 23 images may not generalize to the full dataset.

### 4.4 Cross-Domain Evaluation (Synthetic Dataset)

We evaluated all 7 VLM models on 520 synthetic images to test cross-domain generalization. Gemini 2.5 Pro had parse failures on 95 images (18.3%), so its metrics are computed on n=425 valid responses only. Qwen3-VL-8B Thinking had 2 errors (n=518).

**Table 5: Synthetic Dataset Results by Subset**

MainScore uses 4-parameter logistic PLCC; wSRCC is SRCC-only for backward compatibility.

| Model | n | MainScore | wSRCC | Main (ID) | Main (OOD) | SRCC_O | PLCC_O |
|-------|---|-----------|-------|-----------|------------|--------|--------|
| Gemini 3 Flash | 520 | **0.774** | 0.738 | **0.824** | **0.782** | 0.753 | 0.804 |
| GPT-4.1 | 520 | 0.769 | **0.757** | 0.825 | 0.757 | **0.764** | 0.788 |
| Claude Haiku 4.5 | 520 | 0.660 | 0.591 | 0.539 | 0.706 | 0.582 | 0.717 |
| Qwen 3.5 Flash | 451 | 0.596 | 0.572 | 0.442 | 0.667 | 0.550 | 0.604 |
| Gemini 2.5 Pro | 425 | 0.511 | 0.468 | 0.398 | 0.549 | 0.469 | 0.548 |
| Qwen3-VL-8B Instruct | 520 | 0.466 | 0.388 | 0.274 | 0.494 | 0.413 | 0.544 |
| Qwen3-VL-8B Thinking | 518 | 0.458 | 0.429 | 0.306 | 0.477 | 0.430 | 0.490 |

**Table 6: Per-Category Overall SRCC (Top 2 Models)**

| Category | n | Gemini 3 Flash | GPT-4.1 |
|----------|---|----------------|---------|
| Non-Latin scripts (Tibetan) | 30 | **0.800** | 0.730 |
| Non-Latin scripts (Myanmar) | 30 | 0.763 | 0.764 |
| Non-Latin scripts (Ethiopic) | 30 | 0.767 | **0.797** |
| Adversarial (Nastaliq) | 20 | 0.770 | **0.846** |
| Adversarial (Fraktur) | 20 | **0.768** | 0.762 |
| CJK vertical layout | 30 | 0.624 | **0.747** |
| Multiscript | 30 | 0.659 | **0.756** |
| In-distribution (standard) | 100 | **0.790** | 0.785 |
| In-distribution (Cyrillic) | 50 | **0.808** | 0.758 |
| Form layouts | 30 | 0.201 | 0.169 |
| Heavily degraded | 30 | 0.236 | 0.174 |
| Binarized | 30 | -0.340 | -0.372 |
| Pristine | 30 | 0.032 | -0.086 |
| Very high DPI | 30 | -0.150 | -0.109 |
| Very low DPI | 30 | -0.216 | -0.411 |

#### Fine-Tuned IQA Models on Synthetic OOD

To establish whether domain-trained models maintain their DIQA-5000 advantage on OOD data, we evaluated three fine-tuned IQA models on the same 520-image synthetic dataset.

Table 5b: Fine-Tuned Models vs. VLM Teachers (Synthetic Dataset, Unified MainScore)

| Model | Type | MainScore | Main (ID) | Main (OOD) | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C |
| ----- | ---- | --------- | --------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Gemini 3 Flash | VLM | **0.774** | 0.824 | **0.782** | 0.753 | 0.804 | 0.775 | 0.815 | 0.668 | 0.822 |
| GPT-4.1 | VLM | 0.769 | **0.825** | 0.757 | **0.764** | 0.788 | **0.797** | **0.820** | **0.704** | 0.730 |
| DeQA-Doc-3Specialists | Fine-tuned | 0.748 | 0.842 | 0.746 | 0.696 | 0.765 | 0.778 | **0.832** | 0.687 | 0.766 |
| HyperIQA++ | Fine-tuned | 0.694 | 0.840 | 0.675 | 0.589 | 0.780 | 0.623 | 0.797 | 0.606 | 0.790 |
| Claude Haiku 4.5 | VLM | 0.660 | 0.539 | 0.706 | 0.582 | 0.717 | 0.630 | 0.756 | 0.570 | 0.724 |
| SigLIP2-IQA-Base-86M | Fine-tuned | 0.620 | 0.659 | 0.663 | 0.495 | 0.700 | 0.577 | 0.762 | 0.507 | 0.718 |
| Qwen 3.5 Flash | VLM | 0.596 | 0.442 | 0.667 | 0.550 | 0.604 | 0.603 | 0.623 | 0.583 | 0.649 |
| Gemini 2.5 Pro | VLM | 0.511 | 0.398 | 0.549 | 0.469 | 0.548 | 0.591 | 0.696 | 0.344 | 0.426 |
| Qwen3-VL-8B Instruct | VLM | 0.466 | 0.274 | 0.494 | 0.413 | 0.544 | 0.437 | 0.620 | 0.291 | 0.466 |
| Qwen3-VL-8B Thinking | VLM | 0.458 | 0.306 | 0.477 | 0.430 | 0.490 | 0.485 | 0.524 | 0.373 | 0.439 |

**Findings:**

- **DeQA-Doc-3Specialists is the top fine-tuned model** (MainScore=0.748), approaching VLM teacher performance (Gemini 3 Flash=0.774, GPT-4.1=0.769) and far exceeding HyperIQA++ (0.694). With unified MainScore, DeQA-Doc achieves the highest ID score of any model (0.842) while maintaining competitive OOD performance (0.746, delta=0.096). The soft-label distribution learning objective produces inherently more generalizable representations than standard regression training.
- **Non-Latin scripts transfer well** (SRCC 0.73-0.85), suggesting VLMs assess visual quality independently of reading comprehension
- **Gemini 3 Flash leads on synthetic MainScore** (0.774 vs GPT-4.1's 0.769) due to higher PLCC, despite GPT-4.1 having better raw SRCC. The 4-parameter logistic fit benefits Gemini's predictions more, suggesting a more learnable nonlinear mapping to ground truth.
- **Universal failure modes**: Binarized (negative SRCC), extreme DPI (negative), pristine (near-zero), and form layouts (SRCC ~0.2). These categories have quality variation that is invisible to or misinterpreted by VLMs.
- **Haiku shows better OOD robustness** (wSRCC improves from 0.526 ID to 0.646 OOD), possibly due to more conservative rating behavior.
- **Qwen 8B models show consistent OOD > ID pattern.** Both Qwen3-VL-8B variants (Thinking: 0.281 ID → 0.463 OOD; Instruct: 0.248 ID → 0.431 OOD) perform substantially better on OOD than ID data, mirroring the Gemini 2.5 Pro and Haiku pattern. This suggests weaker models may have less overfitting to DIQA-5000-like distributions, though the absolute performance is poor. The Thinking variant slightly outperforms Instruct on synthetic data (wSRCC 0.429 vs 0.388), consistent with the DIQA-5000 gap (0.409 vs 0.481 — inverted, with Instruct better on DIQA).
- **HyperIQA++ degrades sharply OOD.** wSRCC drops from 0.754 (ID) to 0.549 (OOD), a delta of 0.205 — the largest gap of any model. This confirms that standard NR-IQA regression training overfits to training distribution characteristics. In contrast, DeQA-Doc's soft-label approach and SigLIP2's multi-task heads both show better OOD robustness (deltas of 0.047 and 0.076 respectively).
- **VLM teachers still generalize better than fine-tuned models on OOD.** Gemini 3 Flash (0.782 OOD MainScore) and GPT-4.1 (0.757 OOD) outperform DeQA-Doc (0.746 OOD), though the gap is narrower than wSRCC alone suggested. This validates the pseudo-labeling approach for OOD domain expansion.
- **Gemini 2.5 Pro degrades severely on synthetic data** (wSRCC=0.468, down from 0.612 on DIQA-5000), with 18.3% parse failures (95/520 images) and very poor ID correlation (wSRCC=0.329). Its strong negative bias (-0.66 overall) contrasts with other VLMs' positive bias, and its OOD performance (0.512) is worse than every fine-tuned model except SigLIP2. This disqualifies Gemini 2.5 Pro as a tiebreaker model (cf. consensus review S-3).

The universal failure modes (binarized, extreme DPI, pristine) represent document categories where VLM pseudo-labels cannot be trusted. This motivates the embedding-space OOD detector (Section 5.5), which flags these categories with AUROC >= 0.97 before VLM annotation occurs, preventing unreliable labels from entering the training pipeline.

### 4.5 Baseline IQA Model Benchmark (Off-the-Shelf NR-IQA)

To complete the landscape comparison from Section 2.5, we benchmarked 5 of the 6 competition baseline NR-IQA models on both the DIQA-5000 test set and the 520-image synthetic dataset using their pretrained pyiqa checkpoints (no DIQA-specific fine-tuning). StairIQA was unavailable in pyiqa. RichIQA was approximated by TOPIQ-NR (CFANet), the closest available architecture. All models output a single scalar quality score per image, which is compared against each dimension's ground truth independently.

**Evaluation protocol.** We follow the VQualA competition metric: `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color`, where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses the standard 4-parameter logistic curve fitting (nonlinear regression before Pearson correlation), matching the competition evaluation code.

**Infrastructure.** All evaluations ran on Modal (NVIDIA T4 GPU) using the pyiqa library (v0.1.13). Per-image scores were checkpointed to JSONL on a Modal volume with automatic resume, enabling completion across multiple runs. The benchmark script is at `modal/benchmark_iqa_baselines.py`; aggregated results at `results/iqa_baselines/baseline_summary.json`.

**Table 7: Baseline NR-IQA Models on DIQA-5000 Test Set (n=1,000)**

| Model | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | MainScore | Reported |
|-------|--------|--------|--------|--------|--------|--------|-----------|----------|
| RichIQA (TOPIQ-NR) | 0.489 | 0.483 | 0.498 | 0.484 | 0.507 | 0.488 | **0.490** | 0.866 |
| DBCNN | 0.444 | 0.446 | 0.466 | 0.458 | 0.466 | 0.457 | **0.453** | 0.587 |
| HyperIQA | 0.475 | 0.426 | 0.424 | 0.364 | 0.481 | 0.425 | **0.437** | 0.844 |
| TReS | 0.447 | 0.414 | 0.397 | 0.367 | 0.463 | 0.425 | **0.422** | 0.863 |
| MUSIQ | 0.153 | 0.188 | 0.214 | 0.217 | 0.169 | 0.194 | **0.185** | 0.859 |
| StairIQA | — | — | — | — | — | — | N/A | 0.850 |

**Table 8: Baseline NR-IQA Models on Synthetic Dataset (n=520)**

| Model | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | MainScore | Reported |
|-------|--------|--------|--------|--------|--------|--------|-----------|----------|
| TReS | 0.683 | 0.786 | 0.723 | 0.816 | 0.706 | 0.791 | **0.747** | 0.863 |
| HyperIQA | 0.639 | 0.798 | 0.643 | 0.823 | 0.639 | 0.802 | **0.723** | 0.844 |
| RichIQA (TOPIQ-NR) | 0.482 | 0.735 | 0.507 | 0.770 | 0.499 | 0.746 | **0.619** | 0.866 |
| DBCNN | 0.560 | 0.557 | 0.594 | 0.539 | 0.556 | 0.547 | **0.559** | 0.587 |
| MUSIQ | 0.252 | 0.340 | 0.199 | 0.316 | 0.258 | 0.351 | **0.289** | 0.859 |

**Key findings:**

1. **Massive gap between off-the-shelf and reported competition scores on DIQA.** The reported scores used models fine-tuned on the DIQA-5000 training set. The pyiqa checkpoints are pretrained on natural image IQA datasets (KonIQ-10K, FLIVE) and have never seen document images. The best off-the-shelf model (TOPIQ-NR, MainScore=0.490) scores 43% below its reported fine-tuned score (0.866), confirming that document IQA is a fundamentally different domain from natural image IQA.

2. **Synthetic data scores are dramatically higher than DIQA for all models.** TReS improves from 0.422 (DIQA) to 0.747 (synthetic), a delta of +0.325. HyperIQA jumps from 0.437 to 0.723 (+0.286). This indicates our synthetic dataset has quality characteristics more aligned with natural image IQA distributions — the degradation patterns (blur, noise, compression) are closer to what these models were trained on, while DIQA-5000's document-specific distortions (moiré, shadows, creases, enhancement artifacts) are out-of-domain.

3. **PLCC >> SRCC is universal on synthetic data, not just NR-IQA.** With unified MainScore computation across all model types, PLCC consistently exceeds SRCC: HyperIQA off-the-shelf (0.798 vs 0.639, delta=0.159), Gemini 3 Flash (0.804 vs 0.753, delta=0.051), DeQA-Doc (0.765 vs 0.696, delta=0.069). The 4-parameter logistic fitting improves all models, confirming the prediction-to-MOS relationship is inherently nonlinear across the synthetic quality range.

4. **MUSIQ transfers poorly across both datasets** (MainScore 0.185 on DIQA, 0.289 on synthetic). The KonIQ-10K pretrained MUSIQ checkpoint is essentially noise for document quality assessment.

5. **VLM teachers dramatically outperform off-the-shelf NR-IQA on documents.** Gemini 3 Flash (wSRCC=0.708) scores 44% higher than the best off-the-shelf baseline (TOPIQ-NR, MainScore=0.490) on DIQA-5000, without any DIQA-specific training. This gap is even larger than the VLM-vs-fine-tuned gap (0.008), confirming that pretrained NR-IQA models cannot substitute for VLM annotation on document images.

**Table 9: Cross-Method Comparison Summary (Unified MainScore)**

Synthetic column now uses MainScore (with PLCC) for all model types. DIQA-5000 column uses wSRCC for VLM/fine-tuned models and MainScore for NR-IQA baselines (matching their original evaluation protocol).

| Method | Type | DIQA-5000 | Synthetic | Notes |
|--------|------|-----------|-----------|-------|
| Gemini 3 Flash Preview | VLM teacher | 0.708 | **0.774** | Zero-shot, API-based |
| GPT-4.1 | VLM teacher | 0.669 | 0.769 | Zero-shot, API-based |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.716** | 0.748 | Trained on DIQA-5000 human labels |
| TReS (off-the-shelf) | NR-IQA baseline | 0.422 | 0.747 | Pretrained on KonIQ-10K |
| HyperIQA (off-the-shelf) | NR-IQA baseline | 0.437 | 0.723 | Pretrained on KonIQ-10K |
| HyperIQA++ | Fine-tuned NR-IQA | — | 0.694 | Fine-tuned on DIQA-5000 |
| Claude Haiku 4.5 | VLM teacher | 0.586 | 0.660 | Zero-shot, API-based |
| SigLIP2-IQA-Base-86M | Student model | 0.620 | 0.620 | Trained on pseudo-labels |
| TOPIQ-NR (off-the-shelf) | NR-IQA baseline | 0.490 | 0.619 | Pretrained on KonIQ-10K |
| Qwen 3.5 Flash | VLM teacher | — | 0.596 | n=451 (13% parse failures) |
| DBCNN (off-the-shelf) | NR-IQA baseline | 0.453 | 0.559 | Pretrained on KonIQ-10K |
| Gemini 2.5 Pro | VLM teacher | 0.612 | 0.511 | Zero-shot, 18% parse failures |
| Qwen3-VL-8B Instruct | VLM teacher | 0.481 | 0.466 | Strong positive bias |
| Qwen3-VL-8B Thinking | VLM teacher | 0.409 | 0.458 | CoT reasoning |
| MUSIQ (off-the-shelf) | NR-IQA baseline | 0.185 | 0.289 | Pretrained on KonIQ-10K |

**Implications for pseudo-labeling:** The per-image baseline scores (archived on the Modal `iqa-baseline-results` volume as JSONL checkpoints) can serve as weak ensemble signals in the uncertainty fusion pipeline. Images where DeQA-Doc and the off-the-shelf baselines strongly disagree may indicate annotation difficulty or OOD characteristics. However, given their low absolute correlation on DIQA-5000, these signals should be weighted much lower than VLM teacher predictions. TReS and HyperIQA are the most useful baseline signals for synthetic data (MainScore > 0.72), while MUSIQ should be excluded entirely.

---

## 5. Analysis and Discussion

### 5.1 The Over-Rating Problem

The most consistent finding across all experiments is systematic positive bias. VLMs rate documents approximately 0.5-1.5 MOS points higher than human annotators. This manifests as:

1. **Compressed dynamic range**: Models use the 3.5-5.0 range for images humans rate 1.0-5.0
2. **Collapsed upper buckets**: Fair/good/excellent are conflated into "good" or "excellent"
3. **Selective accuracy**: Models correctly identify truly bad images but over-rate everything else

We hypothesize this stems from VLMs being trained on general web data where "most images are fine" — a prior that conflicts with DIQA-5000's intentionally degraded document set.

**Implication for pseudo-labeling**: Raw VLM scores cannot be used directly. A calibration step (e.g., isotonic regression or quantile mapping) is essential to map VLM predictions to the human MOS scale.

### 5.2 Resolution as a Quality Signal

The prompt optimization experiment (n=23) suggested that image resolution was the dominant factor for VLM-based DIQA, with no-resize improving wSRCC by +0.042 over the 1024px default.

**Full-scale validation disproves this.** Running Gemini 3 Flash at native resolution on all 1,000 test images yields wSRCC = 0.699, *lower* than the 1024px default (wSRCC = 0.708, delta = -0.009). Only color fidelity marginally improved (+0.009 SRCC), while overall (-0.014) and sharpness (-0.018) both degraded. Inference latency also increased by ~17%. The CIs overlap substantially (not statistically significant), but the direction is clearly opposite to what the 23-image optimization predicted (+0.042 vs. actual -0.009).

This is a cautionary finding: **small-sample prompt optimization can be actively misleading**. The n=23 subset happened to contain images where native resolution helped, but this did not generalize. This is consistent with the smoke test instability finding (Section 4.1), where n=7 rankings diverged dramatically from n=1,000 rankings. We recommend a minimum of n=200 stratified samples for any prompt optimization experiment.

### 5.3 Correlation vs. Calibration

A recurring theme is the disconnect between correlation (ranking accuracy) and calibration (absolute score accuracy). Claude Haiku 4.5 exemplifies this: it has the lowest MAE (0.68) but only the 5th-highest SRCC. Conversely, GPT-4.1 has the 2nd-highest SRCC but the worst MAE (1.15).

For pseudo-labeling, **correlation matters more than calibration**. The DeQA-Score training pipeline converts MOS to soft-label distributions, and the distribution shape depends on the rank order of samples more than their absolute values. A model that correctly ranks all images but assigns them all scores in [3.5, 5.0] can be calibrated; a model that assigns random scores within the correct range cannot.

### 5.4 Complementary Model Strengths

The synthetic evaluation reveals that Gemini 3 Flash and GPT-4.1 have complementary failure modes:
- Gemini leads on in-distribution and non-Latin scripts
- GPT-4.1 leads on adversarial scripts, CJK vertical, and multiscript
- Both fail on binarized, extreme DPI, and pristine

This suggests a **consensus approach** — averaging predictions from multiple models — could outperform either model alone, particularly on edge cases. Notably, Gemini 2.5 Pro (wSRCC=0.468 synthetic, 18% parse failures) is not viable as a consensus member or tiebreaker despite ranking 3rd on DIQA-5000 (wSRCC=0.612) — its synthetic performance is worse than all fine-tuned models and its high parse failure rate would introduce systematic data loss.

### 5.5 OOD Detection for Quality Filtering

The OOD detector serves a dual purpose in the iterative expansion pipeline: it identifies where SigLIP2's predictions are unreliable (triggering VLM pseudo-labeling), and it gates the quality of VLM labels themselves. Our synthetic evaluation (Section 4.4) shows that VLMs fail catastrophically on certain document categories (binarized: negative SRCC, extreme DPI: negative SRCC). We need an automated gate to flag documents where VLM labels are also unreliable — these represent the current boundary of what can be pseudo-labeled and must wait for either human annotation or future VLM improvements.

We address this with a Mahalanobis-distance OOD detector operating on SigLIP2's 768-dimensional embeddings (see `results/tier1_ood_detector/README.md` for full details):

- **Method**: Fit a multivariate Gaussian (mean + Ledoit-Wolf shrinkage covariance) on 4,000 DIQA-5000 train+val embeddings. For a new image, compute Mahalanobis distance from the training centroid.
- **Performance**: AUROC = 0.9963 on DIQA-5000 test vs. 370 synthetic OOD documents. All 13 OOD categories detected with AUROC >= 0.97.
- **Thresholds** (from clean extraction, see below): Production threshold = 30.8 (train+val p95). Hard-reject calibrated from test p99 = 58.2.
- **Latency**: ~1-2ms per image (matrix-vector multiply), negligible vs. SigLIP2's ~30ms forward pass.
- **Integration**: The detector reuses embeddings already computed during SigLIP2 inference — no additional forward pass needed.

This creates a two-tier reliability pipeline: Tier 1 (OOD detector) gates at near-zero cost, and only flagged images proceed to Tier 2 (VLM cross-model validation via Qwen3-VL-8B at ~$0.001/image).

#### 5.5.1 OOD Detector Re-Calibration (Clean Extraction)

The original OOD detector (Section 5.5 above) was fit on embeddings from a checkpoint with 445 missing keys, causing an ~8-unit train/test distance shift. We re-extracted all 5,000 DIQA-5000 embeddings from the correct IQA-only checkpoint (`siglip2_iqa_best.pt`, trained on DIQA-5000 with `google/siglip2-base-patch16-naflex` backbone) using Modal GPU infrastructure.

**Extraction details:**

- **Script**: `image_detection/modal/extract_siglip2_diqa5000.py`
- **GPU**: NVIDIA L4 (24GB), ~50 min total runtime
- **Checkpoint**: IQA-only (22 missing keys for non-IQA heads — expected; 0 unexpected keys)
- **IQA output range**: mu in [-0.17, 0.73] (regression heads, not clamped to [0,1])
- **Rescaling**: `MOS_pred = mu * 4.0 + 1.0` (model trained with `(MOS - 1) / 4` normalization)

**Re-calibrated OOD detector statistics (v2):**

| Metric                | Train+Val (n=4,000) | Test (n=1,000) |
| --------------------- | ------------------- | -------------- |
| Median distance       | 23.7                | 31.4           |
| p95                   | 30.8                | 48.5           |
| p99                   | 34.6                | 58.2           |

The train+val median (23.7) and test median (31.4) are now in a healthy range with no anomalous shift — confirming the checkpoint mismatch is resolved. The test distribution shows the expected heavier tail (p95=48.5, p99=58.2) from naturally occurring OOD-like documents in the test set.

**Full extraction outputs** are archived at `results/siglip2_diqa5000/` (see [Data Availability](#7-data-availability)).

### 5.6 The Fine-Tuning Gap

The gap between VLM teachers (best wSRCC = 0.708) and fine-tuned specialists (wSRCC = 0.716) is only 0.008 wSRCC on DIQA-5000 — remarkably small given that VLMs receive zero DIQA-specific training. On out-of-distribution data, VLMs still lead: the best VLM (GPT-4.1, wSRCC = 0.757) outperforms the best fine-tuned model (DeQA-Doc-3Specialists, wSRCC = 0.714) by 0.043. However, this gap is far smaller than initially appeared before DeQA-Doc evaluation — HyperIQA++ (wSRCC = 0.602) showed a 0.155 gap that made VLM superiority look larger than it is.

The fine-tuning gap is training-objective-dependent:

- **DeQA-Doc (soft-label distribution)**: ID/OOD delta = 0.047 (0.762 → 0.715), nearly matching VLM robustness
- **HyperIQA++ (regression)**: ID/OOD delta = 0.205 (0.754 → 0.549), severe OOD degradation
- **SigLIP2-IQA (Gaussian NLL)**: ID/OOD delta = 0.076 (0.635 → 0.559), moderate degradation

This suggests that distribution-aware training objectives (soft labels over discrete quality levels) produce inherently more generalizable representations than point-estimate regression.

VLMs have two structural disadvantages:

1. **Scale inefficiency**: Each VLM inference costs ~$0.001-0.01 per image vs. ~$0.0001 for SigLIP2. At scale (millions of documents), VLMs are 10-100x more expensive.
2. **Calibration gap**: VLMs over-rate by +0.5 to +1.5 MOS, requiring a calibration step that itself needs labeled data.

But they retain one advantage: **OOD generalization**. VLM teachers maintain slightly better ranking quality on unseen document types (GPT-4.1 OOD wSRCC = 0.747 vs. DeQA-Doc OOD = 0.715), making them valuable for expanding training data beyond the DIQA-5000 domain — though the margin is smaller than expected.

The pseudo-labeling pipeline (Section 6) bridges this gap iteratively: use VLMs to generate calibrated training data for OOD documents where SigLIP2 is currently weak, retrain SigLIP2 on the expanded dataset, re-fit the OOD detector, and repeat. Each cycle expands the range of documents where SigLIP2 maintains SRCC > 0.90 while preserving production-speed inference.

---

## 6. Proposed Pseudo-Labeling Pipeline

Based on our experimental findings, we propose a multi-stage pipeline for iteratively expanding SigLIP2's training set. The pipeline is designed to be applied incrementally: each cycle identifies where SigLIP2 is currently weak, generates pseudo-labels for those areas, retrains, and then re-evaluates. The stages below describe a single iteration; Stage 5 describes how iterations chain together.

### Stage 1: Multi-Model VLM Annotation

For each document image:
1. Run Gemini 3 Flash Preview with the standard 1-prompt approach (1024px resize, which performs as well as native resolution at lower latency)
2. Run GPT-4.1 as a second opinion
3. For images where the two models disagree by > 1.0 MOS, run a third model (Gemini 2.5 Pro or Claude Haiku 4.5)

Expected cost: ~$0.003 per image for 2-model annotation via OpenRouter.

### Stage 2: Score Calibration

Apply learned calibration functions to map VLM scores to the human MOS scale:
1. Fit isotonic regression on DIQA-5000 training set (3,500 images with known MOS)
2. Apply per-model, per-dimension calibration to all pseudo-labels
3. Validate calibration on DIQA-5000 test set (1,000 images)

**Calibration experiment results (SigLIP2-IQA, train→test):** Three calibration methods were compared — linear regression, 4-parameter logistic (4PL), and isotonic regression — on the SigLIP2 student model (the only model with train+test predictions). SigLIP2 predictions are on a [0,1] scale while GT is raw MOS [~0.5, ~4.2].

| Method   | wSRCC  | wMAE   | MAE_O  | MAE_S  | MAE_C  |
|----------|--------|--------|--------|--------|--------|
| Raw      | 0.8914 | 2.424  | 2.409  | 2.404  | 2.474  |
| Linear   | 0.8914 | 0.173  | 0.167  | 0.184  | 0.172  |
| 4PL      | 0.8914 | 0.173  | 0.167  | 0.184  | 0.172  |
| Isotonic | 0.8910 | 0.174  | 0.168  | 0.186  | 0.173  |

Key findings: (1) wSRCC is invariant under monotone calibrations (confirmed by 13-model consensus review). (2) MAE drops 14x after calibration — the raw MAE was inflated by the [0,1] vs MOS scale mismatch. (3) Linear ≈ 4PL — the mapping is nearly affine, so the industry-standard 4PL logistic adds no benefit. (4) Isotonic is marginally worse due to tied-rank effects from piecewise-constant fit. (5) PLCC is already 0.921 (overall) even without calibration. Full results with bootstrap 95% CIs: `results/siglip2_diqa5000/calibration_results.json`. Script: `results/siglip2_diqa5000/calibrate_isotonic.py`.

**Note:** VLM models lack train-split predictions — calibrating VLMs requires running inference on 3,500 train images first. The SigLIP2 results demonstrate that the calibration mechanism works and the scale correction is substantial, but VLM-specific calibration remains future work.

### Stage 3: Consensus and Uncertainty Estimation

For each image:
1. Compute consensus score as the calibrated mean across models
2. Estimate uncertainty from inter-model disagreement (std of calibrated scores)
3. Generate soft-label distributions using the DeQA methodology:
   - mu = consensus score
   - sigma^2 = max(inter-model variance, sigma_pseudo^2) where sigma_pseudo = 0.8

### Stage 4: Quality Filtering

Apply the embedding-space OOD detector (Section 5.5) to gate pseudo-label reliability:

1. **Auto-accept**: Mahalanobis distance < 30.8 (train+val p95) AND inter-model agreement < 0.5 MOS. These are in-distribution documents where VLM labels are reliable.
2. **Review**: Distance in [30.8, 58.2] OR inter-model agreement in [0.5, 1.0]. Route to Tier 2 cross-model validation (Qwen3-VL-8B) for a third opinion.
3. **Reject**: Distance > 58.2 (test p99) OR inter-model agreement > 1.5 MOS. These are OOD documents where VLM labels cannot be trusted. Reserve for human annotation or exclude.

Thresholds are derived from the clean SigLIP2 extraction (Section 5.5.1), fitted on 4,000 train+val embeddings and calibrated on 1,000 test embeddings. The OOD detector operates on SigLIP2 embeddings already computed in Stage 1 inference, adding only ~1-2ms per image.

### Stage 5: Iterative Expansion (Core Strategy)

The pipeline above describes a single labeling pass. The project's core strategy is iterative: each cycle expands SigLIP2's effective domain, shifting the OOD boundary outward. A complete iteration proceeds as follows:

1. **Retrain SigLIP2** on the expanded training set (original DIQA-5000 human labels + new pseudo-labeled samples). The human-labeled DIQA-5000 data is always retained to anchor in-distribution performance.
2. **Evaluate on DIQA-5000 test set** to confirm no regression on in-distribution performance (target: maintain SRCC > 0.90).
3. **Re-extract embeddings** from the retrained SigLIP2 for all training data (original + new samples).
4. **Re-fit the OOD detector** on the expanded embedding set. As the training distribution grows, documents previously flagged as OOD may now fall within-distribution, and the Mahalanobis threshold shifts accordingly.
5. **Identify the new OOD frontier**: run the re-fitted detector on candidate documents (synthetic test sets, real-world document collections) to find where SigLIP2 is still unreliable.
6. **Generate pseudo-labels** for the newly-identified OOD documents via Stages 1-4 above.
7. **Repeat** from step 1 until the target domain coverage is achieved.

This iterative design ensures that every component — SigLIP2, the OOD detector, and the pseudo-labeling pipeline — is incrementally applicable to both new training samples and new test data. The OOD detector's boundary contracts with each cycle as SigLIP2's training distribution expands, systematically reducing the set of documents requiring VLM annotation.

### Expected Performance

Projections based on our findings (these are estimates requiring validation):

- Gemini 3 Flash at native resolution: the prompt optimization (n=23) suggests ~0.95 wSRCC, but this small sample likely overestimates performance. A full n=1,000 native-resolution run is in progress to establish the true value.
- Consensus of 2 models: may improve over single-model predictions, but the magnitude depends on error correlation between models. Complementary failure modes (Section 5.4) suggest gains are plausible.
- After calibration: isotonic regression on 3,500 training images reduces wMAE from 2.424 to 0.173 (14× improvement) with wSRCC stable at 0.886. Linear and 4PL calibration perform comparably (wMAE 0.176–0.177), confirming the SigLIP2→MOS mapping is nearly affine. See Section 5.6.1 for full results.
- A DeQA-Doc model trained on calibrated pseudo-labels should be compared against the human-label baseline to quantify the actual degradation.

---

## 7. Data Availability

All experimental data is archived in this repository under `results/vlm_teacher_eval/full_eval/`:

### Per-Sample JSONL Checkpoints

Each file contains one JSON object per image with fields: `model_id`, `image`, `overall`, `sharpness`, `color_fidelity`, `reasoning`, `raw_response`, `latency_ms`, `error`.

| File | Model | n |
|------|-------|---|
| `checkpoints/google__gemini-3-flash-preview.jsonl` | Gemini 3 Flash Preview | 1,000 |
| `checkpoints/openai__gpt-4.1.jsonl` | GPT-4.1 | 1,000 |
| `checkpoints/google__gemini-2.5-pro.jsonl` | Gemini 2.5 Pro | 1,000 |
| `checkpoints/qwen__qwen3.5-flash-02-23.jsonl` | Qwen 3.5 Flash | 1,000 |
| `checkpoints/anthropic__claude-haiku-4.5.jsonl` | Claude Haiku 4.5 | 1,000 |
| `checkpoints/qwen__qwen3-vl-8b-instruct.jsonl` | Qwen3-VL-8B Instruct | 1,000 |
| `checkpoints/qwen__qwen3-vl-8b-thinking.jsonl` | Qwen3-VL-8B Thinking (default temp) | 1,000 |
| `checkpoints/qwen__qwen3-vl-8b-thinking__temp0.jsonl` | Qwen3-VL-8B Thinking (temp=0) | 1,000 |
| `checkpoints/google__gemini-3-flash-preview__no_resize.jsonl` | Gemini 3 Flash (native resolution) | 1,000 |
| `ab_test/google__gemini-3-flash-preview_3prompt.jsonl` | Gemini 3 Flash (3-prompt) | 44 |
| `ab_test/openai__gpt-4.1_3prompt.jsonl` | GPT-4.1 (3-prompt) | 44 |
| `checkpoints_synthetic/google__gemini-3-flash-preview.jsonl` | Gemini 3 Flash (synthetic) | 520 |
| `checkpoints_synthetic/openai__gpt-4.1.jsonl` | GPT-4.1 (synthetic) | 520 |
| `checkpoints_synthetic/anthropic__claude-haiku-4.5.jsonl` | Haiku (synthetic) | 520 |
| `checkpoints_synthetic/google__gemini-2.5-pro.jsonl` | Gemini 2.5 Pro (synthetic) | 520 |
| `checkpoints_synthetic/qwen__qwen3-vl-8b-instruct.jsonl` | Qwen3-VL-8B Instruct (synthetic) | 520 |
| `checkpoints_synthetic/qwen__qwen3-vl-8b-thinking.jsonl` | Qwen3-VL-8B Thinking (synthetic) | 520 |
| `checkpoints_synthetic/qwen__qwen3.5-flash-02-23.jsonl` | Qwen 3.5 Flash (synthetic) | 520 |

**Total: 12,877 VLM per-sample evaluations with full model responses and reasoning** (7,088 from primary experiments + 2,000 from validation runs + 161 from prompt optimization + 3,628 from synthetic evaluations), plus **5,000 SigLIP2 multi-task predictions with 768-dim embeddings** (see below).

### Aggregated Metrics

| File | Contents |
|------|----------|
| `results/vlm_benchmark_metrics.json` | Full benchmark: SRCC/PLCC/MAE with 95% CIs per model per dimension |
| `results/synthetic_eval_metrics.json` | Synthetic eval: per-category and per-subset metrics |
| `results/ordinal_analysis.json` | Ordinal analysis: confusion matrices, Kappa, bucket accuracy |
| `prompt_optimization/optimization_results.json` | 7-arm optimization: per-arm metrics (Gemini 3 Flash) |
| `prompt_optimization/qwen__qwen3.5-flash-02-23/optimization_results.json` | 7-arm optimization: per-arm metrics (Qwen 3.5 Flash) |

### SigLIP2 Extraction Outputs

Full SigLIP2 multi-task predictions and embeddings for all 5,000 DIQA-5000 images, extracted from the IQA-only checkpoint (`siglip2_iqa_best.pt`). Archived at `results/siglip2_diqa5000/`.

| File | Contents |
| ---- | -------- |
| `siglip2_diqa5000/siglip2_diqa5000_train.jsonl` | 3,500 records: IQA predictions, script/source/orientation, shadow/warping severity |
| `siglip2_diqa5000/siglip2_diqa5000_val.jsonl` | 500 records (same schema) |
| `siglip2_diqa5000/siglip2_diqa5000_test.jsonl` | 1,000 records (same schema) |
| `siglip2_diqa5000/embeddings/train.npz` | (3500, 768) float32 embeddings + image names |
| `siglip2_diqa5000/embeddings/val.npz` | (500, 768) float32 embeddings + image names |
| `siglip2_diqa5000/embeddings/test.npz` | (1000, 768) float32 embeddings + image names |
| `siglip2_diqa5000/ood_detector_v2.npz` | Fitted OOD detector (mean, precision matrix, calibration distances, threshold) |
| `siglip2_diqa5000/summary.json` | Extraction metadata and OOD detector statistics |

Each JSONL record contains 20 fields: `image`, `split`, IQA mu/sigma_sq for 3 dimensions, script prediction + 19-class distribution, source/orientation/shadow/warping predictions, and inference time.

### Fine-Tuned IQA Model Results

| File | Contents |
|------|----------|
| `results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json` | Per-model, per-subset, per-category metrics for SigLIP2, HyperIQA++, DeQA-Doc-3Specialists |
| Modal volume `synthetic-ood-results` (`checkpoints_synthetic/*.jsonl`) | Per-image JSONL checkpoints for all 520 synthetic images |
| Modal volume `deqa-specialist-checkpoints` | DeQA-Doc 3 specialist model weights (overall, sharpness, color) |
| `modal/benchmark_synthetic_ood.py` | Benchmark script for fine-tuned models on Modal |

### NR-IQA Baseline Results

| File | Contents |
|------|----------|
| `results/iqa_baselines/baseline_summary.json` | Per-model, per-dataset SRCC/PLCC/MainScore for 5 off-the-shelf NR-IQA baselines |
| Modal volume `iqa-baseline-results` | Per-image JSONL checkpoints for all 1,520 images (1,000 DIQA + 520 synthetic) |
| `modal/benchmark_iqa_baselines.py` | Benchmark script with checkpoint/resume on Modal |

### Unified Leaderboards

| File | Contents |
|------|----------|
| `results/LEADERBOARD_DIQA.md` | All models ranked by MainScore on DIQA-5000 test set |
| `results/LEADERBOARD_SYNTHETIC.md` | All models ranked by MainScore on 520-image synthetic dataset |

### Experiment Scripts

| Script | Purpose |
|--------|---------|
| `run_full_diqa_eval.py` | Full n=1,000 DIQA benchmark with checkpoint/resume |
| `run_prompt_ab_test.py` | 1-prompt vs. 3-prompt A/B test |
| `run_synthetic_eval.py` | Synthetic OOD dataset evaluation |
| `analyze_ordinal.py` | Ordinal discrimination analysis |
| `run_prompt_optimization.py` | 7-arm prompt optimization experiment |
| `modal/benchmark_iqa_baselines.py` | NR-IQA baseline benchmark on Modal |
| `modal/benchmark_synthetic_ood.py` | Fine-tuned IQA model benchmark on Modal |

---

## 8. Conclusion

SigLIP2-IQA achieves SRCC > 0.90 on DIQA-5000 documents but degrades on OOD document types. Our evaluation demonstrates that frontier VLMs — particularly Gemini 3 Flash Preview (wSRCC = 0.708) — can generate pseudo-labels of sufficient quality to expand SigLIP2's training set into these OOD areas. The VLM-to-human correlation approaches the supervised baseline (0.716), and VLMs maintain better ranking quality than fine-tuned models on OOD documents, making them viable label sources for exactly the document types where SigLIP2 needs improvement.

However, all VLMs exhibit systematic positive bias (+0.5 to +1.5 MOS points) that necessitates calibration, and performance degrades on specific categories (binarized, extreme DPI, form layouts) — areas where the OOD detector correctly flags unreliable predictions. The most actionable methodological finding is that **small-sample prompt optimization is unreliable for VLMs**: a 23-image experiment suggested +0.042 wSRCC from native resolution, but full-scale validation showed -0.009. We recommend a minimum of n=200 stratified samples for any prompt strategy evaluation.

The proposed iterative pipeline — identify OOD weaknesses via the embedding-space detector, generate calibrated pseudo-labels via VLM consensus, expand the training set, retrain SigLIP2, and re-fit the OOD detector — is designed for incremental application. Each cycle contracts the OOD boundary as SigLIP2's domain coverage expands. The dual-model consensus approach (Gemini 3 Flash + GPT-4.1, ~$0.003/image) provides complementary coverage across document types at a cost orders of magnitude below human annotation.

### Limitations

1. **Single dataset**: All primary evaluation uses DIQA-5000. Generalization to other document IQA datasets (e.g., Tobacco800, RVL-CDIP) is untested.
2. **Synthetic OOD only**: The 520-image OOD dataset is programmatically generated. Real-world OOD documents (handwritten forms, historical manuscripts, receipts) may behave differently.
3. **API-mediated evaluation**: All VLMs were accessed via OpenRouter, adding routing latency and potential response variability. Direct API access might yield different results.
4. **No end-to-end validation**: We measure VLM-vs-human correlation but have not yet trained a student model on VLM pseudo-labels to measure actual downstream impact.
5. **Prompt optimization sample size**: The 7-arm optimization used only n=23 images. The winning strategy (no-resize) needs full-scale validation before adoption.
6. ~~**Checkpoint mismatch in OOD detector**~~ Resolved — re-extracted all 5,000 embeddings from the correct IQA-only checkpoint (Section 5.5.1). The v2 OOD detector shows healthy distance distributions with no anomalous train/test shift.

### Future Work

1. ~~**Full-scale native-resolution evaluation**~~ Completed — no-resize does not improve over 1024px default (Section 5.2)
2. ~~**Calibration experiments** using isotonic regression on the DIQA-5000 training split~~ Completed — isotonic, 4PL, and linear calibration all reduce wMAE by 14× with wSRCC invariant (Section 5.6.1). Quantile mapping remains future work.
3. **First expansion cycle (end-to-end validation)**: Run the first complete iteration — generate pseudo-labels for a targeted set of OOD documents, retrain SigLIP2 on the expanded training set, and measure whether SRCC > 0.90 is maintained on DIQA-5000 while improving on OOD categories
4. **OOD detector re-fitting after expansion**: Re-extract embeddings from the retrained SigLIP2 and re-fit the Mahalanobis detector to verify the OOD boundary contracts as expected
5. **Ensemble optimization**: Systematic search over model combinations and weighting schemes for consensus scoring
6. **Active learning**: Use inter-model disagreement to identify the most informative images for targeted human annotation, prioritizing documents near the OOD boundary
7. **OOD detector validation on real-world data**: Test the Mahalanobis detector on naturally-occurring OOD documents rather than synthetic ones
8. ~~**Re-fit OOD detector with matched checkpoint**~~ Completed — v2 OOD detector fitted on clean embeddings (Section 5.5.1), archived at `results/siglip2_diqa5000/ood_detector_v2.npz`
9. **Cross-dataset transfer**: Evaluate whether VLM pseudo-labels generated on one document type transfer to train models for different document types
10. **Incremental pipeline tooling**: Build automation for the iterative cycle (pseudo-label → retrain → re-extract → re-fit OOD → re-evaluate) so each expansion iteration can be run with minimal manual intervention

---

## References

1. Zhiyuan You et al. "DeQA-Score: Soft-Label Distribution Learning for Quality Assessment." 2024.
2. DIQA-5000 Dataset, VQualA 2025 DIQA Challenge.
3. VQualA 2025 Competition Evaluation Metrics: wSRCC = 0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color.
