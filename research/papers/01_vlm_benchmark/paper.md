# VLM Benchmark for Document Image Quality Assessment

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 1/10
**Repository:** `results/vlm_teacher_eval/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams
**Keywords:** document image quality assessment, vision-language models, DIQA-5000, pseudo-labeling, benchmark, VQualA

---

> **Update (2026-03-09):** This paper was originally written around 7 VLMs (batch 1). A batch 2 evaluation added 14 more VLMs (21 total). Key tables and data sections have been updated. The original 7-model analysis and findings are confirmed and strengthened by the expanded benchmark. See [LEADERBOARD_DIQA.md](../../../results/LEADERBOARD_DIQA.md) and [LEADERBOARD_SYNTHETIC.md](../../../results/LEADERBOARD_SYNTHETIC.md) for the full 28-model rankings.

## Abstract

Document Image Quality Assessment (DIQA) requires predicting how well a scanned or photographed document can be read by humans. Obtaining human quality annotations is expensive -- the DIQA-5000 dataset required approximately 225,000 individual ratings from 15 annotators -- creating a bottleneck for expanding training data to new document types. We evaluate 21 frontier Vision-Language Models (VLMs) as automated quality annotators on 1,000 DIQA-5000 test images across three dimensions (overall quality, sharpness, color fidelity), comparing their zero-shot ratings against human Mean Opinion Scores (MOS). Our best model, Gemini 3 Flash Preview, achieves MainScore = 0.743 (wSRCC = 0.708), approaching the supervised DeQA-Doc-3Specialists baseline (wSRCC = 0.716) without any DIQA-specific training. All VLMs exhibit systematic positive bias (+0.5 to +1.5 MOS points), collapsing the fair/good/excellent distinction while preserving rank order. We find that small-sample prompt optimization is unreliable: a 23-image experiment suggested +0.042 wSRCC from native resolution, but full-scale validation showed -0.009. Cross-domain evaluation on 520 synthetic out-of-distribution documents confirms that VLMs maintain ranking quality on non-Latin scripts (SRCC 0.73-0.85) but fail on binarized documents, extreme DPI, and pristine digital originals. We release per-sample VLM evaluations with full model reasoning to support reproducible research in VLM-based quality annotation.

---

## 1. Introduction

### 1.1 Motivation

Document Image Quality Assessment predicts how well a scanned or photographed document can be read by humans. High-quality DIQA models enable document processing pipelines to flag poor-quality scans for re-capture, route documents to appropriate OCR engines, and prioritize archival efforts. The field has undergone a paradigm shift: at the VQualA 2025 DIQA Challenge (ICCV 2025), the top four of seven teams all used multimodal large language models (MLLMs), scoring above 0.92 on the competition metric, while CNN-based approaches plateaued below 0.90 [1, 2].

Our SigLIP2-IQA student model achieves SRCC > 0.90 on DIQA-5000 in-distribution documents but degrades on document types not represented in its training set -- non-Latin scripts, unusual layouts, extreme degradation patterns. Expanding its effective domain requires new labeled training data, but the DIQA-5000 annotation protocol required 15 human annotators per image across 3 quality dimensions, making extension prohibitively costly. This chicken-and-egg problem -- needing labels for exactly the documents where existing models fail -- motivates the investigation of VLMs as automated quality annotators.

VLMs offer a potential solution: frontier multimodal models can assess document quality in a zero-shot setting, generating pseudo-labels for out-of-distribution (OOD) documents without additional human annotation. If VLM ratings correlate sufficiently with human judgments, they can provide the supplemental training data needed to expand the student model's effective domain. Each expansion cycle then shifts the OOD boundary outward, systematically broadening coverage.

This paper -- the first in a seven-part technical report series -- establishes the empirical foundation for this approach by benchmarking 21 VLMs against human quality judgments on 1,000 DIQA-5000 test images.

### 1.2 Contributions

1. A benchmark of 21 VLMs on 1,000 DIQA-5000 test images with bootstrapped 95% confidence intervals across 3 quality dimensions, contextualized against the broader DIQA model landscape.
2. An ordinal discrimination analysis revealing that all VLMs exhibit systematic positive bias, with over-rating ranging from 64% to 98% of images.
3. Evidence that small-sample prompt optimization (n=23) can be actively misleading, with results that fail to replicate at full scale.
4. Cross-domain evaluation on 520 synthetic OOD images spanning 13 categories, identifying both robust transfer (non-Latin scripts) and universal failure modes (binarized, extreme DPI).
5. Release of 12,877 per-sample VLM evaluations with full model reasoning and metadata to support reproducible research.

### 1.3 Series Context

This paper establishes the VLM benchmark that subsequent reports build upon. Paper 2 analyzes cross-domain generalization in depth. Paper 3 compares VLMs against traditional NR-IQA baselines. Paper 4 describes the OOD detection system. Paper 5 benchmarks off-the-shelf NR-IQA models. Paper 6 validates that quality scores predict OCR accuracy. Paper 7 integrates these components into a pseudo-labeling pipeline.

---

## 2. Task Definition & Related Work

### 2.1 Task Definition

Given a document image, the task is to predict continuous quality scores across three perceptual dimensions:

- **Overall Quality**: Holistic readability, usability, and visual clarity of the document.
- **Sharpness**: Text edge clarity, blur level, resolution adequacy, and suppression of algorithmic artifacts.
- **Color Fidelity**: Color accuracy, contrast, white balance, and tonal reproduction relative to the physical document.

Each dimension is scored on a 1.0-5.0 continuous scale derived from human Mean Opinion Scores (MOS), where 1.0 represents completely unusable quality and 5.0 represents excellent quality. The evaluation metric, following the VQualA 2025 competition, is weighted SRCC:

```
wSRCC = 0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color
```

This weighting reflects the primacy of overall document utility, with subsidiary importance given to the specific optical properties of sharpness and color rendering.

### 2.2 Related Work

**Document IQA datasets.** Prior to DIQA-5000, document image quality was assessed primarily through proxy metrics such as OCR character error rate or full-reference metrics (PSNR, SSIM) that require pristine reference images unavailable in real capture scenarios [3]. DIQA-5000 (Ma et al., arXiv:2509.17012) fills this gap by providing multi-dimensional MOS from human annotators, enabling direct training and evaluation of no-reference perceptual quality models. Created by researchers at Shanghai Jiao Tong University, the dataset contains 5,000 enhanced document images derived from 500 mobile-captured originals -- curated from publicly accessible PDFs spanning text, tables, diagrams, handwritten notes, and mixed layouts in English, Chinese, and mathematical notation. Each original was subjected to one of 5 distortion categories (shadows, occlusions, blur, creases, moire) and then enhanced via 6 operations (dewarping, demoire, occlusion removal, deblurring, deshadowing, appearance enhancement), with ratings from 15 of 23 trained annotators following ITU-R BT.500 guidelines [4]. The accompanying DocIQ baseline achieves SRCC 0.870 and PLCC 0.900. As of March 2026, the VQualA 2025 DIQA Challenge remains the only competition to have used DIQA-5000, though it has been adopted as a benchmark in subsequent research including the Q-Doc study [6].

**VQualA 2025 DIQA Challenge.** Held at the inaugural Visual Quality Assessment Competition workshop (ICCV 2025, Honolulu, October 19, 2025), the DIQA track attracted 120 registered participants across 16 active teams (183 development submissions, 97 test submissions), with 7 teams submitting final verified solutions. The winning team (DeQA-Doc, score 0.929) adapted the DeQA-Score soft-label distribution learning framework [5] to document images. The top four teams all used MLLMs (mPLUG-Owl2, Qwen2.5-VL, SigLIP2-NaFlex), while CNN-based approaches scored below 0.90. The performance gap between rank 4 (MLLM, 0.924) and rank 5 (CNN, 0.898) marked a clear tier boundary, signaling that multimodal language models possess a superior inductive bias for document quality reasoning [2]. A detailed competition analysis covering all team methodologies is provided in Paper 0 of this series.

**VLMs for image quality assessment.** The Q-Doc benchmark [6] evaluated MLLMs on zero-shot DIQA using DIQA-5000, finding that even GPT-4o achieves only SRCC 0.132, while DeepSeek-VL2 reaches 0.447 -- far below supervised results. Our work differs in three respects: we use structured JSON output with explicit dimension ratings rather than open-ended quality descriptions; we evaluate more recent models (Gemini 3 Flash, GPT-4.1, Qwen3-VL); and we provide per-sample scores for all evaluations rather than aggregate metrics only.

**Soft-label distribution learning.** DeQA-Score [5] treats quality as a probability distribution over five discrete levels (excellent/good/fair/poor/bad) rather than a point estimate. The training loss combines KL divergence between predicted and ground-truth distributions with in-level concentration and pairwise ranking terms. This approach requires both a mean (MOS) and variance (annotation disagreement) for each training sample. DIQA-5000 provides MOS but not variance, leading competition teams to engineer pseudo-variance injection and linear interpolation schemes [1].

**General NR-IQA models.** Models designed for natural images (DBCNN, HyperIQA, MUSIQ, TReS, RichIQA) transfer poorly to documents. Off-the-shelf, the best achieves MainScore 0.490 on DIQA-5000 -- less than half the competition ceiling of 0.929 [2]. Fine-tuning on DIQA-5000 dramatically improves performance (MUSIQ: 0.185 to 0.859, a 4.6x improvement), confirming that document IQA is a fundamentally distinct domain from natural image IQA.

---

## 3. Experimental Setup

### 3.1 Datasets

**Primary: DIQA-5000 Test Set (n=1,000).** One thousand real document images with human MOS ground truth across three dimensions. The test set exhibits strong class imbalance: 613 images fall in the "fair" quality range (MOS 2.6-3.4) for overall quality, with only 5 images rated "excellent" (MOS >= 4.0) and 56 rated "bad" (MOS < 1.8). This concentration in the middle range makes fine-grained discrimination challenging.

| Quality Bucket | MOS Range | Overall (n) | Sharpness (n) | Color Fidelity (n) |
|---|---|---|---|---|
| Bad | [1.0, 1.8) | 56 | 56 | 37 |
| Poor | [1.8, 2.6) | 180 | 172 | 162 |
| Fair | [2.6, 3.4) | 613 | 620 | 606 |
| Good | [3.4, 4.0) | 146 | 141 | 184 |
| Excellent | [4.0, 5.0] | 5 | 11 | 11 |

**Secondary: Synthetic OOD Dataset (n=520).** Programmatically generated images with controlled degradation parameters spanning 15 categories: in-distribution standard (n=100) and Cyrillic (n=50); OOD non-Latin scripts including Tibetan, Myanmar, and Ethiopic (n=30 each); adversarial scripts including Fraktur and Nastaliq (n=20 each); layout variants including CJK vertical and form layouts (n=30 each); extreme degradation categories including binarized and heavily degraded (n=30 each); DPI extremes (n=30 each); multiscript (n=30); and pristine digital originals (n=30). Ground truth MOS is derived from generation parameters.

### 3.2 Models

We evaluated 21 VLMs spanning different architectures, price points, and providers. An initial batch of 7 models was selected via a 26-model smoke test (n=7 images per model); 14 additional models were added in a second batch to broaden coverage.

| Model | Provider | Architecture Type | Cost ($/M in / $/M out) |
|---|---|---|---|
| Gemini 3 Flash Preview | Google | Multimodal | $0.50 / $3.00 |
| GPT-4.1 | OpenAI | Multimodal | $2.00 / $8.00 |
| Gemini 2.5 Pro | Google | Multimodal reasoning | $1.25 / $10.00 |
| Qwen 3.5 Flash | Alibaba | Reasoning + vision | $0.10 / $0.40 |
| Claude Haiku 4.5 | Anthropic | Multimodal | $1.00 / $5.00 |
| Qwen3-VL-8B Instruct | Alibaba | Vision-language (8B) | $0.08 / $0.50 |
| Qwen3-VL-8B Thinking | Alibaba | Vision-language + CoT (8B) | $0.12 / $1.37 |
| Qwen 3.5 Plus | Alibaba | Strong reasoning | $0.26 / $1.56 |
| Qwen 3.5 122B-A10B | Alibaba | MoE vision | $0.26 / $2.08 |
| Qwen3-VL-235B Instruct | Alibaba | Vision-language (235B) | $0.20 / $0.88 |
| Gemini 3.1 Flash Lite | Google | Multimodal (lite) | $0.25 / $1.50 |
| Seed 1.6 | ByteDance | Multimodal | $0.25 / $2.00 |
| Grok 4.1 Fast | xAI | Multimodal | $0.20 / $0.50 |
| Seed 1.6 Flash | ByteDance | Multimodal (fast) | $0.08 / $0.30 |
| Nemotron Nano 12B VL | NVIDIA | Vision-language (12B) | $0.20 / $0.60 |
| Qwen3-VL-30B Thinking | Alibaba | Vision-language + CoT (30B) | free |
| Qwen3-VL-235B Thinking | Alibaba | Vision-language + CoT (235B) | free |
| Mistral Small 3.1 24B | Mistral | Instruction-tuned | $0.35 / $0.56 |
| Gemma 3 4B | Google | Small vision-language | $0.04 / $0.08 |
| Gemma 3 12B | Google | Medium vision-language | $0.04 / $0.13 |
| Gemma 3 27B | Google | Large vision-language | $0.03 / $0.11 |

All models were accessed via OpenRouter API using the OpenAI-compatible SDK. The selection spans a 50x cost range, from free promotional-tier models (Qwen3-VL Thinking) to GPT-4.1 ($2.00/1M input tokens).

### 3.3 Evaluation Protocol

**Prompt design.** Each model received a system prompt establishing it as a document quality assessor and a user prompt containing the document image. The prompt requests ratings on a 1.0-5.0 continuous scale with 0.1 granularity across all three dimensions, with brief reasoning. The response format is structured JSON:

```json
{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}
```

**Image preprocessing.** Images were resized to fit within 1024x1024 pixels (preserving aspect ratio) and encoded as base64 JPEG for API transmission. Section 4.3 investigates the impact of this resolution choice.

**Inference parameters.** Temperature = 0.0 for all models except Qwen3-VL-8B Thinking, which uses the provider default (OpenRouter's thinking-model interface does not accept explicit temperature). Maximum tokens = 1,024 (2,048 for thinking models to accommodate chain-of-thought). Exponential backoff retry with 3 attempts.

**Metrics.** We report:

- **SRCC** (Spearman Rank Correlation Coefficient): Measures monotonic agreement between predicted and human rankings.
- **PLCC** (Pearson Linear Correlation Coefficient): Measures linear agreement after 4-parameter logistic curve fitting.
- **MAE** (Mean Absolute Error): Average absolute prediction error in MOS units.
- **RMSE** (Root Mean Square Error): Penalizes large prediction errors.
- **wSRCC** (Weighted SRCC): Competition metric, `0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color`.
- **95% Confidence Intervals**: Bootstrapped with 1,000 iterations (seed=42).

**Reference baseline.** DeQA-Doc-3Specialists uses three mPLUG-Owl2 models (7B parameters each), each fine-tuned on a single quality dimension of the DIQA-5000 training set. On the test set: wSRCC = 0.716, SRCC_overall = 0.733, SRCC_sharpness = 0.681, SRCC_color = 0.716.

---

## 4. Results

### 4.1 Main Benchmark Results

Table 1 presents the primary results on the DIQA-5000 test set. All 21 models received all 1,000 images. Gemini 2.5 Pro had 70 parse failures (7.0%), Nemotron Nano 12B VL had 25 (2.5%), Grok 4.1 Fast had 14 (1.4%), and Qwen3-VL-8B Thinking had 2 (0.2%); metrics for affected models are computed on valid responses only.

**Table 1: DIQA-5000 Full Benchmark Results (n=1,000)**

| Model | MainScore | wSRCC | SRCC_O | SRCC_S | SRCC_C | PLCC_O | MAE_O |
|---|---|---|---|---|---|---|---|
| **Gemini 3 Flash** | **0.743** | **0.708** | 0.707 | **0.736** | **0.681** | **0.784** | 0.80 |
| Qwen 3.5 122B-A10B | 0.729 | 0.713 | 0.704 | 0.746 | 0.699 | 0.748 | 1.40 |
| Gemini 3.1 Flash Lite | 0.722 | 0.694 | 0.702 | 0.722 | 0.650 | 0.766 | 1.09 |
| GPT-4.1 | 0.715 | 0.669 | **0.683** | 0.679 | 0.631 | 0.775 | 1.15 |
| Qwen 3.5 Plus | 0.707 | 0.679 | 0.668 | 0.700 | 0.679 | 0.727 | 1.13 |
| Gemini 2.5 Pro | 0.655 | 0.612 | 0.613 | 0.603 | 0.621 | 0.662 | 0.84 |
| Qwen 3.5 Flash | 0.626 | 0.593 | 0.560 | 0.643 | 0.608 | 0.624 | 1.50 |
| Qwen3-VL-235B Inst. | 0.598 | 0.577 | 0.639 | 0.595 | 0.434 | 0.658 | 1.23 |
| Claude Haiku 4.5 | 0.601 | 0.579 | 0.598 | 0.539 | 0.579 | 0.636 | **0.68** |
| Seed 1.6 | 0.562 | 0.557 | 0.606 | 0.531 | 0.485 | 0.595 | 1.16 |
| Qwen3-VL-30B Think | 0.556 | 0.537 | 0.559 | 0.523 | 0.507 | 0.599 | 1.25 |
| Qwen3-VL-235B Think | 0.541 | 0.534 | 0.526 | 0.512 | 0.573 | 0.535 | 1.15 |
| Mistral Small 3.1 24B | 0.511 | 0.488 | 0.549 | 0.443 | 0.411 | 0.585 | 0.87 |
| Seed 1.6 Flash | 0.500 | 0.483 | 0.508 | 0.499 | 0.417 | 0.534 | 1.05 |
| Qwen3-VL-8B Instruct | 0.505 | 0.481 | 0.520 | 0.437 | 0.446 | 0.563 | 1.31 |
| Gemma 3 27B | 0.431 | 0.396 | 0.438 | 0.444 | 0.265 | 0.521 | 0.63 |
| Qwen3-VL-8B Think | 0.439 | 0.409 | 0.432 | 0.397 | 0.377 | 0.465 | 0.93 |
| Gemma 3 12B | 0.276 | 0.242 | 0.286 | 0.331 | 0.067 | 0.353 | 0.64 |
| Gemma 3 4B | 0.270 | 0.251 | 0.284 | 0.249 | 0.189 | 0.326 | 1.10 |
| Nemotron Nano 12B VL | 0.185 | 0.187 | 0.181 | 0.251 | 0.134 | 0.171 | 0.87 |
| Grok 4.1 Fast | 0.114 | 0.095 | 0.102 | 0.034 | 0.141 | 0.132 | 1.31 |
| *DeQA-Doc-3Spec.* | *0.716†* | *0.716* | *0.733* | *0.681* | *0.716* | *--* | *--* |

*†DeQA-Doc-3Specialists MainScore is wSRCC only (no PLCC reported for the specialist ensemble).*

Gemini 3 Flash Preview leads all VLMs with MainScore = 0.743 (wSRCC = 0.708), approaching the supervised baseline (wSRCC = 0.716). Qwen 3.5 122B-A10B (0.729) and Gemini 3.1 Flash Lite (0.722) are strong contenders from batch 2, both surpassing GPT-4.1 (0.715). Claude Haiku 4.5 achieves the lowest MAE (0.68) despite ranking ninth in MainScore, indicating conservative but poorly ranked predictions.

The 95% bootstrapped confidence intervals for the top models overlap: Gemini 3 Flash overall SRCC = [0.671, 0.742]; Qwen 3.5 122B-A10B and Gemini 3.1 Flash Lite are within this range. However, the top tier (MainScore > 0.70) is clearly separated from the remaining models. Notably, the bottom tier — Grok 4.1 Fast (0.114), Nemotron Nano 12B VL (0.185), and Gemma 3 4B/12B (0.270/0.276) — shows correlations near random, likely due to poor instruction following for structured JSON output.

**Reasoning models underperform.** Both chain-of-thought models -- Qwen3-VL-8B Thinking (wSRCC = 0.409) and Qwen 3.5 Flash with extended reasoning tokens (wSRCC = 0.593) -- rank below their non-reasoning counterparts or same-tier models. A separate temperature=0 validation run on Qwen3-VL-8B Thinking showed a small decrease (wSRCC 0.409 to 0.383), suggesting stochastic sampling slightly helps this model's ranking quality, though both values remain far below the top models.

### 4.2 Per-Dimension Analysis

Performance varies substantially across quality dimensions (Figure 2). Key patterns include:

**Sharpness is the easiest dimension for VLMs.** The majority of models achieve their highest SRCC on sharpness. Qwen 3.5 122B-A10B leads at SRCC = 0.746, followed by Gemini 3 Flash at 0.736 and Gemini 3.1 Flash Lite at 0.722. This likely reflects VLMs' strong ability to assess text legibility -- a capability grounded in their language understanding training.

**Color fidelity is the hardest dimension.** Most models achieve their lowest SRCC on color fidelity. The gap between sharpness and color can be substantial: GPT-4.1 shows SRCC 0.679 for sharpness vs. 0.631 for color (delta = 0.048); Claude Haiku 4.5 shows 0.539 vs. 0.579, where color is actually marginally better. Color fidelity assessment requires understanding domain-specific artifacts (deshadowing color shifts, white balance errors) that may be underrepresented in VLM training data.

**Overall quality tracks a weighted average.** Overall SRCC generally falls between sharpness and color fidelity for each model, consistent with its holistic nature incorporating both sub-dimensions.

### 4.3 Ordinal Discrimination Analysis

We mapped continuous VLM predictions to the nearest quality bucket (bad/poor/fair/good/excellent) and compared against human MOS buckets.

**Table 2: Ordinal Classification Metrics (Overall Quality)**

| Model | Exact Acc. | Adjacent Acc. | Weighted Kappa | Over-rate % |
|---|---|---|---|---|
| Claude Haiku 4.5 | **31.5%** | **77.6%** | 0.340 | 64.0% |
| Gemini 3 Flash | 24.9% | 68.1% | **0.379** | 73.9% |
| GPT-4.1 | 10.2% | 37.4% | 0.246 | 88.6% |
| Qwen3-VL-8B Instruct | 1.2% | 27.7% | 0.087 | 98.3% |

**Systematic over-rating is universal.** Every evaluated model rates documents higher than human annotators. The over-rating percentage ranges from 64.0% (Claude Haiku 4.5, the most conservative model) to 98.3% (Qwen3-VL-8B Instruct, which rates virtually every image as "good" or "excellent"). GPT-4.1 predicts "excellent" for 754 of 1,000 images despite only 5 images having true MOS >= 4.0 -- a 151x over-prediction of the excellent category.

**Adjacent accuracy reveals preserved ordinal structure.** While exact bucket match is poor (1.2%-31.5%), adjacent accuracy (prediction within one bucket of ground truth) ranges from 27.7% to 77.6%. This indicates that models can distinguish "bad from good" even when they cannot calibrate the absolute quality level. The gap between exact and adjacent accuracy (28-47 percentage points) directly quantifies the calibration deficit that a post-hoc correction step could address.

**Confusion matrix pattern (Figure 4).** The confusion matrix for Gemini 3 Flash reveals a consistent pattern shared by all models. The "bad" category shows the highest precision (0.923 for Gemini 3 Flash) -- models correctly identify truly unusable documents. However, the "fair" bucket (n=613, the mode) is the most problematic: of 613 truly "fair" images, Gemini predicts 272 as "excellent," 174 as "good," 160 as "fair," and only 7 as "poor" or "bad." The fair/good/excellent distinction collapses into a single "above average" cluster from the VLM's perspective.

### 4.4 Prompt Engineering Findings

We conducted two prompt engineering experiments to assess whether prompting strategy can improve VLM quality predictions.

**Single vs. separate prompts (n=44).** We compared the default single-prompt approach (all 3 dimension scores in one API call) against separate prompts (one call per dimension) on 44 stratified images using Gemini 3 Flash and GPT-4.1.

| Dimension | Condition | SRCC (Gemini) | SRCC (GPT-4.1) |
|---|---|---|---|
| Overall | 1-prompt | **0.785** | **0.730** |
| Overall | 3-prompt | 0.768 | 0.679 |
| Sharpness | 1-prompt | 0.767 | 0.741 |
| Sharpness | 3-prompt | **0.803** | **0.760** |
| Color Fidelity | 1-prompt | 0.704 | 0.654 |
| Color Fidelity | 3-prompt | **0.719** | **0.691** |

Separate prompts improve sharpness (+0.019 to +0.036 SRCC) and color fidelity (+0.015 to +0.037) at the cost of overall quality correlation (-0.017 to -0.051) and 2-3x latency. The improvement is most pronounced for color fidelity, the dimension where models are weakest, suggesting that dimension-specific rubrics reduce anchoring effects between correlated quality attributes.

**Seven-arm prompt optimization (n=23).** We tested 7 prompting strategies on Gemini 3 Flash using 23 stratified images:

| Rank | Strategy | wSRCC | MAE_O |
|---|---|---|---|
| 1 | No resize (native resolution) | 0.951 | 0.618 |
| 2 | Multi-sample (3x, temp=0.3, median) | 0.928 | 0.640 |
| 3 | Resize to 2048px | 0.925 | 0.655 |
| 4 | Hybrid (overall combined, sub-dims separate) | 0.923 | 0.686 |
| 5 | Separate 3 prompts | 0.911 | 0.803 |
| 6 | Single prompt (baseline) | 0.909 | 0.633 |
| 7 | Few-shot (3 examples) | 0.836 | 0.715 |

The no-resize strategy appeared dominant (+0.042 wSRCC over baseline), and this ranking replicated on Qwen 3.5 Flash (no-resize best at wSRCC = 0.914). However, when validated on all 1,000 images, native resolution yielded wSRCC = 0.699 -- **lower** than the 1024px default (0.708, delta = -0.009). Only color fidelity showed marginal improvement (+0.009 SRCC), while overall (-0.014) and sharpness (-0.018) both degraded. Inference latency also increased by approximately 17%.

This is a cautionary result: the n=23 optimization suggested a +0.042 improvement that became a -0.009 degradation at full scale. The small sample happened to contain images where native resolution helped, creating a spurious signal. Combined with the smoke test instability finding (Section 4.1 reports that a 7-image smoke test overestimated all models by 0.15-0.23 wSRCC), this demonstrates that small-sample VLM benchmarks are unreliable for both absolute scoring and strategy comparison. We recommend a minimum of n=200 stratified samples for any prompt optimization experiment.

### 4.5 Cross-Domain Evaluation

We evaluated all 21 VLMs on the 520-image synthetic OOD dataset (Table 3). Parse failure rates varied: Gemini 2.5 Pro had 95 failures (18.3%), Qwen 3.5 Flash had 69 (13.3%); metrics for affected models are computed on valid responses only.

**Table 3: Synthetic Dataset Results (All 21 VLMs)**

| Model | MainScore | wSRCC | SRCC_O | PLCC_O |
|---|---|---|---|---|
| **Gemini 3 Flash** | **0.768** | 0.738 | 0.753 | 0.804 |
| GPT-4.1 | 0.768 | **0.757** | **0.764** | 0.788 |
| Claude Haiku 4.5 | 0.646 | 0.591 | 0.582 | 0.717 |
| Gemini 3.1 Flash Lite | 0.642 | 0.581 | 0.576 | 0.700 |
| Qwen 3.5 122B-A10B | 0.625 | 0.609 | 0.614 | 0.653 |
| Qwen 3.5 Flash | 0.567 | 0.542 | 0.550 | 0.604 |
| Qwen 3.5 Plus | 0.570 | 0.559 | 0.558 | 0.590 |
| Seed 1.6 Flash | 0.489 | 0.437 | 0.449 | 0.556 |
| Gemini 2.5 Pro | 0.477 | 0.466 | 0.469 | 0.548 |
| Mistral Small 3.1 24B | 0.476 | 0.421 | 0.453 | 0.539 |
| Gemma 3 12B | 0.459 | 0.402 | 0.432 | 0.539 |
| Qwen3-VL-8B Think | 0.450 | 0.428 | 0.430 | 0.490 |
| Qwen3-VL-8B Instruct | 0.449 | 0.388 | 0.413 | 0.544 |
| Gemma 3 27B | 0.440 | 0.382 | 0.401 | 0.503 |
| Qwen3-VL-30B Think | 0.408 | 0.336 | 0.358 | 0.476 |
| Seed 1.6 | 0.372 | 0.251 | 0.263 | 0.456 |
| Gemma 3 4B | 0.323 | 0.268 | 0.286 | 0.397 |
| Qwen3-VL-235B Inst. | 0.294 | 0.229 | 0.234 | 0.349 |
| Qwen3-VL-235B Think | 0.232 | 0.185 | 0.189 | 0.269 |
| Nemotron Nano 12B VL | 0.207 | 0.183 | 0.193 | 0.231 |
| Grok 4.1 Fast | 0.134 | 0.149 | 0.135 | 0.139 |

**Table 4: Per-Category Overall SRCC (Top 2 Models)**

| Category | n | Gemini 3 Flash | GPT-4.1 |
|---|---|---|---|
| In-distribution (Cyrillic) | 50 | **0.808** | 0.758 |
| Non-Latin (Tibetan) | 30 | **0.800** | 0.730 |
| Non-Latin (Ethiopic) | 30 | 0.767 | **0.797** |
| Adversarial (Nastaliq) | 20 | 0.770 | **0.846** |
| In-distribution (standard) | 100 | **0.790** | 0.785 |
| Non-Latin (Myanmar) | 30 | 0.763 | 0.764 |
| Adversarial (Fraktur) | 20 | **0.768** | 0.762 |
| CJK vertical layout | 30 | 0.624 | **0.747** |
| Multiscript | 30 | 0.659 | **0.756** |
| Heavily degraded | 30 | 0.236 | 0.174 |
| Form layouts | 30 | 0.201 | 0.169 |
| Pristine | 30 | 0.032 | -0.086 |
| Very high DPI | 30 | -0.150 | -0.109 |
| Binarized | 30 | -0.340 | -0.372 |
| Very low DPI | 30 | -0.216 | -0.411 |

The category-level results reveal three performance tiers:

**Tier 1 -- Strong transfer (SRCC 0.7-0.85).** Non-Latin scripts, adversarial scripts, standard documents, and Cyrillic all show SRCC above 0.7 for at least one of the top two models. VLMs assess visual quality independently of reading comprehension -- they do not need to understand Tibetan or Ethiopic script to judge whether text edges are sharp and contrast is adequate.

**Tier 2 -- Moderate transfer (SRCC 0.15-0.25).** Heavily degraded and form layouts show positive but weak correlation. These categories present quality variation that VLMs can partially detect but struggle to rank precisely.

**Tier 3 -- Failure (SRCC < 0.05 or negative).** Binarized documents (SRCC = -0.34/-0.37), extreme DPI variants (negative SRCC), and pristine digital originals (near-zero SRCC) are universal failure modes for all evaluated VLMs. Binarized documents present a quality dimension (ink density, edge smoothness) that VLMs misinterpret. Pristine documents have near-zero quality variance, making rank correlation unstable. Extreme DPI creates artifacts (pixelation at low DPI, file size inflation at high DPI) that VLMs do not map to perceptual quality.

**Complementary strengths.** Gemini 3 Flash leads on in-distribution and Cyrillic documents, while GPT-4.1 leads on adversarial scripts, CJK vertical, and multiscript categories. This complementary pattern motivates a consensus approach (discussed in Section 5.2).

### 4.6 Error Analysis and Failure Cases

We identify three systematic failure patterns:

**Positive bias with compressed dynamic range.** VLMs use approximately the 3.0-5.0 range for images that humans rate 1.0-5.0. Figure 5 shows that mean bias (predicted minus ground truth) is positive for every model and every dimension. GPT-4.1 exhibits the most severe bias: overall +1.15, sharpness +1.08, color fidelity +1.31 MOS points. Claude Haiku 4.5 shows the smallest bias: overall +0.55, sharpness +0.38, color fidelity +0.78. We hypothesize this stems from VLMs being trained on general web data where "most images are fine" -- a prior that conflicts with DIQA-5000's intentionally degraded document set.

**Collapsed upper buckets.** The ordinal analysis (Section 4.3) shows that fair/good/excellent distinctions collapse. GPT-4.1 assigns 75.4% of all predictions to the "excellent" bucket. Even Gemini 3 Flash, the best calibrated model, assigns 42.1% to "excellent" (true proportion: 0.5%). The discriminative failure concentrates in the upper range -- models can distinguish "bad" from "not bad" but cannot distinguish "fair" from "good" from "excellent."

**Category-specific failures on synthetic data.** The binarized document failure (negative SRCC) is particularly notable because binarized documents are common in real-world document processing. VLMs appear to treat binarization as a quality defect rather than a deliberate processing step, assigning low scores to clean binarized documents. This represents a fundamental conceptual gap between VLM quality assessment (which expects natural-looking images) and document processing quality (which values OCR-optimized preprocessing).

---

## 5. Discussion

### 5.1 Key Insights

**VLMs approach supervised baselines without training data.** The best VLM (Gemini 3 Flash, MainScore = 0.743) surpasses the supervised baseline (DeQA-Doc-3Specialists, wSRCC = 0.716) when compared on the unified MainScore metric, which incorporates PLCC. Five VLMs achieve MainScore > 0.70 (Gemini 3 Flash, Qwen 3.5 122B-A10B, Gemini 3.1 Flash Lite, GPT-4.1, Qwen 3.5 Plus). This validates the core hypothesis: frontier VLMs possess sufficient document understanding to serve as quality annotators.

**Correlation matters more than calibration for pseudo-labeling.** The disconnect between correlation (ranking accuracy) and calibration (absolute score accuracy) is a recurring theme. Claude Haiku 4.5 has the lowest MAE (0.68) but only the fifth-highest SRCC. GPT-4.1 has the second-highest SRCC but the worst MAE (1.15). For pseudo-labeling applications, correlation is the primary concern: the DeQA-Score training pipeline converts MOS to soft-label distributions whose shape depends on rank order more than absolute values. A model that correctly ranks all images but assigns them all scores in [3.5, 5.0] can be calibrated post-hoc; a model that assigns random scores within the correct range cannot be recovered.

**Small-sample evaluation is unreliable.** This finding recurs at multiple scales: a 7-image smoke test overestimated models by 0.15-0.23 wSRCC; a 23-image prompt optimization suggested +0.042 from native resolution that became -0.009 at full scale; a 44-image A/B test showed dimension-specific effects that may not generalize. The minimum reliable sample size for VLM quality benchmarking appears to be n=200-500 stratified images. This has practical implications: VLM prompt engineering cannot be cheaply validated on small pilot sets.

**Reasoning does not help quality assessment.** Chain-of-thought reasoning, expected to improve nuanced judgments, actually degrades performance. This pattern holds across all model scales: Qwen3-VL-8B Thinking (MainScore = 0.439) underperforms Qwen3-VL-8B Instruct (0.505); Qwen3-VL-30B Thinking (0.556) and Qwen3-VL-235B Thinking (0.541) both underperform Qwen3-VL-235B Instruct (0.598). Document quality assessment may be a sufficiently "fast thinking" task that extended deliberation introduces noise rather than signal.

### 5.2 Practical Implications

**Dual-model consensus improves correlation.** The complementary failure modes between Gemini 3 Flash and GPT-4.1 (Section 4.5) motivate a consensus approach for pseudo-labeling. Empirical evaluation on the DIQA-5000 test set confirms this: a simple pairwise mean of Gemini 3 Flash + GPT-4.1 achieves wSRCC = 0.744 [95% CI: 0.714-0.769], exceeding either model alone (Gemini: 0.708, GPT-4.1: 0.669) by +0.036 to +0.075. The improvement is consistent across all three dimensions (SRCC overall: 0.745, sharpness: 0.769, color fidelity: 0.716). Extending to the original 7 primary models with wSRCC-proportional weighting yields wSRCC = 0.755, a +0.047 gain over the best single model. On synthetic OOD data (n=520), the pairwise consensus achieves wSRCC = 0.778, exceeding both Gemini (0.738) and GPT-4.1 (0.757). Mean consistently outperforms median aggregation (+0.01-0.02 wSRCC). Using both models costs approximately $0.003 per image via OpenRouter -- orders of magnitude below human annotation. Images where the two models disagree by more than 1.0 MOS can be routed to a third model or flagged for manual review.

**Calibration is mandatory.** Raw VLM scores cannot be used directly as training labels -- all 21 models exhibit positive bias ranging from +0.57 MOS (Claude Haiku 4.5) to +1.50 MOS (Qwen 3.5 Flash). GPT-4.1 over-rates by +1.13 overall and +1.22 on sharpness. A calibration step is essential to map VLM predictions to the human MOS scale. Five-fold cross-validated linear calibration on the 1,000-image test set reduces MAE by 2-4x (e.g., GPT-4.1 overall: 1.15 → 0.28; Qwen 3.5 Flash: 1.50 → 0.35) while preserving or slightly improving wSRCC. Notably, calibration benefits ensembles more than single models: the calibrated All-7 weighted ensemble achieves the lowest MAE (0.28 overall) alongside the highest wSRCC (0.760). Full calibration using the 3,500 DIQA-5000 training images is planned as the first step of the pseudo-labeling pipeline (Paper 7).

**OOD gating is necessary.** The universal VLM failure on binarized documents, extreme DPI, and pristine originals means pseudo-labels for these categories are worse than random. An automated quality gate -- such as the Mahalanobis-distance OOD detector described in Paper 4 (AUROC = 0.9963) -- must identify and exclude these categories before VLM annotation proceeds.

**Cost-performance tradeoff.** Gemini 3 Flash provides the best accuracy at $0.50/1M input tokens. Gemini 3.1 Flash Lite ($0.25/1M) achieves 97% of Gemini Flash's MainScore at half the cost, making it the best value option. GPT-4.1 at $2.00/1M input tokens offers slightly lower MainScore (0.715 vs. 0.743) with complementary category strengths. The bottom tier (Grok 4.1 Fast, Nemotron Nano 12B VL, Gemma 3 4B/12B) shows correlations near random despite moderate pricing, highlighting that model capability matters more than cost for this task.

### 5.3 Limitations and Threats to Validity

**Single dataset.** All primary evaluation uses the DIQA-5000 test set. Results may not generalize to other document IQA datasets (Tobacco800, RVL-CDIP) or real-world document collections with different quality distributions.

**Synthetic OOD only.** The 520-image cross-domain evaluation uses programmatically generated images. Real-world OOD documents (handwritten forms, historical manuscripts, receipts, medical records) may present different quality patterns that VLMs handle better or worse.

**API-mediated evaluation.** All VLMs were accessed via OpenRouter, introducing routing latency and potential response variability from load balancing across provider endpoints. Direct API access might yield marginally different results. However, this reflects realistic deployment conditions where VLMs are typically accessed via API.

**Prompt sensitivity.** We evaluated a fixed prompt design (system + user prompt requesting structured JSON). Alternative prompt structures -- chain-of-thought, comparison-based rating, reference anchoring -- might improve specific models. The prompt optimization experiment (Section 4.4) suggests the effect is small relative to model choice, but we cannot rule out that an undiscovered prompt strategy substantially changes the ranking.

**Temporal instability.** VLM providers frequently update model weights. Our results reflect a specific point in time (March 2026); future model versions may perform differently. The raw checkpoint data enables re-evaluation as models change.

**Missing variance estimation.** We evaluate VLMs on point-estimate accuracy but do not assess their ability to estimate annotation variance (the spread of human ratings). For the pseudo-labeling pipeline, both the mean and variance are needed to construct soft-label distributions. Methods for deriving variance from VLM outputs (multi-sample standard deviation, prompt-based confidence elicitation) remain future work.

---

## 6. Conclusion & Future Work

We benchmarked 21 frontier VLMs on 1,000 DIQA-5000 test images, finding that Gemini 3 Flash Preview (MainScore = 0.743, wSRCC = 0.708) leads the VLM ranking, approaching the supervised DeQA-Doc-3Specialists baseline (wSRCC = 0.716) without any DIQA-specific training. Qwen 3.5 122B-A10B (0.729) and Gemini 3.1 Flash Lite (0.722) closely follow. All VLMs exhibit systematic positive bias that compresses the predicted score range, but the rank ordering of documents is well-preserved, making these models viable as pseudo-label generators after calibration.

The most methodologically significant finding is that small-sample prompt optimization is unreliable: effects observed at n=23 did not replicate at n=1,000. This has broad implications for VLM evaluation methodology beyond document quality assessment.

Cross-domain evaluation confirms that VLMs transfer well to non-Latin scripts (SRCC 0.73-0.85) but fail on specific categories (binarized, extreme DPI, pristine), motivating automated OOD gating in any pseudo-labeling pipeline.

**Future work directly building on these findings:**

1. **End-to-end pseudo-labeling validation.** Train a student model on VLM pseudo-labels for OOD documents and measure whether SRCC > 0.90 is maintained on DIQA-5000 while improving on OOD categories.
2. **Calibration experiments on VLM predictions.** Run VLM inference on 3,500 training images to learn per-model calibration functions, and evaluate the effect on downstream student model quality.
3. **Ensemble optimization.** Systematic search over model combinations and weighting schemes for consensus scoring, exploiting the complementary strengths identified in Section 4.5.
4. **Variance estimation from VLMs.** Evaluate multi-sample strategies (running each model 3-5 times with temperature > 0) for estimating annotation uncertainty, needed to construct soft-label distributions.
5. **Larger-scale model comparison.** Extend the benchmark to newer models as they become available, with a minimum sample size of n=200 per evaluation.

---

## 7. Reproducibility, Data & Governance

### Reproducibility

All experiments are fully reproducible from archived data:

- **Evaluation scripts**: `results/vlm_teacher_eval/full_eval/run_full_diqa_eval.py` (main benchmark), `analyze_ordinal.py` (ordinal analysis), `run_prompt_optimization.py` (7-arm optimization).
- **Figure generation**: `research/papers/01_vlm_benchmark/figures/generate_figures.py` produces all figures from raw checkpoint data.
- **Environment**: Python 3.10+, dependencies managed via `uv` with lockfile.
- **Random seeds**: All bootstrap confidence intervals use seed=42 with 1,000 iterations.

### Data Availability

Per-sample JSONL checkpoints for all 21 models (1,000 DIQA images each) are archived at `results/vlm_teacher_eval/full_eval/checkpoints/`. Synthetic OOD checkpoints (21 models x 520 images) are at `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/`. Each record contains: model_id, image filename, overall/sharpness/color_fidelity scores, free-text reasoning, raw API response, latency in milliseconds, and any error messages.

**Total: ~33,800 VLM per-sample evaluations** comprising 21,000 primary DIQA benchmark records, 10,920 synthetic OOD records, 2,000 validation records (native resolution and temperature=0 runs), 88 A/B test records, and 161 prompt optimization records.

Ground truth: `results/vlm_teacher_eval/full_eval/data/test.csv` (1,000 records with human MOS for 3 dimensions).

Aggregated results: `results/vlm_teacher_eval/full_eval/results/ordinal_analysis.json`, `vlm_benchmark_results.csv`, `synthetic_eval_metrics.json`.

### Data Governance

- **DIQA-5000 ground truth** is used under the terms of the VQualA 2025 Challenge. No individual annotator data is exposed; only aggregate MOS values are used.
- **VLM API responses** contain model-generated text only. No personally identifiable information is present in prompts or responses.
- **Synthetic OOD data** is programmatically generated and contains no real documents or copyrighted content.
- **Cost transparency**: Total VLM API cost for all experiments was approximately $350 via OpenRouter.

---

## Acknowledgments

The DIQA-5000 dataset was created by Zhichao Ma, Fan Huang, Lu Zhao, Xiaohong Liu, Xiongkuo Min, and Guangtao Zhai at Shanghai Jiao Tong University. The VQualA 2025 DIQA Challenge was organized by Fan Huang, Xiongkuo Min, Zhichao Ma, Xiaohong Liu, Chris Wei Zhou, and Guangtao Zhai, with sponsorship from INTSIG Information Co. Ltd. The DeQA-Score framework was developed by Zhiyuan You et al. VLM inference was conducted via OpenRouter's unified API.

---

## References

[1] J. Gao et al., "DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment," arXiv:2507.12796, 2025.

[2] F. Huang et al., "VQualA 2025: Document Image Quality Assessment Challenge Overview," ICCVW 2025, pp. 3313-3320.

[3] Z. Ma et al., "DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment," arXiv:2509.17012, 2025.

[4] ITU-R, "Recommendation BT.500-14: Methodology for the Subjective Assessment of the Quality of Television Pictures," International Telecommunication Union, 2019.

[5] Z. You et al., "DeQA-Score: Soft-Label Distribution Learning for Quality Assessment," 2024.

[6] F. Huang et al., "Q-Doc: Benchmarking Document Image Quality Assessment Capabilities in Multi-modal Large Language Models," PRCV 2025/2026, Springer.

[7] Z. You et al., "Descriptive Image Quality Assessment in the Wild," arXiv:2405.18842, 2024.

---

## Appendix

### A. Complete Per-Dimension Metrics

**Table A1: Full Benchmark Metrics with 95% Bootstrap CIs**

| Model | Dim | SRCC | SRCC CI | PLCC | MAE | RMSE | Bias |
|---|---|---|---|---|---|---|---|
| Gemini 3 Flash | Overall | 0.707 | [0.671, 0.742] | 0.784 | 0.80 | 0.97 | +0.68 |
| Gemini 3 Flash | Sharpness | 0.736 | [0.703, 0.766] | 0.780 | 0.87 | 1.04 | +0.75 |
| Gemini 3 Flash | Color | 0.681 | [0.644, 0.718] | 0.762 | 0.96 | 1.13 | +0.85 |
| GPT-4.1 | Overall | 0.683 | [0.644, 0.719] | 0.775 | 1.15 | 1.32 | +1.15 |
| GPT-4.1 | Sharpness | 0.679 | [0.639, 0.715] | 0.741 | 1.22 | 1.39 | +1.08 |
| GPT-4.1 | Color | 0.631 | [0.588, 0.670] | 0.739 | 1.37 | 1.51 | +1.31 |
| Gemini 2.5 Pro | Overall | 0.613 | [0.568, 0.655] | 0.662 | 0.84 | 1.06 | +0.48 |
| Gemini 2.5 Pro | Sharpness | 0.603 | [0.555, 0.648] | 0.637 | 0.84 | 1.06 | +0.47 |
| Gemini 2.5 Pro | Color | 0.621 | [0.571, 0.668] | 0.679 | 0.89 | 1.10 | +0.57 |
| Qwen 3.5 Flash | Overall | 0.560 | [0.509, 0.607] | 0.624 | 1.50 | 1.67 | +1.44 |
| Qwen 3.5 Flash | Sharpness | 0.643 | [0.601, 0.683] | 0.690 | 1.32 | 1.52 | +1.22 |
| Qwen 3.5 Flash | Color | 0.608 | [0.561, 0.651] | 0.671 | 1.33 | 1.52 | +1.24 |
| Claude Haiku 4.5 | Overall | 0.598 | [0.554, 0.640] | 0.636 | 0.68 | 0.89 | +0.55 |
| Claude Haiku 4.5 | Sharpness | 0.539 | [0.491, 0.584] | 0.577 | 0.67 | 0.89 | +0.38 |
| Claude Haiku 4.5 | Color | 0.579 | [0.534, 0.621] | 0.642 | 0.86 | 1.06 | +0.78 |
| Qwen3-VL-8B | Overall | 0.520 | [0.471, 0.567] | 0.563 | 1.31 | 1.48 | +1.26 |
| Qwen3-VL-8B | Sharpness | 0.437 | [0.381, 0.491] | 0.472 | 1.30 | 1.51 | +1.20 |
| Qwen3-VL-8B | Color | 0.446 | [0.392, 0.498] | 0.508 | 1.52 | 1.68 | +1.47 |
| Qwen3-VL-8B Think | Overall | 0.400 | [0.343, 0.453] | 0.448 | 0.93 | 1.09 | +0.66 |
| Qwen3-VL-8B Think | Sharpness | 0.343 | [0.284, 0.398] | 0.377 | 0.85 | 1.07 | +0.48 |
| Qwen3-VL-8B Think | Color | 0.351 | [0.296, 0.404] | 0.382 | 1.05 | 1.22 | +0.85 |

### B. Landscape Comparison

To contextualize VLM teacher performance, Table B1 positions our results within the broader DIQA model landscape.

**Table B1: DIQA Model Landscape (Representative Models)**

| Family | Representative | Best wSRCC/MainScore | Notes |
|---|---|---|---|
| Competition MLLMs (fine-tuned) | DeQA-Doc ensemble | 0.929 | Trained on DIQA-5000, 5-fold CV |
| Fine-tuned specialists | DeQA-Doc-3Specialists | 0.716 | Single-dim specialists, no ensemble |
| **VLM teachers (this work)** | **Gemini 3 Flash** | **0.708** | **Zero-shot, API-based** |
| Fine-tuned student | SigLIP2-IQA-Base-86M | 0.886 (MainScore) | Trained on pseudo-labels |
| General IQA (zero-shot) | RichIQA (off-the-shelf) | 0.490 | Pretrained on KonIQ-10K |
| General IQA (worst) | MUSIQ (off-the-shelf) | 0.185 | Pretrained on KonIQ-10K |

The VLM teachers fill a critical gap: they substantially outperform off-the-shelf NR-IQA models (0.708 vs. 0.490) and approach the supervised baseline (0.716) without any DIQA-specific training, making them viable as zero-shot pseudo-label generators for domains where labeled data is unavailable.
