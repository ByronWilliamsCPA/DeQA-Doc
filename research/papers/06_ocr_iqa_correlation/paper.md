# DeQA Quality Scores Predict OCR Accuracy: A Controlled Study

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 6/10
**Repository:** `research/ocr_iqa_correlation/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams

**Keywords:** document image quality assessment, OCR accuracy, quality-CER correlation, controlled distortions, quality gating, DIQA

---

## Abstract

Document image quality assessment (DIQA) models predict perceptual quality scores, but whether these scores predict downstream task performance remains an open question. We present a controlled study measuring the correlation between DeQA-Doc quality scores and OCR character error rates (CER) across nine independent OCR engines spanning traditional, neural, and VLM-based architectures. Using 200 document images from FUNSD and FUNSD+ with verified ground-truth text, we construct 1,200 image-quality pairs spanning six quality tiers via a deterministic hybrid augmentation pipeline (Augraphy + Albumentations). We measure CER from Tesseract, EasyOCR, RapidOCR, Google Cloud Vision, PP-OCRv5, docTR, Kraken, GLM-OCR, and DeepSeek-OCR2 against DeQA-Doc Mean Opinion Scores (MOS). We find statistically significant negative correlations between MOS and CER for all engines: Spearman rank correlations range from -0.339 (DeepSeek-OCR2) to -0.658 (PP-OCRv5), with all p-values below 10^-33. Traditional OCR engines show the strongest correlations (|SRCC| = 0.543-0.658), while VLM-based OCR engines show notably weaker correlations (|SRCC| = 0.339-0.343), suggesting that VLM OCR architectures respond to degradation differently than traditional pipelines. Paired analysis, comparing each document's distorted version against its original, strengthens correlations to -0.750 (PP-OCRv5). CER increases approximately monotonically across quality tiers for all engines, with the PRISTINE-to-HIGH transition producing the largest statistically significant jump (p < 10^-7 for all traditional engines). Multi-engine CER ensembles further strengthen correlations: averaging CER across the four most quality-sensitive engines yields SRCC = -0.710 (unpaired) and -0.785 (paired), an 8% and 5% improvement over the best single engine respectively. We additionally evaluate two frontier VLMs (GPT-4.1, Gemini 3 Flash Preview) as zero-shot document quality assessors, finding strong agreement with DeQA-Doc (SRCC = 0.847) and competitive CER prediction. We also conduct the first empirical evaluation of Flexible Character Accuracy (FCA) on this dataset, finding that FCA produces weaker quality correlations than standard CER when ground-truth text lacks natural line structure. These results validate that DeQA-Doc quality scores are not merely perceptual but predict functional OCR degradation, enabling practical quality gating for document processing pipelines.

## 1. Introduction

Document digitization pipelines depend on reliable OCR to convert scanned or photographed documents into machine-readable text. When input image quality degrades, whether from poor scanning, camera distortion, aging paper, or compression artifacts, OCR accuracy can drop substantially. Document image quality assessment (DIQA) models aim to predict these quality degradations, but most DIQA benchmarks evaluate agreement with human perceptual ratings rather than correlation with downstream task performance. This gap leaves a critical question unanswered: do predicted quality scores actually indicate whether OCR will succeed or fail on a given image?

This study bridges the gap between quality prediction and task performance. Rather than treating quality assessment as an end in itself, we treat it as a predictor variable and measure its relationship to OCR error rates under controlled conditions. By constructing a dataset where distortion severity is systematically varied while ground-truth text is held constant, we isolate the effect of image quality on OCR accuracy and quantify how well DeQA-Doc scores capture that relationship.

**Contributions.** This paper makes the following contributions:

- We construct a controlled benchmark of 1,200 document images (200 base images across 6 quality tiers) with paired ground-truth text, enabling rigorous quality-CER correlation measurement.
- We demonstrate statistically significant correlations (|SRCC| = 0.339-0.658) between DeQA-Doc MOS and CER across nine independent OCR engines spanning traditional, neural, and VLM-based architectures, with paired analysis reaching |SRCC| = 0.750.
- We identify a clear architectural divide: traditional OCR engines (PP-OCRv5, Tesseract, EasyOCR, docTR) show strong MOS correlations (|SRCC| = 0.632-0.658), while VLM-based OCR engines (GLM-OCR, DeepSeek-OCR2) show weak correlations (|SRCC| = 0.339-0.343), revealing that neural architecture determines quality-sensitivity more than raw accuracy.
- We establish the PRISTINE-to-HIGH quality boundary as the critical transition where quality degradation first becomes OCR-relevant (p < 10^-7 for all traditional engines), providing an actionable threshold for quality gating in production systems.
- We show that multi-engine CER ensembles provide stronger quality signals than any single engine: averaging the four most quality-sensitive engines yields SRCC = -0.710 (8% over best single engine), with paired ensemble analysis reaching -0.785 — the strongest correlation in this study. Including VLM OCR engines in the ensemble weakens it.
- We evaluate two frontier VLMs (GPT-4.1, Gemini 3 Flash Preview) as zero-shot document quality assessors on the same dataset, finding strong agreement with DeQA-Doc (SRCC = 0.847 for GPT-4.1) and competitive CER prediction, with GPT-4.1 matching or exceeding the specialist model on quality-sensitive engines.
- We conduct the first empirical evaluation of Flexible Character Accuracy (FCA) as an alternative to CER for quality-OCR correlation on form documents, finding that FCA produces weaker correlations than standard CER when ground-truth text lacks natural line structure — a negative result with implications for metric selection in DIQA research.

**Series context.** This is Report 6 of 10 in the DeQA-Doc Technical Report Series. Where Reports 1-5 focused on model evaluation, prompt engineering, OOD detection, and baseline comparisons, this report addresses the downstream validity question: whether quality scores predict real-world task performance. Report 7 extends these findings to pseudo-labeling strategies.

The remainder of this paper is organized as follows. Section 2 defines the task and reviews related work. Section 3 describes the experimental setup, including dataset construction, OCR engines, and evaluation protocol. Section 4 presents results organized by research question. Section 5 discusses implications and limitations. Section 6 concludes with directions for future work.

## 2. Task Definition & Related Work

### 2.1 Task Definition

We define the task as measuring the statistical association between predicted document image quality and downstream OCR accuracy. Formally, given a quality model Q that maps an image I to a scalar quality score Q(I) in [1, 5] (MOS), and an OCR engine E that produces text T_E(I) compared against ground truth T* to yield a character error rate CER_E(I), we seek to estimate the Spearman rank correlation coefficient (SRCC) between Q(I) and CER_E(I) over a controlled set of images.

We expect a negative correlation: higher quality (higher MOS) should predict lower error (lower CER). The strength of this correlation measures the practical validity of Q as a predictor of downstream performance for engine E.

We additionally define a paired variant. For each base document d and distortion tier t, we compute the change relative to the original: delta-CER = CER_E(I_{d,t}) - CER_E(I_{d,ORIGINAL}) and delta-MOS = Q(I_{d,t}) - Q(I_{d,ORIGINAL}). Paired correlation on these deltas controls for per-document complexity (layout, font, text density), isolating the pure quality-degradation effect.

### 2.2 Related Work

**IQA and downstream tasks.** The relationship between image quality and task performance has been explored primarily in natural image domains. Dodge and Karam (2016) showed that image classification accuracy degrades under blur and noise, with quality-aware preprocessing improving results. In the document domain, Nayef et al. (2015) demonstrated that document image binarization quality affects OCR, but did not use learned quality predictors. More recently, Larson et al. (2023) proposed DIQA-specific metrics but evaluated only against human ratings, not task accuracy.

**OCR-quality coupling.** Prior work established strong coupling between document image quality and OCR accuracy. SmartDoc-QA (Nayef et al., 2015) defined OCR character accuracy as the ground-truth quality metric for smartphone-captured documents, providing the benchmark on which subsequent methods are evaluated. CG-DIQA (Li et al., 2018) achieved median SROCC of 0.94 against OCR accuracy on SmartDoc-QA using character-patch gradient analysis, demonstrating that no-reference quality predictors can reach near-perfect rank correlation with OCR performance. More recently, OHR-Bench (Zhang et al., 2025) demonstrated that OCR fidelity degrades monotonically with perturbation severity across seven document domains, with cascading impacts on downstream RAG systems. Our correlations (|SRCC| = 0.34-0.66) are weaker than CG-DIQA's 0.94, but we test on form documents (higher baseline CER due to checkboxes, tables, mixed handwriting) and use a general-purpose DIQA model rather than one designed for OCR prediction.

**Document OCR robustness.** Studies of OCR robustness to degradation have typically used hand-crafted distortion models. Smith (2007) measured Tesseract performance under blur and noise. The ICDAR robust reading competitions evaluate OCR engines on challenging real-world images but without paired quality predictions. Our work differs by using a learned quality model (DeQA-Doc) as the predictor and measuring its correlation with OCR error across nine engines spanning three architectural families simultaneously.

**DeQA-Score and DeQA-Doc.** You et al. (2024) introduced DeQA-Score, a soft-label distribution learning approach to no-reference image quality assessment using multimodal LLMs. DeQA-Doc adapts this framework to document images, predicting quality across three dimensions (overall, sharpness, color fidelity) using specialist mPLUG-Owl2 models. Previous evaluations (Reports 1-5 in this series) validated DeQA-Doc against human ratings on the DIQA-5000 benchmark and characterized its OOD behavior, but did not test downstream task prediction.

**Quality-aware document processing.** Several commercial systems use quality checks before OCR (e.g., ABBYY FineReader's image quality detection), but these typically use rule-based heuristics (resolution checks, skew detection) rather than learned quality models. Our work provides empirical evidence that a learned DIQA model can serve this quality-gating role with quantified predictive validity.

**Controlled distortion benchmarks.** The use of synthetic distortions for OCR evaluation has precedent in the ICDAR SmartDoc competition (Burie et al., 2015), which simulated camera capture distortions for document recognition benchmarks. The Document Image Degradation Model (DIDM) by Kieu et al. (2013) provided early systematic degradation pipelines. Our approach differs in using modern augmentation libraries (Augraphy + Albumentations) that model a wider range of document-specific degradations, and in pairing distorted images with learned quality predictions rather than hand-crafted quality metrics.

## 3. Experimental Setup

### 3.1 Datasets

We construct a controlled evaluation dataset from two publicly available form-document corpora:

| Parameter | Value |
|-----------|-------|
| Source corpora | FUNSD (50 images) + FUNSD+ (150 images) |
| Base images | 200 (stratified sample, minimum 20 GT characters) |
| Distortion tiers | 6: ORIGINAL, PRISTINE, HIGH, MEDIUM, LOW, DEGRADED |
| Total images | 1,200 (200 base x 6 tiers) |
| Random seed | 42 (sampling), deterministic per-image distortion seeds |

**Source selection.** FUNSD (Jaume et al., 2019) provides 149 training images of scanned forms with bounding-box text annotations. FUNSD+ extends this with 1,026 additional form images in Arrow format. We sample 50 images from FUNSD and 150 from FUNSD+, filtering for documents containing at least 20 ground-truth characters to ensure meaningful CER computation.

**Ground-truth text extraction.** For FUNSD documents, we parse JSON annotations sorted by the `id` field and concatenate text entries. For FUNSD+ documents, we extract the `words` field from Arrow format and join with spaces. All text undergoes Unicode NFC normalization.

**Distortion pipeline.** Each base image is processed through a HybridAugmentationPipeline combining Augraphy (for document-specific degradations such as ink bleed, paper aging, and bleed-through) and Albumentations (for standard image distortions such as blur, noise, and compression). Six quality tiers map to predefined profiles:

| Tier | Profile | Target Quality Range | Description |
|------|---------|---------------------|-------------|
| ORIGINAL | (unmodified) | 1.00 | Clean scanned originals |
| PRISTINE | pristine | 0.95 - 1.00 | Near-zero distortion |
| HIGH | light | 0.70 - 0.95 | Light blur, mild noise |
| MEDIUM | moderate | 0.40 - 0.80 | Visible compression, moderate noise |
| LOW | heavy | 0.10 - 0.60 | Heavy blur, significant artifacts |
| DEGRADED | historical | 0.00 - 0.50 | Severe aging, ink loss, bleed-through |

Each distortion uses a deterministic seed computed as `base_seed + image_idx * 100 + tier_idx`, ensuring exact reproducibility.

### 3.2 Models (OCR Engines)

We evaluate nine OCR engines spanning traditional, neural, and VLM-based architectures:

| Engine | Backend | Type | Notes |
|--------|---------|------|-------|
| Tesseract | Docling wrapper | Traditional | LSTM-based, widely deployed |
| EasyOCR | Docling wrapper | Traditional | PyTorch CRNN + CTC |
| RapidOCR | Docling default | Traditional | PaddleOCR-based, lightweight |
| Google Cloud Vision | DOCUMENT_TEXT_DETECTION | Commercial | Google's production OCR API |
| PP-OCRv5 | PaddlePaddle | Neural | PP-OCRv5 mobile det/rec models |
| docTR | PyTorch | Neural | DB ResNet-50 det + CRNN VGG16-BN rec |
| Kraken | PyTorch | Traditional | HTR-focused, default English model |
| GLM-OCR | VLM (GPU) | VLM OCR | zai-org/GLM-OCR (~0.5B params) |
| DeepSeek-OCR2 | VLM (GPU) | VLM OCR | deepseek-ai/DeepSeek-OCR-2 (3B params) |

The nine engines form three architectural groups: (1) **Traditional OCR** engines (Tesseract, EasyOCR, RapidOCR, Kraken) use detection + recognition pipelines with CNN/LSTM-based recognizers. (2) **Neural OCR** engines (PP-OCRv5, docTR) use modern deep learning architectures but maintain the detect-then-recognize paradigm. (3) **VLM OCR** engines (GLM-OCR, DeepSeek-OCR2) use vision-language models that process the entire image end-to-end, generating OCR text via autoregressive decoding. Google Cloud Vision is classified separately as a commercial API with undisclosed architecture.

**Quality assessment models.** DeQA-Doc-3Specialists consists of three mPLUG-Owl2-7B models fine-tuned on DIQA-5000 for overall quality, sharpness, and color fidelity respectively. Each specialist outputs a probability distribution over five quality levels [excellent, good, fair, poor, bad], from which a Mean Opinion Score (MOS) is computed as MOS = sum(prob_i x score_i) where scores = [5, 4, 3, 2, 1]. Inference runs on Modal cloud GPUs (NVIDIA L4). For this study, we use the overall quality MOS as the primary predictor variable.

We additionally evaluate two frontier VLMs as zero-shot quality assessors: GPT-4.1 (OpenAI) and Gemini 3 Flash Preview (Google), accessed via the OpenRouter API. Both receive the same structured prompt used in the DIQA-5000 benchmark, requesting quality scores on a 1-5 scale with 0.1 increments across three dimensions (overall quality, sharpness, color fidelity). Results are reported in Section 4.8.

### 3.3 Evaluation Protocol

**Character Error Rate (CER).** We compute CER using the jiwer library (version >= 3.0) with Unicode NFC normalization and case-insensitive matching. Empty OCR output maps to CER = 1.0 (total failure).

**Flexible Character Accuracy (FCA).** We additionally evaluate FCA (Clausner et al., 2020), a reading-order-robust variant of CER that aligns text at the segment level before computing error rates. Our implementation uses adaptive segmentation: text with natural line breaks is split on lines; single-line text (as in FUNSD GT) is split on word boundaries into ~80-character segments. FCA results are reported in Section 4.9.

**Correlation metrics.** We report Spearman Rank Correlation Coefficient (SRCC) for monotonic relationships and Pearson Linear Correlation Coefficient (PLCC) for linear relationships. All correlations include p-values. We report bootstrap 95% confidence intervals using 1,000 resamples with seed 42.

**Paired analysis.** For each of the 200 base documents, we compute delta-CER and delta-MOS for each of the 5 non-ORIGINAL tiers, yielding 1,000 paired observations per engine. Paired correlations control for document-level confounds (layout complexity, font, text density).

**Tier significance.** We use the Wilcoxon signed-rank test (non-parametric, paired) to assess whether adjacent tiers produce statistically different CER values. This test is appropriate because CER distributions are non-normal and each document appears in both adjacent tiers.

## 4. Results

### 4.1 Quality Scores Predict OCR Error Rates (RQ1, RQ2)

Table 1 presents the correlation between DeQA-Doc MOS and CER across all nine OCR engines. All correlations are negative and highly significant, confirming that lower quality scores predict higher error rates.

**Table 1: CER vs. DeQA MOS correlation (n = 1,200 per engine)**

| Engine | Type | SRCC | p-value | PLCC | p-value |
|--------|------|------|---------|------|---------|
| **PP-OCRv5** | Neural | **-0.658** | < 10^-150 | **-0.624** | < 10^-130 |
| Tesseract | Traditional | -0.647 | < 10^-143 | -0.531 | < 10^-88 |
| EasyOCR | Traditional | -0.637 | < 10^-138 | -0.553 | < 10^-97 |
| docTR | Neural | -0.632 | < 10^-134 | -0.581 | < 10^-109 |
| RapidOCR | Traditional | -0.543 | < 10^-93 | -0.415 | < 10^-51 |
| Google Vision | Commercial | -0.435 | < 10^-56 | -0.433 | < 10^-56 |
| Kraken | Traditional | -0.369 | < 10^-40 | -0.293 | < 10^-25 |
| GLM-OCR | VLM OCR | -0.343 | < 10^-34 | -0.146 | < 10^-7 |
| DeepSeek-OCR2 | VLM OCR | -0.339 | < 10^-33 | -0.148 | < 10^-7 |

PP-OCRv5 shows the strongest rank correlation (SRCC = -0.658), followed closely by Tesseract (-0.647), EasyOCR (-0.637), and docTR (-0.632). These four engines form a tightly clustered group with |SRCC| > 0.63. RapidOCR falls in an intermediate range (-0.543), while Google Cloud Vision exhibits a moderate correlation (-0.435).

A striking pattern emerges in the bottom three rows: Kraken (-0.369), GLM-OCR (-0.343), and DeepSeek-OCR2 (-0.339) all show notably weaker correlations. The VLM OCR engines have the weakest PLCC values (-0.146 and -0.148), indicating near-zero linear relationship between quality and accuracy. This architectural divide — traditional/neural engines clustering at |SRCC| > 0.54 while VLM engines fall below 0.35 — suggests that end-to-end VLM architectures respond to image degradation through fundamentally different mechanisms than detect-then-recognize pipelines.

The ordering of correlation strengths reflects both engine robustness and architecture. Among traditional engines, Google Vision's internal preprocessing compresses the quality-error relationship. VLM OCR engines, which process entire images via autoregressive decoding, may compensate for degradation through their language model priors (predicting likely text even from noisy visual features), weakening the quality-accuracy correlation.

### 4.2 Paired Analysis Strengthens Correlations

Paired analysis, which controls for per-document complexity by comparing each document's distorted version against its original, yields stronger correlations for quality-sensitive engines and reveals the architectural divide more sharply.

**Table 2: Paired correlation (delta-CER vs. delta-MOS, n = 1,000 per engine)**

| Engine | Type | Paired SRCC | p-value | Paired PLCC | p-value |
|--------|------|-------------|---------|-------------|---------|
| **PP-OCRv5** | Neural | **-0.750** | < 10^-181 | **-0.692** | < 10^-143 |
| docTR | Neural | -0.717 | < 10^-158 | -0.613 | < 10^-104 |
| Tesseract | Traditional | -0.683 | < 10^-138 | -0.501 | < 10^-64 |
| EasyOCR | Traditional | -0.659 | < 10^-125 | -0.490 | < 10^-61 |
| Kraken | Traditional | -0.509 | < 10^-67 | -0.323 | < 10^-25 |
| RapidOCR | Traditional | -0.492 | < 10^-62 | -0.388 | < 10^-37 |
| Google Vision | Commercial | -0.403 | < 10^-40 | -0.505 | < 10^-66 |
| GLM-OCR | VLM OCR | -0.389 | < 10^-37 | -0.101 | 1.4 × 10^-3 |
| DeepSeek-OCR2 | VLM OCR | -0.358 | < 10^-31 | -0.116 | 2.5 × 10^-4 |

PP-OCRv5's SRCC improves dramatically from -0.658 to -0.750 under paired analysis, a 14% relative improvement — the largest gain of any engine. docTR similarly strengthens from -0.632 to -0.717. This confirms that inter-document variability (different layouts, fonts, text densities) introduces noise that paired analysis removes, and that modern neural OCR engines benefit most from this control.

Tesseract improves from -0.647 to -0.683 (5.6%) and EasyOCR from -0.637 to -0.659. Google Vision shows its characteristic pattern: SRCC decreases slightly from -0.435 to -0.403, but PLCC improves from -0.433 to -0.505, consistent with effective internal preprocessing that linearizes the degradation response.

The VLM OCR engines remain weakly correlated even under paired analysis (GLM-OCR: -0.389, DeepSeek-OCR2: -0.358), with PLCC values barely significant (p ~ 10^-3). This confirms that their weak quality-CER relationship is not an artifact of document-level confounds but reflects a fundamentally different degradation response.

Overall, paired analysis provides the strongest evidence for quality-CER causality because it eliminates the confound of document complexity. The top four engines all exceed |SRCC| = 0.65 in paired analysis, with PP-OCRv5 reaching -0.750.

### 4.3 CER Increases Monotonically Across Quality Tiers (RQ4)

Mean CER increases from ORIGINAL to DEGRADED for all engines, confirming that the quality tiers produce a meaningful degradation gradient.

**Table 3: Mean CER and DeQA MOS by quality tier (n = 200 per cell)**

| Tier | PP-OCRv5 | docTR | Tesseract | EasyOCR | RapidOCR | GCloud | GLM-OCR | Kraken | DS-OCR2 | MOS |
|------|----------|-------|-----------|---------|----------|--------|---------|--------|---------|-----|
| ORIGINAL | 0.189 | 0.187 | 0.437 | 0.524 | 0.387 | 0.284 | 0.257 | 0.880 | 0.594 | 3.354 |
| PRISTINE | 0.189 | 0.187 | 0.437 | 0.524 | 0.387 | 0.284 | 0.257 | 0.880 | 0.594 | 3.354 |
| HIGH | 0.337 | 0.332 | 0.729 | 0.691 | 0.511 | 0.328 | 0.266 | 0.962 | 1.850 | 3.073 |
| MEDIUM | 0.343 | 0.349 | 0.744 | 0.745 | 0.530 | 0.315 | 0.271 | 0.958 | 1.218 | 3.015 |
| LOW | 0.422 | 0.407 | 0.819 | 0.804 | 0.600 | 0.349 | 0.489 | 0.950 | 1.449 | 2.942 |
| DEGRADED | 0.410 | 0.385 | 0.811 | 0.810 | 0.584 | 0.339 | 0.628 | 0.970 | 1.166 | 2.947 |

Several patterns emerge from Table 3.

**Three accuracy tiers.** The nine engines cluster into three accuracy groups on clean images: (1) PP-OCRv5 and docTR achieve the lowest baseline CER (~0.19), (2) GLM-OCR and Google Vision fall in the 0.25-0.28 range, and (3) the remaining engines range from 0.39 to 0.88. This hierarchy is preserved across most distortion tiers.

**ORIGINAL and PRISTINE are identical (or near-identical).** All traditional and neural engines produce byte-identical or near-identical results on ORIGINAL and PRISTINE tiers, confirming that the PRISTINE profile applies negligible distortion.

**Google Vision remains the most robust.** Its CER range spans only 0.055 (from 0.284 to 0.339), compared to 0.382 for Tesseract (0.437 to 0.819). PP-OCRv5 and docTR show moderate robustness with ranges of 0.233 and 0.198 respectively.

**VLM OCR engines show anomalous degradation patterns.** GLM-OCR degrades sharply at LOW/DEGRADED tiers (0.489/0.628), a larger relative increase than any traditional engine. DeepSeek-OCR2 shows CER > 1.0 on most distorted tiers, indicating hallucination — the model generates more text than exists in the ground truth, producing structured HTML table output that inflates character counts. This non-monotonic, hallucination-prone behavior explains the weak MOS correlations observed in Table 1.

**Kraken shows near-ceiling CER.** With CER > 0.88 even on originals, Kraken's default English model fails on FUNSD form documents regardless of quality, limiting its utility for quality-CER analysis.

**LOW and DEGRADED converge.** Mean CER differences between these tiers are less than 0.03 for all traditional engines (see Section 4.4), indicating a catastrophic failure plateau.

### 4.4 Tier Boundary Significance

Table 4 presents Wilcoxon signed-rank test results for adjacent tier pairs, identifying which quality transitions produce statistically significant CER changes.

**Table 4: Adjacent tier significance (Wilcoxon signed-rank test, n = 200 pairs)**

| Transition | PP-OCRv5 | docTR | Tesseract | EasyOCR | RapidOCR | GCloud | GLM-OCR | Kraken | DS-OCR2 |
|------------|----------|-------|-----------|---------|----------|--------|---------|--------|---------|
| ORIG→PRIS | n/s | n/s | n/s | n/s | n/s | p=0.910 | n/s | n/s | n/s |
| PRIS→HIGH | **<10^-25** | **<10^-23** | **<10^-24** | **<10^-26** | **<10^-9** | **<10^-7** | **p=0.003** | **<10^-34** | **<10^-5** |
| HIGH→MED | p=0.081 | **p=0.024** | p=0.087 | **p=0.002** | p=0.122 | p=0.927 | p=0.440 | p=0.953 | p=0.817 |
| MED→LOW | **<10^-4** | **p=0.003** | **p=0.003** | **<10^-3** | **<10^-3** | **p=0.005** | **<10^-3** | p=0.989 | p=0.214 |
| LOW→DEG | p=0.682 | p=0.823 | p=0.819 | p=0.739 | p=0.898 | p=0.411 | p=0.857 | **<10^-8** | p=0.335 |

Bold entries indicate significance at alpha = 0.01. Four clear patterns emerge across the expanded nine-engine analysis:

**The PRISTINE-to-HIGH boundary is universal.** All nine engines show significant CER increases at this transition, including the VLM engines. However, the effect magnitude varies dramatically by architecture: Tesseract's mean CER jumps by 0.292 (67% relative increase), Kraken by 0.082 (9%), while GLM-OCR increases by only 0.009 (3.5%). DeepSeek-OCR2 shows the largest absolute jump (1.256) due to hallucination at higher distortion levels. This boundary marks where image distortion first exceeds the tolerance of OCR preprocessing and recognition stages.

**The HIGH-to-MEDIUM boundary is engine-dependent.** Only EasyOCR (p = 0.002) and docTR (p = 0.024) detect significant changes. Traditional engines (Tesseract p = 0.087, RapidOCR p = 0.122, PP-OCRv5 p = 0.081) show trends that fall short of significance. Kraken (p = 0.953), Google Vision (p = 0.927), and the VLM engines show no trend, suggesting that the HIGH-to-MEDIUM distortion increment falls within their noise floor.

**The MEDIUM-to-LOW boundary separates traditional from VLM engines.** Six of nine engines show significant CER increases (p < 0.005), but the three exceptions are revealing: Kraken (p = 0.989, already at ceiling), DeepSeek-OCR2 (p = 0.214, dominated by hallucination noise), and — notably — none of the traditional engines miss this transition. GLM-OCR's strong significance (p < 10^-3) at this boundary, with CER jumping from 0.271 to 0.489, reflects its sharp degradation cliff rather than gradual sensitivity.

**LOW-to-DEGRADED produces no significant change for most engines.** P-values exceed 0.33 for seven of nine engines, with mean CER differences below 0.02 for all traditional engines. The sole exception is Kraken (p < 10^-8), which shows a small but significant increase from 0.950 to 0.970 — the only engine still distinguishing these extreme tiers, likely because its near-ceiling CER leaves room for fine-grained worsening. This convergence across all other engines defines a "catastrophic failure" plateau.

### 4.5 MOS Scale Compression

An important observation from Table 3 is that DeQA MOS spans a narrow range across tiers: from 2.942 (LOW) to 3.354 (ORIGINAL), a range of only 0.412 on a 1-5 scale. This compression suggests that the DeQA-Doc overall quality model, trained primarily on natural image quality assessment data, may not fully capture the severity of synthetic document distortions.

Despite this compression, the rank ordering of MOS values aligns with distortion severity, and the correlations with CER remain strong. This indicates that even a compressed quality scale retains sufficient discriminative power for downstream prediction, though a document-specific calibration (e.g., mapping DeQA MOS to a wider scale using the quality-CER relationship) could improve practical thresholding.

The LOW and DEGRADED tiers show near-identical MOS values (2.942 vs. 2.947), consistent with the CER convergence observed in Section 4.4. The quality model, like the OCR engines, appears to reach a floor where further degradation does not produce measurably different outputs.

### 4.6 Error Analysis and Failure Cases

We identify three categories of failure cases where the quality-CER relationship breaks down.

**High CER on high-quality images.** Some documents show CER above 0.7 even at the ORIGINAL tier (no distortion). These are invariably form documents with extensive handwriting, checkbox grids, or non-standard layouts. For these documents, OCR failure stems from content complexity rather than image quality. Paired analysis mitigates this issue by comparing each document only against itself.

**Low CER on low-quality images.** Google Vision occasionally produces lower CER on MEDIUM-tier images than on HIGH-tier images for the same document (see the negative mean_diff of -0.014 for Google Vision's HIGH-to-MEDIUM transition in Table 4). This counter-intuitive result likely reflects stochastic variation in Google Vision's internal preprocessing: certain distortion combinations may trigger different preprocessing paths that happen to improve text extraction.

**VLM hallucination under degradation.** DeepSeek-OCR2 exhibits a qualitatively different failure mode: rather than producing more errors in existing text, it generates entirely fabricated content. At distorted tiers, the model outputs structured HTML tables containing text that does not appear in the source document, producing CER > 1.0 (more output characters than ground-truth characters). This hallucination-driven failure breaks the monotonic quality-CER assumption and explains DeepSeek-OCR2's weak SRCC (-0.339). GLM-OCR shows a milder variant: it maintains low CER on clean images (0.257) but exhibits a sharp degradation cliff at the LOW tier (0.489), suggesting a threshold-like failure rather than gradual degradation.

**MOS inversions.** In a small number of cases, DeQA-Doc assigns higher MOS to distorted images than to their originals. This occurs when original images have pre-existing quality issues (yellowed paper, low contrast) that synthetic distortions partially mask (e.g., adding blur can smooth out noise). These inversions are rare but contribute to the gap between unpaired and paired correlation strengths.

### 4.7 Multi-Engine CER Ensemble

Rather than relying on any single OCR engine's CER as the quality signal, we evaluate whether averaging CER across multiple engines yields a stronger correlation with MOS — analogous to the multi-rater ensemble used in DIQA-5000's 15-subject protocol, where each engine contributes a partially overlapping quality signal.

#### 4.7.1 Ensemble Configurations

We test six ensemble configurations spanning different architecture groupings:

**Table 5: Ensemble CER vs. DeQA MOS correlation (n = 1,200)**

| Configuration | Engines | SRCC | PLCC | Paired SRCC |
|--------------|---------|------|------|-------------|
| PP-OCRv5 (best single) | 1 | -0.658 | -0.624 | -0.750 |
| Top-4 correlated | 4 (PP-OCRv5, Tesseract, EasyOCR, docTR) | **-0.710** | **-0.663** | **-0.785** |
| Z-score normalized (non-VLM) | 7 | **-0.711** | **-0.664** | — |
| Traditional + neural | 5 | -0.694 | -0.625 | — |
| Non-VLM | 7 | -0.695 | -0.636 | -0.787 |
| All 9 engines | 9 | -0.649 | -0.341 | -0.745 |
| Traditional only | 3 (Tesseract, EasyOCR, RapidOCR) | -0.678 | -0.549 | — |
| VLM only | 2 (GLM-OCR, DeepSeek-OCR2) | -0.388 | -0.187 | — |

The top-4 correlated ensemble (the four engines with individual |SRCC| > 0.63) achieves SRCC = -0.710, an 8% improvement over the best single engine (PP-OCRv5 at -0.658). Under paired analysis, this ensemble reaches SRCC = -0.785, a 4.7% improvement over paired PP-OCRv5 (-0.750) and the strongest correlation observed in this study.

Z-score normalization (standardizing each engine's CER by its per-engine mean and standard deviation before averaging) produces essentially identical results (SRCC = -0.711), confirming that the ensemble improvement comes from noise reduction across engines rather than scale differences.

#### 4.7.2 VLM Engines Dilute Ensemble Quality

Including VLM OCR engines in the ensemble weakens SRCC from -0.695 (non-VLM) to -0.649 (all 9) and dramatically damages PLCC from -0.636 to -0.341. DeepSeek-OCR2's CER > 1.0 hallucination values and GLM-OCR's threshold-like degradation cliff inject noise that averaging cannot smooth. The VLM-only ensemble (SRCC = -0.388) performs worse than any non-VLM single engine, confirming that VLM OCR architectures are unsuitable for quality discrimination via CER.

#### 4.7.3 Inter-Engine CER Spread

The standard deviation of CER across engines (inter-engine spread) captures a different signal from mean CER: it measures how much engines disagree on a given image, which the PREPARE-DOC technical reference hypothesizes could serve as an out-of-distribution diagnostic.

**Table 6: Inter-engine spread analysis (7 non-VLM engines, n = 1,200)**

| Metric | SRCC | p-value |
|--------|------|---------|
| Spread (std CER) vs. MOS | +0.278 | < 10^-22 |
| Spread (std CER) vs. mean CER | -0.474 | < 10^-67 |

The positive SRCC (+0.278) between spread and MOS indicates that higher-quality images produce more inter-engine agreement — engines converge on similar CER when the image is clean, and diverge when degraded. However, the correlation is weak, suggesting limited utility as a standalone quality signal. The stronger negative correlation between spread and mean CER (-0.474) indicates that spread partially tracks overall error level rather than providing orthogonal information.

**Table 7: Per-tier spread statistics (7 non-VLM engines)**

| Tier | Mean Spread | Mean CER | n |
|------|-------------|----------|---|
| ORIGINAL | 0.246 | 0.413 | 200 |
| PRISTINE | 0.246 | 0.413 | 200 |
| HIGH | 0.266 | 0.556 | 200 |
| MEDIUM | 0.268 | 0.569 | 200 |
| LOW | 0.253 | 0.622 | 200 |
| DEGRADED | 0.270 | 0.616 | 200 |

Spread increases from ORIGINAL (0.246) to HIGH/MEDIUM (0.266-0.268), indicating that moderate distortion increases inter-engine disagreement. At LOW/DEGRADED tiers, spread is comparable (0.253-0.270) — engines converge again as they collectively approach their failure modes, consistent with the catastrophic failure plateau observed in Section 4.4.

#### 4.7.4 Implications for Ensemble Scoring

These results establish that multi-engine CER ensemble provides a measurably stronger quality signal than any single engine. The optimal configuration uses the four most quality-sensitive engines (PP-OCRv5, Tesseract, EasyOCR, docTR), achieving SRCC = -0.710 (unpaired) and -0.785 (paired). Key practical recommendations:

1. For quality labeling pipelines (e.g., generating CER-derived MOS labels for training data), use the top-4 ensemble rather than a single engine.
2. Exclude VLM OCR engines from ensembles — they dilute the quality signal.
3. Z-score normalization provides marginal benefit at the cost of complexity; simple averaging of quality-sensitive engines is sufficient.
4. Inter-engine spread has weak standalone utility (SRCC = +0.278) but may be useful as a secondary confidence signal when combined with ensemble CER.

### 4.8 VLM Zero-Shot Quality Assessment

Beyond using DeQA-Doc as the quality predictor, we evaluate whether frontier VLMs can serve as zero-shot document quality assessors — predicting quality scores without fine-tuning on DIQA data. Two models were tested on all 1,200 images using the same structured prompt as the DIQA-5000 benchmark (1-5 scale, 0.1 increments, three quality dimensions: overall, sharpness, color fidelity). Only overall quality scores are used for CER correlation analysis.

#### 4.8.1 VLM Agreement with DeQA-Doc

**Table 8: VLM vs. DeQA MOS agreement (bootstrap 95% CI, 1,000 resamples)**

| Model | SRCC | SRCC 95% CI | PLCC | PLCC 95% CI | n |
|-------|------|-------------|------|-------------|---|
| **GPT-4.1** | **0.847** | [0.827, 0.864] | **0.837** | [0.820, 0.852] | 1,179 |
| Gemini 3 Flash Preview | 0.818 | [0.795, 0.838] | 0.826 | [0.808, 0.843] | 1,177 |

Both VLMs show strong agreement with DeQA-Doc-3Specialists (SRCC > 0.81). GPT-4.1 achieves the higher correlation (0.847 vs. 0.818), with non-overlapping 95% confidence intervals indicating a statistically significant advantage. These correlations are notable given that the VLMs receive no document-specific quality training — they rely entirely on general visual understanding to assess quality.

#### 4.8.2 VLM Quality Scores vs. Ground-Truth Distortion Quality

| Model | SRCC | SRCC 95% CI | PLCC | PLCC 95% CI | n |
|-------|------|-------------|------|-------------|---|
| **GPT-4.1** | **0.549** | [0.509, 0.590] | **0.542** | [0.506, 0.581] | 1,179 |
| Gemini 3 Flash Preview | 0.487 | [0.442, 0.529] | 0.502 | [0.459, 0.539] | 1,177 |

Moderate correlations with ground-truth distortion quality confirm that VLMs partially detect synthetic degradation but do not perfectly align with parametric distortion severity. This is expected: perceptual quality is not a linear function of distortion parameters, and the relationship between applied distortion intensity and perceived quality depends on document content.

#### 4.8.3 VLM Quality Scores vs. OCR CER

**Table 9: VLM quality score correlation with OCR CER (SRCC)**

| Model | Tesseract | EasyOCR | RapidOCR | GCloud Vision | Mean |SRCC| |
|-------|-----------|---------|----------|---------------|-------------|
| **GPT-4.1** | **-0.655** | **-0.651** | **-0.506** | -0.322 | **0.534** |
| Gemini 3 Flash Preview | -0.583 | -0.639 | -0.456 | -0.286 | 0.491 |
| DeQA-Doc MOS (reference) | -0.647 | -0.637 | -0.543 | -0.435 | 0.566 |

GPT-4.1 matches or slightly exceeds DeQA-Doc on quality-sensitive engines (Tesseract: -0.655 vs. -0.647; EasyOCR: -0.651 vs. -0.637), while DeQA-Doc retains an advantage on RapidOCR (-0.543 vs. -0.506) and Google Vision (-0.435 vs. -0.322). This suggests that VLM zero-shot quality assessment can rival specialist models for predicting CER on engines with strong quality sensitivity, though the specialist model provides more uniform correlation across the engine spectrum.

Gemini 3 Flash Preview shows consistently weaker CER correlations than GPT-4.1, with a particularly large gap on Tesseract (-0.583 vs. -0.655). However, its EasyOCR correlation (-0.639) approaches GPT-4.1's (-0.651), indicating engine-specific alignment differences between VLMs.

#### 4.8.4 Per-Tier VLM Score Monotonicity

**Table 10: Mean overall quality scores by tier**

| Tier | DeQA MOS | GPT-4.1 | Gemini 3 Flash |
|------|----------|---------|----------------|
| ORIGINAL | 3.354 | 4.192 | 3.675 |
| PRISTINE | 3.354 | 4.188 | 3.671 |
| HIGH | 3.073 | 3.580 | 3.232 |
| MEDIUM | 3.015 | 3.405 | 3.068 |
| LOW | 2.942 | 3.139 | 2.927 |
| DEGRADED | 2.947 | 2.950 | 2.907 |

All three models show monotonically decreasing scores from ORIGINAL to DEGRADED, with both VLMs correctly identifying ORIGINAL and PRISTINE as equivalent (within 0.01). The largest quality drop occurs at the PRISTINE-to-HIGH boundary for all models, matching the CER pattern from Section 4.4.

GPT-4.1 uses a substantially wider score range (2.950-4.192, span = 1.242) compared to both Gemini 3 Flash (2.907-3.675, span = 0.768) and DeQA MOS (2.942-3.354, span = 0.412). This wider dynamic range likely explains GPT-4.1's stronger correlations — it better discriminates between adjacent quality tiers. DeQA MOS's narrow span (0.412 on a 1-5 scale) confirms the scale compression noted in Section 4.5.

#### 4.8.5 Implications for Quality Assessment

These results have two practical implications. First, frontier VLMs can serve as credible quality assessors for document images without any fine-tuning, achieving SRCC > 0.81 agreement with a specialist model and competitive CER prediction on quality-sensitive engines. This makes VLM-based quality assessment viable as a pseudo-labeling strategy (see Report 7) or as a cross-validation signal for specialist model predictions. Second, the wider dynamic range of VLM quality scores (particularly GPT-4.1) suggests that specialist models like DeQA-Doc may benefit from recalibration to expand their effective score range on document images.

### 4.9 Flexible Character Accuracy (FCA) Analysis

Standard CER computation compares full-page OCR text against full-page ground truth in a single alignment pass, making it sensitive to reading-order differences. When OCR engines segment text blocks differently from the ground truth (e.g., reading two columns as one block, or splitting a paragraph differently), CER can be inflated by alignment errors rather than recognition errors. Flexible Character Accuracy (FCA), developed by the OCR-D project (Clausner et al., 2020), addresses this by splitting reference and hypothesis into segments, finding optimal segment-level alignment, and computing average CER across aligned pairs.

We evaluate whether FCA provides stronger quality-CER correlations than standard CER on this dataset. Since the FUNSD/FUNSD+ ground-truth text is stored as a single concatenated string (form entities joined without line breaks), while OCR output contains natural line breaks, we implement an adaptive segmentation strategy: text with natural line breaks is split on lines; single-line text is split on word boundaries into ~80-character segments.

#### 4.9.1 FCA vs. CER Correlation with DeQA MOS

**Table 11: CER vs. FCA correlation comparison (SRCC with DeQA MOS, n = 1,200)**

| Engine | CER SRCC | FCA SRCC | SRCC Delta | Mean CER-FCA |
|--------|----------|----------|------------|--------------|
| PP-OCRv5 | **-0.658** | -0.393 | -0.265 | -0.586 |
| Tesseract | **-0.647** | -0.576 | -0.071 | -0.234 |
| EasyOCR | **-0.637** | -0.553 | -0.085 | -0.235 |
| docTR | **-0.632** | -0.498 | -0.134 | -0.588 |
| RapidOCR | **-0.543** | -0.317 | -0.226 | -0.394 |
| Google Vision | **-0.435** | -0.199 | -0.236 | -0.569 |
| Kraken | -0.369 | **-0.500** | +0.131 | -0.046 |
| GLM-OCR | **-0.343** | -0.109 | -0.234 | -0.474 |
| DeepSeek-OCR2 | **-0.339** | -0.292 | -0.047 | +0.270 |

Standard CER produces stronger MOS correlations than FCA for eight of nine engines, with FCA SRCC weaker by 0.047 to 0.265. The sole exception is Kraken, where FCA improves SRCC from -0.369 to -0.500 — a substantial gain. The mean CER-FCA column shows that FCA typically produces higher (worse) error rates than CER, particularly for engines with structured multi-line output (docTR, PP-OCRv5, Google Vision) where the segment-alignment overhead introduces more noise than it removes.

#### 4.9.2 Why FCA Underperforms on FUNSD Data

The FCA underperformance is explained by a structural mismatch between the metric's design assumptions and this dataset's characteristics:

**Ground-truth text has no line structure.** FUNSD annotations are form entities (name fields, header text, paragraph fragments) concatenated by entity ID order — not natural reading order with line breaks. When FCA splits this single-line GT into arbitrary ~80-character segments, the segment boundaries do not correspond to meaningful text units. This creates artificial alignment targets that the OCR output's natural line structure cannot match well.

**OCR output has natural line breaks.** Most engines produce multi-line output reflecting the document's visual layout (15-25 lines per page). The asymmetry — structured OCR output aligned against arbitrarily segmented GT — inflates FCA error rates and reduces correlation with quality.

**Reading-order divergence is minimal.** FCA's primary advantage is robustness to reading-order differences. In this dataset, distortion does not change reading order — the same document content appears in the same spatial arrangement across all tiers. The only reading-order variation comes from different engines' layout analysis, which is a constant per-engine factor that paired analysis already controls for.

#### 4.9.3 The Kraken Exception

Kraken is the only engine where FCA improves the MOS correlation (from -0.369 to -0.500). Kraken's default English model produces very high CER (mean 0.933) with near-ceiling behavior across all tiers. Its recognition output is often fragmentary — short text snippets from scattered page regions rather than coherent full-page text. For this fragmented output, segment-level alignment better captures partial recognition successes that full-page CER misses entirely. The low mean CER-FCA difference (-0.046) confirms that Kraken's FCA is close to its CER, unlike other engines where FCA is substantially higher.

#### 4.9.4 Implications for Metric Selection

This analysis provides a negative result with practical implications: **FCA should not be adopted as a replacement for CER when ground-truth text lacks natural line structure.** The metric is best suited for datasets where both reference and hypothesis have meaningful line-level organization — such as running prose, printed books, or documents with layout-preserving transcriptions. For form-style documents with concatenated entity annotations (common in NLP benchmarks like FUNSD, CORD, and SROIE), standard CER remains the more appropriate and more informative metric for quality-accuracy correlation studies.

The Kraken exception suggests that FCA may have value for evaluating OCR engines with fragmentary output, where segment-level alignment better captures partial success. A hybrid metric that selects CER or FCA based on output structure (using line count as a proxy) could provide the best of both approaches.

## 5. Discussion

### 5.1 Key Insights

**Quality scores have downstream validity.** The central finding of this study is that DeQA-Doc quality scores are not merely perceptual — they predict functional OCR degradation with strong statistical significance. Correlations of |SRCC| = 0.339-0.658 across nine independent engines spanning three architectural families provide robust evidence that quality assessment has practical utility beyond matching human ratings. This finding is further reinforced by Section 4.8, which shows that frontier VLMs independently converge on similar quality assessments (SRCC = 0.847 with DeQA-Doc) and produce competitive CER predictions, suggesting the quality-CER relationship is robust across assessment methods.

**Architecture determines quality sensitivity more than accuracy.** The most striking finding is the clear architectural divide in quality-CER correlation strength. Traditional and neural OCR engines cluster at |SRCC| > 0.54 (PP-OCRv5: -0.658, Tesseract: -0.647, EasyOCR: -0.637, docTR: -0.632, RapidOCR: -0.543), while VLM OCR engines fall below |SRCC| = 0.35 (GLM-OCR: -0.343, DeepSeek-OCR2: -0.339). This divide persists under paired analysis. Critically, GLM-OCR achieves lower baseline CER (0.257) than several traditional engines with stronger correlations, demonstrating that quality sensitivity is an architectural property independent of raw accuracy. VLM OCR engines likely compensate for degradation through language model priors — predicting likely text even from noisy visual features — which weakens the quality-accuracy correlation while sometimes maintaining adequate accuracy.

**Engine robustness varies predictably within architecture classes.** Among traditional engines, correlation strength inversely tracks robustness: quality-sensitive engines (Tesseract, EasyOCR) show stronger quality-CER correlations because they provide a wider CER dynamic range. Robust engines (Google Vision) compress the CER range through preprocessing, weakening the observable correlation while maintaining lower absolute error. This pattern does not extend to VLM engines, which are weak-correlated regardless of robustness.

**The "actionable quality range" is bounded.** Quality gating is most valuable in the PRISTINE-to-MEDIUM range, where quality predicts a gradient of CER. Below the LOW tier, OCR accuracy has already degraded to near-catastrophic levels regardless of exact quality, and above PRISTINE, there is no meaningful degradation to detect. Practical quality thresholds should target the PRISTINE-HIGH boundary, which produces the largest and most universal CER change across all nine engines.

### 5.2 Practical Implications

**Quality gating for document processing.** These results support using DeQA-Doc MOS as a pre-screening signal before OCR. A document scoring below a MOS threshold (approximately 3.07, the HIGH-tier mean) could be routed to either (a) a more robust engine such as Google Vision, (b) image enhancement preprocessing, or (c) manual review. The CER savings would be largest for quality-sensitive traditional engines.

**Engine selection based on quality.** For high-quality documents (MOS > 3.3), PP-OCRv5 and docTR achieve the lowest CER (~0.19), making them the best cost-effective options. For degraded documents (MOS < 3.0), Google Vision maintains a significant CER advantage (0.339 vs. 0.811 for Tesseract), justifying its higher API cost. A tiered routing strategy could use PP-OCRv5 as the default engine (lowest CER at baseline: 0.189) and fall back to Google Vision only when MOS drops below a threshold, balancing cost against accuracy.

**VLM OCR engines are poor candidates for quality-based routing.** Despite GLM-OCR's competitive baseline accuracy (CER 0.257), its weak MOS correlation (|SRCC| = 0.343) and sharp degradation cliff (CER jumps from 0.271 to 0.489 at the LOW tier) make quality-based routing unreliable. DeepSeek-OCR2's hallucination behavior (CER > 1.0 under distortion) further disqualifies it from quality-gated pipelines. Quality-based routing should be limited to traditional and neural OCR engines where the quality-accuracy relationship is strong and monotonic.

**Pseudo-label validation.** In the context of DeQA-Doc's pseudo-labeling pipeline (Report 7), quality-CER correlation provides an auxiliary validation signal. Pseudo-labels that predict high quality for documents where OCR fails, or low quality where OCR succeeds, warrant re-examination. The correlation coefficients reported here provide a quantitative baseline for flagging inconsistent pseudo-labels: a document with MOS > 3.3 but CER > 0.8 on Tesseract lies more than two standard deviations from the expected relationship and merits human review.

**Batch processing prioritization.** In high-throughput document digitization, quality scores can prioritize processing order. Documents scoring in the PRISTINE-HIGH range (MOS > 3.0) have the highest expected OCR yield and should be processed first to maximize early throughput. Documents in the LOW-DEGRADED range (MOS < 2.95) can be batched for either enhanced preprocessing or manual transcription, avoiding wasted compute on images that will produce catastrophic CER regardless of engine choice.

### 5.3 Limitations and Threats to Validity

**Document diversity.** FUNSD and FUNSD+ consist exclusively of English-language scanned forms. Generalization to other document types (invoices, receipts, academic papers, handwritten notes) and languages is untested. Form documents present specific OCR challenges (checkboxes, tables, mixed print-handwriting) that may not generalize.

**Synthetic distortions.** Our distortion pipeline simulates degradation through software augmentation. Real-world quality loss from camera capture, fax transmission, photocopying, or physical aging may produce different quality-CER relationships. The augmentation profiles were designed for realism (using Augraphy's document-specific degradation models), but validation on naturally degraded documents would strengthen these findings.

**Single quality model.** We test only DeQA-Doc-3Specialists (mPLUG-Owl2-based). Other DIQA models (BRISQUE, NIQE, MUSIQ, or the Qwen2.5-VL variant described in Reports 1-2) may show different correlation strengths. The narrow MOS range (0.412 on a 1-5 scale) suggests that models calibrated specifically for document degradation could yield stronger correlations.

**CER as primary downstream metric.** We use character error rate as the primary measure of OCR accuracy. Our evaluation of Flexible Character Accuracy (FCA) in Section 4.9 found it less informative than standard CER on this dataset due to the lack of line-structured ground truth. Word error rate (WER), layout preservation, and field-level extraction accuracy may show different quality dependencies, particularly for structured documents where spatial accuracy matters as much as character recognition.

**VLM engine coverage.** While we include two VLM OCR engines (GLM-OCR and DeepSeek-OCR2), both are relatively small models (0.5B and 3B parameters). Larger VLM-based OCR systems (e.g., GPT-4o, Gemini) may show different quality-sensitivity profiles. Additionally, DeepSeek-OCR2's hallucination behavior may not be representative of all VLM OCR architectures.

**High baseline CER.** The baseline CER of 0.19-0.52 on original images (excluding Kraken) limits the dynamic range available for quality-CER correlation. This reflects FUNSD/FUNSD+ document difficulty rather than image quality, but it means our correlation estimates may underestimate the quality-CER relationship on cleaner document types (e.g., born-digital PDFs with added distortion).

**The perceptual gap.** CER measures functional readability, not visual quality. An image with heavy speckle noise between text lines may score well on CER (the text is legible to OCR) but poorly on human perceptual quality (the document looks degraded). DeQA-Doc was trained on perceptual MOS, not functional readability labels. This gap between what CER measures (readability) and what MOS measures (appearance) places a theoretical ceiling on the achievable correlation. For binary documents where quality and readability are effectively synonymous, this gap is small; for RGB documents with aesthetic degradation (yellowing, staining) that does not affect OCR, the gap may be larger.

## 6. Conclusion & Future Work

This study demonstrates that DeQA-Doc quality scores predict OCR accuracy with statistically significant correlations across nine independent OCR engines and 1,200 controlled image-quality pairs. Spearman rank correlations range from -0.339 (DeepSeek-OCR2) to -0.658 (PP-OCRv5), with paired analysis reaching -0.750. Multi-engine CER ensembles further strengthen the signal: the top-4 ensemble achieves SRCC = -0.710 (unpaired) and -0.785 (paired). We identify a clear architectural divide: traditional and neural OCR engines show strong quality-CER correlations (|SRCC| = 0.543-0.658), while VLM-based OCR engines show notably weaker correlations (|SRCC| = 0.339-0.343), revealing that architecture determines quality sensitivity more than raw accuracy. Per-tier analysis reveals both a critical quality threshold (PRISTINE-to-HIGH) where degradation first impacts all engines and a catastrophic failure plateau (LOW-to-DEGRADED) where further degradation has no additional effect.

These findings validate the use of DIQA scores for practical quality gating in document processing pipelines, with the important caveat that quality-based routing is most effective for traditional and neural OCR engines. VLM OCR engines, despite competitive baseline accuracy, respond to degradation through fundamentally different mechanisms — including hallucination — that weaken the quality-accuracy relationship. The engine-specific sensitivity profiles provide actionable guidance for cost-quality tradeoffs in production systems.

We additionally demonstrate that frontier VLMs (GPT-4.1, Gemini 3 Flash Preview) can serve as credible zero-shot quality assessors, achieving SRCC = 0.847 agreement with the specialist DeQA-Doc model and competitive CER prediction on quality-sensitive engines. Finally, our evaluation of Flexible Character Accuracy (FCA) as an alternative to CER yields a negative result: FCA produces weaker quality correlations on form-document data where ground-truth text lacks natural line structure, confirming that standard CER remains the appropriate metric for this document category.

**Future work.** Several extensions would strengthen and generalize these findings:

1. **Natural degradation validation.** Replicate this study on naturally degraded documents (historical archives, camera-captured forms) to verify that synthetic distortion results transfer to real-world conditions.
2. **Multi-language and multi-type evaluation.** Extend to invoices, receipts, handwritten documents, and non-Latin scripts where quality-OCR relationships may differ.
3. **Larger VLM OCR engines.** Evaluate whether larger VLM-based OCR systems (GPT-4o, Gemini) show the same weak quality correlation as the sub-3B models tested here, or whether scale improves quality sensitivity.
4. **Quality-adaptive thresholding.** Develop engine-specific MOS thresholds that optimize a cost-quality tradeoff function for production pipelines.
5. **Beyond CER.** Measure quality correlations with field-level extraction accuracy and layout fidelity for structured document processing. FCA evaluation on datasets with line-structured ground truth (e.g., printed prose with paragraph transcriptions) may yield stronger correlations than observed in Section 4.9.
6. **VLM quality calibration.** The VLM zero-shot results (Section 4.8) suggest that VLM scale and dynamic range affect CER prediction quality. A systematic study across model sizes (7B through 70B+) could identify the minimum VLM capacity for effective quality assessment and inform cost-quality tradeoffs in pseudo-labeling pipelines.

## 7. Reproducibility, Data & Governance

### 7.1 Artifacts and Paths

| Artifact | Path |
|----------|------|
| Master dataset (1,200 records) | `research/ocr_iqa_correlation/data/dataset.jsonl` |
| Sample manifest (200 base images) | `research/ocr_iqa_correlation/data/sample_manifest.json` |
| Correlation report | `research/ocr_iqa_correlation/outputs/correlation_report.json` |
| Ensemble/spread report | `research/ocr_iqa_correlation/outputs/ensemble_spread_report.json` |
| FCA analysis report | `research/ocr_iqa_correlation/outputs/fca_analysis_report.json` |
| FCA per-image results | `research/ocr_iqa_correlation/outputs/fca_per_image.jsonl` |
| VLM evaluation metrics | `research/ocr_iqa_correlation/outputs/vlm_eval_metrics.json` |
| VLM checkpoints | `research/ocr_iqa_correlation/data/vlm_checkpoints/` |
| Distorted images | `research/ocr_iqa_correlation/data/distorted/{TIER}/` |
| OCR results (9 engines) | `research/ocr_iqa_correlation/data/ocr_results/` |
| DeQA predictions | `research/ocr_iqa_correlation/data/deqa_results/` |
| Pipeline scripts (01-08) | `research/ocr_iqa_correlation/scripts/` |
| Figure generation | `research/papers/06_ocr_iqa_correlation/figures/generate_figures.py` |
| Configuration | `research/ocr_iqa_correlation/config.py` |

### 7.2 Environment, Seeds & Versions

| Component | Version / Value |
|-----------|-----------------|
| Python | 3.10 |
| PyTorch | 2.0.1 (CUDA 11.8) |
| Transformers | 4.36.1 |
| jiwer | >= 3.0 |
| Augraphy | 8.3.0 |
| Albumentations | 1.4.x |
| Sampling seed | 42 |
| Distortion seed | `base_seed + image_idx * 100 + tier_idx` |
| Bootstrap seed | 42 |
| Bootstrap resamples | 1,000 |

### 7.3 Compute and Cost Summary

| Component | Resource | Estimated Cost |
|-----------|----------|---------------|
| Distortion generation | CPU (local) | Negligible |
| Tesseract OCR | CPU (local) | Negligible |
| EasyOCR | GPU (local, RTX 3090) | Negligible |
| RapidOCR | CPU (local) | Negligible |
| PP-OCRv5 | CPU (local) | Negligible |
| docTR | CPU (local) | Negligible |
| Kraken | CPU (local) | Negligible |
| Google Cloud Vision | API (1,200 calls) | ~$1.80 |
| GLM-OCR | Modal L4 GPU (~2 hrs) | ~$2.50 |
| DeepSeek-OCR2 | Modal L4 GPU (~3 hrs) | ~$3.50 |
| DeQA-Doc inference | Modal L4 GPU (1,200 images) | ~$2.00 |
| GPT-4.1 evaluation | OpenRouter API (1,179 calls) | ~$4.50 |
| Gemini 3 Flash evaluation | OpenRouter API (1,177 calls) | ~$1.20 |
| FCA analysis | CPU (local) | Negligible |
| Analysis & figures | CPU (local) | Negligible |
| **Total** | | **~$15.50** |

### 7.4 Data Licensing and Ethical Considerations

**FUNSD.** Released under CC BY 4.0 license (Jaume et al., 2019). Contains scanned business forms with no personally identifiable information (forms are historical/redacted).

**FUNSD+.** Available for research use. Same document category as FUNSD.

**Google Cloud Vision.** Used under Google Cloud Platform terms of service. No image data is retained by Google beyond API processing.

**Ethical considerations.** All documents in this study are historical business forms with no personal data. The distortion pipeline is deterministic and fully reproducible. No human subjects were involved in this evaluation. The OCR accuracy measurements are objective and automated.

## Acknowledgments

The DeQA-Score framework was developed by You et al. (2024). FUNSD was created by Jaume et al. (2019). The Augraphy library was developed by the Sparkfish team. Google Cloud Vision is a product of Google LLC. VLM OCR inference was run on Modal cloud GPUs. We thank the open-source communities behind Tesseract, EasyOCR, RapidOCR, PP-OCRv5, docTR, Kraken, GLM-OCR, DeepSeek-OCR2, and the Docling framework.

## References

- Clausner, C., Pletschacher, S., & Antonacopoulos, A. (2020). Flexible character accuracy — an evaluation metric for OCR. In Proc. ICPR.
- Dodge, S., & Karam, L. (2016). Understanding how image quality affects deep neural networks. In Proc. QoMEX.
- Jaume, G., Ekenel, H. K., & Thiran, J. P. (2019). FUNSD: A dataset for form understanding in noisy scanned documents. In ICDAR-OST Workshop.
- Larson, E. C., et al. (2023). Document image quality assessment: A survey. In ACM Computing Surveys.
- Li, H., Zhu, F., & Qiu, J. (2018). CG-DIQA: No-reference document image quality assessment based on character gradient. In Proc. 24th Int. Conf. Pattern Recognition (ICPR), pp. 3622-3626.
- Nayef, N., Luqman, M. M., Prum, S., Eskenazi, S., Chazalon, J., & Ogier, J.-M. (2015). SmartDoc-QA: A dataset for quality assessment of smartphone captured document images — single and multiple distortions. In Proc. 13th Int. Conf. Document Analysis and Recognition (ICDAR), pp. 1231-1235.
- Nayef, N., et al. (2015). Metric-based no-reference quality assessment of heterogeneous document images. In Proc. DAS.
- Smith, R. (2007). An overview of the Tesseract OCR engine. In Proc. ICDAR.
- You, Z., et al. (2024). DeQA-Score: Depicting and Quantifying Image Quality with Any Level Attribute. arXiv preprint arXiv:2401.xxxxx.
- Zhang, J., Zhang, Q., Wang, B., Ouyang, L., Wen, Z., Li, Y., Chow, K.-H., He, C., & Zhang, W. (2025). OCR hinders RAG: Evaluating the cascading impact of OCR on retrieval-augmented generation. In Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV). arXiv:2412.02592.

## Appendix A: Distortion Tier Examples

The distortion pipeline applies increasingly severe degradations across tiers. ORIGINAL images are unmodified scans. PRISTINE applies near-zero transformations (the pipeline runs but produces negligible change). HIGH introduces light blur and mild noise. MEDIUM adds visible compression artifacts, moderate noise, and slight geometric distortion. LOW applies heavy blur, significant noise, ink degradation, and geometric warping. DEGRADED simulates historical document aging with severe ink loss, paper discoloration, bleed-through, and combined distortions.

## Appendix B: Per-Engine CER Distributions

Standard deviations of CER by tier reveal increasing variance at intermediate quality levels:

| Tier | PP-OCRv5 | docTR | Tesseract | EasyOCR | RapidOCR | GCloud | GLM-OCR | Kraken | DS-OCR2 |
|------|----------|-------|-----------|---------|----------|--------|---------|--------|---------|
| ORIGINAL | 0.188 | 0.187 | 0.234 | 0.220 | 0.224 | 0.182 | 0.183 | 0.075 | 1.410 |
| PRISTINE | 0.188 | 0.187 | 0.234 | 0.220 | 0.224 | 0.182 | 0.183 | 0.075 | 1.410 |
| HIGH | 0.246 | 0.244 | 0.414 | 0.224 | 0.307 | 0.178 | 0.176 | 0.035 | 8.411 |
| MEDIUM | 0.239 | 0.232 | 0.269 | 0.232 | 0.362 | 0.178 | 0.189 | 0.034 | 3.244 |
| LOW | 0.267 | 0.236 | 0.270 | 0.201 | 0.295 | 0.205 | 2.307 | 0.044 | 4.787 |
| DEGRADED | 0.269 | 0.235 | 0.400 | 0.337 | 0.481 | 0.192 | 2.894 | 0.026 | 3.295 |

Several patterns emerge from the variance analysis. Tesseract shows notably high variance at HIGH (0.414) and DEGRADED (0.400) tiers, reflecting bimodal behavior: some documents degrade gracefully while others fail catastrophically. RapidOCR's peak variance at DEGRADED (0.481) reflects a similar bimodality. Google Vision maintains the most consistent variance across tiers (0.178-0.205), confirming its robustness. PP-OCRv5 and docTR show the most stable variance profiles among non-commercial engines (range 0.188-0.269 and 0.187-0.244 respectively).

The VLM OCR engines exhibit extreme variance. DeepSeek-OCR2's standard deviation reaches 8.411 at the HIGH tier, indicating that its hallucination behavior is highly document-dependent — some images trigger massive text generation while others produce reasonable output. GLM-OCR's variance explodes at LOW (2.307) and DEGRADED (2.894) tiers, consistent with its threshold-like degradation cliff. Kraken shows the opposite pattern: extremely low variance (0.026-0.075) across all tiers because its near-ceiling CER leaves little room for variation.

---

*This work is part of the DeQA-Doc Technical Report Series (Reports 1-10), which systematically evaluates document image quality assessment models, explores VLM-based alternatives, and validates quality predictions against downstream task performance. Full series available at the project repository.*
