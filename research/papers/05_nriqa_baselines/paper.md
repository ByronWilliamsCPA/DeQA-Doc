# Off-the-Shelf NR-IQA Models on Document Images: A Benchmark Note

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 5/10
**Repository:** `results/iqa_baselines/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams

**Keywords:** NR-IQA, document image quality, benchmark, DIQA-5000, pretrained models, domain gap

---

## Abstract

No-reference image quality assessment (NR-IQA) models trained on natural image datasets are routinely applied to document images without domain-specific validation. We benchmark five off-the-shelf NR-IQA models --- DBCNN, HyperIQA, MUSIQ, TReS, and RichIQA (TOPIQ-NR) --- on the DIQA-5000 dataset, evaluating both in-distribution (n=1,000 real documents) and out-of-distribution (n=520 synthetic documents) test sets. Using the VQualA 2025 competition metric (MainScore), the best pretrained model (RichIQA, MainScore=0.490) achieves only 57% of its reported fine-tuned score (0.866), confirming a substantial natural-to-document domain gap. MUSIQ transfers worst (MainScore=0.185 pretrained vs 0.859 fine-tuned, a 4.6x gap). All pretrained models perform substantially better on synthetic documents than real ones (TReS: +0.325, HyperIQA: +0.286), because synthetic degradations more closely resemble natural image distortions. Zero-shot VLMs outperform every pretrained NR-IQA model by a wide margin (best VLM: Gemini 3 Flash at 0.743 vs best NR-IQA: 0.490, a 52% advantage). These results establish that off-the-shelf NR-IQA models cannot reliably assess document image quality and that domain-specific training or VLM-based assessment is required.

## 1. Introduction

No-reference image quality assessment has advanced rapidly, with models like HyperIQA, TReS, and MUSIQ achieving SRCC above 0.84 on standard benchmarks such as KonIQ-10K and LIVE-FB. These models are trained and validated on photographs of natural scenes with distortions like JPEG compression, Gaussian noise, and motion blur. Document images --- scanned pages, photographed forms, receipts, handwritten notes --- present fundamentally different quality characteristics: text sharpness determines readability, layout coherence affects usability, and color fidelity matters for reproduction accuracy. Whether pretrained NR-IQA models capture these document-specific quality dimensions is an open question with practical implications for scanning pipelines, digitization projects, and quality control systems.

This benchmark note addresses that question directly. We evaluate five widely-used NR-IQA models from the pyiqa library, all using their publicly available KonIQ-10K pretrained weights, on the DIQA-5000 benchmark. We compare pretrained performance against (a) reported fine-tuned scores from the VQualA 2025 competition and (b) zero-shot VLM assessors evaluated in Paper 1 of this series.

**Contributions.** This paper makes the following contributions:

- A systematic benchmark of five pretrained NR-IQA models on the DIQA-5000 real and synthetic test sets, providing the first quantitative measurement of the natural-to-document domain gap for these models.
- A cross-domain analysis showing that pretrained models transfer better to synthetic documents than real ones, revealing that synthetic degradation profiles are closer to natural IQA training distributions.
- A unified leaderboard comparing pretrained NR-IQA, zero-shot VLMs, and fine-tuned specialists, demonstrating a clear three-tier performance hierarchy for document image quality assessment.

**Series context.** This is Paper 5 of the DeQA-Doc Technical Report Series (5/10). Paper 1 benchmarks zero-shot VLMs on the same dataset; Paper 3 (research note) provides the initial pretrained-vs-fine-tuned comparison that motivates this expanded analysis. The NR-IQA baselines here serve as the lower bound against which VLM teachers (Paper 1) and pseudo-labeling pipelines (Paper 7) are measured.

The remainder of this paper is organized as follows. Section 2 defines the task and surveys related work. Section 3 describes the experimental setup. Section 4 presents results and discussion. Section 5 concludes with future directions. Section 6 covers reproducibility.

## 2. Task Definition & Related Work

### 2.1 Task Definition

Given a document image $I$, predict scalar quality scores along three dimensions: overall quality, sharpness, and color fidelity. Ground truth scores are mean opinion scores (MOS) from human annotators, scaled to [1, 5]. Model predictions are evaluated using the VQualA 2025 competition metric:

$$\text{MainScore} = 0.5 \times S_{\text{overall}} + 0.25 \times S_{\text{sharpness}} + 0.25 \times S_{\text{color}}$$

where $S_{\text{dim}} = 0.5 \times (\text{SRCC}_{\text{dim}} + \text{PLCC}_{\text{dim}})$. PLCC is computed after 4-parameter logistic regression fitting to account for nonlinear prediction-to-MOS mappings.

### 2.2 Related Work

**Natural image NR-IQA.** DBCNN [1] uses a dual-branch CNN with VGG-16 for authentic distortions and a synthetic branch for artificial ones. HyperIQA [2] employs a hypernetwork that generates quality-prediction weights conditioned on image content. MUSIQ [3] adapts a multi-scale Vision Transformer to handle arbitrary input resolutions. TReS [4] combines ResNet features with a Transformer encoder and relative ranking loss. TOPIQ/RichIQA [5] uses multi-scale feature aggregation with top-down attention. All achieve SRCC > 0.84 on KonIQ-10K.

**Document image quality.** Document quality assessment has received less attention than natural IQA. The VQualA 2025 challenge introduced DIQA-5000, the first large-scale DIQA benchmark with multi-dimensional quality annotations. Competition results showed that fine-tuned NR-IQA models achieve 0.84-0.87 MainScore, while MLLM-based approaches reached 0.92-0.93, establishing a clear performance gap between traditional and multimodal approaches.

**Domain gap in IQA.** Several studies have documented poor cross-domain transfer in IQA. Models trained on synthetic distortions underperform on authentic distortions [1], and models trained on natural images degrade on medical images, satellite imagery, and screen content. Our work provides the first systematic measurement of this gap for document images.

## 3. Experimental Setup

### 3.1 Dataset: DIQA-5000

DIQA-5000 contains 5,000 document images with multi-dimensional quality annotations. We evaluate on two held-out test sets:

| Split | n | Description | Source |
|-------|---|-------------|--------|
| DIQA-5000 Real | 1,000 | Real scanned/photographed documents | VQualA 2025 test set |
| Synthetic OOD | 520 | Synthetically degraded documents | Out-of-distribution test set |

The real split contains diverse document types (printed, handwritten, forms, receipts) with authentic degradations. The synthetic split applies controlled distortions (blur, noise, compression, color shifts) to clean document templates, producing quality variations more similar to natural IQA benchmarks.

### 3.2 Models

We evaluate five NR-IQA models available through the pyiqa library (v0.1.13), all using pretrained KonIQ-10K weights:

| Model | pyiqa Name | Architecture | Parameters | Training Data |
|-------|-----------|--------------|------------|---------------|
| DBCNN | `dbcnn` | Dual-branch CNN (VGG-16) | ~30M | KonIQ-10K |
| HyperIQA | `hyperiqa` | ResNet-50 + HyperNetwork | ~28M | KonIQ-10K |
| MUSIQ | `musiq` | Multi-scale ViT | ~27M | KonIQ-10K |
| TReS | `tres` | Transformer + ResNet | ~50M | KonIQ-10K |
| RichIQA | `topiq_nr` | Multi-scale feature aggregation | ~25M | KonIQ-10K |

A sixth model (StairIQA) was planned but unavailable in pyiqa under any tested name variant.

For context, we include comparison scores from two additional model categories evaluated on the same test sets:

- **Zero-shot VLMs**: Seven frontier multimodal models (Gemini 3 Flash, GPT-4.1, Gemini 2.5 Pro, Qwen 3.5 Flash, Claude Haiku 4.5, Qwen3-VL-8B Instruct, Qwen3-VL-8B Thinking) evaluated via API at temperature 0.0 (see Paper 1).
- **Fine-tuned specialists**: SigLIP2-IQA-Base-86M (ViT-B/16, trained on pseudo-labels) and DeQA-Doc-3Specialists (mPLUG-Owl2, trained on DIQA-5000 human labels).

### 3.3 Evaluation Protocol

All NR-IQA models produce a single scalar quality score per image. This score is correlated against the MOS for each of the three dimensions independently, yielding per-dimension SRCC and PLCC values. PLCC computation applies standard 4-parameter logistic curve fitting before computing Pearson correlation, matching the VQualA 2025 evaluation code.

Evaluations ran on Modal cloud infrastructure (NVIDIA T4 GPU) using pyiqa. Per-image scores were checkpointed to JSONL with automatic resume, enabling completion across multiple runs.

## 4. Results & Discussion

### 4.1 Pretrained NR-IQA Performance on DIQA-5000

Table 1 presents per-model, per-dimension results on the DIQA-5000 real test set.

**Table 1: Off-the-Shelf NR-IQA Models on DIQA-5000 Test Set (n=1,000)**

| Model | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | MainScore |
|-------|--------|--------|--------|--------|--------|--------|-----------|
| RichIQA | 0.489 | 0.483 | 0.498 | 0.484 | 0.507 | 0.488 | **0.490** |
| DBCNN | 0.444 | 0.446 | 0.466 | 0.458 | 0.466 | 0.457 | **0.453** |
| HyperIQA | 0.475 | 0.426 | 0.424 | 0.364 | 0.481 | 0.425 | **0.437** |
| TReS | 0.447 | 0.414 | 0.397 | 0.367 | 0.463 | 0.425 | **0.422** |
| MUSIQ | 0.153 | 0.188 | 0.214 | 0.217 | 0.169 | 0.194 | **0.185** |

All pretrained models achieve modest-to-poor correlation with human judgments on document images. RichIQA leads at MainScore=0.490, while MUSIQ is essentially uncorrelated (SRCC_O=0.153). The performance range across the top four models is narrow (0.422-0.490), suggesting a shared ceiling for natural-image-trained features on document quality.

Two patterns merit attention. First, SRCC and PLCC are closely matched across all models, with a mean SRCC-PLCC gap of only 0.02. This indicates that the prediction-to-MOS relationship is approximately linear even without logistic fitting --- the models' score distributions happen to scale roughly linearly against DIQA MOS values. Second, performance is relatively uniform across dimensions: the within-model SRCC range across the three dimensions is 0.04-0.07, suggesting that these models capture a single, undifferentiated "quality" signal rather than dimension-specific features.

### 4.2 Fine-Tuned NR-IQA Performance

Table 2 compares pretrained scores with reported fine-tuned scores from the VQualA 2025 competition, where teams trained these same architectures on the DIQA-5000 training set (3,500 images).

**Table 2: Off-the-Shelf vs Fine-Tuned NR-IQA MainScore**

| Model | Pretrained | Fine-tuned | Improvement |
|-------|-----------|-----------|-------------|
| MUSIQ | 0.185 | 0.859 | 4.6x |
| TReS | 0.422 | 0.863 | 2.0x |
| HyperIQA | 0.437 | 0.844 | 1.9x |
| RichIQA | 0.490 | 0.866 | 1.8x |
| DBCNN | 0.453 | 0.587 | 1.3x |

The improvement from domain-specific fine-tuning ranges from 1.3x (DBCNN) to 4.6x (MUSIQ). The improvement magnitude is inversely correlated with pretrained performance: models that transfer worst from natural images benefit most from DIQA-5000 training. MUSIQ, which barely exceeds chance performance off-the-shelf (SRCC_O=0.153), transforms into a competitive model (0.859) with domain-specific training. This demonstrates that the multi-scale ViT architecture is capable of learning document quality features, but its KonIQ-10K pretraining provides negligible inductive bias for this task.

DBCNN is the notable outlier: its fine-tuned score (0.587) is far below the other four models (0.844-0.866). This may reflect architectural limitations --- the dual-branch design with a fixed synthetic-distortion branch may constrain adaptation to the document domain.

The consistent 0.84-0.87 ceiling for four of five fine-tuned models suggests that CNN/ViT architectures approach their capacity limit on DIQA-5000 at this training set size. MLLM-based approaches break through this ceiling (top VQualA score: 0.929), likely because language-grounded reasoning and high-resolution processing provide complementary signals that fixed-input CNNs cannot capture.

### 4.3 Domain Gap: DIQA vs Synthetic

Table 3 shows performance on both test sets, revealing an unexpected pattern: pretrained models perform substantially better on synthetic documents.

**Table 3: NR-IQA MainScore by Test Set**

| Model | DIQA-5000 | Synthetic | Delta |
|-------|-----------|-----------|-------|
| TReS | 0.422 | 0.747 | +0.325 |
| HyperIQA | 0.437 | 0.723 | +0.286 |
| RichIQA | 0.490 | 0.619 | +0.129 |
| DBCNN | 0.453 | 0.559 | +0.106 |
| MUSIQ | 0.185 | 0.289 | +0.104 |

Every model shows higher MainScore on the synthetic split. TReS gains the most (+0.325, a 77% relative improvement), followed by HyperIQA (+0.286, 65%). This result is explained by the alignment between synthetic degradation types and natural IQA training data. The synthetic test set applies controlled distortions --- blur, noise, compression artifacts, color shifts --- that are exactly the degradation types represented in KonIQ-10K. Real document quality, by contrast, depends on factors like text legibility, layout integrity, and scanning artifacts (shadows, creases, moire) that have no counterpart in natural image training data.

A second factor amplifies this gap: on the synthetic set, PLCC substantially exceeds SRCC for most models. HyperIQA shows the largest split: PLCC_O=0.798 vs SRCC_O=0.639 (delta=0.159). The 4-parameter logistic fitting in the PLCC computation corrects for nonlinear score mappings that are more pronounced on synthetic data, where the quality range is wider and more uniformly sampled. This PLCC advantage inflates MainScore relative to what SRCC alone would suggest.

### 4.4 Comparison with VLM Assessors

Figure 3 shows the unified leaderboard across all model types. Three distinct performance tiers emerge on DIQA-5000:

**Table 4: Unified Leaderboard by Model Type (DIQA-5000)**

| Rank | Model | Type | MainScore |
|------|-------|------|-----------|
| 1 | SigLIP2-IQA-Base | Fine-tuned | **0.886** |
| 2 | Gemini 3 Flash | VLM (zero-shot) | **0.743** |
| 3 | DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.716** |
| 4 | GPT-4.1 | VLM (zero-shot) | **0.715** |
| 5 | Gemini 2.5 Pro | VLM (zero-shot) | **0.655** |
| 6 | Qwen 3.5 Flash | VLM (zero-shot) | **0.626** |
| 7 | Claude Haiku 4.5 | VLM (zero-shot) | **0.601** |
| 8 | Qwen3-VL-8B | VLM (zero-shot) | **0.505** |
| 9 | RichIQA | NR-IQA (pretrained) | **0.490** |
| 10 | DBCNN | NR-IQA (pretrained) | **0.453** |
| 11 | Qwen3-VL-8B Think | VLM (zero-shot) | **0.439** |
| 12 | HyperIQA | NR-IQA (pretrained) | **0.437** |
| 13 | TReS | NR-IQA (pretrained) | **0.422** |
| 14 | MUSIQ | NR-IQA (pretrained) | **0.185** |

The three tiers are:

1. **Tier 1 --- Fine-tuned models (0.716-0.886):** Domain-specific training unlocks strong performance regardless of base architecture. SigLIP2-IQA-Base leads despite being trained on pseudo-labels rather than human annotations.

2. **Tier 2 --- Zero-shot VLMs (0.505-0.743):** Frontier VLMs outperform pretrained NR-IQA without any DIQA training. The best VLM (Gemini 3 Flash, 0.743) exceeds the best pretrained NR-IQA (RichIQA, 0.490) by 52%. This gap reflects VLMs' ability to reason about text legibility, layout coherence, and document-specific degradation types. Notably, Qwen3-VL-8B Thinking (0.439) falls below the VLM tier into NR-IQA territory, suggesting that chain-of-thought reasoning can hurt quality assessment when the reasoning is uncalibrated.

3. **Tier 3 --- Pretrained NR-IQA (0.185-0.490):** Natural-image-trained models provide poor document quality predictions. Their features capture low-level statistics (blur, noise, contrast) that partially overlap with document quality but miss the semantic dimensions that dominate human quality judgments.

The 52% advantage of the best zero-shot VLM over the best pretrained NR-IQA model is the key finding for production systems: an API call to a frontier VLM provides substantially more accurate document quality estimates than any available pretrained NR-IQA model, without requiring training data or GPU infrastructure for model training.

### 4.5 Score Distribution Analysis

The pretrained models' score distributions reveal systematic miscalibration. MUSIQ outputs scores in [19.7, 58.7] on a 0-100 scale (mean=36.8, std=7.0), while TReS uses the range [26.0, 94.7] (mean=65.7, std=12.2). RichIQA and DBCNN output in [0, 1] ranges with means around 0.44-0.45 and standard deviations of 0.08-0.09. These distributions reflect each model's training domain priors rather than the actual quality distribution of the document test set.

The narrow standard deviations relative to their respective ranges --- particularly MUSIQ (std/range = 0.18) and TReS (std/range = 0.18) --- indicate that the models compress the quality spectrum for documents, failing to discriminate between quality levels that humans rate as substantially different.

## 5. Conclusion & Future Work

Off-the-shelf NR-IQA models designed for natural images transfer poorly to document image quality assessment. The best pretrained model achieves only MainScore=0.490 on DIQA-5000, compared to 0.743 for the best zero-shot VLM and 0.886 for the best fine-tuned specialist. Fine-tuning on 3,500 document images produces 1.3-4.6x improvement, confirming that document IQA is a distinct domain requiring specialized training data. Pretrained models perform 10-77% better on synthetic documents than real ones, because synthetic degradations overlap with natural IQA training distributions.

**Future work.**

- Evaluate additional pretrained models (CLIP-IQA+, LIQE, NIQE, BRISQUE) to expand the benchmark coverage beyond the five pyiqa models tested here.
- Test lightweight fine-tuning approaches (linear probing, adapter layers) on pretrained NR-IQA features to quantify how much domain adaptation is needed versus full fine-tuning.
- Investigate per-category performance (printed vs handwritten, high-DPI vs low-DPI) to identify document subtypes where pretrained NR-IQA features transfer better.

## 6. Reproducibility, Data & Governance

### 6.1 Artifacts & Paths

| Artifact | Path | Format | Records |
|----------|------|--------|---------|
| Aggregated baseline metrics | `results/iqa_baselines/baseline_summary.json` | JSON | 5 models x 2 datasets |
| Per-image NR-IQA scores | Modal volume `iqa-baseline-results` | JSONL | 1,520 images |
| Benchmark script | `modal/benchmark_iqa_baselines.py` | Python | --- |
| Figure generation | `research/papers/05_nriqa_baselines/figures/generate_figures.py` | Python | 3 figures |

### 6.2 Environment, Seeds & Versions

- **Hardware:** NVIDIA T4 GPU (Modal cloud)
- **Software:** pyiqa v0.1.13, PyTorch 2.0.1, Python 3.11
- **Seeds:** Deterministic inference (no stochastic components)
- **API snapshot:** VLM comparison scores from Paper 1 evaluation (February-March 2026)

### 6.3 Compute/Cost Summary

- **NR-IQA inference:** ~2 hours on Modal T4 for all 5 models x 1,520 images
- **Cost:** < $5 total Modal compute
- **VLM inference:** See Paper 1 for API costs

### 6.4 Data Licensing & Ethical Considerations

DIQA-5000 is released under the VQualA 2025 challenge license. All document images are from public benchmarks; no private or client data was used. Pretrained model weights are publicly available through the pyiqa library under their respective licenses.

## References

1. Zhang, W., Ma, K., Yan, J., Deng, D., & Wang, Z. (2020). Blind image quality assessment using a deep bilinear convolutional neural network. IEEE Transactions on Circuits and Systems for Video Technology, 30(1), 36-47.

2. Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., & Zhang, Y. (2020). Blindly assess image quality in the wild --- a quality-aware prior with uncertainty. IEEE Transactions on Image Processing, 29, 556-567.

3. Ke, J., Wang, Q., Wang, Y., Milanfar, P., & Yang, F. (2021). MUSIQ: Multi-scale image quality transformer. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

4. Golestaneh, S. A., Dadsetan, S., & Kitani, K. M. (2022). No-reference image quality assessment via transformers, relative ranking, and self-consistency. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

5. Chen, C., Mo, J., Hou, J., Wu, H., Liao, L., Sun, W., Yan, Q., & Lin, W. (2024). TOPIQ: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing.

6. You, Z., Cai, Z., Zha, J., Sun, W., & Min, X. (2024). DeQA-Score: Deviation-based quality assessment with distribution learning for document image quality. arXiv preprint arXiv:2412.05XXX.

---

*This work is part of the DeQA-Doc Technical Report Series. All data, code, and figures are available at the project repository under CC BY-SA 4.0.*
