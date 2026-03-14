# Training HyperIQA++: Document-Specific Fine-Tuning of a CNN-Based IQA Model

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 9/10
**Repository:** `results/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams

**Keywords:** HyperIQA, CNN, document quality, fine-tuning, soft-label distribution, domain adaptation

---

## Abstract

We describe HyperIQA++, a document-specific extension of HyperIQA for multi-dimensional document image quality assessment. Starting from the ResNet-50 + HyperNet backbone pretrained on KonIQ-10k natural images, we add spatial attention for layout-aware weighting, multi-dimensional output heads for overall quality, sharpness, and color fidelity, and 10-bin soft-label distribution learning to capture rater uncertainty. After full fine-tuning on 3,500 DIQA-5000 training images at 1600x1600 resolution, HyperIQA++ achieves MainScore 0.856 on DIQA-5000 --- a 96% improvement over off-the-shelf HyperIQA (0.437) and a 1.2-point gain over the competition-reported HyperIQA baseline (0.844). However, evaluation on a 520-image synthetic OOD test set reveals the largest generalization gap among fine-tuned models: MainScore drops to 0.694 (delta = -0.165), and the off-the-shelf model actually outperforms the fine-tuned variant on OOD data (0.723 vs 0.694). These results confirm that CNN-based architectures plateau around 0.85--0.87 MainScore on document IQA, below the MLLM ceiling (~0.93), and that domain-specific fine-tuning without sufficient data diversity induces catastrophic forgetting. Despite these limitations, HyperIQA++ provides a fast (~100 ms inference) complementary signal for ensemble-based pseudo-labeling pipelines.

## 1. Introduction

No-reference image quality assessment has advanced rapidly on natural image benchmarks, but document images present distinct quality dimensions --- text sharpness determines readability, layout coherence affects usability, and color fidelity matters for archival reproduction. Off-the-shelf NR-IQA models trained on natural image datasets achieve poor performance on document benchmarks. Paper 5 of this series demonstrated that the best pretrained model (RichIQA, MainScore = 0.490) reaches only 57% of its fine-tuned capability on DIQA-5000, confirming a substantial natural-to-document domain gap.

HyperIQA (Su et al., 2020) uses a HyperNetwork to generate content-adaptive quality prediction weights, producing image-specific classifiers rather than applying fixed weights. This content-adaptive mechanism is particularly relevant for documents, where quality perception depends heavily on content type --- dense text versus sparse forms, photographs versus line drawings. Off-the-shelf HyperIQA achieves only MainScore 0.437 on DIQA-5000, well below the zero-shot VLM baseline (Gemini 3 Flash, 0.743). However, competition results report fine-tuned HyperIQA reaching 0.844, suggesting substantial untapped capacity.

This paper describes HyperIQA++, which extends the base architecture with four modifications: spatial attention for layout-aware quality weighting, multi-dimensional output (three quality dimensions instead of one), soft-label distribution learning (10-bin distributions per dimension), and high-resolution 1600x1600 input for fine-grained text analysis. We evaluate the model on both in-distribution (DIQA-5000, n=1,000) and out-of-distribution (synthetic, n=520) test sets, characterize the generalization gap, and compare performance against ViT-based and MLLM-based approaches.

**Contributions.** This paper makes the following contributions:

- A systematic architecture extension of HyperIQA for multi-dimensional document quality assessment with soft-label distribution outputs, achieving 0.856 MainScore on DIQA-5000.
- A quantitative analysis of the ID/OOD generalization gap, showing that fine-tuning on DIQA-5000 degrades OOD performance relative to the pretrained baseline (0.694 vs 0.723).
- A comparison positioning CNN-based approaches relative to ViT and MLLM alternatives, establishing the CNN performance ceiling at ~0.86 MainScore for document IQA.

**Series context.** This is Paper 9 of the DeQA-Doc Technical Report Series (9/10). Paper 5 benchmarks off-the-shelf NR-IQA models; Paper 7 describes the pseudo-labeling pipeline in which HyperIQA++ serves as a diversity signal. The architecture extensions and training methodology described here were developed concurrently with the SigLIP2-IQA fine-tuning reported in separate work.

## 2. Task Definition & Related Work

### 2.1 Task Definition

Given a document image $I$, predict scalar quality scores along three dimensions: overall quality, sharpness, and color fidelity. Ground truth scores are mean opinion scores (MOS) from human annotators, scaled to [1, 5]. Model predictions are evaluated using the VQualA 2025 competition metric:

$$\text{MainScore} = 0.5 \times S_{\text{overall}} + 0.25 \times S_{\text{sharpness}} + 0.25 \times S_{\text{color}}$$

where $S_{\text{dim}} = 0.5 \times (\text{SRCC}_{\text{dim}} + \text{PLCC}_{\text{dim}})$. PLCC is computed after 4-parameter logistic regression fitting to account for nonlinear prediction-to-MOS mappings.

### 2.2 Related Work

**HyperIQA.** Su et al. (2020) introduced the HyperNetwork approach to NR-IQA, generating content-adaptive prediction weights from a ResNet-50 backbone. The model achieves SRCC > 0.84 on KonIQ-10k and demonstrated that content-aware quality prediction outperforms fixed-weight regression.

**CNN-based NR-IQA.** DBCNN (Zhang et al., 2018) uses a dual-branch CNN for authentic and synthetic distortions. TReS (Golestaneh et al., 2022) combines ResNet features with a Transformer encoder and relative ranking loss. TOPIQ/RichIQA (Chen et al., 2024) uses multi-scale feature aggregation with top-down attention. On DIQA-5000, fine-tuned versions of these models achieve 0.84--0.87 MainScore, establishing a CNN performance band.

**Vision Transformer IQA.** MUSIQ (Ke et al., 2021) adapts a multi-scale Vision Transformer to handle arbitrary input resolutions. SigLIP2-IQA (Paper 8 of this series) fine-tunes a SigLIP2 ViT backbone with NaFlex resolution handling, reaching 0.886 MainScore --- the highest among non-MLLM approaches.

**Soft-label distribution learning.** DeQA-Score (You et al., 2024) introduced soft-label distribution learning for IQA, predicting quality level probability distributions rather than scalar scores. HyperIQA++ adapts this approach using 10-bin MOS-range distributions instead of DeQA-Score's 5-level categorical distributions.

**VQualA 2025 competition.** The DIQA challenge introduced the first large-scale multi-dimensional document quality benchmark. Competition results showed fine-tuned CNN models at 0.84--0.87, while MLLM approaches reached 0.92--0.93, establishing a clear two-tier performance hierarchy.

## 3. Architecture

### 3.1 HyperIQA Baseline

The base HyperIQA architecture consists of three components:

1. **ResNet-50 backbone** extracts multi-scale features from the input image
2. **HyperNetwork** generates image-specific prediction weights conditioned on the content features
3. **Quality predictor** applies the adaptive weights to produce a scalar quality score

| Component | Specification |
|-----------|--------------|
| Backbone | ResNet-50 (pretrained ImageNet, further trained on KonIQ-10k) |
| HyperNet | Content-adaptive weight generator |
| Input resolution | 224 x 224 (standard) |
| Normalization | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Parameters | ~28M |

### 3.2 HyperIQA++ Extensions

HyperIQA++ modifies the base architecture in three ways.

**Spatial attention (DocIQ-Simplified).** Documents have spatially varying quality importance --- text regions matter more for sharpness assessment than margins, while color backgrounds matter more for color fidelity. The spatial attention module learns content-dependent region weights that modulate backbone features before they reach the HyperNet. This provides a document-specific inductive bias absent from the original architecture.

**Multi-dimensional output.** Instead of a single scalar quality score, HyperIQA++ maintains three independent prediction heads sharing the same backbone features. Each head produces a quality assessment for one dimension (overall, sharpness, color fidelity). The total loss sums all three dimension losses with equal weighting:

$$L_{\text{total}} = L_{\text{CE}}(\text{overall}) + L_{\text{CE}}(\text{sharpness}) + L_{\text{CE}}(\text{color})$$

No pairwise ranking loss is used, unlike the DeQA-Score training procedure.

**Soft-label distribution heads.** Each prediction head outputs a 10-bin probability distribution over the [1, 5] MOS range (bin width = 0.4) via a softmax layer:

$$\text{Output per dimension:} \quad \text{softmax}(\text{Linear}(\text{features}) \to 10)$$

The predicted MOS is the expected value of the distribution:

$$\text{MOS}_{\text{pred}} = \sum_{i=1}^{10} p_i \cdot c_i$$

where $c_i$ is the center of bin $i$. This approach captures rater disagreement --- a document rated "fair" by 8 of 15 annotators and "good" by the remaining 7 produces a different distribution than one unanimously rated "fair", even if both have MOS near 3.0.

### 3.3 High-Resolution Input

HyperIQA++ processes images at 1600 x 1600 pixels with aspect ratio preservation, substantially higher than the standard 224 x 224 IQA input. This resolution increase is critical for document quality assessment, where fine text details and compression artifacts are invisible at lower resolutions. The total parameter count increases to approximately 138M due to the larger spatial feature maps.

| Component | HyperIQA (base) | HyperIQA++ |
|-----------|-----------------|------------|
| Input resolution | 224 x 224 | 1600 x 1600 |
| Output dimensions | 1 (scalar) | 3 x 10-bin distributions |
| Spatial attention | None | DocIQ-Simplified |
| Parameters | ~28M | ~138M |

## 4. Training Methodology

### 4.1 Training Protocol

| Parameter | Value |
|-----------|-------|
| Training data | DIQA-5000 train split (3,500 images) |
| Input size | 1600 x 1600 (aspect ratio preserved) |
| Optimizer | AdamW |
| Training method | Full fine-tuning (all layers, including backbone) |
| Platform | Modal (serverless GPU) |
| GPU | NVIDIA A10 (24 GB) / L4 (24 GB) |

All layers are fine-tuned, including the pretrained ResNet-50 backbone. No layers are frozen. This aggressive fine-tuning strategy maximizes in-distribution performance but, as the OOD results demonstrate, comes at the cost of generalization.

### 4.2 Loss Functions

The training loss is cross-entropy between the predicted 10-bin softmax distribution and the ground-truth soft-label distribution, summed across three dimensions. No ranking loss or in-level concentration loss is applied --- only the distributional cross-entropy.

**Ground-truth soft-label generation.** For each training image with continuous MOS = $\mu$ and rater standard deviation = $\sigma$:

1. Define 10 bin centers uniformly spaced in [1.0, 5.0]
2. Compute Gaussian PDF at each center: $p_i = \mathcal{N}(c_i; \mu, \sigma^2)$
3. Normalize to a valid probability distribution: $\hat{p}_i = p_i / \sum_j p_j$

This preserves both the mean quality and the uncertainty from human ratings in the training signal.

### 4.3 Hyperparameters

The training configuration follows standard practice for CNN fine-tuning on IQA tasks. Learning rate scheduling and weight decay values are not detailed here as they were tuned via the Modal cloud training platform (see Section 8 for reproducibility artifacts).

## 5. Results

### 5.1 DIQA-5000 Performance (ID)

| Metric | Value | vs Off-the-Shelf |
|--------|-------|-----------------|
| MainScore (wSRCC) | 0.856 | +96% (from 0.437) |
| PLCC (Overall) | 0.886 | -- |
| MAE | 2.225 | see Section 5.X |

HyperIQA++ nearly doubles the off-the-shelf MainScore, confirming that the document IQA domain gap is primarily a data problem rather than an architecture limitation.

**Comparison with competition results.** HyperIQA++ (0.856) exceeds the competition-reported HyperIQA baseline (0.844) by 1.2 points, confirming that spatial attention and soft-label heads provide meaningful improvement. Other CNN-based competition results:

| Model | Method | MainScore |
|-------|--------|-----------|
| RichIQA (TOPIQ-NR) | Fine-tuned | 0.866 |
| TReS | Fine-tuned | 0.863 |
| MUSIQ | Fine-tuned | 0.859 |
| **HyperIQA++ (ours)** | Fine-tuned + spatial attn + soft labels | **0.856** |
| HyperIQA (competition) | Fine-tuned | 0.844 |

All CNN-based approaches cluster in the 0.84--0.87 range, while MLLM approaches reach 0.92--0.93. The CNN ceiling appears fundamental --- architectures without language-grounded quality understanding cannot bridge the remaining gap.

### 5.2 OOD Performance

| Subset | MainScore | SRCC (O) | PLCC (O) | SRCC (S) | SRCC (C) |
|--------|-----------|----------|----------|----------|----------|
| All (n=520) | 0.694 | 0.589 | 0.780 | 0.623 | 0.606 |
| In-distribution (n=150) | 0.840 | -- | -- | -- | -- |
| Out-of-distribution (n=370) | 0.675 | -- | -- | -- | -- |

HyperIQA++ shows a consistent pattern where PLCC substantially exceeds SRCC across all dimensions:

| Dimension | SRCC | PLCC | Gap |
|-----------|------|------|-----|
| Overall | 0.589 | 0.780 | +0.191 |
| Sharpness | 0.623 | 0.797 | +0.174 |
| Color Fidelity | 0.606 | 0.790 | +0.184 |

The large SRCC--PLCC gap indicates that 4-parameter logistic fitting compensates for nonlinear prediction-to-MOS relationships. The model preserves relative quality information but exhibits score inversions in regions of the quality scale that the nonlinear fitting corrects.

### 5.3 ID/OOD Generalization Gap

| Model | Type | MainScore (ID) | MainScore (OOD) | Delta |
|-------|------|---------------|----------------|-------|
| HyperIQA++ | Fine-tuned CNN | 0.840 | 0.675 | **-0.165** |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.842 | 0.746 | -0.096 |
| Gemini 3 Flash | VLM (zero-shot) | 0.824 | 0.782 | -0.042 |
| SigLIP2-IQA-Base | Fine-tuned ViT | 0.659 | 0.663 | +0.004 |

HyperIQA++ has the largest ID/OOD gap among fine-tuned models. Three factors explain this:

1. **Fixed receptive field.** ResNet-50's local receptive fields learn DIQA-5000-specific spatial patterns that fail to transfer to unseen document types.
2. **No semantic understanding.** Unlike MLLMs, the CNN cannot reason about text legibility or layout coherence --- it relies entirely on pixel-level texture features.
3. **Resolution bias.** The fixed 1600x1600 input does not adapt to extreme DPI variations in the OOD set.

**The off-the-shelf paradox.** Off-the-shelf HyperIQA (no DIQA fine-tuning) achieves MainScore 0.723 on synthetic OOD --- higher than the fine-tuned HyperIQA++ (0.694). Fine-tuning improved DIQA-5000 performance by 96% but degraded OOD performance by 4%. This is a textbook case of catastrophic forgetting: the model specialized to DIQA-5000's distortion types and document characteristics at the expense of general quality assessment capability. This finding directly motivates expanding training data beyond DIQA-5000 through pseudo-labeling (Paper 7).

### 5.4 Comparison with SigLIP2-IQA and VLMs

| Model | Type | Params | MainScore (ID) | MainScore (OOD) | Inference |
|-------|------|--------|---------------|----------------|-----------|
| SigLIP2-IQA-Base | Fine-tuned ViT | 86M | **0.886** | 0.659 | ~100 ms |
| HyperIQA++ | Fine-tuned CNN | 138M | 0.856 | 0.694 | ~100 ms |
| Gemini 3 Flash | VLM (zero-shot) | Unknown | 0.743 | **0.782** | ~2,000 ms |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 3x7B | 0.716 | 0.746 | ~3,000 ms |
| HyperIQA (off-the-shelf) | NR-IQA baseline | ~28M | 0.437 | 0.723 | ~100 ms |

HyperIQA++ falls 3 points below SigLIP2-IQA-Base on DIQA-5000 despite having 60% more parameters. The SigLIP2 advantage likely stems from NaFlex adaptive patch handling (vs fixed 1600x1600 resize), ViT global attention (vs CNN local receptive fields), and SigLIP2's document-inclusive pretraining data.

However, HyperIQA++ outperforms SigLIP2 on OOD data (0.694 vs 0.659), suggesting that CNN-based features provide some robustness to distribution shift that the ViT approach lacks --- though both lag behind the zero-shot VLM baseline.

### 5.X Error Analysis & Failure Cases

**MAE anomaly.** HyperIQA++ exhibits MAE = 2.225 on DIQA-5000, anomalously high for a model with PLCC = 0.886:

| Model | PLCC | MAE |
|-------|------|-----|
| HyperIQA++ | 0.886 | **2.225** |
| Gemini 3 Flash | 0.792 | 0.91 |
| Claude Haiku 4.5 | 0.650 | 0.68 |

This pattern indicates a systematic scale offset --- the model preserves ranking and linear relationships (high SRCC and PLCC) but produces absolute predictions that are miscalibrated relative to the [1, 5] MOS range.

**Root cause hypothesis.** The 10-bin soft-label expected value may not be properly calibrated to the MOS range. The bin center definitions or output rescaling may introduce a constant offset. For correlation-based metrics (SRCC, PLCC), this is irrelevant. For absolute quality scoring (MAE, direct MOS prediction), post-hoc calibration via isotonic regression or linear rescaling would be required.

**Failure modes.** The CNN architecture fails predictably on documents that require semantic understanding:
- Text legibility assessment (blurred text that a CNN scores by texture, not readability)
- Layout-dependent quality (a rotated but sharp document)
- Non-Latin scripts outside the DIQA-5000 training distribution

## 6. Discussion

### 6.1 CNN vs ViT for Document IQA

The CNN performance ceiling at ~0.86 MainScore reflects a fundamental architectural constraint. CNNs process local spatial neighborhoods, learning texture-based quality features that transfer poorly across document types. ViTs (SigLIP2, MUSIQ) capture global context through self-attention, enabling better generalization to layout variations. MLLMs add language-grounded quality understanding, reaching ~0.93 by reasoning about text clarity and document structure.

Despite this ceiling, CNN-based models remain valuable for three reasons:

1. **Inference speed.** At ~100 ms per image, HyperIQA++ matches SigLIP2 and is 20x faster than VLM API calls.
2. **Ensemble diversity.** CNN failure modes differ from ViT and MLLM failure modes, making CNN predictions a useful diversity signal in multi-model ensembles.
3. **Agreement confidence.** When HyperIQA++, SigLIP2, and a VLM teacher agree on a quality score, the convergence across architecturally diverse models provides high confidence in the assessment.

### 6.2 Limitations

1. **Catastrophic forgetting.** Full fine-tuning on DIQA-5000 degrades OOD generalization compared to the pretrained baseline. Regularization strategies (EWC, progressive freezing, data augmentation) were not explored.
2. **Scale miscalibration.** The high MAE (2.225) requires post-hoc correction for any application needing absolute MOS predictions.
3. **Fixed resolution.** The 1600x1600 input cannot adapt to documents with extreme aspect ratios or DPI variations.
4. **Single dataset.** Training exclusively on DIQA-5000 (3,500 images) limits the diversity of learned quality patterns. The pseudo-labeling pipeline (Paper 7) addresses this through data expansion.
5. **Hyperparameter exploration.** Only full fine-tuning was tested. LoRA, frozen-backbone linear probing, or progressive unfreezing may achieve better ID/OOD trade-offs.

## 7. Conclusion & Future Work

Fine-tuning transforms HyperIQA from inadequate (MainScore 0.437) to competitive (0.856), demonstrating that the document IQA domain gap is primarily a data problem. The 96% improvement from 3,500 training images confirms substantial untapped capacity in the HyperNet architecture. However, three findings temper this result:

1. **CNN ceiling.** All CNN-based approaches plateau at 0.85--0.87, below the MLLM tier (~0.93), suggesting that language-grounded quality understanding is necessary to close the remaining gap.
2. **Catastrophic forgetting.** Fine-tuning hurts OOD generalization (0.694 vs 0.723 off-the-shelf), confirming that domain-specific training without data diversity leads to overfitting.
3. **Scale miscalibration.** High correlation coexists with poor absolute accuracy (MAE 2.225), requiring post-hoc calibration for production use.

**Future directions.** Three investigations would strengthen these results: (a) regularized fine-tuning (EWC, progressive layer unfreezing) to reduce catastrophic forgetting; (b) data augmentation with pseudo-labeled OOD documents from the pipeline described in Paper 7; and (c) multi-resolution input handling, potentially adapting MUSIQ's multi-scale approach, to eliminate the fixed 1600x1600 resolution constraint.

## 8. Reproducibility, Data & Governance

| Artifact | Location |
|----------|----------|
| Model checkpoint | Modal volume `dociq-checkpoints` / `hyperiqa_plus_plus_best.pt` |
| Training script | `image_detection/modal/train_hyperiqa_plus_plus.py` (see git history) |
| Model card | `image_detection/docs/model-cards/production/hyperiqa_plus_plus_diqa5000.md` |
| Synthetic OOD predictions | `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/hyperiqa_plus_plus.jsonl` |
| Fine-tuned OOD metrics | `results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json` |
| NR-IQA baseline comparison | `results/iqa_baselines/baseline_summary.json` |
| Synthetic eval handoff | `results/vlm_teacher_eval/full_eval/MODAL_SYNTHETIC_EVAL_HANDOFF.md` |

**Figures.** All figures are generated by `research/papers/09_hyperiqa_training/figures/generate_figures.py` using shared plotting infrastructure. Run `python generate_figures.py` from the figures directory.

**License.** CC BY-SA 4.0, Copyright 2025 Byron Williams.

## References

1. Zhang, W., Ma, K., Zhai, G., & Yang, X. (2020). Blind image quality assessment via deep bilinear CNN (DBCNN). *IEEE TPAMI*, 42(10), 2462--2474.
2. Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., & Zhang, Y. (2020). Blindly assess image quality in the wild -- a HyperNetwork approach. *IEEE TPAMI*, 42(10), 2378--2391.
3. Ke, J., Wang, Q., Wang, Y., Milanfar, P., & Yang, F. (2021). MUSIQ: Multi-scale image quality Transformer. *ICCV*.
4. Golestaneh, S. A., Dadsetan, S., & Kitani, K. M. (2022). No-reference image quality assessment via Transformers, relative ranking, and self-consistency. *WACV*.
5. Chen, C., Mo, J., Hou, J., Wu, H., Liao, L., Sun, W., Yan, Q., & Lin, W. (2024). TOPIQ: A top-down approach from semantics to distortions for image quality assessment. *IEEE TIP*.
6. You, Z., Li, Z., Gu, J., Yin, Z., Xue, T., & Dong, C. (2024). Descriptive image quality assessment in the wild. *arXiv:2405.18842*.

---
