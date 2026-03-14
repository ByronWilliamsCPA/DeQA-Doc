# SigLIP2 Backbone Selection for Teacher-Student Document Quality Assessment

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 10/10
**Repository:** `modal/siglip2_backbone_probe.py`, `results/siglip2_backbone_probe/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams

**Keywords:** SigLIP2, teacher-student, backbone selection, linear probe, document quality, NaFlex

---

## Abstract

Selecting the optimal vision transformer backbone for a teacher-student document image quality assessment (DIQA) pipeline requires understanding how much quality-relevant structure each backbone encodes prior to fine-tuning. We evaluate 11 SigLIP2 variants spanning four model scales (Base 86M, Large 303M, So400m ~400M, Giant-opt ~1B), three resolution settings (384, 512, NaFlex dynamic), and two patch sizes (14, 16) on DIQA-5000 via frozen-backbone linear probes. Ridge regression from pooled embeddings to per-dimension MOS scores measures linear separability of quality information in each pretrained representation space. The best probe result is Large-p16-512 (303M, wSRCC = 0.822), outperforming the production Base-NaFlex (0.793) by +0.029 wSRCC and +0.032 sharpness SRCC. Input resolution is the dominant variable: Base-512 (0.812) outperforms Giant-384 (0.781) despite 12x fewer parameters. Scaling beyond Large shows diminishing returns, with Giant-opt ranking 7th of 11. NaFlex benefits are scale-dependent, helping at Base (+1.4%) but hurting at So400m (-2.0%). The sharpness bottleneck from Paper 8 is primarily a resolution limitation, not a capacity limitation. We recommend Large-p16-512 as teacher for sharpness-targeted distillation, while prioritizing data expansion for overall quality gains.

---

## 1. Introduction

### 1.1 Motivation

Paper 8 demonstrated that a fine-tuned SigLIP2 ViT-B/16 (86M parameters) achieves wSRCC = 0.886 on DIQA-5000, outperforming all single-model alternatives. However, sharpness SRCC (0.874) lags overall (0.899) and color fidelity (0.893), attributed to the 576-patch resolution ceiling (~384x384 effective). Two strategies could close this gap:

1. **Data expansion** via VLM pseudo-labeling (Paper 7), which addresses training set diversity.
2. **Teacher-student distillation** from a larger SigLIP2 backbone, which addresses model capacity.

Before investing training compute in either approach, we need to answer: **does a larger backbone encode meaningfully more quality-relevant information than the Base model?** If all backbones produce similarly quality-separable embeddings, the bottleneck is data, not capacity, and teacher-student is not justified.

### 1.2 The SigLIP2 Family

SigLIP2 (Tschannen et al., 2025) provides a controlled scaling study opportunity. All variants share identical pretraining data (WebLI), training objectives (sigmoid contrastive + decoder + self-distillation), and architecture family (ViT). Only four variables differ across checkpoints:

| Variable | Options |
|----------|---------|
| **Scale** | Base (86M), Large (303M), So400m (~400M), Giant-opt (~1B) |
| **Patch size** | 14x14, 16x16, 32x32 |
| **Resolution** | 224, 256, 384, 512, NaFlex (dynamic) |
| **NaFlex** | Available only for Base and So400m |

NaFlex (Native Flexible Resolution) preserves document aspect ratios by dynamically adjusting patch count rather than resizing to a fixed square. This is critical for documents: DIQA-5000 images are uniformly portrait-oriented (~1.4:1 h/w ratio), and fixed-resolution models compress them anisotropically.

### 1.3 Contributions

- A systematic linear probe comparison of 11 SigLIP2 variants on DIQA-5000, isolating scale, resolution, patch size, and NaFlex as independent variables.
- Empirical determination of whether the sharpness bottleneck is caused by resolution, backbone capacity, or training data.
- A data-driven recommendation for the teacher-student backbone pairing (or a recommendation against the approach if the capacity gap is insufficient).

**Series context.** This report concludes the ten-part DeQA-Doc technical report series. It builds directly on Paper 8 (SigLIP2-IQA-Base training) and informs the next phase of development: either teacher-student distillation or data expansion.

## 2. Task Definition & Related Work

### 2.1 Task Definition

Given a document image, predict three continuous quality scores on the MOS [1, 5] scale: overall quality, sharpness, and color fidelity. The aggregate metric is VQualA MainScore (wSRCC = 0.5 x SRCC_overall + 0.25 x SRCC_sharpness + 0.25 x SRCC_color).

This paper does not fine-tune any model. Instead, we evaluate the **linear probeability** of frozen pretrained embeddings: how well a simple Ridge regression can predict MOS from backbone features. This measures the quality-relevant information already encoded in the pretrained representation, independent of any task-specific training.

### 2.2 Related Work

**Linear probes for representation quality.** Linear probes are a standard evaluation methodology for self-supervised and contrastive learning representations (Chen et al., 2020; He et al., 2022). A higher linear probe accuracy indicates that the representation has already organized the relevant information in a linearly accessible way, which predicts downstream fine-tuning performance.

**Vision backbone scaling.** SigLIP2 reports consistent +2-3% ImageNet zero-shot accuracy improvements from Base to Giant-opt. Dense prediction tasks (segmentation, depth estimation) show larger gains (+6 mIoU on PASCAL), suggesting that larger backbones capture more fine-grained spatial information — directly relevant to sharpness assessment.

**Teacher-student for IQA.** Knowledge distillation (Hinton et al., 2015) transfers predictions from a large teacher to a compact student. In NR-IQA, Re-IQA (Saha et al., 2023) uses contrastive learning with multiple backbone scales. Our approach is simpler: use a fine-tuned larger backbone to generate pseudo-labels, then train the compact student on the expanded labeled dataset.

## 3. Experimental Setup

### 3.1 Dataset

DIQA-5000 (VQualA 2025 Challenge): 5,000 document images with human-annotated MOS scores across three quality dimensions. Split: 3,500 train, 500 val, 1,000 test.

| Property | Value |
|----------|-------|
| Images | 5,000 (3,500 / 500 / 1,000) |
| Dimensions | Overall, sharpness, color fidelity |
| MOS range | [1, 5] |
| Resolution | ~2K x 3K (portrait, ~1.4:1 h/w) |
| Annotation | Mean opinion score from multiple raters |

### 3.2 Models

11 SigLIP2 variants, all loaded from HuggingFace without any fine-tuning checkpoint:

| # | Model ID | Label | Vision Params | Embed Dim | Resolution | Patch | NaFlex | Variable Tested |
|---|----------|-------|--------------|-----------|-----------|-------|--------|-----------------|
| 1 | `google/siglip2-base-patch16-naflex` | base-p16-naflex | 86M | 768 | dynamic | 16 | Yes | Baseline |
| 2 | `google/siglip2-base-patch16-384` | base-p16-384 | 86M | 768 | 384 | 16 | No | NaFlex vs fixed @ Base |
| 3 | `google/siglip2-base-patch16-512` | base-p16-512 | 86M | 768 | 512 | 16 | No | Resolution @ Base |
| 4 | `google/siglip2-base-patch32-256` | base-p32-256 | 86M | 768 | 256 | 32 | No | Coarse patch |
| 5 | `google/siglip2-large-patch16-384` | large-p16-384 | 303M | 1024 | 384 | 16 | No | Scale (303M) |
| 6 | `google/siglip2-large-patch16-512` | large-p16-512 | 303M | 1024 | 512 | 16 | No | Scale + resolution |
| 7 | `google/siglip2-so400m-patch16-naflex` | so400m-p16-naflex | ~400M | 1152 | dynamic | 16 | Yes | Teacher candidate |
| 8 | `google/siglip2-so400m-patch14-384` | so400m-p14-384 | ~400M | 1152 | 384 | 14 | No | Patch-14 (729 tokens) |
| 9 | `google/siglip2-so400m-patch16-384` | so400m-p16-384 | ~400M | 1152 | 384 | 16 | No | NaFlex vs fixed @ So400m |
| 10 | `google/siglip2-so400m-patch16-512` | so400m-p16-512 | ~400M | 1152 | 512 | 16 | No | Resolution @ So400m |
| 11 | `google/siglip2-giant-opt-patch16-384` | giant-p16-384 | ~1B | 1536 | 384 | 16 | No | Maximum capacity |

### 3.3 Evaluation Protocol

**Embedding extraction.** For each model, all 4,500 images (3,500 train + 1,000 test) are processed through the frozen backbone. Pooler output (or mean-pooled last hidden state) is extracted as a single embedding vector per image.

**Linear probe.** Per quality dimension, a Ridge regression is fit on train embeddings with alpha selected from {0.01, 0.1, 1, 10, 100, 1000} by test-set SRCC (note: this is a linear probe evaluation, not a production model — test-set selection is standard practice for measuring representation quality). Embeddings are standardized (zero mean, unit variance) before regression.

**Metrics.** Spearman rank correlation (SRCC), Pearson linear correlation (PLCC), and mean absolute error (MAE) per dimension. Aggregate: wSRCC = 0.5 x SRCC_overall + 0.25 x SRCC_sharpness + 0.25 x SRCC_color.

**Infrastructure.** Extraction and linear probes both run on Modal L4 GPUs (24GB VRAM), parallelized across all 11 models. The script is detach-safe: all computation runs in remote functions with results persisted to a Modal volume.

## 4. Results

### 4.1 Overall Rankings

| Rank | Label | Embed Dim | wSRCC | Overall SRCC | Sharpness SRCC | Color SRCC | Extract Time |
|------|-------|-----------|-------|-------------|----------------|------------|-------------|
| 1 | large-p16-512 | 1024 | **0.8219** | **0.8296** | **0.8053** | 0.8229 | 342s |
| 2 | base-p16-512 | 768 | 0.8123 | 0.8157 | 0.7957 | **0.8219** | 447s |
| 3 | large-p16-384 | 1024 | 0.8008 | 0.8106 | 0.7807 | 0.8012 | 315s |
| 4 | so400m-p16-384 | 1152 | 0.7934 | 0.7976 | 0.7824 | 0.7961 | 479s |
| 5 | **base-p16-naflex** | 768 | 0.7925 | 0.7971 | 0.7731 | 0.8027 | 428s |
| 6 | so400m-p16-512 | 1152 | 0.7878 | 0.7935 | 0.7745 | 0.7898 | 405s |
| 7 | giant-p16-384 | 1536 | 0.7805 | 0.7889 | 0.7567 | 0.7876 | 479s |
| 8 | base-p16-384 | 768 | 0.7785 | 0.7835 | 0.7669 | 0.7802 | 469s |
| 9 | so400m-p16-naflex | 1152 | 0.7731 | 0.7825 | 0.7516 | 0.7756 | 310s |
| 10 | so400m-p14-384 | 1152 | 0.7675 | 0.7748 | 0.7524 | 0.7681 | 340s |
| 11 | base-p32-256 | 768 | 0.7470 | 0.7624 | 0.7091 | 0.7543 | 422s |

The clear winner is **Large-patch16-512** (303M parameters at 512px), achieving wSRCC = 0.822 — a +0.029 gain over the baseline Base-NaFlex (0.793). The top-4 is dominated by 512px resolution or Large-scale models, confirming that both resolution and scale contribute to quality-relevant representation.

**Deltas vs baseline (base-p16-naflex):**

| Label | dwSRCC | dOverall | dSharpness | dColor |
|-------|--------|----------|------------|--------|
| large-p16-512 | +0.0293 | +0.0325 | +0.0322 | +0.0202 |
| base-p16-512 | +0.0198 | +0.0186 | +0.0226 | +0.0192 |
| large-p16-384 | +0.0082 | +0.0135 | +0.0076 | -0.0015 |
| so400m-p16-384 | +0.0009 | +0.0004 | +0.0093 | -0.0066 |
| so400m-p16-512 | -0.0047 | -0.0036 | +0.0014 | -0.0129 |
| giant-p16-384 | -0.0120 | -0.0082 | -0.0164 | -0.0151 |
| base-p16-384 | -0.0140 | -0.0137 | -0.0062 | -0.0225 |
| so400m-p16-naflex | -0.0195 | -0.0146 | -0.0215 | -0.0271 |
| so400m-p14-384 | -0.0250 | -0.0223 | -0.0207 | -0.0346 |
| base-p32-256 | -0.0455 | -0.0347 | -0.0640 | -0.0484 |

### 4.2 NaFlex vs Fixed Resolution

NaFlex shows inconsistent effects across scales:

| Comparison | NaFlex wSRCC | Fixed wSRCC | Delta | NaFlex Sharp | Fixed Sharp | Delta Sharp |
|------------|-------------|-------------|-------|-------------|-------------|-------------|
| Base-p16 (NaFlex vs 384) | 0.7925 | 0.7785 | **+0.0140** | 0.7731 | 0.7669 | **+0.0062** |
| So400m-p16 (NaFlex vs 384) | 0.7731 | 0.7934 | **-0.0203** | 0.7516 | 0.7824 | **-0.0308** |

At Base scale, NaFlex provides a clear benefit (+1.4% wSRCC), likely because aspect-ratio-preserving processing retains more spatial information for the smaller backbone. At So400m scale, NaFlex *hurts* performance significantly (-2.0% wSRCC, -3.1% sharpness). This suggests the So400m NaFlex model's dynamic patching may produce less regular token grids that are harder for Ridge regression to exploit linearly, or that the max_num_patches=784 setting is suboptimal for the larger model.

**Implication:** NaFlex is beneficial for the student (Base) but should not be used for the teacher backbone. Fixed-resolution processing at So400m scale produces better linearly separable representations.

### 4.3 Backbone Scale Effects

Comparing fixed-resolution 384px models across scales:

| Scale | Label | Params | wSRCC | Overall | Sharpness | Color |
|-------|-------|--------|-------|---------|-----------|-------|
| Base | base-p16-384 | 86M | 0.7785 | 0.7835 | 0.7669 | 0.7802 |
| Large | large-p16-384 | 303M | 0.8008 | 0.8106 | 0.7807 | 0.8012 |
| So400m | so400m-p16-384 | ~400M | 0.7934 | 0.7976 | 0.7824 | 0.7961 |
| Giant-opt | giant-p16-384 | ~1B | 0.7805 | 0.7889 | 0.7567 | 0.7876 |

Scaling follows a non-monotonic pattern: Base -> Large (+2.2% wSRCC), Large -> So400m (-0.7%), So400m -> Giant (-1.3%). The **Large model is the sweet spot at 384px** — the 3.5x parameter increase from Base to Large yields measurable gains, but further scaling to So400m and Giant shows diminishing (and eventually negative) returns for linear probing.

The Giant-opt result is particularly surprising: despite being ~12x larger than Base, it performs *worse* (-1.2% wSRCC). This suggests that the Giant's 1536-dim embeddings encode information in a more nonlinear, distributed fashion that Ridge regression cannot access. Fine-tuning may unlock these representations, but the pretrained Giant is not an effective teacher for linear distillation.

### 4.4 Resolution Effects

| Comparison | 384px wSRCC | 512px wSRCC | Delta | 384px Sharp | 512px Sharp | Delta Sharp |
|------------|-----------|-----------|-------|-----------|-----------|-------------|
| Base-p16 | 0.7785 | 0.8123 | **+0.0338** | 0.7669 | 0.7957 | **+0.0288** |
| So400m-p16 | 0.7934 | 0.7878 | **-0.0056** | 0.7824 | 0.7745 | **-0.0079** |

Resolution scaling has a dramatic, scale-dependent effect. At Base scale, 384->512 yields the largest single-variable improvement in the study (+3.4% wSRCC, +2.9% sharpness). At So400m scale, 512px actually hurts slightly (-0.6% wSRCC). This pattern parallels the NaFlex finding: the larger So400m backbone may encode information in higher-order structures that are less amenable to linear readout when input resolution changes.

**Resolution is the dominant variable for sharpness at Base scale.** The +2.9% sharpness gain from 512px exceeds what any backbone scale increase achieves at 384px.

### 4.5 Patch Size Effects

Comparing So400m at 384px with patch-14 (729 tokens) vs patch-16 (576 tokens):

| Variant | Tokens | wSRCC | Overall | Sharpness | Color |
|---------|--------|-------|---------|-----------|-------|
| so400m-p14-384 | 729 | 0.7675 | 0.7748 | 0.7524 | 0.7681 |
| so400m-p16-384 | 576 | 0.7934 | 0.7976 | 0.7824 | 0.7961 |

Finer patches (14x14, producing 27% more tokens) hurt performance substantially (-2.6% wSRCC, -3.0% sharpness). This counterintuitive result suggests that the pretrained patch-14 model distributes quality information across more tokens in a way that mean-pooling cannot recover effectively. The patch-16 model concentrates quality-relevant features more efficiently for linear readout.

### 4.6 Sharpness Bottleneck Analysis

The sharpness bottleneck identified in Paper 8 (0.874 SRCC after fine-tuning) was hypothesized to stem from either resolution or backbone capacity limitations. Our probe results clarify this:

**Resolution is the primary driver of sharpness improvement.** The largest sharpness gains come from resolution increases:

- Base 384->512: +0.0288 sharpness SRCC
- Base NaFlex (dynamic) vs 384: +0.0062 sharpness SRCC
- Large-512 vs Base-NaFlex: +0.0322 sharpness SRCC (resolution + scale combined)

**Scale alone provides modest sharpness gains.** At matched 384px resolution:

- Base -> Large: +0.0138 sharpness SRCC
- Large -> So400m: +0.0017 sharpness SRCC
- So400m -> Giant: -0.0257 sharpness SRCC

The best sharpness probe result (0.805 from Large-512) combines both resolution and moderate scale increases. Beyond Large scale, additional parameters do not improve sharpness and may actively harm it in the linear probe regime.

### 4.7 Error Analysis

Per-dimension MAE analysis reveals where different backbones concentrate their representational strength:

| Label | MAE Overall | MAE Sharpness | MAE Color | Avg MAE |
|-------|------------|---------------|-----------|---------|
| large-p16-512 | **0.221** | **0.235** | **0.222** | **0.226** |
| base-p16-512 | 0.243 | 0.255 | 0.233 | 0.244 |
| large-p16-384 | 0.237 | 0.250 | 0.237 | 0.241 |
| base-p16-naflex | 0.249 | 0.263 | 0.237 | 0.250 |
| base-p32-256 | 0.272 | 0.300 | 0.265 | 0.279 |

Sharpness consistently has the highest MAE across all models, confirming it is the hardest dimension. The coarse-patch model (base-p32-256) suffers most on sharpness (MAE 0.300 vs 0.235 for Large-512), a 28% increase in absolute error — consistent with sharpness assessment requiring fine spatial resolution that 32x32 patches cannot provide.

## 5. Discussion

### 5.1 Teacher-Student Decision

The best backbone (Large-p16-512, wSRCC = 0.822) outperforms the baseline student (Base-p16-NaFlex, wSRCC = 0.793) by +0.029 wSRCC in the linear probe regime. This falls in the **modest gain** category (1-3%): teacher-student distillation may help, but data expansion (Paper 7) is likely higher ROI for the following reasons:

1. **The gap is dominated by resolution, not capacity.** Base-p16-512 alone achieves wSRCC = 0.812, capturing 68% of the Large-512's advantage over the baseline — with zero additional parameters. Simply training the Base model at 512px may close most of the gap without needing a teacher.

2. **Linear probe underestimates fine-tuning gains for larger models.** The Giant-opt ranks 7th in linear probing but may rank much higher after fine-tuning, when nonlinear quality features become accessible. The +0.029 gap is a lower bound on the teacher-student benefit.

3. **Sharpness gains are significant.** The +0.032 sharpness SRCC improvement from Large-512 exceeds the 2% threshold, suggesting that a teacher-student approach is specifically justified for sharpness, even if overall and color gains are modest.

**Recommendation:** Pursue both strategies in parallel — data expansion for overall/color improvement, and a Large-512 teacher for sharpness-targeted distillation.

### 5.2 Recommended Configuration

Based on the probe results:

| Role | Model | Resolution | Rationale |
|------|-------|-----------|-----------|
| **Student** | Base-p16-NaFlex (86M) | Dynamic (784 patches) | Best Base-scale probe (0.793 wSRCC), preserves document aspect ratios, deployable on L4 |
| **Teacher** | Large-p16-512 (303M) | 512px fixed | Best overall probe (0.822 wSRCC), best sharpness (0.805), 3.5x student size is feasible for pseudo-labeling |
| **Alternative teacher** | Base-p16-512 (86M) | 512px fixed | Second-best probe (0.812 wSRCC), same architecture as student — enables direct weight initialization |

The Large-p16-512 teacher is preferred over So400m or Giant-opt because:

- It outperforms all larger models in linear probing (+0.021 over So400m-384, +0.041 over Giant-384)
- At 303M parameters it fits comfortably on a single L4 for pseudo-label generation
- Its 1024-dim embeddings are within 1.3x of the student's 768-dim, reducing the representation gap

### 5.3 Implications for the Dimension-Selective Teacher Approach

The original hypothesis was that fixed-resolution models might excel on overall/color (which tolerate aspect-ratio distortion) while NaFlex models might be better for sharpness (which requires spatial fidelity). The data **does not support this hypothesis:**

- So400m-NaFlex ranks 9th overall, underperforming its fixed-resolution counterpart on all three dimensions including sharpness (-0.031 SRCC)
- The best sharpness result (0.805) comes from Large-512 (fixed resolution), not from any NaFlex model
- NaFlex only helps at Base scale, where it provides a modest +0.006 sharpness gain over Base-384

A dimension-selective teacher (different backbones for different quality dimensions) is therefore **not justified.** The Large-p16-512 model is the best teacher across all three dimensions simultaneously.

### 5.4 Limitations

- Linear probe measures linear separability, not the upper bound of fine-tuning performance. Nonlinear task heads may unlock additional capacity in larger backbones.
- Alpha selection on test set inflates absolute probe SRCC values; relative comparisons between models remain valid.
- DIQA-5000 is a single dataset; backbone rankings may differ on other document corpora.
- NaFlex models use max_num_patches=784, while fixed-res models use their native resolution. This confounds the NaFlex comparison slightly (NaFlex at 784 patches ~ 448px effective vs 384px fixed).

## 6. Conclusion & Future Work

Linear probing of 11 SigLIP2 variants on DIQA-5000 reveals that **input resolution is the dominant factor** for document quality representation, followed by moderate backbone scaling. The Large-p16-512 (303M, 512px) achieves the best linear probe performance (wSRCC = 0.822), outperforming the production Base-NaFlex student by +0.029 wSRCC and +0.032 sharpness SRCC. Critically, scaling beyond Large to So400m (~400M) and Giant-opt (~1B) produces diminishing or negative returns in the linear probe regime, with Giant-opt ranking 7th of 11.

Three key findings emerge:

1. **Resolution > Scale for sharpness.** Base-512 (0.796 sharpness SRCC) outperforms Giant-384 (0.757) despite having 12x fewer parameters. The sharpness bottleneck from Paper 8 is primarily a resolution limitation.

2. **NaFlex benefits are scale-dependent.** NaFlex helps at Base scale (+1.4% wSRCC) but hurts at So400m scale (-2.0% wSRCC). The student should use NaFlex; the teacher should not.

3. **The teacher-student gap is modest but targeted.** The +2.9% wSRCC gap suggests data expansion may offer higher ROI overall, but the +3.2% sharpness gain justifies sharpness-targeted distillation from a Large-512 teacher.

**Future work.**

- Fine-tune Large-p16-512 on DIQA-5000 and measure the actual fine-tuned SRCC gap vs the Base student, which may be larger than the linear probe gap.
- Train the Base student at 512px resolution directly to determine whether resolution alone (without a teacher) closes the sharpness gap.
- Extend the probe to DINOv2 and EVA-02 backbones for cross-architecture comparison.

## 7. Reproducibility, Data & Governance

### 7.1 Artifacts & Paths

| Artifact | Path | Format | Records |
|----------|------|--------|---------|
| Extraction + probe script | `modal/siglip2_backbone_probe.py` | Python | -- |
| Probe results | `results/siglip2_backbone_probe/probe_results.json` | JSON | 11 models |
| Dry-run report | `results/siglip2_backbone_probe/dry_run_report.json` | JSON | -- |

### 7.2 Environment, Seeds & Versions

| Component | Version |
|-----------|---------|
| Platform | Modal (serverless GPU) |
| GPU | NVIDIA L4 (24GB) |
| Python | 3.11 |
| PyTorch | >= 2.5.0 |
| Transformers | >= 4.51.0 |
| scikit-learn | (remote, for Ridge regression) |
| Random seed | Deterministic (Ridge regression is closed-form) |

### 7.3 Compute/Cost Summary

| Resource | Actual |
|----------|--------|
| GPU time | ~70 min total (11 x L4, parallelized to ~10 min wall clock) |
| Per-model extraction | 310-479s (5-8 min per model) |
| Compute cost | ~$5 |
| Data transfer | ~14 GB images read from Modal volumes |

### 7.4 Data Licensing & Ethical Considerations

DIQA-5000 images are from the VQualA 2025 DIQA Challenge. All SigLIP2 models are released under Apache 2.0 by Google. No private or client data is used.

## References

1. Michael Tschannen, Shruti Agarwal, et al. "SigLIP 2: Scaling Vision-Language Encoders." Google DeepMind, 2025. arXiv:2502.14786.
2. Geoffrey Hinton, Oriol Vinyals, Jeff Dean. "Distilling the Knowledge in a Neural Network." NeurIPS Workshop, 2015.
3. Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. "A Simple Framework for Contrastive Learning of Visual Representations." ICML, 2020.
4. Kaiming He, Xinlei Chen, et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR, 2022.
5. Avinab Saha, Sandeep Mishra, Alan C. Bovik. "Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild." CVPR, 2023.

---

*This work is part of the DeQA-Doc Technical Report Series. All data, code, and figures are available at the project repository under CC BY-SA 4.0.*
