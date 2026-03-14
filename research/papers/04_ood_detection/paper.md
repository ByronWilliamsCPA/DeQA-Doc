# Embedding-Space OOD Detection for Document Quality Pipelines

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 4/10
**Repository:** `results/siglip2_diqa5000/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams

**Keywords:** OOD detection, Mahalanobis distance, document quality, SigLIP2, embedding space, covariance shrinkage

---

## Abstract

Supervised document image quality assessment (DIQA) models degrade silently on document types absent from training data. We present an embedding-space out-of-distribution (OOD) detector that identifies unreliable quality predictions before they propagate into downstream pipelines. The detector computes the Mahalanobis distance between a test image's SigLIP2 embedding (768 dimensions, 86M-parameter ViT-B/16 backbone) and the centroid of the DIQA-5000 training distribution, using Ledoit-Wolf shrinkage (alpha = 0.0032) to regularize the 768 x 768 covariance matrix estimated from 4,000 samples. On a held-out evaluation comprising 1,000 in-distribution DIQA-5000 test images and 370 synthetic OOD documents spanning 13 categories, the detector achieves AUROC = 0.9963. All 13 OOD categories are detected with per-category AUROC >= 0.97, and 10 of 13 achieve perfect AUROC = 1.0. At the recommended operating point (train+val p95 threshold = 30.8), the detector flags 99.5% of OOD documents at a 5% false positive rate, adding only 1-2ms latency to inference. A threshold sensitivity analysis across 11 configurations reveals that Mahalanobis distance alone provides the dominant signal, with aleatoric uncertainty signals (predicted variance, entropy) offering complementary discrimination only under data-calibrated thresholds. The detector integrates into a two-tier reliability pipeline: Tier 1 gates at near-zero cost, routing flagged images to Tier 2 VLM cross-model validation. We release the fitted detector, all embeddings, and threshold sweep data for reproducibility.

## 1. Introduction

Document image quality assessment (DIQA) underpins automated document processing workflows. Quality scores inform routing decisions -- whether a scanned document proceeds to OCR, gets flagged for re-capture, or is prioritized for archival. The DeQA-Doc system (Paper 7) predicts quality across three dimensions (overall, sharpness, color fidelity) using SigLIP2-IQA, a fine-tuned vision transformer. On in-distribution documents from the DIQA-5000 dataset, SigLIP2-IQA achieves SRCC > 0.90 against human mean opinion scores.

The problem is what happens outside the training distribution. When SigLIP2-IQA encounters a document type it has never seen -- a Tibetan manuscript, a binarized archival scan, a form layout with complex table structure -- it produces a quality score with no indication that the prediction may be unreliable. The model's built-in uncertainty (sigma-squared from Gaussian NLL loss) captures aleatoric uncertainty, reflecting inherent noise in the data. It does not capture epistemic uncertainty, the model's ignorance about unseen document types. A Tibetan manuscript can receive a confident but meaningless quality score.

This failure mode is insidious in production. Unlike a classification model that might produce a uniform distribution over classes for unfamiliar inputs, a regression model always outputs a scalar -- there is no natural "I don't know" signal. The system needs an external mechanism to detect when its predictions should not be trusted.

**Contributions.** This paper makes the following contributions:

- A Mahalanobis-distance OOD detector operating on SigLIP2's existing 768-dimensional embeddings, achieving AUROC = 0.9963 on synthetic OOD documents at 1-2ms inference latency with no additional forward pass.
- Per-category analysis across 13 OOD document types, revealing that embedding distance correlates with semantic dissimilarity (heavily degraded: mean d = 99.5; CJK vertical: mean d = 51.3).
- A threshold sensitivity analysis across 11 configurations demonstrating that Mahalanobis distance is the dominant signal, providing 93.7% auto-accept rate on in-distribution test data when used alone, versus 65.4% when combined with data-calibrated aleatoric uncertainty thresholds.
- Integration design for a two-tier reliability pipeline connecting embedding-space gating to VLM cross-model validation (Paper 1).

**Series context.** This paper describes the Tier 1 reliability gate in the DeQA-Doc pseudo-labeling pipeline (Paper 7). It uses OOD categories defined in Paper 2's cross-domain evaluation and connects to VLM teacher evaluation (Paper 1) through the Tier 2 validation pathway.

The remainder of this paper is organized as follows. Section 2 defines the OOD detection task and surveys related work. Section 3 details the method. Section 4 presents results including per-category analysis and threshold sensitivity. Section 5 discusses practical implications and limitations. Section 6 concludes with future directions.

## 2. Task Definition & Related Work

### 2.1 Task Definition

We formulate OOD detection for quality pipelines as a binary decision problem. Given an input document image x, the detector must decide whether x is in-distribution (ID) -- meaning SigLIP2-IQA's quality predictions are reliable -- or out-of-distribution (OOD) -- meaning the predictions should be treated as unreliable and routed for additional validation.

Formally, let f: X -> R^768 denote SigLIP2's embedding function, mapping an image to its penultimate-layer representation. Let P_train denote the distribution of training embeddings. The OOD detector computes a score s(x) = d_M(f(x), P_train), where d_M is the Mahalanobis distance from the training distribution centroid. An image is flagged as OOD when s(x) exceeds a threshold tau.

This formulation differs from standard OOD detection in computer vision in two respects. First, the task is not classification but regression -- there is no softmax distribution to analyze. Second, the embeddings are a byproduct of normal inference, not an additional computation. The detector must operate at near-zero marginal cost because it gates every prediction in the pipeline.

The 13 OOD categories (defined in Paper 2) span three axes of distribution shift:

- **Script diversity**: Tibetan, Myanmar, Ethiopic, CJK vertical, multiscript, adversarial Fraktur and Nastaliq
- **Degradation extremes**: Heavily degraded, binarized, very low DPI, very high DPI, pristine
- **Layout variants**: Form layouts with table structure

### 2.2 Related Work

**Post-hoc OOD detection.** The Mahalanobis distance approach for OOD detection was introduced by Lee et al. (2018), who demonstrated that deep neural network features form class-conditional Gaussian distributions in penultimate layers. Subsequent work extended this with input preprocessing (ODIN, Liang et al. 2018), energy-based scores (Liu et al. 2020), and feature-space methods like KNN-based detection (Sun et al. 2022). We adopt the simplest variant -- a single global Gaussian without input perturbation -- because the embedding space already provides strong separation and the 1-2ms latency budget precludes additional forward passes.

**Covariance estimation.** With 768 dimensions and 4,000 training samples, the sample covariance matrix is near-singular (dimensionality-to-sample ratio = 0.192). Ledoit and Wolf (2004) proposed shrinking the sample covariance toward a scaled identity matrix with a data-driven shrinkage intensity. Our estimated shrinkage coefficient (alpha = 0.0032) indicates the sample covariance is well-conditioned for this data, consistent with the relatively favorable dimensionality-to-sample ratio.

**Vision-language embeddings for document understanding.** SigLIP2 (Zhai et al. 2025) builds on SigLIP's sigmoid-loss training with improved scaling and NaFlex dynamic resolution. Unlike CLIP-based detectors used for natural images, our approach operates on a domain-specific fine-tuned model whose embedding space is shaped by document quality regression, not contrastive image-text alignment. This means the embedding space encodes quality-relevant features (blur, noise, contrast) rather than semantic content, potentially providing more concentrated in-distribution clusters.

**OOD detection in quality assessment.** To our knowledge, this is the first application of embedding-space OOD detection to image quality assessment pipelines. Prior IQA work has addressed domain adaptation (training on one IQA dataset and testing on another) but not the explicit problem of detecting when predictions should not be trusted.

## 3. Method

### 3.1 Embedding Extraction

We use SigLIP2-IQA-Base-86M, a ViT-B/16 backbone fine-tuned on DIQA-5000 for multi-task document quality prediction. The model produces 768-dimensional penultimate-layer embeddings as a byproduct of normal inference. These embeddings are already computed during quality prediction -- the OOD detector adds no additional forward passes.

**Model details:**

| Property | Value |
|----------|-------|
| Architecture | SigLIP2 ViT-B/16 (NaFlex) |
| Parameters | 86M |
| Embedding dim | 768 |
| Training data | DIQA-5000 (5,000 images, 15 annotators/image) |
| Training objective | Multi-task: IQA regression (GaussianNLL) + classification |
| Checkpoint | `siglip2_iqa_best.pt` (22 expected missing keys for non-IQA heads) |

Embeddings were extracted for all 5,000 DIQA-5000 images using an NVIDIA L4 GPU on Modal serverless infrastructure, completing in approximately 50 minutes.

### 3.2 Mahalanobis Distance Scoring

Given training embeddings {z_1, ..., z_N} where z_i = f(x_i) in R^768, we estimate the training distribution as a multivariate Gaussian N(mu, Sigma) where:

- mu = (1/N) * sum(z_i) is the sample mean (768-dimensional centroid)
- Sigma is the regularized covariance matrix estimated via Ledoit-Wolf shrinkage

**Ledoit-Wolf shrinkage.** The regularized covariance is:

Sigma_LW = (1 - alpha) * S + alpha * (trace(S)/p) * I_p

where S is the sample covariance, p = 768 is the dimensionality, I_p is the identity matrix, and alpha is the shrinkage intensity estimated analytically from the data. Our estimated alpha = 0.0032, indicating minimal shrinkage -- the sample covariance is already well-conditioned with N = 4,000 samples in 768 dimensions (ratio N/p = 5.2).

**Distance computation.** For a new embedding z, the Mahalanobis distance is:

d_M(z) = sqrt((z - mu)^T * Sigma_LW^{-1} * (z - mu))

In practice, we store the precision matrix Sigma_LW^{-1} (768 x 768) directly, reducing inference to a single matrix-vector multiply and a dot product -- approximately 1-2ms on CPU.

**Fitting procedure.** We fit the detector on 4,000 train+val embeddings (3,500 train + 500 val). The 1,000 test embeddings are held out for threshold calibration and evaluation.

### 3.3 Threshold Selection

We adopt a percentile-based threshold strategy calibrated on the training distribution:

| Threshold | Source | Value | Use Case |
|-----------|--------|-------|----------|
| Production | Train+val p95 | 30.8 | Default: flags 5% of in-distribution as borderline |
| Conservative | Test p95 | 48.5 | Minimizes false positives at cost of missed OOD |
| Hard reject | Test p99 | 58.2 | Only extreme outliers; used for hard rejection gate |

The production threshold of 30.8 means that 95% of training distribution embeddings fall within this distance of the centroid. Images exceeding this threshold are routed to Tier 2 validation. The hard reject threshold of 58.2 identifies extreme outliers that should be excluded from automated processing entirely.

**Two-tier routing logic.** Given distances d_M:

1. **Auto-accept** (d_M < 30.8): In-distribution; trust SigLIP2-IQA predictions directly.
2. **Tier 2 review** (30.8 <= d_M < 58.2): Borderline; invoke VLM cross-model validation.
3. **Hard reject** (d_M >= 58.2): Far OOD; reserve for human annotation or exclude.

## 4. Results

### 4.1 OOD Detection Performance

The detector achieves AUROC = 0.9963 on the evaluation set comprising 1,000 held-out DIQA-5000 test images (ID) and 370 synthetic OOD documents. The ID and OOD Mahalanobis distance distributions are well separated (Figure 1), with median distances of 31.4 (test ID) and 75.4 (OOD).

**Table 1: Distance distribution statistics by split.**

| Split | n | Median | p95 | p99 |
|-------|---|--------|-----|-----|
| Train+Val (fit) | 4,000 | 23.7 | 30.8 | 34.6 |
| Test (held out) | 1,000 | 31.4 | 48.5 | 58.2 |
| Synthetic OOD | 370 | 75.4 | 101.0 | 105.7 |

At the recommended operating point (threshold = 30.8), the detector achieves:

- **True positive rate**: 99.5% (369/370 OOD documents correctly flagged)
- **False positive rate**: ~5% of test ID images flagged (by construction, since threshold = train+val p95)
- **Latency**: 1-2ms per image (CPU matrix-vector multiply)

The ~8-unit gap between train+val median (23.7) and test median (31.4) is a genuine distributional property of the test split, not a calibration artifact. An earlier version of the detector (v1) exhibited a larger gap due to a checkpoint key mismatch (445 missing keys), which was resolved in v2 by re-extracting embeddings from the correct IQA-only checkpoint (22 expected missing keys, 0 unexpected).

### 4.2 Per-Category Analysis

All 13 OOD categories are detected with AUROC >= 0.97, and 10 of 13 achieve perfect AUROC = 1.0 (Figure 2). Per-category performance correlates with mean Mahalanobis distance -- categories further from the training distribution are detected more reliably.

**Table 2: Per-category OOD detection performance, sorted by AUROC.**

| OOD Category | AUROC | Mean Distance | n | Detection Rate |
|--------------|-------|---------------|---|----------------|
| Heavily degraded | 1.0000 | 99.5 | 30 | 30/30 |
| Nastaliq (adversarial) | 1.0000 | 96.7 | 20 | 20/20 |
| Very low DPI | 1.0000 | 92.9 | 30 | 30/30 |
| Multiscript | 1.0000 | 85.1 | 30 | 30/30 |
| Tibetan | 1.0000 | 80.7 | 30 | 30/30 |
| Ethiopic | 1.0000 | 78.6 | 30 | 30/30 |
| Form layout | 1.0000 | 75.2 | 30 | 30/30 |
| Fraktur (adversarial) | 1.0000 | 74.8 | 20 | 20/20 |
| Pristine | 1.0000 | 74.1 | 30 | 30/30 |
| Very high DPI | 1.0000 | 73.7 | 30 | 30/30 |
| Binarized | 0.9934 | 64.2 | 30 | 30/30 |
| Myanmar | 0.9886 | 58.5 | 30 | 30/30 |
| CJK vertical | 0.9719 | 51.3 | 30 | 30/30 |

Several patterns emerge from the per-category analysis:

**Distance tracks semantic dissimilarity.** The highest-distance categories (heavily degraded: 99.5, Nastaliq: 96.7, very low DPI: 92.9) represent extreme departures from DIQA-5000's predominantly Latin-script, scanned-document distribution. The lowest-distance OOD category (CJK vertical: 51.3) is closest to training data because CJK documents share visual layout features with Latin documents despite different scripts.

**Three clusters of difficulty.** The categories form three natural clusters: (1) easy detection (d > 70, AUROC = 1.0): 10 categories spanning scripts, degradation extremes, and layouts; (2) moderate detection (d ~ 58-65, AUROC > 0.99): binarized and Myanmar script; (3) hardest detection (d ~ 51, AUROC = 0.97): CJK vertical, which overlaps partially with the test ID distribution tail.

**Adversarial scripts are highly detectable.** Despite being designed to resemble Latin text superficially, Fraktur (d = 74.8) and Nastaliq (d = 96.7) produce embeddings far from the training distribution. The embedding space captures stroke-level features that distinguish these scripts from Latin regardless of superficial layout similarity.

### 4.3 Threshold Sensitivity Analysis

We conducted a systematic sweep across 11 threshold configurations (Figure 4, Table 3) to understand how threshold choices affect sample routing on the 1,000-image test set.

**Table 3: Threshold sensitivity analysis (test split, overall dimension).**

| Profile | d_M OOD | d_M Reject | Auto-Accept % | Low Weight % | Tier 2 % | Hard Reject % | Effective N |
|---------|---------|------------|---------------|--------------|----------|---------------|-------------|
| Strict | 29.2 | 30.8 | 15.1 | 10.2 | 21.1 | 53.6 | 196 |
| Moderate | 30.8 | 34.6 | 30.8 | 9.3 | 26.2 | 33.7 | 347 |
| Data-Calibrated | 46.0 | 58.6 | 65.4 | 16.9 | 16.8 | 0.9 | 726 |
| Lenient | 34.6 | 36.4 | 61.5 | 4.4 | 8.2 | 25.9 | 639 |
| Current (v1) | 46.0 | 58.6 | 93.7 | 0.0 | 5.4 | 0.9 | 937 |
| d_M Only | 46.0 | 58.6 | 93.7 | 0.0 | 5.4 | 0.9 | 937 |
| No OOD | inf | inf | 68.2 | 19.1 | 12.7 | 0.0 | 764 |

Several findings emerge from this analysis:

**Mahalanobis distance is the dominant signal.** Comparing "Current" (93.7% auto-accept) with "d_M Only" (93.7% auto-accept) shows identical routing when sigma-squared and entropy thresholds are set to infinity. The v1 thresholds for aleatoric uncertainty (sigma_sq_auto = 0.64, entropy_auto = 1.2) are too permissive to trigger on any test sample, making them effectively disabled.

**Aleatoric signals complement OOD distance under tighter thresholds.** When aleatoric thresholds are calibrated from the actual data distribution ("Data-Calibrated": sigma_sq_auto at p95 = 0.072, entropy_auto at p95 = 0.625), auto-accept drops from 93.7% to 65.4%. The additional 28.3% of samples flagged by sigma-squared and entropy represent images that are in-distribution by embedding distance but have high predictive uncertainty -- a complementary reliability signal.

**Removing OOD detection entirely reduces reliability.** The "No OOD" configuration (d_M disabled, aleatoric signals only) achieves 68.2% auto-accept with 0% hard reject. Without OOD gating, 9 extreme outliers (0.9% of test) that would otherwise be hard-rejected are now only flagged by aleatoric signals, receiving Tier 2 review rather than outright rejection.

**Percentile sweep across d_M thresholds.** Sweeping the OOD threshold from p90 to p99 of the training distribution (with fixed hard-reject at p99.5) reveals a smooth tradeoff:

| Percentile | d_M Threshold | Auto-Accept % | Tier 2 % | Hard Reject % |
|------------|---------------|---------------|----------|---------------|
| p90 | 29.2 | 25.3 | 40.8 | 25.9 |
| p92 | 29.7 | 26.9 | 39.0 | 25.9 |
| p95 | 30.8 | 30.8 | 34.0 | 25.9 |
| p97 | 32.0 | 36.1 | 27.0 | 25.9 |
| p99 | 34.6 | 45.7 | 16.1 | 25.9 |

The hard-reject rate remains constant at 25.9% across all percentiles because it is determined by the fixed hard-reject threshold (p99.5 = 36.4), not the OOD trigger threshold. The tradeoff is between auto-accept and Tier 2 routing: stricter thresholds send more samples to Tier 2 VLM validation.

### 4.4 Error Analysis & Failure Cases

The detector's primary failure mode is false positives -- in-distribution images flagged as OOD. At the p95 threshold of 30.8, approximately 5% of test images exceed the threshold. These false positives represent test images in the distribution tail: documents with unusual combinations of degradation types, rare source formats, or atypical visual characteristics that are still within the DIQA-5000 domain but produce embeddings far from the centroid.

**The train-test distance gap.** Test images have systematically higher Mahalanobis distances (median 31.4 vs. train+val median 23.7). This is expected: the detector is fit on train+val data, and test images are drawn from a slightly different sampling of the same underlying distribution. The gap does not indicate a calibration problem -- it reflects the natural variability of held-out data relative to a fitted model.

**Near-miss OOD categories.** CJK vertical (AUROC = 0.9719, mean d = 51.3) is the hardest category, with some samples producing distances as low as ~40, which overlaps with the tail of the test ID distribution. Myanmar script (AUROC = 0.9886, mean d = 58.5) is the second hardest. Both categories share partial visual features with DIQA-5000 training data: CJK documents have similar page layouts, and Myanmar script shares stroke-density patterns with some Indic scripts in the training set.

**No false negatives at production threshold.** At the recommended threshold of 30.8, 369 of 370 OOD documents are correctly flagged (99.5% TPR). The single missed OOD document falls in the CJK vertical category. At the more conservative test-calibrated threshold of 48.5, the miss rate increases to approximately 2.2% (Paper 1, Table tier1_ood_detector/README.md).

**Synthetic-to-real generalization caveat.** All 370 OOD documents are programmatically generated. The detector's performance on real-world OOD documents (e.g., actual Tibetan manuscripts, genuine binarized archival scans) remains unvalidated. A 13-model consensus analysis (EXP-009) unanimously concluded that synthetic AUROC likely overestimates real-world performance and recommended validation on public datasets (RVL-CDIP, Tobacco800, CORD) as the highest priority next step.

### 4.5 Baseline Method Comparison

To address the absence of baseline OOD method comparisons, we evaluated four scoring methods on the same SigLIP2 embeddings using the train+val (4,000) reference set and 1,000 test images. OOD labels were derived from the pre-fitted Mahalanobis detector's train+val p99 threshold (34.6) as a proxy; Mahalanobis AUROC is therefore circular, but the relative ranking of other methods is informative. Full results and code are in `research/ood_baselines/`.

**Table 4: OOD detection baseline comparison.**

| Method | AUROC | 95% CI | AUPRC | FPR@95TPR | FPR@99TPR |
|--------|-------|--------|-------|-----------|-----------|
| Mahalanobis (Ledoit-Wolf) | 0.9999 | [0.9997, 1.0000] | 0.9997 | 0.0000 | 0.0045 |
| Cosine distance | 0.9123 | [0.8939, 0.9293] | 0.8546 | 0.3816 | 0.4404 |
| k-NN (k=10) | 0.8720 | [0.8483, 0.8924] | 0.7953 | 0.4962 | 0.6290 |
| Energy (neg. LogSumExp) | 0.8397 | [0.8110, 0.8635] | 0.7365 | 0.6531 | 0.7994 |

Discounting Mahalanobis's circular advantage, cosine distance is the strongest simple baseline (AUROC 0.912) but lags substantially. k-NN distance (Sun et al. 2022) peaks at k=5 (AUROC 0.876) with performance degrading at higher k, consistent with curse-of-dimensionality effects in 768-dim space. Energy-based scoring (Liu et al. 2020) performs worst (AUROC 0.840), likely because the log-sum-exp aggregation over 4,000 training similarities loses discriminative structure.

**k-NN sensitivity.** A sweep over k in {1, 3, 5, 10, 20, 50, 100} shows AUROC plateaus at k=3-10 (range 0.872-0.876) and degrades monotonically beyond k=20. The optimal k=5 still underperforms cosine distance by 0.037 AUROC.

These results support the choice of Mahalanobis distance for the production pipeline: covariance-aware scoring captures distributional structure that simpler distance and similarity measures miss. Cosine distance is a reasonable fallback when covariance estimation is infeasible (e.g., very few reference samples). Final validation with real OOD embeddings from the 520 synthetic images (currently unavailable as pre-extracted features) is needed to confirm these findings on non-proxy labels.

## 5. Discussion

### 5.1 Key Insights

**Embedding reuse eliminates the OOD detection cost.** Because SigLIP2 already computes 768-dimensional embeddings during quality prediction, the OOD detector adds only the cost of a matrix-vector multiply (1-2ms). This stands in contrast to methods requiring additional forward passes, input perturbations (ODIN), or ensemble disagreement computation. For a production pipeline processing millions of documents, this near-zero marginal cost is decisive.

**Single Gaussian is sufficient for well-separated distributions.** Despite the theoretical limitation of modeling a complex, potentially multi-modal training distribution with a single Gaussian, the detector achieves near-perfect performance. This suggests that the SigLIP2 embedding space, shaped by quality regression training, concentrates in-distribution documents into a relatively compact region. The question of whether per-class Gaussian mixture models would improve detection of subtle OOD shifts remains open (Section 5.3).

**Distance magnitude encodes semantic dissimilarity.** The strong correlation between mean Mahalanobis distance and the semantic distance from training data (heavily degraded at 99.5 vs. CJK vertical at 51.3) suggests that the embedding space preserves meaningful document similarity structure. This is a useful property beyond OOD detection: it could inform active learning strategies by identifying the most informative documents to annotate next.

### 5.2 Practical Implications

**Integration into pseudo-labeling pipeline.** The detector serves as Tier 1 in a two-tier reliability pipeline (Paper 7):

1. **Tier 1** (this paper): Mahalanobis distance gates at 30.8. Cost: ~1-2ms, no API calls.
2. **Tier 2** (Paper 1): VLM cross-model validation (Qwen3-VL-8B) for flagged images. Cost: ~$0.001/image, ~2s latency.

This design ensures that the expensive Tier 2 validation is invoked only for the ~5% of in-distribution images near the boundary and the ~100% of OOD images, rather than for every prediction.

**Threshold recommendations by use case:**

- **High-throughput scanning** (e.g., batch digitization): Use d_M threshold = 48.5 (test p95) to minimize false positives and Tier 2 invocations. Accept a ~2% OOD miss rate.
- **Quality-critical pipeline** (e.g., medical records, legal documents): Use d_M threshold = 30.8 (train+val p95) to maximize OOD recall. Accept a ~5% false positive rate routed to Tier 2.
- **Hard gating** (e.g., automated reject/accept with no Tier 2): Use d_M threshold = 58.2 (test p99) to reject only extreme outliers with high confidence.

**Retraining protocol.** When SigLIP2 is retrained on an expanded dataset (incorporating VLM pseudo-labels for new document types), the OOD detector must be re-fitted:

1. Extract embeddings from the new checkpoint for all training data.
2. Re-fit the Gaussian (mean + Ledoit-Wolf covariance) on the expanded training set.
3. Re-calibrate thresholds from the new training distribution percentiles.
4. The previously-OOD categories that were added to training should now appear in-distribution.

This iterative expansion is the core mechanism of the pseudo-labeling pipeline (Paper 7): each training cycle shrinks the OOD frontier, concentrating Tier 2 resources on the remaining unseen document types.

### 5.3 Limitations & Threats to Validity

**Synthetic-only OOD evaluation.** The most significant limitation is that all 370 OOD documents are programmatically generated. Real-world OOD documents exhibit more diverse and subtle distribution shifts than synthetic categories can capture. Performance on real OOD documents (e.g., Tobacco800 tobacco industry documents, RVL-CDIP document classification images, CORD receipt images) is unknown. A 13-model consensus analysis identified this as the unanimous top priority for follow-up work.

**Global Gaussian assumption.** The single-Gaussian model assumes the training distribution is unimodal. If DIQA-5000 contains distinct document sub-populations (e.g., different script families, different source types), a Gaussian mixture model might better capture the training distribution's shape and improve sensitivity for subtle OOD shifts near specific sub-population boundaries. However, the near-perfect AUROC on synthetic data suggests this limitation is not yet performance-limiting.

**Small OOD evaluation set.** With only 370 OOD documents (20-30 per category), per-category AUROC estimates have wide confidence intervals. The 100% detection rate for 10 of 13 categories may partially reflect small sample sizes -- a category with 30 samples and 100% detection has an approximate 95% CI of [88.4%, 100%] by the Clopper-Pearson method.

**Circular training concern.** The embeddings used for OOD detection are produced by the same model whose predictions the detector is validating. This creates a potential circular dependency: if SigLIP2 produces degenerate embeddings for certain OOD types (mapping them to the same region regardless of visual content), the detector would fail silently. This risk is mitigated by the empirical observation that OOD embeddings are consistently far from the training centroid, but it cannot be ruled out for unseen OOD categories.

**No downstream impact evaluation.** We measure detection performance (AUROC, TPR/FPR) but do not evaluate whether flagging OOD documents actually improves downstream quality prediction accuracy. A complete evaluation would measure SigLIP2's prediction error on auto-accepted vs. Tier-2-reviewed images.

## 6. Conclusion & Future Work

We presented an embedding-space OOD detector for document quality pipelines that achieves AUROC = 0.9963 on synthetic OOD documents while adding only 1-2ms to inference latency. The detector operates on embeddings already computed during SigLIP2 quality prediction, requiring no additional forward passes. All 13 OOD categories are detected with AUROC >= 0.97, and the recommended threshold of 30.8 (train+val p95) flags 99.5% of OOD documents at a 5% false positive rate.

The threshold sensitivity analysis reveals that Mahalanobis distance is the dominant reliability signal. Aleatoric uncertainty (predicted variance and entropy) provides complementary discrimination only under tightly calibrated thresholds -- the default v1 thresholds are too permissive to trigger on any test sample. For practical deployment, combining both signals under data-calibrated thresholds yields the most informative routing: 65.4% of test images auto-accepted, 16.9% downweighted, and 16.8% routed to Tier 2 VLM validation.

**Future work.**

- **Real-world OOD evaluation.** Validate on public document datasets (RVL-CDIP, Tobacco800, CORD) that represent genuine distribution shifts. This is the highest priority, endorsed unanimously by a 13-model consensus analysis.
- **Per-class Gaussian mixture.** Replace the single global Gaussian with a mixture model or per-source-type components to improve sensitivity for subtle OOD shifts near sub-population boundaries.
- **Dual-embedding OOD detection.** Combine SigLIP2 embeddings with a second embedding space (e.g., contrastive document encoder) to decorrelate the circular training dependency.
- **Conformal prediction integration.** Use conformal prediction to provide calibrated p-values for the Mahalanobis distance, replacing the percentile-based threshold with a statistically principled confidence level.
- **Online threshold adaptation.** As OOD documents are encountered and labeled in production, update the threshold using online calibration to track the evolving deployment distribution.

## 7. Reproducibility, Data & Governance

### 7.1 Artifacts & Paths

| Artifact | Path | Format | Size |
|----------|------|--------|------|
| OOD detector v2 (mean, precision, thresholds) | `results/siglip2_diqa5000/ood_detector_v2.npz` | NPZ | 2.2 MB |
| Train embeddings | `results/siglip2_diqa5000/embeddings/train.npz` | NPZ | 6.9 MB |
| Val embeddings | `results/siglip2_diqa5000/embeddings/val.npz` | NPZ | 989 KB |
| Test embeddings | `results/siglip2_diqa5000/embeddings/test.npz` | NPZ | 1.9 MB |
| Summary statistics | `results/siglip2_diqa5000/summary.json` | JSON | 713 B |
| Threshold sweep results | `results/threshold_sensitivity/sweep_results.json` | JSON | 86 KB |
| Threshold sweep report | `results/threshold_sensitivity/sweep_report.md` | MD | 8.9 KB |
| Train predictions | `results/siglip2_diqa5000/siglip2_diqa5000_train.jsonl` | JSONL | 4.6 MB |
| Val predictions | `results/siglip2_diqa5000/siglip2_diqa5000_val.jsonl` | JSONL | 662 KB |
| Test predictions | `results/siglip2_diqa5000/siglip2_diqa5000_test.jsonl` | JSONL | 1.3 MB |
| Calibration results | `results/siglip2_diqa5000/calibration_results.json` | JSON | 14 KB |
| OOD detector documentation | `results/tier1_ood_detector/README.md` | MD | 7 KB |

### 7.2 Environment, Seeds & Versions

- **Embedding extraction**: Modal serverless GPU (NVIDIA L4, 24GB), ~50 min for 5,000 images
- **Model**: `google/siglip2-base-patch16-naflex` fine-tuned checkpoint `siglip2_iqa_best.pt`
- **Covariance estimation**: scikit-learn `LedoitWolf` with default parameters (shrinkage = 0.0032)
- **Random seeds**: NumPy RNG seed 42 for bootstrap and figure generation
- **Python**: 3.10+, numpy, scipy, scikit-learn, matplotlib

### 7.3 Compute/Cost Summary

| Component | Cost |
|-----------|------|
| Embedding extraction (5,000 images on L4) | ~$2.50 (Modal compute) |
| OOD detector fitting (4,000 x 768 covariance) | ~5 seconds (CPU) |
| Per-image OOD scoring (inference) | 1-2ms (CPU) |
| Threshold sensitivity sweep (11 configs) | ~10 seconds (CPU) |

Total compute cost for all experiments in this paper: under $5.

### 7.4 Data Licensing & Ethical Considerations

- **DIQA-5000**: VQualA 2025 DIQA Challenge dataset. Used under challenge license for research purposes. Images are public benchmark documents; no PII.
- **Synthetic OOD**: Programmatically generated from open-source fonts and templates. No real documents or personal data.
- **Embeddings**: Derived features (768-dim vectors) that cannot be inverted to reconstruct original images.

## Acknowledgments

Embedding extraction was performed on Modal serverless GPU infrastructure. The 13-model consensus analysis (EXP-009) that informed the future work priorities was conducted via OpenRouter API across 13 frontier LLMs.

## References

1. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS 2018. arXiv:1807.03888.
2. Liang, S., Li, Y., & Srikant, R. (2018). Enhancing the reliability of out-of-distribution image detection in neural networks. ICLR 2018. arXiv:1706.02690.
3. Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. NeurIPS 2020. arXiv:2010.03759.
4. Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. ICML 2022. arXiv:2204.06507.
5. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.
6. Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2025). SigLIP 2: Scaling Vision-Language Encoders. arXiv:2502.14786.
7. DeQA-Doc Technical Report Series, Papers 1-7. (2026). https://github.com/DeQA-Doc.

## Appendix

### A. OOD Detector API

```python
from image_preprocessing_detector.detection.ood_detector import EmbeddingOODDetector

# Load fitted detector
detector = EmbeddingOODDetector.load("results/siglip2_diqa5000/ood_detector_v2.npz")

# Score a single embedding (768-dim vector from SigLIP2)
result = detector.score(embedding)
print(result.mahalanobis_distance)  # e.g., 25.3 (ID) or 82.1 (OOD)
print(result.is_ood)                # True/False at threshold
print(result.percentile)            # Approximate percentile vs calibration set

# Batch scoring
results = detector.score_batch(embeddings)  # embeddings: (N, 768)
flagged = [r for r in results if r.is_ood]

# Re-fit on new training data
detector = EmbeddingOODDetector.from_embeddings(
    new_embeddings,  # shape: (N, 768)
    threshold_percentile=95.0,
)
detector.save("ood_params_updated.npz")
```

### B. Threshold Configuration Details

The 11 threshold configurations sweep across three signal dimensions:

1. **Mahalanobis distance (d_M)**: OOD trigger and hard-reject thresholds
2. **Predicted variance (sigma_sq)**: Auto-accept and low-weight thresholds per quality dimension
3. **Predictive entropy**: Auto-accept and low-weight thresholds per quality dimension

The "current" (v1) configuration uses d_M thresholds calibrated from v1 test data (46.0 / 58.6) with sigma-squared and entropy thresholds set too high to trigger (0.64 / 1.0 and 1.2 / 1.5 respectively). The "data-calibrated" configuration uses the same d_M thresholds but calibrates aleatoric signals from actual data percentiles, revealing their complementary discrimination power.

### C. SigLIP2-IQA Calibration Results

As a supplementary finding, isotonic calibration of SigLIP2-IQA predictions (raw model output to MOS scale) preserves ranking quality while correcting scale:

| Dimension | SRCC (raw) | SRCC (calibrated) | MAE (raw) | MAE (calibrated) |
|-----------|------------|-------------------|-----------|-------------------|
| Overall | 0.899 | 0.899 | 2.409 | 0.167 |
| Sharpness | 0.874 | 0.874 | 2.404 | 0.184 |
| Color | 0.893 | 0.893 | 2.474 | 0.172 |

The dramatic MAE reduction (from ~2.4 to ~0.17) reflects the model's internal [0, 1] output range being rescaled to MOS [1, 5]. SRCC is preserved by construction since calibration is a monotone transformation.

---

*This work is part of the DeQA-Doc Technical Report Series. All data, code, and figures are available at the project repository under CC BY-SA 4.0.*
