# OOD Detection Baseline Comparison

Comparison of four OOD scoring methods on SigLIP2 embeddings (768-dim) with **ground truth** ID/OOD labels. Eval set: 1150 ID + 370 OOD = 1520 samples.

- **ID**: DIQA-5000 test (1,000) + synthetic in-distribution (150)
- **OOD**: Synthetic out-of-distribution (370) across 13 categories
- **Reference**: train+val (4,000) used for fitting all methods

## Results

| Method | AUROC | 95% CI | AUPRC | FPR@95TPR | FPR@99TPR |
| ------ | ----- | ------ | ----- | --------- | --------- |
| Mahalanobis (Ledoit-Wolf) | 0.9470 | [0.9371, 0.9561] | 0.8438 | 0.1435 | 0.1513 |
| k-NN (k=10) | 0.9570 | [0.9481, 0.9652] | 0.8686 | 0.1391 | 0.1774 |
| Cosine distance | 0.9494 | [0.9389, 0.9588] | 0.8478 | 0.1748 | 0.2070 |
| Energy (neg. LogSumExp) | 0.9106 | [0.8962, 0.9245] | 0.7576 | 0.2757 | 0.3643 |

## Per-Category AUROC Breakdown

| Category | n_OOD | Mahalanobis (Ledoit-Wolf) | k-NN (k=10) | Cosine distance | Energy (neg. LogSumExp) |
| ------ | ---: | ----: | ----: | ----: | ----: |
| adversarial_fraktur | 20 | 0.9583 | 0.9661 | 0.9647 | 0.9385 |
| adversarial_nastaliq | 20 | 1.0000 | 0.9991 | 0.9920 | 0.9923 |
| binarized | 30 | 0.9253 | 0.9032 | 0.8898 | 0.8551 |
| cjk_vertical | 30 | 0.8600 | 0.8588 | 0.8240 | 0.7126 |
| form_layout | 30 | 0.9531 | 0.9603 | 0.9592 | 0.9272 |
| heavily_degraded | 30 | 1.0000 | 1.0000 | 1.0000 | 0.9997 |
| multiscript | 30 | 0.9615 | 0.9756 | 0.9785 | 0.9581 |
| pristine | 30 | 0.9314 | 0.9653 | 0.9543 | 0.9188 |
| script_ethiopic | 30 | 0.9453 | 0.9213 | 0.9441 | 0.8475 |
| script_myanmar | 30 | 0.8766 | 0.9526 | 0.9121 | 0.8406 |
| script_tibetan | 30 | 1.0000 | 0.9981 | 0.9939 | 0.9739 |
| very_high_dpi | 30 | 0.9209 | 0.9583 | 0.9488 | 0.9106 |
| very_low_dpi | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Figures

- [ROC curves](figures/roc_comparison.png)
- [AUROC bar chart](figures/auroc_bar_comparison.png)
- [Per-category heatmap](figures/per_category_heatmap.png)

## Limitations

1. **Single embedding space**: All methods use the same SigLIP2 embeddings. Results may differ with other backbone features.
2. **Synthetic OOD**: OOD images are programmatically generated, not real-world OOD documents. Results indicate separability of the synthetic categories but may not generalize to all OOD types.

## Recommendation

**Best method**: k-NN (k=10) (AUROC=0.9570).

See per-category breakdown above for method-specific strengths. The optimal production threshold should be calibrated at 95% or 99% TPR using the best-performing method's FPR values.

