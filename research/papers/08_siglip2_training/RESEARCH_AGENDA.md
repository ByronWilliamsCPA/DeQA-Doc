# Research Agenda: Paper 8 --- Training SigLIP2-IQA-Base

## Potential Improvements

- **Increase NaFlex max_num_patches to 784+**: The 576-patch ceiling limits effective resolution to ~384x384, which constrains sharpness SRCC (0.874 vs 0.899 for overall). Increasing to 784 patches raises effective resolution to ~448x448 and should yield +2-3% sharpness SRCC based on the known relationship between resolution and text edge detection. Expected impact: close the 0.025 sharpness-overall gap. Effort: low (configuration change + retraining, ~4 hours on A10).

- **Reduce dropout from 0.3 to 0.1-0.15**: The 0.3 dropout rate was inherited from larger-dataset settings. With only 3,500 training images, the heads have ~197K parameters each -- the dropout rate may be suppressing useful capacity. Expected impact: moderate improvement in per-dimension PLCC. Effort: low (hyperparameter change + retraining).

- **Dimension-specific attention pooling**: Replace global average pooling with learned attention weights per dimension. Overall quality benefits from holistic features, while sharpness benefits from high-frequency local features. Per-dimension attention allows each head to attend to the most relevant backbone features. Expected impact: improved sharpness SRCC without sacrificing overall or color performance. Effort: medium (architecture modification, ~200 additional parameters per head).

- **Layer-wise learning rate decay (LLRD)**: Apply a 0.9 decay factor per transformer layer during Phase 2 fine-tuning. Lower layers preserve pretrained features (edges, textures) while upper layers adapt to document quality. Expected impact: improved training stability and potentially higher final SRCC. Effort: low (optimizer configuration change).

- **CosineAnnealingWarmRestarts scheduler**: Replace simple cosine annealing with warm restarts to escape local minima during Phase 2. With 40 epochs and T_0=10, the model gets 4 restart cycles. Expected impact: prevents premature convergence and improves final checkpoint selection. Effort: low (scheduler swap).

- **Wider regression heads (768 -> 512 -> 256 -> 2)**: The current 768 -> 256 -> 2 heads may be too narrow. Adding an intermediate 512-dim layer doubles head capacity (~400K params per head) while remaining negligible relative to the 86M backbone. Expected impact: marginal SRCC improvement, primarily on color fidelity where the feature transformation may be underparameterized. Effort: low (architecture change + retraining).

- **MarginRankingLoss auxiliary**: Add a pairwise ranking loss (sample two images per batch, enforce correct rank ordering) alongside NormInNorm and GaussianNLL. This directly optimizes for SRCC-like rank correlation. Expected impact: +1-2% wSRCC improvement. Effort: medium (loss implementation, batch sampling logic).

- **Document-specific augmentations**: Add synthetic blur, JPEG compression artifacts, and noise injection during training to improve robustness. Currently only standard augmentations (flip, crop, color jitter) are used. Expected impact: improved OOD generalization, potentially reducing the 0.886 -> 0.620 OOD degradation. Effort: medium (augmentation pipeline design and tuning).

## Test Refinements

- **Resolution ablation study**: Train models at 384, 576, 784, and 1024 max patches and measure per-dimension SRCC to quantify the resolution-sharpness relationship. Why it matters: validates the hypothesis that the sharpness bottleneck is resolution-limited and determines diminishing returns. Data: DIQA-5000 training set, same hyperparameters except max_patches. Estimated effort: 4x training runs (~16 hours).

- **Dropout rate sweep**: Evaluate dropout rates of 0.0, 0.05, 0.1, 0.15, 0.2, 0.3 on the validation set to determine the optimal regularization level for 3,500-sample regression. Why it matters: 0.3 may be suboptimal and the sensitivity is unknown. Data: DIQA-5000 val set. Estimated effort: 6x training runs (~24 hours).

- **Phase 1 duration ablation**: Test 5, 10, 15, and 20 epochs for Phase 1 warmup to determine the minimum warmup needed before unfreezing. Why it matters: if heads converge in 5 epochs, Phase 1 is 5 epochs too long, wasting training budget. Data: monitor validation loss convergence. Estimated effort: 4x training runs (~16 hours).

- **Calibration method comparison on held-out documents**: Fit calibration functions (linear, 4PL, isotonic) on DIQA-5000 and evaluate on an independent document quality dataset (e.g., subset of KADID-10K adapted to documents). Why it matters: calibration generalization is a core assumption for production deployment. Data: requires a second document quality dataset with MOS annotations.

- **Per-dimension uncertainty calibration analysis**: Evaluate whether sigma-squared accurately reflects prediction error magnitude per dimension. Plot sigma-squared vs. absolute error and compute calibration curves. Why it matters: the auto-accept threshold (0.64) is shared across dimensions but sharpness may need a tighter threshold. Data: DIQA-5000 test predictions already collected.

- **Multi-task head interaction analysis**: Measure whether training with script/source/orientation heads improves IQA performance vs. IQA-only training. Why it matters: multi-task regularization is claimed but not empirically isolated. Data: IQA-only checkpoint comparison. Estimated effort: 1 additional training run (~4 hours).

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| v2.0 with 784 patches | Sharpness SRCC improves by 2-3% (0.874 -> 0.90) | DIQA-5000, same splits | Critical |
| VLM pseudo-label expansion (first cycle) | wSRCC maintained > 0.88 on DIQA-5000 after retraining on expanded data | 500-1000 OOD documents with VLM pseudo-labels | Critical |
| Dropout + LLRD combined sweep | Combined improvements exceed individual gains due to complementary effects | DIQA-5000, same splits | High |
| ONNX export and CPU inference benchmark | Model achieves < 500ms inference on CPU with ONNX runtime | Exported model, benchmark images | High |
| Cross-dataset transfer evaluation | SigLIP2-IQA-Base trained on DIQA-5000 achieves SRCC > 0.70 on KADID-10K document subset | KADID-10K with document-like images | High |
| Attention map analysis | Sharpness head attends to text regions while color head attends to image regions | DIQA-5000 test images with GradCAM visualization | Medium |
| Knowledge distillation from DeQA-Doc ensemble | Training with ensemble soft targets as auxiliary loss improves single-model accuracy by > 0.01 wSRCC | DeQA-Doc ensemble predictions on training set | Medium |
| SigLIP2-Large backbone (400M params) | Larger backbone improves sharpness SRCC by > 3% to exceed 0.90 | DIQA-5000, A100 GPU for training | Medium |
| Mixed-precision (FP16/BF16) training | Training speed improves 1.5x with < 0.001 wSRCC difference | Same data, A10 GPU | Low |
| Temperature scaling for uncertainty | Post-hoc temperature scaling improves uncertainty calibration curves | DIQA-5000 validation set predictions | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Metric Consistency | wSRCC inconsistency: abstract reports 0.886, Table 5.1 reports 0.891, baseline table reports 0.886. Must unify to single canonical value. | Open |
| 2026-03-08 | 4-model consensus | Ablation Studies | Missing ablations for: (1) two-phase vs single-phase training, (2) 576 vs 784 patch count, (3) dropout 0.3 vs lower, (4) IQA-only vs multi-task training, (5) NormInNorm vs +GaussianNLL. All 4 reviewers flagged this. | Open |
| 2026-03-08 | 4-model consensus | MAE Reporting | Raw MAE of 2.424 confusing if outputs are [0,1] scale. Clarify prediction scale at each stage. Fix Section 5.3 table phrasing. | Open |
| 2026-03-08 | 4-model consensus | OOD Analysis | 30% OOD degradation (0.886->0.620) needs deeper category-level breakdown and uncertainty-error correlation analysis. | Open |
| 2026-03-08 | 3/4 reviewers | Multi-Task Heads | Section 6.3 lists 5 auxiliary heads but provides zero accuracy metrics. Either add evaluation tables or remove section. | Open |
| 2026-03-08 | 2/4 reviewers (GPT-5.2, Qwen 3.5+) | Reproducibility | Private Modal volume checkpoint and unclear DIQA-5000 licensing limit independent verification. Add public access path. | Open |
| 2026-03-08 | Gemini 3.1 Pro | Visuals | Add qualitative examples of extreme quality blindspot (prediction range [0.32, 3.92]) and uncertainty calibration plots. | Open |
| 2026-03-08 | Qwen 3.5+ | Backbone Comparison | Need empirical comparison with DINOv2/CLIP backbones, not just rationale table. | Open |
| 2026-03-08 | Consensus | Overall | Recommendation: Minor Revision (3/4 Minor, 1/4 Major). Mean score 3.70/5.0. See PEER_REVIEW.md for full details. | Open |
