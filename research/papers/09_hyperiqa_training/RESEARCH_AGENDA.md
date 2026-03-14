# Research Agenda: Paper 9 --- Training HyperIQA++

## Potential Improvements

- **Regularized fine-tuning.** Apply elastic weight consolidation (EWC) or progressive layer unfreezing to reduce catastrophic forgetting. The off-the-shelf model outperforms the fine-tuned variant on OOD data (0.723 vs 0.694), indicating that aggressive full fine-tuning destroys general quality features. EWC would penalize large deviations from pretrained weights in the backbone while allowing the new heads to train freely. Expected impact: reduce ID/OOD gap from -0.165 to below -0.10 while maintaining DIQA-5000 MainScore above 0.84.

- **Scale calibration.** The MAE of 2.225 coexists with PLCC of 0.886, indicating a systematic offset in absolute predictions. Fit isotonic regression or a simple linear rescaling on a held-out calibration set to correct the 10-bin expected value to the [1, 5] MOS range. Expected impact: reduce MAE below 1.0 without changing correlation metrics.

- **Multi-resolution input.** Replace the fixed 1600x1600 resize with MUSIQ-style multi-scale patch handling or NaFlex adaptive resolution. Documents vary widely in aspect ratio and DPI; forcing all inputs to a square grid discards structural information. Expected impact: improved handling of extreme aspect ratio documents in the OOD set.

- **Data augmentation with pseudo-labels.** Fine-tune on DIQA-5000 plus pseudo-labeled OOD documents from the pipeline described in Paper 7. Adding 500--1,000 diverse documents should reduce overfitting to DIQA-5000's distortion types. Expected impact: close the ID/OOD gap while preserving or improving DIQA-5000 performance.

- **LoRA vs full fine-tuning comparison.** Test parameter-efficient fine-tuning (LoRA, rank 16--64) against full fine-tuning. LoRA preserves more pretrained knowledge by constraining the update subspace, potentially reducing catastrophic forgetting at the cost of lower peak ID performance.

## Test Refinements

- **Bootstrap confidence intervals.** Add 95% CIs via bootstrap resampling (n=1,000 iterations) to all SRCC and PLCC values. The 3-point gap between HyperIQA++ (0.856) and SigLIP2-IQA (0.886) may or may not be statistically significant at n=1,000.

- **Per-document-type stratification.** Break down performance by document category (printed, handwritten, forms, receipts) on both ID and OOD sets. CNN-based features may transfer better for certain document types where texture is the dominant quality cue.

- **Ablation study.** Isolate the contribution of each extension (spatial attention, multi-dimensional output, soft-label distribution, high-resolution input) by training variants with individual components removed. Determine which extension contributes most to the 1.2-point gain over the competition baseline.

- **Calibration error analysis.** Quantify the scale miscalibration using expected calibration error (ECE) and reliability diagrams. Determine whether the offset is constant (simple bias correction) or varies with predicted quality level (requires nonlinear calibration).

- **OOD category breakdown.** Report per-category OOD performance across the 13 synthetic categories (heavily degraded, adversarial scripts, binarized, etc.) to identify specific failure modes of the CNN architecture.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| EWC regularized fine-tuning | EWC reduces ID/OOD gap below 0.10 while maintaining DIQA MainScore above 0.84 | DIQA-5000 train + pretrained weights + Fisher information matrix | High |
| Pseudo-label augmented training | Adding 1,000 pseudo-labeled OOD documents improves OOD MainScore above 0.75 | Pseudo-labels from Paper 7 pipeline + OOD document images | High |
| LoRA fine-tuning comparison | LoRA (rank 32) achieves within 0.02 MainScore of full fine-tuning with smaller OOD gap | DIQA-5000 train split | High |
| Scale calibration (isotonic) | Post-hoc isotonic calibration reduces MAE below 1.0 on both ID and OOD | Hold-out calibration set (200 images from DIQA-5000 val) | High |
| Component ablation | Soft-label distribution heads contribute more than spatial attention to the 1.2-point gain | DIQA-5000 train + multiple training runs | Medium |
| Multi-resolution input | Adaptive resolution improves OOD MainScore by 0.03+ vs fixed 1600x1600 | Same training data, modified preprocessing | Medium |
| Progressive layer unfreezing | Freezing backbone for first N epochs then unfreezing reduces forgetting | DIQA-5000 train split | Medium |
| Feature visualization (GradCAM) | CNN attention focuses on text regions for sharpness but fails on non-Latin scripts | Trained model + OOD test images | Medium |
| Ensemble weight optimization | Learned ensemble weights for HyperIQA++ + SigLIP2 + VLM outperform equal weighting | Per-image predictions from all three models + ground truth | Low |
| Cross-dataset transfer | Model fine-tuned on DIQA-5000 + pseudo-labels transfers to Tobacco800 quality assessment | Tobacco800 with quality annotations | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Recommendation | Split verdict: 2x Minor Revision, 2x Major Revision. Consensus: Minor-to-Major Revision. Mean score 3.50/5. | Open |
| 2026-03-08 | All 4 models (unanimous) | Ablation | Missing ablation study isolating contributions of spatial attention, soft-label heads, multi-dim output, and 1600x1600 resolution. Cannot attribute 1.2-point gain over competition baseline. | Open |
| 2026-03-08 | All 4 models (unanimous) | Calibration | MAE=2.225 on [1,5] scale is anomalous given PLCC=0.886. Must experimentally validate calibration fix (isotonic regression or linear rescaling), not just hypothesize. | Open |
| 2026-03-08 | All 4 models (unanimous) | Consistency | MainScore inconsistency: 0.856 (abstract) vs 0.840 (Section 5.3). Reconcile to single canonical value and explain evaluation split difference. | Open |
| 2026-03-08 | All 4 models (unanimous) | Formatting | Section 5.X placeholder and "see Section 5.X" cross-reference are draft artifacts. Finalize numbering. | Open |
| 2026-03-08 | GPT-5.2, Qwen 3.5+ | Reproducibility | Section 4.3 omits learning rate, batch size, epochs, weight decay, seeds. Blocks exact replication. | Open |
| 2026-03-08 | GPT-5.2 | Technical | Parameter count claim (28M to 138M "due to larger spatial feature maps") is incorrect — resolution changes affect compute, not params. Clarify what added parameters. | Open |
| 2026-03-08 | GPT-5.2 | Citation | DBCNN cited as (Zhang et al., 2018) in text but (Zhang et al., 2020) in references. | Open |
| 2026-03-08 | Gemini 3.1 Pro | Visuals | No qualitative figures despite discussing spatial attention and OOD failure modes. Add attention maps or failure case examples. | Open |
| 2026-03-08 | Qwen 3.5+ | Analysis | Add training convergence curves showing when catastrophic forgetting begins during fine-tuning. | Open |
| 2026-03-08 | Grok 4.1 Fast, GPT-5.2 | Language | Soften "fundamental" CNN ceiling claim to "observed" — not proven without broader architecture sweeps. | Open |
