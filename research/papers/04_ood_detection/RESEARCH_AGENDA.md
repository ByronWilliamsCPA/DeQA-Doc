# Research Agenda: Paper 4 — OOD Detection

## Potential Improvements

- **Multi-signal fusion**: Combine Mahalanobis distance with aleatoric uncertainty (predicted variance, entropy) using learned weights instead of fixed thresholds. Expected impact: Better discrimination at decision boundaries. Effort: Medium.

- **Adaptive thresholds**: Learn per-category OOD thresholds instead of a single global p95/p99 cutoff. Expected impact: Higher recall on near-OOD categories (CJK vertical, form layouts) without increasing false positives. Effort: Low-Medium.

- ~~**Alternative distance metrics**~~: **COMPLETED** (2026-03-08). Compared Mahalanobis to k-NN, cosine, and energy-based scores on identical embeddings. Mahalanobis dominates (AUROC 0.9999 on proxy labels); cosine is best simple baseline (0.912); k-NN peaks at k=5 (0.876); energy worst (0.840). See `research/ood_baselines/` and Paper 4 Section 4.5.

- **Larger embedding models**: Replace SigLIP2-Base (768-dim) with SigLIP2-Large (1024-dim) or SigLIP2-SO400M for richer representations. Expected impact: Better separation in embedding space. Effort: Low (re-extract embeddings).

- **Online distribution tracking**: Implement running mean/covariance updates so the detector adapts as the training distribution expands. Expected impact: Enables iterative domain expansion without full refit. Effort: Medium-High.

## Test Refinements

- **Expand OOD evaluation set**: Current 370 synthetic OOD samples may not capture all real-world distribution shifts. Add real-world OOD documents from digital archives. Why it matters: AUROC 0.9963 on synthetic may overestimate real-world performance.

- **Near-OOD stress test**: Create samples that are semantically OOD but visually similar to ID (e.g., same document type but different language). Why it matters: Tests the detector's sensitivity to subtle distribution shifts.

- **Calibration analysis**: Evaluate whether Mahalanobis distance is well-calibrated as a probability of being OOD. Data: Existing embeddings + isotonic calibration. Compute: Minimal.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|---|---|---|---|
| ~~k-NN OOD detection~~ | ~~k-NN distance matches Mahalanobis AUROC without covariance estimation~~ | Existing embeddings | **DONE** — hypothesis rejected; k-NN (0.876) << Mahalanobis (0.9999). See Section 4.5. |
| Layer-wise OOD detection | Intermediate SigLIP2 layers capture different OOD signals | Re-extract multi-layer embeddings | Medium |
| Contrastive OOD training | Fine-tuning SigLIP2 with OOD-aware contrastive loss improves separation | OOD samples + training compute | Medium |
| Temporal drift detection | Monitor Mahalanobis distance statistics over time to detect distribution drift | Production deployment data | Medium |
| Cross-encoder OOD detection | Use document-specific features (text density, layout complexity) as auxiliary OOD signals | Document layout analysis | Low |
| Ensemble OOD detection | Combine detectors from multiple embedding models for robustness | Multiple model embeddings | Low |
| Human evaluation of boundary cases | Validate that images near the threshold are genuinely ambiguous | Human annotation budget (~100 images) | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|---|---|---|---|---|
| 2026-03-08 | 5-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Validity | Synthetic-only OOD evaluation likely overestimates real-world AUROC; validate on RVL-CDIP, Tobacco800, CORD | Open |
| 2026-03-08 | 5-model consensus | Completeness | No baseline OOD method comparisons (cosine, KNN, energy, GMM, one-class SVM) on same embeddings | **Partially addressed** — 4 methods compared (Section 4.5); GMM and one-class SVM not yet tested. Proxy labels used; needs real OOD embeddings. |
| 2026-03-08 | 5-model consensus | Completeness | No downstream impact evaluation — show MAE/SRCC improvement from gating (auto-accepted vs flagged) | Open |
| 2026-03-08 | 5-model consensus | Statistics | Per-category AUROC CIs are wide (n=20-30); report bootstrap CIs and increase sample sizes to n=50-100 | Open |
| 2026-03-08 | GPT-5.2 | Reporting | FPR "by construction" claim misleading given train-test distance shift; report explicit test-set FPR at each threshold | Open |
| 2026-03-08 | 3/4 reviewers | Clarity | Table 3 "Current (v1)" and "d_M Only" rows are identical — consolidate or annotate to clarify v1 aleatoric thresholds are effectively disabled | Open |
| 2026-03-08 | GPT-5.2 | Reproducibility | Document synthetic OOD generation scripts, fonts/templates, and parameter ranges | Open |
| 2026-03-08 | Gemini 3.1 Pro | Method | Single global Gaussian assumption may mask subtle OOD shifts as pipeline scales to broader document types | Open |
| 2026-03-08 | Qwen 3.5+ | Formatting | Section numbering inconsistency (Sec 6 "concludes" but Sec 7 exists); figure references not embedded in markdown | Open |
