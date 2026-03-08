# Research Backlog

Prioritized list of untested hypotheses and improvement ideas. When an idea is implemented, move its row to the [Archive](#archive) section with a link to the experiment entry.

---

## Active Ideas

| ID | Idea | Priority | Effort | Expected Impact | Source | Depends On |
|---|---|---|---|---|---|---|
| [calibration](#calibration) | Isotonic regression / quantile mapping to correct VLM bias | P1 | Low | Reduce MAE by ~0.3-0.5 MOS | [VLM_TEACHER_EVALUATION](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S6 | EXP-007 |
| [e2e-validation](#e2e-validation) | Train SigLIP2 on VLM pseudo-labels, measure downstream SRCC | P1 | Medium | Validates entire pipeline | [VLM_TEACHER_EVALUATION](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S6 | calibration |
| [real-world-ood](#real-world-ood) | Evaluate OOD detector on real document datasets | P1 | Low | Validates AUROC claim (expect drop to 0.70-0.85) | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S2b | EXP-006 |
| [public-dataset-expand](#public-dataset-expand) | Extract SigLIP2 embeddings from RVL-CDIP, Tobacco800, CORD | P1 | Low | OOD eval + covariance expansion | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S3 | EXP-006 |
| [ensemble-optimization](#ensemble-optimization) | Systematic model combination search for consensus scoring | P2 | Medium | +0.02-0.05 wSRCC over single model | [VLM_TEACHER_EVALUATION](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S5.4 | EXP-007 |
| [siglip2-v2-arch](#siglip2-v2-arch) | Tier 1 architecture improvements (patch count, scheduler, grad accum, attention pooling) | P2 | Medium | +2-3% VQualA | [research.md](../research.md) S7 | — |
| [vlm-committee](#vlm-committee) | Multi-VLM committee labeling with bias calibration at scale | P2 | High | Scale pseudo-labels to 10K+ images | [research.md](../research.md) S6 | calibration |
| [active-learning](#active-learning) | Use inter-model disagreement + BALD for targeted human annotation | P2 | Medium | Efficient annotation on most informative samples | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S3 | EXP-007 |
| [dual-embedding-ood](#dual-embedding-ood) | Combine SigLIP2 + DiT/LayoutLMv3 embeddings for OOD detection | P3 | Medium | Better boundary discrimination for near-OOD | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S3 | EXP-006 |
| [controlled-degradation](#controlled-degradation) | Apply known degradations to real documents for cheap ground truth | P3 | Medium | Augmentation without human labels | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S3 | public-dataset-expand |
| [pca-ood](#pca-ood) | PCA 768→256 dims before Mahalanobis fitting | P3 | Low | Possible threshold improvement, remove noise dims | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S4a | EXP-006 |
| [energy-ensemble-ood](#energy-ensemble-ood) | ODIN/energy-based ensemble with Mahalanobis + kNN density | P3 | Medium | +5-10% AUROC on near-boundary OOD | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S4b | EXP-006 |
| [conformal-prediction](#conformal-prediction) | Risk-controlled gating with statistical guarantees | P3 | Medium | "Catch 95% of unreliable predictions" guarantee | [NEXT_ITERATION](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) S4d | real-world-ood |

---

## Details

### calibration

Isotonic regression on the 3,500 DIQA-5000 training images to learn a VLM-to-MOS mapping per dimension per model. All VLMs exhibit systematic positive bias (+0.5 to +1.5 MOS). Adjacent bucket accuracy (68-78%) suggests models rank correctly but miscalibrate absolute values. Quantile mapping is an alternative if isotonic regression overfits. See [VLM_TEACHER_EVALUATION.md S5.1, S6](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) for the over-rating analysis.

### e2e-validation

Train SigLIP2-IQA on calibrated VLM pseudo-labels and measure SRCC degradation vs. human-label baseline on DIQA-5000 test. This is the single most important validation — if pseudo-labels degrade the student model, the entire pipeline needs rethinking. Start with Gemini 3 Flash labels only (best single model), then add GPT-4.1 for consensus.

### real-world-ood

Evaluate the OOD detector on naturally-occurring documents from public datasets (RVL-CDIP, Tobacco800, CORD). The current AUROC of 0.9963 is on synthetic data — the 13-model consensus (EXP-009) unanimously agrees real-world performance will be substantially lower (estimated 0.70-0.85). This is a prerequisite for any OOD detector expansion work.

### public-dataset-expand

Extract SigLIP2 embeddings from RVL-CDIP (sample 5-10K across 16 categories), Tobacco800 (1,600 docs), CORD (sample 500 receipts). Use for evaluation first (no label needed — just compute Mahalanobis distances). Only expand OOD covariance after validating SigLIP2's IQA predictions on those document types. Critical: expanding covariance makes those docs "in-distribution" — only do this for validated domains.

### ensemble-optimization

Gemini 3 Flash and GPT-4.1 have complementary failure modes (Gemini leads on in-distribution/non-Latin, GPT-4.1 leads on adversarial/CJK/multiscript). Systematic search over model combinations and weighting schemes could outperform either alone. Start with simple median of top-2, then explore weighted combinations.

### siglip2-v2-arch

Independent of VLM distillation work. Four Tier 1 improvements from [research.md S7](../research.md): (1) Increase max_num_patches 576→784+ for better text resolution, (2) CosineAnnealingWarmRestarts to prevent premature convergence, (3) Gradient accumulation batch 4x4 for PCGrad in Phase 2, (4) Attention pooling per dimension. Each is low-medium effort, collectively expected +2-3% VQualA.

### vlm-committee

Scale pseudo-labeling to 10K+ diverse documents (tax forms, legal, handwritten, receipts) using calibrated VLM committee. Protocol from EXP-009 consensus: minimum 3 diverse models, abstain on disagreement (JSD > 0.1 or std > 0.5), human bridge set of 50-100 images for validation. Estimated cost: $50-75 for 5K images at 5 passes via Sonnet 4.6.

### active-learning

The uncertainty pipeline already implements BALD-based sample selection (`src/uncertainty/active_learning.py`). Deploy OOD detector in shadow mode, log boundary cases (d_M 40-55), prioritize for human annotation using BALD + proximity to boundary. Use adaptive rater counts (Dawid-Skene/MACE aggregation) rather than 15 raters per image.

### dual-embedding-ood

Combine SigLIP2 (IQA-aware) with a document-specialized model (DiT-Base or LayoutLMv3-Base, both 768-dim) for OOD detection. Gate on OR of separate Mahalanobis thresholds per space rather than concatenating (avoids curse of dimensionality). Catches failures in either embedding space independently. ~30ms additional inference per image.

### controlled-degradation

Apply known degradations (Gaussian blur, JPEG compression, noise, resolution reduction) to pristine documents from RVL-CDIP. Ground truth quality is derived directly from degradation parameters — no VLM or human annotation needed. Provides real document semantics with controlled quality variation. Requires calibrating degradation parameters against DIQA-5000 MOS scale.

### pca-ood

768 dimensions with 4,400 samples is a tight ratio even with Ledoit-Wolf shrinkage. PCA to 128-256 dimensions before fitting Mahalanobis could remove noise dimensions that capture identity rather than distribution and improve near-boundary detection. Quick experiment to validate.

### energy-ensemble-ood

Combine Mahalanobis distance with energy-based OOD scores from SigLIP2's logits and kNN density in embedding space. Multiple complementary signals catch different failure modes. `OOD = w1*mahalanobis + w2*energy + w3*knn(k=10)`. Estimated +5-10% AUROC on near-boundary with no additional data.

### conformal-prediction

Instead of optimizing AUROC, use conformal prediction to guarantee maximum error rate under exchangeability. Operationally more meaningful — "we catch 95% of unreliable predictions" rather than "AUROC is 0.99". Requires a proper calibration set with known OOD labels (hence depends on real-world-ood).

---

## Archive

*Implemented ideas move here with links to their experiment entries.*

| ID | Idea | Implemented In | Outcome |
|---|---|---|---|
| checkpoint-fix | Re-extract embeddings from correct IQA-only checkpoint | [EXP-006](experiments.md#exp-006-siglip2-full-extraction--ood-v2) | Resolved: v2 detector healthy, no anomalous shift |
| native-resolution | Full-scale validation of no-resize prompt strategy | [EXP-008](experiments.md#exp-008-prompt-optimization) | Disproved: -0.009 wSRCC vs baseline |
