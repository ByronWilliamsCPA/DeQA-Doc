# DeQA-Doc Research Index

**Project**: Document Image Quality Assessment via VLM Pseudo-Labeling
**Goal**: Improve SigLIP2-IQA-Base from VQualA 0.886 to 0.92+ by expanding training data beyond DIQA-5000
**Last Updated**: 2026-03-08

---

## Status Dashboard

| # | Workstream | Status | Key Metric | Details |
|---|-----------|--------|------------|---------|
| 1 | VQualA 2025 Championship (mPLUG-Owl2) | DONE | wSRCC 0.929 | [Readme.md](../Readme.md) |
| 2 | SigLIP2-IQA-Base training | DONE | VQualA 0.886 | [research.md](../research.md) S3 |
| 3 | VLM smoke test (Claude Sonnet/Opus) | DONE | Scores identical across providers | [EXP-001](experiments.md#exp-001-vlm-smoke-test) |
| 4 | VLM model selection (10 models, n=7) | DONE | Ranked top 10 candidates | [EXP-002](experiments.md#exp-002-vlm-model-selection) |
| 5 | IQA baseline benchmarking (6 models) | DONE | Best: RichIQA wSRCC 0.490 | [EXP-003](experiments.md#exp-003-iqa-baseline-benchmarking) |
| 6 | Synthetic OOD dataset creation | DONE | 520 images, 13 categories | [EXP-004](experiments.md#exp-004-synthetic-ood-dataset-creation) |
| 7 | OOD detector v1 (Mahalanobis) | DONE | AUROC 0.9963 (synthetic) | [EXP-005](experiments.md#exp-005-ood-detector-v1) |
| 8 | SigLIP2 full extraction + OOD v2 | DONE | Checkpoint mismatch resolved | [EXP-006](experiments.md#exp-006-siglip2-full-extraction--ood-v2) |
| 9 | VLM teacher evaluation (7 models, 1K images) | DONE | Best: Gemini 3 Flash wSRCC 0.708 | [EXP-007](experiments.md#exp-007-vlm-teacher-evaluation) |
| 10 | Prompt optimization (7-arm + A/B) | DONE | No improvement over default | [EXP-008](experiments.md#exp-008-prompt-optimization) |
| 11 | 13-model consensus analysis | DONE | 8 strategies ranked | [EXP-009](experiments.md#exp-009-13-model-consensus-analysis) |
| 12 | Pseudo-labeling pipeline (code) | IN PROGRESS | 87 tests passing | [src/uncertainty/](../DeQA-Score/src/uncertainty/) |
| 13 | Real-world OOD evaluation | NOT STARTED | -- | [backlog.md#real-world-ood](backlog.md#real-world-ood) |
| 14 | VLM score calibration | NOT STARTED | -- | [backlog.md#calibration](backlog.md#calibration) |
| 15 | End-to-end pseudo-label training | NOT STARTED | -- | [backlog.md#e2e-validation](backlog.md#e2e-validation) |

---

## Key Results Summary

| Metric | Value | Source |
|--------|-------|--------|
| DeQA-Doc championship score | 0.929 wSRCC | VQualA 2025 |
| SigLIP2-IQA-Base current | 0.886 VQualA | [research.md](../research.md) S3 |
| Target | 0.92 VQualA | [research.md](../research.md) S1 |
| Best VLM teacher | 0.708 wSRCC (Gemini 3 Flash) | [EXP-007](experiments.md#exp-007-vlm-teacher-evaluation) |
| 3Specialist baseline | 0.716 wSRCC | [VLM_TEACHER_EVALUATION.md](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S2.3 |
| OOD detector AUROC (synthetic) | 0.9963 | [EXP-005](experiments.md#exp-005-ood-detector-v1) |
| Best IQA baseline on DIQA | 0.490 wSRCC (RichIQA) | [EXP-003](experiments.md#exp-003-iqa-baseline-benchmarking) |

---

## Document Map

| File | Purpose | Size |
|------|---------|------|
| **This index** | | |
| [experiments.md](experiments.md) | Experiment registry (9 completed experiments) | ~250 lines |
| [backlog.md](backlog.md) | Hypothesis/idea tracker with priorities | ~120 lines |
| [data-inventory.md](data-inventory.md) | All datasets, artifacts, and scripts | ~120 lines |
| **Research papers** | | |
| [diqa_1.md](diqa_1.md) | Literature survey: DIQA-5000 dataset and VQualA 2025 analysis | 259 lines |
| [diqa_2.md](diqa_2.md) | Competition landscape: all 7 DIQA teams and methods | ~100 lines |
| [diqa_3.md](diqa_3.md) | Baseline MLLMs vs Traditional Vision ML: before and after training | ~260 lines |
| [diqa_4_siglip2_training.md](diqa_4_siglip2_training.md) | Training SigLIP2-IQA-Base-86M: architecture, protocol, and results | ~250 lines |
| [diqa_5_hyperiqa_training.md](diqa_5_hyperiqa_training.md) | Training HyperIQA++: CNN fine-tuning for document IQA | ~250 lines |
| [diqa_6_ood_test_analysis.md](diqa_6_ood_test_analysis.md) | Model performance on 520-image synthetic OOD test set | ~350 lines |
| **Technical notes** | | |
| [research.md](../research.md) | Technical handoff: VLM distillation strategy and execution plan | 211 lines |
| **Leaderboards** | | |
| [LEADERBOARD_DIQA.md](../results/LEADERBOARD_DIQA.md) | Unified ranking: all models on DIQA-5000 test (n=1,000) | ~50 lines |
| [LEADERBOARD_SYNTHETIC.md](../results/LEADERBOARD_SYNTHETIC.md) | Unified ranking: all models on synthetic OOD (n=520) | ~85 lines |
| **Results documentation** | | |
| [VLM_TEACHER_EVALUATION.md](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) | Systematic VLM teacher evaluation (quasi-paper) | 595 lines |
| [tier1_ood_detector/README.md](../results/tier1_ood_detector/README.md) | OOD detector architecture and usage | ~120 lines |
| [NEXT_ITERATION_ANALYSIS.md](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) | 13-model consensus: next iteration strategy | 514 lines |
| [siglip2_diqa5000/summary.json](../results/siglip2_diqa5000/summary.json) | SigLIP2 extraction metadata and OOD v2 stats | JSON |
| **Handoff documents** | | |
| [MODAL_SYNTHETIC_EVAL_HANDOFF.md](../results/vlm_teacher_eval/full_eval/MODAL_SYNTHETIC_EVAL_HANDOFF.md) | Fine-tuned IQA eval on synthetic OOD | ~100 lines |
| [MODAL_SIGLIP2_DIQA5000_HANDOFF.md](../results/vlm_teacher_eval/full_eval/MODAL_SIGLIP2_DIQA5000_HANDOFF.md) | SigLIP2 full extraction task spec | ~100 lines |
| **API stability** | | |
| [STABILITY.md](../STABILITY.md) | Stable API surface and level ordering convention | 61 lines |
