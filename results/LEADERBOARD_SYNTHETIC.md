# Synthetic OOD Dataset Leaderboard

Unified leaderboard for all models evaluated on the 520-image synthetic dataset spanning 13 OOD categories.

**Metric:** `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color`
where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting.

**Last updated:** 2026-03-07

---

## Overall Ranking

All models now use the same **MainScore** formula with 4-parameter logistic curve fitting before PLCC, enabling direct comparison across model types.

| Rank | Model | Type | MainScore | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | Notes |
| ---- | ----- | ---- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ----- |
| 1 | Gemini 3 Flash Preview | VLM teacher | **0.774** | 0.753 | 0.804 | 0.775 | 0.815 | 0.668 | 0.822 | Best MainScore; PLCC boosts over GPT-4.1 |
| 2 | GPT-4.1 | VLM teacher | **0.769** | 0.764 | 0.788 | 0.797 | 0.820 | 0.704 | 0.730 | Best SRCC; #2 on MainScore |
| 3 | DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.748** | 0.696 | 0.765 | 0.778 | 0.832 | 0.687 | 0.766 | Best fine-tuned; smallest ID/OOD gap |
| 4 | TReS | NR-IQA baseline | **0.747** | 0.683 | 0.786 | 0.723 | 0.816 | 0.706 | 0.791 | Off-the-shelf, KonIQ-10K pretrained |
| 5 | HyperIQA | NR-IQA baseline | **0.723** | 0.639 | 0.798 | 0.643 | 0.823 | 0.639 | 0.802 | Off-the-shelf, KonIQ-10K pretrained |
| 6 | HyperIQA++ | Fine-tuned IQA | **0.694** | 0.589 | 0.780 | 0.623 | 0.797 | 0.606 | 0.790 | Fine-tuned on DIQA-5000; PLCC ≈ off-the-shelf |
| 7 | Claude Haiku 4.5 | VLM teacher | **0.660** | 0.582 | 0.717 | 0.630 | 0.756 | 0.570 | 0.724 | Better OOD than ID |
| 8 | SigLIP2-IQA-Base-86M | Fine-tuned student | **0.620** | 0.495 | 0.700 | 0.577 | 0.762 | 0.507 | 0.718 | 86M params; trained on pseudo-labels |
| 9 | RichIQA (TOPIQ-NR) | NR-IQA baseline | **0.619** | 0.482 | 0.735 | 0.507 | 0.770 | 0.499 | 0.746 | Off-the-shelf, KonIQ-10K pretrained |
| 10 | Qwen 3.5 Flash | VLM teacher | **0.596** | 0.550 | 0.604 | 0.603 | 0.623 | 0.583 | 0.649 | n=451 (13% parse failures) |
| 11 | DBCNN | NR-IQA baseline | **0.559** | 0.560 | 0.557 | 0.594 | 0.539 | 0.556 | 0.547 | Off-the-shelf, KonIQ-10K pretrained |
| 12 | Gemini 2.5 Pro | VLM teacher | **0.511** | 0.469 | 0.548 | 0.591 | 0.696 | 0.344 | 0.426 | n=425 (18% parse failures) |
| 13 | Qwen3-VL-8B Instruct | VLM teacher | **0.466** | 0.413 | 0.544 | 0.437 | 0.620 | 0.291 | 0.466 | Strong positive bias (+0.66) |
| 14 | Qwen3-VL-8B Thinking | VLM teacher | **0.458** | 0.430 | 0.490 | 0.485 | 0.524 | 0.373 | 0.439 | CoT reasoning; n=518 |
| 15 | MUSIQ | NR-IQA baseline | **0.289** | 0.252 | 0.340 | 0.199 | 0.316 | 0.258 | 0.351 | Off-the-shelf, KonIQ-10K pretrained |
| — | StairIQA | NR-IQA baseline | N/A | — | — | — | — | — | — | Unavailable in pyiqa |

---

## ID vs OOD Breakdown (VLM Teachers and Fine-Tuned Models)

| Model | Type | Main (all) | Main (ID) | Main (OOD) | ID-OOD Δ |
| ----- | ---- | ---------- | --------- | ---------- | -------- |
| Gemini 3 Flash Preview | VLM teacher | 0.774 | 0.824 | 0.782 | -0.042 |
| GPT-4.1 | VLM teacher | 0.769 | 0.825 | 0.757 | -0.068 |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.748 | 0.842 | 0.746 | -0.096 |
| HyperIQA++ | Fine-tuned IQA | 0.694 | 0.840 | 0.675 | **-0.165** |
| Qwen 3.5 Flash | VLM teacher | 0.596 | 0.442 | 0.667 | +0.225 |
| Claude Haiku 4.5 | VLM teacher | 0.660 | 0.539 | 0.706 | +0.167 |
| SigLIP2-IQA-Base-86M | Fine-tuned student | 0.620 | 0.659 | 0.663 | +0.004 |
| Gemini 2.5 Pro | VLM teacher | 0.511 | 0.398 | 0.549 | +0.151 |
| Qwen3-VL-8B Instruct | VLM teacher | 0.466 | 0.274 | 0.494 | +0.220 |
| Qwen3-VL-8B Thinking | VLM teacher | 0.458 | 0.306 | 0.477 | +0.171 |

With unified MainScore (including PLCC), the ID/OOD gap picture is more nuanced than wSRCC alone. Top VLMs (GPT-4.1, Gemini 3 Flash) still show strong ID performance (0.824–0.825) with modest OOD degradation. DeQA-Doc-3Specialists achieves the highest ID score of any model (0.842) while maintaining near-VLM OOD performance (0.746, delta=0.096). HyperIQA++ shows the largest fine-tuned gap (0.165) but is substantially less extreme than the wSRCC-only delta (0.205), because PLCC partially compensates for monotonicity failures. Weaker VLMs still show better OOD than ID (positive delta), confirming they lack DIQA-distribution-specific priors.

---

## Per-Category Overall SRCC (Top 2 VLMs + DeQA-Doc)

| Category | n | Gemini 3 Flash | GPT-4.1 | DeQA-Doc |
| -------- | - | -------------- | ------- | -------- |
| In-distribution (standard) | 100 | 0.790 | 0.785 | 0.744 |
| In-distribution (Cyrillic) | 50 | 0.808 | 0.758 | 0.765 |
| Non-Latin (Tibetan) | 30 | 0.800 | 0.730 | 0.488 |
| Non-Latin (Myanmar) | 30 | 0.763 | 0.764 | **0.786** |
| Non-Latin (Ethiopic) | 30 | 0.767 | 0.797 | **0.796** |
| Adversarial (Nastaliq) | 20 | 0.770 | 0.846 | 0.609 |
| Adversarial (Fraktur) | 20 | 0.768 | 0.762 | 0.705 |
| CJK vertical layout | 30 | 0.624 | 0.747 | **0.772** |
| Multiscript | 30 | 0.659 | 0.756 | 0.651 |
| Form layouts | 30 | 0.201 | 0.169 | 0.124 |
| Heavily degraded | 30 | 0.236 | 0.174 | 0.156 |
| Binarized | 30 | -0.340 | -0.372 | -0.490 |
| Pristine | 30 | 0.032 | -0.086 | 0.023 |
| Very high DPI | 30 | -0.150 | -0.109 | -0.200 |
| Very low DPI | 30 | -0.216 | -0.411 | 0.004 |

DeQA-Doc-3Specialists leads on CJK vertical (+0.025 vs GPT-4.1), Myanmar scripts, and Ethiopic scripts, but trails VLMs on adversarial categories and Tibetan. All models fail on binarized, extreme DPI, and pristine documents.

---

## Notes

- **Unified MainScore**: All models now use the same formula: `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color` where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting (`f(x) = b1 * (1 - 1/(1 + exp(b2*(x-b3)))) + b4`). This eliminates the earlier apples-to-oranges comparison where VLMs used wSRCC and NR-IQA baselines used MainScore.
- **PLCC consistently exceeds SRCC** across all model types (not just NR-IQA baselines), confirming the 4-parameter logistic fit captures nonlinear prediction-to-MOS relationships universal to quality assessment.
- **Synthetic ground truth** is derived from generation parameters (degradation level, noise intensity), not human MOS.
- The synthetic dataset's quality distributions are closer to natural image IQA, explaining why off-the-shelf NR-IQA baselines (TReS=0.747, HyperIQA=0.723) perform much better here than on DIQA-5000 (TReS=0.422, HyperIQA=0.437).
- **Qwen 3.5 Flash** (n=451) and **Gemini 2.5 Pro** (n=425) have parse failures reducing sample counts from 520.

## Data Sources

| Type | Location |
| ---- | -------- |
| VLM per-image scores | `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/*.jsonl` |
| Fine-tuned per-image scores | Modal volume `synthetic-ood-results` (`checkpoints_synthetic/*.jsonl`) |
| NR-IQA per-image scores | Modal volume `iqa-baseline-results` (JSONL checkpoints) |
| Fine-tuned aggregated metrics | `results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json` |
| VLM aggregated metrics (SRCC-only) | `results/vlm_teacher_eval/full_eval/results/synthetic_eval_metrics.json` |
| Unified metrics (all models, SRCC+PLCC) | `results/vlm_teacher_eval/full_eval/results/synthetic_eval_metrics_unified.json` |
| NR-IQA aggregated metrics | `results/iqa_baselines/baseline_summary.json` |
| Synthetic images + metadata | Modal volume `synthetic-ood-data` |
| DeQA-Doc checkpoints | Modal volume `deqa-specialist-checkpoints` |
| Benchmark script (fine-tuned) | `modal/benchmark_synthetic_ood.py` |
| Benchmark script (NR-IQA) | `modal/benchmark_iqa_baselines.py` |
| Evaluation doc | `results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md` |
