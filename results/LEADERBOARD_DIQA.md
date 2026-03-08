# DIQA-5000 Test Set Leaderboard

Unified leaderboard for all models evaluated on the DIQA-5000 test set (n=1,000).

**Metric:** `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color`
where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting.

**Last updated:** 2026-03-07

---

## Overall Ranking

All models now use the same **MainScore** formula with 4-parameter logistic curve fitting before PLCC, enabling direct comparison across model types.

| Rank | Model | Type | MainScore | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | Notes |
| ---- | ----- | ---- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ----- |
| 1 | SigLIP2-IQA-Base-86M | Fine-tuned student | **0.886** | 0.896 | — | 0.869 | — | 0.885 | — | Trained on DIQA-5000; wSRCC; ~100ms inference |
| 2 | Gemini 3 Flash Preview | VLM teacher | **0.743** | 0.707 | 0.792 | 0.736 | 0.777 | 0.681 | 0.752 | Best VLM; zero-shot via OpenRouter |
| 3 | DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.716** | 0.733 | — | 0.681 | — | 0.716 | — | Trained on DIQA-5000 human labels; wSRCC only |
| 4 | GPT-4.1 | VLM teacher | **0.715** | 0.683 | 0.775 | 0.679 | 0.800 | 0.631 | 0.690 | Zero-shot via OpenRouter |
| 5 | Gemini 2.5 Pro | VLM teacher | **0.655** | 0.613 | 0.691 | 0.603 | 0.736 | 0.621 | 0.672 | 7% parse failures (n=930) |
| 6 | Qwen 3.5 Flash | VLM teacher | **0.626** | 0.560 | 0.625 | 0.643 | 0.734 | 0.608 | 0.653 | Zero-shot via OpenRouter |
| 7 | Claude Haiku 4.5 | VLM teacher | **0.601** | 0.598 | 0.650 | 0.539 | 0.603 | 0.579 | 0.593 | Best MAE (0.68) |
| 8 | Qwen3-VL-8B Instruct | VLM teacher | **0.505** | 0.520 | 0.564 | 0.437 | 0.537 | 0.446 | 0.449 | Zero-shot via OpenRouter |
| 9 | RichIQA (TOPIQ-NR) | NR-IQA baseline | **0.490** | 0.489 | 0.483 | 0.498 | 0.484 | 0.507 | 0.488 | Off-the-shelf, KonIQ-10K pretrained |
| 10 | DBCNN | NR-IQA baseline | **0.453** | 0.444 | 0.446 | 0.466 | 0.458 | 0.466 | 0.457 | Off-the-shelf, KonIQ-10K pretrained |
| 11 | Qwen3-VL-8B Thinking | VLM teacher | **0.439** | 0.432 | 0.486 | 0.397 | 0.490 | 0.377 | 0.411 | CoT reasoning mode; n=998 |
| 12 | HyperIQA | NR-IQA baseline | **0.437** | 0.475 | 0.426 | 0.424 | 0.364 | 0.481 | 0.425 | Off-the-shelf, KonIQ-10K pretrained |
| 13 | TReS | NR-IQA baseline | **0.422** | 0.447 | 0.414 | 0.397 | 0.367 | 0.463 | 0.425 | Off-the-shelf, KonIQ-10K pretrained |
| 14 | MUSIQ | NR-IQA baseline | **0.185** | 0.153 | 0.188 | 0.214 | 0.217 | 0.169 | 0.194 | Off-the-shelf, KonIQ-10K pretrained |
| — | StairIQA | NR-IQA baseline | N/A | — | — | — | — | — | — | Unavailable in pyiqa |

---

## Notes

- **Unified MainScore**: All models now use the same formula: `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color` where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting. This eliminates the earlier apples-to-oranges comparison where VLMs used wSRCC and NR-IQA baselines used MainScore.
- **Fine-tuned student** (SigLIP2-IQA-Base-86M) scores are wSRCC from the model trained on DIQA-5000 human labels. Per-dimension PLCC not yet computed. See [research.md](../research.md) S3.
- **Fine-tuned MLLM** (DeQA-Doc-3Specialists) scores are wSRCC only (no PLCC reported for the specialist ensemble).
- **PLCC consistently exceeds SRCC** for VLM teachers (e.g., Gemini 3 Flash: PLCC_O=0.792 vs SRCC_O=0.707), confirming the 4-parameter logistic fit captures nonlinear prediction-to-MOS relationships.
- NR-IQA baselines use pretrained checkpoints from pyiqa (no DIQA fine-tuning). The "Reported" competition scores (after fine-tuning) were: DBCNN=0.587, HyperIQA=0.844, StairIQA=0.850, MUSIQ=0.859, TReS=0.863, RichIQA=0.866.

## Data Sources

| Type | Location |
| ---- | -------- |
| VLM per-image scores | `results/vlm_teacher_eval/full_eval/checkpoints/*.jsonl` |
| NR-IQA per-image scores | Modal volume `iqa-baseline-results` (JSONL checkpoints) |
| NR-IQA aggregated metrics | `results/iqa_baselines/baseline_summary.json` |
| VLM aggregated metrics | `results/vlm_teacher_eval/full_eval/results/vlm_benchmark_metrics.json` |
| Ground truth | `results/vlm_teacher_eval/full_eval/data/test.csv` |
| Benchmark script (NR-IQA) | `modal/benchmark_iqa_baselines.py` |
| Evaluation doc | `results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md` |
