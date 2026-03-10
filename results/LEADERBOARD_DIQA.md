# DIQA-5000 Test Set Leaderboard

Unified leaderboard for all models evaluated on the DIQA-5000 test set (n=1,000).

**Metric:** `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color`
where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting.

**Last updated:** 2026-03-10

---

## Overall Ranking

All models now use the same **MainScore** formula with 4-parameter logistic curve fitting before PLCC, enabling direct comparison across model types.

| Rank | Model | Type | MainScore | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | $/M in | $/M out | Notes |
| ---- | ----- | ---- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ----- |
| 1 | SigLIP2-IQA-Base-86M | Fine-tuned student | **0.886** | 0.896 | — | 0.869 | — | 0.885 | — | — | — | Trained on DIQA-5000; wSRCC; ~100ms inference |
| 2 | Gemini 3 Flash Preview | VLM teacher | **0.743** | 0.707 | 0.792 | 0.736 | 0.777 | 0.681 | 0.752 | $0.50 | $3.00 | Best VLM; zero-shot via OpenRouter |
| 3 | Qwen 3.5 122B-A10B | VLM teacher | **0.729** | 0.704 | 0.748 | 0.746 | 0.776 | 0.699 | 0.709 | $0.26 | $2.08 | Zero-shot via OpenRouter |
| 4 | Gemini 3.1 Flash Lite | VLM teacher | **0.722** | 0.702 | 0.766 | 0.722 | 0.780 | 0.650 | 0.683 | $0.25 | $1.50 | Zero-shot via OpenRouter |
| 5 | DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.716** | 0.733 | — | 0.681 | — | 0.716 | — | — | — | Trained on DIQA-5000 human labels; wSRCC only |
| 6 | GPT-4.1 | VLM teacher | **0.715** | 0.683 | 0.775 | 0.679 | 0.800 | 0.631 | 0.690 | $2.00 | $8.00 | Zero-shot via OpenRouter |
| 7 | Qwen 3.5 Plus | VLM teacher | **0.707** | 0.668 | 0.727 | 0.700 | 0.756 | 0.679 | 0.732 | $0.26 | $1.56 | Zero-shot via OpenRouter |
| 8 | Gemini 2.5 Pro | VLM teacher | **0.655** | 0.613 | 0.691 | 0.603 | 0.736 | 0.621 | 0.672 | $1.25 | $10.00 | 7% parse failures (n=930) |
| 9 | Qwen 3.5 Flash | VLM teacher | **0.626** | 0.560 | 0.625 | 0.643 | 0.734 | 0.608 | 0.653 | $0.10 | $0.40 | Zero-shot via OpenRouter |
| 10 | Claude Haiku 4.5 | VLM teacher | **0.601** | 0.598 | 0.650 | 0.539 | 0.603 | 0.579 | 0.593 | $1.00 | $5.00 | Best MAE (0.68) |
| 11 | Qwen3-VL-235B Instruct | VLM teacher | **0.598** | 0.639 | 0.658 | 0.595 | 0.661 | 0.434 | 0.494 | $0.20 | $0.88 | Zero-shot via OpenRouter |
| 12 | Seed 1.6 | VLM teacher | **0.562** | 0.606 | 0.595 | 0.531 | 0.573 | 0.485 | 0.508 | $0.25 | $2.00 | Zero-shot via OpenRouter |
| 13 | Qwen3-VL-30B Thinking | VLM teacher | **0.556** | 0.559 | 0.599 | 0.523 | 0.571 | 0.507 | 0.531 | free | free | CoT reasoning mode |
| 14 | Qwen3-VL-235B Thinking | VLM teacher | **0.541** | 0.526 | 0.535 | 0.512 | 0.536 | 0.573 | 0.582 | free | free | CoT reasoning mode |
| 15 | Mistral Small 3.1 24B | VLM teacher | **0.511** | 0.549 | 0.585 | 0.443 | 0.487 | 0.411 | 0.476 | $0.35 | $0.56 | Zero-shot via OpenRouter |
| 16 | Qwen3-VL-8B Instruct | VLM teacher | **0.505** | 0.520 | 0.564 | 0.437 | 0.537 | 0.446 | 0.449 | $0.08 | $0.50 | Zero-shot via OpenRouter |
| 17 | Seed 1.6 Flash | VLM teacher | **0.500** | 0.508 | 0.534 | 0.499 | 0.569 | 0.417 | 0.429 | $0.08 | $0.30 | Zero-shot via OpenRouter |
| 18 | RichIQA (TOPIQ-NR) | NR-IQA baseline | **0.490** | 0.489 | 0.483 | 0.498 | 0.484 | 0.507 | 0.488 | — | — | Off-the-shelf, KonIQ-10K pretrained |
| 19 | DBCNN | NR-IQA baseline | **0.453** | 0.444 | 0.446 | 0.466 | 0.458 | 0.466 | 0.457 | — | — | Off-the-shelf, KonIQ-10K pretrained |
| 20 | Qwen3-VL-8B Thinking | VLM teacher | **0.439** | 0.432 | 0.486 | 0.397 | 0.490 | 0.377 | 0.411 | $0.12 | $1.37 | CoT reasoning mode; n=998 |
| 21 | HyperIQA | NR-IQA baseline | **0.437** | 0.475 | 0.426 | 0.424 | 0.364 | 0.481 | 0.425 | — | — | Off-the-shelf, KonIQ-10K pretrained |
| 22 | Gemma 3 27B | VLM teacher | **0.431** | 0.438 | 0.521 | 0.444 | 0.503 | 0.265 | 0.313 | $0.03 | $0.11 | Zero-shot via OpenRouter |
| 23 | TReS | NR-IQA baseline | **0.422** | 0.447 | 0.414 | 0.397 | 0.367 | 0.463 | 0.425 | — | — | Off-the-shelf, KonIQ-10K pretrained |
| 24 | Gemma 3 12B | VLM teacher | **0.276** | 0.286 | 0.353 | 0.331 | 0.410 | 0.067 | 0.120 | $0.04 | $0.13 | Zero-shot via OpenRouter |
| 25 | Gemma 3 4B | VLM teacher | **0.270** | 0.284 | 0.326 | 0.249 | 0.302 | 0.189 | 0.204 | $0.04 | $0.08 | Zero-shot via OpenRouter |
| 26 | MUSIQ | NR-IQA baseline | **0.185** | 0.153 | 0.188 | 0.214 | 0.217 | 0.169 | 0.194 | — | — | Off-the-shelf, KonIQ-10K pretrained |
| 27 | Nemotron Nano 12B VL | VLM teacher | **0.185** | 0.181 | 0.171 | 0.251 | 0.252 | 0.134 | 0.136 | $0.20 | $0.60 | 2% parse failures (n=975) |
| 28 | Grok 4.1 Fast | VLM teacher | **0.114** | 0.102 | 0.132 | 0.034 | 0.085 | 0.141 | 0.184 | $0.20 | $0.50 | 1% parse failures (n=986) |
| — | StairIQA | NR-IQA baseline | N/A | — | — | — | — | — | — | — | — | Unavailable in pyiqa |

---

## Prompt Optimization Results (wSRCC only)

Best prompt-optimized variants for the top 3 VLM teachers, evaluated at n=1,000. See [Paper 3](../research/papers/03_prompt_engineering/paper.md) for methodology and statistical analysis.

### Gemini 3 Flash Preview

| Rank | Arm | wSRCC | SRCC_O | SRCC_S | SRCC_C | Bias_O | MAE_O | Calls/img | Notes |
| ---- | --- | ----- | ------ | ------ | ------ | ------ | ----- | --------- | ----- |
| 1 | **Few-shot (3 examples)** | **0.731** | 0.732 | 0.748 | 0.713 | +1.04 | 1.08 | 1 | Sig. vs baseline (paired bootstrap p=0.027) |
| 2 | **1-10 scale (rescaled)** | **0.726** | 0.724 | 0.756 | 0.702 | +0.75 | 0.79 | 1 | Best bias-correlation tradeoff |
| 3 | Baseline (1024px) | 0.708 | 0.707 | 0.736 | 0.681 | +0.76 | 0.80 | 1 | |
| 4 | 2048px resize | 0.704 | 0.701 | 0.727 | 0.686 | +0.79 | 0.85 | 1 | Resolution doesn't help Gemini |
| 5 | 0.5 increments | 0.698 | 0.695 | 0.707 | 0.697 | +0.63 | 0.72 | 1 | Lowest bias, hurts discrimination |
| 6 | Separate 3 prompts | 0.691 | 0.683 | 0.724 | 0.672 | +0.88 | 0.91 | 3 | Sig. worse (Holm p=0.036) |

### Qwen 3.5 Flash

| Rank | Arm | wSRCC | SRCC_O | SRCC_S | SRCC_C | Bias_O | MAE_O | Calls/img | Notes |
| ---- | --- | ----- | ------ | ------ | ------ | ------ | ----- | --------- | ----- |
| 1 | **2048px resize** | **0.671** | 0.652 | 0.702 | 0.677 | +1.50 | 1.50 | 1 | Sig. (Holm p<0.001); +0.078 over baseline |
| 2 | **No resize (native)** | **0.659** | 0.633 | 0.699 | 0.669 | +1.47 | 1.47 | 1 | Sig. (Holm p<0.001) |
| 3 | Few-shot (3 examples) | 0.630 | 0.612 | 0.651 | 0.644 | +1.27 | 1.29 | 1 | |
| 4 | 1-10 scale (rescaled) | 0.612 | 0.574 | 0.654 | 0.648 | +1.35 | 1.36 | 1 | Reduces bias -10% |
| 5 | Multi-sample (3x) | 0.605 | 0.573 | 0.643 | 0.629 | +1.50 | 1.50 | 3 | |
| 6 | Separate 3 prompts | 0.601 | 0.598 | 0.628 | 0.581 | +1.37 | 1.38 | 3 | |
| 7 | Baseline (1024px) | 0.593 | 0.560 | 0.643 | 0.608 | +1.50 | 1.50 | 1 | |
| 8 | 0.5 increments | 0.585 | 0.554 | 0.646 | 0.586 | +1.45 | 1.45 | 1 | |
| 9 | Hybrid | 0.514 | 0.485 | 0.551 | 0.534 | +1.49 | 1.50 | 2 | n=561 (still running) |

### Qwen 3.5 122B-A10B

| Rank | Arm | wSRCC | SRCC_O | SRCC_S | SRCC_C | Bias_O | MAE_O | Calls/img | Notes |
| ---- | --- | ----- | ------ | ------ | ------ | ------ | ----- | --------- | ----- |
| 1 | **No resize (native)** | **0.728** | 0.729 | 0.758 | 0.695 | +1.39 | 1.40 | 1 | Best wSRCC for this model |
| 2 | 1-10 scale (rescaled) | 0.720 | 0.708 | 0.742 | 0.721 | +1.23 | 1.23 | 1 | Best bias reduction (-12%) |
| 3 | 2048px resize | 0.717 | 0.717 | 0.738 | 0.695 | +1.43 | 1.43 | 1 | |
| 4 | Baseline (1024px) | 0.713 | 0.704 | 0.746 | 0.699 | +1.40 | 1.40 | 1 | |

### Key Findings

- **Best prompt strategy is model-dependent.** Few-shot helps Gemini (+0.023 wSRCC) but resolution helps Qwen Flash (+0.078) and Qwen 122B (+0.014). No single arm dominates across models.
- **Resolution sensitivity scales inversely with model capacity.** Qwen Flash (7B) gains +0.078 from 2048px; Qwen 122B (MoE) gains +0.004; Gemini gains nothing. Weaker models benefit most from higher-resolution input.
- **1-10 scale consistently reduces positive bias** across all models (-0% to -12%) while maintaining or improving correlation. This is the most reliably beneficial prompt intervention.
- **0.5 increments achieve lowest bias** (Gemini: +0.63) but hurt discrimination (-0.010 wSRCC). The coarser scale forces models to bin more aggressively.
- **Separate per-dimension prompting hurts Gemini** (sig. worse, Holm p=0.036) but marginally helps Qwen Flash. The 3x cost increase is not justified for either model.
- **Small-sample regression to mean confirmed.** The n=23 pilot identified no-resize as best for Gemini (+0.042); n=1,000 reversed this (-0.009). See Paper 3 for power analysis showing n≥200 is needed for reliable prompt selection.

---

## Notes

- **Unified MainScore**: All models now use the same formula: `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color` where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC uses 4-parameter logistic curve fitting. This eliminates the earlier apples-to-oranges comparison where VLMs used wSRCC and NR-IQA baselines used MainScore.
- **Fine-tuned student** (SigLIP2-IQA-Base-86M) scores are wSRCC from the model trained on DIQA-5000 human labels. Per-dimension PLCC not yet computed. See [research.md](../research.md) S3.
- **Fine-tuned MLLM** (DeQA-Doc-3Specialists) scores are wSRCC only (no PLCC reported for the specialist ensemble).
- **PLCC consistently exceeds SRCC** for VLM teachers (e.g., Gemini 3 Flash: PLCC_O=0.792 vs SRCC_O=0.707), confirming the 4-parameter logistic fit captures nonlinear prediction-to-MOS relationships.
- **Qwen 3.5 122B-A10B** is the new #1 VLM teacher (MainScore 0.729), surpassing GPT-4.1 and approaching Gemini 3 Flash Preview. Its strong sharpness correlation (SRCC_S=0.746) is the best among all VLMs.
- **Gemini 3.1 Flash Lite** (0.722) performs remarkably close to the full Gemini 3 Flash Preview (0.743) at lower cost.
- **Thinking/CoT models** (Qwen3-VL-30B/235B Thinking) do not outperform their instruct counterparts on this task, consistent with prior findings on Qwen3-VL-8B Thinking.
- **Grok 4.1 Fast** (0.134) and **Nemotron Nano 12B VL** (0.185) perform poorly on DIQA, with correlations near random — likely due to poor instruction following for structured JSON output.
- **Pricing** ($/M in, $/M out) is per million tokens via OpenRouter as of 2026-03-09. "free" indicates promotional free-tier models. NR-IQA and fine-tuned models run locally (no API cost).
- **Best value**: Gemini 3.1 Flash Lite ($0.25/$1.50) achieves 97% of top VLM quality at 50% of Gemini Flash Preview's cost. Qwen 3.5 Flash ($0.10/$0.40) offers 84% quality at 8% of GPT-4.1's cost.
- **z-ai/glm-5** and **z-ai/glm-4.7-flash** were excluded: no image input endpoints available on OpenRouter.
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
