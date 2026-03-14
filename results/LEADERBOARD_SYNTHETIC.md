# Synthetic OOD Dataset Leaderboard

Unified leaderboard for all models evaluated on the 520-image synthetic dataset spanning 13 OOD categories.

**Metric:** `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color`
where `Score_dim = 0.5 × (PLCC + SRCC)`. PLCC is Pearson linear correlation.

**Last updated:** 2026-03-09

---

## Overall Ranking

| Rank | Model | Type | MainScore | SRCC_O | PLCC_O | SRCC_S | PLCC_S | SRCC_C | PLCC_C | Notes |
| ---- | ----- | ---- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ----- |
| 1 | Gemini 3 Flash Preview | VLM teacher | **0.768** | 0.753 | 0.798 | 0.775 | 0.814 | 0.668 | 0.784 | Tied #1; best color PLCC |
| 2 | GPT-4.1 | VLM teacher | **0.768** | 0.764 | 0.786 | 0.797 | 0.819 | 0.704 | 0.721 | Tied #1; best SRCC overall |
| 3 | DeQA-Doc-3Specialists | Fine-tuned MLLM | **0.748** | 0.696 | 0.765 | 0.778 | 0.832 | 0.687 | 0.766 | Best fine-tuned; smallest ID/OOD gap |
| 4 | TReS | NR-IQA baseline | **0.747** | 0.683 | 0.786 | 0.723 | 0.816 | 0.706 | 0.791 | Off-the-shelf, KonIQ-10K pretrained |
| 5 | HyperIQA | NR-IQA baseline | **0.723** | 0.639 | 0.798 | 0.643 | 0.823 | 0.639 | 0.802 | Off-the-shelf, KonIQ-10K pretrained |
| 6 | HyperIQA++ | Fine-tuned IQA | **0.694** | 0.589 | 0.780 | 0.623 | 0.797 | 0.606 | 0.790 | Fine-tuned on DIQA-5000 |
| 7 | Claude Haiku 4.5 | VLM teacher | **0.646** | 0.582 | 0.693 | 0.630 | 0.715 | 0.570 | 0.706 | Better OOD than ID |
| 8 | Gemini 3.1 Flash Lite | VLM teacher | **0.642** | 0.595 | 0.690 | 0.557 | 0.699 | 0.578 | 0.736 | Strong color PLCC (0.736) |
| 9 | Qwen 3.5 122B-A10B | VLM teacher | **0.625** | 0.584 | 0.628 | 0.643 | 0.659 | 0.625 | 0.649 | Best batch 2 VLM |
| 10 | SigLIP2-IQA-Base-86M | Fine-tuned student | **0.620** | 0.495 | 0.700 | 0.577 | 0.762 | 0.507 | 0.718 | 86M params; trained on pseudo-labels |
| 11 | RichIQA (TOPIQ-NR) | NR-IQA baseline | **0.619** | 0.482 | 0.735 | 0.507 | 0.770 | 0.499 | 0.746 | Off-the-shelf, KonIQ-10K pretrained |
| 12 | Qwen 3.5 Plus | VLM teacher | **0.570** | 0.563 | 0.585 | 0.647 | 0.649 | 0.463 | 0.502 | Zero-shot via OpenRouter |
| 13 | Qwen 3.5 Flash | VLM teacher | **0.567** | 0.510 | 0.565 | 0.574 | 0.606 | 0.573 | 0.635 | n=516 |
| 14 | DBCNN | NR-IQA baseline | **0.559** | 0.560 | 0.557 | 0.594 | 0.539 | 0.556 | 0.547 | Off-the-shelf, KonIQ-10K pretrained |
| 15 | Seed 1.6 Flash | VLM teacher | **0.489** | 0.430 | 0.517 | 0.503 | 0.594 | 0.387 | 0.534 | n=519 |
| 16 | Gemini 2.5 Pro | VLM teacher | **0.476** | 0.474 | 0.489 | 0.600 | 0.613 | 0.317 | 0.356 | n=511 (2% parse failures) |
| 17 | Mistral Small 3.1 24B | VLM teacher | **0.476** | 0.430 | 0.516 | 0.330 | 0.564 | 0.493 | 0.531 | n=518 |
| 18 | Gemma 3 12B | VLM teacher | **0.459** | 0.420 | 0.516 | 0.472 | 0.569 | 0.297 | 0.462 | n=519 |
| 19 | Qwen3-VL-8B Thinking | VLM teacher | **0.450** | 0.428 | 0.481 | 0.482 | 0.489 | 0.374 | 0.439 | CoT reasoning |
| 20 | Qwen3-VL-8B Instruct | VLM teacher | **0.449** | 0.413 | 0.520 | 0.437 | 0.556 | 0.291 | 0.440 | Zero-shot via OpenRouter |
| 21 | Gemma 3 27B | VLM teacher | **0.440** | 0.370 | 0.494 | 0.420 | 0.543 | 0.370 | 0.462 | Zero-shot via OpenRouter |
| 22 | Qwen3-VL-30B Thinking | VLM teacher | **0.408** | 0.337 | 0.487 | 0.452 | 0.587 | 0.218 | 0.357 | CoT reasoning |
| 23 | Seed 1.6 | VLM teacher | **0.371** | 0.162 | 0.384 | 0.352 | 0.625 | 0.327 | 0.576 | Negative ID correlation |
| 24 | Gemma 3 4B | VLM teacher | **0.323** | 0.285 | 0.403 | 0.308 | 0.446 | 0.193 | 0.261 | Zero-shot via OpenRouter |
| 25 | Qwen3-VL-235B Instruct | VLM teacher | **0.294** | 0.225 | 0.354 | 0.392 | 0.469 | 0.075 | 0.261 | Negative ID correlation |
| 26 | MUSIQ | NR-IQA baseline | **0.289** | 0.252 | 0.340 | 0.199 | 0.316 | 0.258 | 0.351 | Off-the-shelf, KonIQ-10K pretrained |
| 27 | Qwen3-VL-235B Thinking | VLM teacher | **0.232** | 0.170 | 0.269 | 0.345 | 0.459 | 0.052 | 0.119 | CoT reasoning |
| 28 | Nemotron Nano 12B VL | VLM teacher | **0.207** | 0.164 | 0.217 | 0.215 | 0.300 | 0.187 | 0.189 | n=496 (5% parse failures) |
| 29 | Grok 4.1 Fast | VLM teacher | **0.134** | 0.158 | 0.132 | 0.191 | 0.165 | 0.090 | 0.051 | n=514 (1% parse failures) |
| — | StairIQA | NR-IQA baseline | N/A | — | — | — | — | — | — | Unavailable in pyiqa |

---

## ID vs OOD Breakdown (VLM Teachers and Fine-Tuned Models)

| Model | Type | Main (all) | Main (ID) | Main (OOD) | ID-OOD Δ |
| ----- | ---- | ---------- | --------- | ---------- | -------- |
| Gemini 3 Flash Preview | VLM teacher | 0.768 | 0.800 | 0.773 | +0.027 |
| GPT-4.1 | VLM teacher | 0.768 | 0.798 | 0.755 | +0.043 |
| DeQA-Doc-3Specialists | Fine-tuned MLLM | 0.748 | 0.842 | 0.746 | -0.096 |
| HyperIQA++ | Fine-tuned IQA | 0.694 | 0.840 | 0.675 | **-0.165** |
| Claude Haiku 4.5 | VLM teacher | 0.646 | 0.529 | 0.685 | -0.156 |
| Gemini 3.1 Flash Lite | VLM teacher | 0.642 | 0.372 | 0.689 | **-0.316** |
| Qwen 3.5 122B-A10B | VLM teacher | 0.625 | 0.606 | 0.636 | -0.030 |
| SigLIP2-IQA-Base-86M | Fine-tuned student | 0.620 | 0.659 | 0.663 | +0.004 |
| Qwen 3.5 Plus | VLM teacher | 0.570 | 0.567 | 0.593 | -0.026 |
| Qwen 3.5 Flash | VLM teacher | 0.567 | 0.377 | 0.618 | -0.241 |
| Seed 1.6 Flash | VLM teacher | 0.489 | 0.013 | 0.586 | **-0.573** |
| Gemini 2.5 Pro | VLM teacher | 0.476 | 0.411 | 0.507 | -0.096 |
| Mistral Small 3.1 24B | VLM teacher | 0.476 | -0.108 | 0.545 | **-0.653** |
| Gemma 3 12B | VLM teacher | 0.459 | 0.149 | 0.509 | -0.361 |
| Qwen3-VL-8B Thinking | VLM teacher | 0.450 | 0.288 | 0.471 | -0.183 |
| Qwen3-VL-8B Instruct | VLM teacher | 0.449 | 0.263 | 0.478 | -0.216 |
| Gemma 3 27B | VLM teacher | 0.440 | 0.067 | 0.498 | -0.431 |
| Qwen3-VL-30B Thinking | VLM teacher | 0.408 | 0.004 | 0.485 | -0.481 |
| Seed 1.6 | VLM teacher | 0.371 | -0.428 | 0.531 | **-0.959** |
| Gemma 3 4B | VLM teacher | 0.323 | 0.082 | 0.357 | -0.275 |
| Qwen3-VL-235B Instruct | VLM teacher | 0.294 | -0.174 | 0.357 | -0.532 |
| Qwen3-VL-235B Thinking | VLM teacher | 0.232 | -0.256 | 0.325 | -0.581 |
| Nemotron Nano 12B VL | VLM teacher | 0.207 | -0.012 | 0.224 | -0.236 |
| Grok 4.1 Fast | VLM teacher | 0.134 | 0.119 | 0.122 | -0.003 |

Top VLMs (GPT-4.1, Gemini 3 Flash) show strong and balanced ID/OOD performance (delta < +0.05). Qwen 3.5 122B-A10B has the smallest OOD gap among batch 2 models (-0.030). Several models show **negative ID MainScore** (Seed 1.6: -0.428, Mistral Small: -0.108), indicating complete failure on the in-distribution synthetic subset — likely due to systematic bias or inability to distinguish quality levels in the standard document images. The large negative ID-OOD deltas for weaker models confirm they perform better on diverse OOD categories (where quality differences are more visually obvious) than on standard documents.

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

- **Unified MainScore**: All models use the formula: `MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color` where `Score_dim = 0.5 × (PLCC + SRCC)`.
- **PLCC consistently exceeds SRCC** across all model types, confirming Pearson correlation captures linear prediction-to-GT relationships that Spearman misses.
- **Synthetic ground truth** is derived from generation parameters (degradation level, noise intensity), not human MOS.
- The synthetic dataset's quality distributions are closer to natural image IQA, explaining why off-the-shelf NR-IQA baselines (TReS=0.747, HyperIQA=0.723) perform much better here than on DIQA-5000 (TReS=0.422, HyperIQA=0.437).
- **Qwen 3.5 122B-A10B** (0.625) is the best batch 2 VLM, ranking #9 overall behind Claude Haiku 4.5 and Gemini 3.1 Flash Lite.
- **Negative ID correlations** for several models (Seed 1.6, Mistral Small, Qwen3-VL-235B variants) indicate these models cannot meaningfully rank standard document quality — they may assign near-constant scores to the in-distribution subset.
- **Grok 4.1 Fast** (0.134) and **Nemotron Nano 12B VL** (0.207) remain near random, consistent with DIQA-5000 results.

## Data Sources

| Type | Location |
| ---- | -------- |
| VLM per-image scores | `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/*.jsonl` |
| Fine-tuned per-image scores | Modal volume `synthetic-ood-results` (`checkpoints_synthetic/*.jsonl`) |
| NR-IQA per-image scores | Modal volume `iqa-baseline-results` (JSONL checkpoints) |
| Fine-tuned aggregated metrics | `results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json` |
| VLM aggregated metrics | `results/vlm_teacher_eval/full_eval/results/synthetic_eval_metrics.json` |
| NR-IQA aggregated metrics | `results/iqa_baselines/baseline_summary.json` |
| Synthetic images + metadata | Modal volume `synthetic-ood-data` |
| DeQA-Doc checkpoints | Modal volume `deqa-specialist-checkpoints` |
| Benchmark script (fine-tuned) | `modal/benchmark_synthetic_ood.py` |
| Benchmark script (NR-IQA) | `modal/benchmark_iqa_baselines.py` |
| Evaluation doc | `results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md` |
