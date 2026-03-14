# DIQA-5000 Sample Evaluation — 26 Models x 7 Images

**Date**: 2026-03-07
**Branch**: `docs/ood-detector-next-iteration-analysis`
**Script**: `run_diqa_eval.py`
**Raw data**: `diqa_eval_raw_results.json`, `diqa_eval_metrics.json`

## Overview

Evaluated 26 VLM models on 7 DIQA-5000 test images spanning all 15 quality categories
(5 quality levels x 3 dimensions). All models accessed via OpenRouter. 182/182 API calls
succeeded (100% success rate).

**Goal**: Identify which VLM teachers produce scores best correlated with human MOS (Mean
Opinion Score) for use as pseudo-label generators in the DeQA training pipeline.

**Metric**: VQualA weighted SRCC = `0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color`

## Sample Images

Selected via greedy set-cover to minimize image count while covering all 15 quality buckets.
Quality buckets: bad [1.0-1.8), poor [1.8-2.6), fair [2.6-3.4), good [3.4-4.0), excellent [4.0-5.0).

| Image | MOS Overall | MOS Sharpness | MOS Color | Category (O/S/C) |
|-------|-------------|---------------|-----------|------------------|
| test_res_00354.jpg | 4.053 | 4.027 | 4.067 | excellent / excellent / excellent |
| test_res_00001.jpg | 3.760 | 3.653 | 3.707 | good / good / good |
| test_res_00008.jpg | 2.807 | 2.847 | 2.927 | fair / fair / fair |
| test_res_00756.jpg | 2.700 | 2.407 | 3.020 | fair / poor / fair |
| test_res_00052.jpg | 2.293 | 2.507 | 2.360 | poor / poor / poor |
| test_res_00316.jpg | 1.840 | 1.500 | 2.080 | poor / bad / poor |
| test_res_00312.jpg | 1.700 | 1.653 | 1.667 | bad / bad / bad |

Average MOS: O=2.736, S=2.656, C=2.833

## Model Rankings by Weighted SRCC

| Rank | Model | Tier | wSRCC | SRCC_O | SRCC_S | SRCC_C | PLCC_O | PLCC_S | PLCC_C | MAE_O | MAE_S | MAE_C |
|------|-------|------|-------|--------|--------|--------|--------|--------|--------|-------|-------|-------|
| 1 | openai/gpt-4.1 | Strong | **0.880** | 0.937 | 0.955 | 0.691 | 0.872 | 0.916 | 0.687 | 0.750 | 0.844 | 1.067 |
| 2 | google/gemini-3-flash-preview | Value | **0.857** | 0.873 | 0.847 | 0.837 | 0.757 | 0.754 | 0.832 | 0.935 | 1.246 | 0.838 |
| 3 | openai/gpt-5-mini | Value | **0.841** | 0.768 | 0.906 | 0.919 | 0.680 | 0.841 | 0.871 | 0.822 | 0.751 | 0.753 |
| 4 | google/gemini-2.5-pro | Frontier | **0.829** | 0.893 | 0.821 | 0.709 | 0.893 | 0.747 | 0.709 | 0.331 | 0.611 | 0.606 |
| 5 | anthropic/claude-sonnet-4.6 | Anthropic | **0.827** | 0.879 | 0.945 | 0.606 | 0.888 | 0.858 | 0.675 | 0.579 | 0.566 | 0.910 |
| 6 | openai/gpt-5.1 | Frontier | **0.826** | 0.847 | 0.889 | 0.721 | 0.821 | 0.846 | 0.790 | 0.750 | 1.087 | 0.939 |
| 7 | openai/gpt-5 | Frontier | **0.824** | 0.750 | 0.937 | 0.857 | 0.830 | 0.859 | 0.892 | 0.578 | 0.915 | 0.690 |
| 8 | anthropic/claude-haiku-4.5 | Anthropic | **0.813** | 0.867 | 0.906 | 0.612 | 0.823 | 0.865 | 0.630 | 0.581 | 0.530 | 0.812 |
| 9 | google/gemini-2.5-flash | Value | **0.778** | 0.818 | 0.857 | 0.618 | 0.742 | 0.792 | 0.726 | 0.735 | 0.872 | 0.702 |
| 10 | anthropic/claude-opus-4.6 | Anthropic | **0.742** | 0.704 | 0.847 | 0.711 | 0.766 | 0.752 | 0.708 | 0.596 | 0.667 | 0.631 |
| 11 | google/gemini-3.1-pro-preview | Frontier | **0.729** | 0.821 | 0.857 | 0.414 | 0.775 | 0.779 | 0.496 | 0.975 | 1.189 | 1.353 |
| 12 | qwen/qwen3-vl-32b-instruct | Value | **0.707** | 0.733 | 0.767 | 0.593 | 0.784 | 0.851 | 0.539 | 0.922 | 0.966 | 0.967 |
| 13 | openai/gpt-4.1-mini | Value | **0.684** | 0.694 | 0.767 | 0.582 | 0.706 | 0.801 | 0.499 | 0.821 | 0.846 | 1.539 |
| 14 | openai/gpt-5.2 | Frontier | **0.615** | 0.800 | 0.786 | 0.072 | 0.732 | 0.681 | 0.314 | 0.668 | 0.780 | 1.182 |
| 15 | meta-llama/llama-4-maverick | Value | **0.610** | 0.458 | 0.657 | 0.866 | 0.597 | 0.682 | 0.829 | 0.724 | 0.665 | 0.615 |
| 16 | qwen/qwen-vl-max | VL | **0.585** | 0.624 | 0.364 | 0.727 | 0.519 | 0.380 | 0.749 | 1.607 | 1.758 | 1.553 |
| 17 | qwen/qwen3-vl-235b-a22b-instruct | Strong | **0.514** | 0.524 | 0.593 | 0.414 | 0.571 | 0.578 | 0.431 | 1.235 | 1.415 | 1.267 |
| 18 | mistralai/mistral-large-2512 | Value | **0.489** | 0.612 | 0.463 | 0.267 | 0.645 | 0.603 | 0.151 | 0.925 | 0.944 | 1.110 |
| 19 | nvidia/nemotron-nano-12b-v2-vl | VL | **0.473** | 0.404 | 0.630 | 0.455 | 0.448 | 0.561 | 0.356 | 1.022 | 0.980 | 1.359 |
| 20 | mistralai/pixtral-large-2411 | Strong | **0.468** | 0.449 | 0.617 | 0.356 | 0.570 | 0.593 | 0.472 | 0.992 | 0.944 | 1.044 |
| 21 | qwen/qwen2.5-vl-72b-instruct | Strong | **0.463** | 0.474 | 0.661 | 0.243 | 0.486 | 0.683 | 0.170 | 1.124 | 1.201 | 1.155 |
| 22 | baidu/ernie-4.5-vl-424b-a47b | VL | **0.393** | 0.599 | 0.509 | -0.134 | 0.679 | 0.644 | -0.092 | 0.852 | 1.014 | 1.739 |
| 23 | qwen/qwen3-vl-8b-instruct | VL | **0.317** | 0.259 | 0.319 | 0.433 | 0.408 | 0.381 | 0.408 | 1.407 | 1.344 | 1.453 |
| 24 | qwen/qwen-2.5-vl-7b-instruct | VL | **0.305** | 0.289 | 0.154 | 0.487 | 0.439 | 0.128 | 0.431 | 1.635 | 1.715 | 1.753 |
| 25 | x-ai/grok-4-fast | Value | **0.140** | 0.126 | 0.396 | -0.090 | 0.056 | 0.234 | -0.098 | 0.922 | 0.826 | 1.107 |
| 26 | qwen/qwen2.5-vl-32b-instruct | VL | **-0.130** | -0.113 | -0.275 | -0.019 | -0.166 | -0.275 | -0.083 | 1.352 | 1.445 | 1.212 |

## Calibration Analysis — Average Predicted vs MOS

Human MOS average: O=2.736, S=2.656, C=2.833

| Model | Avg O | Avg S | Avg C | Bias O | Bias S | Bias C | Avg Bias |
|-------|-------|-------|-------|--------|--------|--------|----------|
| google/gemini-2.5-pro | 2.614 | 2.871 | 2.943 | -0.122 | +0.215 | +0.110 | **+0.068** |
| x-ai/grok-4-fast | 2.929 | 2.686 | 3.429 | +0.192 | +0.029 | +0.596 | **+0.273** |
| anthropic/claude-opus-4.6 | 3.100 | 3.043 | 3.386 | +0.364 | +0.387 | +0.553 | **+0.435** |
| anthropic/claude-haiku-4.5 | 3.157 | 3.000 | 3.586 | +0.421 | +0.344 | +0.753 | **+0.506** |
| anthropic/claude-sonnet-4.6 | 3.243 | 3.129 | 3.743 | +0.507 | +0.472 | +0.910 | **+0.630** |
| openai/gpt-5 | 3.229 | 3.571 | 3.329 | +0.492 | +0.915 | +0.496 | **+0.634** |
| meta-llama/llama-4-maverick | 3.214 | 3.071 | 3.429 | +0.478 | +0.415 | +0.596 | **+0.496** |
| openai/gpt-5.2 | 3.343 | 3.371 | 4.014 | +0.607 | +0.715 | +1.182 | **+0.835** |
| openai/gpt-4.1 | 3.486 | 3.500 | 3.900 | +0.750 | +0.844 | +1.067 | **+0.887** |
| openai/gpt-5-mini | 3.543 | 3.400 | 3.586 | +0.807 | +0.744 | +0.753 | **+0.768** |
| openai/gpt-5.1 | 3.486 | 3.743 | 3.771 | +0.750 | +1.087 | +0.939 | **+0.925** |
| google/gemini-2.5-flash | 3.471 | 3.529 | 3.529 | +0.735 | +0.872 | +0.696 | **+0.768** |
| google/gemini-3-flash-preview | 3.529 | 3.729 | 3.457 | +0.792 | +1.072 | +0.625 | **+0.830** |
| google/gemini-3.1-pro-preview | 3.500 | 3.586 | 4.186 | +0.764 | +0.929 | +1.353 | **+1.015** |
| qwen/qwen3-vl-32b-instruct | 3.643 | 3.614 | 3.800 | +0.907 | +0.958 | +0.967 | **+0.944** |
| openai/gpt-4.1-mini | 3.357 | 3.329 | 4.371 | +0.621 | +0.672 | +1.539 | **+0.944** |
| mistralai/mistral-large-2512 | 3.571 | 3.600 | 3.943 | +0.835 | +0.944 | +1.110 | **+0.963** |
| mistralai/pixtral-large-2411 | 3.729 | 3.600 | 3.857 | +0.992 | +0.944 | +1.025 | **+0.987** |
| nvidia/nemotron-nano-12b-v2-vl | 3.400 | 3.486 | 4.186 | +0.664 | +0.829 | +1.353 | **+0.949** |
| qwen/qwen2.5-vl-72b-instruct | 3.786 | 3.857 | 3.843 | +1.050 | +1.201 | +1.010 | **+1.087** |
| baidu/ernie-4.5-vl-424b-a47b | 3.357 | 3.286 | 4.571 | +0.621 | +0.629 | +1.739 | **+0.996** |
| qwen/qwen3-vl-235b-a22b-instruct | 3.971 | 4.071 | 4.100 | +1.235 | +1.415 | +1.267 | **+1.306** |
| qwen/qwen2.5-vl-32b-instruct | 4.014 | 4.057 | 3.986 | +1.278 | +1.401 | +1.153 | **+1.278** |
| qwen/qwen3-vl-8b-instruct | 4.143 | 4.000 | 4.286 | +1.407 | +1.344 | +1.453 | **+1.401** |
| qwen/qwen-vl-max | 4.343 | 4.414 | 4.386 | +1.607 | +1.758 | +1.553 | **+1.639** |
| qwen/qwen-2.5-vl-7b-instruct | 4.371 | 4.371 | 4.586 | +1.635 | +1.715 | +1.753 | **+1.701** |

## Key Findings

### 1. Ranking vs Calibration Are Different Skills

The top-ranked model by wSRCC (gpt-4.1 at 0.880) has MAE_O=0.750 — it **ranks** images
correctly but overrates them. In contrast, gemini-2.5-pro ranks #4 (wSRCC=0.829) but has the
**best calibration** (MAE_O=0.331, bias_O=-0.122). This distinction matters for pseudo-labeling:

- **For ranking-based training** (pairwise loss): wSRCC is the key metric
- **For absolute score training** (SoftKL loss): MAE and bias matter more

### 2. Best Overall Performers (wSRCC >= 0.80)

| Model | wSRCC | MAE_O | Bias_O | Cost Tier |
|-------|-------|-------|--------|-----------|
| openai/gpt-4.1 | 0.880 | 0.750 | +0.750 | Strong |
| google/gemini-3-flash-preview | 0.857 | 0.935 | +0.792 | Value |
| openai/gpt-5-mini | 0.841 | 0.822 | +0.807 | Value |
| google/gemini-2.5-pro | 0.829 | 0.331 | -0.122 | Frontier |
| anthropic/claude-sonnet-4.6 | 0.827 | 0.579 | +0.507 | Anthropic |
| openai/gpt-5.1 | 0.826 | 0.750 | +0.750 | Frontier |
| openai/gpt-5 | 0.824 | 0.578 | +0.492 | Frontier |
| anthropic/claude-haiku-4.5 | 0.813 | 0.581 | +0.421 | Anthropic |

### 3. gemini-2.5-pro Is the Best Calibrated Model

- **Lowest MAE overall** (0.331) — closest absolute scores to human MOS
- **Only model with negative overall bias** (-0.122) — slightly underrates rather than overrating
- **Strong ranking** (wSRCC=0.829)
- **Scores the bad image lowest of all models** (1.2/1.3/1.5 for test_res_00312.jpg vs MOS 1.7/1.65/1.67)
- **Downside**: Slowest model (mean 10s latency), most expensive

### 4. Anthropic Models — Good Calibration, Moderate Ranking

- **Haiku 4.5**: Best calibration among Anthropic (MAE_O=0.581, bias=+0.421), fastest (3.2s mean)
- **Sonnet 4.6**: Best ranking among Anthropic (wSRCC=0.827), slightly higher bias (+0.507)
- **Opus 4.6**: Lowest bias among Anthropic (+0.364), but weakest ranking (wSRCC=0.742)
- All three compress the range — they underrate good images and overrate bad ones

### 5. Qwen VL Models Severely Overrate Everything

All Qwen VL-specific models have avg bias > +1.0 point:
- qwen-2.5-vl-7b: bias +1.70 (scores 4.5/4.8/4.7 on worst image!)
- qwen-vl-max: bias +1.64
- qwen3-vl-8b: bias +1.40
- qwen2.5-vl-32b: bias +1.28

These models are **not usable as pseudo-label generators** without significant recalibration.
The only Qwen model with reasonable performance is qwen3-vl-32b-instruct (wSRCC=0.707, bias=+0.94).

### 6. Color Fidelity Is Universally Overrated

Every model except gemini-2.5-pro and gpt-5 overrates color_fidelity by 0.5-1.7 points.
The worst offenders:
- baidu/ernie: +1.739 color bias (gives 5.0 color to bad images)
- openai/gpt-4.1-mini: +1.539
- google/gemini-3.1-pro-preview: +1.353
- qwen/qwen-2.5-vl-7b: +1.753

### 7. Low-Quality Image Discrimination Fails for Most Models

On test_res_00312.jpg (MOS 1.7 — worst quality):
- **5 models scored it 4.0+** (qwen3-vl-235b, qwen3-vl-8b, qwen-2.5-vl-7b, pixtral-large, qwen2.5-vl-32b)
- Only **gemini-2.5-pro** scored it below the MOS (1.2/1.3/1.5)
- Most models rated it 2.5-3.5 — a full 1-2 points above human judgment

On test_res_00052.jpg (MOS 2.3 — poor quality):
- **Every model overrated it**, most by 1.0-2.0 points
- Only gemini-2.5-pro came close (2.2/2.6/2.5)

### 8. Model Tiers Don't Predict IQA Performance

- **Value tier gpt-5-mini** (wSRCC=0.841) outperforms all Frontier models except gemini-2.5-pro
- **Frontier gpt-5.2** (wSRCC=0.615) is barely better than Maverick
- **Strong tier gpt-4.1** (wSRCC=0.880) beats every Frontier model
- Model size/cost is **not** a reliable predictor of IQA alignment

## Per-Image Score Distributions

| Image (MOS O) | Pred Mean O | Pred Std O | Min O | Max O | Range |
|----------------|-------------|------------|-------|-------|-------|
| test_res_00354.jpg (4.05) | 4.25 | 0.61 | 2.1 | 5.0 | 2.9 |
| test_res_00001.jpg (3.76) | 4.06 | 0.49 | 3.2 | 4.8 | 1.6 |
| test_res_00008.jpg (2.81) | 3.62 | 0.59 | 2.0 | 4.6 | 2.6 |
| test_res_00756.jpg (2.70) | 3.34 | 0.88 | 2.0 | 4.8 | 2.8 |
| test_res_00052.jpg (2.29) | 3.81 | 0.51 | 2.2 | 4.5 | 2.3 |
| test_res_00316.jpg (1.84) | 2.60 | 0.83 | 1.7 | 4.5 | 2.8 |
| test_res_00312.jpg (1.70) | 2.97 | 0.90 | 1.2 | 4.5 | 3.3 |

Key observations:
- **Higher consensus on good images** (std 0.49-0.61) vs **high disagreement on bad images** (std 0.83-0.90)
- **Overrating worsens as quality decreases** — the gap between predicted mean and MOS widens from +0.20 for excellent to +1.27 for bad
- **Range 2.3-3.5 on every image** — some models are consistently 2+ points away from ground truth

## Latency

| Model | Mean (ms) | Median (ms) | Notes |
|-------|-----------|-------------|-------|
| qwen/qwen-2.5-vl-7b-instruct | 1,523 | 1,473 | Fastest overall |
| google/gemini-2.5-flash | 1,582 | 1,585 | Fastest good model |
| meta-llama/llama-4-maverick | 2,049 | 1,965 | |
| qwen/qwen3-vl-8b-instruct | 2,317 | 2,308 | |
| qwen/qwen3-vl-32b-instruct | 2,416 | 2,404 | |
| anthropic/claude-haiku-4.5 | 3,156 | 3,093 | Best Anthropic |
| anthropic/claude-sonnet-4.6 | 3,988 | 4,367 | |
| anthropic/claude-opus-4.6 | 5,012 | 5,208 | |
| openai/gpt-5-mini | 6,792 | 7,379 | |
| openai/gpt-5.1 | 6,008 | 5,729 | |
| google/gemini-2.5-pro | 10,062 | 9,907 | Best calibrated but slowest |
| openai/gpt-4.1 | 16,190 | 4,681 | Spiky (p95=61.8s!) |

## Recommendations for Pseudo-Label Pipeline

### Tier 1 — Primary Teacher Candidates

For the DeQA pseudo-labeling pipeline, we need models with both good ranking AND reasonable calibration:

1. **google/gemini-2.5-pro** — Best calibration (MAE=0.331), strong ranking (wSRCC=0.829).
   Use as primary teacher when accuracy matters most. Slow but accurate.

2. **anthropic/claude-sonnet-4.6** — Good balance (wSRCC=0.827, MAE=0.579).
   Moderate bias (+0.507) can be corrected with a simple offset.

3. **openai/gpt-5** — Strong ranking (wSRCC=0.824), lowest MAE among OpenAI (0.578).
   Good color correlation (SRCC_C=0.857).

### Tier 2 — Value Teachers (Cost-Effective at Scale)

4. **openai/gpt-5-mini** — wSRCC=0.841, much cheaper than gpt-5. Good for bulk labeling.

5. **google/gemini-2.5-flash** — wSRCC=0.778, fastest good model (1.6s). Best for high-throughput.

6. **anthropic/claude-haiku-4.5** — wSRCC=0.813, fast (3.2s), good calibration (MAE=0.581).

### Consensus Strategy

Given that all models overrate (except gemini-2.5-pro), a consensus ensemble should:
1. Use top-K models (K=3-5) and take the **median** (not mean) to reduce overrating bias
2. Weight gemini-2.5-pro higher in the ensemble due to superior calibration
3. Apply a learned bias correction per model based on this 7-image calibration data
4. Focus consensus on low-quality images where model disagreement is highest

### Not Recommended

- **All Qwen VL models** (except qwen3-vl-32b): Severe overrating, poor discrimination
- **x-ai/grok-4-fast**: Near-zero correlation, erratic scoring
- **qwen2.5-vl-32b-instruct**: Negative correlation — actively anticorrelated with humans
- **mistralai models**: Mediocre ranking AND high bias
- **openai/gpt-5.2**: Good overall SRCC but near-zero color correlation (0.072)

## Caveats

1. **N=7 images** — correlations computed on 7 data points have high variance. These rankings
   should be validated on a larger sample (50-100 images) before finalizing the teacher ensemble.

2. **OpenRouter routing** — latency and potentially response quality may differ from direct API access.
   The Anthropic models were accessed via OpenRouter, not the Anthropic API directly.

3. **Temperature=0.0** — all models run deterministic. Stochastic sampling might improve
   distribution estimation but would require multiple runs per image.

4. **Document images only** — DIQA-5000 contains scanned/photographed documents. Performance on
   natural images (KONIQ, LIVE) may differ significantly.
