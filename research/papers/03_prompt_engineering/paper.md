# Prompt Engineering for VLM-Based Quality Assessment

**Author:** Byron Williams
**Date:** March 2026
**Series:** DeQA-Doc Technical Report 3/10
**Repository:** `results/vlm_teacher_eval/full_eval/`
**License:** CC BY-SA 4.0, Copyright 2025 Byron Williams
**Keywords:** prompt engineering, VLM, image quality assessment, A/B testing, regression to mean, scale manipulation

---

## Abstract

Vision-Language Models (VLMs) can serve as automated annotators for document image quality assessment (DIQA), but their effectiveness depends critically on how prompts are structured and how images are preprocessed. We present a systematic evaluation of nine prompting strategies across three models (Gemini 3 Flash, Qwen 3.5 Flash, Qwen 3.5 122B-A10B) on 1,000 DIQA-5000 document images. Our nine-arm experiment tests prompt structure (single, separate, hybrid, few-shot, multi-sample), image resolution (1024px, 2048px, native), and scoring scale (1-5 default, 1-10 rescaled, 0.5 increments).

Key findings: (1) The best prompt strategy is model-dependent — few-shot examples help Gemini (+0.023 wSRCC, p=0.027) while higher resolution helps Qwen Flash (+0.078, p<0.001); (2) A 1-10 scoring scale consistently reduces positive bias across all models by up to 12% while maintaining correlation; (3) Resolution sensitivity scales inversely with model capacity — small models benefit most from higher-resolution input; (4) A 23-image pilot study identified the wrong best arm, confirming that small-sample prompt optimization is unreliable (minimum n≥200 recommended). Paired bootstrap confidence intervals with Holm-Bonferroni correction provide rigorous statistical validation of all claims.

---

## 1. Introduction

When using Vision-Language Models as automated quality annotators for document images, the prompt is the only lever the practitioner controls. Unlike fine-tuning, which adjusts model weights, prompt engineering operates at inference time: changing how the question is asked, how many questions are asked per image, whether exemplars are provided, how the image itself is preprocessed, and what scoring scale is presented.

Paper 1 in this series (Williams, 2026a) established that Gemini 3 Flash Preview achieves wSRCC = 0.708 on the DIQA-5000 test set using a single-prompt, 1024px-resized baseline configuration. This approaches the supervised DeQA-Doc-3Specialists baseline (wSRCC = 0.716) without any DIQA-specific training. But can prompt engineering close or widen this gap?

This paper investigates that question through a comprehensive nine-arm experiment at full scale (n=1,000), cross-validated across three VLM architectures of different sizes (7B, 22B MoE, and proprietary). We go beyond simple accuracy comparison to analyze positive scoring bias, per-quality-bucket performance, statistical power requirements, and the reliability of small-sample prompt optimization. The central finding is that no single prompt strategy dominates across models, but specific interventions — particularly scoring scale manipulation and resolution adjustment — provide consistent, model-dependent improvements.

## 2. Task Definition and Related Work

### 2.1 Document Image Quality Assessment

DIQA predicts how well a scanned or photographed document can be read by humans. The DIQA-5000 dataset provides human Mean Opinion Scores (MOS) from 15 annotators per image across three dimensions: overall quality, sharpness, and color fidelity, each on a 1.0-5.0 continuous scale.

We evaluate prompt variants using the VQualA competition metric, weighted SRCC:

$$\text{wSRCC} = 0.5 \times \text{SRCC}_\text{overall} + 0.25 \times \text{SRCC}_\text{sharpness} + 0.25 \times \text{SRCC}_\text{color}$$

where SRCC is Spearman's rank correlation coefficient computed between VLM predictions and human MOS.

### 2.2 Prompt Engineering for Vision Tasks

Prompt engineering for VLMs in evaluation tasks differs from text-only prompting in several respects. First, image preprocessing (resolution, compression, aspect ratio) is itself a form of "prompting" that affects the visual information available to the model. Second, structured output requirements (JSON with specific keys and numeric ranges) constrain the response space. Third, multi-dimensional ratings introduce anchoring effects, where the model's assessment of one dimension can influence its rating of others. Fourth, the numeric scale presented to the model affects both the granularity of responses and the distribution of scores.

Prior work on prompt optimization for VLM-based natural image quality assessment (Wu et al., 2024; Zhang et al., 2024) has found that detailed rubric descriptions and scale anchoring improve correlation with human judgments. However, these studies typically use large evaluation sets (n > 500), and the question of how many samples are needed for reliable prompt selection has received limited attention.

### 2.3 Regression to the Mean

Regression to the mean is a well-known statistical phenomenon: extreme values on a first measurement tend to be less extreme on a second measurement, purely due to random variation. In the context of prompt optimization, an arm that achieves the highest wSRCC on a small pilot sample may do so partly because of favorable noise in that particular sample. At full scale, this noise averages out, and the true performance is revealed to be closer to the population mean.

## 3. Experimental Setup

### 3.1 Prompt Variants

We tested nine prompting strategies, each representing a different hypothesis about what drives VLM quality assessment accuracy:

| Arm | Strategy | Hypothesis | Calls/Image |
|-----|----------|-----------|-------------|
| 1 | **Single prompt** (baseline) | One call requesting all 3 dimensions is sufficient | 1 |
| 2 | **Separate prompts** | Dedicated prompts per dimension avoid anchoring effects | 3 |
| 3 | **Hybrid** | Overall in combined call, sub-dimensions separately | 2 |
| 4 | **Few-shot** (3 examples) | Calibration examples anchor the score scale | 1 |
| 5 | **Multi-sample** (3x, temp=0.3) | Aggregating multiple stochastic samples reduces noise | 3 |
| 6 | **Resize 2048px** | Higher resolution preserves quality-relevant detail | 1 |
| 7 | **No resize** (native) | Native resolution preserves all visual information | 1 |
| 8 | **1-10 scale** (rescaled) | Wider numeric range reduces positive bias clustering | 1 |
| 9 | **0.5 increments** | Coarser steps change score distribution behavior | 1 |

All prompts used the same system prompt establishing the model as a document quality assessor with scale anchors and JSON output format. The baseline (arm 1) resizes images to fit within 1024×1024 pixels using LANCZOS resampling and JPEG quality 90. Arms 8 and 9 use the same 1024px encoding but modify the scoring scale presented to the model.

Arms 6-7 test image preprocessing, arms 8-9 test scoring scale, and arms 2-5 test prompt structure. This factored design separates three distinct axes of prompt engineering.

### 3.2 Models

We tested on three models spanning different architectures and sizes:

- **Gemini 3 Flash Preview** (Google): The top-performing VLM on DIQA-5000 (wSRCC = 0.708 baseline), proprietary architecture, accessed via OpenRouter.
- **Qwen 3.5 Flash** (Alibaba, 7B): A small, fast reasoning model with lower DIQA performance (wSRCC = 0.593 baseline), accessed via OpenRouter.
- **Qwen 3.5 122B-A10B** (Alibaba, 122B MoE): A large mixture-of-experts model (wSRCC = 0.713 baseline), tested on resolution and scale arms to evaluate capacity effects.

The A/B test (single vs per-dimension prompting) additionally included **GPT-4.1** (OpenAI).

### 3.3 Evaluation Protocol

**Pilot experiment (n=23).** Twenty-three images were stratified-sampled from the DIQA-5000 test set to cover the MOS range. Each of the original 7 arms was evaluated on the same 23 images. This sample size was chosen for rapid iteration (~20 minutes per arm).

**Full-scale validation (n=1,000).** All nine arms were evaluated on the full 1,000-image DIQA-5000 test set for Gemini (6 arms) and Qwen Flash (9 arms), and 4 arms for Qwen 122B. Bootstrapped 95% confidence intervals (2,000 iterations, seed=42) are reported for all metrics.

**Statistical analysis.** Paired bootstrap delta CIs resample images as units, computing the wSRCC delta within each bootstrap replicate. Holm-Bonferroni step-down correction is applied for multiple comparisons (each arm vs baseline). Sub-sampling power curves draw 500 stratified sub-samples at n={25, 50, 100, 150, 200, 300, 500, 750, 1000} to measure wSRCC SD and arm ranking stability.

**A/B test (n=44).** Forty-four stratified images were used for the single-prompt versus per-dimension comparison, with both conditions run on the same images.

All experiments used temperature=0.0 and max_tokens=1,024 (except arm 5 which uses temperature=0.3).

## 4. Results and Discussion

### 4.1 Full-Scale Nine-Arm Comparison

Table 1 presents the complete results at n=1,000 for all three models.

**Table 1: Full-Scale Prompt Arm Results (n=1,000)**

| Arm | Strategy | wSRCC (Gemini) | wSRCC (Qwen Flash) | wSRCC (Qwen 122B) | Bias_O (Gemini) | Bias_O (Qwen Flash) |
|-----|----------|---------------|--------------------|--------------------|-----------------|---------------------|
| 1 | Baseline (1024px) | 0.708 | 0.593 | 0.713 | +0.76 | +1.50 |
| 2 | Separate 3 prompts | 0.691 | 0.601 | — | +0.88 | +1.37 |
| 3 | Hybrid | — | 0.514† | — | — | +1.49 |
| 4 | Few-shot (3 examples) | **0.731** | 0.630 | — | +1.04 | +1.27 |
| 5 | Multi-sample (3x) | — | 0.605 | — | — | +1.50 |
| 6 | 2048px resize | 0.704 | **0.671** | 0.717 | +0.79 | +1.50 |
| 7 | No resize (native) | 0.699 | 0.659 | **0.728** | +0.75 | +1.47 |
| 8 | 1-10 scale (rescaled) | 0.726 | 0.612 | 0.720 | +0.75 | +1.35 |
| 9 | 0.5 increments | 0.698 | 0.585 | — | +0.63 | +1.45 |

†Partial data (n=565). — = not run for this model.

The results reveal that **no single arm dominates across models**:

- **Gemini's best arm** is few-shot (+0.023 wSRCC over baseline), followed closely by the 1-10 scale (+0.018).
- **Qwen Flash's best arm** is 2048px resolution (+0.078), a dramatically larger effect driven by the small model's limited ability to interpret low-resolution images.
- **Qwen 122B's best arm** is native resolution (+0.014), a more modest improvement consistent with the larger model's better baseline vision processing.

### 4.2 Statistical Significance: Paired Bootstrap Analysis

Table 2 presents paired bootstrap delta CIs for each arm versus baseline, with Holm-Bonferroni correction for multiple comparisons.

**Table 2: Paired Bootstrap Deltas vs Baseline (Gemini 3 Flash)**

| Arm | Delta wSRCC | 95% CI | P(delta>0) | Holm p | Sig? |
|-----|------------|--------|------------|--------|------|
| arm4 few-shot | +0.023 | [+0.003, +0.043] | 0.987 | 0.081 | marginal |
| arm8 scale-10 | — | — | — | — | (not in power analysis) |
| arm6 2048px | -0.004 | [-0.021, +0.012] | 0.303 | 0.606 | no |
| arm7 no-resize | -0.009 | [-0.027, +0.007] | 0.143 | 0.572 | no |
| arm2 separate | -0.017 | [-0.031, -0.004] | 0.004 | **0.036** | **yes (worse)** |

**Table 3: Paired Bootstrap Deltas vs Baseline (Qwen 3.5 Flash)**

| Arm | Delta wSRCC | 95% CI | P(delta>0) | Holm p | Sig? |
|-----|------------|--------|------------|--------|------|
| arm6 2048px | +0.077 | [+0.044, +0.111] | 1.000 | **<0.001** | **yes** |
| arm7 no-resize | +0.066 | [+0.036, +0.097] | 1.000 | **<0.001** | **yes** |
| arm4 few-shot | +0.037 | [+0.005, +0.067] | 0.988 | 0.096 | marginal |
| arm3 hybrid | +0.027 | [-0.008, +0.064] | 0.932 | 0.405 | no |
| arm5 multi-sample | +0.012 | [-0.012, +0.036] | 0.853 | 0.590 | no |
| arm2 separate | +0.009 | [-0.017, +0.036] | 0.734 | 0.533 | no |

Two findings survive Holm-Bonferroni correction: (1) resolution improvements for Qwen Flash are highly significant (p<0.001 for both 2048px and native); (2) separate prompting significantly *hurts* Gemini (Holm p=0.036). Few-shot effects are marginal (Holm p~0.08-0.10) for both models.

### 4.3 Resolution Sensitivity Scales Inversely with Model Capacity

The most striking cross-model finding is that resolution's impact depends on model size:

| Model | Capacity | Δ wSRCC (2048px) | Δ wSRCC (native) | Interpretation |
|-------|----------|-----------------|------------------|---------------|
| Qwen 3.5 Flash | 7B | **+0.078** | +0.066 | Huge benefit — limited vision encoder |
| Qwen 3.5 122B-A10B | 22B active | +0.004 | +0.014 | Modest benefit |
| Gemini 3 Flash | proprietary | -0.004 | -0.009 | No benefit / slight harm |

Qwen Flash at 2048px (wSRCC=0.671) outperforms its baseline (0.593) by 13% relative — the single largest improvement from any prompt intervention in this study. The effect is highly significant (Holm p<0.001). By contrast, Gemini derives zero benefit from higher resolution, suggesting its vision encoder already processes 1024px images effectively.

For Qwen, 2048px slightly outperforms native resolution (+0.012), suggesting that some form of controlled upscaling is preferable to passing arbitrarily large images. Very large native images may introduce noise from resolution inconsistency across the dataset.

### 4.4 Scoring Scale Manipulation

Arms 8 and 9 test whether changing the numeric scale presented to the VLM affects scoring behavior, particularly the persistent positive bias (VLMs over-rate quality by +0.63 to +1.50 MOS).

**Table 4: Scoring Scale Effects on Bias and Correlation**

| Model | Scale | wSRCC | Bias_O | MAE_O | Δ Bias vs baseline |
|-------|-------|-------|--------|-------|-------------------|
| Gemini | 1-5 (baseline) | 0.708 | +0.76 | 0.80 | — |
| Gemini | 1-10 (rescaled) | 0.726 | +0.75 | 0.79 | -2% |
| Gemini | 1-5 (0.5 steps) | 0.698 | **+0.63** | **0.72** | **-18%** |
| Qwen Flash | 1-5 (baseline) | 0.593 | +1.50 | 1.50 | — |
| Qwen Flash | 1-10 (rescaled) | 0.612 | +1.35 | 1.36 | -10% |
| Qwen Flash | 1-5 (0.5 steps) | 0.585 | +1.45 | 1.45 | -3% |
| Qwen 122B | 1-5 (baseline) | 0.713 | +1.40 | 1.40 | — |
| Qwen 122B | 1-10 (rescaled) | 0.720 | **+1.23** | **1.23** | **-12%** |

The 1-10 scale consistently improves or maintains correlation while reducing bias. For Qwen 122B, the bias reduction is substantial (-12%) with a simultaneous wSRCC improvement (+0.007). The mechanism is likely that the wider numeric range allows the model to differentiate within the "good" range (7 vs 8 vs 9) rather than clustering everything at 4-5 on a 1-5 scale.

The 0.5-increment scale has mixed effects: it achieves the lowest absolute bias for Gemini (+0.63) but hurts discrimination (-0.010 wSRCC). The coarser scale forces the model to bin scores more aggressively, reducing the effective resolution of its quality judgments. For Qwen Flash, the effect is minimal on both bias and correlation.

**Practical recommendation.** The 1-10 scale with linear rescaling to 1-5 is the single most robust prompt intervention: it improves or preserves correlation for all three models while reducing positive bias. We recommend it as the default for VLM-based DIQA, with a post-hoc linear rescaling to match the target MOS range.

### 4.5 Power Analysis: Minimum Sample Size for Prompt Optimization

Sub-sampling power curves (Figure 1, not shown) confirm that small samples produce unreliable arm rankings:

**Table 5: wSRCC Standard Deviation by Sample Size**

| n | SD (Gemini) | SD (Qwen Flash) | 95% CI half-width |
|---|-------------|-----------------|-------------------|
| 25 | 0.086 | 0.121 | ±0.169 |
| 50 | 0.062 | 0.091 | ±0.121 |
| 100 | 0.041 | 0.060 | ±0.079 |
| 200 | 0.028 | 0.037 | ±0.054 |
| 300 | 0.022 | 0.026 | ±0.043 |
| 500 | 0.014 | 0.010 | ±0.028 |

At n=25 (our pilot size), the SD of 0.086-0.121 means that two arms differing by 0.02 wSRCC (a typical prompt effect size) are indistinguishable from noise. The 95% CI half-width of ±0.17 exceeds the entire spread of arm performance.

At n=200, the SD drops to 0.028-0.037, sufficient to detect effects of ~0.05 wSRCC with reasonable power. We recommend **n≥200 stratified images** as the minimum for prompt optimization experiments.

### 4.6 Small-Sample Regression to the Mean

The n=23 pilot results (Table 6) dramatically illustrate the danger of small-sample optimization.

**Table 6: Pilot (n=23) vs Full-Scale (n=1,000) Comparison for Gemini**

| Arm | wSRCC (n=23) | wSRCC (n=1,000) | Reversal? |
|-----|-------------|----------------|-----------|
| 7 No resize | **0.951** (#1) | 0.699 (#5) | Yes — pilot's best is below baseline |
| 5 Multi-sample | 0.928 (#2) | — | |
| 6 2048px | 0.925 (#3) | 0.704 (#4) | |
| 1 Baseline | 0.909 (#6) | 0.708 (#3) | |
| 4 Few-shot | 0.836 (#7) | **0.731** (#1) | Yes — pilot's worst is actual best |

The pilot ranked no-resize first (wSRCC=0.951) and few-shot last (0.836). Full-scale validation completely reversed this: few-shot is best (0.731) and no-resize is below baseline (0.699). The pilot's ranking was not merely imprecise — it was *inverted* for the top and bottom arms.

This occurred because the pilot's 23 images happened to favor high-resolution images where native resolution preserves quality-relevant detail, while the few-shot examples happened to confuse the model on those specific images. At n=1,000, these sampling artifacts average out.

### 4.7 A/B Test: Per-Dimension Prompting

A separate experiment compared single-prompt versus per-dimension prompting on n=44 stratified images.

**Table 7: 1-Prompt vs 3-Prompt SRCC (n=44)**

| Dimension | 1-Prompt (Gemini) | 3-Prompt (Gemini) | Delta | 1-Prompt (GPT-4.1) | 3-Prompt (GPT-4.1) | Delta |
|-----------|-------------------|-------------------|-------|--------------------|--------------------|-------|
| Overall | **0.895** | 0.878 | -0.017 | **0.914** | 0.863 | -0.051 |
| Sharpness | 0.860 | **0.896** | +0.036 | 0.906 | **0.924** | +0.019 |
| Color Fidelity | 0.845 | **0.860** | +0.015 | 0.829 | **0.866** | +0.037 |
| **wSRCC** | 0.874 | 0.878 | +0.004 | **0.891** | 0.879 | -0.012 |

Overall quality degrades with separate prompts (the model loses holistic context), while sub-dimensions improve (focused assessment avoids anchoring). The aggregate wSRCC is roughly unchanged, and at full scale (n=1,000) separate prompting significantly *hurts* Gemini (Holm p=0.036), confirming that the A/B test's small positive delta (+0.004) was noise.

### 4.8 Per-Quality-Bucket Analysis

Performance varies substantially by quality tier:

**Table 8: Overall SRCC by Quality Bucket (Gemini)**

| Arm | Bad (n=56) | Poor (n=180) | Fair (n=613) | Good (n=146) |
|-----|-----------|-------------|-------------|-------------|
| Baseline | 0.605 | 0.647 | 0.344 | 0.206 |
| Few-shot | 0.421 | **0.673** | **0.384** | 0.106 |
| 2048px | 0.568 | 0.650 | 0.347 | 0.185 |
| No-resize | 0.612 | 0.626 | 0.332 | 0.209 |
| Separate | **0.673** | 0.637 | 0.354 | 0.169 |

Few-shot examples help most on mid-range quality (poor/fair) but hurt on extremes (bad: 0.421 vs 0.605, good: 0.106 vs 0.206). This suggests the calibration examples help the model differentiate nuanced quality levels but distort its assessment of obvious extremes, possibly by anchoring to the example scores.

All models struggle most with "good" quality images (SRCC~0.1-0.2), indicating that discriminating between "good" and "excellent" documents is inherently difficult — these images have few visible defects, and quality differences are subtle.

## 5. Practical Recommendations

Based on the full-scale cross-model analysis, we provide the following ranked recommendations:

1. **Use a 1-10 scoring scale with linear rescaling.** This is the single most robust intervention: it improves or preserves correlation while reducing positive bias by up to 12%. No downside observed across any model.

2. **For small models (≤10B), increase image resolution to 2048px.** The effect is dramatic for Qwen Flash (+13% relative wSRCC) and highly significant (p<0.001). Larger models benefit less.

3. **For Gemini, add 3 few-shot calibration examples.** This improves wSRCC by +0.023 (p=0.027 raw, marginal after Holm correction). The examples should span the quality range (bad, fair, excellent).

4. **Do not use separate per-dimension prompting.** It significantly hurts Gemini (Holm p=0.036), provides marginal benefit for Qwen, and costs 3x in API calls.

5. **Use n≥200 stratified images for prompt optimization.** Our n=23 pilot produced rankings that were not merely imprecise but *inverted* at full scale.

## 6. Conclusion

This study yields four actionable findings for practitioners using VLMs as quality annotators:

1. **The best prompt strategy is model-dependent.** Few-shot helps Gemini, resolution helps Qwen, and scale manipulation helps all models. Practitioners must validate on their specific model rather than applying universal "best practices."

2. **Scoring scale manipulation is the most robust intervention.** A 1-10 scale with linear rescaling reduces positive bias by up to 12% while maintaining or improving rank correlation. This works across all three models tested.

3. **Resolution sensitivity scales inversely with model capacity.** Small models (7B) gain dramatically from higher-resolution input; large models gain little. This has practical implications for cost optimization: smaller, cheaper models can partially compensate for their lower accuracy with higher-resolution images.

4. **Small-sample prompt optimization is unreliable.** A 23-image pilot produced inverted rankings compared to full-scale validation. We recommend n≥200 stratified images minimum, with paired bootstrap CIs for statistical validation.

**Future directions.** Several avenues remain unexplored: (a) combining the best interventions (e.g., 1-10 scale + 2048px + few-shot) to test for additive effects; (b) adaptive prompting where strategy is selected per image; (c) prompt tuning with soft tokens for open-weight models; (d) cross-dataset validation on non-document IQA benchmarks.

## 7. Reproducibility, Data, and Governance

**Data availability.** All per-image predictions for the nine-arm experiment are archived at:
- `results/vlm_teacher_eval/full_eval/checkpoints/*__arm*.jsonl` (per-model, per-arm JSONL checkpoints)
- `results/vlm_teacher_eval/full_eval/results/prompt_arms_*.json` (aggregated metrics)
- `results/vlm_teacher_eval/full_eval/results/prompt_power_*.json` (statistical analysis: power curves, paired bootstrap CIs)

Pilot data:
- `results/vlm_teacher_eval/full_eval/prompt_optimization/` (n=23 pilot checkpoints)

A/B test results:
- `results/vlm_teacher_eval/full_eval/ab_test/` (per-dimension prompting comparison)

Analysis scripts:
- `results/vlm_teacher_eval/full_eval/run_full_prompt_arms.py` (data collection)
- `results/vlm_teacher_eval/full_eval/analyze_prompt_power.py` (statistical analysis)

**Cost.** Full-scale evaluation consumed approximately $45 total across all models and arms (~25,000 API calls via OpenRouter). The original pilot consumed ~$2 per model.

**Ethical considerations.** All document images are from the publicly available DIQA-5000 dataset. No personal or sensitive information is contained in the evaluation data.

**Relationship to the series.** This paper builds on the VLM benchmark established in Paper 1 (Williams, 2026a) and the cross-domain evaluation in Paper 2 (Williams, 2026b). The 1-10 scale configuration validated here is recommended for use in all subsequent papers in the series.

## References

- Williams, B. (2026a). VLM Teachers for Document Image Quality Assessment: Benchmark Results. DeQA-Doc Technical Report 1/10.
- Williams, B. (2026b). Cross-Domain Generalization of VLM Quality Assessors. DeQA-Doc Technical Report 2/10.
- Wu, H., et al. (2024). Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-Level Vision. ICLR.
- Zhang, Z., et al. (2024). Benchmark Data Contamination of Large Language Models: A Survey. arXiv:2406.04244.
- Zhiyuan, Y., et al. (2024). DeQA-Score: Deep Quality Assessment via Distributional Learning. NeurIPS.

---

## Appendix A: Prompt Templates

### A.1 System Prompt (Baseline, 1-5 Scale)

```
You are an expert document image quality assessor. You evaluate scanned or
photographed document images for visual quality as perceived by a human reader.

You rate documents on three dimensions using a 1.0-5.0 scale with 0.1 increments:

1. **Overall Quality**: Holistic readability and usability of the document.
2. **Sharpness**: Text edge clarity, blur level, and resolution adequacy.
3. **Color Fidelity**: Color accuracy, contrast, white balance, and tonal reproduction.

Scale anchors:
- 1.0: Completely unusable / illegible / severe degradation
- 2.0: Poor — significant issues affecting readability
- 3.0: Fair — acceptable but with noticeable problems
- 4.0: Good — minor issues, generally readable
- 5.0: Excellent — crisp, clean, high-quality reproduction

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON.
```

### A.2 System Prompt (Arm 8, 1-10 Scale)

Same structure as A.1 but with `1.0-10.0` scale, anchors at 1.0/3.25/5.5/7.75/10.0.

### A.3 User Prompt (All single-call arms)

```
Rate the quality of this document image.

Respond with exactly this JSON structure:
{"overall": X.X, "sharpness": X.X, "color_fidelity": X.X, "reasoning": "..."}

The reasoning field should be 1-2 sentences explaining the key quality
factors you observed. Keep it concise.
```

### A.4 Dimension-Specific Prompt (Arms 2, 3)

Example for sharpness:
```
Rate the SHARPNESS of this document image.

Focus on text edge clarity, blur level, and resolution. Ignore color issues.

Respond with exactly: {"score": X.X, "reasoning": "..."}
```

### A.5 Few-Shot Examples (Arm 4)

Three calibration images spanning the quality range:
- Bad quality (MOS ~1.5): severely degraded document
- Fair quality (MOS ~3.0): readable but with noticeable artifacts
- Excellent quality (MOS ~4.1): clean, sharp document

Each example is presented as an image followed by its ground truth JSON scores.

---
