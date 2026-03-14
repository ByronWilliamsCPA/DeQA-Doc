# Multi-Model IQA Smoke Test — 2026-03-07

## Configuration

- **Image**: `DeQA-Score/fig/singapore_flyer.jpg` (nighttime motion-blurred photograph)
- **Provider**: OpenRouter (all models accessed via `openai` SDK)
- **Temperature**: 0.0 (omitted for reasoning models)
- **Scale**: 1.0–5.0
- **Prompt**: Standard IQA prompt from `results/vlm_teacher_eval/prompts.py`
- **Max tokens**: 1024

## Anthropic Baseline (via direct API and Claude Code extension)

From [2026-03-07_baseline.md](2026-03-07_baseline.md) — tested separately via Anthropic API, OpenRouter, and Claude Code extension.

| Model | Overall | Sharp | Color | Latency | Access Method |
|-------|---------|-------|-------|---------|---------------|
| claude-sonnet-4.6 | 1.5 | 1.0 | 2.5 | 3547ms | Anthropic API |
| claude-sonnet-4.6 | 1.5 | 1.0 | 2.5 | 4145ms | OpenRouter API |
| claude-sonnet-4.6 | 1.5 | 1.0 | 2.5 | n/a | Claude Code (standard) |
| claude-sonnet-4.6 | 1.5 | 1.0 | 2.5 | n/a | Claude Code (1M ctx) |
| claude-opus-4.6 | 1.5 | 1.0 | 2.5 | n/a | Claude Code |

All Anthropic models produce identical scores (1.5/1.0/2.5) regardless of access method, context window, or model tier. Opus produces richer reasoning text but the same numeric assessment.

## OpenRouter Multi-Model Results (23 models)

Sorted by overall score (ascending), then sharpness, then color fidelity.

| Model | Overall | Sharp | Color | Latency | Tier |
|-------|---------|-------|-------|---------|------|
| google/gemini-3.1-pro-preview | 1.0 | 1.0 | 1.0 | 9621ms | Frontier |
| qwen/qwen3-vl-235b-a22b-instruct | 1.0 | 1.0 | 1.0 | 2247ms | Strong |
| qwen/qwen3-vl-8b-instruct | 1.0 | 1.0 | 1.0 | 1561ms | VL |
| qwen/qwen-2.5-vl-7b-instruct | 1.0 | 1.0 | 1.0 | 2595ms | VL |
| mistralai/mistral-large-2512 | 1.0 | 1.0 | 1.5 | 6230ms | Value |
| google/gemini-2.5-pro | 1.0 | 1.0 | 1.5 | 11119ms | Frontier |
| openai/gpt-5.2 | 1.0 | 1.0 | 1.6 | 6069ms | Frontier |
| openai/gpt-5 | 1.0 | 1.0 | 2.0 | 9387ms | Frontier |
| openai/gpt-4.1 | 1.0 | 1.0 | 2.0 | 6673ms | Strong |
| google/gemini-2.5-flash | 1.0 | 1.0 | 2.0 | 1466ms | Value |
| qwen/qwen3-vl-32b-instruct | 1.0 | 1.0 | 2.0 | 3102ms | Value |
| openai/gpt-5-mini | 1.0 | 1.0 | 2.0 | 10281ms | Value |
| meta-llama/llama-4-maverick | 1.0 | 1.0 | 2.0 | 2120ms | Value |
| openai/gpt-5.1 | 1.0 | 1.0 | 2.5 | 9162ms | Frontier |
| google/gemini-3-flash-preview | 1.0 | 1.0 | 2.5 | 1883ms | Value |
| openai/gpt-4.1-mini | 1.0 | 1.0 | 2.5 | 8874ms | Value |
| x-ai/grok-4-fast | 1.0 | 1.2 | 2.0 | 6128ms | Value |
| mistralai/pixtral-large-2411 | 1.2 | 1.0 | 1.5 | 2627ms | Strong |
| qwen/qwen-vl-max | 1.5 | 1.0 | 2.0 | 2551ms | VL |
| nvidia/nemotron-nano-12b-v2-vl | 1.5 | 1.0 | 2.0 | 3774ms | VL |
| qwen/qwen2.5-vl-72b-instruct | 1.5 | 1.0 | 2.5 | 2618ms | Strong |
| qwen/qwen2.5-vl-32b-instruct | 1.5 | 1.0 | 2.5 | 1586ms | VL |
| baidu/ernie-4.5-vl-424b-a47b | 1.5 | 1.0 | 2.5 | 5027ms | VL |

### Excluded Models (8)

| Model | Reason |
|-------|--------|
| openai/gpt-5.4 | OpenRouter 500 on image input (provider routing issue) |
| openai/o3 | OpenRouter 500 on image input (provider routing issue) |
| openai/o4-mini | OpenRouter 500 on image input (provider routing issue) |
| baidu/ernie-4.5-vl-28b-a3b | Returns `<unk>` tokens — cannot process image |
| moonshotai/kimi-k2.5 | Empty response after 25s timeout |
| qwen/qwen3-vl-30b-a3b-instruct | Poor calibration (2.0/1.5/3.5 on completely blurred image) |
| allenai/molmo-2-8b | Poor calibration (2.5/2.0/2.5 — significant score inflation) |
| meta-llama/llama-3.2-11b-vision-instruct | Poor calibration (3.0/2.0/3.0 — rates blurred image "fair") |

## Score Distribution Analysis

### Overall Quality

All retained models correctly identify this as a severely degraded, non-document image:

- **1.0** (17 models, 74%): The strictest interpretation — completely unusable
- **1.2** (1 model): pixtral-large, slight uplift
- **1.5** (5 models, 22%): Slightly more generous — acknowledging the image exists but is unusable

**Consensus**: The ground-truth score for this image should be ~1.0-1.5 (overall), ~1.0 (sharpness), ~2.0-2.5 (color). Claude Sonnet 4.6 gave 1.5/1.0/2.5 in the baseline test.

### Sharpness

Near-universal agreement across the retained pool:

- **1.0** (22 models, 96%): Correct — zero sharpness in a motion-blurred image
- **1.2** (1 model): grok-4-fast — minor uplift

### Color Fidelity

Most variation here — reasonable, since colors ARE partially identifiable despite blur:

- **1.0** (4 models): gemini-3.1-pro, qwen3-vl-235b, qwen3-vl-8b, qwen-2.5-vl-7b — strictest
- **1.5** (3 models): mistral-large, gemini-2.5-pro, pixtral-large
- **1.6** (1 model): gpt-5.2
- **2.0** (7 models): Most common — reasonable for partially visible colors
- **2.5** (5 models): Slightly generous but defensible

## Reasoning Responses

### Frontier Tier

**google/gemini-3.1-pro-preview** (O=1.0, S=1.0, C=1.0)
> The image is a heavily blurred night photograph, not a document. It contains no readable text or document structure, making it completely unusable for any document-related purpose.

**openai/gpt-5.2** (O=1.0, S=1.0, C=1.6)
> The image is heavily motion-blurred with no readable text or document content, making it unusable as a document scan/photo. Colors are dominated by low-light noise and light streaks with poor tonal consistency.

**openai/gpt-5.1** (O=1.0, S=1.0, C=2.5)
> The image is heavily motion-blurred, making any potential text or document content completely illegible. While colors and contrast are somewhat preserved, the lack of sharpness renders it unusable as a document image.

**google/gemini-2.5-pro** (O=1.0, S=1.0, C=1.5)
> The image is not a document and is completely unusable due to extreme motion blur, which makes any potential content entirely illegible and unsharp.

**openai/gpt-5** (O=1.0, S=1.0, C=2.0)
> Severe motion blur makes any text or details illegible, so the image is unusable as a document. Colors show low contrast and a strong cast due to low-light capture, further reducing fidelity.

### Strong Tier

**openai/gpt-4.1** (O=1.0, S=1.0, C=2.0)
> The image is extremely blurry with no discernible text or details, making it completely unreadable. Color fidelity is slightly better, but overall quality and sharpness are severely compromised.

**mistralai/pixtral-large-2411** (O=1.2, S=1.0, C=1.5)
> The image is extremely blurred and lacks any readable content, making it unusable. Color fidelity is poor due to severe motion blur and distorted lighting.

**qwen/qwen3-vl-235b-a22b-instruct** (O=1.0, S=1.0, C=1.0)
> This is not a document image; it is an abstract long-exposure photograph of light trails at night, making it entirely unsuitable for document evaluation.

**qwen/qwen2.5-vl-72b-instruct** (O=1.5, S=1.0, C=2.5)
> The image is extremely blurry with no discernible text, making it unusable. Colors are present but lack definition due to motion blur.

### Value Tier

**google/gemini-2.5-flash** (O=1.0, S=1.0, C=2.0)
> The image is extremely blurry due to motion, making it impossible to discern any clear details or text. While colors are present, their fidelity is severely compromised by the blur.

**google/gemini-3-flash-preview** (O=1.0, S=1.0, C=2.5)
> This is not a document image, but a photograph of a night skyline with severe motion blur. As a document, it is completely unusable and illegible due to the lack of any text or structured information.

**mistralai/mistral-large-2512** (O=1.0, S=1.0, C=1.5)
> The image is severely blurred due to motion, rendering text or document details completely illegible. Color fidelity is poor, with unnatural hues and low contrast.

**openai/gpt-4.1-mini** (O=1.0, S=1.0, C=2.5)
> The image is extremely blurry with no readable text or clear details, resulting in very poor sharpness and overall usability. Color fidelity is moderate as the colors are distinguishable but the image is dark and lacks clarity.

**qwen/qwen3-vl-32b-instruct** (O=1.0, S=1.0, C=2.0)
> The image exhibits extreme motion blur with no discernible text or structured content, making it completely illegible. Colors are distorted with streaks and low contrast, indicating poor capture quality.

**x-ai/grok-4-fast** (O=1.0, S=1.2, C=2.0)
> This image shows a blurred nighttime cityscape with no text or document content, making it completely unusable for reading. Sharpness is severely compromised by motion blur, and while colors are vibrant from lights, they lack the accuracy and clean background needed for document fidelity.

**openai/gpt-5-mini** (O=1.0, S=1.0, C=2.0)
> Image has extreme motion blur with no readable text or clear details, making it unusable as a document. Colors are visible but smeared and uneven with poor contrast and white balance.

**meta-llama/llama-4-maverick** (O=1.0, S=1.0, C=2.0)
> The image is not a document but an abstract cityscape at night, making it completely unusable as a document. The blur and motion effects further degrade its quality.

### VL-Specific Models

**qwen/qwen3-vl-8b-instruct** (O=1.0, S=1.0, C=1.0)
> The image is heavily blurred with motion trails, making it completely illegible and unusable as a document image.

**qwen/qwen-2.5-vl-7b-instruct** (O=1.0, S=1.0, C=1.0)
> The image is completely unusable due to severe motion blur, making it impossible to discern any details or read any text. The sharpness and color fidelity are also compromised by the blurriness.

**qwen/qwen2.5-vl-32b-instruct** (O=1.5, S=1.0, C=2.5)
> The image is heavily blurred, making it impossible to discern any text or details, which significantly impacts overall readability. While some colors are visible, they are streaked and lack definition, indicating poor sharpness and moderate color fidelity issues.

**qwen/qwen-vl-max** (O=1.5, S=1.0, C=2.0)
> The image exhibits significant motion blur and light trails, indicating poor sharpness and focus. Colors are present but distorted due to the blurring effect, making it difficult to discern details or content.

**nvidia/nemotron-nano-12b-v2-vl** (O=1.5, S=1.0, C=2.0)
> The image is severely blurred with motion streaks, making details indiscernible. Colors are present but distorted by the blur, and sharpness is entirely lacking due to camera movement.

**baidu/ernie-4.5-vl-424b-a47b** (O=1.5, S=1.0, C=2.5)
> The image is severely blurred, making details indistinguishable, which drastically reduces overall readability. Colors are vibrant but the extreme motion blur overshadows any fidelity.

## Key Observations

### 1. Strong consensus on a severely degraded image

74% of retained models gave the minimum score (1.0) for overall quality. All models correctly identified the image as non-document and heavily degraded. The primary disagreement is on color fidelity, which is the most subjective dimension for this test image.

### 2. Calibration clusters

**Exact Sonnet match** (1.5/1.0/2.5) — 3 models:

- `qwen/qwen2.5-vl-72b-instruct`, `qwen/qwen2.5-vl-32b-instruct`, `baidu/ernie-4.5-vl-424b-a47b`

**Close to Sonnet** (1.0/1.0/2.5) — 3 models:

- `openai/gpt-5.1`, `google/gemini-3-flash-preview`, `openai/gpt-4.1-mini`

**Strictest** (1.0/1.0/1.0) — 4 models:

- `gemini-3.1-pro`, `qwen3-vl-235b`, `qwen3-vl-8b`, `qwen-2.5-vl-7b`

### 3. Strict models may underrate color on real documents

Models giving 1.0 for color fidelity (gemini-3.1-pro, qwen3-vl-235b, qwen3-vl-8b, qwen-2.5-vl-7b) may be too aggressive — colors ARE partially identifiable in the test image. These models may also underrate good-but-imperfect documents.

### 4. Cost vs. quality tradeoff

The cheapest well-calibrated models:
- `qwen/qwen2.5-vl-32b-instruct` ($0.20/$0.60) — matches Sonnet exactly
- `google/gemini-2.5-flash` ($0.30/$2.50) — very close calibration
- `qwen/qwen3-vl-32b-instruct` ($0.10/$0.42) — slightly strict but well-calibrated

These cost 10-100x less than Sonnet ($3/$15) while producing comparable scores.

### 5. Qwen-2.5-VL-7B as student baseline

`qwen/qwen-2.5-vl-7b-instruct` (the base model used for DeQA fine-tuning) gave 1.0/1.0/1.0 — the strictest possible scores. After DeQA training, the model should produce more nuanced scores across the quality spectrum. The full evaluation on actual documents will reveal whether the teacher signal meaningfully improves calibration.

## Comparison: Anthropic Baseline vs Multi-Model Field

Claude Sonnet 4.6 and Opus 4.6 both scored **1.5 / 1.0 / 2.5** (see baseline section above).

| Score dimension | Sonnet/Opus 4.6 | Multi-model median | Multi-model mode |
|-----------------|-----------------|--------------------| -----------------|
| Overall         | 1.5             | 1.0                | 1.0              |
| Sharpness       | 1.0             | 1.0                | 1.0              |
| Color fidelity  | 2.5             | 2.0                | 2.0              |

Sonnet/Opus are slightly more generous than the median but within a reasonable range. Their scores match several models exactly (qwen2.5-vl-72b, qwen2.5-vl-32b, ernie-4.5-vl-424b).

## Next Steps

1. **Run on actual document images** — the test image is a motion-blurred nighttime photo, which all models correctly identify. Real IQA calibration requires document images spanning the 1-5 quality range.
2. **Compute SRCC/PLCC correlation** against DIQA-5000 ground truth to quantify which models produce the most human-aligned scores.
3. **Shortlist 2-3 teachers** for pseudo-label generation based on calibration quality and cost.
4. **Retry gpt-5.4, o3, o4-mini** via direct OpenAI API if OpenRouter resolves the routing issue.

## Test Execution

```bash
cd DeQA-Score
PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
    ../results/vlm_teacher_eval/smoke_test_results/run_multimodel_smoke.py
```

Raw JSON results: `multimodel_raw_results.json`
