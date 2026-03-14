# Experiment Registry

Reverse-chronological log of completed experiments. Each entry follows a standardized template.

---

## EXP-009: 13-Model Consensus Analysis

| Field | Value |
|---|---|
| Date | 2026-03-06 |
| Status | DONE |
| Branch | `docs/ood-detector-next-iteration-analysis` |
| Report | [NEXT_ITERATION_ANALYSIS.md](../results/tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) |

**Hypothesis:** A structured multi-model consultation (13 frontier LLMs) can identify blind spots and rank improvement strategies for the OOD detector more reliably than a single analysis.

**Method:** Sent identical prompts with full project context (OOD detector README, CLAUDE.md, 8-option proposal) to 13 models via OpenRouter: GPT-5.2, Gemini 3.1 Pro, Gemini 3 Flash, DeepSeek V3.2, Minimax M2.5, Grok 4.1 Fast, Qwen3.5-397B, Qwen3.5-Plus, Kimi K2.5, GLM-5, Arcee Trinity, Nemotron Nano 9B, GLM-4.5 Air. Each model provided verdict, per-option analysis, ranking, confidence score, and key insights.

**Key Results:**
- Unanimous (13/13): real-world OOD eval must happen before expansion; circular training problem is real; public datasets (Option 1) are highest priority
- Strong consensus (10+/13): GMM should not precede diversity expansion; VLMs unreliable for fine-grained IQA; min 3 diverse VLMs for committee
- Novel approaches surfaced: PCA dimensionality reduction, ODIN/energy-based ensemble (+5-10% AUROC), conformal prediction, "Not-a-Document" class
- Average confidence: 8.2/10

**Conclusions:** Public dataset expansion for evaluation (RVL-CDIP, Tobacco800) is the unanimous top priority. The circular training problem (SigLIP2 labeling its own training data) requires external signals. VLM committee labeling is supported but with major caveats — VLMs measure readability, not optical quality.

**Follow-up:** [real-world-ood](backlog.md#real-world-ood), [public-dataset-expand](backlog.md#public-dataset-expand), [dual-embedding-ood](backlog.md#dual-embedding-ood)

---

## EXP-008: Prompt Optimization

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Branch | `feat/research-infrastructure` |
| Report | [VLM_TEACHER_EVALUATION.md](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S4.3 |
| Data | `results/vlm_teacher_eval/full_eval/prompt_optimization/` (2 models), `results/vlm_teacher_eval/full_eval/ab_test/` (2 models x 44 images) |

**Hypothesis:** Prompt engineering (resolution, few-shot examples, multi-sample averaging, per-dimension prompts) can meaningfully improve VLM-human correlation.

**Method:** Two experiments: (1) 7-arm optimization on 23 stratified images with Gemini 3 Flash and Qwen 3.5 Flash. Arms: baseline, 3-prompt, hybrid, few-shot, multi-sample, 2048px resize, no-resize. (2) A/B test of 1-prompt vs 3-prompt on 44 stratified images with Gemini 3 Flash and GPT-4.1.

**Key Results:**
- No-resize appeared dominant on n=23 (+0.042 wSRCC over baseline for Gemini), but **full-scale validation on n=1,000 disproved this** (actual delta: -0.009)
- Multi-sample averaging: +0.019 wSRCC at 12.5x cost — impractical
- Separate per-dimension prompts improve color fidelity (+0.015 to +0.037 SRCC) at 2-3x latency
- Few-shot examples hurt Gemini (-0.073) but helped Qwen (+0.096) — model-specific
- Standard 1024px resize, single prompt remains the recommended default

**Conclusions:** Small-sample prompt optimization (n=23) is actively misleading. Minimum n=200 stratified samples recommended. The standard single-prompt baseline is not meaningfully improvable through prompt engineering alone.

**Follow-up:** None — this line of investigation is concluded.

---

## EXP-007: VLM Teacher Evaluation

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Branch | `feat/research-infrastructure` |
| Report | [VLM_TEACHER_EVALUATION.md](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) |
| Data | `results/vlm_teacher_eval/full_eval/checkpoints/` (9 JSONL files, 1,000 records each), `results/vlm_teacher_eval/full_eval/checkpoints_synthetic/` (3 JSONL files, 520 records each) |

**Hypothesis:** Frontier VLMs can approximate human quality judgments (SRCC > 0.7) for pseudo-labeling DIQA without human annotation.

**Method:** 7 VLMs evaluated on 1,000 DIQA-5000 test images across 3 quality dimensions. Temperature 0.0, 1024px resize, structured JSON output via OpenRouter. Models: Gemini 3 Flash Preview, GPT-4.1, Gemini 2.5 Pro, Qwen 3.5 Flash, Claude Haiku 4.5, Qwen3-VL-8B Instruct, Qwen3-VL-8B Thinking. Bootstrapped 95% CIs (1,000 iterations). Cross-domain evaluation on 520 synthetic images (13 OOD categories). Fine-tuned models (HyperIQA++, SigLIP2-IQA) also benchmarked on synthetic set.

**Key Results:**
- Best: Gemini 3 Flash Preview wSRCC = 0.708 [95% CI: 0.672-0.740], approaching 3Specialist baseline (0.716)
- GPT-4.1 second (wSRCC = 0.669), best MAE: Claude Haiku 4.5 (0.68)
- All VLMs exhibit systematic positive bias (+0.5 to +1.5 MOS points)
- Reasoning models underperform non-reasoning counterparts
- On synthetic OOD: VLMs (GPT-4.1 wSRCC 0.757) substantially outperform fine-tuned models (HyperIQA++ 0.602), validating pseudo-labeling approach for OOD expansion
- Universal failure modes: binarized (negative SRCC), extreme DPI (negative), pristine (near-zero)

**Conclusions:** VLMs approach but do not exceed the supervised baseline on DIQA-5000. However, they generalize far better to OOD documents. Calibration is mandatory. Multi-model consensus recommended for production pseudo-labeling.

**Follow-up:** [calibration](backlog.md#calibration), [e2e-validation](backlog.md#e2e-validation), [ensemble-optimization](backlog.md#ensemble-optimization)

---

## EXP-006: SigLIP2 Full Extraction + OOD v2

| Field | Value |
|---|---|
| Date | 2026-03-08 |
| Status | DONE |
| Branch | `feat/research-infrastructure` |
| Report | [siglip2_diqa5000/README.md](../results/siglip2_diqa5000/README.md) |
| Data | `results/siglip2_diqa5000/` (3 JSONL files + 3 NPZ embedding files + OOD detector v2 + summary.json) |

**Hypothesis:** Re-extracting embeddings from the correct IQA-only checkpoint will resolve the 8-unit train/test Mahalanobis distance shift caused by 445 missing keys.

**Method:** Extracted all 5,000 DIQA-5000 embeddings from `siglip2_iqa_best.pt` (correct IQA checkpoint, 22 missing keys for non-IQA heads — expected) on NVIDIA L4 via Modal. Re-fitted Mahalanobis OOD detector with Ledoit-Wolf shrinkage on 4,000 train+val embeddings. Full multi-task predictions (IQA mu/sigma_sq, script, source, orientation, shadow, warping) extracted for all splits.

**Key Results:**
- Train+val median Mahalanobis distance: 23.7 (was 24.1 with old checkpoint)
- Test median: 31.4 (was 32.6) — healthy range, no anomalous shift
- v2 thresholds: p95=30.8 (train+val), p99=58.2 (test)
- Confirmed 22 missing keys are expected (non-IQA heads), 0 unexpected keys
- IQA output range: mu in [-0.17, 0.73], rescale via `mu * 4.0 + 1.0` to MOS [1,5]

**Conclusions:** Checkpoint mismatch resolved. The v2 OOD detector is healthy and production-ready for DIQA-5000-like documents. The remaining train/test distance gap (~8 units median) reflects genuine distribution differences in the test set, not an artifact.

**Follow-up:** [real-world-ood](backlog.md#real-world-ood)

---

## EXP-005: OOD Detector v1

| Field | Value |
|---|---|
| Date | 2026-03-06 |
| Status | DONE (superseded by EXP-006 v2) |
| Report | [tier1_ood_detector/README.md](../results/tier1_ood_detector/README.md) |
| Data | `results/tier1_ood_detector/ood_detector_v2.npz`, `results/tier1_ood_detector/embeddings/` |

**Hypothesis:** Mahalanobis distance in SigLIP2's 768-dim embedding space can distinguish DIQA-5000 in-distribution documents from out-of-distribution ones.

**Method:** Fitted multivariate Gaussian (mean + Ledoit-Wolf shrinkage covariance) on 4,400 DIQA-5000 train+val embeddings. Evaluated against 370 synthetic OOD documents (13 categories). Threshold calibrated from test p95.

**Key Results:**
- AUROC = 0.9963 on synthetic OOD vs DIQA-5000 test
- TPR 99.5% at 5% FPR (threshold = 46.0)
- All 13 OOD categories detected with AUROC >= 0.97
- Inference: ~1-2ms per image (matrix-vector multiply)
- Known issue: checkpoint mismatch caused 8-unit train/test shift (resolved in EXP-006)

**Conclusions:** Mahalanobis distance is highly effective for OOD detection on synthetic data. However, 13-model consensus (EXP-009) warns that synthetic AUROC likely overestimates real-world performance. Real-world evaluation needed.

**Follow-up:** [EXP-006](#exp-006-siglip2-full-extraction--ood-v2), [real-world-ood](backlog.md#real-world-ood)

---

## EXP-004: Synthetic OOD Dataset Creation

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Report | [VLM_TEACHER_EVALUATION.md](../results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) S3.3 |

**Hypothesis:** A programmatically generated synthetic dataset spanning diverse OOD categories can test VLM and OOD detector generalization beyond DIQA-5000.

**Method:** Generated 520 images across 13 categories with controlled degradation parameters: in-distribution Latin/Cyrillic (150), non-Latin scripts — Tibetan/Myanmar/Ethiopic (90), adversarial scripts — Fraktur/Nastaliq (40), layout variants — CJK vertical/forms (60), extreme degradation — binarized/heavily degraded (60), multiscript (30), DPI extremes (60), pristine (30). Ground truth MOS derived from generation parameters.

**Key Results:**
- Successfully covers script, layout, degradation, and DPI diversity gaps in DIQA-5000
- Enabled cross-domain evaluation in EXP-007 and OOD detector validation in EXP-005
- Revealed universal VLM failure modes: binarized, extreme DPI, pristine documents

**Conclusions:** Useful for identifying failure modes but synthetic-to-real domain gap limits conclusions about production performance (confirmed by 13-model consensus in EXP-009).

**Follow-up:** [EXP-005](#exp-005-ood-detector-v1), [EXP-007](#exp-007-vlm-teacher-evaluation)

---

## EXP-003: IQA Baseline Benchmarking

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Report | `results/iqa_baselines/baseline_summary.json` |
| Data | `results/iqa_baselines/baseline_summary.json` |

**Hypothesis:** Pretrained natural-image IQA models transfer poorly to document images, establishing the need for document-specific training.

**Method:** Evaluated 6 NR-IQA models from the pyiqa library on DIQA-5000 test (1,000 images) and synthetic OOD (520 images) via Modal GPU: DBCNN, HyperIQA, StairIQA, MUSIQ, TReS, RichIQA. Script: `modal/benchmark_iqa_baselines.py`.

**Key Results:**
- DIQA-5000: Best wSRCC = 0.490 (RichIQA), worst = 0.153 (MUSIQ)
- Synthetic OOD: Better performance, best wSRCC = 0.747 (TReS)
- All models substantially below VLM teachers (0.708) and DeQA-Doc specialists (0.716) on DIQA-5000
- Large DIQA/synthetic gap confirms domain-specific challenges in document IQA

**Conclusions:** Pretrained IQA models designed for natural images perform poorly on documents. Document-specific training (DeQA-Doc) or VLM pseudo-labeling is necessary.

**Follow-up:** [EXP-007](#exp-007-vlm-teacher-evaluation)

---

## EXP-002: VLM Model Selection (Smoke Test)

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Branch | `docs/ood-detector-next-iteration-analysis` |
| Report | [2026-03-07_diqa_sample_eval.md](../results/vlm_teacher_eval/smoke_test_results/2026-03-07_diqa_sample_eval.md), [2026-03-07_multimodel_eval.md](../results/vlm_teacher_eval/smoke_test_results/2026-03-07_multimodel_eval.md) |
| Data | `results/vlm_teacher_eval/smoke_test_results/diqa_eval_raw_results.json`, `results/vlm_teacher_eval/smoke_test_results/multimodel_raw_results.json` |

**Hypothesis:** A small-sample smoke test (n=7 DIQA images) across 26+ models can identify the best VLM teacher candidates for full-scale evaluation.

**Method:** Two phases: (1) 23 models on 1 motion-blurred nighttime photo (singapore_flyer.jpg) to test basic calibration and exclude broken models. 8 models excluded (API failures, poor calibration). (2) 26 models on 7 stratified DIQA-5000 test images covering all 15 quality buckets (5 levels x 3 dimensions). 182/182 API calls succeeded.

**Key Results:**
- Top performers (n=7): GPT-4.1 (wSRCC 0.880), Gemini 3 Flash (0.857), GPT-5-mini (0.841), Gemini 2.5 Pro (0.829), Claude Sonnet 4.6 (0.827)
- Best calibration: Gemini 2.5 Pro (MAE 0.331, only model with negative overall bias)
- 8 models excluded for API failures or severe miscalibration
- All Qwen VL-specific models severely overrate (+1.0 to +1.7 bias)
- **Caution: n=7 rankings diverged dramatically from n=1,000** (Haiku dropped from 0.813 to 0.579, GPT-4.1 from 0.880 to 0.669)

**Conclusions:** Useful for excluding broken models and getting a rough shortlist, but small-sample VLM benchmarks are unreliable for absolute ranking. Full-scale evaluation (EXP-007) is essential.

**Follow-up:** [EXP-007](#exp-007-vlm-teacher-evaluation)

---

## EXP-001: VLM Smoke Test (Claude Sonnet/Opus)

| Field | Value |
|---|---|
| Date | 2026-03-07 |
| Status | DONE |
| Report | [2026-03-07_baseline.md](../results/vlm_teacher_eval/smoke_test_results/2026-03-07_baseline.md) |

**Hypothesis:** Claude models produce consistent IQA scores across different access methods (Anthropic API, OpenRouter, Claude Code extension) and model tiers (Sonnet vs Opus).

**Method:** Rated 1 image (singapore_flyer.jpg, motion-blurred nighttime photo) via 5 access methods: Anthropic API direct, OpenRouter API, Claude Code extension (standard), Claude Code extension (1M context), Claude Code extension (Opus 4.6). Temperature 0.0, standard IQA prompt.

**Key Results:**
- All 5 methods produce identical scores: Overall=1.5, Sharpness=1.0, Color=2.5
- Opus 4.6 produces richer reasoning (identifies "Singapore Flyer" by name, explains physics of blur) but same numeric scores
- Anthropic direct ~15% faster than OpenRouter (3547ms vs 4145ms)

**Conclusions:** Access method and model tier do not affect numeric scoring at temperature 0. Opus is better for generating training rationales but unnecessary for score generation alone.

**Follow-up:** [EXP-002](#exp-002-vlm-model-selection-smoke-test)
