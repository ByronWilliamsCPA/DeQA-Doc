# OCR-IQA Correlation Leaderboard

Unified leaderboard for all models evaluated on the OCR-IQA correlation dataset (n=1,200).

**Dataset:** 200 FUNSD/FUNSD+ documents × 6 distortion tiers (ORIGINAL, PRISTINE, HIGH, MEDIUM, LOW, DEGRADED).

**Last updated:** 2026-03-09

---

## Section 1: IQA Model Ranking

**Question:** How well does each IQA model predict OCR accuracy (CER) on distorted documents?

**Metric:** `MainScore = mean |SRCC(model_score, CER)|` averaged across all OCR engines.
Higher = better at predicting which images will cause OCR failures.

| Rank | Model | Type | MainScore | SRCC_T | SRCC_E | SRCC_R | SRCC_G | SRCC_P | SRCC_D | SRCC_K | SRCC_GLM | SRCC_DS | SRCC_MOS | n | Notes |
| ---- | ----- | ---- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | -------- | ------- | -------- | - | ----- |
| 1 | DeQA-Doc MOS | Fine-tuned MLLM | **0.511** | -0.647 | -0.637 | -0.543 | -0.435 | -0.658 | -0.632 | -0.369 | -0.343 | -0.339 | — | 1,200 | mPLUG-Owl2; ground-truth IQA reference |
| 2 | GPT-4.1 | VLM zero-shot | **0.534** | -0.655 | -0.651 | -0.506 | -0.322 | — | — | — | — | — | 0.847 | 1,179 | Best per-engine on Tesseract & EasyOCR |
| 3 | Gemini 3 Flash Preview | VLM zero-shot | **0.491** | -0.583 | -0.639 | -0.456 | -0.286 | — | — | — | — | — | 0.818 | 1,177 | Zero-shot via OpenRouter |

**Column key:**
- `SRCC_T` = SRCC(model_score, Tesseract CER)
- `SRCC_E` = SRCC(model_score, EasyOCR CER)
- `SRCC_R` = SRCC(model_score, RapidOCR CER)
- `SRCC_G` = SRCC(model_score, Google Cloud Vision CER)
- `SRCC_P` = SRCC(model_score, PP-OCRv5 CER)
- `SRCC_D` = SRCC(model_score, docTR CER)
- `SRCC_K` = SRCC(model_score, Kraken CER)
- `SRCC_GLM` = SRCC(model_score, GLM-OCR CER)
- `SRCC_DS` = SRCC(model_score, DeepSeek-OCR2 CER)
- `SRCC_MOS` = SRCC(VLM_score, DeQA MOS) — agreement with DeQA reference

---

## Section 2: OCR Engine Ranking

**Question:** Which OCR engines are most accurate and most robust to image distortion?

**Metric:** Mean CER across all 1,200 images (lower = better).

| Rank | Engine | Type | Mean CER | CER_ORIG | CER_PRIS | CER_HIGH | CER_MED | CER_LOW | CER_DEG | SRCC_MOS | Notes |
| ---- | ------ | ---- | -------- | -------- | -------- | -------- | ------- | ------- | ------- | -------- | ----- |
| 1 | docTR | Local (PyTorch) | **0.308** | 0.187 | 0.187 | 0.332 | 0.349 | 0.407 | 0.385 | -0.632 | Best local engine; lowest clean CER |
| 2 | PP-OCRv5 | Local (PaddlePaddle) | **0.315** | 0.189 | 0.189 | 0.337 | 0.343 | 0.422 | 0.410 | -0.658 | Strongest MOS correlation; near-GCloud accuracy |
| 3 | Google Cloud Vision | Cloud API | **0.316** | 0.284 | 0.284 | 0.328 | 0.315 | 0.349 | 0.339 | -0.435 | Most robust to distortion (flattest CER curve) |
| 4 | GLM-OCR | VLM OCR (~0.5B) | **0.361** | 0.257 | 0.257 | 0.266 | 0.271 | 0.489 | 0.628 | -0.343 | Best clean CER; sharp degradation on LOW/DEG |
| 5 | RapidOCR | Local (Docling) | **0.500** | 0.387 | 0.387 | 0.511 | 0.530 | 0.600 | 0.584 | -0.543 | Good balance of accuracy and speed |
| 6 | Tesseract | Local (Docling) | **0.663** | 0.437 | 0.437 | 0.729 | 0.744 | 0.819 | 0.811 | -0.647 | Most distortion-sensitive (largest CER gap) |
| 7 | EasyOCR | Local (Docling) | **0.683** | 0.524 | 0.524 | 0.691 | 0.745 | 0.804 | 0.810 | -0.637 | Highest baseline CER on clean images |
| 8 | Kraken | Local (PyTorch) | **0.933** | 0.880 | 0.880 | 0.962 | 0.958 | 0.950 | 0.970 | -0.369 | Historical doc focus; poor on modern forms |
| 9 | DeepSeek-OCR2 | VLM OCR (3B) | **1.145** | 0.594 | 0.594 | 1.850 | 1.218 | 1.449 | 1.166 | -0.339 | Heavy hallucination (CER>1); outputs HTML tables |

**Column key:**
- `CER_ORIG/PRIS/HIGH/MED/LOW/DEG` = Mean CER at each distortion tier
- `SRCC_MOS` = SRCC(CER, DeQA MOS) — how well DeQA predicts this engine's accuracy

---

## Section 3: VLM Quality Assessment vs DeQA MOS

**Question:** How well do VLM zero-shot quality scores agree with DeQA-Doc's trained IQA model?

| Rank | Model | SRCC | PLCC | SRCC 95% CI | PLCC 95% CI | n | Notes |
| ---- | ----- | ---- | ---- | ----------- | ----------- | - | ----- |
| 1 | GPT-4.1 | **0.847** | 0.837 | [0.827, 0.864] | [0.820, 0.852] | 1,179 | Best agreement with DeQA |
| 2 | Gemini 3 Flash Preview | **0.818** | 0.826 | [0.795, 0.838] | [0.808, 0.843] | 1,177 | Slightly lower SRCC but close PLCC |

---

## Section 4: Per-Tier Monotonicity

**Validation:** IQA models should score ORIGINAL/PRISTINE highest and DEGRADED lowest. VLM overall scores (1-5 scale) and DeQA MOS should decrease monotonically with distortion severity.

| Tier | DeQA MOS | GPT-4.1 | Gemini 3 Flash |
| ---- | -------- | ------- | -------------- |
| ORIGINAL | 3.354 | 4.192 | 3.675 |
| PRISTINE | 3.354 | 4.188 | 3.671 |
| HIGH | 3.073 | 3.580 | 3.232 |
| MEDIUM | 3.015 | 3.405 | 3.068 |
| LOW | 2.942 | 3.139 | 2.927 |
| DEGRADED | 2.947 | 2.950 | 2.907 |

All three models show monotonic decrease from ORIGINAL to DEGRADED (with minor LOW/DEGRADED overlap in DeQA MOS).

---

## Pending Models

The following models have been configured but results are not yet available:

| Model | Type | Status | Script |
| ----- | ---- | ------ | ------ |
| Surya | Neural OCR | Staged for Modal (GPU recommended) | `ocr/surya_engine.py` |
| MinerU2.5 (1.2B) | VLM OCR | Pending launch on Modal | `modal/run_vlm_ocr.py` |
| PaddleOCR-VL-1.5 (0.9B) | VLM OCR | Abandoned (~1 img/min on L4) | `modal/run_vlm_ocr.py` |
| MonkeyOCR-pro (3B) | VLM OCR | Blocked (custom lib) | Needs `monkeyocr` library |
| Adobe PDF Extract | Cloud API | Pending credentials | `ocr/adobe_extract.py` |
| TrOCR | Line-level only | Deferred (needs detection wrapper) | — |

Once additional results are available, Section 2 will be updated.

---

## Notes

- **DeQA-Doc MOS** is the ground-truth IQA reference from the fine-tuned mPLUG-Owl2 model (3 specialists ensemble). It is not a zero-shot predictor — it was trained on DIQA human labels.
- **VLM zero-shot** models use a structured prompt asking for 1-5 quality scores across three dimensions (overall, sharpness, color fidelity). Only overall quality is used for CER correlation.
- **SRCC values are negative** because higher quality → lower CER (better OCR). Stronger negative correlation = better prediction.
- **docTR** and **PP-OCRv5** outperform Google Cloud Vision on clean images (CER_ORIG 0.187/0.189 vs 0.284) but are slightly more distortion-sensitive.
- **PP-OCRv5** shows the strongest CER-MOS correlation (-0.658) across all engines, making it the best "canary" for quality prediction.
- **Google Cloud Vision** has the flattest CER curve (smallest ORIG-to-DEG gap), suggesting it is most robust to quality degradation.
- **GLM-OCR** (zai-org/GLM-OCR, ~0.5B) achieves the best clean-image CER (0.257) but degrades sharply on LOW/DEGRADED tiers (0.489/0.628), resulting in weak MOS correlation (-0.343).
- **DeepSeek-OCR2** (3B) produces HTML-table formatted output that inflates CER >1 on most tiers (hallucination). Clean-image CER (0.594) is reasonable but overall accuracy is worst of all engines.
- **Kraken** shows very high CER (0.933) because its default English model is designed for historical documents, not modern forms.
- **Bootstrap 95% CI** computed with 1,000 resamples for VLM metrics.

## Data Sources

| Type | Location |
| ---- | -------- |
| DeQA per-image scores | `research/ocr_iqa_correlation/data/deqa_results/deqa_scores.jsonl` |
| OCR per-image results | `research/ocr_iqa_correlation/data/ocr_results/*.jsonl` |
| VLM per-image scores | `research/ocr_iqa_correlation/outputs/vlm_eval_checkpoints/*.jsonl` |
| Correlation metrics | `research/ocr_iqa_correlation/outputs/correlation_report.json` |
| VLM eval metrics | `research/ocr_iqa_correlation/outputs/vlm_eval_metrics.json` |
| Analysis script | `research/ocr_iqa_correlation/scripts/05_analyze.py` |
| VLM eval script | `research/ocr_iqa_correlation/scripts/06_vlm_eval.py` |
| Modal OCR script | `modal/run_vlm_ocr.py` |
