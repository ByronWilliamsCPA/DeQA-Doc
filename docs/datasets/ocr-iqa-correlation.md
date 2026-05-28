### OCR-IQA Correlation Dataset

> **Quick Stats**: 1,200 images | Controlled distortions | 9 OCR engines x 6 quality tiers | 2 VLM assessors | Paired analysis
>
> **License**: Research (derived from FUNSD + FUNSD+) | **Commercial Use**: Restricted

#### 1. Overview

| Attribute | Value |
|-----------|-------|
| **Full Name** | OCR-IQA Correlation Test Dataset |
| **Version** | 1.0 |
| **Release Date** | 2026-03-09 |
| **Maintainer** | DeQA-Doc project |
| **Paper** | Internal research (no publication) |
| **License** | Research use (inherits FUNSD/FUNSD+ licenses) |
| **Documentation Status** | Complete (all 6 phases executed) |

This dataset tests the hypothesis: **do DeQA image quality scores predict OCR accuracy on document images?** By applying controlled distortions to 200 documents with known ground-truth text, then measuring both OCR error rates (CER/WER) and DeQA quality scores, it quantifies the correlation between predicted quality and actual downstream OCR performance.

**Key finding**: Strong, statistically significant negative correlations (SRCC up to -0.658, paired SRCC up to -0.683) confirm that DeQA-Doc quality scores are predictive of OCR degradation across all 9 engines tested.

#### 2. Source Data Inventory

##### 2.1 Source Datasets

| Source | Path | Images Used | Total Available |
|--------|------|-------------|-----------------|
| **FUNSD** | `/mnt/e/image_detection/01_base_data/forms/funsd/train/` | 50 | 149 (train only) |
| **FUNSD+** | `/mnt/e/image_detection/01_base_data/forms/funsd_plus/` | 150 | 1,026 (train only) |

Base images selected via stratified sampling (seed=42, min 20 chars GT text). Over-samples FUNSD for balance.

##### 2.2 Provided File Types

| File Type | Format(s) | Description |
|-----------|-----------|-------------|
| **Base Images** | PNG | 200 sampled document images (local copies) |
| **Distorted Images** | PNG | 1,000 distorted versions (5 tiers x 200) + 200 originals |
| **Ground Truth Text** | TXT | Per-image text files from FUNSD/FUNSD+ annotations |
| **OCR Results** | JSONL | Per-engine OCR output for all 1,200 images (9 files) |
| **DeQA Scores** | JSONL | MOS + probability distribution per image |
| **VLM Scores** | JSONL | Zero-shot quality scores from 2 frontier VLMs |
| **Sample Manifest** | JSON | Image selection with provenance metadata |
| **Master Record** | JSONL | Unified dataset with all measurements |
| **Distortion Metadata** | JSONL | Per-image distortion parameters and IQA labels |

##### 2.3 Dataset Structure

| Component | Path | Count | Status |
|-----------|------|-------|--------|
| **Base images** | `data/base_images/` | 200 | ✅ Complete |
| **GT text** | `data/gt_text/` | 200 | ✅ Complete |
| **Manifest** | `data/sample_manifest.json` | 1 | ✅ Complete |
| **Distortion metadata** | `data/distortion_metadata.jsonl` | 1,200 records | ✅ Complete |
| **ORIGINAL tier** | `data/distorted/ORIGINAL/` | 200 | ✅ Complete |
| **PRISTINE tier** | `data/distorted/PRISTINE/` | 200 | ✅ Complete |
| **HIGH tier** | `data/distorted/HIGH/` | 200 | ✅ Complete |
| **MEDIUM tier** | `data/distorted/MEDIUM/` | 200 | ✅ Complete |
| **LOW tier** | `data/distorted/LOW/` | 200 | ✅ Complete |
| **DEGRADED tier** | `data/distorted/DEGRADED/` | 200 | ✅ Complete |
| **OCR results** | `data/ocr_results/` | 9 files | ✅ Complete |
| **DeQA scores** | `data/deqa_results/` | 1 file | ✅ Complete |
| **VLM checkpoints** | `data/vlm_checkpoints/` | 2 files | ✅ Complete |
| **Master record** | `data/dataset.jsonl` | 1,200 records | ✅ Complete |

##### 2.4 Labels & Annotations

| Label Type | Format | Granularity | Description |
|------------|--------|-------------|-------------|
| **Ground Truth Text** | TXT | Page-level | Human-validated text from FUNSD/FUNSD+ annotations |
| **OCR Transcriptions** | JSONL | Page-level | OCR output per engine per image |
| **CER/WER Metrics** | JSONL | Image-level | Character/word error rates vs GT |
| **DeQA Quality Scores** | JSONL | Image-level | MOS + 5-level probability distribution |
| **VLM Quality Scores** | JSONL | Image-level | Zero-shot 1-5 scale (overall, sharpness, color fidelity) |
| **Distortion Parameters** | JSONL | Image-level | Seed, tier, IQALabels from hybrid pipeline |

##### 2.5 Ground Truth Provenance

| Aspect | Details |
|--------|---------|
| **Text Annotation Method** | Human Expert (FUNSD) + Human Expert (FUNSD+) |
| **Text Provenance Tier** | Tier 1 (Human-labeled) — form entity annotations |
| **Quality Score Method** | Model-predicted (DeQA mPLUG-Owl2) |
| **Quality Provenance Tier** | Tier 2 (Model) — fine-tuned VLM inference |
| **VLM Zero-Shot Scores** | Tier 2 (Model) — frontier VLMs (GPT-4.1, Gemini 3 Flash) |
| **Distortion Labels** | Tier 0 (Exact) — deterministic pipeline with known parameters |
| **GT Label Coverage** | 100% text GT, 100% DeQA scores, 98%+ VLM scores |

#### 3. Project Usage

| Aspect | Details |
|--------|---------|
| **Purpose** | Correlation analysis: OCR accuracy vs DeQA quality scores |
| **Local Path** | `research/ocr_iqa_correlation/data/` |
| **Pipeline Code** | `research/ocr_iqa_correlation/` |
| **Analysis Report** | `research/ocr_iqa_correlation/OCR_IQA_CORRELATION_ANALYSIS.md` |
| **Leaderboard** | `results/LEADERBOARD_OCR_IQA.md` |
| **Preprocessing** | 6-step pipeline (extract GT -> distort -> OCR -> DeQA -> analyze -> VLM eval) |

**Research questions** (all answered):

1. Does DeQA MOS correlate with OCR accuracy (CER/WER)? **Yes** — SRCC -0.435 to -0.658 across engines
2. Is the correlation consistent across OCR engines? **Yes** — all 9 engines show significant negative SRCC (p < 10^-33)
3. Do paired deltas (DCER vs DMOS) show stronger correlation than absolute values? **Yes** — Tesseract improves from -0.647 to -0.683
4. Do adjacent quality tiers produce statistically significant CER differences? **Partially** — PRISTINE->HIGH is strongly significant (p < 10^-7); LOW->DEGRADED is not (tier collapse)

#### 4. Dataset Statistics

##### 4.1 Sample Counts

| Metric | Value |
|--------|-------|
| **Base Images** | 200 |
| **Total Images** | 1,200 (200 base x 6 tiers including ORIGINAL) |
| **FUNSD Images** | 50 (25% of base) |
| **FUNSD+ Images** | 150 (75% of base) |
| **Quality Tiers** | 6 (ORIGINAL + 5 distortion levels) |
| **OCR Engines** | 9 (4 traditional + 2 neural + 1 cloud + 2 VLM-based) |
| **Total OCR Runs** | 10,800 (1,200 images x 9 engines) |
| **VLM Assessors** | 2 (GPT-4.1, Gemini 3 Flash Preview) |
| **Total Size on Disk** | ~35.8 MB (base images only; full dataset including distortions TBD) |

##### 4.2 Text Statistics (from GT extraction)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Character Count** | 181 | 2,999 | 1,001 |

**Text Source**: `ground_truth` (human-validated FUNSD/FUNSD+ annotations)

##### 4.3 Directory Structure

```text
research/ocr_iqa_correlation/
+-- analysis/                    # Metrics and correlation modules
|   +-- cer_wer.py              # CER/WER computation (jiwer)
|   +-- correlation.py          # SRCC/PLCC computation
|   +-- paired_analysis.py      # Paired deltas + tier significance
|   +-- visualize.py            # Plot generation
+-- ocr/                        # OCR engine wrappers
|   +-- base.py                 # Abstract base class
|   +-- paddleocr_engine.py     # PP-OCRv5 (PaddlePaddle)
|   +-- doctr_engine.py         # docTR (PyTorch)
|   +-- kraken_engine.py        # Kraken
|   +-- docling_engines.py      # Tesseract, RapidOCR, EasyOCR
|   +-- gcloud_vision.py        # Google Cloud Vision API
|   +-- surya_engine.py         # Surya (staged)
|   +-- adobe_extract.py        # Adobe Extract API (pending)
+-- distortion/                 # Distortion pipeline
|   +-- apply_distortions.py    # Augraphy + Albumentations
+-- gt_extraction/              # Ground truth extraction
|   +-- funsd_parser.py         # FUNSD annotation parser
|   +-- funsd_plus_parser.py    # FUNSD+ annotation parser
|   +-- sampler.py              # Stratified sampling
+-- scripts/                    # Pipeline phases
|   +-- 01_extract_gt_and_sample.py
|   +-- 02_apply_distortions.py
|   +-- 03_run_ocr.py
|   +-- 04_run_deqa.py
|   +-- 05_analyze.py
|   +-- 06_vlm_eval.py
+-- data/                       # gitignored
|   +-- base_images/            # 200 base document images
|   +-- gt_text/                # 200 per-image GT text files
|   +-- sample_manifest.json    # Selection metadata
|   +-- distortion_metadata.jsonl  # 1,200 distortion records
|   +-- distorted/              # 6 tier subdirectories
|   |   +-- ORIGINAL/
|   |   +-- PRISTINE/
|   |   +-- HIGH/
|   |   +-- MEDIUM/
|   |   +-- LOW/
|   |   +-- DEGRADED/
|   +-- ocr_results/            # 9 per-engine JSONL files
|   +-- deqa_results/           # DeQA scores JSONL
|   +-- vlm_checkpoints/        # VLM evaluation checkpoints
|   +-- dataset.jsonl           # Master record (1,200 records)
+-- outputs/                    # Analysis results
|   +-- correlation_report.json # Statistical results
|   +-- vlm_eval_metrics.json   # VLM comparison metrics
|   +-- figures/                # 7+ PNG visualizations
+-- tests/                      # Unit tests
    +-- test_funsd_parser.py    # 3 tests
    +-- test_cer_wer.py         # 13 tests
```

#### 5. Quality Profile

##### 5.1 DeQA Quality Dimensions

| Dimension | Available | Score Range | Notes |
|-----------|-----------|-------------|-------|
| **Overall Quality** | ✅ Complete | 2.94 - 3.35 MOS | Primary analysis dimension |
| **Sharpness** | ✅ Complete | 1-5 MOS | Available in DeQA output |
| **Color Fidelity** | ✅ Complete | 1-5 MOS | Available in DeQA output |

##### 5.2 Quality Tier Design

| Tier | HybridProfile | Target Overall Quality | Observed CER Impact |
|------|---------------|------------------------|---------------------|
| **ORIGINAL** | (none) | Varies (source quality) | Baseline |
| **PRISTINE** | PRISTINE | 0.95 - 1.00 | Identical to ORIGINAL |
| **HIGH** | LIGHT | 0.70 - 0.95 | Significant jump (p < 10^-7) |
| **MEDIUM** | MODERATE | 0.40 - 0.80 | Moderate increase |
| **LOW** | HEAVY | 0.10 - 0.60 | High CER for all engines |
| **DEGRADED** | HISTORICAL | 0.00 - 0.50 | Converges with LOW (tier collapse) |

##### 5.3 Distortion Pipeline

Distortions applied via `HybridAugmentationPipeline` from `/home/byron/dev/image_detection/src/image_preprocessing_detector/synthetic/augmentation_hybrid.py`:

- **Augraphy** (document-specific): bleed-through, ink bleed, paper aging, letterpress, moire
- **Albumentations** (general): blur, noise, compression, geometric transforms

Deterministic seeding: `base_seed + image_idx * 100 + tier_idx`

##### 5.4 Degradation Coverage

| Degradation Type | Present | Source | Notes |
|------------------|---------|--------|-------|
| **Blur** | Yes | Albumentations | Gaussian, motion, defocus |
| **Noise** | Yes | Albumentations | Gaussian, salt-pepper |
| **Compression** | Yes | Albumentations | JPEG artifacts |
| **Ink Degradation** | Yes | Augraphy | Ink bleed, letterpress |
| **Paper Aging** | Yes | Augraphy | Yellowing, staining |
| **Geometric Distortion** | Yes | Albumentations | Rotation, perspective |
| **Bleed-Through** | Yes | Augraphy | Show-through from reverse |
| **Moire** | Yes | Augraphy | Screen/scan interference |

#### 6. OCR Evaluation

##### 6.1 OCR Engines

| Rank | Engine | Type | Mean CER | SRCC vs MOS | Notes |
|------|--------|------|----------|-------------|-------|
| 1 | **docTR** | Local (PyTorch) | 0.308 | -0.632 | Best local engine; lowest clean CER |
| 2 | **PP-OCRv5** | Local (PaddlePaddle) | 0.315 | **-0.658** | Strongest MOS correlation |
| 3 | **Google Cloud Vision** | Cloud API | 0.316 | -0.435 | Most robust to distortion |
| 4 | **GLM-OCR** | VLM (~0.5B) | 0.361 | -0.343 | Best clean CER; sharp degradation at LOW |
| 5 | **RapidOCR** | Local (Docling) | 0.500 | -0.543 | Good balance of accuracy and speed |
| 6 | **Tesseract** | Local (Docling) | 0.663 | -0.647 | Most distortion-sensitive (largest CER gap) |
| 7 | **EasyOCR** | Local (Docling) | 0.683 | -0.637 | Highest baseline CER on clean images |
| 8 | **Kraken** | Local (PyTorch) | 0.933 | -0.369 | Historical doc focus; poor on modern forms |
| 9 | **DeepSeek-OCR2** | VLM (3B) | 1.145 | -0.339 | Heavy hallucination (CER>1); outputs HTML tables |

##### 6.2 Architecture Stratification

| Category | Engines | SRCC Range | Quality Sensitivity |
|----------|---------|-----------|---------------------|
| **Traditional OCR** | PP-OCRv5, Tesseract, EasyOCR, RapidOCR | -0.543 to -0.658 | Highest — best quality discriminators |
| **Neural OCR** | docTR, Kraken | -0.369 to -0.632 | Mixed |
| **Cloud API** | Google Cloud Vision | -0.435 | Moderate — internal preprocessing absorbs degradation |
| **VLM-based OCR** | GLM-OCR, DeepSeek-OCR2 | -0.339 to -0.343 | Lowest — language priors compensate for degradation |

##### 6.3 Metrics

| Metric | Computation | Library |
|--------|-------------|---------|
| **CER** | Character Error Rate (Levenshtein distance / reference length) | jiwer |
| **WER** | Word Error Rate (word-level edit distance / reference words) | jiwer |

Text normalization: NFC unicode normalization, lowercase, whitespace collapse, strip.

Empty OCR output -> CER = 1.0, WER = 1.0.

##### 6.4 Results Summary

| Analysis | Method | Finding |
|----------|--------|---------|
| **Per-engine SRCC** | SRCC(CER, MOS) | Negative correlation: -0.339 to -0.658 (all p < 10^-33) |
| **Per-engine PLCC** | PLCC(CER, MOS) | Negative linear correlation: -0.148 to -0.581 |
| **Paired deltas** | SRCC(DCER, DMOS) | Stronger: Tesseract -0.647 -> -0.683 |
| **Tier significance** | Wilcoxon signed-rank | PRISTINE->HIGH strongly significant; LOW->DEGRADED not significant |
| **Cross-engine agreement** | All 9 engines | Consistent negative correlation direction |

#### 7. Model Evaluation Results

##### 7.1 IQA Model Ranking

| Rank | Model | Type | MainScore | Notes |
|------|-------|------|-----------|-------|
| 1 | GPT-4.1 | VLM zero-shot | **0.534** | Best per-engine on Tesseract & EasyOCR |
| 2 | DeQA-Doc MOS | Fine-tuned MLLM | **0.511** | mPLUG-Owl2; ground-truth IQA reference |
| 3 | Gemini 3 Flash Preview | VLM zero-shot | **0.491** | Zero-shot via OpenRouter |

MainScore = mean |SRCC(model_score, CER)| averaged across OCR engines.

##### 7.2 VLM vs DeQA Agreement

| Model | SRCC | SRCC 95% CI | PLCC | n |
|-------|------|-------------|------|---|
| **GPT-4.1** | **0.847** | [0.827, 0.864] | 0.837 | 1,179 |
| Gemini 3 Flash Preview | 0.818 | [0.795, 0.838] | 0.826 | 1,177 |

##### 7.3 Per-Tier Score Monotonicity

| Tier | DeQA MOS | GPT-4.1 | Gemini 3 Flash |
|------|----------|---------|----------------|
| ORIGINAL | 3.354 | 4.192 | 3.675 |
| PRISTINE | 3.354 | 4.188 | 3.671 |
| HIGH | 3.073 | 3.580 | 3.232 |
| MEDIUM | 3.015 | 3.405 | 3.068 |
| LOW | 2.942 | 3.139 | 2.927 |
| DEGRADED | 2.947 | 2.950 | 2.907 |

All three models show monotonic decrease from ORIGINAL to DEGRADED (with minor LOW/DEGRADED overlap in DeQA MOS).

#### 8. Known Issues & Limitations

- **FUNSD/FUNSD+ only**: Limited to form-style documents; does not cover prose, tables, or handwritten content
- **English-only GT**: Both source datasets are English, limiting script diversity
- **GT text is entity-level**: FUNSD annotations are form entities concatenated by ID order, not natural reading order
- **Model-predicted quality**: DeQA scores are model predictions, not human MOS — correlation tests model consistency, not ground truth alignment
- **Single distortion pipeline**: All distortions from one pipeline (HybridAugmentationPipeline); results may not generalize to other degradation sources
- **MOS scale compression**: DeQA MOS ranges only 2.94-3.35 across tiers (narrow dynamic range)
- **High baseline CER**: All engines show CER > 0.28 even on original images, likely due to form-specific OCR challenges (checkboxes, tables, handwriting)
- **DeepSeek-OCR2 hallucination**: CER > 1.0 on most tiers due to HTML-table output format
- **No FCA metric**: Reading-order-sensitive CER may inflate errors when OCR engines produce different text-block segmentation

#### 9. Content Composition

| Aspect | Details |
|--------|---------|
| **Domain** | Administrative forms, business documents |
| **Language(s)** | English (100%) |
| **Script(s)** | Latin (100%) |
| **Document Types** | Scanned forms with structured fields, headers, and handwritten annotations |

#### 10. References

##### Source Dataset Citations

```bibtex
@inproceedings{jaume2019funsd,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Jaume, Guillaume and Ekenel, Hazim Kemal and Thiran, Jean-Philippe},
  booktitle={ICDAR Workshop},
  year={2019}
}
```

##### Related Datasets

- [diqa-5000](diqa-5000.md) - Benchmark dataset with human MOS (what DeQA was trained on)
- [synth-ood-520](synth-ood-520.md) - Synthetic OOD test set (different evaluation methodology)

##### Related Documents

- [OCR-IQA Correlation Analysis](../../research/ocr_iqa_correlation/OCR_IQA_CORRELATION_ANALYSIS.md) - Full analysis report with figures
- [OCR-IQA Leaderboard](../../results/LEADERBOARD_OCR_IQA.md) - Unified engine and model rankings
- [DIQA Dataset Expansion Strategy](../architecture/diqa_dataset_expansion_strategy.md) - Broader context for OCR-based labeling

#### 11. Dataset-Specific Notes

##### 11.1 Pipeline Phases

| Phase | Script | Status | Description |
|-------|--------|--------|-------------|
| 1 | `scripts/01_extract_gt_and_sample.py` | ✅ Complete | Extract GT text, sample 200 images, copy locally |
| 2 | `scripts/02_apply_distortions.py` | ✅ Complete | Apply 5 distortion tiers via HybridAugmentationPipeline |
| 3 | `scripts/03_run_ocr.py` | ✅ Complete | Run 9 OCR engines on 1,200 images |
| 4 | `scripts/04_run_deqa.py` | ✅ Complete | Run DeQA mPLUG-Owl2 scorer on 1,200 images (Modal L4 GPU) |
| 5 | `scripts/05_analyze.py` | ✅ Complete | Compute correlations, paired analysis, visualizations |
| 6 | `scripts/06_vlm_eval.py` | ✅ Complete | Evaluate GPT-4.1 and Gemini 3 Flash as zero-shot quality assessors |

##### 11.2 Two-Venv Strategy

- **Main venv** (`research/ocr_iqa_correlation/.venv`): Phases 1-3, 5-6. Modern Python with docling, jiwer, augraphy, albumentations
- **DeQA venv** (`DeQA-Score/.venv`): Phase 4 only. torch 2.0.1, transformers 4.36.1 (mPLUG-Owl2 compatibility)

##### 11.3 Master Record Schema

```json
{
  "image_id": "funsd_0000971160",
  "source_dataset": "funsd",
  "tier": "MEDIUM",
  "image_path": "data/distorted/MEDIUM/funsd_0000971160.png",
  "gt_text_chars": 1543,
  "seed": 4200302,
  "actual_overall_quality": 0.67,
  "iqa_labels": {"blur": 0.3, "noise": 0.2, "compression": 0.15},
  "deqa_mos": 3.42,
  "ocr": {
    "tesseract": {"cer": 0.437, "wer": 0.612, "chars": 1489, "time_ms": 340},
    "rapidocr": {"cer": 0.387, "wer": 0.532, "chars": 1501, "time_ms": 220},
    "easyocr": {"cer": 0.524, "wer": 0.667, "chars": 1456, "time_ms": 890},
    "gcloud_vision": {"cer": 0.284, "wer": 0.389, "chars": 1520, "time_ms": 450},
    "paddleocr": {"cer": 0.189, "wer": 0.312, "chars": 1510, "time_ms": 180},
    "doctr": {"cer": 0.187, "wer": 0.298, "chars": 1505, "time_ms": 260},
    "kraken": {"cer": 0.880, "wer": 0.945, "chars": 890, "time_ms": 1200},
    "glm-ocr": {"cer": 0.257, "wer": 0.378, "chars": 1495, "time_ms": 3200},
    "deepseek-ocr2": {"cer": 0.594, "wer": 0.712, "chars": 2100, "time_ms": 4500}
  }
}
```

##### 11.4 Verification Checklist

| Check | Expected | Status |
|-------|----------|--------|
| PRISTINE tier has lowest CER | Yes | ✅ Confirmed (identical to ORIGINAL) |
| DEGRADED tier has highest CER | Yes | ✅ Confirmed (with LOW/DEGRADED convergence) |
| MOS decreases monotonically across tiers | Yes | ✅ Confirmed (minor LOW/DEGRADED overlap) |
| Correlations significant (p < 0.05) | Yes | ✅ All p < 10^-33 |
| All engines show consistent correlation direction | Yes | ✅ All 9 engines show negative SRCC |

##### 11.5 Test Coverage

19 unit tests passing:

- `tests/test_funsd_parser.py` (3 tests): annotation parsing, ID sorting, empty text filtering
- `tests/test_cer_wer.py` (16 tests): text normalization, CER, WER, combined metrics

##### 11.6 Pending Models

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| Surya | Neural OCR | Staged for Modal | GPU recommended |
| MinerU2.5 (1.2B) | VLM OCR | Pending | Modal deployment |
| Adobe PDF Extract | Cloud API | Pending | Needs credentials |
| TrOCR | Line-level | Deferred | Needs detection wrapper |
