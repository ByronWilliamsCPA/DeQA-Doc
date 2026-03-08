### OCR-IQA Correlation Dataset

> **Quick Stats**: 1,200 images | Controlled distortions | 4 OCR engines x 5 quality tiers | Paired analysis
>
> **License**: Research (derived from FUNSD + FUNSD+) | **Commercial Use**: Restricted

#### 1. Overview

| Attribute | Value |
|-----------|-------|
| **Full Name** | OCR-IQA Correlation Test Dataset |
| **Version** | 1.0 (in progress) |
| **Release Date** | 2026 |
| **Maintainer** | DeQA-Doc project |
| **Paper** | Internal research (no publication) |
| **License** | Research use (inherits FUNSD/FUNSD+ licenses) |
| **Documentation Status** | Partial (pipeline built, data generation pending) |

This dataset tests the hypothesis: **do DeQA image quality scores predict OCR accuracy on document images?** By applying controlled distortions to 200 documents with known ground-truth text, then measuring both OCR error rates (CER/WER) and DeQA quality scores, it quantifies the correlation between predicted quality and actual downstream OCR performance.

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
| **Distorted Images** | PNG | 1,000 distorted versions (5 tiers x 200) |
| **Ground Truth Text** | TXT | Per-image text files from FUNSD/FUNSD+ annotations |
| **OCR Results** | JSONL | Per-engine OCR output for all 1,200 images |
| **DeQA Scores** | JSONL | MOS + probability distribution per image |
| **Sample Manifest** | JSON | Image selection with provenance metadata |
| **Master Record** | JSONL | Unified dataset with all measurements |

##### 2.3 Dataset Structure

| Component | Path | Count | Status |
|-----------|------|-------|--------|
| **Base images** | `data/base_images/` | 200 | ✅ Generated |
| **GT text** | `data/gt_text/` | 200 | ✅ Generated |
| **Manifest** | `data/sample_manifest.json` | 1 | ✅ Generated |
| **ORIGINAL tier** | `data/distorted/ORIGINAL/` | 200 | Pending |
| **PRISTINE tier** | `data/distorted/PRISTINE/` | 200 | Pending |
| **HIGH tier** | `data/distorted/HIGH/` | 200 | Pending |
| **MEDIUM tier** | `data/distorted/MEDIUM/` | 200 | Pending |
| **LOW tier** | `data/distorted/LOW/` | 200 | Pending |
| **DEGRADED tier** | `data/distorted/DEGRADED/` | 200 | Pending |
| **OCR results** | `data/ocr_results/` | 4 files | Pending |
| **DeQA scores** | `data/deqa_results/` | 1 file | Pending |
| **Master record** | `data/dataset.jsonl` | 1 file | Pending |

##### 2.4 Labels & Annotations

| Label Type | Format | Granularity | Description |
|------------|--------|-------------|-------------|
| **Ground Truth Text** | TXT | Page-level | Human-validated text from FUNSD/FUNSD+ annotations |
| **OCR Transcriptions** | JSONL | Page-level | OCR output per engine per image |
| **CER/WER Metrics** | JSONL | Image-level | Character/word error rates vs GT |
| **DeQA Quality Scores** | JSONL | Image-level | MOS + 5-level probability distribution |
| **Distortion Parameters** | JSONL | Image-level | Seed, tier, IQALabels from hybrid pipeline |

##### 2.5 Ground Truth Provenance

| Aspect | Details |
|--------|---------|
| **Text Annotation Method** | Human Expert (FUNSD) + Human Expert (FUNSD+) |
| **Text Provenance Tier** | Tier 1 (Human-labeled) — form entity annotations |
| **Quality Score Method** | Model-predicted (DeQA mPLUG-Owl2) |
| **Quality Provenance Tier** | Tier 2 (Model) — fine-tuned VLM inference |
| **Distortion Labels** | Tier 0 (Exact) — deterministic pipeline with known parameters |
| **GT Label Coverage** | 100% text GT for base images; quality scores pending |

#### 3. Project Usage

| Aspect | Details |
|--------|---------|
| **Purpose** | Correlation analysis: OCR accuracy vs DeQA quality scores |
| **Local Path** | `research/ocr_iqa_correlation/data/` |
| **Pipeline Code** | `research/ocr_iqa_correlation/` |
| **Preprocessing** | 5-step pipeline (extract GT -> distort -> OCR -> DeQA -> analyze) |

**Research questions**:

1. Does DeQA MOS correlate with OCR accuracy (CER/WER)? Expected: negative SRCC (higher quality -> lower error)
2. Is the correlation consistent across OCR engines?
3. Do paired deltas (DCER vs DMOS) show stronger correlation than absolute values?
4. Do adjacent quality tiers produce statistically significant CER differences?

#### 4. Dataset Statistics

##### 4.1 Sample Counts

| Metric | Value |
|--------|-------|
| **Base Images** | 200 |
| **Total Images** | 1,200 (200 base x 6 tiers including ORIGINAL) |
| **FUNSD Images** | 50 (25% of base) |
| **FUNSD+ Images** | 150 (75% of base) |
| **Quality Tiers** | 6 (ORIGINAL + 5 distortion levels) |
| **OCR Engines** | 4 |
| **Total OCR Runs** | 4,800 (1,200 images x 4 engines) |
| **Total Size on Disk** | ~35.8 MB (base images only; full dataset TBD) |

##### 4.2 Text Statistics (from GT extraction)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| **Character Count** | 181 | 2,999 | 1,001 |

**Text Source**: `ground_truth` (human-validated FUNSD/FUNSD+ annotations)

##### 4.3 Directory Structure

```text
research/ocr_iqa_correlation/
+-- data/                        # gitignored
|   +-- base_images/             # 200 base document images
|   +-- gt_text/                 # 200 per-image GT text files
|   +-- sample_manifest.json     # Selection metadata
|   +-- distorted/               # 6 tier subdirectories
|   |   +-- ORIGINAL/
|   |   +-- PRISTINE/
|   |   +-- HIGH/
|   |   +-- MEDIUM/
|   |   +-- LOW/
|   |   +-- DEGRADED/
|   +-- ocr_results/             # Per-engine JSONL
|   +-- deqa_results/            # DeQA scores JSONL
|   +-- dataset.jsonl            # Master record
+-- outputs/                     # Analysis results (plots, tables)
```

#### 5. Quality Profile

##### 5.1 DeQA Quality Dimensions

| Dimension | Available | Score Range | Notes |
|-----------|-----------|-------------|-------|
| **Overall Quality** | Pending | 1-5 MOS | Primary analysis dimension |
| **Sharpness** | Pending | 1-5 MOS | Follow-up analysis |
| **Color Fidelity** | Pending | 1-5 MOS | Follow-up analysis |

##### 5.2 Quality Tier Design

| Tier | HybridProfile | Target Overall Quality | Expected CER Impact |
|------|---------------|------------------------|---------------------|
| **ORIGINAL** | (none) | Varies (source quality) | Baseline |
| **PRISTINE** | PRISTINE | 0.95 - 1.00 | Minimal |
| **HIGH** | LIGHT | 0.80 - 0.95 | Low |
| **MEDIUM** | MODERATE | 0.60 - 0.80 | Moderate |
| **LOW** | HEAVY | 0.40 - 0.60 | High |
| **DEGRADED** | HISTORICAL | 0.00 - 0.40 | Severe |

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

| Engine | Backend | Notes |
|--------|---------|-------|
| **Tesseract** | Docling (`TesseractOcrOptions`) | Open-source, widely used baseline |
| **RapidOCR** | Docling (default) | Docling's default OCR engine |
| **EasyOCR** | Docling (`EasyOcrOptions`) | PyTorch-based multilingual OCR |
| **Google Cloud Vision** | Cloud API (`DOCUMENT_TEXT_DETECTION`) | Commercial cloud OCR |

##### 6.2 Metrics

| Metric | Computation | Library |
|--------|-------------|---------|
| **CER** | Character Error Rate (Levenshtein distance / reference length) | jiwer |
| **WER** | Word Error Rate (word-level edit distance / reference words) | jiwer |

Text normalization: NFC unicode normalization, lowercase, whitespace collapse, strip.

Empty OCR output -> CER = 1.0, WER = 1.0.

##### 6.3 Expected Results (pending data generation)

| Analysis | Method | Expected Finding |
|----------|--------|------------------|
| **Per-engine SRCC** | SRCC(CER, MOS) | Negative correlation (higher quality -> lower CER) |
| **Per-engine PLCC** | PLCC(CER, MOS) | Negative linear correlation |
| **Paired deltas** | SRCC(DCER, DMOS) | Stronger correlation than absolute values |
| **Tier significance** | Wilcoxon signed-rank | Significant CER differences between adjacent tiers |
| **Cross-engine agreement** | Kendall's W | Concordance across 4 engines |

#### 7. Model Evaluation Results

Pending — requires Phase 4 (DeQA scoring on GPU).

#### 8. Known Issues & Limitations

- **FUNSD/FUNSD+ only**: Limited to form-style documents; does not cover prose, tables, or handwritten content
- **English-only GT**: Both source datasets are English, limiting script diversity
- **GT text is entity-level**: FUNSD annotations are form entities concatenated by ID order, not natural reading order
- **Model-predicted quality**: DeQA scores are model predictions, not human MOS — correlation tests model consistency, not ground truth alignment
- **Single distortion pipeline**: All distortions from one pipeline (HybridAugmentationPipeline); results may not generalize to other degradation sources
- **No GPU phases yet**: Steps 02 (distortion, needs augraphy) and 04 (DeQA, needs GPU) not yet executed

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

#### 11. Dataset-Specific Notes

##### 11.1 Pipeline Phases

| Phase | Script | Status | Description |
|-------|--------|--------|-------------|
| 1 | `scripts/01_extract_gt_and_sample.py` | ✅ Complete | Extract GT text, sample 200 images, copy locally |
| 2 | `scripts/02_apply_distortions.py` | Pending | Apply 5 distortion tiers via HybridAugmentationPipeline |
| 3 | `scripts/03_run_ocr.py` | Pending | Run 4 OCR engines on 1,200 images |
| 4 | `scripts/04_run_deqa.py` | Pending (GPU) | Run DeQA mPLUG-Owl2 scorer on 1,200 images |
| 5 | `scripts/05_analyze.py` | Pending | Compute correlations, paired analysis, visualizations |

##### 11.2 Two-Venv Strategy

- **Main venv** (`research/ocr_iqa_correlation/.venv`): Phases 1-3, 5. Modern Python with docling, jiwer, augraphy, albumentations
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
  "deqa_probs": [0.05, 0.35, 0.40, 0.15, 0.05],
  "ocr": {
    "tesseract": {"cer": 0.082, "wer": 0.145, "chars": 1489, "time_ms": 340},
    "rapidocr": {"cer": 0.075, "wer": 0.132, "chars": 1501, "time_ms": 220},
    "easyocr": {"cer": 0.098, "wer": 0.167, "chars": 1456, "time_ms": 890},
    "gcloud_vision": {"cer": 0.045, "wer": 0.089, "chars": 1520, "time_ms": 450}
  }
}
```

##### 11.4 Verification Checklist

| Check | Expected | Status |
|-------|----------|--------|
| PRISTINE tier has lowest CER | Yes | Pending |
| DEGRADED tier has highest CER | Yes | Pending |
| MOS decreases monotonically across tiers | Yes | Pending |
| Correlations significant (p < 0.05) | Yes (with N=200) | Pending |
| All 4 engines show consistent correlation direction | Yes | Pending |

##### 11.5 Test Coverage

19 unit tests passing:

- `tests/test_funsd_parser.py` (3 tests): annotation parsing, ID sorting, empty text filtering
- `tests/test_cer_wer.py` (16 tests): text normalization, CER, WER, combined metrics
