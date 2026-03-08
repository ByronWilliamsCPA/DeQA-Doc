### Synthetic OOD Test Set (synth-ood-520)

> **Quick Stats**: 520 images | Synthetic | 15 categories (2 ID + 13 OOD) | OOD robustness evaluation
>
> **License**: Internal research | **Commercial Use**: N/A (synthetic, not distributed)

#### 1. Overview

| Attribute | Value |
|-----------|-------|
| **Full Name** | Synthetic OOD Test Set for DIQA Model Robustness |
| **Version** | 1.0 |
| **Release Date** | 2026 |
| **Maintainer** | DeQA-Doc project |
| **Paper** | Internal research (no publication) |
| **License** | Internal research use |
| **Documentation Status** | Complete |

This dataset systematically evaluates DIQA model robustness across 15 document categories spanning scripts, layouts, degradation extremes, and resolution variations. It was designed to identify failure modes in VLM-based quality scoring by providing controlled OOD scenarios that the DIQA-5000 training distribution does not cover.

#### 2. Source Data Inventory

##### 2.1 Provided File Types

| File Type | Format(s) | Description |
|-----------|-----------|-------------|
| **Images** | JPG | 520 synthetically generated document images |
| **Metadata** | JSONL | Per-image generation parameters and synthetic scores |
| **Manifests** | JSON | Category definitions and image path listings |

##### 2.2 Dataset Splits

| Split | Path | Count | Status |
|-------|------|-------|--------|
| **All (no splits)** | `/tmp/ood_poc_test/` | 520 | ✅ |

No train/val/test splits — this is evaluation-only.

##### 2.3 Labels & Annotations

| Label Type | Format | Granularity | Description |
|------------|--------|-------------|-------------|
| **Synthetic Quality Scores** | JSONL | Image-level | 3-dimensional synthetic MOS: overall, sharpness, color |
| **Category Labels** | JSONL | Image-level | ID/OOD classification with reason |
| **Generation Parameters** | JSONL | Image-level | Script, DPI, quality tier, color mode, font, direction |

##### 2.4 Ground Truth Provenance

| Aspect | Details |
|--------|---------|
| **Annotation Method** | Synthetic (algorithmically assigned based on generation parameters) |
| **Provenance Tier** | Tier 3 (Heuristic — synthetic scores from distortion parameters, not human ratings) |
| **Quality Assurance** | Manual verification of category correctness; no human MOS validation |
| **GT Label Coverage** | 100% (all 520 images have synthetic scores and category labels) |

#### 3. Project Usage

| Aspect | Details |
|--------|---------|
| **Purpose** | OOD robustness evaluation for VLM and fine-tuned IQA models |
| **Local Path** | `/tmp/ood_poc_test/` (generated on-demand) |
| **Subset Used** | Full dataset |
| **Preprocessing** | None required |

**Used in**:
- VLM teacher evaluation ([results/vlm_teacher_eval/](../../results/vlm_teacher_eval/)) — tests cross-domain generalization of 7 VLMs
- Fine-tuned model benchmarking ([modal/benchmark_synthetic_ood.py](../../modal/benchmark_synthetic_ood.py)) — benchmarks SigLIP2-IQA, HyperIQA++, DeQA-Doc specialists
- OOD detector validation ([results/tier1_ood_detector/](../../results/tier1_ood_detector/)) — validates Mahalanobis-distance detector (AUROC 0.9963)

#### 4. Dataset Statistics

##### 4.1 Sample Counts

| Metric | Value |
|--------|-------|
| **Total Images** | 520 |
| **In-Distribution (ID)** | 150 (28.8%) |
| **Out-of-Distribution (OOD)** | 370 (71.2%) |
| **File Format(s)** | JPEG |

##### 4.2 Category Breakdown

**In-Distribution (150 images)**:

| Category | Count | Description | Mean Overall |
|----------|-------|-------------|--------------|
| `id_standard` | 100 | Latin-script documents matching DIQA-5000 | 3.52 |
| `id_cyrillic` | 50 | Cyrillic script (in DIQA-5000 distribution) | 3.53 |

**Out-of-Distribution (370 images)**:

| Category | Count | Description | Mean Overall |
|----------|-------|-------------|--------------|
| `ood_script_tibetan` | 30 | Tibetan script | 3.51 |
| `ood_script_myanmar` | 30 | Myanmar script | 3.47 |
| `ood_script_ethiopic` | 30 | Ethiopic script | 3.51 |
| `ood_adversarial_fraktur` | 20 | Latin Blackletter/Fraktur | 3.44 |
| `ood_adversarial_nastaliq` | 20 | Arabic Nastaliq/calligraphic | 3.50 |
| `ood_cjk_vertical` | 30 | CJK vertical text (tategaki) | 3.42 |
| `ood_multiscript` | 30 | Mixed-script (Arabic + Latin) | 3.53 |
| `ood_form_layout` | 30 | Structured form layouts | 3.02 |
| `ood_heavily_degraded` | 30 | Beyond-DIQA-5000 degradation | 1.50 |
| `ood_binarized` | 30 | Black-and-white binarized | 2.97 |
| `ood_very_low_dpi` | 30 | 72 DPI thumbnails | 1.94 |
| `ood_very_high_dpi` | 30 | 600 DPI archival scans | 3.94 |
| `ood_pristine` | 30 | Near-perfect digital documents | 4.51 |

##### 4.3 Directory Structure

```text
ood_poc_test/
+-- metadata.jsonl              # 520 records with scores and params
+-- manifest.json               # Category definitions
+-- id_standard/                # 100 images
+-- id_cyrillic/                # 50 images
+-- ood_script_tibetan/         # 30 images
+-- ood_script_myanmar/         # 30 images
+-- ood_script_ethiopic/        # 30 images
+-- ood_adversarial_fraktur/    # 20 images
+-- ood_adversarial_nastaliq/   # 20 images
+-- ood_cjk_vertical/           # 30 images
+-- ood_multiscript/            # 30 images
+-- ood_form_layout/            # 30 images
+-- ood_heavily_degraded/       # 30 images
+-- ood_binarized/              # 30 images
+-- ood_very_low_dpi/           # 30 images
+-- ood_very_high_dpi/          # 30 images
+-- ood_pristine/               # 30 images
```

#### 5. Quality Profile

##### 5.1 DeQA Quality Dimensions

| Dimension | Available | Score Range | Notes |
|-----------|-----------|-------------|-------|
| **Overall Quality** | Yes | 1.50 - 4.51 (synthetic) | Algorithmically assigned |
| **Sharpness** | Yes | 1.20 - 4.84 (synthetic) | Algorithmically assigned |
| **Color Fidelity** | Yes | 1.49 - 4.45 (synthetic) | Algorithmically assigned |

Scores are synthetic (not human-rated) — derived from generation parameters, not subjective MOS.

##### 5.2 Quality Score Distribution

| Dimension | Mean | Min | Max |
|-----------|------|-----|-----|
| **Overall** | ~3.37 | 1.50 | 4.51 |
| **Sharpness** | ~3.41 | 1.20 | 4.84 |
| **Color Fidelity** | ~3.33 | 1.49 | 4.45 |

##### 5.3 Degradation Coverage

| Degradation Type | Present | Notes |
|------------------|---------|-------|
| **Blur** | Yes | Across quality tiers (PRISTINE through DEGRADED) |
| **Noise** | Yes | Across quality tiers |
| **Compression** | Yes | JPEG quality variation |
| **Binarization** | Yes | `ood_binarized` category |
| **DPI Extremes** | Yes | 72 DPI (`ood_very_low_dpi`) to 600 DPI (`ood_very_high_dpi`) |
| **Heavy Degradation** | Yes | `ood_heavily_degraded` (beyond DIQA-5000 range) |

#### 6. OCR Evaluation

No OCR evaluation — this dataset has no ground truth text transcriptions.

#### 7. Model Evaluation Results

##### VLM Teacher Evaluation (7 models)

Key findings from cross-domain evaluation:

**Universal failure modes** (all VLMs):

| Category | SRCC Range | Issue |
|----------|------------|-------|
| `ood_binarized` | -0.340 to -0.372 | Negative correlation — VLMs rate B&W incorrectly |
| `ood_very_low_dpi` | -0.150 to -0.411 | Negative correlation |
| `ood_pristine` | 0.032 to -0.086 | Near-zero — no discriminative ability |
| `ood_form_layout` | 0.169 to 0.201 | Very weak correlation |

**Successful transfer**:

| Category | SRCC Range | Notes |
|----------|------------|-------|
| Non-Latin scripts | 0.73 - 0.85 | Tibetan, Myanmar, Ethiopic transfer well |
| Adversarial scripts | 0.70 - 0.85 | Fraktur, Nastaliq handled adequately |
| ID (standard/Cyrillic) | 0.79 - 0.81 | Baseline performance |

##### OOD Detector Performance

| Metric | Value |
|--------|-------|
| **AUROC** | 0.9963 |
| **Method** | Mahalanobis distance on SigLIP2 embeddings |
| **Threshold** | 46.0 (test p95) |

#### 8. Known Issues & Limitations

- **Synthetic scores, not human MOS**: Quality labels are algorithmically derived, not validated by human raters
- **Generated on-demand**: Dataset lives in `/tmp/` and must be regenerated — not persisted
- **No text ground truth**: Synthetically rendered text is not stored as GT for OCR evaluation
- **Category balance**: Uneven counts (20-100 per category) limit statistical power for smaller groups
- **Script rendering fidelity**: Font availability affects visual quality of non-Latin scripts

#### 9. Content Composition

| Aspect | Details |
|--------|---------|
| **Domain** | Synthetic document images across multiple domains |
| **Language(s)** | Multilingual: Latin, Cyrillic, Tibetan, Myanmar, Ethiopic, Arabic, CJK |
| **Script(s)** | 7+ script families |
| **Document Types** | Prose, forms, vertical text, mixed-script |

##### 9.1 Generation Parameters

| Parameter | Values |
|-----------|--------|
| **Script** | Latn, Cyrl, Tibt, Mymr, Ethi, Arab, CJK |
| **DPI** | 72, 200, 300, 600 |
| **Quality Tier** | PRISTINE, HIGH, MEDIUM, LOW, DEGRADED |
| **Color Mode** | color, grayscale, binarized |
| **Writing Direction** | ltr, rtl, ttb |
| **Font Families** | DejaVuSans, NotoSans, Fraktur, Nastaliq, others |

#### 10. References

##### Related Datasets

- [diqa-5000](diqa-5000.md) - The benchmark this OOD set tests robustness against
- [ocr-iqa-correlation](ocr-iqa-correlation.md) - OCR-based quality validation (different methodology)

#### 11. Dataset-Specific Notes

##### 11.1 Per-Image Metadata Schema

Each record in `metadata.jsonl`:

```json
{
  "image_id": "id_standard/id_standard_0000.jpg",
  "image_path": "/tmp/ood_poc_test/id_standard/id_standard_0000.jpg",
  "category": "id_standard",
  "is_ood": false,
  "ood_reason": null,
  "generation_params": {
    "script": "Latn",
    "dpi": 200,
    "quality_tier": "MEDIUM",
    "color_mode": "color",
    "font_family": "DejaVuSans",
    "writing_direction": "ltr"
  },
  "synthetic_scores": {
    "overall": 2.83,
    "sharpness": 3.19,
    "color": 2.66
  },
  "synthetic_categories": {
    "overall": "fair",
    "sharpness": "fair",
    "color": "fair"
  }
}
```

##### 11.2 Categorical Score Buckets

| Category | MOS Range |
|----------|-----------|
| bad | [1.0, 1.8) |
| poor | [1.8, 2.6) |
| fair | [2.6, 3.4) |
| good | [3.4, 4.0) |
| excellent | [4.0, 5.0] |

##### 11.3 Regeneration

This dataset is generated on-demand and not version-controlled. To regenerate:

```bash
# VLM evaluation pipeline
python results/vlm_teacher_eval/full_eval/run_synthetic_eval.py

# Fine-tuned model benchmarking
python modal/benchmark_synthetic_ood.py
```
