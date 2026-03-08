# Dataset Documentation Template

> **Version**: 1.0.0
> **Last Updated**: 2026-03-07
> **Purpose**: Standardized template for documenting datasets used in DeQA-Doc research
> **Adapted from**: image_detection project template v1.6.0

---

## Quick Reference Format (for DATASET_CATALOG.md)

```markdown
### [Dataset Name]

> **Quick Stats**: [count] images | [source_type] | [primary characteristics]
>
> **License**: [license] | **Commercial Use**: Yes/No/Restricted

- **Path**: `path/to/dataset/`
- **Paper**: [Title (Year)](link)
- **Quality Dimensions**: overall / sharpness / color_fidelity
- **Usage**: Training / Benchmark / Evaluation

[2-3 sentence description of dataset and its role in DeQA-Doc.]
```

---

## Detailed Dataset Card Template

For individual dataset files (`docs/datasets/[dataset_name].md`):

### [Dataset Name]

> **Quick Stats**: N images | Source type | Key characteristics

#### 1. Overview

| Attribute | Value |
|-----------|-------|
| **Full Name** | Complete official dataset name |
| **Version** | Version number |
| **Release Date** | YYYY-MM-DD |
| **Maintainer** | Organization or authors |
| **Paper** | [Citation Title (Year)](paper_url) |
| **Repository** | [Official Source](repo_url) |
| **License** | License type |
| **Commercial Use** | Yes / No / Restricted |
| **Documentation Status** | Complete / Partial / Inferred |

#### 2. Source Data Inventory

##### 2.1 Provided File Types

| File Type | Format(s) | Description |
|-----------|-----------|-------------|
| **Images** | JPG / PNG / TIFF | Document images |
| **Annotations** | JSON / CSV / Arrow | Labels and scores |
| **Metadata** | JSON / JSONL / CSV | Per-image or dataset-level metadata |

##### 2.2 Dataset Splits

| Split | Path | Count | Status |
|-------|------|-------|--------|
| **Train** | `dataset/train/` | N | status |
| **Validation** | `dataset/val/` | N | status |
| **Test** | `dataset/test/` | N | status |

##### 2.3 Labels & Annotations

| Label Type | Format | Granularity | Description |
|------------|--------|-------------|-------------|
| **Quality Scores** | CSV / JSON | Image-level | MOS or predicted quality |
| **Text Transcriptions** | TXT / JSON | Page-level | Ground truth or OCR output |
| **Distortion Parameters** | JSON | Image-level | Applied degradation metadata |

> Delete rows that don't apply.

##### 2.4 Ground Truth Provenance

| Aspect | Details |
|--------|---------|
| **Annotation Method** | Human Expert / Crowdsourced / Synthetic / Model-predicted |
| **Provenance Tier** | Tier 0 (Exact) / Tier 1 (Human) / Tier 2 (Model) / Tier 3 (Heuristic) |
| **Annotator Details** | Number of annotators, expertise level |
| **Inter-Annotator Agreement** | IAA metric and value |
| **Quality Assurance** | QA process description |
| **GT Label Coverage** | Percentage of images with ground truth |

#### 3. Project Usage

| Aspect | Details |
|--------|---------|
| **Purpose** | Training / Benchmark / Evaluation / OOD Testing |
| **Local Path** | `path/to/dataset/` |
| **Subset Used** | Full / Specific subset |
| **Preprocessing** | Required steps before use |

#### 4. Dataset Statistics

##### 4.1 Sample Counts

| Metric | Value |
|--------|-------|
| **Total Images** | N |
| **Training Split** | N (%) |
| **Validation Split** | N (%) |
| **Test Split** | N (%) |
| **Image Dimensions** | WxH range |
| **File Format(s)** | JPG / PNG |
| **Total Size on Disk** | X GB |

##### 4.2 Text Statistics (if GT text available)

| Metric | Mean +/- Std | Min | Max |
|--------|--------------|-----|-----|
| **Character Count** | | | |
| **Word Count** | | | |

**Text Source**: `ground_truth` / `ocr_extracted` / `synthetic`

##### 4.3 Directory Structure

```text
dataset/
+-- split_a/
+-- split_b/
```

#### 5. Quality Profile

##### 5.1 DeQA Quality Dimensions

| Dimension | Available | Score Range | Notes |
|-----------|-----------|-------------|-------|
| **Overall Quality** | Yes/No | 1-5 MOS | Holistic readability |
| **Sharpness** | Yes/No | 1-5 MOS | Text clarity and edge definition |
| **Color Fidelity** | Yes/No | 1-5 MOS | Color accuracy and contrast |

##### 5.2 Quality Score Distribution

| Dimension | Mean +/- Std | Min | Max | P25 / P50 / P75 |
|-----------|--------------|-----|-----|------------------|
| **Overall** | | | | |
| **Sharpness** | | | | |
| **Color Fidelity** | | | | |

##### 5.3 Source Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Source Type** | Born-digital / Scanned / Camera-captured / Synthetic |
| **Capture Device** | Scanner model / Camera / N/A |
| **Original Quality** | Clean / Mixed / Degraded |
| **Known Artifacts** | List of common artifacts |

##### 5.4 Degradation Coverage

| Degradation Type | Present | Severity Range | Notes |
|------------------|---------|----------------|-------|
| **Blur** | Yes/No | Low-High | |
| **Noise** | Yes/No | Low-High | |
| **Compression** | Yes/No | Low-High | |
| **Ink Degradation** | Yes/No | Low-High | |
| **Paper Aging** | Yes/No | Low-High | |
| **Geometric Distortion** | Yes/No | Low-High | |
| **Bleed-Through** | Yes/No | Low-High | |

> Delete rows that don't apply. Add custom rows for dataset-specific degradation types.

#### 6. OCR Evaluation (if applicable)

> Delete this section if no OCR evaluation was performed on this dataset.

##### 6.1 OCR Engines Tested

| Engine | Version | Backend | Notes |
|--------|---------|---------|-------|
| Tesseract | X.Y | Docling | |
| RapidOCR | X.Y | Docling | |
| EasyOCR | X.Y | Docling | |
| Google Vision | API | Cloud | |

##### 6.2 OCR Accuracy Summary

| Engine | Mean CER | Mean WER | Notes |
|--------|----------|----------|-------|
| Tesseract | | | |
| RapidOCR | | | |

##### 6.3 OCR-IQA Correlation

| Engine | SRCC (CER vs MOS) | PLCC (CER vs MOS) | p-value |
|--------|--------------------|--------------------|---------|
| Tesseract | | | |
| RapidOCR | | | |

#### 7. Model Evaluation Results (if applicable)

> Delete this section if no DeQA model evaluation was performed.

| Model | SRCC (avg) | PLCC (avg) | Notes |
|-------|------------|------------|-------|
| DeQA mPLUG-Owl2 | | | |
| DeQA Qwen2.5-VL | | | |

#### 8. Known Issues & Limitations

- **Issue 1**: Description
- **Issue 2**: Description

#### 9. Content Composition

| Aspect | Details |
|--------|---------|
| **Domain** | Document types covered |
| **Language(s)** | Primary languages |
| **Script(s)** | Writing systems |
| **Document Types** | Forms / Reports / etc. |

##### 9.1 Category Distribution (if applicable)

| Category | Count | Percentage |
|----------|-------|------------|
| Category A | N | X% |
| Category B | N | X% |

#### 10. References

##### Primary Citation

```bibtex
@article{...,
  title={...},
  author={...},
  year={...}
}
```

##### Related Datasets

- [Dataset A](dataset_a.md) - Relationship description

#### 11. Dataset-Specific Notes

> Capture unique characteristics, caveats, and implementation details.
> Delete subsections that don't apply.

##### 11.1 Implementation Notes

- Notes on parsing, schema quirks, file format details

##### 11.2 Custom Metrics

- Dataset-specific scoring, tier definitions, conversion tables

---

## Documentation Status Markers

| Marker | Meaning |
|--------|---------|
| `[Official]` | From official documentation/paper |
| `[Empirically Derived]` | Computed from actual samples |
| `[Inferred]` | Reasoned from available evidence |
| `[NEEDS_PROFILING]` | Requires empirical analysis |
| `[NEEDS_VERIFICATION]` | Needs confirmation |

---

## Template Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-07 | Initial template adapted from image_detection v1.6.0. Removed Layer 2 audit, training head coverage, reliability & bottleneck sections. Added DeQA quality dimensions (Section 5), OCR evaluation (Section 6), and model evaluation (Section 7) sections. |
