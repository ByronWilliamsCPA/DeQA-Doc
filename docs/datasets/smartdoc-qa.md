### SmartDoc-QA

> **Quick Stats**: 4,260 images | Camera-captured (robotic arm) | 30 documents x 142 distortion variants | OCR accuracy GT
>
> **License**: Research | **Commercial Use**: No

#### 1. Overview

| Attribute | Value |
|-----------|-------|
| **Full Name** | SmartDoc Quality Assessment Dataset |
| **Version** | 1.0 (CBDAR@ICDAR 2015) |
| **Release Date** | 2015 |
| **Maintainer** | L3i Lab, Universite de La Rochelle |
| **Paper** | [Nayef et al. 2015](https://ieeexplore.ieee.org/document/7333960/) |
| **Repository** | [smartdoc.univ-lr.fr](http://smartdoc.univ-lr.fr/smartdoc-qa/) |
| **License** | Research Only |
| **Commercial Use** | No |
| **Documentation Status** | Complete |

SmartDoc-QA is a controlled benchmark for document image quality assessment via OCR accuracy as an objective quality proxy. 30 paper documents (3 categories) were captured using a Fanuc LR Mate 200iD robotic arm under systematically varied conditions (lighting, perspective, blur), producing 4,260 images with pre-computed Tesseract and FineReader OCR accuracy per image plus human-keyed ground truth text. In DeQA-Doc, it serves dual purpose: VLM-labeled training data (train split) and three-way correlation evaluation (VLM vs OCR accuracy vs DeQA predictions).

#### 2. Source Data Inventory

##### 2.1 Provided File Types

| File Type | Format(s) | Description |
|-----------|-----------|-------------|
| **Captured Images** | JPG | 4,260 camera-captured document images (2 phones x 30 docs x 71 conditions) |
| **Original Documents** | PDF / TIF / JPG | 30 source documents (10 modern PDF, 10 admin TIF, 10 receipt JPG) |
| **Ground Truth Text** | TXT | 30 human-keyed transcriptions (one per document) |
| **OCR Results** | TXT | Tesseract + FineReader output per captured image |
| **OCR Accuracy** | TXT (UNLV-ISRI) | Character accuracy (`.cacc`) and word accuracy (`.wacc`) per image per engine |

##### 2.2 Dataset Splits

Document-level stratified split (seed=42), ensuring each category is represented in every split. No document appears in multiple splits.

| Split | Docs | Images | Categories | Status |
|-------|------|--------|------------|--------|
| **Train** | 21 | 2,982 (70%) | Modern: D3,4,6-10 / Admin: D12-16,18,19 / Receipts: D21-23,25,26,28,30 | Active |
| **Validation** | 3 | 426 (10%) | Modern: D5 / Admin: D11 / Receipts: D27 | Active |
| **Test** | 6 | 852 (20%) | Modern: D1,2 / Admin: D17,20 / Receipts: D24,29 | Active |

**Split Manifest**: `/mnt/e/image_detection/02_benchmark_only/smartdoc-qa/splits/smartdoc_qa_splits.json`

##### 2.3 Labels & Annotations

| Label Type | Format | Granularity | Description |
|------------|--------|-------------|-------------|
| **Ground Truth Text** | TXT | Document-level | Human-keyed transcriptions for all 30 documents |
| **OCR Accuracy (FineReader)** | UNLV-ISRI TXT | Image-level | Character and word accuracy vs GT per image |
| **OCR Accuracy (Tesseract)** | UNLV-ISRI TXT | Image-level | Character and word accuracy vs GT per image |
| **OCR Transcriptions** | TXT | Image-level | Raw Tesseract + FineReader output per image |
| **Capture Parameters** | Filename-encoded | Image-level | Document, lighting, angle, blur type/level (see 11.1) |
| **VLM Quality Labels** | JSON | Image-level | 2-model consensus MOS (Gemini Flash Lite + Qwen3-VL-235B) [in progress] |

##### 2.4 Ground Truth Provenance

| Aspect | Details |
|--------|---------|
| **Text Annotation Method** | Human Expert (manual transcription) |
| **Text Provenance Tier** | Tier 1 (Human) |
| **OCR Accuracy Method** | UNLV-ISRI accuracy tool (character/word level vs human GT) |
| **OCR Accuracy Provenance** | Tier 0 (Exact) — deterministic comparison against GT |
| **VLM Quality Labels** | Tier 2 (Model) — 2-model consensus via OpenRouter |
| **Capture Parameters** | Tier 0 (Exact) — robotic arm with known settings |
| **GT Label Coverage** | 100% text GT, 100% OCR accuracy, 100% VLM labels (pending) |

#### 3. Project Usage

| Aspect | Details |
|--------|---------|
| **Purpose** | Training (train split) + Evaluation (three-way correlation) |
| **Local Path** | `/mnt/e/image_detection/02_benchmark_only/smartdoc-qa/Dataset SmartDoc-QA/` |
| **Subset Used** | Full dataset VLM-labeled; train split (2,982 images) for DIQA-5000_1 Tier 1 training |
| **Preprocessing** | VLM consensus labeling via Stream 3 pipeline; OCR accuracy parsed from UNLV-ISRI format |

**Evaluation roles**:

1. **VLM vs OCR accuracy**: Does VLM quality prediction correlate with objective OCR char accuracy?
2. **DeQA vs OCR accuracy**: Does trained DeQA model predict OCR-impacting quality factors? (Same methodology as [OCR-IQA Correlation](ocr-iqa-correlation.md))
3. **DeQA vs VLM**: Consistency check between DeQA predictions and VLM consensus labels

#### 4. Dataset Statistics

##### 4.1 Sample Counts

| Metric | Value |
|--------|-------|
| **Total Images** | 4,260 |
| **Training Split** | 2,982 (70%) |
| **Validation Split** | 426 (10%) |
| **Test Split** | 852 (20%) |
| **Source Documents** | 30 (10 modern + 10 admin + 10 receipts) |
| **Captures per Document** | 142 (71 conditions x 2 phones) |
| **Image Dimensions** | Samsung: 4128x3096, Nokia: 3264x2448 |
| **File Format** | JPEG |
| **File Size Range** | 1.9 - 5.6 MB (avg 3.2 MB) |
| **Total Size on Disk** | 13.78 GB (Captured_Images only) |

##### 4.2 Text Statistics

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| **Character Count** | 1,888 | 291 | 5,517 |
| **Word Count** | 304 | 38 | 897 |

**Text Source**: `ground_truth` (human-keyed, one file per document shared across all captures)

##### 4.3 Directory Structure

```text
Dataset SmartDoc-QA/
+-- Original_documents/           # 30 source documents (PDF/TIF/JPG)
+-- Ground_truth/                 # 30 text transcriptions
|   +-- page_1.txt ... page_30.txt
+-- Captured_Images/
|   +-- Samsung_phone/            # Samsung Galaxy S4
|   |   +-- Images/               # 2,130 JPG captures
|   |   +-- Results_Tesseract/    # OCR output per image
|   |   +-- Results_Finereader/
|   |   +-- OCR_Accuracy_Tesseract/   # .cacc + .wacc per image
|   |   +-- OCR_Accuracy_Finereader/
|   +-- Nokia_phone/              # Nokia Lumia 920 (same structure)
+-- README.txt
+-- SmartDoc-QA_Dataset_License.pdf
```

#### 5. Quality Profile

##### 5.1 DeQA Quality Dimensions

| Dimension | Available | Score Range | Notes |
|-----------|-----------|-------------|-------|
| **Overall Quality** | Pending | 1-5 MOS | Via VLM consensus labeling (Stream 3) |
| **Sharpness** | No | — | Not assessed separately |
| **Color Fidelity** | No | — | Not assessed separately |

##### 5.2 OCR Accuracy as Quality Proxy

OCR character accuracy provides an objective, continuous quality signal (0-100%) that correlates with document readability.

| Engine | Mean +/- Std | Min | Max | P25 / P50 / P75 |
|--------|--------------|-----|-----|------------------|
| **FineReader** | 29.1 +/- 37.5 | 0.0 | 100.0 | 0.0 / 7.2 / 62.5 |
| **Tesseract** | 13.5 +/- 22.3 | 0.0 | 99.8 | 0.0 / 3.6 / 14.4 |

**Note**: Distribution is heavily skewed toward low accuracy (62.8% of images have < 20% FineReader accuracy). This reflects the dataset's emphasis on distorted captures — blur and out-of-focus conditions severely impact OCR.

##### 5.3 Source Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Source Type** | Camera-captured (controlled robotic arm) |
| **Capture Devices** | Samsung Galaxy S4 + Nokia Lumia 920 |
| **Capture Distance** | 35 cm from document |
| **Original Quality** | Mixed — systematically varies from clean to severely distorted |
| **Known Artifacts** | Perspective distortion, motion blur, defocus blur, uneven lighting, shadows, folds |

##### 5.4 Degradation Coverage

| Degradation Type | Present | Severity Range | Notes |
|------------------|---------|----------------|-------|
| **Perspective Distortion** | Yes | 0-10 degrees | Longitudinal (a) and lateral (b) angles |
| **Motion Blur** | Yes | Horizontal + 2D | Mb1 (horizontal), Mb2 (2D motion) |
| **Defocus Blur** | Yes | Multiple levels | Focus distance varied (Ob1-Ob4 Samsung, Ob820-Ob940 Nokia) |
| **Lighting Variation** | Yes | 5 conditions | L1: daylight, L2: +neon, L3: night+lamp, L4: +shadow object, L5: +grid shadow |
| **Folds** | Yes | Present in admin letters | Old documents from Tobacco corpus |

##### 5.5 Quality by Factor [Empirically Derived]

| Factor | Mean FineReader Accuracy | Impact |
|--------|--------------------------|--------|
| No blur | 44.9% | Baseline |
| With blur (Mb/Ob) | 22.9% | -22 pp |
| Daylight (L1) | 51.1% | Best |
| Shadow (L4) | 24.0% | Worst |
| Modern docs (D1-10) | 39.8% | Highest |
| Receipts (D21-30) | 15.7% | Lowest (small text, complex layout) |

#### 6. OCR Evaluation

##### 6.1 OCR Engines

| Engine | Version | Notes |
|--------|---------|-------|
| **ABBYY FineReader** | [Official] | Commercial OCR, higher accuracy baseline |
| **Tesseract** | [Official] | Open-source OCR, lower accuracy baseline |

##### 6.2 OCR Accuracy Summary

| Engine | Mean Char Acc | P50 | Images with > 80% | Images with < 20% |
|--------|--------------|-----|--------------------|--------------------|
| **FineReader** | 29.1% | 7.2% | 848 (19.9%) | 2,677 (62.8%) |
| **Tesseract** | 13.5% | 3.6% | 151 (3.5%) | 3,506 (82.3%) |

##### 6.3 OCR-IQA Correlation

Pending — requires Phase 4 (DeQA scoring on GPU). Will follow the same methodology as [OCR-IQA Correlation](ocr-iqa-correlation.md), with the advantage of having controlled distortion parameters enabling per-factor analysis.

#### 7. Model Evaluation Results

Pending — requires completion of Tier 1 training and DeQA inference on SmartDoc-QA test split.

| Model | SRCC (avg) | PLCC (avg) | Notes |
|-------|------------|------------|-------|
| DeQA mPLUG-Owl2 (DIQA-5000_0) | — | — | Pending |
| DeQA mPLUG-Owl2 (DIQA-5000_1) | — | — | Pending |

#### 8. Known Issues & Limitations

- **Content redundancy**: Only 30 unique documents, each captured 142 times. Models could overfit to document content rather than learning quality features. Mitigated by document-level splits (no content leakage between train/val/test).
- **Controlled environment**: Robotic arm capture does not fully represent handheld smartphone use. Natural hand tremor, varied distances, and spontaneous angles are not captured.
- **Latin script only**: All 30 documents are English/Latin. No script diversity.
- **2015 smartphone cameras**: Samsung Galaxy S4 and Nokia Lumia 920 have lower sensor quality than modern phones. Noise characteristics may not generalize.
- **Heavy low-quality skew**: 62.8% of images have < 20% FineReader accuracy. Quality distribution is imbalanced toward the low end.
- **OCR accuracy > 100% anomalies**: Some FineReader results show > 100% accuracy (max 152.1% raw). Capped at 100% for analysis. Likely due to text insertion errors in the UNLV-ISRI tool.

#### 9. Content Composition

| Aspect | Details |
|--------|---------|
| **Domain** | Modern documents, administrative correspondence, retail receipts |
| **Language(s)** | English (100%) |
| **Script(s)** | Latin (100%) |
| **Document Types** | SmartDoc competition papers, Tobacco corpus letters, shop receipts |

##### 9.1 Category Distribution

| Category | Documents | Images | Source | Train / Val / Test Docs |
|----------|-----------|--------|--------|-------------------------|
| **Modern Documents** | 10 (D1-D10) | 1,420 | SmartDoc competition | 7 / 1 / 2 |
| **Old Administrative Letters** | 10 (D11-D20) | 1,420 | Tobacco corpus | 7 / 1 / 2 |
| **Receipts** | 10 (D21-D30) | 1,420 | Various retail shops | 7 / 1 / 2 |

#### 10. References

##### Primary Citation

```bibtex
@inproceedings{nayef2015smartdocqa,
  title={SmartDoc-QA: A Dataset for Quality Assessment of Smartphone Captured Document Images - Single and Multiple Distortions},
  author={Nayef, Nibal and Luqman, Muhammad Muzzamil and Prum, Sophea and Eskenazi, Sebastien and Chazalon, Joseph and Ogier, Jean-Marc},
  booktitle={CBDAR@ICDAR},
  year={2015}
}
```

##### Related Datasets

- [diqa-5000](diqa-5000.md) - Primary DeQA training dataset with human MOS annotations
- [ocr-iqa-correlation](ocr-iqa-correlation.md) - FUNSD+-based OCR-IQA correlation study (similar evaluation methodology)
- [synth-ood-520](synth-ood-520.md) - Synthetic OOD test set

#### 11. Dataset-Specific Notes

##### 11.1 Filename Convention

Image filenames encode all capture parameters:

```text
{S|M}_Img_{Android|WP}_D{doc}_L{light}_r{dist}_a{lon_angle}_b{lat_angle}[_Mb{blur}][_Ob{focus}].jpg
```

| Field | Values | Meaning |
|-------|--------|---------|
| `S` / `M` | Single / Multiple | Distortion mode |
| `Android` / `WP` | Samsung / Nokia | Capture device |
| `D1`-`D30` | Document number | Maps to `Ground_truth/page_{N}.txt` |
| `L1`-`L5` | Lighting condition | L1=daylight, L2=+neon, L3=night+lamp, L4=+shadow, L5=+grid shadow |
| `r35` | Distance (cm) | Always 35 cm |
| `a{-10,-5,0,5}` | Longitudinal angle | Rotation around Y-axis (degrees) |
| `b{-5,0,5,10}` | Lateral angle | Rotation around X-axis (degrees) |
| `Mb{1,2}` | Motion blur | 1=horizontal, 2=2D |
| `Ob{N}` | Out-of-focus blur | Samsung: focus distance level; Nokia: focus parameter (850-900 = in focus) |

##### 11.2 OCR Accuracy to MOS Mapping

For Tier 1 training, FineReader character accuracy can be mapped to DeQA MOS:

| FineReader Char Accuracy | Approximate Quality | MOS Range |
|--------------------------|---------------------|-----------|
| 80-100% | Excellent / Good | 4.0 - 5.0 |
| 60-80% | Good / Fair | 3.0 - 4.0 |
| 40-60% | Fair / Poor | 2.0 - 3.0 |
| 20-40% | Poor / Bad | 1.0 - 2.0 |
| 0-20% | Bad | 1.0 |

This mapping is approximate. The VLM consensus labels provide an independent quality signal for cross-validation.

##### 11.3 Relationship to image_detection Project

SmartDoc-QA resides in `02_benchmark_only/` in the parent image_detection project, where it is reserved exclusively for SigLIP2 model benchmarking. For DeQA-Doc (a separate model), the train-split images are used for Tier 1 training data, while the test/val splits are used for evaluation. The document-level split ensures no content leakage affects either project.

##### 11.4 Extended SmartDoc Dataset

An [Extended SmartDoc Dataset](https://github.com/ricardobnjunior/Extended-Smartdoc-Dataset) exists (HU-PageScan, IET 2020) but is designed for page crop/segmentation, not quality assessment. It composites documents onto smartphone-captured backgrounds. Not used in DeQA-Doc.
