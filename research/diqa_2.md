# DIQA-5000 and the VQualA 2025 competition landscape

**The DIQA-5000 dataset has been used in exactly one competition to date: the Document Image Quality Assessment Challenge at the inaugural VQualA Workshop, held at ICCV 2025.** Seven teams submitted final solutions, with the top four all leveraging multimodal large language models (MLLMs) to achieve scores above 0.92. The winning team, DeQA-Doc, scored **0.929** by adapting the DeQA-Score framework with soft-label Gaussian distributions. No other competition on any platform — including Kaggle, EvalAI, or ICDAR 2025 — has used DIQA-5000 as of March 2026.

The dataset was created by researchers at **Shanghai Jiao Tong University** and introduced alongside a baseline model called DocIQ (arXiv:2509.17012). It fills a critical gap in the document analysis field: prior document image quality datasets relied on proxy metrics like OCR accuracy rather than direct human perceptual judgments. DIQA-5000 provides multi-dimensional Mean Opinion Scores from human raters, making it a first-of-its-kind resource for training and benchmarking no-reference document quality assessment models.

---

## What DIQA-5000 contains and why it matters

DIQA-5000 comprises **5,000 enhanced document images** derived from 500 original mobile-captured documents. The originals were curated from publicly accessible PDFs spanning diverse content types — text, tables, diagrams, handwritten notes, and mixed layouts in English, Chinese, and mathematical notation — printed at 300 dpi to create paper originals that were then photographed under real-world conditions.

Each original was subjected to one of **five distortion categories** (shadows, occlusions, blur, creases, and moiré patterns), yielding 500 distorted images. Six enhancement operations — dewarping, demoiré processing, occlusion removal, deblurring, deshadowing, and appearance enhancement — were then applied in randomized combinations to produce **10 enhanced versions per distorted original**, totaling 5,000 images.

The annotation protocol follows **ITU-R BT.500 guidelines**. A pool of 23 experienced raters evaluated images across three quality dimensions: **overall quality**, **sharpness** (text and figure clarity), and **color fidelity**. Each image received ratings from 15 raters, with outlier filtering applied to produce Mean Opinion Scores (MOS) on a continuous scale. The dataset is split into training (3,500), validation (500), and testing (1,000) sets at a 7:1:2 ratio, with all enhanced versions of the same original confined to the same partition to prevent data leakage.

The dataset was created by **Zhichao Ma, Fan Huang, Lu Zhao, Xiaohong Liu, Xiongkuo Min, and Guangtao Zhai** at Shanghai Jiao Tong University's Department of Electronic Engineering. The accompanying paper, "DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment," was posted to arXiv on September 21, 2025 (arXiv:2509.17012), and introduces a layout-aware feature fusion baseline model achieving SRCC of 0.8704 and PLCC of 0.8999.

---

## The VQualA 2025 workshop at ICCV: seven tracks, one venue

The **Visual Quality Assessment Competition (VQualA)** workshop held its first edition at ICCV 2025 in Honolulu, Hawaii, on **October 19, 2025**. The workshop featured seven challenge tracks spanning diverse quality assessment domains, all hosted on the CodaLab platform:

1. **DIQA** — Document Image Enhancement Quality Assessment (CodaLab #23020)
2. **ISRGC-Q** — Image Super-Resolution Generated Content Quality Assessment (#22924)
3. **FIQA** — Face Image Quality Assessment (#23017)
4. **EVQA-SnapUGC** — Engagement Prediction for Short Videos (#23005)
5. **Visual Quality Comparison for Large Multimodal Models** (#23016)
6. **GenAI-Bench AIGC Video Quality Assessment Track I** (#23067)
7. **GenAI-Bench AIGC Video Quality Assessment Track II** (#23070)

The workshop was organized by **20 researchers** led by Chris Wei Zhou (Cardiff University), Jian Wang and Sizhuo Ma (Snap Research), Xiongkuo Min, Xiaohong Liu, and Guangtao Zhai (SJTU), and Zhengzhong Tu (Texas A&M). Sponsors included **Snap Research, INTSIG Information Co. Ltd, and TAOBAO & TMALL Group (Alibaba)**. All challenge papers were published in the IEEE/CVF ICCV 2025 Workshop Proceedings and are available through CVF Open Access and IEEE Xplore.

The DIQA challenge specifically was organized by Fan Huang, Xiongkuo Min, Zhichao Ma, Xiaohong Liu (all SJTU), Chris Wei Zhou (Cardiff), and Guangtao Zhai (SJTU), with sponsorship from INTSIG. The competition ran from **May 25 to July 4, 2025**, with winners announced July 30.

---

## All seven DIQA challenge teams and their methods

The DIQA track attracted **120 registered participants**, with 16 teams active during development (183 submissions) and testing (97 submissions) phases. Seven teams submitted final models and technical fact sheets. The evaluation metric combined PLCC and SROCC across three quality dimensions using the formula: **MainScore = 0.5 × Score_overall + 0.25 × Score_sharpness + 0.25 × Score_color**.

| Rank | Team | Affiliation | Score | Core method |
|------|------|-------------|-------|-------------|
| **1** | **DeQA-Doc** | Ant Group, CUHK, Southeast Univ., SJTU, MBZUAI | **0.929** | DeQA-Score MLLM with soft-label Gaussian distributions; ensemble of mPLUG-Owl2 and Qwen2.5-VL |
| **2** | **mapleyzzzz** | Southern Univ. of Science and Technology | **0.927** | Ensemble of 3 MLLMs (Qwen2.5-VL-7B, mPLUG-Owl2-LLaMA2-7B, Qwen2.5-VL-32B) with 5-fold cross-validation |
| **3** | **QA-Veteran** | SJTU, East China Normal Univ., CityU Hong Kong | **0.925** | SigLIP2-NaFlex vision-language model with resolution-adaptive quality templates |
| **4** | **NJUST-KMG** | Nanjing Univ. of Science and Technology | **0.924** | LoRA fine-tuned Qwen2.5-VL-7B + MiMoVL-7B-RL + Qwen2-VL-7B with test-time augmentation |
| **5** | **GoldenChef** | Nanjing Univ. of Posts and Telecommunications | **0.898** | ConvNeXt + Swin Transformer dual-pathway with 25-patch decomposition |
| **6** | **2077Agent** | Chengdu Univ. of Tech, Hebei Univ. of Tech, Sichuan Univ., Hokkaido Univ. | **0.828** | Multi-expert fusion (ViT + ResNet + MANIQA) with Rectified Flow regularization |
| **7** | **BIT ssvgg** | Beijing Institute of Technology | N/A | ResNet-50 encoder with Mixture-of-Experts hypernetwork |

For context, the strongest baseline method was **RichIQA at 0.866**, meaning the top five teams all surpassed it, while the weakest baseline (DBCNN) scored just 0.587. The performance gap between the MLLM-based approaches (ranks 1–4, all > 0.92) and the CNN-based approaches (ranks 5–7) was striking — **a clear signal that multimodal LLMs now dominate document quality assessment**.

The winning team's paper, "DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment" by Junjie Gao et al. (arXiv:2507.12796), and the official challenge overview paper by Fan Huang et al. (ICCVW 2025, pp. 3313–3320) are both available through CVF Open Access.

---

## No other competitions have used DIQA-5000

An exhaustive search across all major competition platforms and academic venues confirms that **the VQualA 2025 DIQA Challenge is the only competition to have used DIQA-5000** as of March 2026. Specifically:

- **Kaggle**: No datasets or competitions referencing DIQA-5000 exist.
- **EvalAI**: No challenges found.
- **CodaLab**: Only competition #23020 (VQualA 2025 DIQA).
- **ICDAR 2025** (Wuhan, September 2025): Hosted 9+ competitions on topics including historical map analysis, document machine translation, glyph detection, and handwriting recognition — none involving document image quality assessment or DIQA-5000.
- **CVPR 2025 / ECCV 2024**: No competitions referencing DIQA-5000 were identified.

This is unsurprising given the dataset's recency — DIQA-5000 was first publicly described in September 2025, and the VQualA challenge was its debut venue. However, the dataset is already being adopted as a benchmark in subsequent research.

---

## Growing academic footprint beyond the competition

Three key papers anchor the DIQA-5000 literature. The **DocIQ dataset paper** (arXiv:2509.17012) by Ma et al. introduces the dataset and a baseline feature fusion network. The **challenge overview paper** by Huang et al. (ICCVW 2025, pp. 3313–3320) documents all seven team submissions, methods, and results. The **DeQA-Doc paper** (arXiv:2507.12796) by Gao et al. details the winning approach and is accompanied by a public GitHub repository.

A fourth paper signals broader adoption: **"Q-Doc: Benchmarking Document Image Quality Assessment Capabilities in Multi-modal Large Language Models"** by Huang et al. (arXiv:2511.11410, published in PRCV 2025/2026 proceedings via Springer) uses DIQA-5000 as one of its primary benchmarks alongside SmartDoc-QA and DocImg-QA. This paper found that even state-of-the-art MLLMs like GPT-4o achieve only **SRCC 0.1321** on zero-shot DIQA tasks using DIQA-5000, while DeepSeek-VL2 reaches 0.4474 — far below the supervised competition results, highlighting significant room for improvement in zero-shot document quality understanding.

---

## Conclusion

DIQA-5000 represents the first large-scale, human-annotated, multi-dimensional document image quality assessment dataset — a departure from the OCR-proxy approach that dominated prior work. Its sole competition deployment at VQualA 2025 (ICCV) revealed that **MLLM-based architectures have decisively outperformed traditional CNN approaches** for this task, with the top four teams all using variants of Qwen2.5-VL or mPLUG-Owl2 fine-tuned with quality-aware objectives. The extremely tight clustering of top scores (0.924–0.929) suggests the current MLLM paradigm may be approaching a performance ceiling on this benchmark, making DIQA-5000 a ripe target for novel architectural innovations. As the dataset is less than a year old, its use in future competitions — potentially at ICDAR 2027, CVPR, or a second edition of VQualA — seems highly likely given its rapid early adoption in the research literature.