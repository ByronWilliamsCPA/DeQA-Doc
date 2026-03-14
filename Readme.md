# DeQA-Doc: Document Image Quality Assessment with Pseudo-Label Scaling

Fork of [Junjie-Gao19/DeQA-Doc](https://github.com/Junjie-Gao19/DeQA-Doc) -- the VQualA 2025 DIQA Challenge Championship solution -- extended with a confidence-weighted pseudo-labeling pipeline for scaling beyond the 5,000 human-labeled DIQA-5000 training set.

## What This Fork Adds

The original DeQA-Doc trains on only 5,000 human-labeled document images. This fork adds infrastructure to scale to 100K+ images via pseudo-labeling with multi-signal uncertainty quantification:

- **4-signal uncertainty fusion** -- Mahalanobis OOD distance, cross-model JSD (SigLIP2 vs DeQA), aleatoric variance, and prediction entropy
- **Tiered acceptance** -- auto-accept, low-weight, VLM veto, hard-reject decisions per sample
- **Cross-model validation** -- detects where SigLIP2-IQA-Base-86M disagrees with DeQA public models
- **Active learning** -- BALD-based sample selection for efficient human annotation
- **Validation safeguards** -- bootstrap CI, harm checks, distribution drift monitoring

All pseudo-labels output in the existing DeQA training JSON format -- no training code changes required.

### Uncertainty Pipeline (`DeQA-Score/src/uncertainty/`)

| Module | Purpose |
|--------|---------|
| `ood_wrapper.py` | Mahalanobis-distance OOD detector from SigLIP2 embeddings |
| `gaussian_to_discrete.py` | Convert SigLIP2 (mu, sigma-sq) to 5-level quality probabilities |
| `discrete_metrics.py` | JSD, KL divergence, entropy, BALD for discrete distributions |
| `cross_validator.py` | Compare SigLIP2 vs DeQA model predictions |
| `fusion.py` | 4-signal uncertainty fusion with per-dimension thresholds |
| `vlm_validator.py` | Tier-2 VLM veto via Qwen3-VL-8B (OpenRouter) |
| `pseudo_label.py` | End-to-end pipeline orchestrator |
| `format_training_data.py` | Convert to `SingleDataset`-compatible training JSON |
| `active_learning.py` | BALD-based annotation queue generation |
| `validation.py` | Bootstrap CI, harm checks, distribution drift |

### CLI Scripts

```bash
# Run full pseudo-labeling pipeline
python scripts/run_pseudo_label.py \
    --siglip2-results predictions/siglip2_iqa.json \
    --embeddings embeddings/unlabeled.npy \
    --ood-params embeddings/ood_params_4400.npz \
    --deqa-specialist predictions/specialist_labels.jsonl \
    --output-dir Data-DeQA-Score/pseudo/ \
    --per-dimension

# Select samples for human annotation
python scripts/run_active_learning.py \
    --pseudo-label-dir Data-DeQA-Score/pseudo/ \
    --sacred-test-ids artifacts/sacred_test_ids.json \
    --output-queue artifacts/annotation_queue.json \
    --k 1000

# Validate OOD detector calibration
python scripts/validate_ood_checkpoint.py \
    --ood-params embeddings/ood_params_4400.npz \
    --test-embeddings embeddings/diqa5000_test_all.npy
```

### Research Results (`results/`)

- `tier1_ood_detector/` -- Mahalanobis OOD detector methodology and calibration
- `vlm_model_selection/` -- VLM model comparison for Tier-2 cross-validation (selected Qwen3-VL-8B)

See [research/INDEX.md](research/INDEX.md) for the full research index, experiment registry, hypothesis backlog, and data inventory.

### Free Model Monitor (`results/vlm_teacher_eval/full_eval/check_free_models.py`)

Daily check for free vision models on OpenRouter. Queries the `/models` API for `:free` models with image input, diffs against previous runs to detect newly appeared or removed models, and optionally auto-launches DIQA-5000 evaluation for new models.

```bash
cd DeQA-Score
PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
    ../results/vlm_teacher_eval/full_eval/check_free_models.py          # dry run
    ../results/vlm_teacher_eval/full_eval/check_free_models.py --run    # evaluate new/incomplete
    ../results/vlm_teacher_eval/full_eval/check_free_models.py --json   # machine-readable output
    ../results/vlm_teacher_eval/full_eval/check_free_models.py --all    # include completed models
```

State is tracked in `free_models_state.json` so successive runs can report additions and removals.

### Technical Report Series (`research/papers/`)

A 10-paper arXiv-style technical report series documenting the full research program. Each paper includes a figure generation script, a living research agenda, and a 5-model consensus peer review. Generated from the comprehensive evaluation in [`results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md`](results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md).

| # | Title | Words | Focus |
|---|-------|-------|-------|
| 0 | [VQualA 2025 DIQA Challenge: A Competition Analysis](research/papers/00_competition_analysis/paper.md) | 3,939 | Competition landscape, all 7 team solutions |
| 1 | [VLM Benchmark for Document Image Quality Assessment](research/papers/01_vlm_benchmark/paper.md) | 6,232 | 7 VLMs vs human MOS on DIQA-5000 |
| 2 | [Cross-Domain Generalization of VLM Quality Assessors](research/papers/02_cross_domain/paper.md) | 5,407 | 13 OOD categories, ID vs OOD performance |
| 3 | [Prompt Engineering for VLM-Based Quality Assessment](research/papers/03_prompt_engineering/paper.md) | 3,501 | 7-arm prompt comparison, regression to mean |
| 4 | [Embedding-Space OOD Detection for Document Quality Pipelines](research/papers/04_ood_detection/paper.md) | 4,980 | Mahalanobis distance, AUROC 0.9963 |
| 5 | [Off-the-Shelf NR-IQA Models on Document Images](research/papers/05_nriqa_baselines/paper.md) | 3,011 | 26 NR-IQA models, domain gap analysis |
| 6 | [DeQA Quality Scores Predict OCR Accuracy](research/papers/06_ocr_iqa_correlation/paper.md) | 5,017 | 1,200 images x 4 OCR engines |
| 7 | [Iterative Pseudo-Labeling Pipeline for Domain Expansion](research/papers/07_pseudo_labeling/paper.md) | 6,079 | End-to-end pipeline design (capstone) |
| 8 | [Training SigLIP2-IQA-Base: A Lightweight Document IQA Model](research/papers/08_siglip2_training/paper.md) | 3,720 | 86M-param student, MainScore 0.886 |
| 9 | [Training HyperIQA++: CNN Fine-Tuning for Document IQA](research/papers/09_hyperiqa_training/paper.md) | 3,124 | CNN baseline, generalization gap analysis |

```bash
# Generate all paper figures
python research/papers/generate_all.py

# Generate figures for specific papers
python research/papers/generate_all.py --paper 1 6 8
```

**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

## Original DeQA-Doc

> **Paper**: [DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment](https://arxiv.org/abs/2507.12796)
>
> **Authors**: Junjie Gao, Runze Liu, Yingzhe Peng, Shujian Yang, Jin Zhang, Kai Yang, Zhiyuan You
>
> **Achievement**: Championship in VQualA 2025 DIQA Challenge

The system predicts quality scores across three dimensions -- **overall quality**, **sharpness**, and **color fidelity** -- using discrete quality levels (excellent/good/fair/poor/bad) with soft-label distribution learning.

Two model backends:

- **mPLUG-Owl2-7B** -- trained via the DeQA-Score codebase in `DeQA-Score/`
- **Qwen2.5-VL-7B** -- trained via LLaMA-Factory with patched files in `Llamafactory/`

### Installation

```bash
cd DeQA-Score
pip install -e .          # inference only
pip install -e ".[train]" # training
pip install -e ".[dev]"   # development (pytest, ruff)
```

### Pre-trained Models

- Initial weights: [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)
- DIQA dimension-specific models: [ModelScope DeQA-Doc](https://www.modelscope.cn/models/zhalala/DeQA-Doc/summary)
- Multi-dimension mixed model: [ModelScope DeQA-Doc-Mix](https://www.modelscope.cn/models/zhalala/DeQA-Doc-Mix/summary)

### Training and Inference

```bash
# mPLUG-Owl2
sh scripts/train.sh           # full fine-tuning
sh scripts/train_lora.sh      # LoRA fine-tuning
sh scripts/infer.sh            # inference
sh scripts/diqa_eval.sh        # format results for DIQA evaluation

# Qwen2.5-VL (requires LLaMA-Factory installation)
llamafactory-cli train examples/train_full/qwen2.5_vl_diqa_sft.yaml
sh scripts/infer_qwen.sh
```

### Testing

```bash
cd DeQA-Score
.venv/bin/python -m pytest tests/uncertainty/ -v  # 122 tests
```

## Acknowledgements

- [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score) -- the foundation this work builds on
- [DeQA-Doc](https://github.com/Junjie-Gao19/DeQA-Doc) -- the original DIQA adaptation and challenge winner

## Citation

If you use this work, please cite the original DeQA-Doc paper:

```bibtex
@inproceedings{deqadoc,
  title={{DeQA-Doc}: Adapting {DeQA-Score} to Document Image Quality Assessment},
  author={Gao, Junjie and Liu, Runze and Peng, Yingzhe and Yang, Shujian and Zhang, Jin and Yang, Kai and You, Zhiyuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop},
  year={2025},
}
```
