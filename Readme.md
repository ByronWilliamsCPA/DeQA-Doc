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
