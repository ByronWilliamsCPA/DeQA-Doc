# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeQA-Doc adapts [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score) to **Document Image Quality Assessment (DIQA)**. It won the Championship in the VQualA 2025 DIQA Challenge. The system predicts quality scores across three dimensions: **overall quality**, **sharpness**, and **color fidelity**, using discrete quality levels (excellent/good/fair/poor/bad) with soft-label distribution learning.

Two model backends are supported:
- **mPLUG-Owl2-7B** (primary) — trained via the DeQA-Score codebase in `DeQA-Score/`
- **Qwen2.5-VL-7B** — trained via LLaMA-Factory with patched files in `Llamafactory/`

## Architecture

### Training Pipeline (mPLUG-Owl2)

The core training loop is in `DeQA-Score/src/train/train_mem.py`. It uses HuggingFace Trainer (via `MPLUGOwl2Trainer`) with DeepSpeed ZeRO-3. Key components:

- **Model**: `MPLUGOwl2LlamaForCausalLM` — a multimodal LLM with CLIP vision encoder + visual abstractor + LLaMA-2 decoder. Loaded via `src/model/builder.py:load_pretrained_model()`.
- **Loss** (`src/train/loss.py`): Multi-component DeQA loss combining:
  - **Next-token CE loss** (standard autoregressive)
  - **SoftKL loss**: KL divergence between predicted token distribution at the quality-level position and ground-truth soft label distribution over the 5 level tokens. Uses `find_prefix()` to locate the quality prefix in the label sequence, then extracts logits at the next position.
  - **In-level loss**: Encourages probability mass to concentrate on the 5 level tokens
  - **Ranking loss**: Pairwise ranking between image pairs (pair dataset mode)
- **Datasets** (`src/datasets/`): Two modes — `SingleDataset` and `PairDataset`. Pair mode samples two images from the same dataset and computes pairwise ranking loss. Data format is JSON with conversations, `level_probs`, `gt_score`, and `std` fields.
- **Training args**: `level_prefix` identifies where quality levels appear in the output sequence; `level_names` maps to tokenizer IDs for the 5 quality words.

### Training Pipeline (Qwen2.5-VL via LLaMA-Factory)

Files in `Llamafactory/` are **patches** to be copied into a LLaMA-Factory installation (paths match). Key modifications:
- `train/sft/loss.py`: Reimplements DeQA's SoftKL loss for Qwen2.5-VL architecture (handles `pixel_values` + `image_grid_thw`)
- `train/sft/trainer.py` / `workflow.py`: Custom trainer with DeQA loss integration
- `data/collator.py`, `data/converter.py`, `data/parser.py`: Modified data pipeline for DIQA format
- `hyparams/finetuning_args.py`: Adds `use_deqa_loss` flag

### Inference & Evaluation

- `src/evaluate/iqa_eval.py`: mPLUG-Owl2 inference — loads model, runs per-image prediction, extracts quality level probabilities
- `src/evaluate/iqa_eval_qwen.py`: Qwen2.5-VL inference with multi-GPU support
- `src/evaluate/cal_plcc_srcc.py`: Computes SRCC/PLCC correlation metrics between predictions and ground truth
- `src/evaluate/cal_distribution_gap.py`: KL/JS divergence and Wasserstein distance for distribution evaluation
- `src/evaluate/scorer.py`: Quick-start scorer API (`from src import Scorer`)

### Research Results (`results/`)

- `tier1_ood_detector/`: Mahalanobis-distance OOD detector using SigLIP2 embeddings for flagging out-of-distribution documents
- `vlm_model_selection/`: VLM model comparison for Tier 2 cross-model validation (selected Qwen3-VL-8B via OpenRouter)

## Common Commands

All commands assume working directory is `DeQA-Score/`:

```bash
# Install (inference + dev tools)
uv sync --extra dev

# Install (training — adds deepspeed, ninja, wandb)
uv sync --extra train --extra dev

# Install flash_attn separately (needs --no-build-isolation)
uv pip install flash_attn --no-build-isolation

# Set PYTHONPATH (required for all scripts)
export PYTHONPATH=./:$PYTHONPATH

# Training (mPLUG-Owl2)
sh scripts/train.sh         # Full fine-tuning (8x A6000 or 4x A100)
sh scripts/train_lora.sh    # LoRA fine-tuning (2x RTX3090)

# Inference
sh scripts/infer.sh $GPU_ID           # mPLUG-Owl2 full model
sh scripts/infer_lora.sh $GPU_ID      # mPLUG-Owl2 LoRA
sh scripts/infer_qwen.sh              # Qwen2.5-VL

# Evaluation
sh scripts/eval_score.sh              # SRCC/PLCC metrics
sh scripts/eval_dist.sh               # Distribution gap metrics
sh scripts/diqa_eval.sh               # DIQA-format evaluation

# Quick scorer
uv run python src/evaluate/scorer.py --img_path fig/singapore_flyer.jpg

# Training Qwen2.5-VL (requires separate LLaMA-Factory installation)
llamafactory-cli train examples/train_full/qwen2.5_vl_diqa_sft.yaml
```

## Key Dependencies

Managed via `uv` with lockfile (`uv.lock`). Python 3.8-3.11 required (torch 2.0.1 has no 3.12+ wheels). PyTorch CUDA 11.8 wheels are configured via `[[tool.uv.index]]` in `pyproject.toml`.

- `torch==2.0.1`, `torchvision==0.15.2` (from PyTorch cu118 index)
- `transformers==4.36.1` (mPLUG-Owl2 compatibility)
- `deepspeed==0.9.5` (training only, via `--extra train`)
- `peft==0.4.0` (LoRA)
- `accelerate==0.21.0`

## Data Layout

Training data JSON files live in `Data-DeQA-Score/` (sibling to `DeQA-Score/`). Each sample has:
- `image`: relative path to image file
- `conversations`: chat-format Q&A with quality descriptions
- `level_probs`: 5-element soft label distribution over [excellent, good, fair, poor, bad]
- `gt_score`: ground truth MOS score
- `std`: standard deviation of human ratings

DIQA-specific metas are in `DeQA-Score/Data-DeQA-Score/DIQA/metas/` and `KONIQ/metas/`.

## Testing

Tests live in `DeQA-Score/tests/` (run from `DeQA-Score/` with PYTHONPATH set):

```bash
export PYTHONPATH=./:$PYTHONPATH
uv run python -m pytest tests/                                   # All tests
uv run python -m pytest tests/datasets/test_pair_dataset.py      # Single file
uv run python -m pytest tests/model/test_find_prefix.py::test    # Single test
```

Note: The test files in `tests/model/` are exploratory scripts (module-level code with `if __name__ == "__main__"` blocks), not proper pytest functions. They require model weights or tokenizer files to run. `tests/datasets/` has proper test structure but needs the transformers tokenizer import chain.

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- **CI** (`ci.yml`) — PRs and pushes to main. Quality checks (Ruff lint/format on `results/` only, Python syntax check on all files, Bandit security scan, secrets detection), CodeQL security analysis, dependency review on PRs. All actions pinned to commit SHAs.

Tests requiring torch/CUDA are **not** run in CI — the GPU dependencies can't install on standard runners. Run tests locally with a GPU environment.

### Qlty

Quality platform config in `.qlty/qlty.toml` with 9 plugins: ruff, bandit, osv-scanner, radarlint-python, trufflehog, actionlint, markdownlint, yamllint, ripgrep.

Upstream code (`DeQA-Score/src/`, `Llamafactory/`) has line-length violations suppressed in triage rules. Only new code in `results/` and project-level files are strictly checked.

## Org Standards Deviations

This is a research codebase forked from an upstream paper implementation. It intentionally deviates from the org-wide standards in `~/.claude/CLAUDE.md`:

- **Selective Ruff/BasedPyright enforcement** — upstream code uses its own style, has Chinese comments, and would require extensive refactoring to comply. Do not auto-format or add type annotations to existing upstream code. CI only lints `results/` and new files.
- **Pinned legacy dependencies** — `torch==2.0.1`, `transformers==4.36.1`, etc. are required for mPLUG-Owl2 compatibility. Do not upgrade without testing the full training pipeline.
- **Minimal `pyproject.toml` tooling config** — the existing `pyproject.toml` has `[tool.uv]` for dependency management but no Ruff, BasedPyright, or pytest tool sections (upstream code doesn't comply).
- **No signed commits required** — research collaboration repo with external contributors.
- **Lightweight CI only** — no reusable org workflows. Uses standalone workflow with targeted checks.

When adding **new code** (e.g., in `results/`, new evaluation scripts, or new modules), follow org standards: conventional commits, type hints, docstrings, and clean Python style. Apply Ruff formatting to new files only.

## Git Workflow

Follow org conventions for new work:

- **Branch naming**: `feat/`, `fix/`, `docs/`, `refactor/` prefixes with lowercase hyphen-separated slugs
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:` prefixes
- **Never commit directly to `main`** — use feature branches

## Environment

Requires `GEMINI_API_KEY` and `OPENROUTER_API_KEY` in `.env` for VLM experiment scripts (see `.env.example`). Not needed for core training/inference.
