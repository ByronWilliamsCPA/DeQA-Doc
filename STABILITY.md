# Stability Guarantees

This document lists the modules, functions, and constants that constitute the
stable API surface of DeQA-Doc. These exports are consumed by the
[image_detection](https://github.com/…/image_detection) project via subprocess
isolation (see `docs/handoff/DEQA_INTEGRATION_HANDOFF.md` in that repo).

Breaking changes to stable interfaces require coordination with the
image_detection maintainer.

## Stable API Surface

### src/constants.py

| Export | Value | Used By |
|--------|-------|---------|
| `DEFAULT_IMAGE_TOKEN` | `"<\|image\|>"` | Prompt construction |
| `IMAGE_TOKEN_INDEX` | `-200` | Tokenizer input building |

### src/mm_utils.py

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_model_name_from_path` | `(model_path: str) -> str` | Extract model name from checkpoint path |
| `tokenizer_image_token` | `(prompt, tokenizer, image_token_index, return_tensors) -> list \| Tensor` | Build input IDs with image token placeholders |
| `expand2square` | `(pil_img, background_color) -> Image` | Pad non-square images to square (center-paste) |

### src/conversation.py

| Export | Interface |
|--------|-----------|
| `conv_templates["mplug_owl2"]` | Conversation object with `.copy()`, `.append_message(role, content)`, `.get_prompt()`, `.roles` |

### src/model/builder.py

| Function | Signature |
|----------|-----------|
| `load_pretrained_model` | `(model_path, model_base, model_name, load_8bit, load_4bit, device, preprocessor_path, ...) -> (tokenizer, model, image_processor, context_len)` |

### src/evaluate/scorer.py

| Class | Interface |
|-------|-----------|
| `Scorer` | `__init__(pretrained, device)`, `forward(images: list[Image]) -> Tensor` |

## Level Ordering Convention (Critical)

The quality level ordering is `[excellent, good, fair, poor, bad]` mapping to
MOS scores `[5, 4, 3, 2, 1]`. This is confirmed in `loss.py:25`,
`gen_soft_label.py:76`, and `cal_distribution_gap.py:79`. MOS reconstruction:
`np.inner(probs, [5, 4, 3, 2, 1])`.

## Experimental (No Stability Guarantees)

The following are under active development and may change without notice:

- `src/uncertainty/` -- Pseudo-labeling pipeline modules
- `results/` -- Research results and evaluation infrastructure
- `Llamafactory/` -- Qwen2.5-VL patches for LLaMA-Factory
- `scripts/` -- CLI orchestration scripts
