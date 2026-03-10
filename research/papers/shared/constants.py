"""Constants shared across all papers in the DeQA-Doc technical report series."""

from __future__ import annotations

# Project root (two levels up from shared/)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # research/papers/shared -> DeQA-Doc
RESULTS_DIR = PROJECT_ROOT / "results"
VLM_EVAL_DIR = RESULTS_DIR / "vlm_teacher_eval" / "full_eval"
OCR_IQA_DIR = PROJECT_ROOT / "research" / "ocr_iqa_correlation"

# License
LICENSE = "CC BY-SA 4.0, Copyright 2025 Byron Williams"

# VQualA metric weights: MainScore = 0.5*overall + 0.25*sharpness + 0.25*color
WSRCC_WEIGHTS = [0.5, 0.25, 0.25]
DIMENSIONS = ["overall", "sharpness", "color_fidelity"]
DIMENSION_LABELS = {"overall": "Overall", "sharpness": "Sharpness", "color_fidelity": "Color Fidelity"}

# DeQA quality levels (high to low)
QUALITY_LEVELS = ["excellent", "good", "fair", "poor", "bad"]
MOS_WEIGHTS = [5, 4, 3, 2, 1]

# Model display names (OpenRouter ID -> paper label)
MODEL_NAMES: dict[str, str] = {
    "google__gemini-3-flash-preview": "Gemini 3 Flash",
    "google__gemini-3-flash-preview__no_resize": "Gemini 3 Flash (no resize)",
    "google__gemini-2.5-pro": "Gemini 2.5 Pro",
    "openai__gpt-4.1": "GPT-4.1",
    "anthropic__claude-haiku-4.5": "Claude Haiku 4.5",
    "qwen__qwen3.5-flash-02-23": "Qwen 3.5 Flash",
    "qwen__qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "qwen__qwen3-vl-8b-thinking": "Qwen3-VL-8B Think",
    "qwen__qwen3-vl-8b-thinking__temp0": "Qwen3-VL-8B Think (t=0)",
}

# Ordered list of primary VLM models (for consistent plot ordering)
PRIMARY_MODELS = [
    "google__gemini-3-flash-preview",
    "openai__gpt-4.1",
    "google__gemini-2.5-pro",
    "qwen__qwen3.5-flash-02-23",
    "anthropic__claude-haiku-4.5",
    "qwen__qwen3-vl-8b-instruct",
    "qwen__qwen3-vl-8b-thinking",
]

# OOD synthetic categories
OOD_CATEGORIES = [
    "ood_heavily_degraded",
    "ood_adversarial_nastaliq",
    "ood_very_low_dpi",
    "ood_multiscript",
    "ood_script_tibetan",
    "ood_script_ethiopic",
    "ood_form_layout",
    "ood_adversarial_fraktur",
    "ood_pristine",
    "ood_very_high_dpi",
    "ood_binarized",
    "ood_script_myanmar",
    "ood_cjk_vertical",
]

# OCR engines
OCR_ENGINES = ["tesseract", "easyocr", "rapidocr", "gcloud_vision"]
OCR_ENGINE_LABELS = {
    "tesseract": "Tesseract",
    "easyocr": "EasyOCR",
    "rapidocr": "RapidOCR",
    "gcloud_vision": "Google Vision",
}

# Quality tiers for OCR-IQA study
QUALITY_TIERS = ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]
