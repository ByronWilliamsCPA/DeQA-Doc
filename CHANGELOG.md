# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This fork of [Junjie-Gao19/DeQA-Doc](https://github.com/Junjie-Gao19/DeQA-Doc),
the VQualA 2025 DIQA Challenge championship solution, adds a confidence-weighted
pseudo-labeling pipeline for scaling the training set beyond the 5,000
human-labeled DIQA-5000 images.

### Added

- Pseudo-label pipeline in `DeQA-Score/src/uncertainty/` that emits labels in
  the existing DeQA training JSON format, requiring no training code changes.
- 4-signal uncertainty fusion combining Mahalanobis OOD distance, cross-model
  JSD (SigLIP2 vs DeQA), aleatoric variance, and prediction entropy.
- Tiered acceptance logic producing auto-accept, low-weight, VLM-veto, and
  hard-reject decisions per sample, with optional per-dimension thresholds.
- Cross-model validation that flags where SigLIP2-IQA-Base-86M disagrees with
  the DeQA public models.
- Active learning via BALD-based sample selection for efficient human
  annotation queue generation.
- Validation safeguards: bootstrap confidence intervals, harm checks, and
  distribution drift monitoring.
- CLI entry points for the pipeline (`scripts/run_pseudo_label.py`),
  active learning (`scripts/run_active_learning.py`), and OOD checkpoint
  validation (`scripts/validate_ood_checkpoint.py`).

## Baseline

The upstream DeQA-Doc history was not previously tracked in a changelog. The
fork starts from upstream version 1.2.0; entries above describe additions made
on top of that baseline.

[Unreleased]: https://github.com/ByronWilliamsCPA/DeQA-Doc
