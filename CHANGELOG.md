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

### Changed

- Set `requires-python` to `>=3.10,<3.12` (the torch 2.0.1 cu118 wheel range),
  so all transitive security floors resolve unconditionally with no vulnerable
  Python 3.8/3.9 fallback.
- Lowered the training `deepspeed` pin to `0.14.5`, the highest release still
  compatible with the mandatory `pydantic<2` pin (0.15.x requires pydantic v2).

### Removed

- Removed the unused upstream `gradio` / `gradio_client` web-demo stack (and its
  transitive web dependencies); it is not imported by any train/infer/eval path.
- Removed `.github/workflows/codeql.yml` and `.github/workflows/dependency-review.yml`,
  and the inline `actions/dependency-review-action` step (and its now-empty
  `dependency-review` job) from `ci.yml`. GitHub now bills Advanced Security
  (Code Security) for CodeQL code scanning and the dependency-review action,
  so neither ran anymore on this repo. Ruff, Bandit, and the secrets-grep
  step in `ci.yml`'s Quality Checks job, and the SonarCloud Code Analysis
  gate, remain as the active static-analysis controls. Automated dependency
  vulnerability scanning is not currently running in CI (all scanner inputs
  in `security-analysis.yml`'s reusable workflow call are `false`); known
  residual CVEs continue to be tracked manually in
  `docs/known-vulnerabilities.md`.

### Security

- Raised transitive dependency floors (idna, pillow, urllib3, requests,
  starlette, h11, certifi, GitPython, pygments, filelock) to patched releases.
- Documented residual unfixable CVEs (transformers, torch, sentencepiece,
  scikit-learn, deepspeed/CVE-2024-43497) in `docs/known-vulnerabilities.md`.

## Baseline

The upstream DeQA-Doc history was not previously tracked in a changelog. The
fork starts from upstream version 1.2.0; entries above describe additions made
on top of that baseline.

[Unreleased]: https://github.com/ByronWilliamsCPA/DeQA-Doc
