# 01 Dependencies and Supply Chain

The core model stack is pinned to releases that are 2 to 3 years old (torch 2.0.1 from May 2023, transformers 4.36.1 from December 2023), and those pins are load-bearing for the championship mPLUG-Owl2 model, so they are accepted residuals rather than accidental drift. The honest risk is age stratification: torch 2.0.1 caps the runtime at Python 3.11 and CUDA 11.8, both of which the rest of the ecosystem has moved past, and the dependency floor list in `DeQA-Score/pyproject.toml` is a manual workaround for the inability to bump the core. Lockfile coverage is partial: `DeQA-Score/uv.lock` exists (92 packages) but `research/ocr_iqa_correlation/` has a `pyproject.toml` with no lockfile. No SBOM is generated. The accepted-CVE register in `docs/known-vulnerabilities.md` is current as of 2026-05-28 with a reassessment date (2026-07-27) that has not yet lapsed.

Tooling note: pip-audit and osv-scanner were not available in this environment, so CVE conclusions rely on `docs/known-vulnerabilities.md` and known advisory data, not a fresh live scan. Release dates are from package history knowledge.

## DEP-01 Core ML stack pinned to releases 2 to 3 years old
- Severity: High
- Effort: L (a transformers/torch bump requires re-running and re-validating the full training pipeline)
- CVE:
- Affected files: `DeQA-Score/pyproject.toml:16-29`
- Evidence: `torch==2.0.1` (released 2023-05-08), `transformers==4.36.1` (2023-12-18), `accelerate==0.21.0` (2023-07), `peft==0.4.0` (2023-07), `scikit-learn==1.2.2` (2023-03), `timm==0.6.13` (2023-01), `einops==0.6.1` (2023-04). All exceed 18 months since release as of 2026-05-29. `transformers==4.36.1` alone carries 32 open advisories per `docs/known-vulnerabilities.md`.
- Recommendation: Keep the pins (they are required and documented), but treat the eventual model-stack migration as a tracked epic with the reassessment date, not an open-ended deferral. The longer the gap grows, the larger the eventual jump.

## DEP-02 torch 2.0.1 caps runtime at Python 3.11 / CUDA 11.8
- Severity: High
- Effort: L (coupled to DEP-01)
- CVE:
- Affected files: `DeQA-Score/pyproject.toml:11` (`requires-python = ">=3.10,<3.12"`), `DeQA-Score/pyproject.toml:88-99` (`[[tool.uv.index]]` pytorch-cu118)
- Evidence: torch 2.0.1 publishes cu118 wheels only for Python 3.8 to 3.11, so the project cannot move to 3.12+. Python 3.10 reaches end of security support 2026-10 (about 5 months out); 3.11 reaches it 2027-10. CUDA 11.8 is two major CUDA generations behind current driver stacks.
- Recommendation: Plan the 3.12 move together with the torch upgrade; do not let 3.10 reach EOL as the floor of the supported range. Document the CUDA 11.8 requirement in the README setup section.

## DEP-03 Unlocked second project (research/ocr_iqa_correlation)
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `research/ocr_iqa_correlation/pyproject.toml`, (no `research/ocr_iqa_correlation/uv.lock`)
- Evidence: `ls research/ocr_iqa_correlation/uv.lock` returns nothing; the directory declares its own `[project]` with ranged deps (`jiwer>=3.0`, `datasets>=2.14`, `docling>=2.0`, `google-cloud-vision>=3.0`) and no lockfile. Builds here are not reproducible.
- Recommendation: Run `uv lock` in that subproject and commit the lockfile, or fold its deps into an optional extra of the root project if it is not meant to be installed standalone.

## DEP-04 Transitive security floors are a manual list that can rot
- Severity: Medium
- Effort: M
- CVE:
- Affected files: `DeQA-Score/pyproject.toml:30-50`
- Evidence: 11 transitive floors are pinned by hand (`idna>=3.15`, `pillow>=12.2.0`, `GitPython>=3.1.50`, `h11>=0.16.0`, `certifi>=2024.7.4`, `pygments>=2.20.0`, `requests>=2.34.2`, `starlette>=0.49.3`, `urllib3>=2.7.0`, `filelock>=3.20.1`) with a comment block explaining each is a security floor. This list has no automated check that it stays ahead of new advisories; it drifts the moment a new CVE lands on one of these or on a floor not yet listed.
- Recommendation: Add a scheduled `osv-scanner`/`pip-audit` job against `uv.lock` (the org has osv-scanner in `.qlty`); let it flag when a new floor is needed rather than relying on manual edits.

## DEP-05 No SBOM generated
- Severity: Low
- Effort: S
- CVE:
- Affected files: `.github/workflows/` (none emit CycloneDX/SPDX/syft output)
- Evidence: `grep -rilE 'sbom|cyclonedx|syft|spdx' .github/` matches only `reuse.yml` (license compliance, an SPDX-adjacent but not a dependency SBOM) and `dependency-review.yml` (PR-time diff, not a published SBOM). No artifact inventories the resolved dependency graph.
- Recommendation: If supply-chain provenance matters for releases, add a CycloneDX SBOM step from `uv.lock`. Low priority for an internal research fork.

## Clean areas (one line each)
- Migration residue: none; no `requirements*.txt`, `setup.py`, `setup.cfg`, `poetry.lock`, or `Pipfile` alongside the `pyproject.toml` + `uv.lock` pair.
- Known-vulnerabilities register is current: documented 2026-05-28, reassessment 2026-07-27 (not yet lapsed as of 2026-05-29).
- Renovate is configured with vulnerability alerts, OSV alerts, transitive remediation, and Actions SHA-pinning (`renovate.json`).
