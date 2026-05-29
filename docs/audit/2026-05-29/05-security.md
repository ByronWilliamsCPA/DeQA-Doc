# 05 Security and Secrets

No live secrets are committed: `.env` is untracked, `.env.example` holds only empty placeholders, and the only `api_key=` literals are `"test-key"` in unit tests. The known-vulnerable dependencies (transformers 4.36.1, deepspeed) are real but reachable only through trusted, project-owned model artifacts, and `docs/known-vulnerabilities.md` documents that compensating control honestly, so they rate Medium, not Critical. The genuine gaps are process-level: there is no detect-secrets baseline and no git-history secret scan (trufflehog runs on staged files only), so a secret committed on a branch before the hook existed would not be caught. Code-pattern risk is low: one `eval()` and one `pickle.load` exist, both on self-owned inputs, and there are no `shell=True` subprocess calls.

Tooling note: bandit, trufflehog, and pip-audit were not installed; the bandit-equivalent findings below come from `ruff --select S` (flake8-bandit rules), not bandit itself.

## SEC-01 No git-history secret scan and no detect-secrets baseline
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `.pre-commit-config.yaml:34-40`, `.github/workflows/ci.yml` (secret grep step), (no `.secrets.baseline`)
- Evidence: The trufflehog pre-commit hook scans `git diff --cached` (staged files) only, with an inline comment stating history mode is deliberately excluded. CI's secret check is a regex `grep` over `*.py/*.sh/*.yaml`, not a scanner. No `.secrets.baseline` exists. A secret introduced on any branch before this hook, or in a file type the grep misses (`.json`, `.ipynb`, `.env`-style), is not detected.
- Recommendation: Add a CI job that runs trufflehog (or gitleaks) in full-history mode on push to main; the pre-commit staged-only scan is correct for local use but is not a backstop.

## SEC-02 Pinned dependencies with open CVEs (accepted residual, trusted-input control)
- Severity: Medium
- Effort: L (resolution is the model-stack migration)
- CVE: CVE-2024-3568, CVE-2025-5197, GHSA-8cp5-3rf8-8gfh (deepspeed RCE), plus 29 more on transformers
- Affected files: `DeQA-Score/pyproject.toml:18` (`transformers==4.36.1`), `pyproject.toml:53` (`deepspeed==0.14.5`), `docs/known-vulnerabilities.md`
- Evidence: `docs/known-vulnerabilities.md` lists 32 transformers advisories and the deepspeed RCE `GHSA-8cp5-3rf8-8gfh` (allowlisted in `dependency-review.yml`). Most transformers advisories are ReDoS / unsafe-deserialization triggered by malicious model configs or tokenizer files; this project loads only its own mPLUG-Owl2/Qwen weights from controlled paths. The deepspeed RCE has no fix that keeps `pydantic<2`.
- Recommendation: The compensating control (trusted artifacts only) holds; keep it documented and hold to the 2026-07-27 reassessment. Do not rate these Critical given no untrusted-input path exists.

## SEC-03 eval() on a path component in weight conversion
- Severity: Low
- Effort: S
- CVE:
- Affected files: `DeQA-Score/src/model/convert_mplug_owl2_weight_to_hf.py:118`
- Evidence: `ruff S307` flags `iteration = eval(input_base_path.split('/')[-1].replace('iter_', '').lstrip('0'))`. Input is a local checkpoint directory name controlled by the operator, not a remote input, so injection risk is low, but `eval` on a string is unsafe by construction.
- Recommendation: Replace with `int(...)` or `ast.literal_eval`. The value is meant to be an integer iteration count. Upstream-style file; change only if touched.

## SEC-04 requests calls without timeout
- Severity: Low
- Effort: S
- CVE:
- Affected files: `DeQA-Score/src/evaluate/eval_qbench_mcq.py:32`, `src/evaluate/iqa_eval.py:30`, `src/evaluate/iqa_eval_qwen.py:26`
- Evidence: `ruff S113` flags 3 `requests` calls with no `timeout=`. A hung server stalls the eval indefinitely.
- Recommendation: Add `timeout=` to each call. These are image/URL fetches in eval helpers; a 30s timeout is safe.

## SEC-05 pickle.load on a cache file
- Severity: Low
- Effort: S
- CVE:
- Affected files: `research/vlm_calibration/evaluate_calibration.py:139`
- Evidence: `return pickle.load(f)  # noqa: S301`. The file is a calibration cache the script itself writes, so the input is self-owned, but a tampered cache deserializes arbitrary objects.
- Recommendation: Acceptable for a self-written cache; if the cache is ever shared or downloaded, switch to JSON or numpy `.npz`.

## SEC-06 Broad except Exception handlers (43 sites)
- Severity: Low
- Effort: M
- CVE:
- Affected files: 43 sites across `*.py` (mix of upstream and research scripts)
- Evidence: `git grep -c 'except Exception'` returns 43; 0 of them are `except: pass` swallow blocks, so failures are at least logged or re-raised in the cases sampled. Breadth is the concern, not silent swallowing.
- Recommendation: No bulk change. Narrow exception types opportunistically in new code when a handler is edited.

## Clean areas (one line each)
- No committed secrets: `.env` untracked, `.env.example` placeholders empty, only `"test-key"` literals in `tests/uncertainty/test_vlm_validator.py`.
- No `shell=True` subprocess calls (0 hits); the 6 `S603`/4 `S607` are list-form subprocess calls, the safe form.
- GitHub Actions: all third-party actions are SHA-pinned and `permissions:` are declared per-job (least privilege); the CI/CD report (06) covers the one mutable `@main` reusable-workflow reference.
- The 14 `S311` (non-crypto `random`) hits are ML sampling/seeding, not security-sensitive.
