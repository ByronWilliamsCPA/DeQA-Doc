#!/usr/bin/env python3
"""Step 6: Evaluate VLM quality predictions on OCR-IQA distorted images.

Runs Gemini 3 Flash Preview and GPT-4.1 via OpenRouter on all 1,200
distorted document images, then computes SRCC/PLCC between VLM quality
predictions and:
  (a) DeQA MOS scores
  (b) Ground-truth distortion overall_quality
  (c) OCR CER per engine

Usage:
    python -m research.ocr_iqa_correlation.scripts.06_vlm_eval

    # Single model:
    python -m research.ocr_iqa_correlation.scripts.06_vlm_eval \
        --model google/gemini-3-flash-preview

    # Resume (auto):
    python -m research.ocr_iqa_correlation.scripts.06_vlm_eval

    # Dry run:
    python -m research.ocr_iqa_correlation.scripts.06_vlm_eval --limit 10

    # Metrics only (from existing checkpoints):
    python -m research.ocr_iqa_correlation.scripts.06_vlm_eval --metrics-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.image_utils import encode_image_base64
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt
from results.vlm_teacher_eval.response_parser import parse_iqa_response

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_JSONL = DATA_DIR / "dataset.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "vlm_checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS: list[tuple[str, str]] = [
    ("google/gemini-3-flash-preview", "Value"),
    ("openai/gpt-4.1", "Strong"),
]

RATE_LIMIT_S = 0.3
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ImageResult:
    """Result from a single VLM evaluation."""

    model_id: str
    image_id: str
    tier: str
    overall: float | None
    sharpness: float | None
    color_fidelity: float | None
    reasoning: str
    raw_response: str
    latency_ms: int
    error: str


@dataclass(frozen=True)
class MasterRecord:
    """Parsed record from dataset.jsonl."""

    image_id: str
    tier: str
    image_path: str
    actual_overall_quality: float
    deqa_mos: float | None
    ocr_cer: dict[str, float]  # engine -> CER


# --- Data Loading ---


def load_env() -> None:
    """Load .env file from repo root."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_master_dataset() -> list[MasterRecord]:
    """Load the OCR-IQA master dataset."""
    records: list[MasterRecord] = []
    with DATASET_JSONL.open() as f:
        for line in f:
            raw = json.loads(line)
            ocr_cer = {}
            for engine, data in raw.get("ocr", {}).items():
                cer = data.get("cer")
                if cer is not None:
                    ocr_cer[engine] = cer
            records.append(
                MasterRecord(
                    image_id=raw["image_id"],
                    tier=raw["tier"],
                    image_path=raw["image_path"],
                    actual_overall_quality=raw.get("actual_overall_quality", 1.0),
                    deqa_mos=raw.get("deqa_mos"),
                    ocr_cer=ocr_cer,
                )
            )
    return records


# --- API Calls ---


def rate_image_openrouter(
    model_id: str,
    image_b64: str,
    media_type: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 3,
) -> tuple[str, int, str]:
    """Rate an image via OpenRouter with retry logic.

    Returns:
        Tuple of (raw_text, latency_ms, error).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    for attempt in range(max_retries):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_id,
                temperature=0.0,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
            )
            latency_ms = int((time.time() - start) * 1000)
            raw_text = response.choices[0].message.content or ""
            return raw_text, latency_ms, ""
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            err_str = str(exc)

            if attempt < max_retries - 1 and any(
                code in err_str for code in ("429", "500", "502", "503", "timeout")
            ):
                wait = 2 ** (attempt + 1)
                logger.warning("RETRY(%d) in %ds: %s", attempt + 1, wait, err_str[:80])
                time.sleep(wait)
                continue

            return "", latency_ms, err_str

    return "", 0, "max retries exceeded"


# --- Checkpoint Management ---


def checkpoint_path(model_id: str) -> Path:
    """Get checkpoint file path for a model."""
    safe_name = model_id.replace("/", "__")
    return CHECKPOINT_DIR / f"{safe_name}.jsonl"


def load_checkpoint(model_id: str) -> dict[str, dict[str, Any]]:
    """Load existing checkpoint results for a model.

    Returns:
        Dict mapping '{image_id}_{tier}' to result dict.
    """
    cp = checkpoint_path(model_id)
    results: dict[str, dict[str, Any]] = {}
    if not cp.exists():
        return results

    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not item.get("error"):
                key = f"{item['image_id']}_{item['tier']}"
                results[key] = item
        except json.JSONDecodeError:
            continue

    return results


def append_checkpoint(model_id: str, result: ImageResult) -> None:
    """Append a single result to the model's checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp = checkpoint_path(model_id)
    with cp.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# --- Metrics Computation ---


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    metric_fn: Any,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute metric with bootstrapped 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(pred)
    point = float(metric_fn(pred, true))

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            val = float(metric_fn(pred[idx], true[idx]))
            if not np.isnan(val):
                boot_vals.append(val)
        except (ValueError, FloatingPointError):
            continue

    if len(boot_vals) < 30:
        return point, float("nan"), float("nan")

    ci_lower = float(np.percentile(boot_vals, 2.5))
    ci_upper = float(np.percentile(boot_vals, 97.5))
    return point, ci_lower, ci_upper


def srcc_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation."""
    return float(stats.spearmanr(pred, true).statistic)


def plcc_fn(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson linear correlation."""
    return float(stats.pearsonr(pred, true).statistic)


def compute_metrics(
    results: list[ImageResult],
    records_lookup: dict[str, MasterRecord],
) -> dict[str, Any]:
    """Compute all correlation metrics for a model's results.

    Computes SRCC/PLCC between VLM overall prediction and:
    - DeQA MOS
    - Ground-truth distortion overall_quality
    - OCR CER per engine (negative correlation expected)
    Plus per-dimension metrics (overall, sharpness, color_fidelity) vs DeQA.
    """
    # Pair results with master records
    pairs: list[tuple[ImageResult, MasterRecord]] = []
    for r in results:
        if r.error or r.overall is None:
            continue
        key = f"{r.image_id}_{r.tier}"
        rec = records_lookup.get(key)
        if rec:
            pairs.append((r, rec))

    n = len(pairs)
    success_rate = n / len(results) if results else 0.0

    metrics: dict[str, Any] = {
        "num_samples": n,
        "total_images": len(results),
        "success_rate": round(success_rate, 4),
    }

    if n < 30:
        logger.warning("Only %d valid pairs, skipping metrics", n)
        return metrics

    # --- VLM overall vs DeQA MOS ---
    vlm_overall = np.array([r.overall for r, _ in pairs])
    deqa_mos = np.array([rec.deqa_mos for _, rec in pairs if rec.deqa_mos is not None])

    deqa_pairs = [(r, rec) for r, rec in pairs if rec.deqa_mos is not None]
    if len(deqa_pairs) >= 30:
        vlm_o = np.array([r.overall for r, _ in deqa_pairs])
        d_mos = np.array([rec.deqa_mos for _, rec in deqa_pairs])

        srcc, srcc_lo, srcc_hi = bootstrap_ci(vlm_o, d_mos, srcc_fn)
        plcc, plcc_lo, plcc_hi = bootstrap_ci(vlm_o, d_mos, plcc_fn)
        metrics["vs_deqa_mos"] = {
            "srcc": round(srcc, 4),
            "srcc_ci": [round(srcc_lo, 4), round(srcc_hi, 4)],
            "plcc": round(plcc, 4),
            "plcc_ci": [round(plcc_lo, 4), round(plcc_hi, 4)],
            "n": len(deqa_pairs),
        }

    # --- VLM overall vs distortion overall_quality ---
    vlm_all = np.array([r.overall for r, _ in pairs])
    gt_quality = np.array([rec.actual_overall_quality for _, rec in pairs])

    srcc, srcc_lo, srcc_hi = bootstrap_ci(vlm_all, gt_quality, srcc_fn)
    plcc, plcc_lo, plcc_hi = bootstrap_ci(vlm_all, gt_quality, plcc_fn)
    metrics["vs_distortion_quality"] = {
        "srcc": round(srcc, 4),
        "srcc_ci": [round(srcc_lo, 4), round(srcc_hi, 4)],
        "plcc": round(plcc, 4),
        "plcc_ci": [round(plcc_lo, 4), round(plcc_hi, 4)],
        "n": n,
    }

    # --- VLM overall vs OCR CER per engine (negative correlation expected) ---
    engines = sorted({eng for _, rec in pairs for eng in rec.ocr_cer})
    cer_metrics: dict[str, Any] = {}
    for engine in engines:
        eng_pairs = [
            (r, rec) for r, rec in pairs if engine in rec.ocr_cer
        ]
        if len(eng_pairs) < 30:
            continue

        vlm_e = np.array([r.overall for r, _ in eng_pairs])
        cer_e = np.array([rec.ocr_cer[engine] for _, rec in eng_pairs])

        srcc, srcc_lo, srcc_hi = bootstrap_ci(vlm_e, cer_e, srcc_fn)
        plcc, plcc_lo, plcc_hi = bootstrap_ci(vlm_e, cer_e, plcc_fn)
        cer_metrics[engine] = {
            "srcc": round(srcc, 4),
            "srcc_ci": [round(srcc_lo, 4), round(srcc_hi, 4)],
            "plcc": round(plcc, 4),
            "plcc_ci": [round(plcc_lo, 4), round(plcc_hi, 4)],
            "n": len(eng_pairs),
        }

    metrics["vs_ocr_cer"] = cer_metrics

    # --- Per-tier breakdown ---
    tier_counts: dict[str, dict[str, float]] = {}
    for r, rec in pairs:
        tier = rec.tier
        if tier not in tier_counts:
            tier_counts[tier] = {"sum_overall": 0.0, "n": 0}
        tier_counts[tier]["sum_overall"] += r.overall  # type: ignore[operator]
        tier_counts[tier]["n"] += 1

    tier_means: dict[str, dict[str, float]] = {}
    for tier, data in tier_counts.items():
        tier_means[tier] = {
            "mean_vlm_overall": round(data["sum_overall"] / data["n"], 4),
            "n": int(data["n"]),
        }
    metrics["per_tier"] = tier_means

    # --- Latency stats ---
    latencies = [r.latency_ms for r in results if not r.error]
    if latencies:
        metrics["latency"] = {
            "mean_ms": round(float(np.mean(latencies)), 1),
            "p50_ms": round(float(np.median(latencies)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        }

    return metrics


# --- Evaluation ---


def evaluate_model(
    model_id: str,
    records: list[MasterRecord],
    api_key: str,
    system_prompt: str,
    limit: int | None = None,
    max_pixels: int = 1024,
) -> list[ImageResult]:
    """Evaluate a single model on all images with resume support."""
    existing = load_checkpoint(model_id)
    images_to_eval = records[:limit] if limit else records
    total = len(images_to_eval)

    results: list[ImageResult] = []
    skipped = 0

    for idx, rec in enumerate(images_to_eval):
        key = f"{rec.image_id}_{rec.tier}"

        # Resume from checkpoint
        if key in existing:
            r = ImageResult(**existing[key])
            results.append(r)
            skipped += 1
            continue

        if skipped and idx == skipped:
            logger.info("Resumed from checkpoint (%d cached)", skipped)

        img_path = rec.image_path
        if not Path(img_path).exists():
            logger.warning("[%d/%d] MISSING: %s", idx + 1, total, img_path)
            continue

        print(
            f"  [{idx + 1}/{total}] {rec.image_id}/{rec.tier}",
            end=" ",
            flush=True,
        )

        img_b64, media_type = encode_image_base64(img_path, max_pixels=max_pixels)

        raw_text, latency_ms, error = rate_image_openrouter(
            model_id=model_id,
            image_b64=img_b64,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=USER_PROMPT,
            api_key=api_key,
        )

        if error:
            r = ImageResult(
                model_id=model_id,
                image_id=rec.image_id,
                tier=rec.tier,
                overall=None,
                sharpness=None,
                color_fidelity=None,
                reasoning="",
                raw_response="",
                latency_ms=latency_ms,
                error=error,
            )
            print(f"ERROR ({latency_ms}ms): {error[:80]}")
        else:
            try:
                rating = parse_iqa_response(raw_text)
                r = ImageResult(
                    model_id=model_id,
                    image_id=rec.image_id,
                    tier=rec.tier,
                    overall=rating.overall,
                    sharpness=rating.sharpness,
                    color_fidelity=rating.color_fidelity,
                    reasoning=rating.reasoning,
                    raw_response=raw_text,
                    latency_ms=latency_ms,
                    error="",
                )
                print(
                    f"O={rating.overall:.1f} S={rating.sharpness:.1f} "
                    f"C={rating.color_fidelity:.1f} ({latency_ms}ms)"
                )
            except ValueError as exc:
                r = ImageResult(
                    model_id=model_id,
                    image_id=rec.image_id,
                    tier=rec.tier,
                    overall=None,
                    sharpness=None,
                    color_fidelity=None,
                    reasoning="",
                    raw_response=raw_text,
                    latency_ms=latency_ms,
                    error=f"Parse error: {exc}",
                )
                print(f"PARSE ({latency_ms}ms): {exc!s:.60s}")

        results.append(r)
        append_checkpoint(model_id, r)
        time.sleep(RATE_LIMIT_S)

    if skipped == total:
        logger.info("All %d images cached from checkpoint", total)

    return results


def main() -> None:
    """Run VLM evaluation on OCR-IQA dataset."""
    parser = argparse.ArgumentParser(
        description="VLM evaluation on OCR-IQA distorted images"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Evaluate only this model (e.g. google/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N images (for dry run)",
    )
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Compute metrics from existing checkpoints only",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=1024,
        help="Max longest-edge pixels for resize (0 = no resize). Default: 1024",
    )
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.metrics_only:
        logger.error("OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    # Load master dataset
    logger.info("Loading master dataset from %s", DATASET_JSONL)
    records = load_master_dataset()
    records_lookup = {f"{r.image_id}_{r.tier}": r for r in records}
    logger.info("Loaded %d records (%d unique images)", len(records),
                len({r.image_id for r in records}))

    system_prompt = build_system_prompt()

    # Select models
    models = MODELS
    if args.model:
        models = [(m, t) for m, t in MODELS if m == args.model]
        if not models:
            # Allow arbitrary model IDs
            models = [(args.model, "Custom")]

    all_metrics: dict[str, dict[str, Any]] = {}

    for model_id, tier in models:
        logger.info("=" * 70)
        logger.info("Evaluating: %s (tier=%s)", model_id, tier)
        logger.info("=" * 70)

        if args.metrics_only:
            existing = load_checkpoint(model_id)
            if not existing:
                logger.warning("No checkpoint found for %s, skipping", model_id)
                continue
            results = [ImageResult(**v) for v in existing.values()]
            logger.info("Loaded %d results from checkpoint", len(results))
        else:
            results = evaluate_model(
                model_id=model_id,
                records=records,
                api_key=api_key,
                system_prompt=system_prompt,
                limit=args.limit,
                max_pixels=args.max_pixels,
            )

        ok = sum(1 for r in results if not r.error)
        err = sum(1 for r in results if r.error)
        logger.info("Results: %d success, %d errors", ok, err)

        model_metrics = compute_metrics(results, records_lookup)
        all_metrics[model_id] = model_metrics

        # Print summary
        if "vs_deqa_mos" in model_metrics:
            m = model_metrics["vs_deqa_mos"]
            logger.info(
                "  vs DeQA MOS: SRCC=%.4f %s, PLCC=%.4f %s (n=%d)",
                m["srcc"], m["srcc_ci"], m["plcc"], m["plcc_ci"], m["n"],
            )

        if "vs_distortion_quality" in model_metrics:
            m = model_metrics["vs_distortion_quality"]
            logger.info(
                "  vs Distortion Quality: SRCC=%.4f %s, PLCC=%.4f %s (n=%d)",
                m["srcc"], m["srcc_ci"], m["plcc"], m["plcc_ci"], m["n"],
            )

        if "vs_ocr_cer" in model_metrics:
            for engine, m in model_metrics["vs_ocr_cer"].items():
                logger.info(
                    "  vs CER(%s): SRCC=%.4f, PLCC=%.4f (n=%d)",
                    engine, m["srcc"], m["plcc"], m["n"],
                )

        if "per_tier" in model_metrics:
            logger.info("  Per-tier mean VLM overall:")
            for t in ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]:
                if t in model_metrics["per_tier"]:
                    d = model_metrics["per_tier"][t]
                    logger.info("    %s: %.3f (n=%d)", t, d["mean_vlm_overall"], d["n"])

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "vlm_eval_metrics.json"
    report_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Metrics saved: %s", report_path)

    # Print comparison leaderboard
    print(f"\n{'=' * 80}")
    print("VLM Quality Prediction on OCR-IQA Dataset (n=1200)")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<35s} {'vs DeQA':>8s} {'vs GT_Q':>8s} "
        f"{'vs CER_t':>8s} {'vs CER_g':>8s} {'n':>5s}"
    )
    print("-" * 80)

    for model_id, m in all_metrics.items():
        deqa_srcc = m.get("vs_deqa_mos", {}).get("srcc", 0)
        gt_srcc = m.get("vs_distortion_quality", {}).get("srcc", 0)
        cer_t = m.get("vs_ocr_cer", {}).get("tesseract", {}).get("srcc", 0)
        cer_g = m.get("vs_ocr_cer", {}).get("gcloud_vision", {}).get("srcc", 0)
        n = m.get("num_samples", 0)
        print(
            f"{model_id:<35s} {deqa_srcc:>8.4f} {gt_srcc:>8.4f} "
            f"{cer_t:>8.4f} {cer_g:>8.4f} {n:>5d}"
        )

    print("-" * 80)
    print("DeQA MOS correlation = VLM agrees with specialist model")
    print("GT Quality = VLM detects synthetic distortion levels")
    print("CER = VLM quality predicts OCR accuracy (negative = correct)")


if __name__ == "__main__":
    main()
