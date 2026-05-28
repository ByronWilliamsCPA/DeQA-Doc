"""Step 3: Compute MOS-CER/WER correlation on SmartDoc-QA.

Joins DeQA inference results with pre-computed OCR accuracy files and computes
Spearman/Pearson correlations, matching the analysis in paper 06.

Usage:
    cd DeQA-Score
    .venv/bin/python ../research/smartdoc_qa_ocr_analysis/03_compute_correlation.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ERROR_RATES_PATH = SCRIPT_DIR / "outputs" / "smartdoc_qa_error_rates.jsonl"
DEQA_RESULTS_DIR = SCRIPT_DIR / "data" / "deqa_results"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# DeQA level convention: [excellent, good, fair, poor, bad] = [5, 4, 3, 2, 1]
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]
LEVEL_SCORES = np.array([5.0, 4.0, 3.0, 2.0, 1.0])


def load_deqa_results() -> dict[str, dict]:
    """Load DeQA inference results keyed by relative image path.

    The iqa_eval.py output JSONL has fields: image, logits, probs.
    Image paths are like: Samsung_phone/Images/M_Img_Android_D10_L2_r35_a-10_b-5.jpg

    Returns:
        Dict mapping filename stem (without phone prefix) to
        {mos, probs, phone} dict.
    """
    results: dict[str, dict] = {}

    # Find result file(s) in deqa_results directory
    result_files = list(DEQA_RESULTS_DIR.glob("*.json*"))
    if not result_files:
        raise FileNotFoundError(
            f"No DeQA results found in {DEQA_RESULTS_DIR}. "
            "Run 02_run_deqa.sh first."
        )

    for result_file in result_files:
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                image_path = record["image"]
                # Extract phone and filename
                # e.g. "Samsung_phone/Images/M_Img_Android_D10_L2_r35_a-10_b-5.jpg"
                parts = image_path.split("/")
                phone = parts[0].replace("_phone", "")  # "Samsung" or "Nokia"
                img_filename = parts[-1]  # full filename with .jpg
                stem = img_filename.removesuffix(".jpg")

                # Compute MOS from probabilities
                if "probs" in record:
                    probs = record["probs"]
                    prob_array = np.array(
                        [probs.get(name, 0.0) for name in LEVEL_NAMES]
                    )
                    mos = float(prob_array @ LEVEL_SCORES)
                else:
                    # Fallback to logits -> softmax
                    logits = record["logits"]
                    logit_array = np.array(
                        [logits.get(name, 0.0) for name in LEVEL_NAMES]
                    )
                    exp_logits = np.exp(logit_array - np.max(logit_array))
                    prob_array = exp_logits / exp_logits.sum()
                    mos = float(prob_array @ LEVEL_SCORES)

                key = f"{phone}/{stem}"
                results[key] = {
                    "mos": mos,
                    "probs": prob_array.tolist(),
                    "phone": phone,
                }

    return results


def load_error_rates() -> dict[str, list[dict]]:
    """Load OCR error rates keyed by phone/filename.

    Returns:
        Dict mapping "Phone/filename_stem" to list of metric records.
    """
    records: dict[str, list[dict]] = defaultdict(list)

    with open(ERROR_RATES_PATH) as f:
        for line in f:
            record = json.loads(line)
            phone = record["phone"]
            # Filename in error rates uses the stem without phone prefix
            # Reconstruct the key to match DeQA results
            filename = record["filename"]
            key = f"{phone}/{filename}"
            records[key].append(record)

    return records


def compute_correlations(
    mos_values: np.ndarray,
    error_values: np.ndarray,
    label: str,
) -> dict[str, float]:
    """Compute SRCC and PLCC with p-values.

    Args:
        mos_values: Array of DeQA MOS scores.
        error_values: Array of error rates (CER or WER).
        label: Description for logging.

    Returns:
        Dict with srcc, srcc_p, plcc, plcc_p, n.
    """
    mask = np.isfinite(mos_values) & np.isfinite(error_values)
    mos = mos_values[mask]
    err = error_values[mask]

    if len(mos) < 3:
        return {"srcc": float("nan"), "srcc_p": 1.0, "plcc": float("nan"), "plcc_p": 1.0, "n": len(mos)}

    srcc, srcc_p = stats.spearmanr(mos, err)
    plcc, plcc_p = stats.pearsonr(mos, err)

    return {
        "srcc": float(srcc),
        "srcc_p": float(srcc_p),
        "plcc": float(plcc),
        "plcc_p": float(plcc_p),
        "n": int(len(mos)),
    }


def main() -> None:
    """Join DeQA scores with OCR error rates and compute correlations."""
    print("Loading DeQA results...")
    deqa = load_deqa_results()
    print(f"  Loaded {len(deqa)} DeQA scores")

    print("Loading OCR error rates...")
    error_rates = load_error_rates()
    unique_images = set(error_rates.keys())
    print(f"  Loaded error rates for {len(unique_images)} unique image keys")

    # Match DeQA results to error rates
    matched_keys = set(deqa.keys()) & unique_images
    print(f"  Matched: {len(matched_keys)} images with both DeQA + OCR data")

    if not matched_keys:
        print("\nERROR: No matches found. Check key formats.")
        print(f"  Sample DeQA keys: {list(deqa.keys())[:5]}")
        print(f"  Sample error rate keys: {list(error_rates.keys())[:5]}")
        return

    # Build joined dataset
    joined: list[dict] = []
    for key in sorted(matched_keys):
        deqa_record = deqa[key]
        for err_record in error_rates[key]:
            joined.append({
                "key": key,
                "phone": deqa_record["phone"],
                "engine": err_record["engine"],
                "metric_type": err_record["metric_type"],
                "mos": deqa_record["mos"],
                "error_rate": err_record["error_rate"],
                "accuracy_pct": err_record["accuracy_pct"],
                "document_id": err_record["document_id"],
                "distortion_type": err_record["distortion_type"],
                "lighting": err_record["lighting"],
                "blur_type": err_record["blur_type"],
                "category": err_record["category"],
            })

    print(f"\nJoined dataset: {len(joined)} records")

    # Save joined dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joined_path = OUTPUT_DIR / "smartdoc_qa_joined.jsonl"
    with open(joined_path, "w") as f:
        for record in joined:
            f.write(json.dumps(record) + "\n")
    print(f"Saved joined dataset to {joined_path}")

    # ── Correlation Analysis ─────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("  CORRELATION ANALYSIS: DeQA MOS vs OCR Error Rate")
    print("=" * 80)

    # Overall by engine x metric
    print(f"\n{'Engine':<15} {'Metric':<6} {'N':>6} {'SRCC':>8} {'p(SRCC)':>12} "
          f"{'PLCC':>8} {'p(PLCC)':>12}")
    print("-" * 75)

    results_summary: list[dict] = []

    for engine in ["FineReader", "Tesseract"]:
        for metric in ["CER", "WER"]:
            subset = [r for r in joined if r["engine"] == engine and r["metric_type"] == metric]
            if not subset:
                continue
            mos = np.array([r["mos"] for r in subset])
            err = np.array([r["error_rate"] for r in subset])
            corr = compute_correlations(mos, err, f"{engine}/{metric}")
            print(
                f"{engine:<15} {metric:<6} {corr['n']:>6} {corr['srcc']:>8.4f} "
                f"{corr['srcc_p']:>12.2e} {corr['plcc']:>8.4f} {corr['plcc_p']:>12.2e}"
            )
            results_summary.append({"engine": engine, "metric": metric, **corr})

    # By phone
    print(f"\n{'Phone':<15} {'Engine':<15} {'Metric':<6} {'N':>6} {'SRCC':>8} {'PLCC':>8}")
    print("-" * 65)

    for phone in ["Nokia", "Samsung"]:
        for engine in ["FineReader", "Tesseract"]:
            for metric in ["CER"]:
                subset = [
                    r for r in joined
                    if r["phone"] == phone and r["engine"] == engine and r["metric_type"] == metric
                ]
                if not subset:
                    continue
                mos = np.array([r["mos"] for r in subset])
                err = np.array([r["error_rate"] for r in subset])
                corr = compute_correlations(mos, err, f"{phone}/{engine}/{metric}")
                print(
                    f"{phone:<15} {engine:<15} {metric:<6} {corr['n']:>6} "
                    f"{corr['srcc']:>8.4f} {corr['plcc']:>8.4f}"
                )

    # By distortion type
    print(f"\n{'Distortion':<15} {'Engine':<15} {'N':>6} {'SRCC':>8} {'PLCC':>8}")
    print("-" * 55)

    for dist_type in ["single", "multiple"]:
        for engine in ["FineReader", "Tesseract"]:
            subset = [
                r for r in joined
                if r["distortion_type"] == dist_type
                and r["engine"] == engine
                and r["metric_type"] == "CER"
            ]
            if not subset:
                continue
            mos = np.array([r["mos"] for r in subset])
            err = np.array([r["error_rate"] for r in subset])
            corr = compute_correlations(mos, err, f"{dist_type}/{engine}")
            print(
                f"{dist_type:<15} {engine:<15} {corr['n']:>6} "
                f"{corr['srcc']:>8.4f} {corr['plcc']:>8.4f}"
            )

    # By document category
    print(f"\n{'Category':<18} {'Engine':<15} {'N':>6} {'SRCC':>8} {'PLCC':>8}")
    print("-" * 58)

    for category in ["modern", "administrative", "receipts"]:
        for engine in ["FineReader", "Tesseract"]:
            subset = [
                r for r in joined
                if r["category"] == category
                and r["engine"] == engine
                and r["metric_type"] == "CER"
            ]
            if not subset:
                continue
            mos = np.array([r["mos"] for r in subset])
            err = np.array([r["error_rate"] for r in subset])
            corr = compute_correlations(mos, err, f"{category}/{engine}")
            print(
                f"{category:<18} {engine:<15} {corr['n']:>6} "
                f"{corr['srcc']:>8.4f} {corr['plcc']:>8.4f}"
            )

    # By blur type (CER only, both engines combined)
    print(f"\n{'Blur Type':<12} {'N':>6} {'Mean MOS':>10} {'Mean CER':>10} {'SRCC':>8}")
    print("-" * 52)

    blur_types = sorted({r["blur_type"] for r in joined})
    for blur in blur_types:
        subset = [r for r in joined if r["blur_type"] == blur and r["metric_type"] == "CER"]
        if len(subset) < 10:
            continue
        mos = np.array([r["mos"] for r in subset])
        err = np.array([r["error_rate"] for r in subset])
        corr = compute_correlations(mos, err, f"blur={blur}")
        print(
            f"{blur:<12} {corr['n']:>6} {np.mean(mos):>10.3f} {np.mean(err):>10.4f} "
            f"{corr['srcc']:>8.4f}"
        )

    # Save summary
    summary_path = OUTPUT_DIR / "smartdoc_qa_correlation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved correlation summary to {summary_path}")


if __name__ == "__main__":
    main()
