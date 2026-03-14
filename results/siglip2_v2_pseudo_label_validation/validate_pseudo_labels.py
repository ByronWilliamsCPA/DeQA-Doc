#!/usr/bin/env python3
"""Step 2.1: Validate pseudo-label quality against DIQA val GT.

Runs the pseudo-labeling pipeline on 500 DIQA val images (which have GT
for overall, sharpness, color) and compares pseudo-label MOS to GT MOS.

Metrics computed:
- Per-dimension MAE, RMSE, SRCC, PLCC
- Acceptance rate per tier (auto_accept, low_weight, tier2_trigger, hard_reject)
- MAE broken down by tier
- Distribution of confidence weights
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "DeQA-Score"
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ("overall", "sharpness", "color")


def load_siglip2_val_predictions() -> list[dict]:
    """Load SigLIP2 v1.0 val predictions and convert mu to MOS scale."""
    path = RESULTS_ROOT / "siglip2_diqa5000" / "siglip2_diqa5000_val.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            # Convert from [0,1] to MOS [1,5]: MOS = 1 + 4 * mu
            converted = {
                "image_id": rec["image"],
                "overall_mu": 1.0 + 4.0 * rec["iqa_overall_mu"],
                "overall_sigma_sq": rec["iqa_overall_sigma_sq"],
                "sharpness_mu": 1.0 + 4.0 * rec["iqa_sharpness_mu"],
                "sharpness_sigma_sq": rec["iqa_sharpness_sigma_sq"],
                "color_mu": 1.0 + 4.0 * rec["iqa_color_mu"],
                "color_sigma_sq": rec["iqa_color_sigma_sq"],
            }
            records.append(converted)
    return records


def load_val_gt() -> dict[str, dict[str, float]]:
    """Load DIQA val GT scores. Returns {image_filename: {dim: mos}}."""
    path = PROJECT_ROOT / "Data-DeQA-Score" / "DIQA" / "metas" / "diqa_val.json"
    with open(path) as f:
        samples = json.load(f)

    gt = {}
    for s in samples:
        # Extract filename from path like "DIQA/val/res/val_res_00001.jpg"
        image_name = Path(s["image"]).name
        gt[image_name] = {
            "overall": s["overall"],
            "sharpness": s["sharpness"],
            "color": s.get("color", s.get("color_fidelity", 0.0)),
        }
    return gt


def load_val_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load val embeddings and image names."""
    path = RESULTS_ROOT / "siglip2_diqa5000" / "embeddings" / "val.npz"
    data = np.load(path)
    return data["embeddings"], list(data["image_names"])


def run_validation() -> dict:
    """Run pseudo-label pipeline on val set and compare to GT."""
    from src.uncertainty.cross_validator import CrossValidator
    from src.uncertainty.fusion import UncertaintyFusion
    from src.uncertainty.ood_wrapper import OODDetectorWrapper
    from src.uncertainty.pseudo_label import PseudoLabelPipeline

    # Load data
    print("Loading SigLIP2 val predictions...")
    siglip2_outputs = load_siglip2_val_predictions()
    print(f"  {len(siglip2_outputs)} images")

    print("Loading val GT...")
    gt = load_val_gt()
    print(f"  {len(gt)} images with GT")

    print("Loading val embeddings...")
    embeddings, emb_names = load_val_embeddings()
    print(f"  {embeddings.shape}")

    # Verify alignment
    for i, rec in enumerate(siglip2_outputs):
        assert rec["image_id"] == emb_names[i], (
            f"Mismatch at {i}: {rec['image_id']} vs {emb_names[i]}"
        )

    # Load OOD detector
    ood_path = RESULTS_ROOT / "siglip2_diqa5000" / "ood_detector_v2.npz"
    print(f"Loading OOD detector from {ood_path}...")
    ood_detector = OODDetectorWrapper.from_npz(str(ood_path))

    # No DeQA cross-validator (we don't have DeQA predictions for val)
    cross_validator = CrossValidator({})

    # Create fusion engine with default thresholds
    fusion = UncertaintyFusion()

    # Create pipeline
    pipeline = PseudoLabelPipeline(
        ood_detector=ood_detector,
        cross_validator=cross_validator,
        fusion=fusion,
    )

    # Process batch
    print("Running pseudo-label pipeline...")
    all_samples = pipeline.process_batch(
        siglip2_outputs=siglip2_outputs,
        embeddings=embeddings,
        dimensions=DIMENSIONS,
    )

    # Analyze results
    results = analyze_results(all_samples, gt)
    return results


def analyze_results(
    samples: list,
    gt: dict[str, dict[str, float]],
) -> dict:
    """Compare pseudo-label MOS to GT MOS across dimensions and tiers."""
    results: dict = {"dimensions": {}, "tiers": {}, "overall_stats": {}}

    # Group by dimension
    for dim in DIMENSIONS:
        dim_samples = [s for s in samples if s.dimension == dim]
        pred_mos = []
        gt_mos = []
        tiers = []
        weights = []

        for s in dim_samples:
            if s.image_id not in gt:
                continue
            pred_mos.append(s.mos)
            gt_mos.append(gt[s.image_id][dim])
            tiers.append(s.tier.value)
            weights.append(s.confidence_weight)

        pred_arr = np.array(pred_mos)
        gt_arr = np.array(gt_mos)
        errors = np.abs(pred_arr - gt_arr)

        srcc, srcc_p = stats.spearmanr(pred_arr, gt_arr)
        plcc, plcc_p = stats.pearsonr(pred_arr, gt_arr)

        dim_result = {
            "n_samples": len(pred_mos),
            "mae": float(errors.mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
            "srcc": float(srcc),
            "srcc_p": float(srcc_p),
            "plcc": float(plcc),
            "plcc_p": float(plcc_p),
            "median_ae": float(np.median(errors)),
            "p90_ae": float(np.percentile(errors, 90)),
            "mean_weight": float(np.mean(weights)),
        }

        # Per-tier breakdown
        tier_breakdown = {}
        unique_tiers = sorted(set(tiers))
        for tier in unique_tiers:
            mask = np.array([t == tier for t in tiers])
            tier_errors = errors[mask]
            tier_gt = gt_arr[mask]
            tier_pred = pred_arr[mask]
            tier_weights = np.array(weights)[mask]

            tb = {
                "count": int(mask.sum()),
                "pct": float(mask.sum() / len(mask) * 100),
                "mae": float(tier_errors.mean()),
                "rmse": float(np.sqrt((tier_errors**2).mean())),
                "mean_weight": float(tier_weights.mean()),
            }
            # SRCC only meaningful with enough samples
            if mask.sum() >= 10:
                t_srcc, _ = stats.spearmanr(tier_pred, tier_gt)
                tb["srcc"] = float(t_srcc)

            tier_breakdown[tier] = tb

        dim_result["tiers"] = tier_breakdown
        results["dimensions"][dim] = dim_result

    # Overall tier distribution
    tier_counts: dict[str, int] = {}
    for s in samples:
        tier_counts[s.tier.value] = tier_counts.get(s.tier.value, 0) + 1
    total = len(samples)
    results["tiers"] = {
        t: {"count": c, "pct": round(c / total * 100, 1)}
        for t, c in sorted(tier_counts.items())
    }

    # Acceptance summary
    accepted = [s for s in samples if s.confidence_weight > 0]
    results["overall_stats"] = {
        "total_samples": total,
        "total_images": total // 3,
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / total * 100, 1),
    }

    return results


def print_report(results: dict) -> None:
    """Print a human-readable validation report."""
    print("\n" + "=" * 70)
    print("STEP 2.1: PSEUDO-LABEL VALIDATION REPORT")
    print("Dataset: DIQA val (500 images, 3 dimensions)")
    print("Pipeline: SigLIP2 v1.0 -> OOD + Fusion (no DeQA cross-validator)")
    print("=" * 70)

    # Overall stats
    stats_info = results["overall_stats"]
    print(f"\nTotal samples: {stats_info['total_samples']} "
          f"({stats_info['total_images']} images x 3 dims)")
    print(f"Accepted (weight > 0): {stats_info['accepted']} "
          f"({stats_info['acceptance_rate']}%)")

    # Tier distribution
    print("\n--- Tier Distribution ---")
    for tier, info in results["tiers"].items():
        print(f"  {tier:20s}: {info['count']:4d} ({info['pct']:5.1f}%)")

    # Per-dimension results
    for dim in DIMENSIONS:
        dr = results["dimensions"][dim]
        print(f"\n--- {dim.upper()} (n={dr['n_samples']}) ---")
        print(f"  MAE:    {dr['mae']:.4f}  (median: {dr['median_ae']:.4f}, "
              f"p90: {dr['p90_ae']:.4f})")
        print(f"  RMSE:   {dr['rmse']:.4f}")
        print(f"  SRCC:   {dr['srcc']:.4f}  (p={dr['srcc_p']:.2e})")
        print(f"  PLCC:   {dr['plcc']:.4f}  (p={dr['plcc_p']:.2e})")
        print(f"  Mean confidence weight: {dr['mean_weight']:.3f}")

        print("\n  Per-tier breakdown:")
        print(f"  {'Tier':<20s} {'Count':>6s} {'%':>6s} {'MAE':>8s} "
              f"{'RMSE':>8s} {'SRCC':>8s} {'Weight':>8s}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for tier, tb in dr["tiers"].items():
            srcc_str = f"{tb['srcc']:.4f}" if "srcc" in tb else "   n/a"
            print(f"  {tier:<20s} {tb['count']:6d} {tb['pct']:5.1f}% "
                  f"{tb['mae']:8.4f} {tb['rmse']:8.4f} {srcc_str:>8s} "
                  f"{tb['mean_weight']:8.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = run_validation()

    # Save JSON results
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "validation_results.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")

    print_report(results)
