#!/usr/bin/env python3
"""Validate OOD detector checkpoint against known calibration data.

Re-scores stored embeddings and compares distances to calibration values
from the OOD fitting run. Use this to diagnose checkpoint mismatch issues.

Usage:
    python scripts/validate_ood_checkpoint.py \
        --ood-params /path/to/ood_params_4400.npz \
        --test-embeddings /path/to/diqa5000_test_all.npy \
        [--threshold 46.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OOD detector checkpoint")
    parser.add_argument(
        "--ood-params",
        type=str,
        required=True,
        help="Path to ood_params .npz file (mean + precision_matrix)",
    )
    parser.add_argument(
        "--test-embeddings",
        type=str,
        default=None,
        help="Path to test embeddings .npy file for calibration check",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=46.0,
        help="OOD threshold (test p95). Default: 46.0",
    )
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from src.uncertainty.ood_wrapper import OODDetectorWrapper

    # Load detector
    print(f"Loading OOD params from: {args.ood_params}")
    detector = OODDetectorWrapper.from_npz(args.ood_params, threshold=args.threshold)
    print(f"  Mean shape: {detector.mean.shape}")
    print(f"  Precision matrix shape: {detector.precision_matrix.shape}")
    print(f"  Threshold: {detector.threshold}")

    # Check stored calibration distances
    data = np.load(args.ood_params)
    print(f"\n  Keys in .npz: {list(data.keys())}")
    if "calibration_distances" in data:
        cal_dists = data["calibration_distances"]
        print(f"  Calibration distances: N={len(cal_dists)}")
        print(f"    median = {np.median(cal_dists):.2f}")
        print(f"    mean   = {np.mean(cal_dists):.2f}")
        print(f"    p95    = {np.percentile(cal_dists, 95):.2f}")
        print(f"    p99    = {np.percentile(cal_dists, 99):.2f}")
        print(f"    max    = {np.max(cal_dists):.2f}")
    if "threshold" in data:
        print(f"  Stored threshold: {float(data['threshold'])}")

    # Score test embeddings if provided
    if args.test_embeddings:
        print(f"\nScoring test embeddings from: {args.test_embeddings}")
        embeddings = np.load(args.test_embeddings)
        print(f"  Shape: {embeddings.shape}")

        results = detector.score_batch(embeddings)
        distances = np.array([r.mahalanobis_distance for r in results])

        print(f"\n  Test set distances (N={len(distances)}):")
        print(f"    median = {np.median(distances):.2f}")
        print(f"    mean   = {np.mean(distances):.2f}")
        print(f"    p5     = {np.percentile(distances, 5):.2f}")
        print(f"    p25    = {np.percentile(distances, 25):.2f}")
        print(f"    p75    = {np.percentile(distances, 75):.2f}")
        print(f"    p95    = {np.percentile(distances, 95):.2f}")
        print(f"    p99    = {np.percentile(distances, 99):.2f}")
        print(f"    max    = {np.max(distances):.2f}")

        n_ood = sum(1 for r in results if r.is_ood)
        print(
            f"\n  OOD flagged: {n_ood}/{len(results)} ({n_ood / len(results) * 100:.1f}%)"
        )
        print(f"  Expected: ~5% at threshold={args.threshold}")

        # Compare to stored calibration if available
        if "calibration_distances" in data:
            cal_median = np.median(data["calibration_distances"])
            test_median = np.median(distances)
            shift = test_median - cal_median
            print(f"\n  Median shift (test - calibration): {shift:.2f}")
            if abs(shift) > 5.0:
                print("  WARNING: Large shift detected. Possible checkpoint mismatch.")
            else:
                print("  OK: Shift within expected range.")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
