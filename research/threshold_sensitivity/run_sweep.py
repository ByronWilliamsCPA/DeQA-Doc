"""U-3: OOD Threshold Sensitivity Analysis.

Sweeps uncertainty thresholds for the pseudo-labeling pipeline and reports
how tier proportions (AUTO_ACCEPT/LOW_WEIGHT/TIER2_TRIGGER/HARD_REJECT) shift.

Part 1: Tier-1 fusion threshold sweep (d_M, σ², entropy)
Part 2: Tier-2 VLM veto threshold sweep (disagreement threshold)

Run:
    cd DeQA-Score && PYTHONPATH=./:$PYTHONPATH \
        .venv/bin/python -m research.threshold_sensitivity.run_sweep
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import types

import numpy as np

# Add DeQA-Score to path and patch src module to avoid CUDA-dependent imports
_deqa_root = Path(__file__).resolve().parent.parent.parent / "DeQA-Score"
if str(_deqa_root) not in sys.path:
    sys.path.insert(0, str(_deqa_root))

# Patch: src/__init__.py imports MPLUGOwl2LlamaForCausalLM which needs CUDA.
# Create a lightweight src module that only exposes the subpackage path.
if "src" not in sys.modules:
    _src_mod = types.ModuleType("src")
    _src_mod.__path__ = [str(_deqa_root / "src")]
    sys.modules["src"] = _src_mod

from src.uncertainty.discrete_metrics import discrete_entropy  # noqa: E402
from src.uncertainty.gaussian_to_discrete import (  # noqa: E402
    siglip2_output_to_level_probs,
)
from src.uncertainty.ood_wrapper import OODDetectorWrapper  # noqa: E402

logger = logging.getLogger(__name__)

# Dimensions in our pipeline
DIMENSIONS = ["overall", "sharpness", "color"]

# Tier enum values (matching fusion.py AcceptanceTier)
TIER_AUTO_ACCEPT = 0
TIER_LOW_WEIGHT = 1
TIER_TIER2_TRIGGER = 2
TIER_HARD_REJECT = 3
TIER_NAMES = ["AUTO_ACCEPT", "LOW_WEIGHT", "TIER2_TRIGGER", "HARD_REJECT"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SweepData:
    """All pre-computed data needed for the sweep."""

    image_ids: list[str]  # (N,) image filenames
    splits: list[str]  # (N,) "train"/"val"/"test"
    dm: np.ndarray  # (N,) Mahalanobis distances
    sigma_sq: np.ndarray  # (N, 3) MOS-scale σ² per dimension
    entropy: np.ndarray  # (N, 3) entropy per dimension
    mu_mos: np.ndarray  # (N, 3) MOS-scale μ per dimension
    n_train_val: int  # count of train+val images
    split_mask_train_val: np.ndarray  # (N,) bool mask for train+val
    split_mask_test: np.ndarray  # (N,) bool mask for test


@dataclass
class ThresholdConfig:
    """A single threshold configuration for the sweep."""

    name: str
    dm_ood: float
    dm_reject: float
    sigma_sq_auto: float
    sigma_sq_low: float
    entropy_auto: float
    entropy_low: float


@dataclass
class TierStats:
    """Tier distribution statistics for one config × split × dimension."""

    n_total: int
    n_auto_accept: int
    n_low_weight: int
    n_tier2_trigger: int
    n_hard_reject: int
    pct_auto_accept: float
    pct_low_weight: float
    pct_tier2_trigger: float
    pct_hard_reject: float
    mean_weight: float
    effective_n: float  # sum of weights


@dataclass
class VetoSweepResult:
    """Veto sweep result for one threshold × model × dimension."""

    threshold: float
    model_id: str
    dimension: str
    n_total: int
    n_vetoed: int
    veto_rate: float
    mean_disagreement: float
    median_disagreement: float


# ---------------------------------------------------------------------------
# Step 1: Data Loading
# ---------------------------------------------------------------------------


def _load_siglip2_jsonl(path: Path) -> list[dict]:
    """Load a SigLIP2 predictions JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_data(
    siglip2_dir: Path,
    embedding_dir: Path,
    ood_npz_path: Path,
) -> SweepData:
    """Load all pre-computed artifacts for the sweep.

    Args:
        siglip2_dir: Directory containing siglip2_diqa5000_{split}.jsonl files.
        embedding_dir: Directory containing {split}.npz embedding files.
        ood_npz_path: Path to ood_detector_v2.npz.

    Returns:
        SweepData with aligned arrays for all 5000 images.
    """
    # Load SigLIP2 predictions
    all_records: list[dict] = []
    for split in ["train", "val", "test"]:
        records = _load_siglip2_jsonl(siglip2_dir / f"siglip2_diqa5000_{split}.jsonl")
        for r in records:
            r["split"] = split
        all_records.extend(records)

    n = len(all_records)
    logger.info("Loaded %d SigLIP2 predictions", n)

    # Load embeddings and compute Mahalanobis distances
    embeddings_list = []
    image_names_list = []
    for split in ["train", "val", "test"]:
        npz = np.load(embedding_dir / f"{split}.npz")
        embeddings_list.append(npz["embeddings"])
        image_names_list.extend(npz["image_names"].tolist())

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    logger.info("Loaded embeddings: shape %s", all_embeddings.shape)

    # Load OOD detector and score all embeddings
    # Construct directly to avoid shape (1,) → float conversion bug in from_npz
    ood_npz = np.load(str(ood_npz_path))
    ood_detector = OODDetectorWrapper(
        mean=ood_npz["mean"],
        precision_matrix=ood_npz["precision_matrix"],
        threshold=46.0,
    )
    ood_results = ood_detector.score_batch(all_embeddings)
    dm_from_embeddings = np.array([r.mahalanobis_distance for r in ood_results])

    # Build aligned arrays — match JSONL records to embeddings by image name
    embedding_name_to_idx = {name: i for i, name in enumerate(image_names_list)}

    image_ids: list[str] = []
    splits: list[str] = []
    dm = np.empty(n, dtype=np.float64)
    sigma_sq = np.empty((n, 3), dtype=np.float64)
    entropy_arr = np.empty((n, 3), dtype=np.float64)
    mu_mos = np.empty((n, 3), dtype=np.float64)

    dim_keys = [
        ("iqa_overall_mu", "iqa_overall_sigma_sq"),
        ("iqa_sharpness_mu", "iqa_sharpness_sigma_sq"),
        ("iqa_color_mu", "iqa_color_sigma_sq"),
    ]

    for i, record in enumerate(all_records):
        img = record["image"]
        image_ids.append(img)
        splits.append(record["split"])

        # Mahalanobis distance
        emb_idx = embedding_name_to_idx.get(img)
        if emb_idx is not None:
            dm[i] = dm_from_embeddings[emb_idx]
        else:
            dm[i] = 0.0
            logger.warning("No embedding found for %s", img)

        # Per-dimension signals
        for j, (mu_key, sq_key) in enumerate(dim_keys):
            raw_mu = record[mu_key]
            raw_sq = record[sq_key]

            # Convert to MOS scale
            mos = raw_mu * 4.0 + 1.0
            sq_mos = raw_sq * 16.0

            mu_mos[i, j] = mos
            sigma_sq[i, j] = sq_mos

            # Compute entropy using the pipeline's actual function
            level_probs = siglip2_output_to_level_probs(mos, sq_mos)
            entropy_arr[i, j] = discrete_entropy(level_probs)

    split_arr = np.array(splits)

    return SweepData(
        image_ids=image_ids,
        splits=splits,
        dm=dm,
        sigma_sq=sigma_sq,
        entropy=entropy_arr,
        mu_mos=mu_mos,
        n_train_val=int(np.sum((split_arr == "train") | (split_arr == "val"))),
        split_mask_train_val=(split_arr == "train") | (split_arr == "val"),
        split_mask_test=(split_arr == "test"),
    )


# ---------------------------------------------------------------------------
# Step 2: Threshold Configurations
# ---------------------------------------------------------------------------


def build_threshold_configs(data: SweepData) -> list[ThresholdConfig]:
    """Build named profiles and d_M percentile sweep configs.

    Percentiles are computed from train+val data only.

    Args:
        data: SweepData with all signals.

    Returns:
        List of ThresholdConfig for the sweep.
    """
    tv = data.split_mask_train_val

    # Compute train+val percentiles for each signal
    dm_tv = data.dm[tv]
    # σ² and entropy: average across dimensions for percentile computation
    sq_tv = data.sigma_sq[tv].ravel()
    ent_tv = data.entropy[tv].ravel()

    def pctl(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p))

    # Log signal distributions
    logger.info(
        "Train+val d_M percentiles: p50=%.1f p75=%.1f p90=%.1f p95=%.1f p99=%.1f p99.5=%.1f",
        pctl(dm_tv, 50), pctl(dm_tv, 75), pctl(dm_tv, 90),
        pctl(dm_tv, 95), pctl(dm_tv, 99), pctl(dm_tv, 99.5),
    )
    logger.info(
        "Train+val σ² percentiles: p50=%.6f p75=%.6f p90=%.6f p95=%.6f p99=%.6f",
        pctl(sq_tv, 50), pctl(sq_tv, 75), pctl(sq_tv, 90),
        pctl(sq_tv, 95), pctl(sq_tv, 99),
    )
    logger.info(
        "Train+val entropy percentiles: p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f",
        pctl(ent_tv, 50), pctl(ent_tv, 75), pctl(ent_tv, 90),
        pctl(ent_tv, 95), pctl(ent_tv, 99),
    )

    inf = float("inf")

    configs = [
        # Current hardcoded defaults
        ThresholdConfig(
            name="current",
            dm_ood=46.0, dm_reject=58.6,
            sigma_sq_auto=0.64, sigma_sq_low=1.0,
            entropy_auto=1.2, entropy_low=1.5,
        ),
        # Data-calibrated (keep existing d_M, fix σ²/entropy)
        ThresholdConfig(
            name="data_calibrated",
            dm_ood=46.0, dm_reject=58.6,
            sigma_sq_auto=pctl(sq_tv, 75), sigma_sq_low=pctl(sq_tv, 90),
            entropy_auto=pctl(ent_tv, 75), entropy_low=pctl(ent_tv, 90),
        ),
        # Strict: reject more
        ThresholdConfig(
            name="strict",
            dm_ood=pctl(dm_tv, 90), dm_reject=pctl(dm_tv, 95),
            sigma_sq_auto=pctl(sq_tv, 50), sigma_sq_low=pctl(sq_tv, 75),
            entropy_auto=pctl(ent_tv, 50), entropy_low=pctl(ent_tv, 75),
        ),
        # Moderate
        ThresholdConfig(
            name="moderate",
            dm_ood=pctl(dm_tv, 95), dm_reject=pctl(dm_tv, 99),
            sigma_sq_auto=pctl(sq_tv, 75), sigma_sq_low=pctl(sq_tv, 90),
            entropy_auto=pctl(ent_tv, 75), entropy_low=pctl(ent_tv, 90),
        ),
        # Lenient: accept more
        ThresholdConfig(
            name="lenient",
            dm_ood=pctl(dm_tv, 99), dm_reject=pctl(dm_tv, 99.5),
            sigma_sq_auto=pctl(sq_tv, 95), sigma_sq_low=pctl(sq_tv, 99),
            entropy_auto=pctl(ent_tv, 95), entropy_low=pctl(ent_tv, 99),
        ),
        # d_M only (σ²/entropy disabled)
        ThresholdConfig(
            name="dm_only",
            dm_ood=46.0, dm_reject=58.6,
            sigma_sq_auto=inf, sigma_sq_low=inf,
            entropy_auto=inf, entropy_low=inf,
        ),
        # No OOD (σ²/entropy only)
        ThresholdConfig(
            name="no_ood",
            dm_ood=inf, dm_reject=inf,
            sigma_sq_auto=pctl(sq_tv, 75), sigma_sq_low=pctl(sq_tv, 90),
            entropy_auto=pctl(ent_tv, 75), entropy_low=pctl(ent_tv, 90),
        ),
    ]

    # d_M percentile sweep (5 values)
    # Hard reject always at p99.5 to ensure it's above the OOD threshold
    dm_reject_pctl = pctl(dm_tv, 99.5)
    for p_ood in [90, 92, 95, 97, 99]:
        configs.append(ThresholdConfig(
            name=f"dm_p{p_ood}",
            dm_ood=pctl(dm_tv, p_ood),
            dm_reject=dm_reject_pctl,
            sigma_sq_auto=pctl(sq_tv, 75),
            sigma_sq_low=pctl(sq_tv, 90),
            entropy_auto=pctl(ent_tv, 75),
            entropy_low=pctl(ent_tv, 90),
        ))

    return configs


# ---------------------------------------------------------------------------
# Step 3: Vectorized Tier Assignment
# ---------------------------------------------------------------------------


def assign_tiers(
    dm: np.ndarray,
    sigma_sq: np.ndarray,
    entropy: np.ndarray,
    config: ThresholdConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tier assignment replicating UncertaintyFusion.decide().

    Args:
        dm: (N,) Mahalanobis distances.
        sigma_sq: (N, 3) σ² per dimension.
        entropy: (N, 3) entropy per dimension.
        config: Threshold configuration.

    Returns:
        tiers: (N, 3) int8 tier assignments.
        weights: (N, 3) float64 confidence weights.
    """
    n = dm.shape[0]
    n_dims = 3
    tiers = np.full((n, n_dims), TIER_TIER2_TRIGGER, dtype=np.int8)
    weights = np.zeros((n, n_dims), dtype=np.float64)

    # Broadcast d_M to (N, 3) for element-wise comparison with per-dim arrays
    dm_broad = np.broadcast_to(dm[:, np.newaxis], (n, n_dims))

    # 1. Hard reject: d_M > hard_reject
    hard_reject_mask = dm_broad > config.dm_reject
    tiers[hard_reject_mask] = TIER_HARD_REJECT

    # 2. OOD trigger: d_M > ood_threshold (and not already hard-rejected)
    ood_mask = (dm_broad > config.dm_ood) & ~hard_reject_mask
    tiers[ood_mask] = TIER_TIER2_TRIGGER

    # Remaining: not OOD and not hard-rejected
    remaining = ~hard_reject_mask & ~ood_mask

    # 3. Auto-accept: σ² ≤ auto AND entropy ≤ auto
    auto_mask = remaining & (sigma_sq <= config.sigma_sq_auto) & (entropy <= config.entropy_auto)
    tiers[auto_mask] = TIER_AUTO_ACCEPT
    weights[auto_mask] = 1.0

    # 4. Low-weight: σ² ≤ low AND entropy ≤ low (and not auto-accepted)
    low_mask = remaining & ~auto_mask & (sigma_sq <= config.sigma_sq_low) & (entropy <= config.entropy_low)
    tiers[low_mask] = TIER_LOW_WEIGHT

    # Interpolate weights for low-weight tier
    if np.any(low_mask):
        lo, hi = 0.3, 0.6
        positions = []

        sq_range = config.sigma_sq_low - config.sigma_sq_auto
        if sq_range > 0:
            sq_pos = np.clip(
                (sigma_sq[low_mask] - config.sigma_sq_auto) / sq_range, 0.0, 1.0,
            )
            positions.append(sq_pos)

        ent_range = config.entropy_low - config.entropy_auto
        if ent_range > 0:
            ent_pos = np.clip(
                (entropy[low_mask] - config.entropy_auto) / ent_range, 0.0, 1.0,
            )
            positions.append(ent_pos)

        if positions:
            worst = np.maximum.reduce(positions) if len(positions) > 1 else positions[0]
            weights[low_mask] = hi - worst * (hi - lo)
        else:
            weights[low_mask] = hi

    # 5. Remaining → TIER2_TRIGGER (already set as default)

    return tiers, weights


def compute_tier_stats(
    tiers: np.ndarray,
    weights: np.ndarray,
    dim_idx: int,
    mask: np.ndarray,
) -> TierStats:
    """Compute tier statistics for a specific dimension and split mask.

    Args:
        tiers: (N, 3) tier assignments.
        weights: (N, 3) confidence weights.
        dim_idx: Dimension index (0=overall, 1=sharpness, 2=color).
        mask: (N,) boolean mask for the split.

    Returns:
        TierStats with counts and percentages.
    """
    t = tiers[mask, dim_idx]
    w = weights[mask, dim_idx]
    n = len(t)

    counts = {
        "auto_accept": int(np.sum(t == TIER_AUTO_ACCEPT)),
        "low_weight": int(np.sum(t == TIER_LOW_WEIGHT)),
        "tier2_trigger": int(np.sum(t == TIER_TIER2_TRIGGER)),
        "hard_reject": int(np.sum(t == TIER_HARD_REJECT)),
    }

    return TierStats(
        n_total=n,
        n_auto_accept=counts["auto_accept"],
        n_low_weight=counts["low_weight"],
        n_tier2_trigger=counts["tier2_trigger"],
        n_hard_reject=counts["hard_reject"],
        pct_auto_accept=counts["auto_accept"] / n * 100 if n > 0 else 0,
        pct_low_weight=counts["low_weight"] / n * 100 if n > 0 else 0,
        pct_tier2_trigger=counts["tier2_trigger"] / n * 100 if n > 0 else 0,
        pct_hard_reject=counts["hard_reject"] / n * 100 if n > 0 else 0,
        mean_weight=float(np.mean(w)) if n > 0 else 0,
        effective_n=float(np.sum(w)),
    )


# ---------------------------------------------------------------------------
# Step 5: Spot-check validation against scalar UncertaintyFusion
# ---------------------------------------------------------------------------


def spot_check_validation(
    data: SweepData,
    tiers: np.ndarray,
    weights: np.ndarray,
    config: ThresholdConfig,
    n_checks: int = 10,
) -> bool:
    """Validate vectorized results against scalar UncertaintyFusion.decide().

    Prints comparison to stderr. Returns True if all match.
    """
    from src.uncertainty.cross_validator import CrossValidationResult
    from src.uncertainty.fusion import AcceptanceTier, JSDThresholds, UncertaintyFusion

    tier_map = {
        AcceptanceTier.AUTO_ACCEPT: TIER_AUTO_ACCEPT,
        AcceptanceTier.LOW_WEIGHT: TIER_LOW_WEIGHT,
        AcceptanceTier.TIER2_TRIGGER: TIER_TIER2_TRIGGER,
        AcceptanceTier.HARD_REJECT: TIER_HARD_REJECT,
    }

    fusion = UncertaintyFusion(
        mahalanobis_ood_threshold=config.dm_ood,
        mahalanobis_hard_reject=config.dm_reject,
        sigma_sq_auto=config.sigma_sq_auto,
        sigma_sq_low=config.sigma_sq_low,
        entropy_auto=config.entropy_auto,
        entropy_low=config.entropy_low,
        jsd_thresholds={
            dim: JSDThresholds(auto_accept=0.06, low_weight=0.12)
            for dim in DIMENSIONS
        },
    )

    rng = np.random.default_rng(42)
    indices = rng.choice(len(data.image_ids), size=min(n_checks, len(data.image_ids)), replace=False)

    all_match = True
    for idx in indices:
        for dim_idx, dim_name in enumerate(DIMENSIONS):
            # Build a CrossValidationResult with JSD=0 (disabled)
            mu = data.mu_mos[idx, dim_idx]
            sq = data.sigma_sq[idx, dim_idx]
            sigma = max(math.sqrt(max(sq, 0.0)), 0.1)
            level_probs = siglip2_output_to_level_probs(mu, sq)

            cross_val = CrossValidationResult(
                image_id=data.image_ids[idx],
                dimension=dim_name,
                siglip2_probs=level_probs,
                siglip2_mu=float(np.clip(mu, 1.0, 5.0)),
                siglip2_sigma_sq=sq,
                deqa_probs=level_probs,  # same → JSD=0
                deqa_mos=float(np.clip(mu, 1.0, 5.0)),
                cross_model_jsd=0.0,
                mos_delta=0.0,
                siglip2_entropy=data.entropy[idx, dim_idx],
            )

            decision = fusion.decide(cross_val, data.dm[idx])
            expected_tier = tier_map[decision.tier]
            actual_tier = tiers[idx, dim_idx]
            actual_weight = weights[idx, dim_idx]

            if expected_tier != actual_tier:
                print(
                    f"MISMATCH idx={idx} dim={dim_name}: "
                    f"expected {TIER_NAMES[expected_tier]} got {TIER_NAMES[actual_tier]} "
                    f"d_M={data.dm[idx]:.1f} σ²={sq:.6f} H={data.entropy[idx, dim_idx]:.4f}",
                    file=sys.stderr,
                )
                all_match = False
            elif abs(decision.confidence_weight - actual_weight) > 0.01:
                print(
                    f"WEIGHT MISMATCH idx={idx} dim={dim_name}: "
                    f"expected {decision.confidence_weight:.3f} got {actual_weight:.3f}",
                    file=sys.stderr,
                )
                all_match = False

    status = "PASS" if all_match else "FAIL"
    print(f"Spot-check validation: {status} ({n_checks} images × 3 dims)", file=sys.stderr)
    return all_match


# ---------------------------------------------------------------------------
# Step 6-7: Tier-2 VLM Veto Sweep
# ---------------------------------------------------------------------------


def load_vlm_data(
    checkpoint_dir: Path,
    siglip2_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    """Load VLM checkpoint predictions and SigLIP2 test predictions.

    Args:
        checkpoint_dir: Directory containing VLM checkpoint JSONL files.
        siglip2_dir: Directory containing SigLIP2 JSONL files.

    Returns:
        vlm_predictions: {model_id: {image_id: {dim: score}}}
        siglip2_mos: {image_id: {dim: mos_score}}
    """
    # Load VLM predictions
    vlm_predictions: dict[str, dict[str, dict[str, float]]] = {}
    for jsonl_path in sorted(checkpoint_dir.glob("*.jsonl")):
        # Use filename stem as model_id to handle duplicate model_ids across checkpoints
        file_model_id = jsonl_path.stem.replace("__", "/")
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line.strip())
                model_id = file_model_id
                image = record["image"]

                if model_id not in vlm_predictions:
                    vlm_predictions[model_id] = {}

                # Normalize color_fidelity → color
                vlm_predictions[model_id][image] = {
                    "overall": record.get("overall", 0.0),
                    "sharpness": record.get("sharpness", 0.0),
                    "color": record.get("color_fidelity", 0.0),
                }

    # Load SigLIP2 test predictions
    siglip2_mos: dict[str, dict[str, float]] = {}
    test_path = siglip2_dir / "siglip2_diqa5000_test.jsonl"
    with open(test_path) as f:
        for line in f:
            record = json.loads(line.strip())
            image = record["image"]
            siglip2_mos[image] = {
                "overall": record["iqa_overall_mu"] * 4.0 + 1.0,
                "sharpness": record["iqa_sharpness_mu"] * 4.0 + 1.0,
                "color": record["iqa_color_mu"] * 4.0 + 1.0,
            }

    logger.info(
        "Loaded VLM predictions: %d models, %d test images with SigLIP2 MOS",
        len(vlm_predictions),
        len(siglip2_mos),
    )
    return vlm_predictions, siglip2_mos


def run_veto_sweep(
    vlm_predictions: dict[str, dict[str, dict[str, float]]],
    siglip2_mos: dict[str, dict[str, float]],
    thresholds: list[float],
) -> list[VetoSweepResult]:
    """Sweep veto thresholds across all models and dimensions.

    Args:
        vlm_predictions: {model_id: {image_id: {dim: score}}}.
        siglip2_mos: {image_id: {dim: mos_score}}.
        thresholds: List of veto thresholds to sweep.

    Returns:
        List of VetoSweepResult for each threshold × model × dimension.
    """
    results: list[VetoSweepResult] = []

    # Pre-compute disagreements per model × dimension
    model_ids = sorted(vlm_predictions.keys())
    common_images = sorted(
        set.intersection(
            set(siglip2_mos.keys()),
            *(set(vlm_predictions[m].keys()) for m in model_ids),
        ),
    )
    n_images = len(common_images)
    logger.info("Common images for VLM sweep: %d", n_images)

    # Build disagreement arrays: (n_models, n_images, 3)
    n_models = len(model_ids)
    disagreements = np.zeros((n_models, n_images, 3), dtype=np.float64)

    for m_idx, model_id in enumerate(model_ids):
        for i_idx, image in enumerate(common_images):
            for d_idx, dim in enumerate(DIMENSIONS):
                vlm_score = vlm_predictions[model_id][image].get(dim)
                siglip2_score = siglip2_mos[image][dim]
                if vlm_score is not None and siglip2_score is not None:
                    disagreements[m_idx, i_idx, d_idx] = abs(vlm_score - siglip2_score)
                else:
                    disagreements[m_idx, i_idx, d_idx] = 0.0  # treat missing as no disagreement

    # Sweep thresholds
    for threshold in thresholds:
        for m_idx, model_id in enumerate(model_ids):
            for d_idx, dim in enumerate(DIMENSIONS):
                disag = disagreements[m_idx, :, d_idx]
                n_vetoed = int(np.sum(disag >= threshold))
                results.append(VetoSweepResult(
                    threshold=threshold,
                    model_id=model_id,
                    dimension=dim,
                    n_total=n_images,
                    n_vetoed=n_vetoed,
                    veto_rate=n_vetoed / n_images if n_images > 0 else 0,
                    mean_disagreement=float(np.mean(disag)),
                    median_disagreement=float(np.median(disag)),
                ))

        # Ensemble majority vote
        for d_idx, dim in enumerate(DIMENSIONS):
            majority_threshold = math.ceil(n_models / 2)
            per_model_vetoes = disagreements[:, :, d_idx] >= threshold  # (n_models, n_images)
            vote_counts = per_model_vetoes.sum(axis=0)  # (n_images,)
            ensemble_vetoed = int(np.sum(vote_counts >= majority_threshold))
            # Mean/median of average disagreement across models
            mean_disag_per_image = disagreements[:, :, d_idx].mean(axis=0)
            results.append(VetoSweepResult(
                threshold=threshold,
                model_id="ensemble_majority",
                dimension=dim,
                n_total=n_images,
                n_vetoed=ensemble_vetoed,
                veto_rate=ensemble_vetoed / n_images if n_images > 0 else 0,
                mean_disagreement=float(np.mean(mean_disag_per_image)),
                median_disagreement=float(np.median(mean_disag_per_image)),
            ))

    return results


# ---------------------------------------------------------------------------
# Step 4 & 9: Reporting
# ---------------------------------------------------------------------------


def generate_json_output(
    tier1_results: dict[str, dict[str, dict[str, dict]]],
    tier2_results: list[VetoSweepResult],
    signal_distributions: dict,
    configs: list[ThresholdConfig],
) -> dict:
    """Generate machine-readable JSON output."""
    return {
        "tier1_sweep": {
            "configs": [asdict(c) for c in configs],
            "results": tier1_results,
            "signal_distributions": signal_distributions,
        },
        "tier2_sweep": {
            "results": [asdict(r) for r in tier2_results],
        },
    }


def _format_pctl_table(label: str, values: dict[str, float]) -> str:
    """Format a single row of percentile values."""
    cols = " | ".join(f"{v:>8.4f}" for v in values.values())
    return f"| {label:<12} | {cols} |"


def generate_markdown_report(
    tier1_results: dict[str, dict[str, dict[str, dict]]],
    tier2_results: list[VetoSweepResult],
    signal_distributions: dict,
    configs: list[ThresholdConfig],
    data: SweepData,
) -> str:
    """Generate human-readable markdown report."""
    lines: list[str] = []
    lines.append("# U-3: OOD Threshold Sensitivity Analysis")
    lines.append("")
    lines.append("## 1. Signal Distributions")
    lines.append("")
    lines.append("Percentile values computed from **train+val** data (N={}).".format(data.n_train_val))
    lines.append("")

    # Signal distribution tables
    for signal_name in ["d_M", "sigma_sq", "entropy"]:
        lines.append(f"### {signal_name}")
        lines.append("")
        pctls = ["p50", "p75", "p90", "p95", "p99", "p99.5"]
        header = "| Split/Dim    | " + " | ".join(f"{p:>8}" for p in pctls) + " |"
        sep = "|" + "-" * 14 + "|" + (("-" * 10 + "|") * len(pctls))
        lines.append(header)
        lines.append(sep)

        dist = signal_distributions.get(signal_name, {})
        for key, vals in dist.items():
            cols = " | ".join(f"{vals.get(p, 0):>8.4f}" for p in pctls)
            lines.append(f"| {key:<12} | {cols} |")
        lines.append("")

    # Named profiles comparison (test split)
    lines.append("## 2. Named Profiles Comparison (Test Split)")
    lines.append("")
    lines.append("| Profile | Dim | AUTO_ACCEPT% | LOW_WEIGHT% | TIER2% | HARD_REJECT% | Mean Weight | Effective N |")
    lines.append("|---------|-----|-------------|------------|--------|-------------|-------------|-------------|")

    named_profiles = ["current", "data_calibrated", "strict", "moderate", "lenient", "dm_only", "no_ood"]
    for profile in named_profiles:
        if profile not in tier1_results:
            continue
        for dim in DIMENSIONS:
            test_stats = tier1_results[profile].get("test", {}).get(dim, {})
            if not test_stats:
                continue
            lines.append(
                f"| {profile:<15} | {dim:<9} | "
                f"{test_stats['pct_auto_accept']:>11.1f} | "
                f"{test_stats['pct_low_weight']:>10.1f} | "
                f"{test_stats['pct_tier2_trigger']:>6.1f} | "
                f"{test_stats['pct_hard_reject']:>11.1f} | "
                f"{test_stats['mean_weight']:>11.3f} | "
                f"{test_stats['effective_n']:>11.1f} |"
            )
    lines.append("")

    # d_M sweep
    lines.append("## 3. d_M Percentile Sweep (Test Split, Overall Dimension)")
    lines.append("")
    lines.append("| Config | d_M OOD | d_M Reject | AUTO_ACCEPT% | TIER2% | HARD_REJECT% | Effective N |")
    lines.append("|--------|---------|-----------|-------------|--------|-------------|-------------|")

    dm_configs = [c for c in configs if c.name.startswith("dm_p")]
    for config in dm_configs:
        test_stats = tier1_results.get(config.name, {}).get("test", {}).get("overall", {})
        if not test_stats:
            continue
        lines.append(
            f"| {config.name:<10} | {config.dm_ood:>7.1f} | {config.dm_reject:>9.1f} | "
            f"{test_stats['pct_auto_accept']:>11.1f} | "
            f"{test_stats['pct_tier2_trigger']:>6.1f} | "
            f"{test_stats['pct_hard_reject']:>11.1f} | "
            f"{test_stats['effective_n']:>11.1f} |"
        )
    lines.append("")

    # Key findings
    lines.append("## 4. Key Findings")
    lines.append("")

    current_test = tier1_results.get("current", {}).get("test", {}).get("overall", {})
    calibrated_test = tier1_results.get("data_calibrated", {}).get("test", {}).get("overall", {})

    if current_test:
        lines.append(f"- **Current thresholds**: {current_test.get('pct_auto_accept', 0):.1f}% AUTO_ACCEPT "
                     f"on test (σ²/entropy thresholds never trigger)")
    if calibrated_test:
        lines.append(f"- **Data-calibrated thresholds**: {calibrated_test.get('pct_auto_accept', 0):.1f}% AUTO_ACCEPT "
                     f"on test (σ²/entropy now discriminate)")

    dm_only_test = tier1_results.get("dm_only", {}).get("test", {}).get("overall", {})
    no_ood_test = tier1_results.get("no_ood", {}).get("test", {}).get("overall", {})
    if dm_only_test:
        lines.append(f"- **d_M only**: {dm_only_test.get('pct_auto_accept', 0):.1f}% AUTO_ACCEPT "
                     f"(σ²/entropy disabled → pure OOD gating)")
    if no_ood_test:
        lines.append(f"- **No OOD**: {no_ood_test.get('pct_auto_accept', 0):.1f}% AUTO_ACCEPT "
                     f"(d_M disabled → σ²/entropy only)")
    lines.append("")

    # Tier-2 VLM Veto Sweep
    lines.append("## 5. Tier-2 VLM Veto Threshold Sweep")
    lines.append("")

    # Disagreement distribution
    lines.append("### 5.1 VLM Disagreement Distribution (|VLM - SigLIP2|)")
    lines.append("")

    # Get unique models and thresholds
    models_in_results = sorted({r.model_id for r in tier2_results if r.model_id != "ensemble_majority"})
    threshold_values = sorted({r.threshold for r in tier2_results})

    # Per-model mean disagreement table
    lines.append("| Model | overall | sharpness | color |")
    lines.append("|-------|---------|-----------|-------|")
    for model_id in models_in_results:
        model_results = [r for r in tier2_results if r.model_id == model_id and r.threshold == threshold_values[0]]
        dims = {r.dimension: r.mean_disagreement for r in model_results}
        lines.append(
            f"| {model_id:<40} | {dims.get('overall', 0):>7.3f} | "
            f"{dims.get('sharpness', 0):>9.3f} | {dims.get('color', 0):>5.3f} |"
        )
    lines.append("")

    # Veto rate sweep table
    lines.append("### 5.2 Veto Rate by Threshold (Overall Dimension)")
    lines.append("")

    # Header
    model_short_names = [m.split("/")[-1] if "/" in m else m for m in models_in_results]
    header_models = " | ".join(f"{n[:12]:>12}" for n in model_short_names)
    lines.append(f"| Threshold | {header_models} | ensemble |")
    sep_cols = " | ".join("-" * 12 for _ in model_short_names)
    lines.append(f"|-----------|{sep_cols.replace(' ', '-')}|----------|")

    for threshold in threshold_values:
        rates = []
        for model_id in models_in_results:
            match = [r for r in tier2_results
                     if r.model_id == model_id and r.threshold == threshold and r.dimension == "overall"]
            rates.append(f"{match[0].veto_rate * 100:>11.1f}%" if match else f"{'N/A':>12}")
        ensemble_match = [r for r in tier2_results
                         if r.model_id == "ensemble_majority" and r.threshold == threshold and r.dimension == "overall"]
        ens_rate = f"{ensemble_match[0].veto_rate * 100:.1f}%" if ensemble_match else "N/A"
        rates_str = " | ".join(rates)
        lines.append(f"| {threshold:>9.1f} | {rates_str} | {ens_rate:>8} |")
    lines.append("")

    # Per-dimension breakdown
    lines.append("### 5.3 Per-Dimension Veto Rates at Current Threshold (1.5)")
    lines.append("")
    lines.append("| Model | overall | sharpness | color |")
    lines.append("|-------|---------|-----------|-------|")

    for model_id in [*models_in_results, "ensemble_majority"]:
        dim_rates = {}
        for dim in DIMENSIONS:
            match = [r for r in tier2_results
                     if r.model_id == model_id and r.threshold == 1.5 and r.dimension == dim]
            dim_rates[dim] = match[0].veto_rate * 100 if match else 0
        display_name = model_id.split("/")[-1] if "/" in model_id else model_id
        lines.append(
            f"| {display_name:<40} | {dim_rates.get('overall', 0):>6.1f}% | "
            f"{dim_rates.get('sharpness', 0):>8.1f}% | {dim_rates.get('color', 0):>4.1f}% |"
        )
    lines.append("")

    # Limitations
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append("- **JSD thresholds not swept**: DeQA predictions unavailable per-image; JSD=0 for all. "
                 "JSD sensitivity requires separate DeQA inference.")
    lines.append("- **GT validation for Tier-2 not available**: DIQA-5000 test set has no ground-truth MOS, "
                 "so True/False veto accuracy cannot be computed.")
    lines.append("- **Calibration split**: All percentile thresholds computed from train+val (N={}). "
                 "Test split (N={}) used for evaluation only.".format(data.n_train_val, int(np.sum(data.split_mask_test))))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full threshold sensitivity sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    siglip2_dir = repo_root / "results" / "siglip2_diqa5000"
    embedding_dir = siglip2_dir / "embeddings"
    ood_npz_path = siglip2_dir / "ood_detector_v2.npz"
    checkpoint_dir = repo_root / "results" / "vlm_teacher_eval" / "full_eval" / "checkpoints"
    output_dir = repo_root / "results" / "threshold_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Part 1: Tier-1 Fusion Sweep ---
    logger.info("=== Part 1: Tier-1 Fusion Threshold Sweep ===")
    data = load_data(siglip2_dir, embedding_dir, ood_npz_path)
    configs = build_threshold_configs(data)

    # Compute signal distributions for reporting
    tv = data.split_mask_train_val
    test = data.split_mask_test
    percentiles_list = [50, 75, 90, 95, 99, 99.5]
    pctl_names = ["p50", "p75", "p90", "p95", "p99", "p99.5"]

    signal_distributions: dict[str, dict[str, dict[str, float]]] = {}

    # d_M (scalar per image)
    signal_distributions["d_M"] = {}
    for label, mask in [("train+val", tv), ("test", test), ("all", np.ones(len(data.dm), dtype=bool))]:
        vals = data.dm[mask]
        signal_distributions["d_M"][label] = {
            p: float(np.percentile(vals, pv)) for p, pv in zip(pctl_names, percentiles_list)
        }

    # σ² and entropy (per dimension)
    for signal_name, arr in [("sigma_sq", data.sigma_sq), ("entropy", data.entropy)]:
        signal_distributions[signal_name] = {}
        for label, mask in [("train+val", tv), ("test", test)]:
            for d_idx, dim in enumerate(DIMENSIONS):
                key = f"{label}_{dim}"
                vals = arr[mask, d_idx]
                signal_distributions[signal_name][key] = {
                    p: float(np.percentile(vals, pv)) for p, pv in zip(pctl_names, percentiles_list)
                }

    # Run tier assignment for each config
    tier1_results: dict[str, dict[str, dict[str, dict]]] = {}
    for config in configs:
        tiers, weights = assign_tiers(data.dm, data.sigma_sq, data.entropy, config)
        tier1_results[config.name] = {}

        for split_name, mask in [("train_val", tv), ("test", test)]:
            tier1_results[config.name][split_name] = {}
            for d_idx, dim in enumerate(DIMENSIONS):
                stats = compute_tier_stats(tiers, weights, d_idx, mask)
                tier1_results[config.name][split_name][dim] = asdict(stats)

        # Spot-check the first config (current defaults)
        if config.name == "current":
            spot_check_validation(data, tiers, weights, config)

    # --- Part 2: Tier-2 VLM Veto Sweep ---
    logger.info("=== Part 2: Tier-2 VLM Veto Threshold Sweep ===")
    vlm_predictions, siglip2_mos = load_vlm_data(checkpoint_dir, siglip2_dir)
    veto_thresholds = [0.3, 0.5, 0.8, 1.0, 1.5]
    tier2_results = run_veto_sweep(vlm_predictions, siglip2_mos, veto_thresholds)

    # --- Generate outputs ---
    logger.info("=== Generating outputs ===")

    json_output = generate_json_output(tier1_results, tier2_results, signal_distributions, configs)
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    logger.info("JSON results written to %s", json_path)

    md_report = generate_markdown_report(tier1_results, tier2_results, signal_distributions, configs, data)
    md_path = output_dir / "sweep_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    logger.info("Markdown report written to %s", md_path)

    # Print summary to stdout
    print(f"\nResults written to:")
    print(f"  JSON: {json_path}")
    print(f"  Report: {md_path}")

    # Quick summary
    current_test = tier1_results.get("current", {}).get("test", {}).get("overall", {})
    calibrated_test = tier1_results.get("data_calibrated", {}).get("test", {}).get("overall", {})
    if current_test and calibrated_test:
        print(f"\nKey finding:")
        print(f"  Current thresholds:        {current_test['pct_auto_accept']:.1f}% AUTO_ACCEPT (test, overall)")
        print(f"  Data-calibrated thresholds: {calibrated_test['pct_auto_accept']:.1f}% AUTO_ACCEPT (test, overall)")


if __name__ == "__main__":
    main()
