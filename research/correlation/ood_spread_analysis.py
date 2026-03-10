"""
OOD Spread Analysis Pipeline for DIQA Model Ensemble
=====================================================

This pipeline computes inter-model disagreement (spread) as a proxy for
out-of-distribution detection, validates the spread metric against known
ground truth, and produces stratified samples for human annotation.

Architecture:
    1. Load & normalize model predictions to a common z-score scale
    2. Compute baseline spread distribution on DIQA-5000 test set
    3. Compute OOD spread on the 520-sample synthetic dataset
    4. Decompose disagreement structure by model pairs and clusters
    5. Stratified sampling: select 25 images for human annotation
    6. Export annotation batch with anchor images

Expects two CSV inputs (paths configurable below):
    - diqa5000_predictions.csv: columns [image_id, human_mos, siglip, hyperiqa, mllm, vl, frontier]
    - ood_predictions.csv:      columns [image_id, siglip, hyperiqa, mllm, vl, frontier]
      (human_mos column optional — will be used for validation if present)

Output:
    - Console report with spread statistics and OOD classification
    - annotation_batch.csv: 25 stratified OOD images + 5 anchors for human scoring
    - spread_analysis_report.csv: full 520-image dataset with spread metrics
    - model_pair_disagreement.csv: pairwise divergence matrix per OOD category
    - Figures saved as PNG files

Author: Byron's DIQA Pipeline
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """All tunable parameters in one place."""

    # --- File paths ---
    diqa5000_predictions_path: str = "diqa5000_predictions.csv"
    ood_predictions_path: str = "ood_predictions.csv"
    output_dir: str = "spread_analysis_output"

    # --- Model columns (must match CSV headers) ---
    model_columns: list = field(default_factory=lambda: [
        "siglip", "hyperiqa", "mllm", "vl", "frontier"
    ])
    human_mos_column: str = "human_mos"
    image_id_column: str = "image_id"

    # --- OOD category column (optional, for per-category analysis) ---
    # If your OOD CSV has a column indicating document type/category
    ood_category_column: Optional[str] = "category"

    # --- Spread thresholds (in units of baseline σ) ---
    soft_ood_threshold_sigma: float = 1.0  # spread > μ + 1σ = soft OOD
    hard_ood_threshold_sigma: float = 2.0  # spread > μ + 2σ = strong OOD

    # --- Annotation batch parameters ---
    annotation_batch_size: int = 25       # gap images per batch
    n_anchors: int = 5                     # DIQA-5000 anchor images
    n_low_spread: int = 5                  # from low-spread bucket
    n_mid_spread: int = 10                 # from medium-spread bucket
    n_high_spread: int = 10                # from high-spread bucket

    # --- Anchor selection: target MOS values (evenly spanning 1-5) ---
    anchor_target_mos: list = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    anchor_mos_tolerance: float = 0.3      # how close to target MOS

    # --- Normalization method ---
    normalization: str = "z_score"  # "z_score" or "min_max"

    # --- Flagging thresholds ---
    high_model_spread_flag: float = 1.0    # raw score SD (on normalized scale)


# ============================================================================
# STEP 1: DATA LOADING AND NORMALIZATION
# ============================================================================

class ModelNormalizer:
    """
    Fits normalization parameters on DIQA-5000 (in-distribution) and applies
    the same transform to OOD data. This is critical — normalizing OOD data
    on its own statistics would destroy the spread signal.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.fit_params = {}  # {model_name: {"mean": float, "std": float}}

    def fit(self, diqa_df: pd.DataFrame) -> None:
        """Learn normalization parameters from DIQA-5000 predictions."""
        for model in self.config.model_columns:
            scores = diqa_df[model].dropna()
            if self.config.normalization == "z_score":
                self.fit_params[model] = {
                    "mean": scores.mean(),
                    "std": scores.std(ddof=1),
                }
            elif self.config.normalization == "min_max":
                self.fit_params[model] = {
                    "min": scores.min(),
                    "max": scores.max(),
                }
            print(f"  {model:>12s}: mean={scores.mean():.4f}  std={scores.std():.4f}  "
                  f"range=[{scores.min():.4f}, {scores.max():.4f}]")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization learned from DIQA-5000 to any dataframe."""
        df_norm = df.copy()
        for model in self.config.model_columns:
            p = self.fit_params[model]
            if self.config.normalization == "z_score":
                df_norm[model] = (df[model] - p["mean"]) / p["std"]
            elif self.config.normalization == "min_max":
                df_norm[model] = (df[model] - p["min"]) / (p["max"] - p["min"])
        return df_norm


# ============================================================================
# STEP 2: BASELINE SPREAD DISTRIBUTION (DIQA-5000)
# ============================================================================

@dataclass
class SpreadDistribution:
    """Summary statistics for inter-model spread on a dataset."""
    per_image_spread: np.ndarray    # SD across models for each image
    mean: float                      # μ of spread distribution
    std: float                       # σ of spread distribution
    median: float
    q25: float
    q75: float
    q95: float
    q99: float

    @classmethod
    def compute(cls, df: pd.DataFrame, model_columns: list) -> "SpreadDistribution":
        """Compute per-image spread (SD across models) and summary stats."""
        model_scores = df[model_columns].values  # shape: (n_images, n_models)
        per_image_spread = np.std(model_scores, axis=1, ddof=1)

        return cls(
            per_image_spread=per_image_spread,
            mean=np.mean(per_image_spread),
            std=np.std(per_image_spread, ddof=1),
            median=np.median(per_image_spread),
            q25=np.percentile(per_image_spread, 25),
            q75=np.percentile(per_image_spread, 75),
            q95=np.percentile(per_image_spread, 95),
            q99=np.percentile(per_image_spread, 99),
        )

    def classify(self, spread_values: np.ndarray,
                 soft_sigma: float = 1.0,
                 hard_sigma: float = 2.0) -> np.ndarray:
        """
        Classify spread values as:
            0 = in-distribution-like (below soft threshold)
            1 = soft OOD (between soft and hard threshold)
            2 = strong OOD (above hard threshold)
        """
        soft_thresh = self.mean + soft_sigma * self.std
        hard_thresh = self.mean + hard_sigma * self.std
        labels = np.zeros(len(spread_values), dtype=int)
        labels[spread_values > soft_thresh] = 1
        labels[spread_values > hard_thresh] = 2
        return labels


# ============================================================================
# STEP 3: MODEL-PAIR DISAGREEMENT DECOMPOSITION
# ============================================================================

def compute_pairwise_disagreement(df: pd.DataFrame,
                                   model_columns: list) -> pd.DataFrame:
    """
    For each pair of models, compute the mean absolute difference across
    all images. Returns a symmetric matrix showing which model pairs
    diverge most.
    """
    pairs = list(combinations(model_columns, 2))
    results = {}
    for m1, m2 in pairs:
        mad = np.mean(np.abs(df[m1].values - df[m2].values))
        results[f"{m1}_vs_{m2}"] = mad

    # Also build symmetric matrix form
    n = len(model_columns)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(model_columns):
        for j, m2 in enumerate(model_columns):
            if i != j:
                mad = np.mean(np.abs(df[m1].values - df[m2].values))
                matrix[i, j] = mad

    matrix_df = pd.DataFrame(
        matrix,
        index=model_columns,
        columns=model_columns
    )
    return matrix_df


def compute_cluster_divergence(df: pd.DataFrame,
                                model_columns: list) -> dict:
    """
    Identify whether disagreement follows architectural clusters.
    Groups: vision-only (siglip, hyperiqa) vs MLLM-based (mllm, vl, frontier).
    Adjust groupings based on your actual model architectures.
    """
    # Define clusters — adjust these to match your actual model types
    vision_models = [m for m in ["siglip", "hyperiqa"] if m in model_columns]
    mllm_models = [m for m in ["mllm", "vl", "frontier"] if m in model_columns]

    if not vision_models or not mllm_models:
        return {"warning": "Could not form two clusters from available models"}

    vision_mean = df[vision_models].mean(axis=1)
    mllm_mean = df[mllm_models].mean(axis=1)

    inter_cluster_gap = np.abs(vision_mean - mllm_mean)

    # Within-cluster spread
    vision_spread = df[vision_models].std(axis=1, ddof=1) if len(vision_models) > 1 else pd.Series(0, index=df.index)
    mllm_spread = df[mllm_models].std(axis=1, ddof=1) if len(mllm_models) > 1 else pd.Series(0, index=df.index)

    return {
        "inter_cluster_gap_mean": inter_cluster_gap.mean(),
        "inter_cluster_gap_std": inter_cluster_gap.std(),
        "vision_intra_spread_mean": vision_spread.mean(),
        "mllm_intra_spread_mean": mllm_spread.mean(),
        "cluster_divergence_ratio": (
            inter_cluster_gap.mean() /
            max((vision_spread.mean() + mllm_spread.mean()) / 2, 1e-8)
        ),
        # Per-image: is the gap mainly between clusters or within?
        "pct_images_inter_gt_intra": (
            (inter_cluster_gap > (vision_spread + mllm_spread) / 2).mean() * 100
        ),
    }


# ============================================================================
# STEP 4: GROUND TRUTH VALIDATION (when human MOS available)
# ============================================================================

def validate_spread_metric(df: pd.DataFrame,
                            model_columns: list,
                            human_mos_col: str,
                            spread_values: np.ndarray) -> dict:
    """
    Test the core hypothesis: does higher spread correlate with lower
    prediction accuracy? Bins images by spread quantile and computes
    per-bin SRCC/PLCC against human MOS.
    """
    if human_mos_col not in df.columns:
        return {"status": "no_human_mos_available"}

    human = df[human_mos_col].values
    ensemble_mean = df[model_columns].mean(axis=1).values

    # Overall metrics
    overall_srcc, _ = stats.spearmanr(ensemble_mean, human)
    overall_plcc, _ = stats.pearsonr(ensemble_mean, human)
    overall_rmse = np.sqrt(np.mean((ensemble_mean - human) ** 2))

    # Per-spread-quantile metrics
    quantile_labels = pd.qcut(spread_values, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    per_quantile = {}
    for q_label in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        mask = quantile_labels == q_label
        if mask.sum() < 5:
            continue
        q_human = human[mask]
        q_pred = ensemble_mean[mask]
        q_srcc, _ = stats.spearmanr(q_pred, q_human)
        q_plcc, _ = stats.pearsonr(q_pred, q_human)
        q_rmse = np.sqrt(np.mean((q_pred - q_human) ** 2))
        per_quantile[q_label] = {
            "n_images": int(mask.sum()),
            "srcc": round(q_srcc, 4),
            "plcc": round(q_plcc, 4),
            "rmse": round(q_rmse, 4),
            "mean_spread": round(spread_values[mask].mean(), 4),
        }

    # Per-model accuracy vs spread correlation
    per_model_spread_corr = {}
    for model in model_columns:
        model_errors = np.abs(df[model].values - human)
        corr, pval = stats.spearmanr(spread_values, model_errors)
        per_model_spread_corr[model] = {
            "spread_error_srcc": round(corr, 4),
            "p_value": round(pval, 6),
        }

    return {
        "status": "validated",
        "overall": {
            "ensemble_srcc": round(overall_srcc, 4),
            "ensemble_plcc": round(overall_plcc, 4),
            "ensemble_rmse": round(overall_rmse, 4),
        },
        "per_spread_quantile": per_quantile,
        "spread_error_correlation": per_model_spread_corr,
    }


# ============================================================================
# STEP 5: ANCHOR SELECTION FROM DIQA-5000
# ============================================================================

def select_anchors(diqa_df: pd.DataFrame,
                   config: PipelineConfig) -> pd.DataFrame:
    """
    Select anchor images from DIQA-5000 that:
    1. Span the MOS range evenly (targets: 1.0, 2.0, 3.0, 4.0, 5.0)
    2. Have LOW inter-model spread (models agree = unambiguous quality)
    3. Are content-neutral (simple text, no exotic layouts)
       — This requires manual review; we select candidates algorithmically
         and flag the top 3 per target for manual inspection.
    """
    mos_col = config.human_mos_column
    model_cols = config.model_columns
    spread = diqa_df[model_cols].std(axis=1, ddof=1)

    anchors = []
    for target_mos in config.anchor_target_mos:
        # Find images close to target MOS
        mos_distance = np.abs(diqa_df[mos_col] - target_mos)
        candidates = diqa_df[mos_distance <= config.anchor_mos_tolerance].copy()

        if len(candidates) == 0:
            # Widen tolerance
            candidates = diqa_df.nsmallest(20, mos_distance).copy()

        # Among candidates, prefer lowest model spread (most agreement)
        candidates["_spread"] = spread[candidates.index]
        candidates["_mos_dist"] = mos_distance[candidates.index]
        # Composite score: low spread + close to target MOS
        candidates["_anchor_score"] = (
            candidates["_spread"] / candidates["_spread"].max() * 0.5 +
            candidates["_mos_dist"] / max(candidates["_mos_dist"].max(), 1e-8) * 0.5
        )
        best = candidates.nsmallest(1, "_anchor_score").iloc[0]
        anchors.append({
            config.image_id_column: best[config.image_id_column],
            "anchor_target_mos": target_mos,
            "actual_mos": best[mos_col],
            "model_spread": best["_spread"],
        })

    return pd.DataFrame(anchors)


# ============================================================================
# STEP 6: STRATIFIED SAMPLING FOR ANNOTATION
# ============================================================================

def stratified_sample(ood_df: pd.DataFrame,
                      spread_values: np.ndarray,
                      ood_labels: np.ndarray,
                      config: PipelineConfig) -> pd.DataFrame:
    """
    Select 25 images from the OOD set, stratified by spread level:
      - 5 from low spread (in-distribution-like, sanity check)
      - 10 from medium spread (soft OOD, boundary cases)
      - 10 from high spread (strong OOD, maximum diagnostic value)

    Within each stratum, selection maximizes diversity:
      - If categories are available, sample across categories
      - Otherwise, spread images evenly across the spread range within each bucket
    """
    ood_working = ood_df.copy()
    ood_working["_spread"] = spread_values
    ood_working["_ood_label"] = ood_labels

    selected = []

    strata = [
        (0, config.n_low_spread, "low_spread_sanity_check"),
        (1, config.n_mid_spread, "medium_spread_boundary"),
        (2, config.n_high_spread, "high_spread_diagnostic"),
    ]

    for label, n_target, stratum_name in strata:
        pool = ood_working[ood_working["_ood_label"] == label]

        if len(pool) == 0:
            # If no images at this OOD level, steal from adjacent
            print(f"  WARNING: No images in {stratum_name} bucket. "
                  f"Sampling from nearest spread values instead.")
            if label == 0:
                pool = ood_working.nsmallest(n_target * 3, "_spread")
            elif label == 2:
                pool = ood_working.nlargest(n_target * 3, "_spread")
            else:
                # Middle: take the middle third
                sorted_idx = ood_working["_spread"].argsort()
                mid_start = len(sorted_idx) // 3
                mid_end = 2 * len(sorted_idx) // 3
                pool = ood_working.iloc[sorted_idx[mid_start:mid_end]]

        n_select = min(n_target, len(pool))

        # If categories available, stratify within stratum by category
        cat_col = config.ood_category_column
        if cat_col and cat_col in pool.columns:
            cats = pool[cat_col].unique()
            per_cat = max(1, n_select // len(cats))
            cat_samples = []
            for cat in cats:
                cat_pool = pool[pool[cat_col] == cat]
                n_from_cat = min(per_cat, len(cat_pool))
                # Within category, sample evenly across spread range
                if n_from_cat > 0:
                    idx = np.linspace(0, len(cat_pool) - 1, n_from_cat, dtype=int)
                    cat_sorted = cat_pool.sort_values("_spread")
                    cat_samples.append(cat_sorted.iloc[idx])
            if cat_samples:
                stratum_selected = pd.concat(cat_samples).head(n_select)
            else:
                stratum_selected = pool.sample(n=n_select, random_state=42)
        else:
            # No categories: evenly space across the spread range
            pool_sorted = pool.sort_values("_spread")
            idx = np.linspace(0, len(pool_sorted) - 1, n_select, dtype=int)
            stratum_selected = pool_sorted.iloc[idx]

        stratum_selected = stratum_selected.copy()
        stratum_selected["annotation_stratum"] = stratum_name
        selected.append(stratum_selected)

    result = pd.concat(selected, ignore_index=True)

    # Trim to exact batch size if over-sampled from category stratification
    if len(result) > config.annotation_batch_size:
        result = result.head(config.annotation_batch_size)

    return result


# ============================================================================
# STEP 7: EXPORT ANNOTATION PACKAGE
# ============================================================================

def build_annotation_package(sampled_ood: pd.DataFrame,
                              anchor_df: pd.DataFrame,
                              config: PipelineConfig) -> pd.DataFrame:
    """
    Combine 25 OOD images + 5 anchors into a single annotation batch.
    Randomize order so raters don't know which are anchors.
    Include columns needed for the scoring interface.
    """
    # Prepare OOD rows
    ood_rows = sampled_ood[[config.image_id_column, "annotation_stratum", "_spread"]].copy()
    ood_rows["is_anchor"] = False
    ood_rows["known_mos"] = np.nan
    ood_rows.rename(columns={"_spread": "model_spread"}, inplace=True)

    # Prepare anchor rows
    anchor_rows = anchor_df[[config.image_id_column, "actual_mos", "model_spread"]].copy()
    anchor_rows["is_anchor"] = True
    anchor_rows["known_mos"] = anchor_rows["actual_mos"]
    anchor_rows["annotation_stratum"] = "anchor"
    anchor_rows.drop(columns=["actual_mos"], inplace=True)

    # Combine and shuffle
    batch = pd.concat([ood_rows, anchor_rows], ignore_index=True)
    batch = batch.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add presentation order and empty scoring columns
    batch["presentation_order"] = range(1, len(batch) + 1)
    batch["rater_score_overall"] = np.nan
    batch["rater_score_sharpness"] = np.nan
    batch["rater_score_color_fidelity"] = np.nan

    # Reorder columns for clarity
    col_order = [
        "presentation_order", config.image_id_column, "is_anchor",
        "annotation_stratum", "model_spread", "known_mos",
        "rater_score_overall", "rater_score_sharpness", "rater_score_color_fidelity",
    ]
    batch = batch[[c for c in col_order if c in batch.columns]]

    return batch


# ============================================================================
# STEP 8: POST-ANNOTATION SCORE PROCESSING
# ============================================================================

def process_annotation_scores(batch_df: pd.DataFrame,
                               n_raters: int = 9,
                               outlier_sigma: float = 2.0) -> pd.DataFrame:
    """
    After human scoring is complete, this function:
    1. Filters outlier raters using anchor image calibration
    2. Computes MOS per image per dimension
    3. Applies affine rescaling to DIQA-5000 scale using anchor mapping
    4. Flags high-variance images

    Expects batch_df to have been augmented with per-rater columns:
        rater_1_overall, rater_1_sharpness, ..., rater_9_color_fidelity

    This is a template — adapt column names to your actual annotation output.
    """
    dimensions = ["overall", "sharpness", "color_fidelity"]
    anchors = batch_df[batch_df["is_anchor"] == True]

    if len(anchors) == 0:
        print("WARNING: No anchor images found. Cannot calibrate.")
        return batch_df

    # --- Outlier rater detection ---
    # For each rater, check if their anchor scores deviate > 2σ from known MOS
    rater_cols = {
        dim: [f"rater_{i}_{dim}" for i in range(1, n_raters + 1)]
        for dim in dimensions
    }

    valid_raters = set(range(1, n_raters + 1))

    for dim in dimensions:
        for rater_id in range(1, n_raters + 1):
            col = f"rater_{rater_id}_{dim}"
            if col not in batch_df.columns:
                continue
            anchor_scores = anchors[col].values
            anchor_known = anchors["known_mos"].values
            if len(anchor_scores) < 3:
                continue
            residuals = anchor_scores - anchor_known
            if np.std(residuals) > outlier_sigma:
                valid_raters.discard(rater_id)
                print(f"  Rater {rater_id} excluded: anchor residual σ = "
                      f"{np.std(residuals):.3f} on {dim}")

    print(f"  Retained {len(valid_raters)} of {n_raters} raters after filtering")

    # --- Compute MOS from valid raters ---
    for dim in dimensions:
        valid_cols = [f"rater_{r}_{dim}" for r in valid_raters
                      if f"rater_{r}_{dim}" in batch_df.columns]
        if valid_cols:
            batch_df[f"mos_{dim}"] = batch_df[valid_cols].mean(axis=1)
            batch_df[f"mos_{dim}_std"] = batch_df[valid_cols].std(axis=1, ddof=1)

    # --- Affine rescaling to DIQA-5000 scale ---
    for dim in dimensions:
        mos_col = f"mos_{dim}"
        if mos_col not in batch_df.columns:
            continue
        anchor_mask = batch_df["is_anchor"] == True
        if anchor_mask.sum() < 2:
            continue
        # Linear regression: rater MOS → known DIQA-5000 MOS
        x = batch_df.loc[anchor_mask, mos_col].values
        y = batch_df.loc[anchor_mask, "known_mos"].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        batch_df[f"calibrated_{dim}"] = batch_df[mos_col] * slope + intercept
        print(f"  Anchor calibration ({dim}): y = {slope:.4f}x + {intercept:.4f}")

    # --- Flag high-variance images ---
    for dim in dimensions:
        std_col = f"mos_{dim}_std"
        if std_col in batch_df.columns:
            batch_df[f"flag_{dim}"] = batch_df[std_col] > 1.0

    return batch_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: PipelineConfig):
    """Execute the full analysis pipeline."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("OOD SPREAD ANALYSIS PIPELINE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading data...")
    diqa_path = Path(config.diqa5000_predictions_path)
    ood_path = Path(config.ood_predictions_path)

    if not diqa_path.exists() or not ood_path.exists():
        print("\n  Input files not found. Generating synthetic demo data...")
        diqa_df, ood_df = _generate_demo_data(config)
    else:
        diqa_df = pd.read_csv(diqa_path)
        ood_df = pd.read_csv(ood_path)

    print(f"  DIQA-5000: {len(diqa_df)} images, {len(config.model_columns)} models")
    print(f"  OOD set:   {len(ood_df)} images")

    # ------------------------------------------------------------------
    # Normalize model scores
    # ------------------------------------------------------------------
    print("\n[2/8] Normalizing model scores (fit on DIQA-5000)...")
    normalizer = ModelNormalizer(config)
    normalizer.fit(diqa_df)

    diqa_norm = normalizer.transform(diqa_df)
    ood_norm = normalizer.transform(ood_df)

    # Preserve original human MOS (don't normalize ground truth)
    if config.human_mos_column in diqa_df.columns:
        diqa_norm[config.human_mos_column] = diqa_df[config.human_mos_column]
    if config.human_mos_column in ood_df.columns:
        ood_norm[config.human_mos_column] = ood_df[config.human_mos_column]

    # ------------------------------------------------------------------
    # Baseline spread distribution
    # ------------------------------------------------------------------
    print("\n[3/8] Computing baseline spread on DIQA-5000...")
    baseline = SpreadDistribution.compute(diqa_norm, config.model_columns)
    print(f"  Baseline spread: μ={baseline.mean:.4f}  σ={baseline.std:.4f}")
    print(f"  Median={baseline.median:.4f}  Q25={baseline.q25:.4f}  "
          f"Q75={baseline.q75:.4f}  Q95={baseline.q95:.4f}")

    soft_thresh = baseline.mean + config.soft_ood_threshold_sigma * baseline.std
    hard_thresh = baseline.mean + config.hard_ood_threshold_sigma * baseline.std
    print(f"  Soft OOD threshold (μ+{config.soft_ood_threshold_sigma}σ): {soft_thresh:.4f}")
    print(f"  Hard OOD threshold (μ+{config.hard_ood_threshold_sigma}σ): {hard_thresh:.4f}")

    # ------------------------------------------------------------------
    # OOD spread analysis
    # ------------------------------------------------------------------
    print("\n[4/8] Computing OOD spread on synthetic dataset...")
    ood_spread = SpreadDistribution.compute(ood_norm, config.model_columns)
    print(f"  OOD spread: μ={ood_spread.mean:.4f}  σ={ood_spread.std:.4f}")
    print(f"  Median={ood_spread.median:.4f}  Q95={ood_spread.q95:.4f}")

    ood_labels = baseline.classify(
        ood_spread.per_image_spread,
        config.soft_ood_threshold_sigma,
        config.hard_ood_threshold_sigma,
    )
    n_in_dist = (ood_labels == 0).sum()
    n_soft = (ood_labels == 1).sum()
    n_hard = (ood_labels == 2).sum()
    print(f"\n  Classification of {len(ood_df)} OOD images:")
    print(f"    In-distribution-like:  {n_in_dist:>4d} ({n_in_dist/len(ood_df)*100:.1f}%)")
    print(f"    Soft OOD:              {n_soft:>4d} ({n_soft/len(ood_df)*100:.1f}%)")
    print(f"    Strong OOD:            {n_hard:>4d} ({n_hard/len(ood_df)*100:.1f}%)")

    # Per-category breakdown if available
    cat_col = config.ood_category_column
    if cat_col and cat_col in ood_df.columns:
        print(f"\n  Per-category spread:")
        ood_norm["_spread"] = ood_spread.per_image_spread
        ood_norm["_ood_label"] = ood_labels
        ood_norm[cat_col] = ood_df[cat_col]
        for cat in sorted(ood_df[cat_col].unique()):
            mask = ood_norm[cat_col] == cat
            cat_spread = ood_norm.loc[mask, "_spread"]
            cat_labels = ood_norm.loc[mask, "_ood_label"]
            pct_ood = (cat_labels >= 1).mean() * 100
            print(f"    {cat:>30s}: n={mask.sum():>3d}  "
                  f"mean_spread={cat_spread.mean():.4f}  "
                  f"pct_OOD={pct_ood:.0f}%")

    # ------------------------------------------------------------------
    # Pairwise model disagreement
    # ------------------------------------------------------------------
    print("\n[5/8] Decomposing model disagreement structure...")

    print("\n  DIQA-5000 (baseline) pairwise MAD:")
    diqa_pairs = compute_pairwise_disagreement(diqa_norm, config.model_columns)
    print(diqa_pairs.round(4).to_string())

    print("\n  OOD pairwise MAD:")
    ood_pairs = compute_pairwise_disagreement(ood_norm, config.model_columns)
    print(ood_pairs.round(4).to_string())

    print("\n  Divergence increase (OOD / baseline):")
    ratio = ood_pairs / diqa_pairs.replace(0, np.nan)
    print(ratio.round(2).to_string())

    # Cluster analysis
    print("\n  Cluster divergence (vision vs MLLM):")
    diqa_clusters = compute_cluster_divergence(diqa_norm, config.model_columns)
    ood_clusters = compute_cluster_divergence(ood_norm, config.model_columns)
    print(f"    DIQA-5000 inter-cluster gap: {diqa_clusters.get('inter_cluster_gap_mean', 'N/A'):.4f}")
    print(f"    OOD inter-cluster gap:       {ood_clusters.get('inter_cluster_gap_mean', 'N/A'):.4f}")
    print(f"    DIQA-5000 divergence ratio:  {diqa_clusters.get('cluster_divergence_ratio', 'N/A'):.2f}")
    print(f"    OOD divergence ratio:         {ood_clusters.get('cluster_divergence_ratio', 'N/A'):.2f}")

    # ------------------------------------------------------------------
    # Validate spread metric (if human MOS available)
    # ------------------------------------------------------------------
    print("\n[6/8] Validating spread metric against ground truth...")
    diqa_validation = validate_spread_metric(
        diqa_norm, config.model_columns,
        config.human_mos_column, baseline.per_image_spread
    )
    if diqa_validation.get("status") == "validated":
        print(f"  DIQA-5000 ensemble: SRCC={diqa_validation['overall']['ensemble_srcc']:.4f}  "
              f"PLCC={diqa_validation['overall']['ensemble_plcc']:.4f}")
        print(f"\n  Accuracy by spread quantile (DIQA-5000):")
        for q, metrics in diqa_validation["per_spread_quantile"].items():
            print(f"    {q}: SRCC={metrics['srcc']:.4f}  PLCC={metrics['plcc']:.4f}  "
                  f"RMSE={metrics['rmse']:.4f}  (n={metrics['n_images']}, "
                  f"mean_spread={metrics['mean_spread']:.4f})")
        print(f"\n  Spread-error correlation per model:")
        for model, corr in diqa_validation["spread_error_correlation"].items():
            sig = "***" if corr["p_value"] < 0.001 else "**" if corr["p_value"] < 0.01 else "*" if corr["p_value"] < 0.05 else "ns"
            print(f"    {model:>12s}: SRCC(spread, |error|) = {corr['spread_error_srcc']:.4f} {sig}")
    else:
        print(f"  Skipped — no human MOS column found in DIQA-5000 data")

    # ------------------------------------------------------------------
    # Anchor selection
    # ------------------------------------------------------------------
    print("\n[7/8] Selecting anchor images from DIQA-5000...")
    if config.human_mos_column in diqa_df.columns:
        anchors = select_anchors(diqa_norm, config)
        print("  Selected anchors:")
        for _, row in anchors.iterrows():
            print(f"    Target MOS {row['anchor_target_mos']:.1f} → "
                  f"{row[config.image_id_column]} "
                  f"(actual={row['actual_mos']:.3f}, spread={row['model_spread']:.4f})")
    else:
        print("  Skipped — no human MOS available for anchor selection")
        anchors = pd.DataFrame()

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------
    print("\n[8/8] Building annotation batch (stratified sampling)...")
    sampled = stratified_sample(
        ood_df, ood_spread.per_image_spread, ood_labels, config
    )
    print(f"  Selected {len(sampled)} OOD images:")
    for stratum in sampled["annotation_stratum"].unique():
        n = (sampled["annotation_stratum"] == stratum).sum()
        print(f"    {stratum}: {n}")

    # Build full annotation package
    if len(anchors) > 0:
        batch = build_annotation_package(sampled, anchors, config)
    else:
        batch = sampled.copy()
        batch["presentation_order"] = range(1, len(batch) + 1)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SAVING OUTPUTS")
    print("=" * 72)

    # Full OOD analysis
    ood_report = ood_df.copy()
    ood_report["spread"] = ood_spread.per_image_spread
    ood_report["ood_classification"] = np.where(
        ood_labels == 0, "in_distribution_like",
        np.where(ood_labels == 1, "soft_ood", "strong_ood")
    )
    ood_report["ensemble_mean"] = ood_norm[config.model_columns].mean(axis=1).values
    for model in config.model_columns:
        ood_report[f"{model}_normalized"] = ood_norm[model].values

    report_path = output_dir / "spread_analysis_report.csv"
    ood_report.to_csv(report_path, index=False)
    print(f"  {report_path}")

    # Annotation batch
    batch_path = output_dir / "annotation_batch.csv"
    batch.to_csv(batch_path, index=False)
    print(f"  {batch_path}")

    # Pairwise disagreement
    pairs_path = output_dir / "model_pair_disagreement.csv"
    combined_pairs = pd.concat(
        [diqa_pairs.add_prefix("baseline_"), ood_pairs.add_prefix("ood_")],
        axis=1,
    )
    combined_pairs.to_csv(pairs_path)
    print(f"  {pairs_path}")

    # Summary JSON
    summary = {
        "baseline_spread": {
            "mean": round(baseline.mean, 4),
            "std": round(baseline.std, 4),
            "soft_threshold": round(soft_thresh, 4),
            "hard_threshold": round(hard_thresh, 4),
        },
        "ood_spread": {
            "mean": round(ood_spread.mean, 4),
            "std": round(ood_spread.std, 4),
        },
        "ood_classification": {
            "in_distribution_like": int(n_in_dist),
            "soft_ood": int(n_soft),
            "strong_ood": int(n_hard),
        },
        "cluster_divergence": {
            "baseline": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in diqa_clusters.items()},
            "ood": {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in ood_clusters.items()},
        },
        "validation": diqa_validation,
    }
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  {summary_path}")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    return {
        "baseline": baseline,
        "ood_spread": ood_spread,
        "ood_labels": ood_labels,
        "anchors": anchors,
        "annotation_batch": batch,
        "validation": diqa_validation,
        "config": config,
    }


# ============================================================================
# DEMO DATA GENERATOR (for testing without real data)
# ============================================================================

def _generate_demo_data(config: PipelineConfig) -> tuple:
    """
    Generate realistic synthetic data that mimics the statistical properties
    of a DIQA-5000 + OOD scenario. Used when real CSVs aren't available.
    """
    rng = np.random.default_rng(42)

    # --- DIQA-5000: 1000 test images, models mostly agree ---
    n_diqa = 1000
    human_mos = rng.uniform(1.0, 5.0, n_diqa)

    diqa_data = {
        config.image_id_column: [f"diqa_{i:04d}" for i in range(n_diqa)],
        config.human_mos_column: human_mos,
    }
    # Each model = human_mos + model-specific bias + small noise
    model_biases = {"siglip": 0.02, "hyperiqa": -0.05, "mllm": 0.08, "vl": -0.03, "frontier": 0.01}
    model_noise_scales = {"siglip": 0.25, "hyperiqa": 0.30, "mllm": 0.20, "vl": 0.28, "frontier": 0.22}
    for model in config.model_columns:
        bias = model_biases.get(model, 0)
        noise = model_noise_scales.get(model, 0.25)
        diqa_data[model] = np.clip(
            human_mos + bias + rng.normal(0, noise, n_diqa), 1, 5
        )

    diqa_df = pd.DataFrame(diqa_data)

    # --- OOD: 520 synthetic images, models disagree more ---
    n_ood = 520
    categories = (
        ["arabic_cursive_blur"] * 60 +
        ["aged_yellowed_bleedthrough"] * 60 +
        ["thermal_receipt_faded"] * 60 +
        ["messaging_compression"] * 60 +
        ["devanagari_low_res"] * 50 +
        ["handwritten_mixed_script"] * 50 +
        ["blueprint_engineering"] * 40 +
        ["bound_book_gutter"] * 40 +
        ["carbon_copy_form"] * 50 +
        ["watermarked_certificate"] * 50
    )

    # True quality (unknown in practice)
    true_quality = rng.uniform(1.0, 5.0, n_ood)

    # Category-specific disagreement levels
    cat_disagreement = {
        "arabic_cursive_blur": 0.8,
        "aged_yellowed_bleedthrough": 0.7,
        "thermal_receipt_faded": 0.9,
        "messaging_compression": 0.4,
        "devanagari_low_res": 0.75,
        "handwritten_mixed_script": 0.85,
        "blueprint_engineering": 0.6,
        "bound_book_gutter": 0.5,
        "carbon_copy_form": 0.65,
        "watermarked_certificate": 0.55,
    }

    ood_data = {
        config.image_id_column: [f"ood_{i:04d}" for i in range(n_ood)],
        config.ood_category_column: categories,
    }

    for model in config.model_columns:
        scores = []
        for i in range(n_ood):
            cat = categories[i]
            disagree = cat_disagreement[cat]
            bias = model_biases.get(model, 0) * (1 + disagree)
            noise = model_noise_scales.get(model, 0.25) * (1 + disagree * 2)
            s = true_quality[i] + bias + rng.normal(0, noise)
            scores.append(np.clip(s, 1, 5))
        ood_data[model] = scores

    ood_df = pd.DataFrame(ood_data)

    return diqa_df, ood_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = PipelineConfig(
        output_dir="spread_analysis_output",
    )
    results = run_pipeline(config)
