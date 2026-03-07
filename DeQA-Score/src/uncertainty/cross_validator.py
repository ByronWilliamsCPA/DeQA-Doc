"""Cross-model validation comparing SigLIP2 predictions against DeQA public models.

Loads pre-computed DeQA inference results (specialist and ensemble) and compares
them against SigLIP2's predictions using Jensen-Shannon divergence to detect
disagreement that indicates epistemic uncertainty.

DeQA inference output format (JSONL from iqa_eval.py):
    {"image": "path.jpg", "logits": {"excellent": 1.2, ...}, "probs": {"excellent": 0.1, ...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .discrete_metrics import discrete_entropy, discrete_jsd
from .gaussian_to_discrete import (
    gaussian_to_level_probs,
    level_probs_to_mos,
)

# DeQA level names in order: index 0=excellent, index 4=bad
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]


@dataclass(frozen=True)
class CrossValidationResult:
    """Result of comparing SigLIP2 vs DeQA for one image+dimension."""

    image_id: str
    dimension: str
    siglip2_probs: np.ndarray  # (5,) from gaussian_to_level_probs
    siglip2_mu: float
    siglip2_sigma_sq: float
    deqa_probs: np.ndarray  # (5,) from DeQA model
    deqa_mos: float
    cross_model_jsd: float  # JSD(siglip2_probs, deqa_probs)
    mos_delta: float  # |siglip2_mu - deqa_mos|
    siglip2_entropy: float  # H(siglip2_probs)


def _extract_level_probs_from_deqa(
    record: dict,
    use_openset_probs: bool = True,
) -> np.ndarray:
    """Extract 5-element level_probs from a DeQA inference record.

    Args:
        record: Dict with 'logits' and/or 'probs' keyed by level name.
        use_openset_probs: If True, use softmax over raw probs dict.
            If False, use logits and compute softmax ourselves.

    Returns:
        Array (5,) in DeQA convention [excellent→bad].
    """
    if use_openset_probs and "probs" in record:
        raw = np.array([record["probs"].get(name, 0.0) for name in LEVEL_NAMES])
    elif "logits" in record:
        logprobs = np.array([record["logits"].get(name, -1e6) for name in LEVEL_NAMES])
        raw = np.exp(logprobs)
    else:
        msg = f"Record has neither 'probs' nor 'logits': {list(record.keys())}"
        raise ValueError(msg)

    total = raw.sum()
    if total > 0:
        return raw / total
    return np.full(5, 0.2)  # uniform fallback


def _load_deqa_jsonl(path: str | Path) -> dict[str, dict]:
    """Load DeQA JSONL file into {image_id: record} dict.

    Args:
        path: Path to JSONL file from iqa_eval.py.

    Returns:
        Dict mapping image path to the full record.
    """
    records: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record.get("image", record.get("id", ""))
            records[image_id] = record
    return records


class CrossValidator:
    """Compares SigLIP2 predictions against DeQA public model predictions.

    Supports two sources of DeQA predictions:
    - Specialist models: dimension-specific (overall, sharpness, color)
    - Ensemble model: DeQA-Score-Mix3 (overall only)

    For each image+dimension, computes JSD between the two distributions
    as a measure of epistemic uncertainty.

    Args:
        deqa_predictions: Nested dict {dimension: {image_id: record}}.
    """

    def __init__(self, deqa_predictions: dict[str, dict[str, dict]]) -> None:
        self._predictions = deqa_predictions

    @classmethod
    def from_jsonl(
        cls,
        specialist_path: str | None = None,
        ensemble_path: str | None = None,
        dimension_paths: dict[str, str] | None = None,
    ) -> CrossValidator:
        """Load from JSONL files.

        Args:
            specialist_path: Path to specialist_true_labels.jsonl (3 dimensions).
                Each line has dimension-tagged predictions.
            ensemble_path: Path to ensemble_labels.jsonl (overall only).
            dimension_paths: Dict mapping dimension name to JSONL path,
                e.g. {"overall": "path/to/overall.jsonl", ...}.
                This is the most flexible option.

        Returns:
            CrossValidator instance.
        """
        predictions: dict[str, dict[str, dict]] = {}

        if dimension_paths:
            for dim_name, path in dimension_paths.items():
                predictions[dim_name] = _load_deqa_jsonl(path)

        if specialist_path:
            records = _load_deqa_jsonl(specialist_path)
            # Specialist JSONL may have a "dimension" field per record,
            # or may be dimension-specific (one file per dimension)
            for image_id, record in records.items():
                dim = record.get("dimension", "overall")
                if dim not in predictions:
                    predictions[dim] = {}
                predictions[dim][image_id] = record

        if ensemble_path:
            ensemble_records = _load_deqa_jsonl(ensemble_path)
            if "overall" not in predictions:
                predictions["overall"] = {}
            # Ensemble provides an additional overall signal
            for image_id, record in ensemble_records.items():
                # Prefix to distinguish from specialist
                predictions["overall"].setdefault(image_id, record)

        return cls(deqa_predictions=predictions)

    @classmethod
    def from_level_probs_dict(
        cls,
        level_probs: dict[str, dict[str, np.ndarray]],
    ) -> CrossValidator:
        """Create from pre-extracted level_probs arrays.

        Args:
            level_probs: {dimension: {image_id: ndarray(5,)}}.

        Returns:
            CrossValidator instance with synthetic records.
        """
        predictions: dict[str, dict[str, dict]] = {}
        for dim, images in level_probs.items():
            predictions[dim] = {}
            for image_id, probs in images.items():
                predictions[dim][image_id] = {"probs_array": np.asarray(probs)}
        return cls(deqa_predictions=predictions)

    def has_prediction(self, image_id: str, dimension: str) -> bool:
        """Check if a DeQA prediction exists for this image+dimension."""
        return (
            dimension in self._predictions and image_id in self._predictions[dimension]
        )

    def get_deqa_probs(self, image_id: str, dimension: str) -> np.ndarray:
        """Get DeQA level_probs for a specific image+dimension.

        Returns:
            Array (5,) in DeQA convention [excellent→bad].

        Raises:
            KeyError: If no prediction exists.
        """
        if not self.has_prediction(image_id, dimension):
            msg = f"No DeQA prediction for image={image_id}, dimension={dimension}"
            raise KeyError(msg)

        record = self._predictions[dimension][image_id]

        # Support pre-extracted arrays (from from_level_probs_dict)
        if "probs_array" in record:
            return np.asarray(record["probs_array"], dtype=np.float64)

        return _extract_level_probs_from_deqa(record)

    def validate(
        self,
        image_id: str,
        dimension: str,
        siglip2_mu: float,
        siglip2_sigma_sq: float,
    ) -> CrossValidationResult:
        """Compare SigLIP2 prediction against DeQA for one image+dimension.

        Args:
            image_id: Image identifier (path or ID).
            dimension: Quality dimension ("overall", "sharpness", "color").
            siglip2_mu: SigLIP2's predicted mean quality score.
            siglip2_sigma_sq: SigLIP2's predicted variance.

        Returns:
            CrossValidationResult with JSD, MOS delta, entropy.
        """
        sigma = max(np.sqrt(max(siglip2_sigma_sq, 0.0)), 0.1)
        siglip2_probs = gaussian_to_level_probs(siglip2_mu, sigma)
        deqa_probs = self.get_deqa_probs(image_id, dimension)

        deqa_mos = level_probs_to_mos(deqa_probs)
        siglip2_mu_clamped = float(np.clip(siglip2_mu, 1.0, 5.0))

        return CrossValidationResult(
            image_id=image_id,
            dimension=dimension,
            siglip2_probs=siglip2_probs,
            siglip2_mu=siglip2_mu_clamped,
            siglip2_sigma_sq=siglip2_sigma_sq,
            deqa_probs=deqa_probs,
            deqa_mos=deqa_mos,
            cross_model_jsd=discrete_jsd(siglip2_probs, deqa_probs),
            mos_delta=abs(siglip2_mu_clamped - deqa_mos),
            siglip2_entropy=discrete_entropy(siglip2_probs),
        )

    def validate_batch(
        self,
        siglip2_results: list[dict],
    ) -> list[CrossValidationResult]:
        """Validate a batch of SigLIP2 results.

        Args:
            siglip2_results: List of dicts with keys:
                image_id, dimension, mu, sigma_sq.

        Returns:
            List of CrossValidationResult. Skips images without DeQA predictions.
        """
        results = []
        for item in siglip2_results:
            image_id = item["image_id"]
            dimension = item["dimension"]
            if not self.has_prediction(image_id, dimension):
                continue
            results.append(
                self.validate(
                    image_id=image_id,
                    dimension=dimension,
                    siglip2_mu=item["mu"],
                    siglip2_sigma_sq=item["sigma_sq"],
                )
            )
        return results
