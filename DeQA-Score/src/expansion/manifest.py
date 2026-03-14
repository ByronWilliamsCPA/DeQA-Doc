"""Dataset version manifest for DIQA-5000 expansion tracking.

Each dataset version (DIQA-5000_0, DIQA-5000_1, ...) is tracked by a manifest
file that records provenance, source composition, quality distribution, and
training configuration.

Manifest files live alongside the training JSON files:
    Data-DeQA-Score/DIQA-5000_0/manifest.json
    Data-DeQA-Score/DIQA-5000_1/manifest.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SourceEntry:
    """A single data source contributing to a dataset version."""

    name: str
    stream: str  # "base", "stream1_degradation", "stream2_synth", "stream3_vlm"
    count: int
    label_method: str  # "human_gt", "deterministic", "vlm_consensus"
    weight: float  # training loss weight
    description: str = ""
    cost_usd: float = 0.0


@dataclass
class QualityDistribution:
    """Quality tier distribution stats for a dataset version."""

    excellent_pct: float = 0.0  # MOS >= 4.5
    good_pct: float = 0.0  # 3.5 <= MOS < 4.5
    fair_pct: float = 0.0  # 2.5 <= MOS < 3.5
    poor_pct: float = 0.0  # 1.5 <= MOS < 2.5
    bad_pct: float = 0.0  # MOS < 1.5
    mos_mean: float = 0.0
    mos_std: float = 0.0
    mos_min: float = 0.0
    mos_max: float = 0.0


@dataclass
class DatasetManifest:
    """Complete manifest for a versioned DIQA dataset.

    Tracks everything needed to reproduce and audit the dataset:
    provenance, composition, quality distribution, and training config.
    """

    version: str  # e.g. "DIQA-5000_0", "DIQA-5000_1"
    tier: int  # 0 = base, 1 = tier 1, etc.
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    parent_version: str | None = None  # "DIQA-5000_0" for tier 1
    description: str = ""

    # Composition
    sources: list[SourceEntry] = field(default_factory=list)
    total_samples: int = 0
    new_samples: int = 0  # samples added in this tier
    holdout_pct: float = 0.0  # % reserved for validation

    # Quality stats
    quality_distribution: QualityDistribution = field(
        default_factory=QualityDistribution
    )

    # Training config
    training_files: dict[str, str] = field(default_factory=dict)
    # e.g. {"overall": "train_overall.json", "sharpness": "train_sharpness.json"}
    base_upsampling: int = 1  # how many times base data is upsampled

    # VLM labeling
    vlm_models: list[str] = field(default_factory=list)
    vlm_cost_usd: float = 0.0

    # Validation gates
    validation_gates: dict[str, float] = field(default_factory=dict)
    # e.g. {"id_wsrcc_min": 0.69, "ood_wsrcc_min": 0.72}

    def add_source(self, source: SourceEntry) -> None:
        """Add a source entry and update totals."""
        self.sources.append(source)
        self.new_samples += source.count
        self.total_samples += source.count
        if source.cost_usd > 0:
            self.vlm_cost_usd += source.cost_usd

    def save(self, output_dir: str | Path) -> Path:
        """Save manifest to JSON file in the dataset directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"

        data = asdict(self)
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path) -> DatasetManifest:
        """Load manifest from JSON file."""
        with open(manifest_path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        sources = [SourceEntry(**s) for s in data.pop("sources", [])]
        quality = QualityDistribution(**data.pop("quality_distribution", {}))
        manifest = cls(**data, sources=sources, quality_distribution=quality)
        return manifest

    def summary(self) -> str:
        """Human-readable summary of the manifest."""
        lines = [
            f"Dataset: {self.version} (Tier {self.tier})",
            f"Created: {self.created_at}",
            f"Parent: {self.parent_version or 'none'}",
            f"Total samples: {self.total_samples:,}",
            f"New samples: {self.new_samples:,}",
            "",
            "Sources:",
        ]
        for src in self.sources:
            lines.append(
                f"  {src.name}: {src.count:,} ({src.label_method}, w={src.weight})"
            )
        lines.extend([
            "",
            f"Quality: MOS {self.quality_distribution.mos_mean:.2f} "
            f"± {self.quality_distribution.mos_std:.2f} "
            f"[{self.quality_distribution.mos_min:.2f}, "
            f"{self.quality_distribution.mos_max:.2f}]",
            f"VLM cost: ${self.vlm_cost_usd:.2f}",
        ])
        return "\n".join(lines)


def compute_quality_distribution(mos_scores: list[float]) -> QualityDistribution:
    """Compute quality tier percentages from a list of MOS scores.

    Args:
        mos_scores: List of MOS values in [1.0, 5.0].

    Returns:
        QualityDistribution with tier percentages and stats.
    """
    if not mos_scores:
        return QualityDistribution()

    import numpy as np

    arr = np.array(mos_scores)
    n = len(arr)

    return QualityDistribution(
        excellent_pct=round(float(np.sum(arr >= 4.5)) / n * 100, 1),
        good_pct=round(float(np.sum((arr >= 3.5) & (arr < 4.5))) / n * 100, 1),
        fair_pct=round(float(np.sum((arr >= 2.5) & (arr < 3.5))) / n * 100, 1),
        poor_pct=round(float(np.sum((arr >= 1.5) & (arr < 2.5))) / n * 100, 1),
        bad_pct=round(float(np.sum(arr < 1.5)) / n * 100, 1),
        mos_mean=round(float(np.mean(arr)), 4),
        mos_std=round(float(np.std(arr)), 4),
        mos_min=round(float(np.min(arr)), 4),
        mos_max=round(float(np.max(arr)), 4),
    )
