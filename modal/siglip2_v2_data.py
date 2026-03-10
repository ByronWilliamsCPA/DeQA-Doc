"""SigLIP2-IQA v2.0 data loading and dataset classes.

Supports DIQA-5000 ground truth data and optional pseudo-label expansion
datasets with configurable mixing strategies.

No Modal dependencies — can be tested locally with mock processors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from modal.siglip2_v2_model import SigLIP2V2Config
except (ImportError, ModuleNotFoundError):
    # When running locally, 'modal' resolves to the pip package, not this dir.
    # Fall back to importlib for the local file.
    import importlib.util
    import os

    import sys as _sys

    _model_path = os.path.join(os.path.dirname(__file__), "siglip2_v2_model.py")
    _spec = importlib.util.spec_from_file_location("siglip2_v2_model", _model_path)
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    _sys.modules["siglip2_v2_model"] = _mod
    _spec.loader.exec_module(_mod)
    SigLIP2V2Config = _mod.SigLIP2V2Config  # type: ignore[misc]

# ============================================================================
# DIQA-5000 Dataset
# ============================================================================


class DIQADataset(Dataset[dict[str, Any]]):
    """DIQA-5000 dataset loading from per-dimension meta JSON files.

    Loads ``train_diqa_overall.json``, ``train_diqa_sharpness.json``, and
    ``train_diqa_color.json`` from the meta directory, merging by image ID.

    Each sample is normalized to [0, 1] via ``(gt_score - 1) / 4``.

    Args:
        meta_dir: Directory containing per-dimension meta JSON files.
        image_root: Root directory prepended to relative image paths.
        processor: HuggingFace AutoProcessor for SigLIP2.
        max_num_patches: Maximum NaFlex patch count.
        split: ``"train"`` or ``"val"`` — determines which meta files to load.
    """

    def __init__(
        self,
        meta_dir: str | Path,
        image_root: str | Path,
        processor: Any,
        max_num_patches: int = 784,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_num_patches = max_num_patches

        meta_dir = Path(meta_dir)
        if split == "train":
            self.samples = self._load_train_metas(meta_dir)
        else:
            self.samples = self._load_val_meta(meta_dir)

    def _load_train_metas(
        self, meta_dir: Path
    ) -> list[dict[str, Any]]:
        """Load and merge per-dimension train meta files by image ID."""
        overall = self._read_json(meta_dir / "train_diqa_overall.json")
        sharpness = self._read_json(meta_dir / "train_diqa_sharpness.json")
        color = self._read_json(meta_dir / "train_diqa_color.json")

        # Index by image ID for merging
        sharp_by_id = {s["id"]: s for s in sharpness}
        color_by_id = {s["id"]: s for s in color}

        merged = []
        for item in overall:
            img_id = item["id"]
            sharp_item = sharp_by_id.get(img_id)
            color_item = color_by_id.get(img_id)
            if sharp_item is None or color_item is None:
                continue
            merged.append({
                "image_id": img_id,
                "image_path": item["image"],
                "overall": item["gt_score"],
                "sharpness": sharp_item["gt_score"],
                "color": color_item["gt_score"],
            })
        return merged

    def _load_val_meta(self, meta_dir: Path) -> list[dict[str, Any]]:
        """Load val/test meta file (single file with all dimensions)."""
        data = self._read_json(meta_dir / "diqa_test.json")
        samples = []
        for item in data:
            samples.append({
                "image_id": item.get("id", item.get("image_id", "")),
                "image_path": item.get("image", item.get("image_path", "")),
                "overall": item.get("overall", item.get("gt_score", 0.0)),
                "sharpness": item.get("sharpness", 0.0),
                "color": item.get("color_fidelity", item.get("color", 0.0)),
            })
        return samples

    @staticmethod
    def _read_json(path: Path) -> list[dict[str, Any]]:
        """Read a JSON file containing a list of dicts."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Normalize MOS from [1, 5] to [0, 1]."""
        return (score - 1.0) / 4.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.image_root / sample["image_path"]
        pil_img = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=pil_img,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
            padding="max_length",
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "spatial_shapes": inputs["spatial_shapes"].squeeze(0),
            "targets": {
                "overall": self._normalize_score(sample["overall"]),
                "sharpness": self._normalize_score(sample["sharpness"]),
                "color": self._normalize_score(sample["color"]),
            },
            "image_id": sample["image_id"],
        }


# ============================================================================
# Pseudo-Label Dataset
# ============================================================================


class PseudoLabelDataset(Dataset[dict[str, Any]]):
    """Dataset from VLM pseudo-labeled expansion data (JSONL format).

    Expected JSONL format per line::

        {
            "image_path": "/path/to/image.jpg",
            "overall": 3.45,
            "sharpness": 3.12,
            "color_fidelity": 3.67,
            "confidence": 0.85,
            "source": "vlm_pseudo_label",
            "labeler": "gemini-3-flash-calibrated"
        }

    Args:
        jsonl_path: Path to the JSONL file.
        processor: HuggingFace AutoProcessor for SigLIP2.
        max_num_patches: Maximum NaFlex patch count.
        confidence_threshold: Minimum confidence to include a sample.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        processor: Any,
        max_num_patches: int = 784,
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.max_num_patches = max_num_patches

        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("confidence", 1.0) >= confidence_threshold:
                    self.samples.append(item)

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Normalize MOS from [1, 5] to [0, 1]."""
        return (score - 1.0) / 4.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pil_img = Image.open(sample["image_path"]).convert("RGB")

        inputs = self.processor(
            images=pil_img,
            return_tensors="pt",
            max_num_patches=self.max_num_patches,
            padding="max_length",
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "spatial_shapes": inputs["spatial_shapes"].squeeze(0),
            "targets": {
                "overall": self._normalize_score(sample["overall"]),
                "sharpness": self._normalize_score(sample["sharpness"]),
                "color": self._normalize_score(
                    sample.get("color_fidelity", sample.get("color", 3.0))
                ),
            },
            "image_id": Path(sample["image_path"]).stem,
            "confidence": sample.get("confidence", 1.0),
        }


# ============================================================================
# Mixed Dataset
# ============================================================================


class MixedDataset(Dataset[dict[str, Any]]):
    """Combines DIQA ground truth with optional pseudo-label datasets.

    Mixing strategies:
    - ``"interleave"``: Alternates between DIQA and pseudo samples.
    - ``"weighted_sample"``: Returns all samples, use WeightedRandomSampler.
    - ``"epoch_alternate"``: DIQA on even epochs, pseudo on odd. Set
      ``dataset.epoch = N`` before each epoch.

    Args:
        diqa: DIQA-5000 dataset (always included).
        pseudo: Optional pseudo-label dataset.
        strategy: Mixing strategy name.
    """

    def __init__(
        self,
        diqa: DIQADataset,
        pseudo: PseudoLabelDataset | None = None,
        strategy: str = "interleave",
    ) -> None:
        super().__init__()
        self.diqa = diqa
        self.pseudo = pseudo
        self.strategy = strategy
        self.epoch = 0  # Set by training loop for epoch_alternate

        if pseudo is None or strategy == "epoch_alternate":
            # For epoch_alternate, length is max of the two
            self._total = max(len(diqa), len(pseudo) if pseudo else 0)
        elif strategy == "interleave":
            self._total = len(diqa) + len(pseudo)
        else:  # weighted_sample
            self._total = len(diqa) + len(pseudo)

    def __len__(self) -> int:
        if self.strategy == "epoch_alternate":
            if self.pseudo is None or self.epoch % 2 == 0:
                return len(self.diqa)
            return len(self.pseudo)
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.pseudo is None:
            return self.diqa[idx]

        if self.strategy == "epoch_alternate":
            if self.epoch % 2 == 0:
                return self.diqa[idx % len(self.diqa)]
            return self.pseudo[idx % len(self.pseudo)]

        if self.strategy == "interleave":
            n_diqa = len(self.diqa)
            if idx < n_diqa:
                return self.diqa[idx]
            return self.pseudo[idx - n_diqa]

        # weighted_sample: index directly, sampler handles weighting
        n_diqa = len(self.diqa)
        if idx < n_diqa:
            return self.diqa[idx]
        return self.pseudo[idx - n_diqa]

    def get_sample_weights(
        self,
        diqa_weight: float = 1.0,
        pseudo_weight: float = 0.5,
    ) -> list[float]:
        """Build per-sample weights for WeightedRandomSampler.

        Args:
            diqa_weight: Weight for DIQA ground truth samples.
            pseudo_weight: Weight for pseudo-labeled samples.

        Returns:
            List of weights, one per sample.
        """
        weights = [diqa_weight] * len(self.diqa)
        if self.pseudo is not None:
            weights.extend([pseudo_weight] * len(self.pseudo))
        return weights


# ============================================================================
# Dataloader Factory
# ============================================================================


def build_dataloaders(
    config: SigLIP2V2Config,
    processor: Any,
    image_root: str | Path = ".",
    num_workers: int = 4,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Build train and validation dataloaders from config.

    Validation always uses DIQA-5000 val split only (no pseudo-labels).

    Args:
        config: Training configuration.
        processor: HuggingFace AutoProcessor for SigLIP2.
        image_root: Root directory for resolving relative image paths.
        num_workers: DataLoader worker count.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    # Training dataset
    diqa_train = DIQADataset(
        meta_dir=config.data.diqa_meta_dir,
        image_root=image_root,
        processor=processor,
        max_num_patches=config.max_num_patches,
        split="train",
    )

    pseudo_train = None
    if config.data.pseudo_label_jsonl is not None:
        pseudo_train = PseudoLabelDataset(
            jsonl_path=config.data.pseudo_label_jsonl,
            processor=processor,
            max_num_patches=config.max_num_patches,
        )

    train_dataset: Dataset[dict[str, Any]]
    sampler = None
    if pseudo_train is not None:
        mixed = MixedDataset(diqa_train, pseudo_train, config.data.mix_strategy)
        train_dataset = mixed
        if config.data.mix_strategy == "weighted_sample":
            weights = mixed.get_sample_weights(
                diqa_weight=1.0,
                pseudo_weight=config.data.pseudo_label_weight,
            )
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
    else:
        train_dataset = diqa_train

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataset — DIQA-5000 val only
    val_dataset = DIQADataset(
        meta_dir=config.data.diqa_meta_dir,
        image_root=image_root,
        processor=processor,
        max_num_patches=config.max_num_patches,
        split="val",
    )

    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
