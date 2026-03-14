"""Tests for dataset version manifest system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.expansion.manifest import (
    DatasetManifest,
    QualityDistribution,
    SourceEntry,
    compute_quality_distribution,
)


class TestSourceEntry:
    def test_frozen(self):
        src = SourceEntry(
            name="test", stream="base", count=100,
            label_method="human_gt", weight=1.0,
        )
        with pytest.raises(AttributeError):
            src.count = 200


class TestQualityDistribution:
    def test_defaults(self):
        qd = QualityDistribution()
        assert qd.mos_mean == 0.0


class TestComputeQualityDistribution:
    def test_empty_list(self):
        qd = compute_quality_distribution([])
        assert qd.mos_mean == 0.0

    def test_all_excellent(self):
        qd = compute_quality_distribution([4.8, 4.9, 5.0])
        assert qd.excellent_pct == 100.0
        assert qd.good_pct == 0.0

    def test_mixed_distribution(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        qd = compute_quality_distribution(scores)
        assert qd.excellent_pct == 20.0
        assert qd.good_pct == 20.0
        assert qd.fair_pct == 20.0
        assert qd.poor_pct == 20.0
        assert qd.bad_pct == 20.0
        assert qd.mos_mean == pytest.approx(3.0)

    def test_stats(self):
        scores = [2.0, 3.0, 4.0]
        qd = compute_quality_distribution(scores)
        assert qd.mos_min == pytest.approx(2.0)
        assert qd.mos_max == pytest.approx(4.0)
        assert qd.mos_mean == pytest.approx(3.0)


class TestDatasetManifest:
    def test_add_source(self):
        m = DatasetManifest(version="test", tier=0)
        m.add_source(SourceEntry(
            name="s1", stream="base", count=100,
            label_method="human_gt", weight=1.0,
        ))
        assert m.new_samples == 100
        assert m.total_samples == 100
        assert len(m.sources) == 1

    def test_add_source_with_cost(self):
        m = DatasetManifest(version="test", tier=1)
        m.add_source(SourceEntry(
            name="vlm", stream="stream3", count=500,
            label_method="vlm_consensus", weight=0.5, cost_usd=1.0,
        ))
        assert m.vlm_cost_usd == 1.0

    def test_save_and_load_roundtrip(self):
        m = DatasetManifest(
            version="DIQA-5000_1",
            tier=1,
            parent_version="DIQA-5000_0",
            description="Test manifest",
        )
        m.add_source(SourceEntry(
            name="test_source", stream="stream1", count=100,
            label_method="deterministic", weight=0.7,
        ))
        m.quality_distribution = compute_quality_distribution([2.5, 3.5, 4.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            m.save(tmpdir)
            loaded = DatasetManifest.load(Path(tmpdir) / "manifest.json")

        assert loaded.version == "DIQA-5000_1"
        assert loaded.tier == 1
        assert loaded.parent_version == "DIQA-5000_0"
        assert loaded.total_samples == 100
        assert len(loaded.sources) == 1
        assert loaded.sources[0].name == "test_source"
        assert loaded.quality_distribution.mos_mean == pytest.approx(
            m.quality_distribution.mos_mean
        )

    def test_summary(self):
        m = DatasetManifest(version="DIQA-5000_0", tier=0)
        m.add_source(SourceEntry(
            name="base", stream="base", count=3500,
            label_method="human_gt", weight=1.0,
        ))
        m.quality_distribution = compute_quality_distribution([3.0] * 100)
        summary = m.summary()
        assert "DIQA-5000_0" in summary
        assert "3,500" in summary
        assert "human_gt" in summary
