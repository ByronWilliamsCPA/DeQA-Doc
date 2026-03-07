"""Tests for cross_validator.py."""

import numpy as np
import pytest

from src.uncertainty.cross_validator import (
    CrossValidationResult,
    CrossValidator,
    _extract_level_probs_from_deqa,
)


class TestExtractLevelProbs:
    def test_from_probs_dict(self):
        record = {
            "probs": {
                "excellent": 0.1,
                "good": 0.3,
                "fair": 0.4,
                "poor": 0.15,
                "bad": 0.05,
            }
        }
        probs = _extract_level_probs_from_deqa(record, use_openset_probs=True)
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-10
        # Check ordering: index 0 = excellent
        assert probs[0] == pytest.approx(0.1, abs=1e-6)

    def test_from_logits_dict(self):
        record = {
            "logits": {
                "excellent": 1.0,
                "good": 2.0,
                "fair": 0.5,
                "poor": -1.0,
                "bad": -2.0,
            }
        }
        probs = _extract_level_probs_from_deqa(record, use_openset_probs=False)
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-10
        # "good" has highest logit → highest prob
        assert np.argmax(probs) == 1

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="neither"):
            _extract_level_probs_from_deqa({})


class TestCrossValidator:
    @pytest.fixture
    def validator(self):
        """Create validator with known DeQA predictions."""
        level_probs = {
            "overall": {
                "img001.jpg": np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
                "img002.jpg": np.array([0.8, 0.15, 0.05, 0.0, 0.0]),
            },
            "sharpness": {
                "img001.jpg": np.array([0.05, 0.2, 0.5, 0.2, 0.05]),
            },
        }
        return CrossValidator.from_level_probs_dict(level_probs)

    def test_has_prediction(self, validator):
        assert validator.has_prediction("img001.jpg", "overall")
        assert not validator.has_prediction("img999.jpg", "overall")
        assert not validator.has_prediction("img001.jpg", "color")

    def test_get_deqa_probs(self, validator):
        probs = validator.get_deqa_probs("img001.jpg", "overall")
        assert probs.shape == (5,)
        assert probs[0] == pytest.approx(0.1, abs=1e-10)

    def test_get_deqa_probs_missing_raises(self, validator):
        with pytest.raises(KeyError):
            validator.get_deqa_probs("img999.jpg", "overall")

    def test_validate_returns_result(self, validator):
        result = validator.validate(
            image_id="img001.jpg",
            dimension="overall",
            siglip2_mu=3.5,
            siglip2_sigma_sq=0.64,
        )
        assert isinstance(result, CrossValidationResult)
        assert result.image_id == "img001.jpg"
        assert result.dimension == "overall"
        assert result.cross_model_jsd >= 0
        assert result.mos_delta >= 0
        assert result.siglip2_entropy >= 0

    def test_validate_perfect_agreement(self, validator):
        """When SigLIP2 matches DeQA exactly, JSD ≈ 0."""
        # DeQA for img002 is [0.8, 0.15, 0.05, 0, 0] → MOS ≈ 4.75
        result = validator.validate(
            image_id="img002.jpg",
            dimension="overall",
            siglip2_mu=4.75,
            siglip2_sigma_sq=0.01,  # Very small sigma → sharp distribution
        )
        # JSD should be relatively low (though not exactly 0 due to discretization)
        assert result.cross_model_jsd < 0.3

    def test_validate_disagreement(self, validator):
        """When SigLIP2 strongly disagrees with DeQA, JSD should be high."""
        # DeQA for img002 is [0.8, 0.15, 0.05, 0, 0] → MOS ≈ 4.75
        # SigLIP2 says MOS ≈ 1.5 (very different)
        result = validator.validate(
            image_id="img002.jpg",
            dimension="overall",
            siglip2_mu=1.5,
            siglip2_sigma_sq=0.1,
        )
        assert result.cross_model_jsd > 0.3
        assert result.mos_delta > 2.0

    def test_validate_batch(self, validator):
        results = validator.validate_batch(
            [
                {
                    "image_id": "img001.jpg",
                    "dimension": "overall",
                    "mu": 3.5,
                    "sigma_sq": 0.64,
                },
                {
                    "image_id": "img001.jpg",
                    "dimension": "sharpness",
                    "mu": 3.0,
                    "sigma_sq": 0.5,
                },
                {
                    "image_id": "img999.jpg",
                    "dimension": "overall",
                    "mu": 3.0,
                    "sigma_sq": 0.5,
                },
            ]
        )
        assert len(results) == 2  # img999 skipped
