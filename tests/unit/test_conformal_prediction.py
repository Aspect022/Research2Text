"""Unit tests for conformal prediction."""

import pytest
import numpy as np
from conformal.predictor import ConformalPredictor, ConformalMethodExtractor


class TestConformalPredictor:
    """Test cases for ConformalPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ConformalPredictor(alpha=0.1)
        assert predictor.alpha == 0.1
        assert predictor.coverage == 0.9
        assert predictor.is_calibrated == False

    def test_string_nonconformity_exact_match(self):
        """Test string nonconformity with exact match."""
        predictor = ConformalPredictor()
        score = predictor._string_nonconformity("Hello", "Hello")
        assert score == 0.0

    def test_string_nonconformity_different(self):
        """Test string nonconformity with different strings."""
        predictor = ConformalPredictor()
        score = predictor._string_nonconformity("Hello", "World")
        assert score > 0.0
        assert score <= 1.0

    def test_string_nonconformity_empty(self):
        """Test string nonconformity with empty strings."""
        predictor = ConformalPredictor()
        assert predictor._string_nonconformity("", "") == 0.0
        assert predictor._string_nonconformity("Hello", "") == 1.0
        assert predictor._string_nonconformity("", "Hello") == 1.0

    def test_numeric_nonconformity_exact_match(self):
        """Test numeric nonconformity with exact match."""
        predictor = ConformalPredictor()
        assert predictor._numeric_nonconformity(1.0, 1.0) == 0.0
        assert predictor._numeric_nonconformity(100, 100) == 0.0

    def test_numeric_nonconformity_relative_error(self):
        """Test numeric nonconformity with relative error."""
        predictor = ConformalPredictor()
        score = predictor._numeric_nonconformity(1.1, 1.0)
        assert abs(score - 0.1) < 1e-9  # 10% relative error (with tolerance)

    def test_numeric_nonconformity_zero_true(self):
        """Test numeric nonconformity when true value is zero."""
        predictor = ConformalPredictor()
        assert predictor._numeric_nonconformity(0.0, 0.0) == 0.0
        assert predictor._numeric_nonconformity(1.0, 0.0) == 1.0

    def test_jaccard_nonconformity_exact_match(self):
        """Test Jaccard nonconformity with exact match."""
        predictor = ConformalPredictor()
        score = predictor._jaccard_nonconformity(["a", "b"], ["a", "b"])
        assert score == 0.0

    def test_jaccard_nonconformity_no_overlap(self):
        """Test Jaccard nonconformity with no overlap."""
        predictor = ConformalPredictor()
        score = predictor._jaccard_nonconformity(["a", "b"], ["c", "d"])
        assert score == 1.0

    def test_jaccard_nonconformity_partial_overlap(self):
        """Test Jaccard nonconformity with partial overlap."""
        predictor = ConformalPredictor()
        score = predictor._jaccard_nonconformity(["a", "b"], ["b", "c"])
        # Intersection = {b}, Union = {a, b, c}, Jaccard = 1/3, Nonconformity = 1 - 1/3 = 2/3
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_list_nonconformity(self):
        """Test list nonconformity."""
        predictor = ConformalPredictor()
        score = predictor._list_nonconformity(["a", "b"], ["a", "b"])
        assert score == 0.0

    def test_get_nested_value(self):
        """Test getting nested dictionary values."""
        predictor = ConformalPredictor()
        data = {"a": {"b": {"c": "value"}}}

        assert predictor._get_nested_value(data, "a") == {"b": {"c": "value"}}
        assert predictor._get_nested_value(data, "a.b") == {"c": "value"}
        assert predictor._get_nested_value(data, "a.b.c") == "value"
        assert predictor._get_nested_value(data, "nonexistent") is None
        assert predictor._get_nested_value(data, "a.nonexistent") is None

    def test_calculate_nonconformity(self):
        """Test nonconformity score calculation."""
        predictor = ConformalPredictor()
        predicted = {
            "algorithm_name": "CNN",
            "training": {"learning_rate": 0.01},
            "datasets": ["CIFAR-10"]
        }
        true = {
            "algorithm_name": "CNN",
            "training": {"learning_rate": 0.001},
            "datasets": ["CIFAR-10"]
        }

        scores = predictor._calculate_nonconformity(predicted, true)
        assert "algorithm_name" in scores
        assert "training.learning_rate" in scores
        assert "datasets" in scores

    def test_generate_candidates(self):
        """Test candidate generation."""
        from conformal.predictor import ConformalMethodExtractor

        extractor = ConformalMethodExtractor(lambda x: {})
        candidates = extractor._generate_candidates("training.learning_rate", 0.001, {})

        assert len(candidates) > 1  # Should include variants
        assert 0.001 in candidates  # Original value should be included


class TestConformalMethodExtractor:
    """Test cases for ConformalMethodExtractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        base_extractor = lambda x: {"algorithm_name": "Test"}
        extractor = ConformalMethodExtractor(base_extractor, alpha=0.1)

        assert extractor.base_extractor == base_extractor
        assert extractor.predictor.alpha == 0.1
        assert extractor.is_calibrated == False

    def test_extract_without_calibration(self):
        """Test extraction without calibration (should warn and return None)."""
        base_extractor = lambda x: {"algorithm_name": "Test"}
        extractor = ConformalMethodExtractor(base_extractor)

        # Should raise RuntimeError when not calibrated
        with pytest.raises(RuntimeError):
            extractor.extract("some text")
