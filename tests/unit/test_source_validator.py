"""Unit tests for source validator."""

import pytest
from validation.source_validator import SourceValidator, validate_sources


class TestSourceValidator:
    """Test cases for SourceValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = SourceValidator(current_year=2024, recency_decay=0.9)
        assert validator.current_year == 2024
        assert validator.recency_decay == 0.9

    def test_score_credibility_tier1_venue(self):
        """Test credibility scoring for tier 1 venues."""
        validator = SourceValidator()
        score = validator._score_credibility("NeurIPS", "Some text")
        assert score >= 7.0  # Tier 1 venues should score high

    def test_score_credibility_tier2_venue(self):
        """Test credibility scoring for tier 2 venues."""
        validator = SourceValidator()
        score = validator._score_credibility("arXiv", "Some text")
        assert score >= 5.0  # Tier 2 venues should score medium

    def test_score_credibility_unknown_venue(self):
        """Test credibility scoring for unknown venues."""
        validator = SourceValidator()
        score = validator._score_credibility("Random Blog", "Some text")
        assert score >= 3.0  # Base score

    def test_score_recency_recent(self):
        """Test recency scoring for recent papers."""
        validator = SourceValidator(current_year=2024)
        score = validator._score_recency(2024)
        assert score == 10.0

    def test_score_recency_old(self):
        """Test recency scoring for old papers."""
        validator = SourceValidator(current_year=2024)
        score = validator._score_recency(2000)
        assert score < 5.0  # Old papers should score lower

    def test_score_recency_none(self):
        """Test recency scoring for unknown year."""
        validator = SourceValidator()
        score = validator._score_recency(None)
        assert score == 5.0  # Default for unknown

    def test_score_technical_depth_with_equations(self):
        """Test technical depth scoring with equations."""
        validator = SourceValidator()
        text = "This paper uses $E=mc^2$ and $\\sum_{i=1}^n x_i$ for calculations with gradient descent optimization."
        score = validator._score_technical_depth(text)
        assert score >= 5.0  # Should score higher with equations

    def test_score_technical_depth_with_technical_terms(self):
        """Test technical depth scoring with technical terms."""
        validator = SourceValidator()
        text = "We use gradient descent optimization with neural networks and deep learning."
        score = validator._score_technical_depth(text)
        assert score > 5.0  # Should score higher with technical terms

    def test_check_peer_reviewed(self):
        """Test peer review detection."""
        validator = SourceValidator()
        assert validator._check_peer_reviewed("NeurIPS", "") == True
        assert validator._check_peer_reviewed("Blog", "This is peer-reviewed") == True
        assert validator._check_peer_reviewed("Blog", "Just a blog post") == False

    def test_validate_source(self):
        """Test full source validation."""
        validator = SourceValidator()
        result = validator.validate_source(
            source_text="This is a paper about deep learning.",
            source_id="test_1",
            title="Test Paper",
            venue="NeurIPS",
            year=2023,
            citations=100
        )

        assert result.source_id == "test_1"
        assert result.title == "Test Paper"
        assert result.credibility_score > 0
        assert result.recency_score > 0
        assert result.technical_depth_score > 0
        assert result.overall_score > 0
        assert result.is_peer_reviewed == True

    def test_validate_paper_sources(self, sample_sources):
        """Test validating multiple sources."""
        validator = SourceValidator()
        results = validator.validate_paper_sources("Some paper text", sample_sources)

        assert len(results) == len(sample_sources)
        assert all(r.overall_score > 0 for r in results)

    def test_filter_top_n(self, sample_sources):
        """Test top-N filtering."""
        validator = SourceValidator()
        results = validator.validate_paper_sources("", sample_sources)
        top = validator.filter_top_n(results, n=2)

        assert len(top) <= 2
        assert all(r.overall_score >= top[-1].overall_score for r in top)

    def test_filter_top_n_with_threshold(self, sample_sources):
        """Test filtering with minimum score threshold."""
        validator = SourceValidator()
        results = validator.validate_paper_sources("", sample_sources)
        filtered = validator.filter_top_n(results, n=10, min_score=7.0)

        assert all(r.overall_score >= 7.0 for r in filtered)

    def test_get_validation_summary(self, sample_sources):
        """Test validation summary."""
        validator = SourceValidator()
        results = validator.validate_paper_sources("", sample_sources)
        summary = validator.get_validation_summary(results)

        assert summary["total_sources"] == len(sample_sources)
        assert "average_overall_score" in summary
        assert "highest_score" in summary
        assert "lowest_score" in summary


class TestValidateSources:
    """Test cases for validate_sources convenience function."""

    def test_basic_usage(self, sample_sources):
        """Test basic usage."""
        result = validate_sources(sample_sources, top_n=2)

        assert "top_sources" in result
        assert "summary" in result
        assert len(result["top_sources"]) <= 2

    def test_with_paper_text(self, sample_sources):
        """Test with paper text context."""
        paper_text = "This paper discusses transformers and attention mechanisms."
        result = validate_sources(sample_sources, paper_text=paper_text, top_n=3)

        assert len(result["top_sources"]) > 0
