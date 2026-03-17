"""
Source validation for academic papers and research content.

Ported from NewResearcher (CrewAI) with improvements:
- Credibility scoring (1-10)
- Recency scoring
- Technical depth scoring
- Top-N filtering
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SourceValidationResult:
    """Result of source validation."""
    source_id: str
    title: str
    credibility_score: float  # 0-10
    recency_score: float  # 0-10
    technical_depth_score: float  # 0-10
    overall_score: float  # 0-10
    is_peer_reviewed: bool
    publication_year: Optional[int]
    venue: Optional[str]
    citations: Optional[int]
    validation_notes: List[str] = field(default_factory=list)


class SourceValidator:
    """
    Validator for academic sources with multi-dimensional scoring.

    Scores sources on:
        - Credibility (venue reputation, peer review status)
        - Recency (publication year)
        - Technical depth (equation density, methodology detail)

    Usage:
        validator = SourceValidator()
        results = validator.validate_paper_sources(paper_text, sources)
        top_sources = validator.filter_top_n(results, n=5)
    """

    # Venue reputation scores (simplified)
    TIER_1_VENUES = {
        "nature", "science", "cell", "ieee transactions", "acm transactions",
        "neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl", "emnlp",
        "aaai", "ijcai", "jmlr", "tpami", "ijcv"
    }

    TIER_2_VENUES = {
        "arxiv", "biorxiv", "medrxiv", "ieee", "acm", "springer",
        "elsevier", "wiley", "mdpi", "frontiers"
    }

    def __init__(
        self,
        current_year: Optional[int] = None,
        recency_decay: float = 0.9,
        min_credibility_threshold: float = 3.0
    ):
        """
        Initialize source validator.

        Args:
            current_year: Current year for recency scoring (default: now)
            recency_decay: Decay factor for older papers (default: 0.9)
            min_credibility_threshold: Minimum credibility to pass (default: 3.0)
        """
        self.current_year = current_year or datetime.now().year
        self.recency_decay = recency_decay
        self.min_credibility_threshold = min_credibility_threshold

        logger.info(f"Initialized SourceValidator (current_year={self.current_year})")

    def validate_source(
        self,
        source_text: str,
        source_id: str = "unknown",
        title: Optional[str] = None,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        citations: Optional[int] = None
    ) -> SourceValidationResult:
        """
        Validate a single source.

        Args:
            source_text: Full text of the source
            source_id: Unique identifier
            title: Source title
            venue: Publication venue
            year: Publication year
            citations: Citation count

        Returns:
            SourceValidationResult with scores
        """
        notes = []

        # Calculate credibility score
        credibility = self._score_credibility(venue, source_text)
        if credibility < self.min_credibility_threshold:
            notes.append(f"Low credibility score: {credibility:.1f}")

        # Calculate recency score
        recency = self._score_recency(year)
        if recency < 5:
            notes.append(f"Older paper (year={year}), recency score: {recency:.1f}")

        # Calculate technical depth score
        technical = self._score_technical_depth(source_text)
        if technical < 4:
            notes.append(f"Low technical depth: {technical:.1f}")

        # Calculate overall score (weighted average)
        overall = (
            credibility * 0.4 +
            recency * 0.3 +
            technical * 0.3
        )

        # Check peer review status
        is_peer_reviewed = self._check_peer_reviewed(venue, source_text)

        return SourceValidationResult(
            source_id=source_id,
            title=title or source_id,
            credibility_score=credibility,
            recency_score=recency,
            technical_depth_score=technical,
            overall_score=overall,
            is_peer_reviewed=is_peer_reviewed,
            publication_year=year,
            venue=venue,
            citations=citations,
            validation_notes=notes
        )

    def _score_credibility(self, venue: Optional[str], text: str) -> float:
        """Score source credibility (0-10)."""
        score = 5.0  # Base score

        # Venue-based scoring
        if venue:
            venue_lower = venue.lower()
            if any(v in venue_lower for v in self.TIER_1_VENUES):
                score += 3.0
            elif any(v in venue_lower for v in self.TIER_2_VENUES):
                score += 1.5

        # Check for indicators of credibility in text
        credibility_indicators = [
            r"peer.?review",
            r"doi[\s:]",
            r"arxiv:\d",
            r"published in",
            r"conference",
            r"journal",
        ]
        for indicator in credibility_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                score += 0.5

        return min(score, 10.0)

    def _score_recency(self, year: Optional[int]) -> float:
        """Score source recency (0-10)."""
        if year is None:
            return 5.0  # Unknown year

        years_old = self.current_year - year

        if years_old <= 1:
            return 10.0  # Very recent
        elif years_old <= 3:
            return 9.0
        elif years_old <= 5:
            return 8.0
        elif years_old <= 10:
            return 6.0
        else:
            # Exponential decay for older papers
            return max(2.0, 10.0 * (self.recency_decay ** years_old))

    def _score_technical_depth(self, text: str) -> float:
        """Score technical depth (0-10)."""
        score = 5.0  # Base score

        # Count equations (LaTeX style)
        equation_patterns = [
            r"\$[^$]+\$",  # Inline math
            r"\\\[.*?\\\]",  # Display math
            r"\\begin\{equation\}.*?\\end\{equation\}",
            r"[\^\-_]?\d+",  # Superscripts/subscripts
        ]
        equation_count = sum(len(re.findall(p, text, re.DOTALL)) for p in equation_patterns)

        if equation_count > 20:
            score += 3.0
        elif equation_count > 10:
            score += 2.0
        elif equation_count > 5:
            score += 1.0

        # Check for technical terms
        technical_terms = [
            "algorithm", "methodology", "implementation", "architecture",
            "hyperparameter", "optimization", "gradient", "loss function",
            "neural network", "deep learning", "machine learning",
            "dataset", "training", "validation", "test set",
            "accuracy", "precision", "recall", "f1-score"
        ]
        term_count = sum(1 for term in technical_terms if term.lower() in text.lower())
        score += min(term_count * 0.2, 2.0)

        return min(score, 10.0)

    def _check_peer_reviewed(self, venue: Optional[str], text: str) -> bool:
        """Check if source is peer-reviewed."""
        if venue and any(v in venue.lower() for v in self.TIER_1_VENUES):
            return True

        peer_review_indicators = [
            "peer-reviewed",
            "peer reviewed",
            "refereed",
            "published in",
        ]
        return any(ind in text.lower() for ind in peer_review_indicators)

    def validate_paper_sources(
        self,
        paper_text: str,
        sources: List[Dict[str, Any]]
    ) -> List[SourceValidationResult]:
        """
        Validate multiple sources from a paper.

        Args:
            paper_text: Full paper text for context
            sources: List of source dictionaries

        Returns:
            List of validation results
        """
        results = []

        for source in sources:
            result = self.validate_source(
                source_text=source.get("text", ""),
                source_id=source.get("id", "unknown"),
                title=source.get("title"),
                venue=source.get("venue"),
                year=source.get("year"),
                citations=source.get("citations")
            )
            results.append(result)

        return results

    def filter_top_n(
        self,
        results: List[SourceValidationResult],
        n: int = 5,
        min_score: Optional[float] = None
    ) -> List[SourceValidationResult]:
        """
        Filter to top N sources by overall score.

        Args:
            results: Validation results
            n: Number of sources to keep
            min_score: Optional minimum score threshold

        Returns:
            Top N sources sorted by score
        """
        filtered = results

        if min_score is not None:
            filtered = [r for r in filtered if r.overall_score >= min_score]

        sorted_results = sorted(filtered, key=lambda x: x.overall_score, reverse=True)
        return sorted_results[:n]

    def get_validation_summary(
        self,
        results: List[SourceValidationResult]
    ) -> Dict[str, Any]:
        """Get summary statistics for validation results."""
        if not results:
            return {"error": "No results to summarize"}

        scores = [r.overall_score for r in results]
        credibility = [r.credibility_score for r in results]
        recency = [r.recency_score for r in results]
        technical = [r.technical_depth_score for r in results]

        peer_reviewed = sum(1 for r in results if r.is_peer_reviewed)

        return {
            "total_sources": len(results),
            "peer_reviewed": peer_reviewed,
            "average_overall_score": sum(scores) / len(scores),
            "average_credibility": sum(credibility) / len(credibility),
            "average_recency": sum(recency) / len(recency),
            "average_technical": sum(technical) / len(technical),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "sources_above_threshold": sum(1 for s in scores if s >= self.min_credibility_threshold),
        }


def validate_sources(
    sources: List[Dict[str, Any]],
    paper_text: str = "",
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to validate sources.

    Args:
        sources: List of source dictionaries
        paper_text: Optional paper text for context
        top_n: Number of top sources to return

    Returns:
        Dictionary with top sources and summary
    """
    validator = SourceValidator()
    results = validator.validate_paper_sources(paper_text, sources)
    top = validator.filter_top_n(results, n=top_n)
    summary = validator.get_validation_summary(results)

    return {
        "top_sources": [
            {
                "id": r.source_id,
                "title": r.title,
                "overall_score": r.overall_score,
                "credibility": r.credibility_score,
                "recency": r.recency_score,
                "technical": r.technical_depth_score,
                "peer_reviewed": r.is_peer_reviewed,
                "year": r.publication_year,
            }
            for r in top
        ],
        "summary": summary,
        "all_results": results,
    }
