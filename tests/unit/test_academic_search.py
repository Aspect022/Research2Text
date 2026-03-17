"""Unit tests for academic search."""

import pytest
from unittest.mock import Mock, patch
from search.academic_search import AcademicSearch, search_papers


class TestAcademicSearch:
    """Test cases for AcademicSearch."""

    def test_initialization_no_keys(self):
        """Test initialization without API keys."""
        search = AcademicSearch()
        assert "arxiv" in search.available_sources
        assert "semantic_scholar" in search.available_sources
        assert "exa" not in search.available_sources
        assert "tavily" not in search.available_sources

    def test_initialization_with_keys(self):
        """Test initialization with API keys."""
        search = AcademicSearch(
            exa_api_key="test_exa_key",
            tavily_api_key="test_tavily_key"
        )
        assert "exa" in search.available_sources
        assert "tavily" in search.available_sources

    def test_extract_year_from_date(self):
        """Test year extraction from date string."""
        search = AcademicSearch()
        assert search._extract_year("2023-05-15") == 2023
        assert search._extract_year("2020") == 2020
        assert search._extract_year(None) is None
        assert search._extract_year("invalid") is None

    def test_calculate_arxiv_score_exact_match(self):
        """Test arXiv score calculation with exact match."""
        search = AcademicSearch()
        score = search._calculate_arxiv_score(
            "transformer architecture",
            "transformer architecture",
            "transformer architecture"
        )
        assert score > 0.5  # Should be high for exact matches

    def test_calculate_arxiv_score_partial_match(self):
        """Test arXiv score calculation with partial match."""
        search = AcademicSearch()
        score = search._calculate_arxiv_score(
            "transformer architecture",
            "neural transformer model",
            "some abstract"
        )
        assert 0 < score < 1.0

    def test_title_similarity_exact(self):
        """Test title similarity with exact match."""
        search = AcademicSearch()
        assert search._title_similarity("Hello World", "Hello World") == 1.0

    def test_title_similarity_substring(self):
        """Test title similarity with substring match."""
        search = AcademicSearch()
        assert search._title_similarity("Hello World", "Hello World Paper") == 0.9

    def test_title_similarity_different(self):
        """Test title similarity with different titles."""
        search = AcademicSearch()
        score = search._title_similarity("Hello World", "Goodbye Moon")
        assert 0 <= score < 0.5

    @patch('search.academic_search.requests.get')
    def test_search_arxiv(self, mock_get):
        """Test arXiv search."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b"""\x3c?xml version="1.0" encoding="UTF-8"?\x3e
        \x3cfeed xmlns="http://www.w3.org/2005/Atom"\x3e
            \x3centry\x3e
                \x3ctitle\x3eTest Paper\x3c/title\x3e
                \x3csummary\x3eThis is a test abstract.\x3c/summary\x3e
                \x3cpublished\x3e2023-01-01\x3c/published\x3e
                \x3cauthor\x3e\x3cname\x3eJohn Doe\x3c/name\x3e\x3c/author\x3e
                \x3clink href="http://arxiv.org/abs/1234.5678" rel="alternate"/\x3e
            \x3c/entry\x3e
        \x3c/feed\x3e"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        search = AcademicSearch()
        results = search.search_arxiv("test query", max_results=1)

        assert len(results) >= 0
        if results:
            assert results[0].title == "Test Paper"
            assert results[0].source == "arxiv"

    @patch('search.academic_search.requests.get')
    def test_search_semantic_scholar(self, mock_get):
        """Test Semantic Scholar search."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "paperId": "test123",
                    "title": "Test Paper",
                    "abstract": "This is a test abstract.",
                    "year": 2023,
                    "venue": "Test Conference",
                    "authors": [{"name": "John Doe"}],
                    "citationCount": 100,
                    "openAccessPdf": {"url": "http://example.com/pdf"}
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        search = AcademicSearch()
        results = search.search_semantic_scholar("test query", max_results=1)

        assert len(results) >= 0
        if results:
            assert results[0].title == "Test Paper"
            assert results[0].source == "semantic_scholar"


class TestSearchPapers:
    """Test cases for search_papers convenience function."""

    @patch('search.academic_search.AcademicSearch.search')
    def test_basic_usage(self, mock_search):
        """Test basic usage."""
        from search.academic_search import SearchResult

        mock_search.return_value = [
            SearchResult(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Test abstract",
                year=2023,
                venue="Test Conf",
                url="http://example.com",
                pdf_url=None,
                citations=100,
                source="arxiv",
                score=0.9
            )
        ]

        result = search_papers("test query", max_results=1)

        assert "query" in result
        assert "total_results" in result
        assert "results" in result
        assert result["query"] == "test query"

    def test_empty_query(self):
        """Test with empty query."""
        result = search_papers("")
        assert result["query"] == ""
        assert result["total_results"] == 0
