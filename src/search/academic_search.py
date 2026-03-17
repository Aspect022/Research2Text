"""
Academic search tools for paper discovery and validation.

Supports:
- Exa API (exa.ai) - Neural search for research papers
- Tavily API (tavily.com) - AI-powered search
- arXiv API - Open access academic papers
- Semantic Scholar - Academic paper database
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from academic search."""
    title: str
    authors: List[str]
    abstract: Optional[str]
    year: Optional[int]
    venue: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    citations: Optional[int]
    source: str  # Which API returned this
    score: float  # Relevance score


class AcademicSearch:
    """
    Academic paper search across multiple sources.

    Usage:
        search = AcademicSearch()
        results = search.search("transformer architecture", max_results=10)

        # Or search specific sources
        arxiv_results = search.search_arxiv("attention mechanism")
    """

    def __init__(
        self,
        exa_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize academic search.

        Args:
            exa_api_key: Exa API key (or from EXA_API_KEY env var)
            tavily_api_key: Tavily API key (or from TAVILY_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout

        # Check available sources
        self.available_sources = ["arxiv", "semantic_scholar"]
        if self.exa_api_key:
            self.available_sources.append("exa")
        if self.tavily_api_key:
            self.available_sources.append("tavily")

        logger.info(f"AcademicSearch initialized with sources: {self.available_sources}")

    def search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search across all available sources.

        Args:
            query: Search query
            max_results: Maximum results to return
            sources: Specific sources to use (default: all available)
            year_from: Filter by publication year (from)
            year_to: Filter by publication year (to)

        Returns:
            Combined and ranked search results
        """
        sources = sources or self.available_sources
        all_results = []

        if "exa" in sources and self.exa_api_key:
            try:
                results = self.search_exa(query, max_results // 2, year_from, year_to)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Exa search failed: {e}")

        if "tavily" in sources and self.tavily_api_key:
            try:
                results = self.search_tavily(query, max_results // 2)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")

        if "arxiv" in sources:
            try:
                results = self.search_arxiv(query, max_results // 2, year_from, year_to)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"arXiv search failed: {e}")

        if "semantic_scholar" in sources:
            try:
                results = self.search_semantic_scholar(query, max_results // 2, year_from, year_to)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:max_results]

    def search_exa(
        self,
        query: str,
        max_results: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[SearchResult]:
        """Search using Exa API (neural search)."""
        if not self.exa_api_key:
            raise ValueError("Exa API key not configured")

        url = "https://api.exa.ai/search"
        headers = {
            "Authorization": f"Bearer {self.exa_api_key}",
            "Content-Type": "application/json"
        }

        # Build query with filters
        search_query = query
        if year_from or year_to:
            year_filter = []
            if year_from:
                year_filter.append(f"after:{year_from}")
            if year_to:
                year_filter.append(f"before:{year_to}")
            search_query += " " + " ".join(year_filter)

        payload = {
            "query": search_query,
            "numResults": max_results,
            "type": "neural",
            "contents": {
                "text": True,
                "highlights": True
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", "Unknown"),
                authors=item.get("authors", []),
                abstract=item.get("text", "")[:500] if item.get("text") else None,
                year=self._extract_year(item.get("published_date")),
                venue=item.get("source"),
                url=item.get("url"),
                pdf_url=None,
                citations=item.get("citation_count"),
                source="exa",
                score=item.get("score", 0.5)
            )
            results.append(result)

        return results

    def search_tavily(
        self,
        query: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        if not self.tavily_api_key:
            raise ValueError("Tavily API key not configured")

        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}

        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "max_results": max_results
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", "Unknown"),
                authors=[],  # Tavily doesn't provide authors
                abstract=item.get("content", "")[:500],
                year=None,
                venue=None,
                url=item.get("url"),
                pdf_url=None,
                citations=None,
                source="tavily",
                score=item.get("score", 0.5)
            )
            results.append(result)

        return results

    def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[SearchResult]:
        """Search using arXiv API."""
        import xml.etree.ElementTree as ET

        # Build arXiv query
        search_query = quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)

        # Define namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            title_text = title.text.strip() if title is not None else "Unknown"

            summary = entry.find("atom:summary", ns)
            abstract = summary.text.strip() if summary is not None else None

            published = entry.find("atom:published", ns)
            year = self._extract_year(published.text) if published is not None else None

            # Filter by year
            if year_from and year and year < year_from:
                continue
            if year_to and year and year > year_to:
                continue

            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            # Get links
            url = None
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                elif not link.get("type"):
                    url = link.get("href")

            # Calculate relevance score (simplified)
            score = self._calculate_arxiv_score(query, title_text, abstract)

            result = SearchResult(
                title=title_text,
                authors=authors,
                abstract=abstract,
                year=year,
                venue="arXiv",
                url=url,
                pdf_url=pdf_url,
                citations=None,
                source="arxiv",
                score=score
            )
            results.append(result)

        return results

    def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[SearchResult]:
        """Search using Semantic Scholar API."""
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,venue,abstract,citationCount,openAccessPdf"
        }

        if year_from:
            params["year"] = f"{year_from}-"
        if year_to:
            params["year"] = f"-{year_to}"

        response = requests.get(base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("data", []):
            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

            # Get PDF URL if available
            pdf_url = None
            oa_pdf = item.get("openAccessPdf")
            if oa_pdf:
                pdf_url = oa_pdf.get("url")

            result = SearchResult(
                title=item.get("title", "Unknown"),
                authors=authors,
                abstract=item.get("abstract"),
                year=item.get("year"),
                venue=item.get("venue"),
                url=f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                pdf_url=pdf_url,
                citations=item.get("citationCount"),
                source="semantic_scholar",
                score=item.get("citationCount", 0) / 1000 if item.get("citationCount") else 0.5
            )
            results.append(result)

        return results

    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        match = re.search(r'(\d{4})', date_str)
        return int(match.group(1)) if match else None

    def _calculate_arxiv_score(self, query: str, title: str, abstract: Optional[str]) -> float:
        """Calculate relevance score for arXiv results."""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        abstract_words = set(abstract.lower().split()) if abstract else set()

        title_match = len(query_words & title_words) / len(query_words) if query_words else 0
        abstract_match = len(query_words & abstract_words) / len(query_words) if query_words else 0

        return min(1.0, title_match * 0.6 + abstract_match * 0.4)

    def validate_paper_exists(
        self,
        title: str,
        authors: Optional[List[str]] = None
    ) -> Optional[SearchResult]:
        """
        Check if a paper exists in academic databases.

        Args:
            title: Paper title
            authors: Optional list of authors

        Returns:
            SearchResult if found, None otherwise
        """
        # Search by title
        query = title
        if authors:
            query += " " + " ".join(authors[:2])  # Add first 2 authors

        results = self.search(query, max_results=5)

        # Find best match
        for result in results:
            # Simple title similarity check
            if self._title_similarity(title, result.title) > 0.8:
                return result

        return None

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity."""
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        if t1 == t2:
            return 1.0

        # Check if one contains the other
        if t1 in t2 or t2 in t1:
            return 0.9

        # Word overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        if words1 and words2:
            overlap = len(words1 & words2)
            return overlap / max(len(words1), len(words2))

        return 0.0


def search_papers(
    query: str,
    max_results: int = 10,
    sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to search for papers.

    Args:
        query: Search query
        max_results: Maximum results
        sources: Specific sources to use

    Returns:
        Dictionary with results and metadata
    """
    search = AcademicSearch()
    results = search.search(query, max_results=max_results, sources=sources)

    return {
        "query": query,
        "total_results": len(results),
        "sources_used": list(set(r.source for r in results)),
        "results": [
            {
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "venue": r.venue,
                "url": r.url,
                "pdf_url": r.pdf_url,
                "citations": r.citations,
                "source": r.source,
                "score": r.score,
            }
            for r in results
        ]
    }
