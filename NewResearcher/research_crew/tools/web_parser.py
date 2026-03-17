"""
web_parser.py
-------------
Fetches a webpage and extracts human-readable text using BeautifulSoup.

Extraction strategy
-------------------
1. Prefer <article>, <main>, and <section role="main"> elements — these
   contain primary content on research pages and documentation sites.
2. Fall back to the full <body> with boilerplate tags stripped.
3. Hard-truncate the result to HARD_TEXT_LIMIT characters before returning.

GitHub repos
------------
When the URL is a GitHub repository root, the tool fetches the rendered
README text from the API rather than parsing the HTML UI chrome.
"""

import logging
import re
from typing import Optional, Type

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from crewai.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool  # type: ignore[no-redef]

from research_crew.utils.token_utils import truncate_text

logger = logging.getLogger(__name__)

HARD_TEXT_LIMIT = 3_000   # characters returned to the agent
REQUEST_TIMEOUT = 20       # seconds

# Tags whose content is nearly always boilerplate
STRIP_TAGS = {
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "button", "svg", "img",
    "figure", "iframe", "advertisement",
}


class _WebInput(BaseModel):
    url: str = Field(..., description="URL of the webpage or GitHub repo to parse")


class WebParserTool(BaseTool):
    name: str = "Webpage Text Parser"
    description: str = (
        "Fetch a webpage and extract clean, readable text. "
        "Handles documentation sites, arXiv abstracts, and GitHub repositories. "
        "Returns at most 3 000 characters of the primary content."
    )
    args_schema: Type[BaseModel] = _WebInput

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _run(self, url: str) -> str:
        # Redirect GitHub repo root to README via API
        if _is_github_repo_root(url):
            return _fetch_github_readme(url)

        logger.info("Fetching webpage: %s", url)
        resp = requests.get(
            url,
            headers={
                "User-Agent": "ResearchCrew/1.0 (academic crawler)",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate elements in-place
        for tag in soup(list(STRIP_TAGS)):
            tag.decompose()

        text = _extract_main_content(soup)
        text = _normalise_whitespace(text)
        text = text[:HARD_TEXT_LIMIT]

        logger.info("Webpage parsed: %d chars from %s", len(text), url)
        return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_main_content(soup: BeautifulSoup) -> str:
    """Try semantic containers first; fall back to body."""
    for selector in [
        "article",
        "main",
        '[role="main"]',
        ".content",
        "#content",
        ".post-content",
        ".entry-content",
    ]:
        element = soup.select_one(selector)
        if element:
            return element.get_text(separator="\n")

    body = soup.find("body")
    return body.get_text(separator="\n") if body else soup.get_text(separator="\n")


def _normalise_whitespace(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    # Collapse runs of 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result


def _is_github_repo_root(url: str) -> bool:
    """Return True for URLs like https://github.com/owner/repo (no sub-path)."""
    pattern = r"^https?://github\.com/[^/]+/[^/]+/?$"
    return bool(re.match(pattern, url))


def _fetch_github_readme(repo_url: str) -> str:
    """Use the GitHub API to retrieve the rendered README as plain text."""
    # Extract owner/repo
    match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
    if not match:
        return f"Could not parse GitHub repo URL: {repo_url}"

    api_url = f"https://api.github.com/repos/{match.group(1)}/readme"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    github_token = None  # Optionally read from env

    try:
        resp = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = _normalise_whitespace(resp.text)
        return text[:HARD_TEXT_LIMIT]
    except Exception as exc:
        logger.warning("GitHub README fetch failed for %s: %s", repo_url, exc)
        return f"Could not fetch README for {repo_url}: {exc}"
