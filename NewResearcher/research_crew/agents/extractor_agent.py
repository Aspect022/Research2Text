"""
extractor_agent.py
------------------
Evidence Extraction Agent — Technical Evidence Extractor

Responsibilities
----------------
This is the most compute-intensive agent. For each validated source the
agent:

    1. Fetches the content (PDF via PDFExtractorTool, webpage via WebParserTool)
    2. Receives pre-chunked, truncated text (≤ 3 000 chars per source)
    3. Extracts only the technical core:
         - quantitative metrics / benchmark numbers
         - dataset names
         - methodology snippets (1–3 sentences)
         - direct quotes (max 2 sentences, verbatim)
         - key findings

Token safety
------------
The agent is instructed NEVER to return more than 300 tokens per source.
The task description further enforces this cap.

Output contract (per source)
-----------------------------
    {
        "source_id": "url or short hash",
        "title": "...",
        "url": "...",
        "metrics": ["accuracy 94.2% on SQuAD 2.0", "..."],
        "datasets": ["SQuAD 2.0", "GLUE", "..."],
        "key_findings": ["...", "..."],
        "quotes": ["\"...exact quote...\""]
    }

Design notes
------------
* Both PDFExtractorTool and WebParserTool already enforce the 3 000-char
  hard limit before text reaches the LLM.
* The agent receives structured source objects, not raw search results.
* max_iter is set to 20 to accommodate up to 5 sources × multiple tool calls.
"""

import logging

from crewai import Agent, LLM

from research_crew.tools.pdf_extractor import PDFExtractorTool
from research_crew.tools.web_parser import WebParserTool

logger = logging.getLogger(__name__)


def build_extractor_agent(llm: LLM) -> Agent:
    """Construct and return the Evidence Extraction Agent.

    Parameters
    ----------
    llm:
        A configured CrewAI LLM instance.

    Returns
    -------
    Agent
        Ready-to-use CrewAI Agent with content-access tools attached.
    """
    logger.info("Initialising Evidence Extraction Agent.")

    pdf_tool = PDFExtractorTool()
    web_tool = WebParserTool()

    return Agent(
        role="Technical Evidence Extractor",
        goal=(
            "For each validated source, fetch the content and extract ONLY the "
            "technical core: numeric metrics, dataset names, methodology descriptions, "
            "and up to 2 verbatim quotes. Discard all boilerplate, ads, and navigation "
            "text. Return a JSON array — one object per source — with keys: "
            "source_id, title, url, metrics, datasets, key_findings, quotes. "
            "Never exceed 300 tokens per source object."
        ),
        backstory=(
            "You are a technical analyst specialising in distilling dense research papers "
            "into machine-readable evidence records. You have processed thousands of arXiv "
            "papers and know exactly where to look for the key contribution: the abstract, "
            "the results table, and the conclusion. You never paraphrase carelessly — if "
            "you quote something, it must be verbatim from the source text. If you cannot "
            "find a metric or dataset name, you leave the field empty rather than guessing. "
            "You are ruthlessly concise: your output is used directly in a citation engine "
            "so accuracy beats completeness every time."
        ),
        tools=[pdf_tool, web_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=20,
        max_retry_limit=3,
    )
