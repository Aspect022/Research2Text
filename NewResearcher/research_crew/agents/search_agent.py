"""
search_agent.py
---------------
Search Agent — Academic Source Finder

Responsibilities
----------------
Accepts the query list produced by the Planner Agent and executes each
query against the Exa (primary) and Tavily (fallback) search APIs.

For each query the agent should return up to 8 structured sources.
Sources are filtered by the tools to exclude junk domains (SEO blogs,
listicles, etc.).

Output contract
---------------
The agent returns a JSON array of SourceResult objects:

    [
        {
            "title": "...",
            "url": "...",
            "source_type": "paper | repo | documentation | web",
            "published_date": "...",
            "snippet": "..."
        },
        ...
    ]

Design notes
------------
* The agent is armed with both ExaSearchTool and TavilySearchTool so it
  can fall back automatically when one API fails or is rate-limited.
* max_iter is capped at 15 to prevent runaway tool loops on edge cases.
"""

import logging

from crewai import Agent, LLM

from research_crew.tools.search_tool import ExaSearchTool, TavilySearchTool

logger = logging.getLogger(__name__)


def build_search_agent(llm: LLM) -> Agent:
    """Construct and return the Search Agent.

    Parameters
    ----------
    llm:
        A configured CrewAI LLM instance.

    Returns
    -------
    Agent
        Ready-to-use CrewAI Agent with search tools attached.
    """
    logger.info("Initialising Search Agent.")

    exa_tool    = ExaSearchTool()
    tavily_tool = TavilySearchTool()

    return Agent(
        role="Academic Source Finder",
        goal=(
            "Execute each research query against the available search APIs and collect "
            "a diverse set of high-quality sources. For each query, retrieve up to 8 "
            "sources. Prefer arXiv papers, IEEE publications, ACL anthology papers, "
            "GitHub repositories, and official project documentation. "
            "Return all results as a single JSON array of structured source objects."
        ),
        backstory=(
            "You are a specialist academic librarian with deep knowledge of scientific "
            "databases, pre-print servers, and open-source repositories. You excel at "
            "combining search terms to find the most relevant and credible sources. "
            "You always prioritise primary sources (papers, official docs, codebases) "
            "over secondary commentary. When one search API is unavailable, you "
            "immediately switch to the backup without complaint."
        ),
        tools=[exa_tool, tavily_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        max_retry_limit=3,
    )
