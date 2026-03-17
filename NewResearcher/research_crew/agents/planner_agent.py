"""
planner_agent.py
----------------
Planner Agent — Research Strategist

Responsibilities
----------------
Given a high-level research topic from the user, the Planner breaks it into
4 - 6 focused, specific search queries that the Search Agent can execute.

Output contract
---------------
The agent MUST emit a valid JSON object:

    {
        "queries": [
            "query 1",
            "query 2",
            ...
        ]
    }

Design notes
------------
* No external tools are needed — this agent reasons purely from the topic.
* Temperature is kept moderate (0.4) to balance creativity with precision.
* The backstory primes the agent to prefer technical, measurable angles
  over broad survey questions.
"""

import logging
import os

from crewai import Agent, LLM

logger = logging.getLogger(__name__)


def build_planner_agent(llm: LLM) -> Agent:
    """Construct and return the Planner Agent.

    Parameters
    ----------
    llm:
        A configured CrewAI LLM instance (e.g. wrapping GPT-4o).

    Returns
    -------
    Agent
        Ready-to-use CrewAI Agent.
    """
    logger.info("Initialising Planner Agent.")

    return Agent(
        role="Research Strategist",
        goal=(
            "Decompose a user-supplied research topic into 4–6 precise, specific "
            "search queries that will surface high-quality academic papers, technical "
            "reports, and authoritative documentation. Each query must be targeted "
            "enough to return relevant results on arXiv, IEEE, ACL, GitHub, or "
            "official documentation sites. Avoid vague, overly broad queries."
        ),
        backstory=(
            "You are a senior research strategist with 15 years of experience navigating "
            "academic literature across machine learning, systems engineering, and software "
            "architecture. You know how to translate a broad topic into the exact search "
            "strings that surface seminal papers and state-of-the-art benchmarks. You "
            "always think in terms of: (1) the foundational concept, (2) key techniques, "
            "(3) benchmark datasets, (4) recent advances, (5) real-world applications, "
            "and (6) open problems. You never return vague queries like 'latest AI research' "
            "— every query must be specific and actionable."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )
