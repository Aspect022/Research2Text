"""
validator_agent.py
------------------
Source Validator Agent — Research Source Quality Evaluator

Responsibilities
----------------
Receives the full list of sources from the Search Agent and scores each one
on a 1–10 scale across four dimensions:

    credibility   – journal/venue reputation, author credentials
    recency       – publication or commit date (newer = higher for fast-moving fields)
    depth         – does the snippet indicate technical substance?
    relevance     – how closely does the source match the original research topic?

Only the top 5 sources (by combined score) are kept and forwarded.

Scoring heuristics (agent reasoning guidance)
---------------------------------------------
    +3  arXiv, IEEE, ACL, NeurIPS, ICML, ICLR, CVPR
    +2  GitHub repo with >100 stars (inferred from description)
    +2  Official project documentation
    -2  Personal blog, Medium, Towards Data Science
    -3  Listicle, SEO bait, "Top N" articles
    +1  Published within the last 2 years
    -1  No date or older than 5 years (for rapidly evolving topics)

Output contract
---------------
    {
        "validated_sources": [
            {
                "title": "...",
                "url": "...",
                "source_type": "paper | repo | documentation | web",
                "published_date": "...",
                "snippet": "...",
                "score": 8,
                "rationale": "Peer-reviewed NeurIPS 2023 paper with benchmark results."
            },
            ...   (up to 5 entries)
        ]
    }

Design notes
------------
* No external tools — scoring is done via LLM reasoning over the JSON list.
* The agent is instructed to return ONLY valid JSON; the task description
  reinforces this contract.
"""

import logging

from crewai import Agent, LLM

logger = logging.getLogger(__name__)


def build_validator_agent(llm: LLM) -> Agent:
    """Construct and return the Source Validator Agent.

    Parameters
    ----------
    llm:
        A configured CrewAI LLM instance.

    Returns
    -------
    Agent
        Ready-to-use CrewAI Agent.
    """
    logger.info("Initialising Source Validator Agent.")

    return Agent(
        role="Research Source Quality Evaluator",
        goal=(
            "Evaluate each source in the provided list and score it from 1–10 based on "
            "credibility, recency, technical depth, and relevance to the research topic. "
            "Return only the top 5 highest-scoring sources as a JSON object. "
            "For each source include the score and a one-sentence rationale."
        ),
        backstory=(
            "You are a peer reviewer with 20 years of experience evaluating research quality "
            "across top-tier ML, NLP, and systems venues. You can instantly distinguish "
            "a genuine academic paper from an SEO-optimised blog post. Your scoring is "
            "rigorous and reproducible: you explicitly check venue reputation, publication "
            "recency, citation signals in the snippet, and whether the source contains "
            "verifiable technical claims rather than vague commentary. You ruthlessly "
            "filter noise — if a source is a listicle or a personal blog without primary "
            "data, it gets a score of 1 and is discarded."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )
