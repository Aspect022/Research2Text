"""
synthesizer_agent.py
--------------------
Research Synthesizer Agent — Research Writer

Responsibilities
----------------
Receives the structured evidence records from the Extraction Agent and
produces a well-organised research summary with inline citations.

Anti-hallucination rules (enforced in task prompt and backstory)
----------------------------------------------------------------
* ONLY cite sources that appear in the provided evidence JSON.
* Every factual claim must be followed by [N] where N is the source index.
* If evidence is missing for a claim, omit the claim entirely.

Output format
-------------
The synthesizer emits plain Markdown:

    # Research Summary: <Topic>

    ## Key Insights

    1. **<Insight headline>**
       <Supporting evidence, 2–4 sentences.>
       *Source: [1]*

    2. ...

    ## Methodology Overview
    <Optional section if methodology snippets were extracted.>

    ## Benchmarks & Metrics
    <Table or bullet list of quantitative results, if available.>

    ## Sources

    [1] <Title>
        <URL>

    [2] ...

Design notes
------------
* No external tools — the agent works purely from the evidence JSON context.
* The task's expected_output schema further constrains the format.
* Temperature is low (0.2) to minimise hallucination risk.
"""

import logging

from crewai import Agent, LLM

logger = logging.getLogger(__name__)


def build_synthesizer_agent(llm: LLM) -> Agent:
    """Construct and return the Research Synthesizer Agent.

    Parameters
    ----------
    llm:
        A configured CrewAI LLM instance.

    Returns
    -------
    Agent
        Ready-to-use CrewAI Agent.
    """
    logger.info("Initialising Research Synthesizer Agent.")

    return Agent(
        role="Research Writer",
        goal=(
            "Synthesise the extracted evidence into a clear, well-structured research "
            "summary in Markdown format. Every factual claim must be supported by an "
            "inline citation [N] referring to a source in the provided evidence list. "
            "Never introduce facts that are not present in the evidence. Include sections: "
            "Key Insights, Methodology Overview (if applicable), Benchmarks & Metrics "
            "(if quantitative results are present), and a numbered Sources list."
        ),
        backstory=(
            "You are a senior research writer who has authored survey papers for top-tier "
            "AI conferences. You combine evidence from multiple sources into coherent, "
            "accurate narratives without embellishment or speculation. Your cardinal rule "
            "is: if it is not in the evidence, it does not go in the report. You cite "
            "inline with bracketed numbers [1], [2] etc. and list all sources at the end "
            "with their full titles and URLs. You write for a technical audience: precise, "
            "concise, and jargon-aware without being inaccessible."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_retry_limit=3,
    )
