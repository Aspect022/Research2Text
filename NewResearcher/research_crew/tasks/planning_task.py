"""
planning_task.py
----------------
Task definition for the Planner Agent.

The task instructs the Planner to produce 4–6 specific search queries from
the user-supplied topic. The output is a strict JSON object consumed by the
Search Task.
"""

from crewai import Task


def build_planning_task(planner_agent, topic: str) -> Task:
    """Create the Planning Task.

    Parameters
    ----------
    planner_agent:
        The Planner Agent instance (from planner_agent.py).
    topic:
        The research topic provided by the user.

    Returns
    -------
    Task
        Configured CrewAI Task ready for execution.
    """
    return Task(
        description=(
            f"The user wants to research the following topic:\n\n"
            f"    TOPIC: {topic}\n\n"
            "Your job is to decompose this topic into exactly 4–6 highly specific, "
            "actionable search queries. Each query should:\n"
            "  • Target a distinct sub-aspect of the topic (foundation, key methods, "
            "    benchmarks, recent advances, open problems, applications)\n"
            "  • Be formulated as a natural-language string that returns relevant "
            "    results when submitted to arXiv, Semantic Scholar, IEEE Xplore, "
            "    ACL Anthology, or GitHub search\n"
            "  • Avoid repetition — each query must cover unique ground\n\n"
            "DO NOT perform any web searches. Your output is a planning artefact only.\n\n"
            "Return ONLY a valid JSON object — no prose, no markdown fences:\n"
            '{\n  "queries": ["query1", "query2", ...]\n}'
        ),
        expected_output=(
            'A valid JSON object with a single key "queries" containing a list of '
            "4–6 specific search query strings. No extra text, no markdown code fences."
        ),
        agent=planner_agent,
    )
