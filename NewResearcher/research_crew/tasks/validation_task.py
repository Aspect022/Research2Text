"""
validation_task.py
------------------
Task definition for the Source Validator Agent.

Receives the raw source list from the Search Task and asks the Validator
to score each source and return only the top 5.
"""

from crewai import Task


def build_validation_task(validator_agent, search_task: Task) -> Task:
    """Create the Validation Task.

    Parameters
    ----------
    validator_agent:
        The Source Validator Agent instance.
    search_task:
        The completed Search Task whose output is the raw source list.

    Returns
    -------
    Task
        Configured CrewAI Task with context dependency on search_task.
    """
    return Task(
        description=(
            "You will receive a JSON array of source objects from the Search Agent.\n\n"
            "For EACH source, assign a quality score from 1–10 using this rubric:\n\n"
            "  Credibility (0–4 points)\n"
            "    4 – arXiv / IEEE / ACL / NeurIPS / ICML / ICLR / CVPR\n"
            "    3 – Other peer-reviewed venue or well-known GitHub repo\n"
            "    2 – Official project documentation\n"
            "    1 – General web page with technical content\n"
            "    0 – Blog, Medium, listicle, SEO content\n\n"
            "  Recency (0–3 points)\n"
            "    3 – Published / committed within the last 12 months\n"
            "    2 – 1–2 years old\n"
            "    1 – 2–5 years old\n"
            "    0 – Older than 5 years OR no date available\n\n"
            "  Technical depth (0–3 points)\n"
            "    3 – Snippet contains quantitative results, datasets, or algorithms\n"
            "    2 – Snippet contains methodology but no numbers\n"
            "    1 – Snippet is descriptive / conceptual\n"
            "    0 – Snippet is vague, marketing-like, or empty\n\n"
            "After scoring ALL sources, keep only the TOP 5 by score.\n"
            "If two sources tie, prefer the more recent one.\n\n"
            "Return ONLY valid JSON — no prose:\n"
            "{\n"
            '  "validated_sources": [\n'
            "    {\n"
            '      "title": "...",\n'
            '      "url": "...",\n'
            '      "source_type": "...",\n'
            '      "published_date": "...",\n'
            '      "snippet": "...",\n'
            '      "score": 9,\n'
            '      "rationale": "NeurIPS 2023 paper with SOTA benchmark results."\n'
            "    }\n"
            "  ]\n"
            "}"
        ),
        expected_output=(
            'A JSON object with key "validated_sources" containing an array of at most 5 '
            "source objects, each with the original fields plus 'score' (int 1–10) and "
            "'rationale' (string). No markdown fences."
        ),
        agent=validator_agent,
        context=[search_task],
    )
