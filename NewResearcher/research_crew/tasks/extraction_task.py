"""
extraction_task.py
------------------
Task definition for the Evidence Extraction Agent.

For each of the top 5 validated sources the agent fetches the content
(PDF or webpage) and extracts structured technical evidence.

Token budget enforcement
------------------------
The task description explicitly caps the output per source at 300 tokens.
The tools themselves cap input text at 3 000 characters, so the agent
never receives an overwhelming context for a single source.
"""

from crewai import Task


def build_extraction_task(extractor_agent, validation_task: Task) -> Task:
    """Create the Evidence Extraction Task.

    Parameters
    ----------
    extractor_agent:
        The Evidence Extraction Agent instance.
    validation_task:
        The completed Validation Task whose output is the validated source list.

    Returns
    -------
    Task
        Configured CrewAI Task with context dependency on validation_task.
    """
    return Task(
        description=(
            "You will receive a JSON object from the Validator Agent containing a "
            "'validated_sources' array (up to 5 sources).\n\n"
            "For EACH source in the array, follow this exact process:\n\n"
            "  Step 1 — Fetch content\n"
            "    • If source_type is 'paper' and URL contains 'arxiv.org': "
            "      call 'PDF Text Extractor' with the URL\n"
            "    • Otherwise: call 'Webpage Text Parser' with the URL\n"
            "    • If the fetch fails, use the snippet from the source object\n\n"
            "  Step 2 — Extract evidence\n"
            "    From the fetched text, extract ONLY:\n"
            "    a) metrics  – numeric results, accuracy %, BLEU scores, F1, latency ms, etc.\n"
            "    b) datasets – named benchmarks or dataset names mentioned\n"
            "    c) key_findings – 2–3 concise findings (1 sentence each)\n"
            "    d) quotes   – up to 2 verbatim sentences from the source\n\n"
            "  Step 3 — Output budget\n"
            "    Each source record MUST NOT exceed 300 tokens.\n"
            "    If you are over budget, cut key_findings first, then quotes.\n\n"
            "Return a JSON array — one object per source:\n"
            "[\n"
            "  {\n"
            '    "source_id": "<url>",\n'
            '    "title": "...",\n'
            '    "url": "...",\n'
            '    "metrics": ["..."],\n'
            '    "datasets": ["..."],\n'
            '    "key_findings": ["..."],\n'
            '    "quotes": ["\\\"exact quote...\\\""]'
            "\n  }\n"
            "]\n\n"
            "Return ONLY the JSON array. No prose. No markdown fences."
        ),
        expected_output=(
            "A JSON array of evidence objects (one per validated source). "
            "Each object has keys: source_id, title, url, metrics, datasets, "
            "key_findings, quotes. Each object is under 300 tokens. "
            "No markdown fences."
        ),
        agent=extractor_agent,
        context=[validation_task],
    )
