"""
main.py
-------
Entry point for the Multi-Agent Research Citation Engine.

Usage
-----
    python -m research_crew.main --topic "attention mechanisms in transformers"
    python -m research_crew.main   # will prompt for a topic interactively

Pipeline
--------
    User Topic
        → Planner Agent   (generate queries)
        → Search Agent    (retrieve sources via Exa / Tavily)
        → Validator Agent (score & filter to top 5)
        → Extractor Agent (fetch content & extract evidence)
        → Synthesizer Agent (produce final Markdown report)

Environment variables required
-------------------------------
    OPENAI_API_KEY   – OpenAI API key (used by all LLM calls)
    EXA_API_KEY      – Exa neural search API key
    TAVILY_API_KEY   – Tavily search API key (fallback)

Optional
--------
    LLM_MODEL        – Override model name (default: gpt-4o)
    LLM_TEMPERATURE  – Override temperature (default: 0.3)
    OUTPUT_FILE      – Path to save the final report (default: research_report.md)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything that reads environment variables
load_dotenv()

from crewai import Crew, LLM, Process

# Agents
from research_crew.agents.planner_agent     import build_planner_agent
from research_crew.agents.search_agent      import build_search_agent
from research_crew.agents.validator_agent   import build_validator_agent
from research_crew.agents.extractor_agent   import build_extractor_agent
from research_crew.agents.synthesizer_agent import build_synthesizer_agent

# Tasks
from research_crew.tasks.planning_task    import build_planning_task
from research_crew.tasks.search_task      import build_search_task
from research_crew.tasks.validation_task  import build_validation_task
from research_crew.tasks.extraction_task  import build_extraction_task
from research_crew.tasks.summary_task     import build_summary_task

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("research_crew.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------
def _check_env() -> None:
    """Fail early with a helpful message if required env vars are missing."""
    missing = []
    for var in ("OPENAI_API_KEY", "EXA_API_KEY"):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        logger.error(
            "Missing required environment variables: %s\n"
            "Create a .env file (see .env.example) or export them in your shell.",
            ", ".join(missing),
        )
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning(
            "TAVILY_API_KEY not set — Tavily fallback search will be unavailable."
        )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def _build_llm() -> LLM:
    model       = os.getenv("LLM_MODEL", "gpt-4o")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    logger.info("Using LLM: %s  temperature=%.1f", model, temperature)
    return LLM(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_research_pipeline(topic: str) -> str:
    """Execute the full multi-agent research pipeline for *topic*.

    Parameters
    ----------
    topic:
        The research question or subject to investigate.

    Returns
    -------
    str
        The final Markdown research report.
    """
    logger.info("=" * 70)
    logger.info("Starting research pipeline for topic: %s", topic)
    logger.info("=" * 70)
    start_time = time.time()

    # ── Build shared LLM ────────────────────────────────────────────────────
    llm = _build_llm()

    # ── Instantiate agents ──────────────────────────────────────────────────
    logger.info("[1/5] Initialising Planner Agent …")
    planner_agent    = build_planner_agent(llm)

    logger.info("[2/5] Initialising Search Agent …")
    search_agent     = build_search_agent(llm)

    logger.info("[3/5] Initialising Validator Agent …")
    validator_agent  = build_validator_agent(llm)

    logger.info("[4/5] Initialising Extractor Agent …")
    extractor_agent  = build_extractor_agent(llm)

    logger.info("[5/5] Initialising Synthesizer Agent …")
    synthesizer_agent = build_synthesizer_agent(llm)

    # ── Build tasks (chained via context) ───────────────────────────────────
    logger.info("Building task pipeline …")
    planning_task   = build_planning_task(planner_agent, topic)
    search_task     = build_search_task(search_agent, planning_task)
    validation_task = build_validation_task(validator_agent, search_task)
    extraction_task = build_extraction_task(extractor_agent, validation_task)
    summary_task    = build_summary_task(synthesizer_agent, extraction_task, topic)

    # ── Assemble the Crew ───────────────────────────────────────────────────
    crew = Crew(
        agents=[
            planner_agent,
            search_agent,
            validator_agent,
            extractor_agent,
            synthesizer_agent,
        ],
        tasks=[
            planning_task,
            search_task,
            validation_task,
            extraction_task,
            summary_task,
        ],
        process=Process.sequential,   # tasks execute in the order listed
        verbose=True,
        memory=False,                 # keep stateless; evidence flows via task context
    )

    # ── Execute ─────────────────────────────────────────────────────────────
    logger.info("Kicking off CrewAI pipeline …")
    result = crew.kickoff()

    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.1f seconds.", elapsed)

    # CrewAI returns a CrewOutput object; extract the string
    report = str(result)

    # ── Append runtime metadata ─────────────────────────────────────────────
    metadata = (
        f"\n\n---\n"
        f"*Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
        f"in {elapsed:.0f}s using model `{os.getenv('LLM_MODEL', 'gpt-4o')}`.*\n"
    )
    return report + metadata


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
def _save_report(report: str, topic: str) -> Path:
    output_path = Path(os.getenv("OUTPUT_FILE", "research_report.md"))
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to: %s", output_path.resolve())
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _check_env()

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Citation Engine (CrewAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m research_crew.main --topic "transformer attention mechanisms"\n'
            '  python -m research_crew.main --topic "diffusion models for image synthesis" '
            "--output diffusion_report.md\n"
        ),
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research topic to investigate (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the Markdown report (overrides OUTPUT_FILE env var)",
    )
    args = parser.parse_args()

    # Override output path if passed on CLI
    if args.output:
        os.environ["OUTPUT_FILE"] = args.output

    # Get topic
    topic = args.topic
    if not topic:
        topic = input("\nEnter research topic: ").strip()
    if not topic:
        logger.error("No research topic provided. Exiting.")
        sys.exit(1)

    # Run
    report = run_research_pipeline(topic)

    # Save and print
    output_path = _save_report(report, topic)
    print("\n" + "=" * 70)
    print("RESEARCH REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
