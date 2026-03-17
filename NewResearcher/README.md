# Multi-Agent Research Citation Engine

A production-grade AI research assistant built with **CrewAI** in Python.  
Enter any research topic and receive a structured report with accurate citations,
extracted evidence, and verified references — similar to Perplexity Deep Research
or Elicit, but fully open and customisable.

---

## What It Does

1. **Planner Agent** decomposes your topic into 4–6 targeted search queries
2. **Search Agent** retrieves up to 8 sources per query from arXiv, IEEE, ACL, GitHub, and official docs via Exa and Tavily APIs
3. **Validator Agent** scores every source (1–10) on credibility, recency, and technical depth — keeps only the top 5
4. **Extractor Agent** fetches each source (PDF or webpage), chunks the text, and extracts metrics, datasets, findings, and verbatim quotes
5. **Synthesizer Agent** merges all evidence into a structured Markdown report with inline citations — no hallucination, every claim is grounded

---

## Architecture

```
User Topic
    ↓
Planner Agent      → { "queries": [...] }
    ↓
Search Agent       → [ { title, url, source_type, snippet, ... } ]
    ↓
Validator Agent    → { "validated_sources": [ top 5 scored ] }
    ↓
Extractor Agent    → [ { metrics, datasets, key_findings, quotes } ]
    ↓
Synthesizer Agent  → Final Markdown Report
```

All agents communicate via **structured JSON only** — never raw documents.

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd multi-agent-researcher-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `EXA_API_KEY` | ✅ | [Exa](https://exa.ai) neural search key |
| `TAVILY_API_KEY` | Recommended | [Tavily](https://tavily.com) fallback search |
| `LLM_MODEL` | Optional | Model name (default: `gpt-4o`) |
| `LLM_TEMPERATURE` | Optional | Temperature (default: `0.3`) |
| `OUTPUT_FILE` | Optional | Report save path (default: `research_report.md`) |

### 3. Run

```bash
# Interactive mode
python -m research_crew.main

# Topic as argument
python -m research_crew.main --topic "attention mechanisms in transformers"

# Custom output file
python -m research_crew.main --topic "diffusion models" --output diffusion.md
```

---

## Project Structure

```
multi-agent-researcher-2/
├── research_crew/
│   ├── agents/
│   │   ├── planner_agent.py       # Research Strategist
│   │   ├── search_agent.py        # Academic Source Finder
│   │   ├── validator_agent.py     # Source Quality Evaluator
│   │   ├── extractor_agent.py     # Technical Evidence Extractor
│   │   └── synthesizer_agent.py   # Research Writer
│   ├── tasks/
│   │   ├── plannsing_task.py       # Query decomposition task
│   │   ├── search_task.py         # Source retrieval task
│   │   ├── validation_task.py     # Source scoring & filtering task
│   │   ├── extraction_task.py     # Evidence extraction task
│   │   └── summary_task.py        # Final report generation task
│   ├── tools/
│   │   ├── search_tool.py         # Exa + Tavily search tools
│   │   ├── pdf_extractor.py       # PyMuPDF PDF text extractor
│   │   └── web_parser.py          # BeautifulSoup webpage parser
│   ├── utils/
│   │   ├── token_utils.py         # count_tokens, truncate_text
│   │   └── text_chunker.py        # chunk_text with overlap
│   └── main.py                    # CLI entry point & pipeline runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## Token Safety

The system enforces strict limits at every layer:

| Layer | Limit | Mechanism |
|---|---|---|
| Document download | 10 MB | Streaming cap in `pdf_extractor.py` |
| Extracted text per source | 3 000 chars | Hard truncation in tools |
| Text chunks | 800 tokens | `chunk_text()` in `text_chunker.py` |
| Evidence per source | 300 tokens | Agent instruction + task constraint |
| LLM calls | Retried on 429 | `tenacity` exponential backoff |

---

## Output Format

```markdown
# Research Summary: <Topic>

## Key Insights

1. **<Headline>**
   <Supporting evidence, 2–4 sentences.>
   *Source: [1]*

## Methodology Overview
<Concise description drawn from extracted methodology snippets.>

## Benchmarks & Metrics
| Metric | Value | Source |
|--------|-------|--------|
| ...    | ...   | [1]    |

## Sources

[1] <Title>
    <URL>
```

---

## Extending the System

| Goal | Where to change |
|---|---|
| Add a new search backend | `tools/search_tool.py` — create a new `BaseTool` subclass |
| Change number of top sources | `tasks/validation_task.py` — update the "keep TOP N" instruction |
| Support local LLMs (Ollama) | `main.py` `_build_llm()` — swap `LLM(model="ollama/...")` |
| Add memory across sessions | `main.py` Crew constructor — set `memory=True` and configure a vector store |
| Export to PDF | Post-process `research_report.md` with `pandoc` or `weasyprint` |

---

## Requirements

- Python ≥ 3.10
- OpenAI API key with GPT-4o access
- Exa API key (free tier available at [exa.ai](https://exa.ai))
- Tavily API key (optional, free tier at [tavily.com](https://tavily.com))
