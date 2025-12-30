# Multi-Agent Architecture Implementation

This document describes the new multi-agent architecture implemented based on the NLM_Final.pdf specifications.

## Overview

The system has been converted from a monolithic pipeline to a **9-agent multi-agent system** with an orchestration layer, maintaining full backward compatibility with existing functionality.

## Architecture

### Layer 1: Orchestration Layer

- **Orchestrator** (`src/agents/orchestrator.py`): Central coordinator managing workflow, task dispatch, result aggregation, and error recovery

### Layer 2: Agent Layer (9 Specialized Agents)

1. **Ingest Agent** (`src/agents/ingest_agent.py`)

   - PDF processing and content extraction
   - Text and image extraction
   - Metadata collection

2. **Vision Agent** (`src/agents/vision_agent.py`)

   - OCR using Tesseract
   - Image captioning with BLIP
   - Table extraction with Camelot

3. **Chunking Agent** (`src/agents/chunking_agent.py`)

   - Semantic chunking (750 words, 100 overlap)
   - Embedding generation using Sentence Transformers

4. **Method Extractor Agent** (`src/agents/method_extractor_agent.py`)

   - LLM-based structured information extraction
   - Heuristic fallback
   - Extracts: algorithms, equations, datasets, training configs

5. **Equation Agent** (`src/agents/equation_agent.py`)

   - LaTeX normalization
   - SymPy conversion
   - PyTorch code generation

6. **Dataset Loader Agent** (`src/agents/dataset_loader_agent.py`)

   - Fuzzy matching for dataset canonicalization (87% accuracy)
   - Automatic loader code generation
   - Supports: CIFAR-10, CIFAR-100, MNIST, ImageNet, Cora, CiteSeer, PubMed

7. **Code Architect Agent** (`src/agents/code_architect_agent.py`)

   - Complete Python project synthesis
   - Generates: model.py, train.py, dataset_loader.py, requirements.txt
   - 98% syntax correctness, 97% import resolution

8. **Graph Builder Agent** (`src/agents/graph_builder_agent.py`)

   - Knowledge graph construction
   - Node types: Paper, Section, Concept, Equation, Algorithm, Dataset, Metric, Figure, Table, Citation
   - Average: 47 nodes, 82 edges per paper

9. **Validator Agent** (`src/agents/validator_agent.py`)

   - AST parsing for syntax validation
   - Import resolution checking
   - Error reporting with suggestions

10. **Cleaner Agent** (`src/agents/cleaner_agent.py`) ✨ **NEW!**

- Remove outdated chunks from RAG database
- Clean orphaned entries (in DB but not in files, or vice versa)
- Refresh ChromaDB index
- Base-specific cleanup
- Age-based cleanup (remove chunks older than N days)
- Dry-run mode for safe preview

## Usage

### Streamlit Interface

The Streamlit app now includes a checkbox to enable multi-agent processing:

```python
# In the "Paper → Code" tab
use_multiagent = st.checkbox(
    "Use Multi-Agent Architecture (9 specialized agents)",
    value=False
)
```

### Programmatic Usage

```python
from src.agents.orchestrator import Orchestrator
from pathlib import Path

# Initialize orchestrator
orchestrator = Orchestrator()

# Process a paper
pdf_path = Path("data/raw_pdfs/paper.pdf")
results = orchestrator.process_paper(pdf_path, paper_base="paper")

# Or use the wrapper function
from src.paper_to_code_multiagent import run_paper_to_code
out_dir = run_paper_to_code("paper_base", use_multiagent=True)
```

### Individual Agent Usage

```python
from src.agents.ingest_agent import IngestAgent
from src.agents.base import AgentMessage

agent = IngestAgent()
message = AgentMessage(
    agent_id="user",
    message_type="request",
    payload={"pdf_path": "path/to/paper.pdf"}
)
response = agent.process(message)
```

## Backward Compatibility

The original `paper_to_code.py` function remains unchanged and fully functional. The new multi-agent system is available as an **optional enhancement**:

- **Legacy mode**: `run_paper_to_code(paper_base)` - uses original pipeline
- **Multi-agent mode**: `run_paper_to_code(paper_base, use_multiagent=True)` - uses new architecture

## New Features

1. **Knowledge Graph**: Generates `knowledge_graph.json` with nodes and edges
2. **Enhanced Vision Processing**: OCR, captioning, and table extraction
3. **Better Dataset Handling**: Fuzzy matching and automatic loader generation
4. **Improved Validation**: AST-based syntax checking and import resolution
5. **Modular Design**: Each agent can be improved independently
6. **Database Cleaner**: Remove outdated chunks, refresh index, maintain database hygiene

## File Structure

```
src/
├── agents/
│   ├── __init__.py
│   ├── base.py              # Base classes and message schemas
│   ├── orchestrator.py      # Orchestration layer
│   ├── ingest_agent.py      # Agent 1
│   ├── vision_agent.py      # Agent 2
│   ├── chunking_agent.py    # Agent 3
│   ├── method_extractor_agent.py  # Agent 4
│   ├── equation_agent.py    # Agent 5
│   ├── dataset_loader_agent.py   # Agent 6
│   ├── code_architect_agent.py   # Agent 7
│   ├── graph_builder_agent.py    # Agent 8
│   ├── validator_agent.py   # Agent 9
│   └── cleaner_agent.py      # Agent 10
├── paper_to_code.py         # Original (unchanged)
└── paper_to_code_multiagent.py  # New multi-agent wrapper
```

## Performance Metrics

Based on the paper specifications:

- **Method Extraction**: 85-92% accuracy
- **Code Generation**: 98% syntax correctness, 97% import resolution
- **Dataset Canonicalization**: 87% accuracy
- **Processing Time**: ~45 seconds per 10-page paper
- **Knowledge Graph**: Average 47 nodes, 82 edges per paper

## Future Enhancements

1. Asynchronous agent processing
2. Enhanced BLIP integration for image captioning
3. Im2LaTeX for advanced equation recognition
4. Fine-tuned domain-specific models
5. Multi-paper synthesis and cross-paper reasoning

## Cleaner Agent Usage

The Cleaner Agent helps maintain database hygiene by removing outdated chunks:

### Streamlit UI

- Available in the sidebar under "🧹 Database Cleaner"
- Options: Refresh Index, Clean Old Chunks, Clean Base, Full Clean
- Dry-run mode available for safe preview

### Command Line

```bash
# Refresh the index
python src/clean_rag.py refresh

# Clean chunks older than 30 days (dry run)
python src/clean_rag.py clean_old --days 30 --dry-run

# Clean chunks older than 30 days (apply)
python src/clean_rag.py clean_old --days 30 --apply

# Clean all chunks for a specific base
python src/clean_rag.py clean_base --base "paper_name" --apply

# Full cleanup (old chunks + orphans + refresh)
python src/clean_rag.py full_clean --days 30 --apply
```

### Programmatic

```python
from src.clean_rag import clean_old_chunks, refresh_index, clean_base

# Clean old chunks
result = clean_old_chunks(days_old=30, dry_run=False)

# Refresh index
result = refresh_index()

# Clean specific base
result = clean_base("paper_name", dry_run=False)
```

## Notes

- All agents use standardized `AgentMessage` and `AgentResponse` formats
- Error handling is built into each agent
- The orchestrator manages error recovery and fallbacks
- Existing functionality is preserved - no breaking changes
- **Cleaner Agent** automatically removes outdated chunks and keeps RAG index fresh
