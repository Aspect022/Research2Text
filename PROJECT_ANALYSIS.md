# Research2Text: Comprehensive Project Analysis

> **Generated on:** February 2, 2026
> **Project Path:** `D:\Projects\Research2Text-main`
> **Version:** Phase 2 (Live) + Multi-Agent Architecture (Implemented)

## 1. Executive Summary

**Research2Text** is an advanced, local-first AI research assistant designed to bridge the gap between academic literature and executable code. It solves two primary problems:
1.  **Information Retrieval**: It allows users to chat with research papers using RAG (Retrieval-Augmented Generation).
2.  **Implementation**: It automatically converts research methodologies, equations, and algorithms described in PDFs into runnable Python/PyTorch code.

The system is built to run entirely locally using **Ollama** for LLMs and **ChromaDB** for vector storage, ensuring data privacy and zero API costs. It recently evolved from a linear pipeline into a sophisticated **Multi-Agent System** comprising 10 specialized agents.

---

## 2. System Architecture

The project operates in three distinct modes/phases:

### Phase 1: RAG-Based Assistant
*   **Goal**: Interactive Q&A with research papers.
*   **Mechanism**:
    1.  **Ingestion**: PyMuPDF extracts text from PDFs.
    2.  **Chunking**: Word-based sliding window (700 words, 100 overlap).
    3.  **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` maps text to 384-dim vectors.
    4.  **Storage**: Vectors stored locally in ChromaDB.
    5.  **Retrieval**: Cosine similarity search (HNSW index) retrieves top-k chunks.
    6.  **Generation**: Local LLM (Ollama) synthesizes answers from context.

### Phase 2: Paper-to-Code Pipeline (Linear)
*   **Goal**: Turn a PDF into a working PyTorch project.
*   **Pipeline Steps**:
    1.  **Method Extraction**: Regex & heuristics identify "Methods" sections.
    2.  **Structuring**: Pydantic schemas structure algorithms, equations, and datasets.
    3.  **Code Synthesis**: Two-tier generation (Template fallback -> LLM generation).
    4.  **Self-Healing Validation**:
        *   Code is executed in a sandbox.
        *   If it fails (syntax/runtime error), the error is fed back to the LLM.
        *   Max 3 attempts to "heal" the code.
    5.  **Export**: Generates a ZIP artifact with code, logs, and a markdown report.

### Phase 3: Multi-Agent Architecture (Advanced)
A sophisticated layer that replaces the linear pipeline with **10 specialized agents** orchestrated by a central controller.

| Agent Name | Responsibility | Key Tech |
| :--- | :--- | :--- |
| **Orchestrator** | Manages workflow, error recovery, and data passing. | Central Logic |
| **Ingest Agent** | Extracts text, images, and metadata from PDFs. | PyMuPDF |
| **Vision Agent** | OCR, image captioning, table extraction. | Tesseract, BLIP, Camelot |
| **Chunking Agent** | Semantic chunking & embedding generation. | SentenceTransformers |
| **Method Extractor** | Structured info extraction (Algo, Equations, Datasets). | LLM, Regex |
| **Equation Agent** | Normalizes LaTeX, converts to SymPy/Python. | SymPy |
| **Dataset Loader** | Canonicalizes dataset names & generates loader code. | Fuzzy Matching |
| **Code Architect** | Synthesizes full project structure (`model.py`, `train.py`). | LLM |
| **Graph Builder** | Constructs a Knowledge Graph (Nodes: Paper, Algo, etc.). | NetworkX |
| **Validator Agent** | AST parsing, syntax checking, import resolution. | AST, Subprocess |
| **Cleaner Agent** | Database hygiene, removes orphans/old chunks. | Chroma API |

---

## 3. Technology Stack

*   **Core Logic**: Python 3.10+
*   **UI**: Streamlit (Dual-tab interface)
*   **LLM Inference**: Ollama (Local models: `gpt-oss:120b`, `mistral`, `llama3`)
*   **Vector DB**: ChromaDB (Persistent, local)
*   **PDF Engine**: PyMuPDF (fitz)
*   **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Validation**: Python `subprocess` (Sandbox), `ast` module
*   **Data Validation**: Pydantic (Strict schemas)
*   **Math**: SymPy (Equation parsing)

---

## 4. Key Capabilities & Performance

### RAG Performance
*   **Chunking**: Optimized for academic text (700 words preserves context).
*   **Retrieval**: Uses Cosine Similarity (Range 0-2).
*   **Context Window**: Character-based counting (approx 4000 chars) for precision.

### Code Generation Stats
*   **Syntax Correctness**: ~98% (via Code Architect).
*   **Import Resolution**: ~97%.
*   **Self-Healing**: Capable of fixing syntax, import, and simple logic errors automatically.
*   **Dataset Handling**: 87% accuracy in canonicalizing names (e.g., mapping "CIFAR10" -> `torchvision.datasets.CIFAR10`).

### Unique Features
*   **Knowledge Graph**: Builds a structured graph of concepts within the paper (Avg 47 nodes/paper).
*   **Vision Support**: Can "read" images and tables in papers (via Vision Agent).
*   **Dual-Mode Export**:
    *   **Full Artifact**: Code + PDF + Logs + Report.
    *   **Code Only**: Minimal Python package.

---

## 5. File Structure Analysis

```text
D:\Projects\Research2Text-main\
├── src/
│   ├── agents/                 # Multi-Agent Implementation
│   │   ├── orchestrator.py     # The "Brain"
│   │   ├── *_agent.py          # Individual specialized agents
│   │   └── base.py             # Message schemas
│   ├── app_streamlit.py        # Main UI entry point
│   ├── paper_to_code.py        # Legacy linear pipeline
│   ├── paper_to_code_multiagent.py # Wrapper for agent system
│   ├── query_rag.py            # RAG logic (Phase 1)
│   ├── validator.py            # Self-healing loop
│   ├── schemas.py              # Pydantic models
│   └── ... (utils, config)
├── data/
│   ├── raw_pdfs/               # Input PDFs
│   ├── chroma_db/              # Vector Storage
│   └── outputs/                # Generated Code & Reports
├── TECHNICAL_DOCUMENTATION.md   # Deep dive docs
├── MULTI_AGENT_ARCHITECTURE.md  # Agent specific docs
└── requirements.txt             # Dependencies
```

## 6. Usage Guide

### Setup
1.  **Install Ollama**: `ollama pull gpt-oss:120b-cloud` (or `mistral`).
2.  **Install Python Deps**: `pip install -r requirements.txt`.

### Running
```bash
streamlit run src/app_streamlit.py
```
*   **Tab 1 (RAG)**: Upload PDF -> Click "Process" -> Chat.
*   **Tab 2 (Code)**: Select Paper -> Check "Use Multi-Agent" -> Generate -> Download ZIP.

### Programmatic
```python
# Multi-Agent Generation
from src.paper_to_code_multiagent import run_paper_to_code
output_path = run_paper_to_code("my_paper_base_name", use_multiagent=True)
```

---

## 7. Conclusion

Research2Text is a robust, privacy-centric tool for researchers. Its transition to a **Multi-Agent Architecture** significantly boosts its ability to handle complex papers by delegating tasks (like equation parsing and dataset loading) to specialized sub-systems. It stands out by not just summarizing papers, but actively attempting to **reproduce** them via code.
