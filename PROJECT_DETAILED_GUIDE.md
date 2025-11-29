# 📖 Research2Text - Complete Project Guide

> **A Comprehensive Guide to Understanding Every Aspect of the Research2Text Project**

This guide provides a detailed walkthrough of the entire Research2Text project, explaining every component, file, function, and design decision. Whether you're a developer looking to contribute, a researcher wanting to understand the system, or a student learning about RAG systems and AI-powered code generation, this guide has you covered.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Core Concepts Explained](#3-core-concepts-explained)
4. [Phase 1: RAG System Deep Dive](#4-phase-1-rag-system-deep-dive)
5. [Phase 2: Paper-to-Code Pipeline](#5-phase-2-paper-to-code-pipeline)
6. [Source Files Explained](#6-source-files-explained)
7. [Data Flow Walkthrough](#7-data-flow-walkthrough)
8. [Configuration and Settings](#8-configuration-and-settings)
9. [How to Use the System](#9-how-to-use-the-system)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Future Development](#11-future-development)

---

## 1. Project Overview

### What is Research2Text?

Research2Text is an AI-powered research assistant that does two main things:

1. **Phase 1 - RAG (Retrieval-Augmented Generation)**: Allows you to upload research papers (PDFs), process them, and then ask questions about them in natural language. The system retrieves relevant parts of the paper and uses a local LLM (Large Language Model) to generate answers.

2. **Phase 2 - Paper-to-Code**: Automatically analyzes research papers to extract the methodology, algorithms, and equations, then generates runnable Python code that implements the described approach.

### Why Was This Built?

- **Privacy**: All processing happens locally on your machine - no data leaves your computer
- **Cost-Free**: Uses local LLMs (via Ollama) instead of paid APIs
- **Research Accessibility**: Makes it easier to understand and implement research papers
- **Time-Saving**: Automates the tedious process of manually implementing paper methodologies

### Key Technologies Used

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **PyMuPDF (fitz)** | PDF text extraction | Fast, reliable, handles complex layouts |
| **ChromaDB** | Vector database | Local storage, easy setup, efficient search |
| **Sentence Transformers** | Text embeddings | High-quality semantic vectors, runs locally |
| **Ollama** | Local LLM inference | Privacy, no API costs, easy model management |
| **Streamlit** | Web interface | Rapid development, Python-native |
| **Pydantic** | Data validation | Type safety, easy serialization |

---

## 2. Project Structure

```
Research2Text/
├── .git/                          # Git version control
├── .gitignore                     # Files to ignore in git
├── CONTRIBUTING.md                # Guide for contributors
├── Documents/                     # Design documents
│   ├── base.md                    # Original project concept
│   └── future.md                  # Phase 2 planning document
├── LICENSE                        # MIT License
├── README.md                      # Main project documentation
├── TECHNICAL_DOCUMENTATION.md     # Deep technical details
├── data/                          # Data directory (mostly gitignored)
│   └── outputs/                   # Generated code and artifacts
├── main.py                        # Legacy PyCharm entry point
├── notebooks/                     # Jupyter notebooks
│   ├── __init__.py
│   └── test.ipynb                 # Testing notebook
├── requirements-dev.txt           # Development dependencies
├── requirements.txt               # Production dependencies
├── setup.py                       # Package installation config
└── src/                           # Source code directory
    ├── __init__.py                # Package initializer
    ├── app_streamlit.py           # Main web application
    ├── code_generator.py          # Phase 2: Code generation
    ├── datasets.py                # Dataset helpers (Phase 2.5 stub)
    ├── equation_parser.py         # Equation processing
    ├── export_utils.py            # ZIP export functionality
    ├── index_documents.py         # Vector database indexing
    ├── ingest_pdf.py              # PDF processing pipeline
    ├── linker.py                  # Report generation
    ├── method_extractor.py        # Phase 2: Method extraction
    ├── paper_to_code.py           # Phase 2: Main pipeline
    ├── query_rag.py               # RAG search and answer
    ├── schemas.py                 # Data models (Pydantic)
    ├── utils.py                   # Utility functions
    └── validator.py               # Code validation & self-healing
```

### Directory Explanations

#### `/data/` - Data Storage
This directory stores all processed data:
- `raw_pdfs/` - Uploaded PDF files
- `raw_texts/` - Extracted text and chunks
- `chroma_db/` - Vector database files
- `outputs/` - Generated code and reports

#### `/src/` - Source Code
All Python source code lives here. Each file has a specific responsibility (explained in detail below).

#### `/Documents/` - Design Documents
Contains the original project planning documents that describe the vision and architecture.

#### `/notebooks/` - Jupyter Notebooks
For experimentation and testing. The `test.ipynb` shows basic Ollama integration.

---

## 3. Core Concepts Explained

### 3.1 What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that combines:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using an LLM to generate answers based on retrieved information

**Why RAG instead of just feeding the entire paper to the LLM?**
- LLMs have limited context windows (token limits)
- RAG is more accurate because it focuses on relevant parts
- It's faster and uses less memory

**How RAG works in Research2Text:**
```
User Question → Embed Question → Search Vector DB → Retrieve Top Chunks → Feed to LLM → Generate Answer
```

### 3.2 What are Embeddings?

Embeddings are numerical representations (vectors) of text that capture semantic meaning.

**Example:**
- "machine learning" → [0.23, -0.45, 0.78, ...]  (384 numbers)
- "artificial intelligence" → [0.21, -0.43, 0.81, ...]  (similar numbers!)
- "banana" → [0.89, 0.12, -0.56, ...]  (very different numbers)

**Why use embeddings?**
- Similar concepts have similar vectors
- We can find related text by comparing vectors
- This is called "semantic search"

### 3.3 What is Vector Similarity Search?

When you ask a question, the system:
1. Converts your question to a vector (embedding)
2. Compares it to all stored chunk vectors
3. Returns chunks with the most similar vectors

**Similarity Metrics:**
- **Cosine Similarity**: Measures angle between vectors (used by ChromaDB)
- Range: -1 (opposite) to 1 (identical)
- Research2Text uses cosine distance = 1 - cosine similarity

### 3.4 What is Text Chunking?

Research papers can be very long. We break them into smaller pieces (chunks) because:
- Embeddings work better on shorter text
- Retrieval is more precise with smaller chunks
- It respects LLM context limits

**Research2Text Chunking Strategy:**
- Chunk size: 700 words
- Overlap: 100 words (ensures context isn't lost at boundaries)

**Example:**
```
Original: [word1, word2, ..., word2000]

Chunk 1: [word1, ..., word700]
Chunk 2: [word601, ..., word1300]  (100 word overlap)
Chunk 3: [word1201, ..., word2000]
```

### 3.5 What is Ollama?

Ollama is a tool for running Large Language Models locally on your computer.

**Benefits:**
- No API keys or costs
- Complete privacy
- Works offline
- Easy model switching

**How it works:**
```python
import ollama
response = ollama.chat(model="gpt-oss:120b-cloud", messages=[...])
```

---

## 4. Phase 1: RAG System Deep Dive

### 4.1 Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Chunking  │───▶│   Embeddings    │
│   (PyMuPDF)     │    │  (Word-based)   │    │   (MiniLM-L6)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Ollama LLM     │◀───│   RAG Query     │◀───│   ChromaDB      │
│  (Answer Gen)   │    │   (Retrieval)   │    │  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 4.2 Step-by-Step Process

#### Step 1: PDF Upload and Text Extraction

**File:** `src/utils.py` → `extract_text_from_pdf()`

```python
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open PDF
    try:
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))  # Extract text per page
        text = "\n\n".join(pages)  # Join with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Clean excessive newlines
        return text
    finally:
        doc.close()  # Always close file handle
```

**What happens:**
1. PyMuPDF opens the PDF file
2. Iterates through each page
3. Extracts text preserving reading order
4. Joins pages with paragraph breaks
5. Cleans up formatting artifacts

#### Step 2: Text Chunking

**File:** `src/utils.py` → `chunk_text_by_words()`

```python
def chunk_text_by_words(text, chunk_size_words=700, overlap_words=100):
    words = text.split()  # Split into words
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunks.append(" ".join(words[start:end]))
        
        if end >= len(words):
            break
            
        start = max(0, end - overlap_words)  # Move with overlap
    
    return chunks
```

**What happens:**
1. Splits text into individual words
2. Creates 700-word chunks
3. Each chunk overlaps with the previous by 100 words
4. Returns list of chunk strings

#### Step 3: Vector Database Indexing

**File:** `src/index_documents.py` → `index_all()`

```python
def index_all():
    # Setup ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Configure embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="research_papers",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    # Index chunks in batches
    for base in bases:
        chunks = load_chunks_for_base(base)
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"base": c["base"], "chunk_id": c["chunk_id"]} for c in chunks]
        )
```

**What happens:**
1. Creates/opens persistent ChromaDB client
2. Sets up Sentence Transformer embedding model
3. Creates collection with cosine similarity metric
4. Loads all chunks from text files
5. Inserts chunks with metadata (batch processing for efficiency)

#### Step 4: Query and Retrieval

**File:** `src/query_rag.py` → `retrieve()`

```python
def retrieve(query: str, top_k: int = 5) -> List[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(...)
    collection = client.get_collection(name="research_papers", embedding_function=embed_fn)
    
    # Query returns similar chunks
    results = collection.query(query_texts=[query], n_results=top_k)
    
    # Format results
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]  # Lower = more similar
        })
    return docs
```

**What happens:**
1. Opens ChromaDB connection
2. Embeds the user's query
3. Finds top-k most similar chunks
4. Returns chunks with similarity scores

#### Step 5: Answer Generation

**File:** `src/query_rag.py` → `answer_with_ollama()`

```python
def answer_with_ollama(query: str, context: str, model: str, stream: bool = False):
    prompt = (
        "You are a helpful research assistant. Answer the question using ONLY the provided context.\n"
        "If the answer cannot be found in the context, say 'I cannot find this in the provided paper.'\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    
    if stream:
        return ollama.chat(model=model, stream=True, messages=[{"role": "user", "content": prompt}])
    
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.get("message", {}).get("content", "")
```

**What happens:**
1. Constructs a prompt with retrieved context
2. Sends to Ollama LLM
3. Returns generated answer (streaming or complete)

---

## 5. Phase 2: Paper-to-Code Pipeline

### 5.1 Pipeline Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Paper Full Text  │───▶│ Method Extractor │───▶│ Structured Data  │
│                  │    │ (Regex + Rules)  │    │     (JSON)       │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Final Artifacts │◀───│    Validator     │◀───│  Code Generator  │
│  (ZIP, Report)   │    │  (Self-Healing)  │    │  (LLM/Templates) │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### 5.2 Step-by-Step Process

#### Step 1: Method Section Extraction

**File:** `src/method_extractor.py`

```python
SECTION_HEADERS = [
    r"^\s*methods?\b",
    r"^\s*methodolog(y|ies)\b",
    r"^\s*materials? and methods\b",
]

def find_method_sections(text: str) -> List[str]:
    chunks = re.split(r"\n\s*\n", text)  # Split by paragraphs
    method_like = []
    
    for block in chunks:
        head = block.strip().splitlines()[0].lower() if block.strip().splitlines() else ""
        if any(re.match(pat, head) for pat in SECTION_HEADERS):
            method_like.append(block.strip())
    
    return method_like
```

**What happens:**
1. Defines regex patterns for method section headers
2. Splits text into paragraph blocks
3. Checks if first line matches a method header pattern
4. Returns matching sections

#### Step 2: Entity Extraction

**File:** `src/method_extractor.py`

```python
def extract_method_entities(method_text: str) -> MethodStruct:
    algo = None
    datasets = []
    equations = []
    
    # Detect algorithm names
    if "transformer" in method_text.lower():
        algo = "Transformer"
    
    # Detect dataset names
    if re.search(r"cifar[-\s]?10", method_text, flags=re.I):
        datasets.append("CIFAR-10")
    if re.search(r"cora", method_text, flags=re.I):
        datasets.append("Cora")
    
    # Detect mathematical terms
    for m in re.finditer(r"\bQK^?T\b|softmax|cross[-\s]entropy", method_text, flags=re.I):
        equations.append(m.group(0))
    
    return MethodStruct(
        algorithm_name=algo,
        equations=equations,
        datasets=datasets,
    )
```

**What happens:**
1. Searches for known algorithm patterns
2. Detects common dataset names
3. Extracts mathematical notation
4. Returns structured method data

#### Step 3: Code Generation

**File:** `src/code_generator.py`

The system has two modes:

**Mode 1: Template-Based (Fallback)**
```python
def _pytorch_stub(method: MethodStruct) -> List[GeneratedFile]:
    model_code = """import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)
"""
    # Also generates train.py
    return [GeneratedFile(path="model.py", content=model_code), ...]
```

**Mode 2: LLM-Based (Primary)**
```python
def generate_code(method: MethodStruct, ...) -> List[GeneratedFile]:
    sys_prompt = (
        "You are an AI research engineer. Generate runnable Python source files..."
    )
    
    user_prompt = (
        "Method JSON:\n" + method.model_dump_json(indent=2) + "\n\n"
        "Constraints:\n"
        "- Include all necessary imports.\n"
        "- Provide a small training loop...\n"
    )
    
    resp = ollama.chat(model=model, messages=[...])
    return _parse_files_json(resp.get("message", {}).get("content", ""))
```

**What happens:**
1. Tries LLM-based generation first
2. Falls back to templates if LLM fails
3. Parses JSON response into file objects
4. Returns list of generated files

#### Step 4: Code Validation and Self-Healing

**File:** `src/validator.py`

```python
def self_heal_cycle(base_dir: Path, files: List[GeneratedFile], 
                   method: MethodStruct, max_attempts: int = 3) -> ValidationResult:
    sandbox = base_dir / "sandbox"
    write_files(sandbox, files)
    
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        
        # Run the code
        res = run_and_capture(["python", "train.py"], cwd=sandbox, timeout=90)
        
        # Log output
        (logs_dir / f"attempt_{attempts}.out").write_text(res.stdout or "")
        (logs_dir / f"attempt_{attempts}.err").write_text(res.stderr or "")
        
        if res.returncode == 0:
            return ValidationResult(success=True, attempts=attempts, ...)
        
        # Try to fix errors with LLM
        last_err = res.stderr or res.stdout
        files = _llm_fix(files, last_err)
        write_files(sandbox, files)
    
    return ValidationResult(success=False, attempts=attempts, last_error=last_err, ...)
```

**What happens:**
1. Writes generated files to sandbox directory
2. Attempts to run the code
3. If it fails, sends error to LLM for fixing
4. Retries up to max_attempts times
5. Logs all outputs for debugging

#### Step 5: Report Generation and Export

**File:** `src/linker.py` and `src/export_utils.py`

```python
def build_markdown(paper_meta, method, code_paths, run_logs_dir) -> str:
    md = []
    md.append(f"# 📘 Paper\n{paper_meta.get('title', 'Unknown Title')}")
    md.append(f"**Authors:** {paper_meta.get('authors', '?')}")
    
    md.append("# 🧮 Extracted Method")
    if method.algorithm_name:
        md.append(f"- Algorithm: {method.algorithm_name}")
    # ... more details
    
    md.append("# 🧠 Generated Code")
    for p in code_paths:
        md.append(f"- `{p}`")
    
    return "\n".join(md)
```

```python
def build_artifacts_zip(base_name: str) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        # Add outputs
        out_dir = OUTPUTS_DIR / base_name
        if out_dir.exists():
            for p in out_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(out_dir)
                    zf.write(p, arcname=f"outputs/{base_name}/{rel.as_posix()}")
        # Add text files and PDF
        # ...
    return buf.getvalue()
```

**What happens:**
1. Generates human-readable Markdown report
2. Creates ZIP archive with all artifacts
3. Includes generated code, logs, and source materials

---

## 6. Source Files Explained

### 6.1 `src/app_streamlit.py` - Main Application

**Purpose:** Provides the web interface for both Phase 1 and Phase 2 features.

**Key Components:**

```python
def main():
    st.set_page_config(page_title="Research2Text RAG", page_icon="📄")
    st.title("Research2Text - Local RAG")
    
    # Sidebar for configuration
    with st.sidebar:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        model = st.text_input("Ollama model", value="gpt-oss:120b-cloud")
        top_k = st.slider("Top K chunks", min_value=3, max_value=10, value=5)
        # ...
    
    # Two tabs for different features
    tabs = st.tabs(["RAG Search", "Paper → Code"])
    
    with tabs[0]:
        # RAG search interface
        # ...
    
    with tabs[1]:
        # Paper-to-Code interface
        # ...
```

**Functions:**
- `save_uploaded_pdf(upload)`: Saves uploaded PDF to `data/raw_pdfs/`
- `ingest_pdf(pdf_path)`: Extracts text and creates chunks
- `main()`: Main application entry point

### 6.2 `src/utils.py` - Utility Functions

**Purpose:** Core utility functions for text processing.

**Functions:**
- `extract_text_from_pdf(pdf_path)`: Extracts text from PDF using PyMuPDF
- `chunk_text_by_words(text, chunk_size_words, overlap_words)`: Splits text into overlapping chunks

### 6.3 `src/index_documents.py` - Vector Indexing

**Purpose:** Manages ChromaDB vector database indexing.

**Functions:**
- `load_chunks_for_base(base_name)`: Loads chunk files for a paper
- `index_all()`: Indexes all chunks into ChromaDB

**Key Details:**
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Stores metadata: paper base name, chunk ID
- Batch processing (64 chunks at a time) for efficiency

### 6.4 `src/query_rag.py` - RAG Search and Answer

**Purpose:** Handles retrieval and answer generation.

**Functions:**
- `retrieve(query, top_k)`: Searches vector database
- `format_context(chunks, max_chars)`: Formats retrieved chunks as context
- `answer_with_ollama(query, context, model, stream)`: Generates answer using LLM

### 6.5 `src/schemas.py` - Data Models

**Purpose:** Defines data structures using Pydantic.

**Models:**

```python
class TrainingConfig(BaseModel):
    optimizer: Optional[str] = None
    loss: Optional[str] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None

class MethodStruct(BaseModel):
    algorithm_name: Optional[str] = None
    equations: List[str] = []
    datasets: List[str] = []
    training: TrainingConfig = TrainingConfig()
    inputs: Dict[str, str] = {}
    outputs: Dict[str, str] = {}
    references: List[str] = []

class GeneratedFile(BaseModel):
    path: str
    content: str

class RunResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str

class ValidationResult(BaseModel):
    success: bool
    attempts: int
    last_error: Optional[str] = None
    logs_dir: Optional[str] = None
```

### 6.6 `src/ingest_pdf.py` - PDF Processing

**Purpose:** Command-line PDF processing.

**Usage:**
```bash
python src/ingest_pdf.py  # Processes all PDFs in data/raw_pdfs/
```

### 6.7 `src/method_extractor.py` - Method Extraction

**Purpose:** Extracts methodology from papers.

**Functions:**
- `find_method_sections(text)`: Finds method-related sections
- `extract_method_entities(method_text)`: Extracts algorithms, datasets, equations

### 6.8 `src/code_generator.py` - Code Generation

**Purpose:** Generates Python code from method structures.

**Functions:**
- `_pytorch_stub(method)`: Template-based code generation
- `_parse_files_json(text)`: Parses LLM JSON output
- `generate_code(method, framework, use_llm, model)`: Main generation function

### 6.9 `src/validator.py` - Code Validation

**Purpose:** Validates and fixes generated code.

**Functions:**
- `write_files(dst, files)`: Writes files to disk
- `run_and_capture(cmd, cwd, timeout)`: Runs command and captures output
- `_llm_fix(files, error_text)`: Uses LLM to fix code errors
- `self_heal_cycle(base_dir, files, method, max_attempts)`: Main validation loop

### 6.10 `src/linker.py` - Report Generation

**Purpose:** Creates Markdown reports linking papers to code.

**Functions:**
- `build_markdown(paper_meta, method, code_paths, run_logs_dir)`: Generates report

### 6.11 `src/export_utils.py` - Export Functionality

**Purpose:** Creates ZIP archives for download.

**Functions:**
- `_add_file_to_zip(zf, file_path, arc_prefix)`: Helper to add files
- `build_artifacts_zip(base_name)`: Creates full artifact ZIP
- `list_known_bases()`: Lists all processed papers
- `build_code_zip(base_name)`: Creates code-only ZIP

### 6.12 `src/paper_to_code.py` - Main Pipeline

**Purpose:** Orchestrates the entire Paper-to-Code pipeline.

**Functions:**
- `run_paper_to_code(paper_base)`: Main entry point

### 6.13 `src/equation_parser.py` - Equation Processing

**Purpose:** Processes mathematical equations (stub for future expansion).

**Functions:**
- `normalize_equation_strings(equations)`: Cleans equation strings
- `to_sympy(expr)`: Converts to SymPy symbolic expression

### 6.14 `src/datasets.py` - Dataset Helpers

**Purpose:** Dataset loading stubs for Phase 2.5.

**Functions:**
- `load_cifar10()`: Placeholder for CIFAR-10 loading
- `load_cora()`: Placeholder for Cora dataset loading

---

## 7. Data Flow Walkthrough

### 7.1 Phase 1: RAG Query Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ USER UPLOADS PDF                                                 │
│                                                                  │
│  paper.pdf → save_uploaded_pdf() → data/raw_pdfs/paper.pdf      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ TEXT EXTRACTION                                                  │
│                                                                  │
│  extract_text_from_pdf() → "Full paper text..."                 │
│  chunk_text_by_words() → ["chunk 0...", "chunk 1...", ...]      │
│                                                                  │
│  Saved to:                                                       │
│  - data/raw_texts/paper.txt (full text)                         │
│  - data/raw_texts/paper_chunk_0.txt, paper_chunk_1.txt, ...     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ VECTOR INDEXING                                                  │
│                                                                  │
│  For each chunk:                                                 │
│    text → embedding model → [0.23, -0.45, 0.78, ...]           │
│                                                                  │
│  Stored in ChromaDB:                                            │
│    ID: "paper:0", "paper:1", ...                                │
│    Document: chunk text                                          │
│    Embedding: 384-dim vector                                     │
│    Metadata: {base: "paper", chunk_id: 0}                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ USER ASKS QUESTION                                               │
│                                                                  │
│  "What is the main contribution?"                               │
│                                                                  │
│  Question → embedding → vector similarity search                │
│  Returns: Top 5 most similar chunks                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ANSWER GENERATION                                                │
│                                                                  │
│  Chunks → format_context() → context string                     │
│                                                                  │
│  Prompt = question + context → Ollama LLM → Answer              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Phase 2: Paper-to-Code Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Paper base name (e.g., "attention_paper")                │
│                                                                  │
│  Read: data/raw_texts/attention_paper.txt                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ METHOD EXTRACTION                                                │
│                                                                  │
│  Full text → find_method_sections() → method sections           │
│  Method sections → extract_method_entities() → MethodStruct     │
│                                                                  │
│  MethodStruct = {                                               │
│    algorithm_name: "Transformer",                               │
│    equations: ["softmax", "cross-entropy"],                     │
│    datasets: ["CIFAR-10"],                                      │
│    ...                                                          │
│  }                                                              │
│                                                                  │
│  Saved: data/outputs/attention_paper/method.json                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CODE GENERATION                                                  │
│                                                                  │
│  MethodStruct → generate_code() → List[GeneratedFile]           │
│                                                                  │
│  Files: [                                                        │
│    {path: "model.py", content: "import torch..."},              │
│    {path: "train.py", content: "from model import..."},         │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ VALIDATION & SELF-HEALING                                        │
│                                                                  │
│  write files → data/outputs/attention_paper/sandbox/            │
│  run "python train.py"                                          │
│                                                                  │
│  If error:                                                       │
│    error message → LLM → fixed code → retry                     │
│                                                                  │
│  Logs saved: data/outputs/attention_paper/run_logs/             │
│    - attempt_1.out, attempt_1.err                               │
│    - attempt_2.out, attempt_2.err (if retry needed)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ REPORT GENERATION                                                │
│                                                                  │
│  build_markdown() → report.md                                   │
│                                                                  │
│  Saved: data/outputs/attention_paper/report.md                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Complete Artifact Directory                              │
│                                                                  │
│  data/outputs/attention_paper/                                  │
│  ├── method.json          (extracted method)                    │
│  ├── report.md            (human-readable report)               │
│  ├── sandbox/             (generated code)                      │
│  │   ├── model.py                                               │
│  │   └── train.py                                               │
│  └── run_logs/            (execution logs)                      │
│      ├── attempt_1.out                                          │
│      └── attempt_1.err                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration and Settings

### 8.1 Model Settings

| Setting | Default Value | Location | Description |
|---------|---------------|----------|-------------|
| Ollama Model | `gpt-oss:120b-cloud` | app_streamlit.py | LLM for answer generation |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | index_documents.py | Embedding model |
| Top-K Retrieval | 5 | app_streamlit.py | Number of chunks to retrieve |

### 8.2 Chunking Settings

| Setting | Default Value | Location | Description |
|---------|---------------|----------|-------------|
| Chunk Size | 700 words | utils.py | Maximum words per chunk |
| Overlap | 100 words | utils.py | Words shared between chunks |

### 8.3 Phase 2 Settings

| Setting | Default Value | Location | Description |
|---------|---------------|----------|-------------|
| Max Validation Attempts | 3 | paper_to_code.py | Maximum code fix attempts |
| Execution Timeout | 90 seconds | validator.py | Max time for code execution |
| Framework | PyTorch | code_generator.py | Target code framework |

### 8.4 Directory Paths

All paths are relative to the project root:

| Directory | Purpose |
|-----------|---------|
| `data/raw_pdfs/` | Uploaded PDF files |
| `data/raw_texts/` | Extracted text and chunks |
| `data/chroma_db/` | ChromaDB vector database |
| `data/outputs/` | Generated code and reports |

---

## 9. How to Use the System

### 9.1 Installation

```bash
# Clone the repository
git clone https://github.com/Aspect022/Research2Text.git
cd Research2Text

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
# Visit https://ollama.ai for installation
ollama pull gpt-oss:120b-cloud
```

### 9.2 Running the Web Interface

```bash
cd Research2Text
streamlit run src/app_streamlit.py
```

Then open http://localhost:8501 in your browser.

### 9.3 Using RAG Search (Phase 1)

1. **Upload PDF**: Use the sidebar to upload a research paper
2. **Process**: Click "Process PDF" button
3. **Wait**: System extracts text, chunks it, and indexes
4. **Ask Questions**: Type your question and click "Search"
5. **View Results**: See relevant chunks and generated answer

### 9.4 Using Paper-to-Code (Phase 2)

1. **Select Paper**: Upload new PDF or select from existing
2. **Generate**: Click "Generate Code" button
3. **Wait**: System extracts methods and generates code
4. **Download**: Click "Download Artifacts Zip" or "Download Code Only ZIP"

### 9.5 Command Line Usage

```bash
# Process all PDFs in data/raw_pdfs/
python src/ingest_pdf.py

# Index all processed documents
python src/index_documents.py

# Query from command line
python src/query_rag.py "What is the main methodology?" --answer --model gpt-oss:120b-cloud

# Generate code from paper
python src/paper_to_code.py --paper-base "paper_name"
```

---

## 10. Troubleshooting Guide

### 10.1 Common Issues

#### Issue: "Ollama connection refused"
**Cause:** Ollama server is not running
**Solution:** Start Ollama with `ollama serve` or ensure it's running in the background

#### Issue: "No results found" in RAG search
**Cause:** Documents haven't been indexed
**Solution:** 
1. Check if chunks exist: `ls data/raw_texts/`
2. Run indexing: `python src/index_documents.py`

#### Issue: "ChromaDB collection not found"
**Cause:** Index hasn't been created
**Solution:** Run `python src/index_documents.py`

#### Issue: PDF extraction returns empty text
**Cause:** PDF is scanned images, not text
**Solution:** Use OCR-enabled PDF or a different PDF

#### Issue: Code generation fails repeatedly
**Cause:** Complex paper or LLM limitations
**Solution:** 
1. Check logs in `data/outputs/{paper}/run_logs/`
2. Try with simpler paper
3. Manually edit generated code

### 10.2 Performance Issues

#### Slow PDF processing
- **Cause:** Large PDF files
- **Solution:** Wait or use smaller files

#### Slow embedding generation
- **Cause:** CPU-only processing
- **Solution:** Use GPU if available, reduce chunk count

#### Slow answer generation
- **Cause:** Large LLM model
- **Solution:** Use smaller model like `mistral:7b`

### 10.3 Debug Tips

```python
# Enable ChromaDB debug info
import chromadb
collection = client.get_collection("research_papers")
print(f"Total documents: {collection.count()}")

# Check if Ollama is working
import ollama
print(ollama.list())

# Test embedding generation
from chromadb.utils import embedding_functions
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print(embed_fn(["test text"]))
```

---

## 11. Future Development

### 11.1 Planned Features

#### Phase 2.5: Enhanced Code Generation
- Improved method detection using LLM
- Automatic dataset downloading
- Better hyperparameter extraction

#### Phase 3: Advanced AI Features
- Paper summarization
- Quiz generation
- Multi-paper analysis
- Trend detection

#### Phase 4: Integration & Collaboration
- REST API
- Mobile app
- Obsidian integration
- Collaborative workspaces

#### Phase 5: Enhanced Processing
- Audio/video processing
- Multi-language support
- Real-time paper updates

### 11.2 How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Appendix: Quick Reference

### File Extensions

| Extension | Purpose |
|-----------|---------|
| `.py` | Python source code |
| `.txt` | Extracted text/chunks |
| `.json` | Structured data |
| `.md` | Documentation/reports |
| `.pdf` | Original papers |
| `.zip` | Exported archives |

### Key Imports

```python
# PDF Processing
import fitz  # PyMuPDF

# Vector Database
import chromadb
from chromadb.utils import embedding_functions

# LLM
import ollama

# Web Interface
import streamlit as st

# Data Models
from pydantic import BaseModel

# Standard Library
import re
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile
```

### Model Recommendations

| Use Case | Model | VRAM Required |
|----------|-------|---------------|
| Quick testing | `mistral:7b` | 4GB |
| General use | `llama3.1:8b` | 8GB |
| Best quality | `gpt-oss:120b-cloud` | Cloud-based |

---

*This guide is maintained as part of the Research2Text project. For the latest updates, see the [GitHub repository](https://github.com/Aspect022/Research2Text).*
