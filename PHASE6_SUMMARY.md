# Research2Text Phase 6 - NewResearcher Integration & UI Overhaul

## Summary

This release introduces the NewResearcher components and a completely redesigned 3-phase workflow in the Streamlit UI.

## New Components

### 1. Token-Aware Chunking (`src/chunking/`)
- **File**: `src/chunking/token_chunker.py`
- **Features**:
  - Sentence boundary preservation using NLTK
  - Token counting with tiktoken (OpenAI's tokenizer)
  - Configurable chunk size (default: 800 tokens) and overlap (default: 100 tokens)
  - Metadata for each chunk (token count, sentence count, character positions)
- **Usage**:
  ```python
  from chunking.token_chunker import TokenChunker
  chunker = TokenChunker(chunk_size=800, chunk_overlap=100)
  chunks = chunker.chunk_text(long_text)
  ```

### 2. Source Validation (`src/validation/`)
- **File**: `src/validation/source_validator.py`
- **Features**:
  - Multi-dimensional scoring (credibility, recency, technical depth)
  - Venue reputation scoring (Tier 1: NeurIPS, ICML, CVPR, etc.)
  - Peer review detection
  - Top-N filtering
- **Scoring**:
  - Credibility (0-10): Based on venue reputation and indicators
  - Recency (0-10): Based on publication year with decay
  - Technical Depth (0-10): Based on equation density and technical terms
- **Usage**:
  ```python
  from validation.source_validator import validate_sources
  result = validate_sources(sources, top_n=5)
  ```

### 3. Academic Search (`src/search/`)
- **File**: `src/search/academic_search.py`
- **Features**:
  - Multi-source search (arXiv, Semantic Scholar, Exa, Tavily)
  - Year filtering
  - Relevance scoring
  - PDF link extraction
- **APIs Supported**:
  - arXiv (no API key required)
  - Semantic Scholar (no API key required)
  - Exa (requires EXA_API_KEY)
  - Tavily (requires TAVILY_API_KEY)
- **Usage**:
  ```python
  from search.academic_search import search_papers
  result = search_papers("transformer architecture", max_results=10)
  ```

## Updated Orchestrator

### New Methods

1. **`process_paper_to_knowledge_graph()`** - Phase 1
   - Runs ingestion through knowledge graph construction
   - Returns method_struct, equations, datasets, knowledge_graph
   - Sets `ready_for_code_gen` flag

2. **`generate_code()`** - Phase 2
   - Generates PyTorch code from extracted method information
   - Returns generated files

3. **`run_sandbox_validation()`** - Phase 3
   - Creates Windows Sandbox and executes code
   - Returns validation results with execution output

## Streamlit UI Overhaul

### New Tab Structure

1. **RAG Search** - Original RAG functionality
2. **Pipeline (v2)** - New 3-phase workflow
3. **NewResearcher** - New tools (chunking, validation, search)
4. **Testing** - Testing and validation suite
5. **Dashboard** - Project overview

### 3-Phase Workflow

#### Phase 1: Research Phase
Stages:
1. Document Ingestion
2. Vision Processing
3. Chunking
4. Method Extraction
5. Equation Processing
6. Dataset Processing
7. Knowledge Graph Construction

**UI**: Shows progress for each stage, extracted method summary, confidence scores

#### Phase 2: Code Generation
- **Trigger**: Manual button press
- **Action**: Generate PyTorch code from extracted method
- **Output**: View generated code files before proceeding

#### Phase 3: Sandbox Execution
- **Trigger**: Manual button press
- **Action**: Create Windows Sandbox, execute code
- **Output**: Live execution results, stdout/stderr, logs

### NewResearcher Tab

#### Token-Aware Chunking Section
- Select paper
- Configure chunk size and overlap
- View chunk metrics (total tokens, avg tokens/chunk)
- Preview individual chunks

#### Source Validation Section
- Manual entry or load from paper
- Multi-dimensional scoring
- Top sources display with detailed metrics

#### Academic Search Section
- Multi-source search
- Year filtering
- Results with abstracts, citations, PDF links

### Testing & Validation Tab

#### Conformal Prediction Tests
- Run coverage evaluation
- Generate calibration data

#### Source Validation Tests
- Test with sample sources
- View scoring breakdown

#### Sandbox Tests
- Test sandbox creation
- Execute sample code
- View results

#### Pipeline Validation Tests
- Validate pipeline results for selected paper
- Check for required output files

## Files Added/Modified

### New Files
- `src/chunking/__init__.py`
- `src/chunking/token_chunker.py`
- `src/validation/__init__.py`
- `src/validation/source_validator.py`
- `src/search/__init__.py`
- `src/search/academic_search.py`
- `src/app_streamlit_v2.py` (new UI)
- `PHASE6_SUMMARY.md` (this file)

### Modified Files
- `src/agents/orchestrator.py` - Added 3-phase workflow methods
- `src/app_streamlit.py` - Replaced with v2

## Dependencies

New optional dependencies:
```
tiktoken>=0.5.0      # For token counting
nltk>=3.8.0          # For sentence tokenization
requests>=2.31.0     # For API calls (already required)
```

Install with:
```bash
pip install tiktoken nltk
```

## Environment Variables

For academic search APIs:
```bash
EXA_API_KEY=your_exa_key
TAVILY_API_KEY=your_tavily_key
```

## Usage

### Run the new UI
```bash
streamlit run src/app_streamlit.py
```

### Run the 3-phase pipeline
1. Upload a PDF in the "Pipeline (v2)" tab
2. Click "Start Research Phase" (Phase 1)
3. Review extracted method and knowledge graph
4. Click "Generate Code" (Phase 2)
5. Review generated code
6. Click "Create Sandbox & Run" (Phase 3)
7. View execution results

### Use NewResearcher tools
1. Go to "NewResearcher" tab
2. Select tool: Token-Aware Chunking, Source Validation, or Academic Search
3. Configure parameters
4. Run and view results

## Success Criteria

- [x] Token-aware chunking with sentence preservation
- [x] Source validation with multi-dimensional scoring
- [x] Academic search across multiple sources
- [x] 3-phase workflow with manual triggers
- [x] Code preview before sandbox execution
- [x] Sandbox execution with live output
- [x] Testing & Validation tab
- [x] Modern UI with phase cards

## Next Steps

1. Test the new workflow end-to-end
2. Add more academic search sources (Google Scholar, PubMed)
3. Implement caching for search results
4. Add export functionality for generated code
5. Integrate conformal prediction confidence scores into UI

## Notes

- The old `process_paper()` method is still available for backward compatibility
- NewResearcher components are optional - the system works without API keys
- Sandbox execution requires Windows Sandbox to be enabled on Windows
- Token chunking falls back to approximate counting if tiktoken is not installed
