# Research2Text Startup Guide

Quick start guide for testing the Research2Text system with NewResearcher components.

## Prerequisites

### Required
- Python 3.10+
- Windows 10/11 (for Sandbox features)
- Git

### Optional (for enhanced features)
- Windows Sandbox enabled (for code execution)
- Ollama (for local LLM inference)
- API keys for academic search (Exa, Tavily)

## Installation

### 1. Clone and Setup

```bash
# Navigate to project directory
cd D:\Projects\Research2Text-main

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Optional Dependencies

```bash
# For token-aware chunking
pip install tiktoken nltk

# For academic search (already in requirements)
pip install requests

# For testing
pip install pytest
```

### 3. Download NLTK Data (for chunking)

```python
import nltk
nltk.download('punkt')
```

Or it will auto-download on first use.

### 4. Configure Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# Academic Search APIs (optional)
EXA_API_KEY=your_exa_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_OLLAMA_MODEL=llama3.1
```

## Quick Start

### Option 1: Run Streamlit UI (Recommended)

```bash
# Run the new UI with 3-phase workflow
streamlit run src/app_streamlit.py
```

Then open your browser to: http://localhost:8501

### Option 2: Run Tests

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run specific test file
python tests/run_tests.py --type specific --file tests/unit/test_token_chunker.py
```

### Option 3: Run Individual Components

#### Test Token-Aware Chunking

```python
from chunking.token_chunker import TokenChunker

chunker = TokenChunker(chunk_size=800, chunk_overlap=100)
text = "Your long text here..."
chunks = chunker.chunk_text(text)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}: {chunk.token_count} tokens, {len(chunk.sentences)} sentences")
```

#### Test Source Validation

```python
from validation.source_validator import validate_sources

sources = [
    {
        "id": "src_1",
        "title": "Attention Is All You Need",
        "venue": "NeurIPS",
        "year": 2017,
        "text": "We propose a new simple network architecture..."
    }
]

result = validate_sources(sources, top_n=3)
print(f"Top source: {result['top_sources'][0]['title']}")
print(f"Score: {result['top_sources'][0]['overall_score']:.1f}/10")
```

#### Test Academic Search

```python
from search.academic_search import search_papers

# Search without API keys (uses arXiv + Semantic Scholar)
result = search_papers("transformer architecture", max_results=5)

for paper in result['results']:
    print(f"{paper['title'][:60]}... ({paper['year']})")
    print(f"  Score: {paper['score']:.2f}")
```

#### Test Conformal Prediction

```python
from conformal.predictor import ConformalPredictor

predictor = ConformalPredictor(alpha=0.1)  # 90% coverage

# Calibration data
validation_data = [
    ("text1", {"algorithm_name": "CNN"}),
    ("text2", {"algorithm_name": "RNN"}),
]

def mock_predictor(text):
    return {"algorithm_name": "CNN"}

predictor.calibrate(validation_data, mock_predictor)
print("Calibrated predictor with 90% coverage guarantee")
```

## Using the Streamlit UI

### Tab 1: RAG Search
- Ask questions about processed papers
- Uses vector search + Ollama for answers

### Tab 2: Pipeline (v2) - 3-Phase Workflow

#### Phase 1: Research
1. Upload a PDF or select existing paper
2. Click "Start Research Phase"
3. Wait for completion (ingestion → knowledge graph)
4. Review extracted method and confidence scores

#### Phase 2: Code Generation
1. Click "Generate Code" (only available after Phase 1)
2. Review generated PyTorch code
3. Download individual files if needed

#### Phase 3: Sandbox Execution
1. Click "Create Sandbox & Run" (only available after Phase 2)
2. View execution results
3. Check stdout/stderr output

### Tab 3: NewResearcher Tools

#### Token-Aware Chunking
1. Select a processed paper
2. Adjust chunk size and overlap
3. Click "Chunk Text"
4. View chunk metrics and preview

#### Source Validation
1. Enter sources manually or load from paper
2. Click "Validate Sources"
3. View credibility, recency, and technical scores

#### Academic Search
1. Enter search query
2. Select sources (arXiv, Semantic Scholar, etc.)
3. Click "Search"
4. Browse results with abstracts and PDF links

### Tab 4: Testing & Validation
- Run conformal prediction tests
- Test source validation
- Test sandbox execution
- Validate pipeline results

### Tab 5: Dashboard
- View processed papers
- Check pipeline status
- Browse output files

## Testing the System

### Quick Smoke Test

```bash
# 1. Run unit tests
python -m pytest tests/unit/ -v --tb=short

# 2. Test chunking
python -c "from chunking.token_chunker import TokenChunker; c = TokenChunker(); print('Chunking OK')"

# 3. Test validation
python -c "from validation.source_validator import SourceValidator; v = SourceValidator(); print('Validation OK')"

# 4. Test search
python -c "from search.academic_search import AcademicSearch; s = AcademicSearch(); print('Search OK')"

# 5. Test conformal prediction
python -c "from conformal.predictor import ConformalPredictor; p = ConformalPredictor(); print('Conformal OK')"
```

### End-to-End Test

1. Start Streamlit: `streamlit run src/app_streamlit.py`
2. Upload a sample PDF (or use existing)
3. Run Phase 1 (Research)
4. Run Phase 2 (Code Generation)
5. Run Phase 3 (Sandbox) - if on Windows with Sandbox enabled

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in the project root
cd D:\Projects\Research2Text-main

# Add to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;D:\Projects\Research2Text-main
```

#### NLTK data not found
```python
import nltk
nltk.download('punkt')
```

#### Windows Sandbox not available
- Sandbox execution only works on Windows 10/11 Pro/Enterprise
- Home edition doesn't support Windows Sandbox
- Code generation still works without sandbox

#### Ollama connection errors
- Ensure Ollama is running: `ollama serve`
- Check default model exists: `ollama pull llama3.1`

#### API key errors (for academic search)
- Exa and Tavily are optional
- arXiv and Semantic Scholar work without API keys

### Debug Mode

```bash
# Run with debug logging
set LOG_LEVEL=DEBUG
streamlit run src/app_streamlit.py
```

## File Structure

```
Research2Text-main/
├── src/
│   ├── app_streamlit.py          # Main UI (v2 with 3-phase workflow)
│   ├── agents/
│   │   └── orchestrator.py       # Updated with 3-phase methods
│   ├── chunking/                 # NEW: Token-aware chunking
│   │   ├── __init__.py
│   │   └── token_chunker.py
│   ├── validation/               # NEW: Source validation
│   │   ├── __init__.py
│   │   └── source_validator.py
│   ├── search/                   # NEW: Academic search
│   │   ├── __init__.py
│   │   └── academic_search.py
│   └── conformal/                # Conformal prediction
│       ├── __init__.py
│       ├── predictor.py
│       └── evaluate.py
├── tests/                        # NEW: Test suite
│   ├── unit/
│   │   ├── test_token_chunker.py
│   │   ├── test_source_validator.py
│   │   ├── test_academic_search.py
│   │   └── test_conformal_prediction.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── run_tests.py
├── data/
│   └── calibration/              # Conformal prediction calibration data
├── STARTUP_GUIDE.md             # This file
└── requirements.txt
```

## Next Steps

1. **Test the 3-phase workflow** with a sample paper
2. **Try NewResearcher tools** (chunking, validation, search)
3. **Run the test suite** to verify everything works
4. **Explore the code** to understand the architecture

## Support

- Check `PHASE6_SUMMARY.md` for detailed component documentation
- Review test files for usage examples
- Check logs in `outputs/` directory for debugging

## Quick Reference

| Component | Import Path | Key Class |
|-----------|-------------|-----------|
| Token Chunking | `chunking.token_chunker` | `TokenChunker` |
| Source Validation | `validation.source_validator` | `SourceValidator` |
| Academic Search | `search.academic_search` | `AcademicSearch` |
| Conformal Prediction | `conformal.predictor` | `ConformalPredictor` |
| Orchestrator | `agents.orchestrator` | `Orchestrator` |

## Happy Testing! 🚀

Run `streamlit run src/app_streamlit.py` and start exploring!
