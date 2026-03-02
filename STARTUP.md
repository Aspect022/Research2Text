# 🚀 Research2Code-GenAI — Startup Guide

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Git** — [Download](https://git-scm.com/)
- **Ollama** (optional, for LLM features) — [Download](https://ollama.com/)

---

## 1. Clone & Enter the Project

```bash
git clone https://github.com/YOUR_USERNAME/Research2Text.git
cd Research2Text-main
```

---

## 2. Create a Virtual Environment

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt.

---

## 3. Install All Dependencies (Covers All 5 Phases)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> This single command installs everything for Phases 1–5: PyMuPDF, ChromaDB, Streamlit, PyTorch, Tesseract bindings, Ollama, SymPy, NetworkX, and all dev tools.

### Optional: Best-in-class PDF Extraction

These are large packages and only needed if you want the premium extraction pipeline:

```bash
# MinerU — preserves LaTeX equations, tables, and reading order
pip install "mineru[all]"

# olmOCR — VLM-based fallback for scanned/complex PDFs
pip install "olmocr[gpu]"
```

> Without these, the pipeline auto-falls back to PyMuPDF (already installed).

---

## 4. Setup Ollama (Optional but Recommended)

If you want LLM-powered method extraction and code generation:

```bash
# Install a coder model (pick one)
ollama pull deepseek-coder-v2
ollama pull qwen2.5-coder
ollama pull codellama

# Verify it's running
ollama list
```

> The pipeline auto-detects which model is available. If none are installed, it falls back to heuristic extraction.

---

## 5. Create Data Directories

```bash
mkdir -p data/raw_pdfs data/raw_texts outputs
```

On Windows CMD:
```cmd
mkdir data\raw_pdfs data\raw_texts outputs
```

---

## 6. Run the App

### Option A: Streamlit UI (Recommended)

```bash
cd src
streamlit run app_streamlit.py
```

Open **http://localhost:8501** in your browser.

### Option B: Run Tests

```bash
python tests/test_multiagent_pipeline.py
```

Expected output:
```
TEST 1 PASSED: All agents imported successfully.
TEST 2 PASSED: Orchestrator initialized with all 10 agents.
TEST 3 PASSED: All agents process mock data correctly.
TEST 4 PASSED: Full pipeline completed successfully.
🎉 ALL TESTS PASSED!
```

### Option C: CLI Pipeline

```bash
cd src
python paper_to_code_multiagent.py --pdf ../data/raw_pdfs/your_paper.pdf
```

---

## 7. Quick Test Workflow

1. **Start the app** → `streamlit run src/app_streamlit.py`
2. **Upload a PDF** → Drag any research paper PDF into the upload area
3. **Click "Run Multi-Agent Pipeline"** → Watch the 9-stage progress tracker
4. **Review results** → Check confidence scores, generated code, knowledge graph
5. **Download artifacts** → Click "Download Artifacts" for the ZIP bundle

---

## Troubleshooting

| Problem                  | Fix                                                    |
| ------------------------ | ------------------------------------------------------ |
| `ModuleNotFoundError`    | Make sure venv is activated: `.\venv\Scripts\activate` |
| `chromadb` errors        | `pip install chromadb --upgrade`                       |
| Empty PDF extraction     | Install MinerU: `pip install mineru[all]`              |
| No LLM answer            | Install Ollama + pull a model: `ollama pull codellama` |
| Sandbox validation fails | Install PyTorch: `pip install torch`                   |

---

## Project Structure

```
Research2Text-main/
├── src/
│   ├── app_streamlit.py        ← Main UI (streamlit run this)
│   ├── agents/
│   │   ├── orchestrator.py     ← 10-agent pipeline coordinator
│   │   ├── ingest_agent.py     ← MinerU/olmOCR/PyMuPDF extraction
│   │   ├── vision_agent.py     ← Image/table batch processing
│   │   ├── method_extractor_agent.py  ← Conformal prediction + LLM
│   │   ├── validator_agent.py  ← Sandbox execution + self-heal
│   │   └── ...                 ← 5 more specialized agents
│   ├── schemas.py              ← Pydantic models + ConfidenceScore
│   ├── code_generator.py       ← Confidence-aware code generation
│   └── ...
├── tests/
│   └── test_multiagent_pipeline.py
├── data/                       ← PDFs and extracted text
├── outputs/                    ← Generated code + pipeline results
├── requirements.txt
├── start.bat                   ← Windows launcher
└── STARTUP.md                  ← This file
```
