# 📄 Research2Text - AI-Powered Research Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF6B6B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-4A90E2.svg)](https://www.trychroma.com/)
[![Phase 2](https://img.shields.io/badge/Phase%202-Paper%20to%20Code-success.svg)](https://github.com/yourusername/research2text)

> **🚀 Phase 2 Now Live!** An advanced AI-powered research assistant that not only processes academic papers and enables intelligent querying through RAG, but also automatically converts research methodologies into executable Python code implementations.

## 🎯 Overview

Research2Text transforms the way you interact with academic literature by providing:

### Phase 1: RAG-Based Research Assistant
- **📚 Automated PDF Processing** - Extract and chunk text from research papers
- **🔍 Semantic Search** - Find relevant information using natural language queries  
- **🤖 AI-Powered Answers** - Generate contextual responses using local LLMs via Ollama
- **💾 Local Storage** - Keep your data private with local vector database
- **🌐 Web Interface** - User-friendly Streamlit application for easy interaction

### Phase 2: Paper-to-Code Generation ✨ **NEW!**
- **🧮 Method Extraction** - Automatically identify algorithms, equations, and datasets
- **💻 Code Generation** - Convert research methodologies to runnable Python/PyTorch code
- **🔧 Self-Healing Validation** - Iteratively fix code errors using AI
- **📊 Structured Outputs** - Generate comprehensive reports with code and logs
- **📦 Export System** - Download complete project artifacts as ZIP files

## 🏗️ Enhanced Architecture (Phase 2)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Chunking  │───▶│   Embeddings    │
│   (PyMuPDF)     │    │  (Word-based)   │    │   (MiniLM-L6)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼─────────┐
│  Ollama LLM     │◀───│   RAG Query     │◀───│   ChromaDB      │
│  (gpt-oss:120b) │    │   (Retrieval)   │    │  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                     │
         ▼                        ▼                     │
┌─────────────────────────────────────────────┐        │
│            Streamlit Web Interface           │        │
│    ┌─────────────┐  ┌─────────────────────┐ │        │
│    │  RAG Search │  │   Paper → Code      │ │        │
│    │    Tab      │  │      Tab ✨         │ │        │
│    └─────────────┘  └─────────────────────┘ │        │
└─────────────────────────────────────────────┘        │
                                                       │
              ┌────────────────────────────────────────┘
              │
              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Method Extractor│───▶│  Code Generator │───▶│   Validator     │
│ (Algorithms,    │    │  (PyTorch Code) │    │ (Self-Healing)  │
│  Equations)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                              ┌─────────────────────────────────┐
                              │         Output System           │
                              │  ┌─────────┐  ┌─────────────┐   │
                              │  │ Reports │  │ Artifacts   │   │
                              │  │   .md   │  │  ZIP Files  │   │
                              │  └─────────┘  └─────────────┘   │
                              └─────────────────────────────────┘
```

## 🚀 Features

### Phase 1: RAG Functionality
- **PDF Text Extraction**: Robust text extraction from research papers using PyMuPDF
- **Intelligent Chunking**: Word-based text chunking with configurable overlap
- **Vector Embeddings**: Generate semantic embeddings using Sentence Transformers
- **Similarity Search**: Fast cosine similarity search in ChromaDB
- **Local LLM Integration**: Generate answers using Ollama with various models
- **Streaming Responses**: Real-time answer generation with streaming support

### Phase 2: Paper-to-Code Generation ✨
- **Method Recognition**: Automatically detect algorithm names, equations, and datasets
- **Structured Data Extraction**: Convert unstructured paper text to JSON schemas
- **Code Synthesis**: Generate complete PyTorch implementations from research descriptions
- **Self-Healing Validation**: Automatically detect and fix runtime errors in generated code
- **Comprehensive Reporting**: Create detailed markdown reports linking papers to code
- **Artifact Management**: Export complete project packages with code, logs, and documentation

### User Interface
- **Dual-Tab Interface**: Separate RAG search and Paper-to-Code functionality
- **Real-time Processing**: Live PDF upload and processing for both workflows
- **Download System**: ZIP export for generated code and complete artifacts
- **Progress Tracking**: Visual feedback for long-running code generation processes

### Data Management
- **Persistent Storage**: Local ChromaDB for vector storage
- **Structured Outputs**: Organized artifact directories with logs and generated code
- **Metadata Tracking**: Comprehensive tracking from paper to executable implementation
- **Export Utilities**: Multiple export formats (code-only, full artifacts)

## 🛠️ Enhanced Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from PDFs |
| **Embeddings** | Sentence Transformers | Generate semantic vectors |
| **Vector Database** | ChromaDB | Store and query embeddings |
| **LLM Integration** | Ollama (gpt-oss:120b-cloud) | Local language model inference |
| **Web Framework** | Streamlit | Dual-tab user interface |
| **Data Modeling** | Pydantic | Structured schemas and validation |
| **Code Generation** | PyTorch Templates | Deep learning framework code |
| **Validation** | Subprocess + AI | Self-healing code execution |
| **Symbolic Math** | SymPy | Equation parsing and manipulation |
| **Export System** | ZIP + Markdown | Artifact packaging and documentation |
| **Backend** | Python 3.10+ | Core application logic |

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- At least 8GB RAM (16GB recommended for larger models)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/research2text.git
cd research2text
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Ollama Model
```bash
# Install the recommended model for best results
ollama pull gpt-oss:120b-cloud

# Alternative lightweight models
ollama pull mistral:7b
ollama pull llama3.1:8b
```

### 5. Initialize Directory Structure
The application will automatically create the required directories:
```
data/
├── raw_pdfs/          # Store uploaded PDFs
├── raw_texts/         # Extracted text and chunks
├── chroma_db/         # ChromaDB vector database
└── outputs/           # Generated code and artifacts (Phase 2)
    └── {paper_name}/
        ├── method.json        # Extracted method structure
        ├── report.md         # Generated report
        ├── sandbox/          # Generated code files
        └── run_logs/         # Validation logs
```

## 🎮 Usage

### Starting the Application
```bash
# Navigate to project directory
cd research2text

# Launch Streamlit app
streamlit run src/app_streamlit.py
```

The application will be available at `http://localhost:8501`

### Basic Workflow

#### RAG Search (Phase 1)
1. **Upload PDF**: Use the sidebar to upload research papers (PDF format)
2. **Process Document**: Click "Process PDF" to extract text and create embeddings
3. **Ask Questions**: Enter natural language queries in the main interface
4. **Get Answers**: Retrieve relevant chunks and generate AI-powered responses

#### Paper-to-Code (Phase 2) ✨
1. **Select Paper**: Upload new PDF or choose from existing processed papers
2. **Generate Code**: Click "Generate Code" to run the paper-to-code pipeline
3. **Review Results**: Examine generated PyTorch code, validation logs, and reports
4. **Download Artifacts**: Export complete project as ZIP or code-only package

### Command Line Usage

#### Phase 1: RAG Operations
```bash
# Process PDFs
python src/ingest_pdf.py

# Build Vector Index
python src/index_documents.py

# Query Documents
python src/query_rag.py "What methodology was used in the study?"
```

#### Phase 2: Paper-to-Code Generation
```bash
# Generate code from processed paper
python src/paper_to_code.py --paper-base "paper_name"

# Example with actual paper
python src/paper_to_code.py --paper-base "An_Improved_Facial_Expression_Recognition"
```

## ⚙️ Configuration

### Model Settings
- **Default Model**: `gpt-oss:120b-cloud`
- **Alternative Models**: `mistral:7b`, `llama3.1:8b`
- **Top-K Retrieval**: 5 (configurable 3-10)
- **Max Context**: 4000 characters (configurable 1000-12000)

### Phase 1: RAG Parameters
- **Chunk Size**: 700 words
- **Overlap**: 100 words
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Phase 2: Code Generation Parameters
- **Framework**: PyTorch (default)
- **Validation Attempts**: 3 (configurable)
- **Timeout**: 90 seconds per validation
- **Self-Healing**: Enabled by default

### Customization
Edit configuration in the respective source files:
- `src/utils.py` - Chunking parameters
- `src/app_streamlit.py` - UI defaults and model settings
- `src/query_rag.py` - Retrieval settings
- `src/code_generator.py` - Code generation templates
- `src/validator.py` - Validation and self-healing parameters

## 📊 Performance

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores (Intel i5/AMD Ryzen 5) | 8+ cores |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB | 50GB+ |
| **GPU** | None (CPU only) | GTX 1660+ (faster inference) |

### Benchmarks
- **PDF Processing**: ~2-5 seconds per paper
- **Embedding Generation**: ~1-3 seconds per chunk
- **Query Response**: ~5-15 seconds (model dependent)
- **Index Size**: ~1MB per 100 pages
- **Code Generation**: ~30-60 seconds per paper (Phase 2)
- **Code Validation**: ~15-45 seconds per attempt (Phase 2)

## 🔧 Advanced Features

### Batch Processing
```bash
# Phase 1: RAG Pipeline
python src/ingest_pdf.py  # Process all PDFs in data/raw_pdfs/
python src/index_documents.py  # Update vector index

# Phase 2: Bulk Code Generation
for paper in $(ls data/raw_texts/*.txt | cut -d'/' -f3 | cut -d'.' -f1); do
    python src/paper_to_code.py --paper-base "$paper"
done
```

### API Integration

#### Phase 1: RAG Operations
```python
from src.utils import extract_text_from_pdf, chunk_text_by_words
from src.query_rag import retrieve, answer_with_ollama

# Extract and process
text = extract_text_from_pdf("paper.pdf")
chunks = chunk_text_by_words(text)

# Query
results = retrieve("What is the main contribution?")
answer = answer_with_ollama("question", "context", "model")
```

#### Phase 2: Paper-to-Code Operations
```python
from src.paper_to_code import run_paper_to_code
from src.method_extractor import extract_method_entities
from src.export_utils import build_artifacts_zip

# Generate code from paper
output_dir = run_paper_to_code("paper_base_name")

# Export artifacts
zip_data = build_artifacts_zip("paper_base_name")
with open("artifacts.zip", "wb") as f:
    f.write(zip_data)
```

## 🚀 Future Scope

### Phase 2.5: Enhanced Code Generation
**Phase 2 is now complete!** Next enhancements include:

- **🎯 Improved Method Detection**: Advanced NLP for better algorithm identification
- **📊 Dataset Integration**: Automatic dataset downloading and preprocessing
- **🔬 Full Experiment Reproduction**: Complete training pipelines with hyperparameter tuning
- **📈 Benchmarking Suite**: Automated performance comparison across implementations
- **🔍 Multi-Paper Analysis**: Cross-paper method comparison and synthesis

### Phase 3: Advanced AI Features
- **📝 Auto-Summarization**: Generate structured summaries and key takeaways
- **🧠 Quiz Generation**: Create flashcards and Q&A for active learning
- **🎯 Enhanced Section Detection**: ML-based parsing of paper sections
- **📈 Trend Analysis**: Identify patterns across multiple papers in a domain
- **🔍 Cross-Paper Code Synthesis**: Generate code by combining insights from multiple papers

### Phase 4: Integration & Collaboration  
- **📱 Mobile App**: iOS/Android interface for on-the-go research
- **🔗 Obsidian Integration**: Export summaries and generated code to knowledge vaults
- **👥 Collaborative Mode**: Share code implementations and insights with research teams
- **🌐 REST API**: Full API for third-party integrations and automation
- **🐳 Docker Deployment**: Containerized deployment for easy scaling

### Phase 5: Enhanced Processing
- **🎤 Audio Processing**: Upload and process research presentations/lectures
- **📊 Multi-modal Input**: Handle papers with complex figures and tables
- **🌍 Multi-language Support**: Process papers in multiple languages
- **⚡ Real-time Updates**: Live processing of newly published papers
- **🏗️ Framework Agnostic**: Support for TensorFlow, JAX, and other frameworks

### Phase 6: Intelligence Augmentation
- **🔍 Cross-Paper Queries**: "Compare methodologies across these 10 papers"
- **💡 Research Recommendations**: Suggest related papers and research directions
- **📚 Literature Gap Analysis**: Identify unexplored research areas
- **🎯 Hypothesis Generation**: AI-assisted research question formulation
- **🧪 Automated Experimentation**: Run generated code across multiple datasets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

### Areas for Contribution
- 🐛 Bug fixes and performance improvements
- 📚 Documentation and examples
- 🧪 New models and embedding techniques
- 🎨 UI/UX enhancements
- 🔌 Integration with other tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[ChromaDB](https://www.trychroma.com/)** - Excellent vector database
- **[Streamlit](https://streamlit.io/)** - Rapid web app development
- **[Ollama](https://ollama.ai/)** - Easy local LLM deployment
- **[Sentence Transformers](https://www.sbert.net/)** - High-quality embeddings
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - Reliable PDF processing

## 📞 Support & Contact

- 📧 **Email**: [jayeshrl2005@gmail.com](mailto:jayeshrl2005@gmail.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/research2text/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/research2text/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/research2text/wiki)

---

<div align="center">

**Built with ❤️ for the research community**

*Transforming how we interact with academic literature, one paper at a time.*

[⭐ Star this repository](https://github.com/yourusername/research2text) | [🍴 Fork it](https://github.com/yourusername/research2text/fork) | [📝 Report Issues](https://github.com/yourusername/research2text/issues)

</div>
