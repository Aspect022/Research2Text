# 📄 Research2Text - AI-Powered Research Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF6B6B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-4A90E2.svg)](https://www.trychroma.com/)

> An AI-powered research assistant that automatically processes academic papers, extracts key information, and enables intelligent querying through a local RAG (Retrieval-Augmented Generation) system.

## 🎯 Overview

Research2Text transforms the way you interact with academic literature by providing:

- **📚 Automated PDF Processing** - Extract and chunk text from research papers
- **🔍 Semantic Search** - Find relevant information using natural language queries  
- **🤖 AI-Powered Answers** - Generate contextual responses using local LLMs via Ollama
- **💾 Local Storage** - Keep your data private with local vector database
- **🌐 Web Interface** - User-friendly Streamlit application for easy interaction

## 🏗️ Architecture

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
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Functionality
- **PDF Text Extraction**: Robust text extraction from research papers using PyMuPDF
- **Intelligent Chunking**: Word-based text chunking with configurable overlap
- **Vector Embeddings**: Generate semantic embeddings using Sentence Transformers
- **Similarity Search**: Fast cosine similarity search in ChromaDB
- **Local LLM Integration**: Generate answers using Ollama with various models

### User Interface
- **Streamlit Dashboard**: Clean, intuitive web interface
- **Real-time Processing**: Live PDF upload and processing
- **Configurable Parameters**: Adjust chunk retrieval, context length, and model settings
- **Streaming Responses**: Real-time answer generation with streaming support

### Data Management
- **Persistent Storage**: Local ChromaDB for vector storage
- **Metadata Tracking**: Chunk-level metadata for precise source attribution
- **Batch Processing**: Efficient handling of multiple documents
- **Auto-indexing**: Automatic vector database updates

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from PDFs |
| **Embeddings** | Sentence Transformers | Generate semantic vectors |
| **Vector Database** | ChromaDB | Store and query embeddings |
| **LLM Integration** | Ollama | Local language model inference |
| **Web Framework** | Streamlit | User interface |
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
# Install the recommended model
ollama pull gpt-oss:120b-cloud

# Alternative lightweight model
ollama pull mistral:7b
```

### 5. Initialize Directory Structure
The application will automatically create the required directories:
```
data/
├── raw_pdfs/          # Store uploaded PDFs
├── raw_texts/         # Extracted text and chunks
└── chroma_db/         # ChromaDB vector database
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

1. **Upload PDF**: Use the sidebar to upload research papers (PDF format)
2. **Process Document**: Click "Process PDF" to extract text and create embeddings
3. **Ask Questions**: Enter natural language queries in the main interface
4. **Get Answers**: Retrieve relevant chunks and generate AI-powered responses

### Command Line Usage

#### Process PDFs
```bash
python src/ingest_pdf.py
```

#### Build Vector Index
```bash
python src/index_documents.py
```

#### Query Documents
```bash
python src/query_rag.py "What methodology was used in the study?"
```

## ⚙️ Configuration

### Model Settings
- **Default Model**: `gpt-oss:120b-cloud`
- **Alternative Models**: `mistral:7b`, `llama3.1:8b`
- **Top-K Retrieval**: 5 (configurable 3-10)
- **Max Context**: 4000 characters (configurable 1000-12000)

### Chunking Parameters
- **Chunk Size**: 700 words
- **Overlap**: 100 words
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Customization
Edit configuration in the respective source files:
- `src/utils.py` - Chunking parameters
- `src/app_streamlit.py` - UI defaults
- `src/query_rag.py` - Retrieval settings

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

## 🔧 Advanced Features

### Batch Processing
```bash
# Place multiple PDFs in data/raw_pdfs/
python src/ingest_pdf.py  # Processes all PDFs
python src/index_documents.py  # Updates vector index
```

### API Integration
The core functions can be imported for programmatic use:
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

## 🚀 Future Scope

### Phase 2: Research-to-Code Generation
Transform research papers into executable code implementations:

- **🧮 Method Extraction**: Automatically identify algorithms and equations
- **💻 Code Generation**: Convert methodologies to runnable Python/PyTorch code
- **🔬 Experiment Reproduction**: Auto-generate training loops and evaluation scripts
- **📊 Model Comparison**: Compare implementations across multiple papers
- **🔗 Citation Graphs**: Visualize paper relationships and dependencies

### Advanced AI Features
- **📝 Auto-Summarization**: Generate structured summaries and key takeaways
- **🧠 Quiz Generation**: Create flashcards and Q&A for active learning
- **🎯 Section Detection**: Intelligent parsing of paper sections (Abstract, Methods, Results)
- **📈 Trend Analysis**: Identify patterns across multiple papers in a domain

### Integration & Collaboration  
- **📱 Mobile App**: iOS/Android interface for on-the-go research
- **🔗 Obsidian Integration**: Export summaries to personal knowledge vaults
- **👥 Collaborative Mode**: Share insights and annotations with research teams
- **🌐 API Development**: RESTful API for third-party integrations

### Enhanced Processing
- **🎤 Audio Processing**: Upload and process research presentations/lectures
- **📊 Multi-modal Input**: Handle papers with complex figures and tables
- **🌍 Multi-language Support**: Process papers in multiple languages
- **⚡ Real-time Updates**: Live processing of newly published papers

### Intelligence Augmentation
- **🔍 Cross-Paper Queries**: "Compare methodologies across these 10 papers"
- **💡 Research Recommendations**: Suggest related papers and research directions
- **📚 Literature Gap Analysis**: Identify unexplored research areas
- **🎯 Hypothesis Generation**: AI-assisted research question formulation

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
