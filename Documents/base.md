
## 🎯 Project Overview

An AI-powered research assistant that automatically processes academic papers, extracts key information, generates summaries and quizzes, then stores everything in your personal knowledge base. Perfect for students, researchers, and lifelong learners.

**Core Features:**

- Upload PDFs of research papers
- Auto-extract sections (abstract, methodology, results, conclusions)
- Generate summaries in multiple formats (bullets, detailed notes, flashcards)
- Create Q&A and quizzes for active recall
- Auto-save to Notion or Obsidian for easy reference

---

## 🛠️ Tech Stack (100% Free & Open Source)

### Core Framework

- **LangChain (Community Edition)** - Orchestrates the RAG pipeline, chains, and document processing
- **Python 3.10+** - Base programming language

### Vector Database

- **ChromaDB** - Local, lightweight vector database (no cloud costs)
- Stores document embeddings for semantic search
- Persistent storage on your machine

### Embeddings

- **HuggingFace Sentence Transformers** - `all-MiniLM-L6-v2` model
- Free, runs locally, no API costs
- 384-dimensional embeddings, fast and efficient

### LLM (Language Model)

- **Ollama + Llama 3.1 (8B)** - Run locally on your machine
- Alternative: **Mistral 7B** for lower resource usage
- Zero API costs, full privacy

### PDF Processing

- **PyMuPDF (fitz)** - Fast PDF text extraction
- Handles scanned PDFs with optional OCR via Tesseract (free)

### Frontend

- **Streamlit** - Simple web interface for uploading PDFs
- Alternative: **Gradio** for even simpler UI

### Automation

- **Python Watchdog** - Monitor folder for new PDFs (replaces paid n8n)
- Triggers processing automatically when files appear

### Storage Options

- **Obsidian** - Local markdown vault (free forever)
- **Notion (Free tier)** - 1000 blocks limit (sufficient for personal use)
- Plain text files as backup

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  PDF Upload     │
│  (Local folder  │
│   or Streamlit) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PDF Parser     │
│  (PyMuPDF)      │
│  • Extract text │
│  • Clean format │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunker   │
│  500-1000 tokens│
│  50-100 overlap │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │
│  (MiniLM-L6-v2) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector DB      │
│  (ChromaDB)     │
│  + Metadata     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Retrieval  │
│  (Similarity    │
│   Search)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Processing │
│  (Llama 3.1)    │
│  • Summarize    │
│  • Generate Q&A │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Format  │
│  • Markdown     │
│  • Flashcards   │
│  • Quiz         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Storage        │
│  • Obsidian     │
│  • Notion       │
│  • Local files  │
└─────────────────┘
```

---

## 🔄 Detailed Workflow

### 1. **Input Stage**

**Options:**

- Manual upload via Streamlit web interface
- Drop PDF into watched folder (auto-trigger)
- Batch upload multiple papers at once

**File Support:**

- Text-based PDFs (native digital documents)
- Scanned PDFs with OCR enabled (requires Tesseract installation)

---

### 2. **PDF Preprocessing**

**Text Extraction:**

- PyMuPDF reads each page sequentially
- Preserves paragraph structure and basic formatting
- Filters out headers, footers, page numbers automatically

**Cleaning Steps:**

- Remove excessive whitespace and line breaks
- Strip citation markers like `[1], [2]` for cleaner text
- Identify section headers (Abstract, Methods, Results, etc.)
- Separate references/bibliography section

**Section Detection:**

- Uses regex patterns to find common academic section headers
- Creates metadata tags for each chunk: `{"section": "methods", "page": 5}`

---

### 3. **Text Chunking**

**Why Chunking?**

- LLMs have token limits (4K-8K for local models)
- Smaller chunks improve retrieval accuracy
- Better semantic matching in vector search

**Strategy:**

- **Chunk Size:** 500-1000 tokens (~400-800 words)
- **Overlap:** 50-100 tokens between chunks
- **Reasoning:** Overlap ensures context isn't lost at boundaries

**Example:**

```
Chunk 1: [Tokens 1-500] + overlap [501-550]
Chunk 2: [Tokens 501-1000] + overlap [1001-1050]
```

---

### 4. **Embedding Generation**

**Model:** `all-MiniLM-L6-v2` (free HuggingFace model)

- Converts text chunks into 384-dimensional vectors
- Captures semantic meaning (not just keywords)
- Runs locally on CPU (no GPU required, though faster with GPU)

**Process:**

- Each chunk → Embedding vector
- Stored in ChromaDB with metadata

**Metadata Example:**

```json
{
  "title": "Attention Is All You Need",
  "authors": "Vaswani et al.",
  "section": "methods",
  "page": 3,
  "chunk_id": 15
}
```

---

### 5. **Vector Database Storage**

**ChromaDB Features:**

- Persistent local storage (data saved between sessions)
- Fast cosine similarity search
- Metadata filtering (e.g., "only search methods section")

**Database Structure:**

- Collection per paper (or one collection for all papers)
- Each entry: `[embedding_vector, text_chunk, metadata]`

**Query Example:**

- User asks: "How did they evaluate the model?"
- System embeds the question → Searches ChromaDB → Returns top 5 relevant chunks

---

### 6. **Summarization Pipeline**

**LangChain Chains Used:**

**A) High-Level Summary (TL;DR)**

- Input: Full paper text (or top chunks)
- Prompt: "Summarize this research paper in 3-4 sentences"
- Output: Quick overview for skimming

**B) Section Summaries**

- Processes Abstract, Methods, Results, Conclusions separately
- Prompt per section: "Summarize the methodology in 2-3 paragraphs"
- Preserves technical details while condensing

**C) Key Takeaways**

- Prompt: "Extract 5 most important findings/contributions"
- Output: Bulleted list of core insights

**D) Layman Explanation (Optional)**

- Prompt: "Explain this paper to a non-expert in simple terms"
- Great for understanding complex topics

---

### 7. **Q&A and Quiz Generation**

**Flashcard Generator:**

- Chain extracts important concepts
- Generates 10-15 question-answer pairs
- Format: `Q: What is the attention mechanism? A: ...`

**Quiz Mode:**

- Multiple choice questions (4 options)
- True/False statements
- Fill-in-the-blank style

**Spaced Repetition Ready:**

- Exports in Anki-compatible format
- Or Obsidian with Spaced Repetition plugin tags

---

### 8. **Output Formatting**

**Obsidian Markdown Template:**

```markdown
---
title: "Attention Is All You Need"
authors: Vaswani et al.
year: 2017
tags: [transformers, NLP, attention]
---

# 📄 TL;DR
[3-sentence summary]

# 🔍 Key Takeaways
- Contribution 1
- Contribution 2
- ...

# 📝 Section Summaries
## Abstract
[Summary]

## Methodology
[Summary]

## Results
[Summary]

# 🧠 Flashcards
Q: What problem does the paper solve?
A: ...

# 🔗 Links
- [Original PDF](file:///path/to/pdf)
```

**Notion Database Structure:**

|Title|Authors|Year|Status|Summary|Tags|PDF Link|
|---|---|---|---|---|---|---|

---

### 9. **Automation Layer**

**File Watcher Setup:**

- Python Watchdog monitors `~/Research/Inbox/` folder
- When new PDF detected → Auto-trigger processing pipeline
- Notification sent when complete

**Google Drive Integration (Optional):**

- Use Google Drive API (free tier)
- Sync watched folder with Drive
- Access papers from anywhere

---

## 💾 Storage Options Explained

### Option A: Obsidian (Recommended)

**Pros:**

- 100% free, local-first
- Markdown = future-proof, portable
- Powerful linking and graph view
- Works offline

**Setup:**

- Create vault: `~/ObsidianVault/Research/`
- Papers saved as: `YYYY-MM-DD_AuthorName_Title.md`
- Auto-link related papers using tags

### Option B: Notion

**Pros:**

- Cloud sync across devices
- Better for collaboration
- Database views (table, board, calendar)

**Cons:**

- Free tier: 1000 blocks limit (sufficient for ~50-100 papers)
- Requires internet

**API Setup:**

- Create Notion integration (free)
- Get database ID
- Use `notion-client` Python library

### Option C: Local Files

- Plain `.md` files in organized folders
- Use GitHub for version control and backup
- Works with any text editor

---

## 🔍 RAG Retrieval Explained

**What is RAG?** Retrieval-Augmented Generation = Fetch relevant docs + Generate answer

**How It Works Here:**

1. User asks: "What dataset did they use?"
2. Question is embedded → Vector search in ChromaDB
3. Top 5 most relevant chunks retrieved
4. Chunks + Question fed to Llama 3.1
5. LLM generates answer grounded in retrieved text

**Why Not Just Summarize Everything?**

- Token limits prevent processing entire papers
- RAG focuses on relevant sections only
- More accurate answers to specific questions

---

## ⚙️ System Requirements

**Minimum:**

- CPU: 4 cores (Intel i5 / AMD Ryzen 5)
- RAM: 8GB (16GB recommended)
- Storage: 10GB for models + papers
- OS: Windows, macOS, or Linux

**For Llama 3.1 8B:**

- Runs on CPU (slow but works)
- GPU: GTX 1660 or better (much faster)
- 8GB VRAM ideal

**For Mistral 7B (lighter alternative):**

- Runs smoothly on 4GB RAM
- Faster inference on modest hardware

---

## 🚀 Key Advantages

1. **100% Free** - No API costs, no subscriptions
2. **Private** - All processing happens locally
3. **Offline** - Works without internet (except Notion sync)
4. **Customizable** - Tweak prompts, chunk sizes, output formats
5. **Portable** - Markdown outputs work anywhere
6. **Scalable** - Add more papers without hitting usage limits

---

## 📊 Use Cases

- **Students:** Quickly review papers for assignments
- **Researchers:** Build personal knowledge base of literature
- **PhD Candidates:** Organize dozens of papers by topic
- **Self-Learners:** Create study materials from technical papers

---

## 🔮 Future Enhancements

- **Citation Graph:** Visualize paper relationships in Obsidian graph
- **Multi-Paper Queries:** "Compare methods across 5 papers"
- **Voice Notes:** Upload audio summaries of papers
- **Mobile App:** iOS/Android interface via Flutter
- **Collaborative Mode:** Share summaries with study groups

---

## 📚 Learning Resources

- **LangChain Docs:** langchain.com/docs
- **ChromaDB Guide:** docs.trychroma.com
- **Ollama Setup:** ollama.ai/docs
- **Obsidian Plugins:** obsidian.md/plugins

---

## ⚠️ Limitations

- **Local LLMs** are slower than GPT-4 but good enough for summaries
- **Scanned PDFs** need OCR (Tesseract) for text extraction
- **ChromaDB** not ideal for 1000+ papers (consider Qdrant for scale)
- **Notion Free Tier** has 1000 block limit (monitor usage)

---

_Built for curious minds who want to learn faster without breaking the bank._ 🧠✨