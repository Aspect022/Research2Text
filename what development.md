# Research2Text: Development Analysis & Recommendations

## Executive Summary

This document analyzes the current state of Research2Text, identifies critical issues, and provides actionable recommendations for taking the project to the next level. It covers the NewResearcher integration, code execution isolation alternatives, knowledge graph quality issues, embedding strategies, and architectural improvements.

---

## 1. NewResearcher Integration Analysis

### What NewResearcher Brings (CrewAI-based)

**Architecture:**
- Uses **CrewAI** framework (higher-level abstraction than custom orchestrator)
- 5 specialized agents: Planner → Search → Validator → Extractor → Synthesizer
- Token-aware pipeline with strict limits (800 tokens/chunk, 3000 chars/source)
- External search integration (Exa, Tavily) for live research

**Key Strengths:**
1. **Better chunking strategy**: Sentence-aware with paragraph boundary preservation
2. **Source validation**: Explicit credibility scoring (1-10) before extraction
3. **Token safety**: Hard limits at every layer prevent LLM overflow
4. **Citation-focused**: Built for generating reports with verified references
5. **Production patterns**: Proper logging, env-based config, retry logic

**Integration Opportunities:**

| Component | Current Research2Text | NewResearcher | Integration Strategy |
|-----------|----------------------|---------------|---------------------|
| Orchestration | Custom Orchestrator | CrewAI | **Keep custom** - more control over paper-to-code flow |
| Chunking | Word-based (700 words) | Token-based (800 tokens) | **Adopt NewResearcher** - sentence boundaries are better |
| Validation | Post-hoc AST check | Pre-extraction source scoring | **Add source validation** before method extraction |
| Search | Local ChromaDB only | Exa + Tavily APIs | **Add as optional** for live paper discovery |
| Token Management | None | Strict limits everywhere | **Implement throughout** |

**Recommendation:**
Integrate NewResearcher's **chunking**, **token management**, and **source validation** patterns into Research2Text. Keep the custom orchestrator for the paper-to-code pipeline (more control than CrewAI's black box).

---

## 2. Code Execution Isolation: Docker Alternatives

### The Problem

Current `validator.py:78` runs generated code directly:
```python
res = run_and_capture(["python", "train.py"], cwd=sandbox, timeout=90)
```

**Risks:**
- No isolation from host system
- Generated code can access filesystem, network
- Malicious paper → malicious code execution
- Dependency conflicts with host Python

### Why Docker is Heavyweight

| Concern | Reality |
|---------|---------|
| **Startup time** | 5-30 seconds for container creation |
| **Resource overhead** | ~100MB+ per container |
| **Complexity** | Requires Docker daemon, image management |
| **Cross-platform** | Docker Desktop issues on Windows/Mac |
| **Cleanup** | Orphaned containers/volumes |

### Better Alternatives (In Order of Preference)

#### Option 1: **Firejail** (Recommended for Linux)
```bash
# Install: sudo apt install firejail
firejail --noprofile --private=/tmp/sandbox python train.py
```

**Pros:**
- Native Linux namespaces (no VM overhead)
- Filesystem sandboxing (--private)
- Network isolation (--net=none)
- Resource limits (--rlimit)
- Sub-second startup

**Cons:**
- Linux only
- Requires root for installation

#### Option 2: **subprocess with RestrictedPython**
```python
from restrictedpython import compile_restricted
from restrictedpython.Guards import safe_builtins

# Compile code with restricted builtins
gl = {'__builtins__': safe_builtins}
exec(compile_restricted(user_code), gl)
```

**Pros:**
- Pure Python (no external deps)
- Prevents dangerous builtins (eval, exec, open)
- Fast execution

**Cons:**
- Can be bypassed by determined attackers
- Limited protection against infinite loops

#### Option 3: **PyPy Sandbox** (Experimental)
- True sandboxing at interpreter level
- Very secure but limited library support

#### Option 4: **Systemd-nspawn** (Linux)
```bash
systemd-nspawn --directory=/var/lib/sandbox --private-network python train.py
```
- Lightweight container (shares kernel)
- Better than Docker for single-shot execution

#### Option 5: **Cloud Functions** (AWS Lambda, etc.)
- True isolation
- Pay-per-execution
- Requires internet + cloud account

### Recommended Implementation

**Hybrid Approach:**
```python
# validator.py - improved isolation

def run_isolated(code_dir: Path, timeout: int = 90) -> RunResult:
    """Run code with best available isolation."""

    # Try Firejail first (Linux production)
    if shutil.which("firejail"):
        return _run_firejail(code_dir, timeout)

    # Fallback to RestrictedPython (cross-platform dev)
    return _run_restricted(code_dir, timeout)

def _run_firejail(code_dir: Path, timeout: int) -> RunResult:
    cmd = [
        "firejail",
        "--noprofile",
        "--private=/tmp/sandbox",
        "--net=none",  # No network
        "--rlimit-cpu=90",  # CPU limit
        "--rlimit-as=1g",  # Memory limit
        "python", "train.py"
    ]
    return run_and_capture(cmd, cwd=code_dir, timeout=timeout)

def _run_restricted(code_dir: Path, timeout: int) -> RunResult:
    # Use RestrictedPython for the entry point
    # Fall back to subprocess with timeout for imports
    pass
```

**Verdict:** Use **Firejail for production Linux** + **RestrictedPython fallback** for cross-platform development. Avoid Docker for this use case.

---

## 3. Knowledge Graph Quality Issues

### Current Implementation Analysis

Looking at `graph_builder_agent.py:20-121`:

```python
# Current issues:
1. Nodes created only from method_struct (limited fields)
2. No section/chunk-level node extraction
3. No relationship inference (only hardcoded "contains", "uses")
4. No entity linking (datasets not canonicalized)
5. Properties are empty dictionaries
```

### Why Quality is Low

| Issue | Current Behavior | Expected Behavior |
|-------|---------------|-------------------|
| **Limited node types** | Only Paper, Algorithm, Dataset, Equation, Citation | Should include: Method, Architecture, Hyperparameter, Metric, Baseline, Ablation |
| **No section context** | Single paper node | Should have: Abstract, Introduction, Method, Experiments, Results nodes |
| **Flat structure** | Paper → Algorithm (1 hop) | Should have: Paper → Section → Subsection → Method → Component |
| **No semantic edges** | Only "contains", "uses" | Should have: "implements", "extends", "compares_to", "achieves", "outperforms" |
| **Empty properties** | `{}` | Should have: confidence scores, source text, page numbers |

### Root Causes

1. **Input data is too coarse**: Only uses `method_struct` (high-level) not chunks (detailed)
2. **No LLM involvement**: Graph is built with simple dictionary lookups
3. **No entity resolution**: "CIFAR-10" and "CIFAR10" become separate nodes
4. **Static schema**: Node types hardcoded, not inferred from content

### Recommendations for High-Quality Knowledge Graphs

#### Approach 1: LLM-Assisted Graph Extraction (Recommended)

Add a new extraction step using the LLM:

```python
# New: graph_extraction_agent.py
GRAPH_EXTRACTION_PROMPT = """
Analyze the following research paper text and extract a detailed knowledge graph.

TEXT:
{text}

Extract:
1. **Entities** (with type, label, and properties)
   - Paper sections (Abstract, Introduction, Method, etc.)
   - Algorithms and models
   - Datasets and benchmarks
   - Hyperparameters (with values)
   - Metrics and results
   - Claims and contributions

2. **Relationships** (with type and evidence)
   - "achieves" (model → metric)
   - "uses" (model → dataset)
   - "outperforms" (model → baseline)
   - "implements" (code → algorithm)
   - "depends_on" (component → component)

Return JSON:
{
  "entities": [
    {"id": "...", "type": "...", "label": "...", "properties": {...}}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "evidence": "..."}
  ]
}
"""
```

#### Approach 2: Chunk-Level Graph Building

Process each chunk individually, then merge:

```python
# Modified graph_builder_agent.py
def process(self, message: AgentMessage) -> AgentResponse:
    chunks = payload.get("chunks", [])

    # Build subgraph for each chunk
    subgraphs = []
    for chunk in chunks:
        subgraph = self._extract_subgraph(chunk)  # LLM call per chunk
        subgraphs.append(subgraph)

    # Merge subgraphs with entity resolution
    merged = self._merge_subgraphs(subgraphs)

    return AgentResponse(data=merged)
```

#### Approach 3: Hybrid (Current + LLM)

Keep current structure for speed, add LLM enrichment for detail:

```python
# Two-phase graph building
phase1 = self._build_base_graph(method_struct)  # Fast, rule-based
phase2 = self._enrich_with_llm(chunks)  # Detailed, LLM-based
return merge(phase1, phase2)
```

### Quick Wins

1. **Add section nodes**: Parse paper structure ("Abstract", "1. Introduction", etc.)
2. **Canonicalize entities**: Use dataset_loader_agent's fuzzy matching for datasets
3. **Add confidence scores**: Copy from method_struct.confidence
4. **Add source evidence**: Store which chunk each node came from

---

## 4. Content Chunking Strategy Analysis

### Current Implementation

**Research2Text** (`utils.py:18-73`):
```python
def chunk_text_by_words(text, chunk_size_words=700, overlap_words=100):
    words = text.split()
    # Simple word splitting - no sentence awareness
```

**NewResearcher** (`text_chunker.py:35-116`):
```python
def chunk_text(text, max_tokens=800, overlap_tokens=100):
    sentences = _split_into_sentences(text)
    # Sentence-aware with token counting
```

### Comparison

| Aspect | Research2Text | NewResearcher | Winner |
|--------|--------------|---------------|--------|
| **Unit** | Words (700) | Tokens (800) | Tokens (LLM-native) |
| **Boundary** | Arbitrary word count | Sentence boundaries | Sentences |
| **Overlap** | Words | Tokens | Tokens |
| **Hard limit** | None | 3000 chars | NewResearcher |
| **Token counting** | None | tiktoken | NewResearcher |

### Problems with Current Chunking

1. **Sentence breaking**: "The attention mechanism (see Section 3) works by..." might split mid-sentence
2. **Context loss**: Mathematical expressions spanning chunks get broken
3. **Inefficient LLM usage**: 700 words ≈ 1000+ tokens (exceeds optimal)
4. **No semantic chunking**: Related paragraphs may be separated

### Recommended Chunking Strategy

```python
# Enhanced chunker combining both approaches

def chunk_text_enhanced(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    respect_sections: bool = True
) -> List[Chunk]:
    """
    Advanced chunking with:
    - Sentence boundaries
    - Section awareness
    - Semantic coherence
    """

    # Step 1: Identify section boundaries
    sections = parse_sections(text) if respect_sections else []

    # Step 2: Split into sentences
    sentences = split_sentences(text)

    # Step 3: Group sentences respecting:
    #   - Token limit
    #   - Section boundaries
    #   - Semantic similarity (optional)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # Check if adding this sentence exceeds limit
        if current_tokens + sent_tokens > max_tokens:
            # Save current chunk
            chunks.append(create_chunk(current_chunk, overlap_tokens))

            # Start new chunk with overlap
            current_chunk = get_overlap_sentences(current_chunk, overlap_tokens)
            current_tokens = count_tokens(current_chunk)

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    return chunks
```

### Advanced: Semantic Chunking

```python
# Use embeddings to keep semantically similar content together
from sklearn.cluster import KMeans

def semantic_chunking(sentences: List[str], n_chunks: int) -> List[List[str]]:
    embeddings = model.encode(sentences)
    clusters = KMeans(n_clusters=n_chunks).fit_predict(embeddings)

    # Group sentences by cluster
    chunks = defaultdict(list)
    for sent, cluster in zip(sentences, clusters):
        chunks[cluster].append(sent)

    return list(chunks.values())
```

**Recommendation:** Adopt NewResearcher's token-based, sentence-aware chunking. Add section boundary detection for academic papers.

---

## 5. Embedding Strategy: Gemini 2.0 vs Current

### Current Setup

```python
# index_documents.py:61
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 384 dimensions, 22M parameters
```

### Gemini 2.0 Text Embedding

**Specs (from Google's announcement):**
- Model: `models/text-embedding-004` (or similar)
- Dimensions: 768 (configurable)
- Context window: 8,192 tokens (vs MiniLM's 512)
- Multilingual: Yes
- MTEB Score: ~68+ (vs MiniLM's ~62)

### Comparison

| Feature | MiniLM-L6-v2 | Gemini 2.0 | Advantage |
|---------|--------------|------------|-------------|
| **Size** | 22M params | Unknown (likely larger) | MiniLM (local) |
| **Speed** | Fast (CPU) | API latency | MiniLM |
| **Quality** | MTEB ~62 | MTEB ~68+ | Gemini |
| **Context** | 512 tokens | 8,192 tokens | Gemini (16x!) |
| **Cost** | Free (local) | API pricing | MiniLM |
| **Privacy** | Local | Cloud | MiniLM |
| **Multilingual** | Limited | Strong | Gemini |
| **Offline** | Yes | No | MiniLM |

### Critical Consideration: Context Window

**The 8K context window is GAME-CHANGING for research papers:**

```python
# Current: 512 tokens ≈ 350 words
# Can fit: 1-2 paragraphs

# Gemini: 8,192 tokens ≈ 6,000 words
# Can fit: Entire paper section or short paper
```

This means:
- **No more chunking needed** for short papers
- **Section-level embeddings** instead of paragraph-level
- **Better semantic coherence** in retrieval

### Hybrid Recommendation

```python
# config.py - embedding configuration

class EmbeddingConfig:
    # Primary: Gemini for quality (if API available)
    GEMINI_MODEL = "models/text-embedding-004"
    GEMINI_DIMENSIONS = 768
    GEMINI_CONTEXT = 8192

    # Fallback: Local MiniLM (offline/privacy)
    LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_DIMENSIONS = 384
    LOCAL_CONTEXT = 512

    # Strategy
    @classmethod
    def get_embedder(cls):
        if os.getenv("GEMINI_API_KEY"):
            return GeminiEmbedder(cls.GEMINI_MODEL)
        return LocalEmbedder(cls.LOCAL_MODEL)
```

### Implementation Plan

**Phase 1: Add Gemini as Optional**
```python
# new file: src/embedders.py

class GeminiEmbedder:
    def __init__(self, model_name="models/text-embedding-004"):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding with 8k context
        result = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
```

**Phase 2: Adaptive Chunking**
- If using Gemini: Larger chunks (2000-4000 tokens)
- If using MiniLM: Keep current 700 words

**Phase 3: Unified Interface**
```python
# Both embedders implement same interface
embedder = EmbeddingConfig.get_embedder()
embeddings = embedder.embed(chunks)  # Works with either
```

### Verdict on Gemini 2.0

**YES, integrate it**, but as an **optional enhancement**, not replacement:

1. **Keep MiniLM as default** (local, free, offline)
2. **Add Gemini as premium option** (better quality, larger context)
3. **Let users choose** based on their needs

The 8K context window alone justifies the integration for research papers.

---

## 6. Comprehensive Improvement Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Code Execution Safety**
- [ ] Implement Firejail sandbox (Linux)
- [ ] Add RestrictedPython fallback (cross-platform)
- [ ] Add resource limits (CPU, memory, time)
- [ ] Security audit of generated code patterns

**Chunking Improvements**
- [ ] Adopt token-based chunking from NewResearcher
- [ ] Add sentence boundary detection
- [ ] Add section-aware splitting
- [ ] Benchmark: chunk quality vs retrieval accuracy

### Phase 2: Quality (Weeks 3-4)

**Knowledge Graph Enhancement**
- [ ] Add LLM-assisted graph extraction
- [ ] Implement entity canonicalization
- [ ] Add relationship inference ("outperforms", "implements")
- [ ] Add confidence scores to nodes/edges
- [ ] Visual graph export (NetworkX + Matplotlib)

**Embedding Upgrade**
- [ ] Add Gemini 2.0 embedder (optional)
- [ ] Implement adaptive chunking based on embedder
- [ ] Add embedding quality evaluation
- [ ] Support for multimodal embeddings (future)

### Phase 3: Integration (Weeks 5-6)

**NewResearcher Integration**
- [ ] Port token management system
- [ ] Add source validation agent
- [ ] Integrate search tools (Exa/Tavily) as optional
- [ ] Unified logging format

**Performance & Scale**
- [ ] Async agent execution (where possible)
- [ ] Caching layer for LLM responses
- [ ] Parallel chunk processing
- [ ] Database migration (ChromaDB → PostgreSQL + pgvector for scale)

### Phase 4: Polish (Weeks 7-8)

**Testing & Reliability**
- [ ] Unit tests for each agent
- [ ] Integration tests for full pipeline
- [ ] Benchmark suite (extraction accuracy, code correctness)
- [ ] CI/CD pipeline

**Documentation**
- [ ] API documentation
- [ ] Architecture decision records (ADRs)
- [ ] Deployment guide
- [ ] Contributing guidelines

---

## 7. Quick Wins (Do These First)

1. **Fix chunking** (1 day)
   - Copy NewResearcher's `text_chunker.py`
   - Replace `chunk_text_by_words` with token-based version

2. **Add Firejail** (1 day)
   - Install on Linux dev machine
   - Add `_run_firejail()` to validator.py
   - Test with generated PyTorch code

3. **Improve knowledge graph** (2 days)
   - Add section parsing (regex for "1. Introduction")
   - Add properties (confidence, source chunk)
   - Canonicalize dataset names

4. **Add Gemini embeddings** (1 day)
   - Get API key
   - Implement `GeminiEmbedder` class
   - Add toggle in config

---

## 8. Architecture Decision Summary

| Decision | Current | Recommended | Rationale |
|----------|---------|-------------|-----------|
| **Sandbox** | None | Firejail + RestrictedPython | Security without Docker overhead |
| **Chunking** | Word-based | Token-based, sentence-aware | Better LLM compatibility |
| **Embeddings** | MiniLM only | MiniLM + Gemini (optional) | Quality vs cost flexibility |
| **Knowledge Graph** | Rule-based | LLM-assisted + rule-based | Richer, more accurate graphs |
| **Orchestration** | Custom | Keep custom | More control than CrewAI |
| **NewResearcher** | Separate | Integrate components | Best of both worlds |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Firejail Linux-only | High | Medium | Provide RestrictedPython fallback |
| Gemini API costs | Medium | Low | Make optional, keep MiniLM default |
| LLM graph extraction slow | High | Low | Cache results, make async |
| Token-based chunking breaks existing | Low | High | Test with existing papers |
| Integration complexity | Medium | Medium | Incremental integration, not rewrite |

---

## 10. Success Metrics

Define measurable targets:

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Knowledge graph nodes (avg) | 10-15 | 40-60 | Nodes per paper |
| Code validation success | ~60% | >85% | % of papers with runnable code |
| RAG retrieval accuracy | Unknown | >80% | Human evaluation on 50 queries |
| Pipeline execution time | ~120s | <60s | End-to-end paper processing |
| Token efficiency | Unknown | <4000 tokens/paper | Total LLM tokens used |

---

## Conclusion

Research2Text has a solid foundation but needs focused improvements in:

1. **Security** (sandboxing)
2. **Quality** (knowledge graphs, chunking)
3. **Flexibility** (embedding options)
4. **Integration** (NewResearcher components)

The recommended path is **incremental enhancement**, not a rewrite. Keep the custom orchestrator (it's working well), adopt NewResearcher's best practices (chunking, token management), and add Gemini as an optional upgrade.

**Next immediate actions:**
1. Implement Firejail sandbox
2. Port NewResearcher's chunking
3. Add Gemini embedder
4. Enhance knowledge graph with LLM assistance

---

*Document Version: 1.0*
*Date: 2026-03-17*
*Author: Claude Code Analysis*
