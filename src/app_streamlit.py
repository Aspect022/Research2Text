"""
Research2Code-GenAI — Streamlit Interface (Phase 6 Overhaul)

New Workflow:
  1. Research Phase: Ingestion → Vision → Chunking → Method Extraction → Equations → Datasets → Knowledge Graph
  2. Code Generation Phase: Manual trigger → Generate code → View code
  3. Sandbox Phase: Manual trigger → Create sandbox → Run → View results

New Features:
  - NewResearcher tab (token-aware chunking, source validation, academic search)
  - Step-by-step pipeline with manual triggers
  - Code preview before sandbox execution
  - Sandbox execution with live output
"""

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ─── Project Imports ──────────────────────────────────
from config import (
    PROJECT_ROOT, RAW_PDF_DIR, RAW_TEXT_DIR, OUTPUTS_DIR,
    DEFAULT_OLLAMA_MODEL, DEFAULT_TOP_K, MAX_CONTEXT_CHARS,
    APP_TITLE, APP_ICON, APP_SUBTITLE,
)
from utils import extract_text_from_pdf, chunk_text_by_words
from export_utils import build_artifacts_zip, list_known_bases, build_code_zip
from paper_to_code import run_paper_to_code
from index_documents import index_documents
from query_rag import retrieve, format_context, answer_with_ollama

# NewResearcher imports
from chunking.token_chunker import TokenChunker, chunk_text_by_tokens
from validation.source_validator import SourceValidator, validate_sources
from search.academic_search import AcademicSearch, search_papers


# ═══════════════════════════════════════════════════════
#  CSS Theme
# ═══════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent-primary: #00D4AA;
    --accent-secondary: #0088FF;
    --accent-warning: #FF6B35;
    --accent-error: #FF3366;
    --bg-dark: #0A0E1A;
    --bg-card: rgba(15, 20, 35, 0.85);
    --bg-glass: rgba(255, 255, 255, 0.04);
    --text-primary: #E8ECF4;
    --text-secondary: #8B95A8;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --glow: 0 0 20px rgba(0, 212, 170, 0.15);
}

.main { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #0D1224 50%, #101830 100%);
}

/* Pipeline Stage Card */
.stage-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stage-card.active {
    border-color: var(--accent-primary);
    box-shadow: var(--glow);
}
.stage-card.done {
    border-left: 3px solid var(--accent-primary);
    opacity: 0.85;
}
.stage-card.error {
    border-left: 3px solid var(--accent-error);
}

/* Phase Cards */
.phase-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    backdrop-filter: blur(10px);
}
.phase-card.ready {
    border-color: var(--accent-primary);
    box-shadow: var(--glow);
}
.phase-card.pending {
    opacity: 0.6;
}
.phase-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}
.phase-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
}
.phase-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
}
.phase-status {
    margin-left: auto;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.status-ready {
    background: rgba(0, 212, 170, 0.15);
    color: #00D4AA;
}
.status-pending {
    background: rgba(255, 184, 0, 0.15);
    color: #FFB800;
}
.status-complete {
    background: rgba(0, 136, 255, 0.15);
    color: #0088FF;
}

/* Confidence Meter */
.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
    margin-top: 6px;
}
.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.confidence-high { background: linear-gradient(90deg, #00D4AA, #00FF88); }
.confidence-mid { background: linear-gradient(90deg, #FFB800, #FF8800); }
.confidence-low { background: linear-gradient(90deg, #FF6B35, #FF3366); }

/* Metric Cards */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* KG Node */
.kg-node {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 4px;
}
.kg-paper { background: rgba(0, 136, 255, 0.15); color: #4DA8FF; border: 1px solid rgba(0, 136, 255, 0.3); }
.kg-algo { background: rgba(0, 212, 170, 0.15); color: #00D4AA; border: 1px solid rgba(0, 212, 170, 0.3); }
.kg-dataset { background: rgba(255, 184, 0, 0.15); color: #FFB800; border: 1px solid rgba(255, 184, 0, 0.3); }
.kg-equation { background: rgba(180, 120, 255, 0.15); color: #B478FF; border: 1px solid rgba(180, 120, 255, 0.3); }

/* Header */
.app-header {
    text-align: center;
    padding: 20px 0 10px;
}
.app-header h1 {
    background: linear-gradient(135deg, #00D4AA 0%, #0088FF 50%, #B478FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: -8px;
}

/* Code Preview */
.code-preview {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

/* Sandbox Output */
.sandbox-output {
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    max-height: 400px;
    overflow-y: auto;
}
.sandbox-output .stdout { color: #00D4AA; }
.sandbox-output .stderr { color: #FF6B35; }
.sandbox-output .error { color: #FF3366; }
</style>
"""


# ═══════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════

def save_uploaded_pdf(upload) -> Path:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_PDF_DIR / upload.name
    out_path.write_bytes(upload.read())
    return out_path


def ingest_pdf(pdf_path: Path):
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_pdf(str(pdf_path))
    if not text or len(text.strip()) < 10:
        raise ValueError("PDF extraction returned empty text. The PDF might be scanned or corrupted.")
    base = pdf_path.stem
    (RAW_TEXT_DIR / f"{base}.txt").write_text(text, encoding="utf-8")
    chunks = chunk_text_by_words(text)
    if not chunks:
        raise ValueError(f"No chunks created from text ({len(text)} chars).")
    for i, c in enumerate(chunks):
        (RAW_TEXT_DIR / f"{base}_chunk_{i}.txt").write_text(c, encoding="utf-8")
    return base, len(chunks)


def render_confidence_bar(score: float, label: str) -> str:
    """Render an HTML confidence bar."""
    pct = int(score * 100)
    css_class = "confidence-high" if score >= 0.7 else "confidence-mid" if score >= 0.4 else "confidence-low"
    return f"""
    <div style="margin: 6px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span style="color: #E8ECF4;">{label}</span>
            <span style="color: #8B95A8;">{pct}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {css_class}" style="width: {pct}%;"></div>
        </div>
    </div>"""


def render_pipeline_stage(name: str, status: str, detail: str = "") -> str:
    """Render a pipeline stage card."""
    icons = {"done": "&#9989;", "active": "&#9203;", "pending": "&#11036;", "error": "&#10060;"}
    icon = icons.get(status, "&#11036;")
    detail_str = f'<div style="color: #8B95A8; font-size: 0.8rem; margin-top: 4px;">{detail}</div>' if detail else ""
    html = f"""<div class="stage-card {status}">
        <span style="font-size: 1.1rem;">{icon}</span>
        <span style="color: #E8ECF4; font-weight: 500; margin-left: 8px;">{name}</span>
        {detail_str}
    </div>"""
    return html.replace("\n", "")


def render_metrics(data: Dict[str, Any]) -> str:
    """Render metric cards row."""
    cards = ""
    for label, value in data.items():
        cards += f"""<div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>"""
    return f'<div class="metric-row">{cards}</div>'


def render_phase_card(phase_num: int, title: str, status: str, content: str) -> str:
    """Render a phase card for the new workflow."""
    status_class = f"status-{status}"
    status_text = status.upper()

    return f"""
    <div class="phase-card {status}">
        <div class="phase-header">
            <div class="phase-number">{phase_num}</div>
            <div class="phase-title">{title}</div>
            <div class="phase-status {status_class}">{status_text}</div>
        </div>
        <div class="phase-content">
            {content}
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════
#  NewResearcher Tab
# ═══════════════════════════════════════════════════════

def render_newresearcher_tab():
    """Render the NewResearcher tab with token chunking, source validation, and academic search."""
    st.header("🔬 NewResearcher Tools")

    subtab = st.radio(
        "Select Tool",
        ["Token-Aware Chunking", "Source Validation", "Academic Search"],
        horizontal=True
    )

    if subtab == "Token-Aware Chunking":
        render_token_chunking_section()
    elif subtab == "Source Validation":
        render_source_validation_section()
    elif subtab == "Academic Search":
        render_academic_search_section()


def render_token_chunking_section():
    """Render the token-aware chunking section."""
    st.subheader("Token-Aware Text Chunking")
    st.markdown("""
    Advanced chunking with:
    - Sentence boundary preservation
    - Token counting with tiktoken
    - Configurable overlap between chunks
    """)

    # Select paper
    bases = list_known_bases()
    if not bases:
        st.info("No papers available. Process a paper in the Pipeline tab first.")
        return

    selected_base = st.selectbox("Select paper", bases, key="chunking_paper")
    text_file = RAW_TEXT_DIR / f"{selected_base}.txt"

    if not text_file.exists():
        st.error(f"Text file not found: {text_file}")
        return

    text = text_file.read_text(encoding="utf-8")
    st.info(f"Loaded text: {len(text):,} characters")

    # Chunking parameters
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size (tokens)", 200, 2000, 800, 100)
    with col2:
        chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 400, 100, 50)

    if st.button("Chunk Text", type="primary"):
        with st.spinner("Chunking..."):
            chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_text(text)

        st.success(f"Created {len(chunks)} chunks")

        # Display metrics
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        st.markdown(
            render_metrics({
                "Chunks": len(chunks),
                "Total Tokens": f"{total_tokens:,}",
                "Avg Tokens/Chunk": f"{avg_tokens:.0f}",
            }),
            unsafe_allow_html=True
        )

        # Display chunks
        for i, chunk in enumerate(chunks[:5]):  # Show first 5
            with st.expander(f"Chunk {i+1} ({chunk.token_count} tokens, {len(chunk.sentences)} sentences)"):
                st.text_area(f"Chunk {i+1} text", chunk.text, height=150, label_visibility="collapsed")
                st.caption(f"Sentences: {len(chunk.sentences)} | Characters: {len(chunk.text)}")

        if len(chunks) > 5:
            st.info(f"... and {len(chunks) - 5} more chunks")


def render_source_validation_section():
    """Render the source validation section."""
    st.subheader("Source Validation")
    st.markdown("""
    Validate academic sources with multi-dimensional scoring:
    - Credibility (venue reputation, peer review)
    - Recency (publication year)
    - Technical depth (equation density, methodology)
    """)

    # Input sources
    input_method = st.radio("Input method", ["Manual Entry", "From Paper"], horizontal=True)

    sources = []
    if input_method == "Manual Entry":
        num_sources = st.number_input("Number of sources", 1, 10, 3)
        for i in range(num_sources):
            with st.expander(f"Source {i+1}", expanded=i==0):
                title = st.text_input(f"Title {i+1}", key=f"src_title_{i}")
                venue = st.text_input(f"Venue {i+1}", key=f"src_venue_{i}")
                year = st.number_input(f"Year {i+1}", 1900, 2030, 2023, key=f"src_year_{i}")
                text = st.text_area(f"Abstract/Text {i+1}", key=f"src_text_{i}", height=100)

                if title:
                    sources.append({
                        "id": f"src_{i+1}",
                        "title": title,
                        "venue": venue,
                        "year": year,
                        "text": text
                    })
    else:
        bases = list_known_bases()
        if bases:
            selected = st.selectbox("Select paper", bases)
            # Extract references from paper
            method_file = OUTPUTS_DIR / selected / "method.json"
            if method_file.exists():
                data = json.loads(method_file.read_text())
                refs = data.get("references", [])
                st.info(f"Found {len(refs)} references in paper")
                # Create synthetic sources from references
                for i, ref in enumerate(refs[:5]):
                    sources.append({
                        "id": f"ref_{i+1}",
                        "title": f"Reference {ref}",
                        "venue": "Unknown",
                        "year": None,
                        "text": ""
                    })

    if sources and st.button("Validate Sources", type="primary"):
        with st.spinner("Validating..."):
            result = validate_sources(sources, top_n=5)

        # Display summary
        summary = result.get("summary", {})
        st.markdown(
            render_metrics({
                "Total": summary.get("total_sources", 0),
                "Peer Reviewed": summary.get("peer_reviewed", 0),
                "Avg Score": f"{summary.get('average_overall_score', 0):.1f}/10",
            }),
            unsafe_allow_html=True
        )

        # Display top sources
        st.subheader("Top Validated Sources")
        for src in result.get("top_sources", []):
            with st.expander(f"{src['title'][:50]}... (Score: {src['overall_score']:.1f})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Credibility", f"{src['credibility']:.1f}")
                with col2:
                    st.metric("Recency", f"{src['recency']:.1f}")
                with col3:
                    st.metric("Technical", f"{src['technical']:.1f}")
                st.caption(f"Year: {src.get('year', 'Unknown')} | Peer Reviewed: {src.get('peer_reviewed', False)}")


def render_academic_search_section():
    """Render the academic search section."""
    st.subheader("Academic Paper Search")
    st.markdown("""
    Search across multiple academic databases:
    - arXiv (open access)
    - Semantic Scholar
    - Exa (if API key configured)
    - Tavily (if API key configured)
    """)

    query = st.text_input("Search query", placeholder="e.g., transformer architecture attention mechanism")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_results = st.slider("Max results", 5, 50, 10)
    with col2:
        year_from = st.number_input("Year from", 1900, 2030, 2015)
    with col3:
        year_to = st.number_input("Year to", 1900, 2030, 2025)

    sources = st.multiselect(
        "Sources",
        ["arxiv", "semantic_scholar", "exa", "tavily"],
        default=["arxiv", "semantic_scholar"]
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching academic databases..."):
            try:
                result = search_papers(
                    query,
                    max_results=max_results,
                    sources=sources if sources else None
                )

                st.success(f"Found {result['total_results']} results")

                # Display results
                for i, paper in enumerate(result.get("results", []), 1):
                    with st.expander(f"{i}. {paper['title'][:80]}..."):
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', [])[:3])}")
                        if paper.get('year'):
                            st.markdown(f"**Year:** {paper['year']}")
                        if paper.get('venue'):
                            st.markdown(f"**Venue:** {paper['venue']}")
                        if paper.get('citations'):
                            st.markdown(f"**Citations:** {paper['citations']:,}")

                        st.markdown(f"**Source:** {paper['source']}")

                        if paper.get('abstract'):
                            st.markdown("**Abstract:**")
                            st.markdown(paper['abstract'][:500] + "...")

                        col1, col2 = st.columns(2)
                        with col1:
                            if paper.get('url'):
                                st.markdown(f"[View Paper]({paper['url']})")
                        with col2:
                            if paper.get('pdf_url'):
                                st.markdown(f"[PDF]({paper['pdf_url']})")

            except Exception as e:
                st.error(f"Search failed: {e}")


# ═══════════════════════════════════════════════════════
#  New Pipeline Tab (3-Phase Workflow)
# ═══════════════════════════════════════════════════════

def render_pipeline_tab_v2():
    """Render the new 3-phase pipeline tab."""
    st.header("⚡ Research → Code Pipeline (v2)")

    # Initialize session state
    if "pipeline_phase" not in st.session_state:
        st.session_state.pipeline_phase = "idle"  # idle, research_complete, code_generated, sandbox_complete
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = {}
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = None
    if "sandbox_results" not in st.session_state:
        st.session_state.sandbox_results = None

    # Upload or select paper
    up = st.file_uploader("Upload a PDF", type=["pdf"], key="p2_pdf_v2")
    base_name = None

    if up is not None:
        if "last_uploaded_file_v2" not in st.session_state or st.session_state.last_uploaded_file_v2 != up.name:
            pdf_path = save_uploaded_pdf(up)
            with st.spinner(f"Extracting text from {up.name}..."):
                base_name, n_chunks = ingest_pdf(pdf_path)
                index_documents(base_name)
            st.session_state.last_uploaded_file_v2 = up.name
            st.session_state.last_base_name_v2 = base_name
            st.session_state.last_n_chunks_v2 = n_chunks
            st.success(f"✅ {up.name}: {n_chunks} chunks indexed")
        else:
            base_name = st.session_state.last_base_name_v2
            n_chunks = st.session_state.last_n_chunks_v2
            st.success(f"✅ {up.name}: {n_chunks} chunks indexed")
    else:
        bases = list_known_bases()
        if bases:
            base_name = st.selectbox("Or select an existing paper", options=bases, index=0, key="existing_paper_v2")

    if not base_name:
        st.info("📄 Upload a PDF or select a paper to start the pipeline.")
        return

    # Store base_name in session
    st.session_state.current_base_name = base_name

    # ═══════════════════════════════════════════════════════
    # Phase 1: Research Phase (Ingestion → Knowledge Graph)
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    phase1_content = render_phase1_content(base_name)
    phase1_status = "complete" if st.session_state.pipeline_phase in ["research_complete", "code_generated", "sandbox_complete"] else "pending"
    st.markdown(render_phase_card(1, "Research Phase", phase1_status, phase1_content), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # Phase 2: Code Generation (Manual Trigger)
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    phase2_content = render_phase2_content(base_name)
    phase2_status = "ready" if st.session_state.pipeline_phase == "research_complete" else \
                    "complete" if st.session_state.pipeline_phase in ["code_generated", "sandbox_complete"] else "pending"
    st.markdown(render_phase_card(2, "Code Generation", phase2_status, phase2_content), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # Phase 3: Sandbox Execution (Manual Trigger)
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    phase3_content = render_phase3_content(base_name)
    phase3_status = "ready" if st.session_state.pipeline_phase == "code_generated" else \
                    "complete" if st.session_state.pipeline_phase == "sandbox_complete" else "pending"
    st.markdown(render_phase_card(3, "Sandbox Execution", phase3_status, phase3_content), unsafe_allow_html=True)


def render_phase1_content(base_name: str) -> str:
    """Render Phase 1 content (Research Phase)."""
    if st.session_state.pipeline_phase == "idle":
        if st.button("🚀 Start Research Phase", type="primary", key="start_research"):
            with st.spinner("Running research pipeline..."):
                run_research_phase(base_name)
            st.rerun()
        return "Click to start the research phase: ingestion, vision processing, chunking, method extraction, equations, datasets, and knowledge graph construction."
    else:
        # Show research results
        results = st.session_state.pipeline_results
        stages = results.get("stages", {})

        html = "<div style='margin-top: 12px;'>"

        # Show stage status
        stage_names = [
            ("Ingestion", "ingestion"),
            ("Vision", "vision"),
            ("Chunking", "chunking"),
            ("Method Extraction", "method_extraction"),
            ("Equations", "equations"),
            ("Datasets", "datasets"),
            ("Knowledge Graph", "knowledge_graph"),
        ]

        for name, key in stage_names:
            stage = stages.get(key, {})
            success = stage.get("success", False) if isinstance(stage, dict) else True
            icon = "&#9989;" if success else "&#10060;"
            html += f'<div style="margin: 4px 0; color: {'#00D4AA' if success else '#FF6B35'};">{icon} {name}</div>'

        html += "</div>"

        # Show method extraction summary
        method_data = stages.get("method_extraction", {}).get("data", {})
        if method_data:
            method_struct = method_data.get("method_struct", {})
            html += f"""
            <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <strong>Extracted Method:</strong> {method_struct.get('algorithm_name', 'N/A')}<br>
                <strong>Datasets:</strong> {', '.join(method_struct.get('datasets', []))}<br>
                <strong>Confidence:</strong> {method_data.get('overall_confidence', 0):.0%}
            </div>
            """

        return html


def render_phase2_content(base_name: str) -> str:
    """Render Phase 2 content (Code Generation)."""
    if st.session_state.pipeline_phase not in ["research_complete", "code_generated", "sandbox_complete"]:
        return "Complete Phase 1 (Research) to enable code generation."

    if st.session_state.pipeline_phase == "research_complete":
        if st.button("&#128187; Generate Code", type="primary", key="generate_code_btn"):
            with st.spinner("Generating code..."):
                run_code_generation(base_name)
            st.rerun()
        return "Click to generate PyTorch code from the extracted method information."

    # Show generated code
    code_data = st.session_state.generated_code
    if not code_data:
        return "No code generated yet."

    files = code_data.get("files", [])
    html = f"<div style='margin-top: 12px;'><strong>Generated {len(files)} files:</strong></div>"

    for f in files:
        path = f.get("path", "unknown")
        html += f'<div style="margin: 4px 0; color: #0088FF;">&#128196; {path}</div>'

    return html


def render_phase3_content(base_name: str) -> str:
    """Render Phase 3 content (Sandbox Execution)."""
    if st.session_state.pipeline_phase not in ["code_generated", "sandbox_complete"]:
        return "Complete Phase 2 (Code Generation) to enable sandbox execution."

    if st.session_state.pipeline_phase == "code_generated":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("&#128736; Create Sandbox & Run", type="primary", key="run_sandbox_btn"):
                with st.spinner("Creating sandbox and running code..."):
                    run_sandbox_execution(base_name)
                st.rerun()
        with col2:
            if st.button("&#128194; View Code First", key="view_code_btn"):
                st.session_state.show_code_preview = True
                st.rerun()
        return "Click to create a Windows Sandbox and execute the generated code."

    # Show sandbox results
    sandbox_data = st.session_state.sandbox_results
    if not sandbox_data:
        return "No sandbox results yet."

    validation = sandbox_data.get("validation", {}).get("data", {})
    success = validation.get("success", False)

    html = f"<div style='margin-top: 12px;'>"
    html += f"<div style='color: {'#00D4AA' if success else '#FF6B35'}; font-size: 1.2rem; font-weight: 600;'>"
    html += f"{'&#9989; Execution Successful' if success else '&#10060; Execution Failed'}"
    html += "</div>"

    execution = validation.get("execution", {})
    if execution:
        html += f"<div style='margin-top: 8px;'>Attempts: {execution.get('attempts', 0)}</div>"

    html += "</div>"

    return html


def run_research_phase(base_name: str):
    """Run the research phase (up to knowledge graph)."""
    from agents.orchestrator import Orchestrator

    text_file = RAW_TEXT_DIR / f"{base_name}.txt"
    if not text_file.exists():
        st.error(f"Text file not found: {text_file}")
        return

    text = text_file.read_text(encoding="utf-8")

    orchestrator = Orchestrator()
    results = orchestrator.process_paper_to_knowledge_graph(text=text, paper_base=base_name)

    st.session_state.pipeline_results = results
    st.session_state.pipeline_phase = "research_complete"

    # Save results
    out_dir = OUTPUTS_DIR / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "research_phase_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def run_code_generation(base_name: str):
    """Run code generation phase."""
    from agents.orchestrator import Orchestrator

    results = st.session_state.pipeline_results
    method_struct = results.get("method_struct", {})
    equations = results.get("equations", [])
    datasets = results.get("datasets", {})
    text = results.get("extracted_text", "")

    orchestrator = Orchestrator()
    code_results = orchestrator.generate_code(
        paper_base=base_name,
        method_struct=method_struct,
        equations=equations,
        datasets=datasets,
        paper_text=text
    )

    st.session_state.generated_code = code_results
    st.session_state.pipeline_phase = "code_generated"

    # Save results
    out_dir = OUTPUTS_DIR / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code_generation_results.json").write_text(
        json.dumps(code_results, indent=2, default=str), encoding="utf-8"
    )


def run_sandbox_execution(base_name: str):
    """Run sandbox execution phase."""
    from agents.orchestrator import Orchestrator

    code_data = st.session_state.generated_code
    files = code_data.get("files", [])

    orchestrator = Orchestrator()
    sandbox_results = orchestrator.run_sandbox_validation(
        paper_base=base_name,
        files=files
    )

    st.session_state.sandbox_results = sandbox_results
    st.session_state.pipeline_phase = "sandbox_complete"

    # Save results
    out_dir = OUTPUTS_DIR / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sandbox_results.json").write_text(
        json.dumps(sandbox_results, indent=2, default=str), encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════
#  Code Preview Section
# ═══════════════════════════════════════════════════════

def render_code_preview():
    """Render code preview section."""
    if not st.session_state.get("show_code_preview"):
        return

    st.markdown("---")
    st.header("&#128196; Generated Code Preview")

    code_data = st.session_state.get("generated_code", {})
    files = code_data.get("files", [])

    if not files:
        st.warning("No code files to display.")
        return

    # File selector
    file_names = [f.get("path", f"file_{i}") for i, f in enumerate(files)]
    selected_file = st.selectbox("Select file to view", file_names)

    # Display selected file
    for f in files:
        if f.get("path") == selected_file:
            content = f.get("content", "")
            lang = "python" if selected_file.endswith(".py") else "text"
            st.code(content, language=lang)

            # Download button
            st.download_button(
                f"Download {selected_file}",
                content,
                file_name=selected_file,
                mime="text/plain"
            )

    if st.button("Hide Code Preview", key="hide_code"):
        st.session_state.show_code_preview = False
        st.rerun()


# ═══════════════════════════════════════════════════════
#  Sandbox Results Section
# ═══════════════════════════════════════════════════════

def render_sandbox_results():
    """Render sandbox results section."""
    if st.session_state.pipeline_phase != "sandbox_complete":
        return

    st.markdown("---")
    st.header("&#128736; Sandbox Execution Results")

    sandbox_data = st.session_state.get("sandbox_results", {})
    validation = sandbox_data.get("validation", {}).get("data", {})

    if not validation:
        st.warning("No validation data available.")
        return

    # Status
    success = validation.get("success", False)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "PASS" if success else "FAIL")
    with col2:
        attempts = validation.get("execution", {}).get("attempts", 0)
        st.metric("Attempts", attempts)
    with col3:
        syntax = validation.get("syntax_correctness", 0)
        st.metric("Syntax Score", f"{syntax:.0%}")

    # Execution output
    execution = validation.get("execution", {})
    if execution:
        st.subheader("Execution Output")

        stdout = execution.get("stdout", "")
        stderr = execution.get("stderr", "")

        if stdout:
            with st.expander("Standard Output", expanded=True):
                st.code(stdout, language="text")

        if stderr:
            with st.expander("Standard Error", expanded=True):
                st.code(stderr, language="text")

        if execution.get("last_error"):
            st.error(f"Error: {execution['last_error']}")

    # Logs
    logs_dir = validation.get("logs_dir")
    if logs_dir:
        st.subheader("Execution Logs")
        st.info(f"Logs saved to: {logs_dir}")


# ═══════════════════════════════════════════════════════
#  Testing & Validation Tab
# ═══════════════════════════════════════════════════════

def render_testing_tab():
    """Render the testing and validation tab."""
    st.header("&#127919; Testing & Validation")

    subtab = st.radio(
        "Select Test",
        ["Conformal Prediction", "Source Validation", "Sandbox Tests", "Pipeline Validation"],
        horizontal=True
    )

    if subtab == "Conformal Prediction":
        render_conformal_prediction_tests()
    elif subtab == "Source Validation":
        render_source_validation_tests()
    elif subtab == "Sandbox Tests":
        render_sandbox_tests()
    elif subtab == "Pipeline Validation":
        render_pipeline_validation_tests()


def render_conformal_prediction_tests():
    """Render conformal prediction tests."""
    st.subheader("Conformal Prediction Tests")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Coverage Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                import subprocess
                result = subprocess.run(
                    ["python", "-m", "src.conformal.evaluate",
                     "--calibration-file", "data/calibration/validation_papers.json",
                     "--alpha", "0.1"],
                    capture_output=True,
                    text=True
                )
                st.session_state.conformal_test_output = result.stdout + "\n" + result.stderr

    with col2:
        if st.button("Generate Calibration Data"):
            with st.spinner("Generating..."):
                import subprocess
                result = subprocess.run(
                    ["python", "-m", "src.conformal.generate_calibration", "--synthetic", "--num-synthetic", "10"],
                    capture_output=True,
                    text=True
                )
                st.success("Calibration data generated!")

    if "conformal_test_output" in st.session_state:
        st.code(st.session_state.conformal_test_output, language="text")


def render_source_validation_tests():
    """Render source validation tests."""
    st.subheader("Source Validation Tests")

    # Test with sample sources
    test_sources = [
        {"id": "test_1", "title": "Attention Is All You Need", "venue": "NeurIPS", "year": 2017,
         "text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."},
        {"id": "test_2", "title": "Deep Residual Learning", "venue": "CVPR", "year": 2016,
         "text": "We present a residual learning framework to ease the training of networks that are substantially deeper."},
        {"id": "test_3", "title": "Random Blog Post", "venue": "Blog", "year": 2020,
         "text": "I think AI is cool and stuff. Machine learning is the future!"},
    ]

    if st.button("Validate Test Sources", type="primary"):
        with st.spinner("Validating..."):
            result = validate_sources(test_sources, top_n=3)

        st.subheader("Results")
        for src in result.get("top_sources", []):
            with st.expander(f"{src['title']} (Score: {src['overall_score']:.1f})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Credibility", f"{src['credibility']:.1f}")
                with col2:
                    st.metric("Recency", f"{src['recency']:.1f}")
                with col3:
                    st.metric("Technical", f"{src['technical']:.1f}")


def render_sandbox_tests():
    """Render sandbox tests."""
    st.subheader("Sandbox Tests")

    st.markdown("""
    Test the Windows Sandbox integration:
    1. Create a test sandbox
    2. Execute sample code
    3. View results
    """)

    test_code = st.text_area("Test Code", """
import sys
print("Hello from sandbox!")
print(f"Python version: {sys.version}")

# Test PyTorch availability
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not available")
""", height=200)

    if st.button("Run in Sandbox", type="primary"):
        with st.spinner("Creating sandbox and running code..."):
            from sandbox.windows_sandbox import WindowsSandbox

            sandbox = WindowsSandbox()
            files = [{"path": "test.py", "content": test_code}]
            result = sandbox.run_project(files)

        st.subheader("Results")
        st.json(result)


def render_pipeline_validation_tests():
    """Render pipeline validation tests."""
    st.subheader("Pipeline Validation Tests")

    bases = list_known_bases()
    if not bases:
        st.info("No papers processed yet.")
        return

    selected = st.selectbox("Select paper to validate", bases)

    if st.button("Validate Pipeline Results", type="primary"):
        with st.spinner("Validating..."):
            out_dir = OUTPUTS_DIR / selected

            checks = {}

            # Check for required files
            checks["method.json"] = (out_dir / "method.json").exists()
            checks["knowledge_graph.json"] = (out_dir / "knowledge_graph.json").exists()
            checks["research_phase_results.json"] = (out_dir / "research_phase_results.json").exists()

            # Check code generation
            code_gen_file = out_dir / "code_generation_results.json"
            checks["code_generation_results.json"] = code_gen_file.exists()

            # Check sandbox
            sandbox_file = out_dir / "sandbox_results.json"
            checks["sandbox_results.json"] = sandbox_file.exists()

            # Display results
            st.subheader("Validation Results")
            for check, passed in checks.items():
                icon = "&#9989;" if passed else "&#10060;"
                color = "#00D4AA" if passed else "#FF6B35"
                st.markdown(f"<span style='color: {color};'>{icon} {check}</span>", unsafe_allow_html=True)

            # Summary
            passed_count = sum(checks.values())
            total_count = len(checks)
            st.metric("Checks Passed", f"{passed_count}/{total_count}")


# ═══════════════════════════════════════════════════════
#  Legacy Tabs (RAG Search, Dashboard)
# ═══════════════════════════════════════════════════════

def render_rag_tab():
    """Render the RAG Search tab."""
    st.header("&#128269; Ask a Question")

    bases = list_known_bases()
    search_base = st.selectbox(
        "Select paper to search",
        options=["All Papers"] + bases if bases else ["All Papers"],
        index=0,
    )
    selected_base = None if search_base == "All Papers" else search_base

    q = st.text_input("Your question", placeholder="e.g. What optimizer does the paper use?")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL, label_visibility="collapsed")
    with col2:
        top_k = st.slider("Chunks", 3, 10, DEFAULT_TOP_K, label_visibility="collapsed")
    with col3:
        stream = st.checkbox("Stream", value=True)

    if st.button("&#128270; Search", type="primary", use_container_width=True) and q:
        with st.spinner("Retrieving..."):
            hits = retrieve(q, top_k=top_k, base_name=selected_base)
        if not hits:
            st.warning("No results found.")
        else:
            if selected_base:
                st.info(f"&#128269; Searching within: **{selected_base}**")

            for i, h in enumerate(hits, 1):
                with st.expander(f"[{i}] {h['id']}  (dist={h['distance']:.4f})"):
                    st.caption(f"Base: {h['metadata'].get('base')}  |  Chunk: {h['metadata'].get('chunk_id')}")
                    st.write(h["text"])

            try:
                ctx = format_context(hits, max_chars=MAX_CONTEXT_CHARS)
                st.subheader("&#128161; Answer")
                if stream:
                    ph = st.empty()
                    acc = []
                    for chunk in answer_with_ollama(q, ctx, model, stream=True):
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            acc.append(content)
                            ph.markdown("".join(acc))
                else:
                    with st.spinner("Generating..."):
                        ans = answer_with_ollama(q, ctx, model)
                    st.write(ans)
            except Exception as e:
                st.error(f"Answer generation failed: {e}")


def render_dashboard_tab():
    """Render the project dashboard with latest results."""
    st.header("&#128202; Project Dashboard")

    # Show latest pipeline results
    bases = list_known_bases()
    if not bases:
        st.info("No papers processed yet. Use the Pipeline tab to process a paper.")
        return

    selected = st.selectbox("Select paper", bases)
    results_file = OUTPUTS_DIR / selected / "pipeline_results.json"
    research_file = OUTPUTS_DIR / selected / "research_phase_results.json"

    # Try to load new format results first
    if research_file.exists():
        results = json.loads(research_file.read_text(encoding="utf-8"))
        stages = results.get("stages", {})
        errors = results.get("errors", [])

        st.markdown(
            render_metrics({
                "Stages": f"{len(stages)}/7",
                "Errors": str(len(errors)),
                "Paper": selected[:20],
            }),
            unsafe_allow_html=True
        )

        # Show phase status
        st.subheader("Pipeline Status")
        phase_cols = st.columns(3)
        with phase_cols[0]:
            st.metric("Phase 1: Research", "&#9989; Complete" if "knowledge_graph" in stages else "&#9203; Pending")
        with phase_cols[1]:
            code_file = OUTPUTS_DIR / selected / "code_generation_results.json"
            st.metric("Phase 2: Code Gen", "&#9989; Complete" if code_file.exists() else "&#9203; Pending")
        with phase_cols[2]:
            sandbox_file = OUTPUTS_DIR / selected / "sandbox_results.json"
            st.metric("Phase 3: Sandbox", "&#9989; Complete" if sandbox_file.exists() else "&#9203; Pending")

        # Knowledge Graph Visualization
        st.subheader("&#128203; Knowledge Graph")
        kg_data = stages.get("knowledge_graph", {}).get("data", {})
        if kg_data:
            nodes = kg_data.get("nodes", [])
            edges = kg_data.get("edges", [])

            # Show graph stats
            kg_cols = st.columns(4)
            with kg_cols[0]:
                st.metric("Total Nodes", len(nodes))
            with kg_cols[1]:
                st.metric("Total Edges", len(edges))
            with kg_cols[2]:
                node_types = {}
                for n in nodes:
                    t = n.get("type", "Unknown")
                    node_types[t] = node_types.get(t, 0) + 1
                st.metric("Node Types", len(node_types))
            with kg_cols[3]:
                st.metric("Algorithm", len([n for n in nodes if n.get("type") == "Algorithm"]))

            # Node type filter
            if node_types:
                selected_types = st.multiselect(
                    "Filter by node type",
                    options=list(node_types.keys()),
                    default=list(node_types.keys())
                )

                # Filter nodes
                filtered_nodes = [n for n in nodes if n.get("type") in selected_types]

                # Display nodes in expandable sections
                with st.expander(f"&#128202; View {len(filtered_nodes)} Nodes", expanded=False):
                    for node in filtered_nodes:
                        node_type = node.get("type", "Unknown")
                        node_label = node.get("label", "Unknown")
                        node_id = node.get("id", "")

                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                st.markdown(f"**{node_type}**")
                            with cols[1]:
                                st.markdown(f"{node_label}")
                                if node.get("properties"):
                                    for k, v in node["properties"].items():
                                        st.caption(f"{k}: {v}")
                            st.divider()

            # Edge visualization
            if edges:
                with st.expander(f"&#128203; View {len(edges)} Relationships", expanded=False):
                    for edge in edges:
                        source = edge.get("source", "Unknown")
                        target = edge.get("target", "Unknown")
                        relation = edge.get("relation", "Unknown")

                        st.markdown(f"**{source}** → *{relation}* → **{target}**")
                        if edge.get("properties"):
                            for k, v in edge["properties"].items():
                                st.caption(f"{k}: {v}")
                        st.divider()

            # Simple graph visualization using nodes/edges
            st.subheader("Graph Structure")
            if nodes and edges:
                # Create a simple adjacency list display
                adj = {}
                for edge in edges:
                    src = edge.get("source", "").split("_")[-1] if edge.get("source") else "?"
                    tgt = edge.get("target", "").split("_")[-1] if edge.get("target") else "?"
                    rel = edge.get("relation", "?")
                    if src not in adj:
                        adj[src] = []
                    adj[src].append((tgt, rel))

                # Show top connections
                for src, targets in list(adj.items())[:10]:
                    st.markdown(f"**{src}**")
                    for tgt, rel in targets:
                        st.caption(f"└─ {rel} → {tgt}")

        else:
            st.info("No knowledge graph data available for this paper.")

        # Pipeline Verification Section
        st.subheader("&#128200; Pipeline Verification")

        # Show all stages with their data
        verification_tabs = st.tabs([
            "Ingestion", "Vision", "Chunking", "Method Extraction",
            "Equations", "Datasets", "Knowledge Graph"
        ])

        with verification_tabs[0]:
            ingestion_data = stages.get("ingestion", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('ingestion', {}).get('success') else '⏳ Pending'}")
            if ingestion_data:
                st.json(ingestion_data)

        with verification_tabs[1]:
            vision_data = stages.get("vision", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('vision', {}).get('success') else '⏳ Pending'}")
            if vision_data:
                st.json(vision_data)

        with verification_tabs[2]:
            chunking_data = stages.get("chunking", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('chunking', {}).get('success') else '⏳ Pending'}")
            if chunking_data:
                chunks = chunking_data.get("chunks", [])
                st.metric("Total Chunks", len(chunks))
                for i, chunk in enumerate(chunks[:5]):
                    with st.expander(f"Chunk {i+1}", expanded=False):
                        if isinstance(chunk, str):
                            st.text(chunk[:500])
                        else:
                            st.text(chunk.get("text", "")[:500])

        with verification_tabs[3]:
            method_data = stages.get("method_extraction", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('method_extraction', {}).get('success') else '⏳ Pending'}")
            if method_data:
                method_struct = method_data.get("method_struct", {})
                st.write(f"**Algorithm:** {method_struct.get('algorithm_name', 'N/A')}")
                st.write(f"**Datasets:** {', '.join(method_struct.get('datasets', []))}")
                if method_struct.get('architecture', {}).get('layer_types'):
                    st.write(f"**Layer Types:** {', '.join(method_struct['architecture']['layer_types'])}")
                st.json(method_struct)

        with verification_tabs[4]:
            equation_data = stages.get("equations", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('equations', {}).get('success') else '⏳ Pending'}")
            if equation_data:
                equations = equation_data.get("equations", [])
                for eq in equations:
                    st.markdown(f"**{eq.get('name', 'Equation')}**")
                    st.code(eq.get('content', ''), language="latex")

        with verification_tabs[5]:
            dataset_data = stages.get("datasets", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('datasets', {}).get('success') else '⏳ Pending'}")
            if dataset_data:
                st.json(dataset_data)

        with verification_tabs[6]:
            kg_data_detail = stages.get("knowledge_graph", {}).get("data", {})
            st.write(f"**Status:** {'✅ Complete' if stages.get('knowledge_graph', {}).get('success') else '⏳ Pending'}")
            if kg_data_detail:
                st.json(kg_data_detail)

    elif results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
        stages = results.get("stages", {})
        errors = results.get("errors", [])

        st.markdown(
            render_metrics({
                "Stages": f"{len(stages)}/9",
                "Errors": str(len(errors)),
                "Paper": selected[:20],
            }),
            unsafe_allow_html=True
        )
    else:
        st.info(f"No results for '{selected}'.")
        return

    # File explorer
    with st.expander("&#128193; Output Files"):
        out_dir = OUTPUTS_DIR / selected
        if out_dir.exists():
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    st.caption(f"&#128196; {f.relative_to(out_dir)}")
        else:
            st.info("No output directory found.")


# ═══════════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div class="app-header">
        <h1>{APP_TITLE}</h1>
        <p class="app-subtitle">{APP_SUBTITLE} | Phase 6: NewResearcher + 3-Phase Workflow</p>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tabs = st.tabs([
        "&#128269; RAG Search",
        "&#9889; Pipeline (v2)",
        "&#128300; NewResearcher",
        "&#127919; Testing",
        "&#128202; Dashboard"
    ])

    with tabs[0]:
        render_rag_tab()

    with tabs[1]:
        render_pipeline_tab_v2()
        render_code_preview()
        render_sandbox_results()

    with tabs[2]:
        render_newresearcher_tab()

    with tabs[3]:
        render_testing_tab()

    with tabs[4]:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
