"""
Research2Code-GenAI — Streamlit Interface (Phase 5 Overhaul)

Premium UI with:
  - Real-time multi-agent pipeline progress tracker
  - Confidence score dashboard (from Phase 3 conformal prediction)
  - Interactive knowledge graph explorer
  - Modern dark-theme styling with glassmorphism
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


def render_pipeline_stage(name: str, status: str, duration: Optional[float] = None, detail: str = "") -> str:
    """Render a pipeline stage card."""
    icons = {"done": "✅", "active": "⏳", "pending": "⬜", "error": "❌"}
    icon = icons.get(status, "⬜")
    time_str = f" ({duration:.1f}s)" if duration else ""
    detail_str = f'<div style="color: #8B95A8; font-size: 0.8rem; margin-top: 4px;">{detail}</div>' if detail else ""
    html = f"""<div class="stage-card {status}">
        <span style="font-size: 1.1rem;">{icon}</span>
        <span style="color: #E8ECF4; font-weight: 500; margin-left: 8px;">{name}</span>
        <span style="color: #8B95A8; font-size: 0.8rem;">{time_str}</span>
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


# ═══════════════════════════════════════════════════════
#  Pipeline Runner with Live Progress
# ═══════════════════════════════════════════════════════

PIPELINE_STAGES = [
    ("📥 Ingestion", "ingest"),
    ("👁️ Vision Processing", "vision"),
    ("✂️ Chunking", "chunking"),
    ("🔬 Method Extraction", "method_extractor"),
    ("📐 Equation Processing", "equations"),
    ("📊 Dataset Processing", "datasets"),
    ("💻 Code Generation", "code_architect"),
    ("🕸️ Knowledge Graph", "graph_builder"),
    ("✅ Validation", "validator"),
]


def run_multiagent_with_progress(base_name: str, text: str, progress_container):
    """Run the multi-agent pipeline with real-time progress updates."""
    from agents.orchestrator import Orchestrator
    from agents.base import AgentMessage

    orchestrator = Orchestrator()
    stage_status = {key: "pending" for _, key in PIPELINE_STAGES}

    # Create placeholders for each stage
    stage_placeholders = {}
    with progress_container:
        for display_name, key in PIPELINE_STAGES:
            stage_placeholders[key] = st.empty()
            stage_placeholders[key].markdown(
                render_pipeline_stage(display_name, "pending"),
                unsafe_allow_html=True,
            )

    # Run the pipeline
    results = orchestrator.process_paper(text=text, paper_base=base_name)

    # Update stages based on results
    stages_data = results.get("stages", {})
    stage_key_map = {
        "ingestion": "ingest",
        "vision": "vision",
        "chunking": "chunking",
        "method_extraction": "method_extractor",
        "equations": "equations",
        "datasets": "datasets",
        "code_generation": "code_architect",
        "knowledge_graph": "graph_builder",
        "validation": "validator",
    }

    for stage_name, stage_key in stage_key_map.items():
        stage_data = stages_data.get(stage_name, {})
        display_name = next((d for d, k in PIPELINE_STAGES if k == stage_key), stage_key)

        if isinstance(stage_data, dict):
            success = stage_data.get("success", True)
            status = "done" if success else "error"
            detail = ""
            if stage_key == "method_extractor" and stage_data.get("data"):
                conf = stage_data["data"].get("overall_confidence")
                if conf is not None:
                    detail = f"Confidence: {conf:.0%}"
        elif isinstance(stage_data, list):
            status = "done"
            detail = f"{len(stage_data)} items"
        else:
            status = "done"
            detail = ""

        stage_placeholders[stage_key].markdown(
            render_pipeline_stage(display_name, status, detail=detail),
            unsafe_allow_html=True,
        )

    return results


# ═══════════════════════════════════════════════════════
#  Tab Renderers
# ═══════════════════════════════════════════════════════

def render_rag_tab():
    """Render the RAG Search tab."""
    st.header("🔍 Ask a Question")

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

    if st.button("🔎 Search", type="primary", use_container_width=True) and q:
        with st.spinner("Retrieving..."):
            hits = retrieve(q, top_k=top_k, base_name=selected_base)
        if not hits:
            st.warning("No results found.")
        else:
            if selected_base:
                st.info(f"🔍 Searching within: **{selected_base}**")

            for i, h in enumerate(hits, 1):
                with st.expander(f"[{i}] {h['id']}  (dist={h['distance']:.4f})"):
                    st.caption(f"Base: {h['metadata'].get('base')}  |  Chunk: {h['metadata'].get('chunk_id')}")
                    st.write(h["text"])

            try:
                ctx = format_context(hits, max_chars=MAX_CONTEXT_CHARS)
                st.subheader("💡 Answer")
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


def render_pipeline_tab():
    """Render the Paper → Code pipeline tab with live progress."""
    st.header("⚡ Research → Code Pipeline")

    # Upload or select paper
    up = st.file_uploader("Upload a PDF", type=["pdf"], key="p2_pdf")
    base_name = None

    if up is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != up.name:
            pdf_path = save_uploaded_pdf(up)
            with st.spinner(f"Extracting text from {up.name}..."):
                base_name, n_chunks = ingest_pdf(pdf_path)
                index_documents(base_name)
            st.session_state.last_uploaded_file = up.name
            st.session_state.last_base_name = base_name
            st.session_state.last_n_chunks = n_chunks
            st.success(f"✅ {up.name}: {n_chunks} chunks indexed")
        else:
            base_name = st.session_state.last_base_name
            n_chunks = st.session_state.last_n_chunks
            st.success(f"✅ {up.name}: {n_chunks} chunks indexed")
    else:
        bases = list_known_bases()
        if bases:
            base_name = st.selectbox("Or select an existing paper", options=bases, index=0)

    if not base_name:
        st.info("📄 Upload a PDF or select a paper to start the pipeline.")
        return

    # Pipeline controls
    col1, col2, col3 = st.columns(3)
    with col1:
        run_pipeline = st.button("🚀 Run Multi-Agent Pipeline", type="primary", use_container_width=True)
    with col2:
        run_simple = st.button("▶️ Run Simple Pipeline", use_container_width=True)
    with col3:
        download_btn = st.button("📦 Download Artifacts", use_container_width=True)

    if download_btn:
        try:
            data = build_artifacts_zip(base_name)
            st.download_button(
                "⬇️ Download ZIP", data=data,
                file_name=f"{base_name}_artifacts.zip", mime="application/zip",
            )
        except Exception as e:
            st.error(f"Failed: {e}")

    if run_simple:
        with st.spinner("Running simple pipeline..."):
            try:
                out_dir = run_paper_to_code(base_name)
                st.success(f"✅ Output: {out_dir}")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    if run_pipeline:
        # Load text for the pipeline
        text_file = RAW_TEXT_DIR / f"{base_name}.txt"
        if not text_file.exists():
            st.error(f"Text file not found: {text_file}")
            return

        text = text_file.read_text(encoding="utf-8")

        st.subheader("📊 Pipeline Progress")
        progress_container = st.container()

        start = time.time()
        results = run_multiagent_with_progress(base_name, text, progress_container)
        elapsed = time.time() - start

        errors = results.get("errors", [])

        # Summary metrics
        stages = results.get("stages", {})
        n_stages = len(stages)
        n_errors = len(errors)

        st.markdown(
            render_metrics({
                "Stages": f"{n_stages}/9",
                "Errors": str(n_errors),
                "Time": f"{elapsed:.1f}s",
                "Status": "✅" if n_errors == 0 else "⚠️",
            }),
            unsafe_allow_html=True,
        )

        if errors:
            with st.expander("⚠️ Errors", expanded=True):
                for err in errors:
                    st.error(err)

        # Confidence Dashboard
        method_data = stages.get("method_extraction", {})
        if isinstance(method_data, dict) and method_data.get("data"):
            render_confidence_dashboard(method_data["data"])

        # Knowledge Graph
        kg_data = stages.get("knowledge_graph", {})
        if isinstance(kg_data, dict) and kg_data.get("data"):
            render_knowledge_graph(kg_data["data"])

        # Generated Code
        code_data = stages.get("code_generation", {})
        if isinstance(code_data, dict) and code_data.get("data"):
            render_generated_code(code_data["data"])

        # Validation Results
        val_data = stages.get("validation", {})
        if isinstance(val_data, dict) and val_data.get("data"):
            render_validation_results(val_data["data"])

        # Save results
        out_dir = OUTPUTS_DIR / base_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pipeline_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8",
        )


def render_confidence_dashboard(method_data: Dict[str, Any]):
    """Render the confidence score dashboard from conformal prediction."""
    st.subheader("🎯 Confidence Dashboard")

    confidence = method_data.get("confidence", {})
    overall = method_data.get("overall_confidence", 0)
    low_fields = method_data.get("low_confidence_fields", [])

    if not confidence:
        st.info("No confidence data available (LLM was not used for extraction).")
        return

    # Overall confidence meter
    overall_pct = int(overall * 100)
    color = "#00D4AA" if overall >= 0.7 else "#FFB800" if overall >= 0.4 else "#FF3366"
    st.markdown(f"""
    <div style="text-align: center; margin: 16px 0;">
        <div style="font-size: 3rem; font-weight: 700; color: {color};">{overall_pct}%</div>
        <div style="color: #8B95A8; font-size: 0.9rem;">Overall Extraction Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Per-field confidence bars
    bars_html = ""
    for field_name, conf_data in confidence.items():
        if isinstance(conf_data, dict):
            score = conf_data.get("score", 0)
            source = conf_data.get("source", "unknown")
            label = f"{field_name} ({source})"
            bars_html += render_confidence_bar(score, label)

    if bars_html:
        st.markdown(bars_html, unsafe_allow_html=True)

    # Low confidence warnings
    if low_fields:
        st.warning(f"⚠️ Low confidence fields: **{', '.join(low_fields)}** — these may be hallucinated by the LLM.")


def render_knowledge_graph(kg_data: Dict[str, Any]):
    """Render the knowledge graph visualization."""
    st.subheader("🕸️ Knowledge Graph")

    nodes = kg_data.get("nodes", [])
    edges = kg_data.get("edges", [])

    if not nodes:
        st.info("No knowledge graph data.")
        return

    # Render nodes by type
    node_html = ""
    type_classes = {
        "paper": "kg-paper",
        "algorithm": "kg-algo",
        "dataset": "kg-dataset",
        "equation": "kg-equation",
        "citation": "kg-paper",
    }

    for node in nodes:
        ntype = node.get("type", "paper")
        label = node.get("label", node.get("id", "?"))
        css_class = type_classes.get(ntype, "kg-paper")
        node_html += f'<span class="kg-node {css_class}">{label}</span>'

    st.markdown(node_html, unsafe_allow_html=True)

    # Edge list
    if edges:
        with st.expander(f"🔗 Connections ({len(edges)} edges)"):
            for edge in edges:
                src = edge.get("source", "?")
                tgt = edge.get("target", "?")
                rel = edge.get("relation", "→")
                st.caption(f"{src} — {rel} → {tgt}")


def render_generated_code(code_data: Dict[str, Any]):
    """Render generated code files."""
    st.subheader("💻 Generated Code")

    files = code_data.get("files", [])
    if not files:
        st.info("No code generated.")
        return

    for f in files:
        path = f.get("path", "unknown")
        content = f.get("content", "")
        lang = "python" if path.endswith(".py") else "text"
        with st.expander(f"📄 {path}", expanded=(path == "model.py")):
            st.code(content, language=lang)


def render_validation_results(val_data: Dict[str, Any]):
    """Render validation results."""
    st.subheader("🔍 Validation Results")

    syntax = val_data.get("syntax_correctness", 0)
    imports = val_data.get("import_resolution", 0)
    execution = val_data.get("execution", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Syntax", f"{syntax:.0%}")
    with col2:
        st.metric("Imports", f"{imports:.0%}")
    with col3:
        if execution:
            exec_status = "✅ Pass" if execution.get("success") else "❌ Fail"
            attempts = execution.get("attempts", 0)
            st.metric("Execution", exec_status, delta=f"Attempt {attempts}")
        else:
            st.metric("Execution", "—")

    # Show execution details
    if execution and execution.get("stdout"):
        with st.expander("📋 Execution Output"):
            st.code(execution["stdout"][:2000])


def render_cleaner_sidebar():
    """Render the database cleaner in the sidebar."""
    st.divider()
    st.header("🧹 Database Cleaner")
    cleaner_action = st.selectbox(
        "Action",
        ["Refresh Index", "Clean Old Chunks", "Clean Base", "Full Clean"],
    )

    days_old = 30
    clean_base_name = None

    if cleaner_action in ["Clean Old Chunks", "Full Clean"]:
        days_old = st.number_input("Days old", 1, 365, 30)

    if cleaner_action == "Clean Base":
        bases = list_known_bases()
        clean_base_name = st.selectbox("Base", options=bases) if bases else None

    dry_run = st.checkbox("Dry run", value=True)

    if st.button("Run Cleaner"):
        from agents.cleaner_agent import CleanerAgent
        from agents.base import AgentMessage

        cleaner = CleanerAgent()
        action_map = {
            "Refresh Index": "refresh_index",
            "Clean Old Chunks": "clean_old",
            "Clean Base": "clean_base",
            "Full Clean": "full_clean",
        }
        payload = {"action": action_map[cleaner_action], "dry_run": dry_run}
        if cleaner_action in ["Clean Old Chunks", "Full Clean"]:
            payload["days_old"] = days_old
        if cleaner_action == "Clean Base" and clean_base_name:
            payload["base_name"] = clean_base_name

        msg = AgentMessage(
            agent_id="user", message_type="request", payload=payload,
        )
        with st.spinner("Running..."):
            response = cleaner.process(msg)
        if response.success:
            st.success("✅ Done!")
            st.json(response.data)
        else:
            st.error(f"❌ {response.error}")


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
        <p class="app-subtitle">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        render_cleaner_sidebar()

    # Main Tabs
    tabs = st.tabs(["🔍 RAG Search", "⚡ Pipeline", "📊 Dashboard"])

    with tabs[0]:
        render_rag_tab()

    with tabs[1]:
        render_pipeline_tab()

    with tabs[2]:
        render_dashboard_tab()


def render_dashboard_tab():
    """Render the project dashboard with latest results."""
    st.header("📊 Project Dashboard")

    # Show latest pipeline results
    bases = list_known_bases()
    if not bases:
        st.info("No papers processed yet. Use the Pipeline tab to process a paper.")
        return

    selected = st.selectbox("Select paper", bases)
    results_file = OUTPUTS_DIR / selected / "pipeline_results.json"

    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
        stages = results.get("stages", {})
        errors = results.get("errors", [])

        st.markdown(
            render_metrics({
                "Stages": f"{len(stages)}/9",
                "Errors": str(len(errors)),
                "Paper": selected[:20],
            }),
            unsafe_allow_html=True,
        )

        # Method extraction details
        method_data = stages.get("method_extraction", {})
        if isinstance(method_data, dict) and method_data.get("data"):
            render_confidence_dashboard(method_data["data"])

        # Knowledge graph
        kg_data = stages.get("knowledge_graph", {})
        if isinstance(kg_data, dict) and kg_data.get("data"):
            render_knowledge_graph(kg_data["data"])
    else:
        st.info(f"No pipeline results for '{selected}'. Run the multi-agent pipeline first.")

    # File explorer
    with st.expander("📁 Output Files"):
        out_dir = OUTPUTS_DIR / selected
        if out_dir.exists():
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    st.caption(f"📄 {f.relative_to(out_dir)}")
        else:
            st.info("No output directory found.")


if __name__ == "__main__":
    main()
