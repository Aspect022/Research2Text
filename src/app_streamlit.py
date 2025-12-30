import io
from pathlib import Path

import streamlit as st

from utils import extract_text_from_pdf, chunk_text_by_words
from index_documents import index_all
from query_rag import retrieve, format_context, answer_with_ollama
from export_utils import build_artifacts_zip, list_known_bases, build_code_zip
from paper_to_code import run_paper_to_code


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"


def save_uploaded_pdf(upload) -> Path:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_PDF_DIR / upload.name
    out_path.write_bytes(upload.read())
    return out_path


def ingest_pdf(pdf_path: Path):
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_pdf(str(pdf_path))
    base = pdf_path.stem
    (RAW_TEXT_DIR / f"{base}.txt").write_text(text, encoding="utf-8")
    chunks = chunk_text_by_words(text)
    for i, c in enumerate(chunks):
        (RAW_TEXT_DIR / f"{base}_chunk_{i}.txt").write_text(c, encoding="utf-8")
    return base, len(chunks)


def main():
    st.set_page_config(page_title="Research2Text RAG", page_icon="📄")
    st.title("Research2Text - Local RAG")

    with st.sidebar:
        st.header("Upload & Ingest")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        model = st.text_input("Ollama model", value="gpt-oss:120b-cloud")
        top_k = st.slider("Top K chunks", min_value=3, max_value=10, value=5)
        run_answer = st.checkbox("Generate answer with Ollama", value=True)
        stream = st.checkbox("Stream answer", value=True)
        max_ctx = st.number_input("Max context chars", min_value=1000, max_value=12000, value=4000, step=500)
        
        st.divider()
        st.header("🧹 Database Cleaner")
        cleaner_action = st.selectbox(
            "Cleanup Action",
            ["Refresh Index", "Clean Old Chunks", "Clean Base", "Full Clean"],
            help="Refresh index or remove outdated chunks"
        )
        if cleaner_action in ["Clean Old Chunks", "Full Clean"]:
            days_old = st.number_input("Days old", min_value=1, max_value=365, value=30, help="Remove chunks older than this")
        else:
            days_old = 30
        
        if cleaner_action == "Clean Base":
            bases = list_known_bases()
            clean_base_name = st.selectbox("Select base to clean", options=bases) if bases else None
        else:
            clean_base_name = None
        
        dry_run = st.checkbox("Dry run (preview only)", value=True, help="Preview changes without applying")
        
        if st.button("Run Cleaner", type="secondary"):
            from agents.cleaner_agent import CleanerAgent
            from agents.base import AgentMessage
            
            cleaner = CleanerAgent()
            
            action_map = {
                "Refresh Index": "refresh_index",
                "Clean Old Chunks": "clean_old",
                "Clean Base": "clean_base",
                "Full Clean": "full_clean"
            }
            
            payload = {
                "action": action_map[cleaner_action],
                "dry_run": dry_run
            }
            
            if cleaner_action in ["Clean Old Chunks", "Full Clean"]:
                payload["days_old"] = days_old
            if cleaner_action == "Clean Base" and clean_base_name:
                payload["base_name"] = clean_base_name
            
            msg = AgentMessage(
                agent_id="user",
                message_type="request",
                payload=payload
            )
            
            with st.spinner("Running cleaner..."):
                response = cleaner.process(msg)
            
            if response.success:
                data = response.data
                st.success(f"✅ Cleaner completed!")
                st.json(data)
            else:
                st.error(f"❌ Cleaner failed: {response.error}")

    if uploaded is not None and st.button("Process PDF"):
        pdf_path = save_uploaded_pdf(uploaded)
        base, n_chunks = ingest_pdf(pdf_path)
        st.success(f"Ingested {uploaded.name}: {n_chunks} chunks")
        with st.spinner("Indexing chunks into Chroma..."):
            index_all()
        st.success("Index updated.")

    tabs = st.tabs(["RAG Search", "Paper → Code"])

    with tabs[0]:
        st.header("Ask a question")
    q = st.text_input("Your question")
    if st.button("Search") and q:
        with st.spinner("Retrieving relevant chunks..."):
            hits = retrieve(q, top_k=top_k)
        if not hits:
            st.warning("No results found.")
        else:
            for i, h in enumerate(hits, 1):
                with st.expander(f"[{i}] {h['id']}  (distance={h['distance']:.4f})"):
                    st.write(f"Base: {h['metadata'].get('base')}  |  Chunk: {h['metadata'].get('chunk_id')}")
                    st.write(h["text"])

            if run_answer:
                try:
                    ctx = format_context(hits, max_chars=int(max_ctx))
                    st.subheader("Answer")
                    if stream:
                        ph = st.empty()
                        acc = []
                        with st.spinner("Generating answer with Ollama (streaming)..."):
                            for chunk in answer_with_ollama(q, ctx, model, stream=True):
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    acc.append(content)
                                    ph.markdown("".join(acc))
                    else:
                        with st.spinner("Generating answer with Ollama..."):
                            ans = answer_with_ollama(q, ctx, model)
                        st.write(ans)
                except Exception as e:
                    st.error(f"Answer generation failed: {e}")

    with tabs[1]:
        st.header("Executable Research (Phase 2)")
        
        # Multi-agent architecture option
        use_multiagent = st.checkbox(
            "Use Multi-Agent Architecture (9 specialized agents)",
            value=False,
            help="Enable the new multi-agent system with Vision, Graph Builder, and enhanced processing"
        )
        
        # Option A: upload a PDF and use it directly
        up = st.file_uploader("Upload a PDF to generate code", type=["pdf"], key="p2_pdf")
        base_name = None
        if up is not None:
            # Save and ingest + index
            pdf_path = save_uploaded_pdf(up)
            base_name, n_chunks = ingest_pdf(pdf_path)
            with st.spinner("Indexing chunks into Chroma..."):
                index_all()
            st.success(f"Prepared {base_name}: {n_chunks} chunks indexed")
        else:
            # Option B: pick an existing base
            bases = list_known_bases()
            base_name = st.selectbox(
                "Or select an existing paper base",
                options=bases,
                index=0 if bases else None,
                placeholder="No processed papers found yet",
            )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Code") and base_name:
                with st.spinner("Running paper → code pipeline..."):
                    try:
                        if use_multiagent:
                            from paper_to_code_multiagent import run_paper_to_code as run_multiagent
                            out_dir = run_multiagent(base_name, use_multiagent=True)
                        else:
                            out_dir = run_paper_to_code(base_name)
                        st.success(f"Artifacts created in: {out_dir}")
                        if use_multiagent:
                            st.info("✨ Multi-agent processing complete! Check knowledge_graph.json for graph data.")
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
        with col2:
            if st.button("Download Artifacts Zip") and base_name:
                try:
                    data = build_artifacts_zip(base_name)
                    st.download_button(
                        label="Download ZIP",
                        data=data,
                        file_name=f"{base_name}_artifacts.zip",
                        mime="application/zip",
                    )
                except Exception as e:
                    st.error(f"Failed to build zip: {e}")
        # Code-only download next to artifacts
        with st.container():
            if base_name:
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Download Code Only ZIP"):
                        try:
                            data = build_code_zip(base_name)
                            st.download_button(
                                label="Download Code ZIP",
                                data=data,
                                file_name=f"{base_name}_code.zip",
                                mime="application/zip",
                            )
                        except Exception as e:
                            st.error(f"Failed to build code zip: {e}")


if __name__ == "__main__":
    main()


