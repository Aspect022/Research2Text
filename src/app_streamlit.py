import io
from pathlib import Path

import streamlit as st

from utils import extract_text_from_pdf, chunk_text_by_words
from index_documents import index_all
from query_rag import retrieve, format_context, answer_with_ollama


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

    if uploaded is not None and st.button("Process PDF"):
        pdf_path = save_uploaded_pdf(uploaded)
        base, n_chunks = ingest_pdf(pdf_path)
        st.success(f"Ingested {uploaded.name}: {n_chunks} chunks")
        with st.spinner("Indexing chunks into Chroma..."):
            index_all()
        st.success("Index updated.")

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


if __name__ == "__main__":
    main()


