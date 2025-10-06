# src/query_rag.py
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "research_papers"


def retrieve(query: str, top_k: int = 5) -> List[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    results = collection.query(query_texts=[query], n_results=top_k)

    docs: List[dict] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for i in range(len(documents)):
        docs.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        })
    return docs


def format_context(chunks: List[dict], max_chars: int = 4000) -> str:
    lines = []
    used = 0
    for i, d in enumerate(chunks, 1):
        base = d["metadata"].get("base")
        cid = d["metadata"].get("chunk_id")
        body = d["text"]
        remaining = max(0, max_chars - used)
        if remaining <= 0:
            break
        take = body[:remaining]
        lines.append(f"[Chunk {i} | {base}:{cid}]\n{take}")
        used += len(take)
        if used >= max_chars:
            break
    return "\n\n".join(lines)


def answer_with_ollama(query: str, context: str, model: str, stream: bool = False):
    import ollama
    prompt = (
        "You are a helpful research assistant. Answer the question using ONLY the provided context.\n"
        "If the answer cannot be found in the context, say 'I cannot find this in the provided paper.'\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    if stream:
        return ollama.chat(model=model, stream=True, messages=[{"role": "user", "content": prompt}])
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.get("message", {}).get("content", "")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query text")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answer", action="store_true", help="Generate an answer with Ollama")
    parser.add_argument("--model", type=str, default="gpt-oss:120b-cloud", help="Ollama model name")
    parser.add_argument("--stream", action="store_true", help="Stream tokens from Ollama")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    args = parser.parse_args()

    hits = retrieve(args.query, top_k=args.k)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['id']} (distance={h['distance']:.4f}) base={h['metadata'].get('base')} chunk={h['metadata'].get('chunk_id')}")
        print(h["text"][:400].replace("\n", " ") + ("..." if len(h["text"]) > 400 else ""))
        print()

    if args.answer and hits:
        ctx = format_context(hits, max_chars=args.max_context_chars)
        try:
            if args.stream:
                print("=== Answer (streaming) ===")
                stream = answer_with_ollama(args.query, ctx, args.model, stream=True)
                for chunk in stream:
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        print(content, end="", flush=True)
                print()
            else:
                ans = answer_with_ollama(args.query, ctx, args.model)
                print("=== Answer ===")
                print(ans)
        except Exception as e:
            print(f"Answer generation failed: {e}")

