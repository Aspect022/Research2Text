# src/query_rag.py
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL


def retrieve(query: str, top_k: int = 5, base_name: str = None) -> List[dict]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: The search query text
        top_k: Number of results to return
        base_name: Optional filter to only search within a specific paper base name
    
    Returns:
        List of dictionaries containing id, text, metadata, and distance
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    
    # If base_name is provided, filter by metadata
    # ChromaDB where filter syntax: {"metadata_field": "value"}
    where_filter = None
    if base_name:
        where_filter = {"base": base_name}
    
    try:
        if where_filter:
            results = collection.query(
                query_texts=[query], 
                n_results=top_k,
                where=where_filter
            )
        else:
            results = collection.query(
                query_texts=[query], 
                n_results=top_k
            )
    except Exception as e:
        # Fallback: query without filter if filter syntax fails
        print(f"Warning: Filter failed ({e}), searching across all papers")
        results = collection.query(
            query_texts=[query], 
            n_results=top_k
        )

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
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--stream", action="store_true", help="Stream tokens from Ollama")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--base", type=str, default=None, help="Filter search to a specific paper base name")
    args = parser.parse_args()

    hits = retrieve(args.query, top_k=args.k, base_name=args.base)
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

