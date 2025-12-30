#!/usr/bin/env python
# src/index_documents.py
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "research_papers"


def load_chunks_for_base(base_name: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for p in sorted(RAW_TEXT_DIR.glob(f"{base_name}_chunk_*.txt")):
        try:
            idx_part = p.stem.split("_chunk_")[-1]
            chunk_id = int(idx_part)
        except ValueError:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks.append({
            "id": f"{base_name}:{chunk_id}",
            "text": text,
            "chunk_id": chunk_id,
            "base": base_name,
        })
    return chunks


def index_all() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    bases = sorted({p.stem.split("_chunk_")[0] for p in RAW_TEXT_DIR.glob("*_chunk_*.txt")})
    if not bases:
        print("No chunk files found in", RAW_TEXT_DIR)
        return

    for base in bases:
        chunks = load_chunks_for_base(base)
        if not chunks:
            continue
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"base": c["base"], "chunk_id": c["chunk_id"]} for c in chunks]

        batch_size = 64
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )
        print(f"Indexed {len(ids)} chunks for {base}")


if __name__ == "__main__":
    index_all()