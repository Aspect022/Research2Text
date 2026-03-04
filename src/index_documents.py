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
    """Load chunks for a specific base name, handling special characters in filenames."""
    chunks: List[Dict[str, str]] = []
    
    # Instead of using glob with special characters, iterate all chunk files
    # and filter by base name manually
    all_chunk_files = list(RAW_TEXT_DIR.glob("*_chunk_*.txt"))
    
    for p in all_chunk_files:
        try:
            # Extract base and chunk_id from filename
            stem = p.stem
            if "_chunk_" not in stem:
                continue
            
            file_base = stem.split("_chunk_")[0]
            # Match base name (handle special characters)
            if file_base != base_name:
                continue
            
            idx_part = stem.split("_chunk_")[-1]
            chunk_id = int(idx_part)
            
            text = p.read_text(encoding="utf-8", errors="ignore")
            chunks.append({
                "id": f"{base_name}:{chunk_id}",
                "text": text,
                "chunk_id": chunk_id,
                "base": base_name,
            })
        except (ValueError, IndexError) as e:
            # Skip files that don't match expected pattern
            continue
    
    # Sort by chunk_id
    chunks.sort(key=lambda x: x["chunk_id"])
    return chunks


def index_documents(target_base: str = None) -> None:
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

    # Find all chunk files - need to escape special characters in glob
    # Get all files ending with _chunk_*.txt and extract base names
    all_chunk_files = list(RAW_TEXT_DIR.glob("*_chunk_*.txt"))
    bases = set()
    for p in all_chunk_files:
        # Extract base name by finding _chunk_ in the stem
        stem = p.stem
        if "_chunk_" in stem:
            base = stem.split("_chunk_")[0]
            bases.add(base)
    
    bases = sorted(bases)
    if target_base:
        bases = [b for b in bases if b == target_base]
        
    if not bases:
        print(f"No chunk files found in {RAW_TEXT_DIR} to index.")
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
    import sys
    if len(sys.argv) > 1:
        index_documents(sys.argv[1])
    else:
        index_documents()