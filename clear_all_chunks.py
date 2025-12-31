#!/usr/bin/env python
"""
Quick script to clear ALL chunks from files and ChromaDB.
Use this to start fresh when testing.
"""
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "research_papers"

print("🧹 Clearing all chunks...")

# 1. Delete all chunk files
chunk_files = list(RAW_TEXT_DIR.glob("*_chunk_*.txt"))
deleted_files = 0
for chunk_file in chunk_files:
    try:
        chunk_file.unlink()
        deleted_files += 1
    except Exception as e:
        print(f"Warning: Could not delete {chunk_file}: {e}")

print(f"✅ Deleted {deleted_files} chunk files")

# 2. Delete ChromaDB directory entirely (will be recreated on next index)
try:
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"✅ Deleted ChromaDB directory (will be recreated fresh on next indexing)")
    else:
        print("ℹ️  ChromaDB directory doesn't exist")
except Exception as e:
    print(f"⚠️  Could not delete ChromaDB directory: {e}")
    print("   (This is okay - it will be recreated fresh when you index again)")

print("\n✨ All chunks cleared! You can now upload a new paper and it will be processed fresh.")

