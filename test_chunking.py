#!/usr/bin/env python
"""Test script to manually chunk a PDF and verify it works."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import extract_text_from_pdf, chunk_text_by_words

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"

# Get the most recent PDF
pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)

if not pdfs:
    print("No PDFs found in data/raw_pdfs/")
    sys.exit(1)

pdf_path = pdfs[0]
print(f"Testing with: {pdf_path.name}")

# Extract text
print("\n1. Extracting text from PDF...")
try:
    text = extract_text_from_pdf(str(pdf_path))
    print(f"   ✅ Extracted {len(text)} characters, {len(text.split())} words")
    if len(text.strip()) < 10:
        print("   ⚠️  WARNING: Text is very short - PDF might be scanned or corrupted")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Chunk text
print("\n2. Chunking text...")
try:
    chunks = chunk_text_by_words(text)
    print(f"   ✅ Created {len(chunks)} chunks")
    if chunks:
        print(f"   First chunk: {chunks[0][:100]}...")
        print(f"   Last chunk: {chunks[-1][:100]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Save chunks
print("\n3. Saving chunks...")
base = pdf_path.stem
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Save main text
text_path = RAW_TEXT_DIR / f"{base}.txt"
text_path.write_text(text, encoding="utf-8")
print(f"   ✅ Saved main text: {text_path}")

# Save chunks
saved_count = 0
for i, chunk in enumerate(chunks):
    chunk_path = RAW_TEXT_DIR / f"{base}_chunk_{i}.txt"
    chunk_path.write_text(chunk, encoding="utf-8")
    saved_count += 1

print(f"   ✅ Saved {saved_count} chunk files")

# Verify files exist
print("\n4. Verifying files...")
chunk_files = list(RAW_TEXT_DIR.glob(f"{base}_chunk_*.txt"))
print(f"   Found {len(chunk_files)} chunk files on disk")
if len(chunk_files) != len(chunks):
    print(f"   ⚠️  WARNING: Expected {len(chunks)} files, found {len(chunk_files)}")
else:
    print(f"   ✅ All chunk files verified!")

print(f"\n✨ Done! Base name: '{base}'")
print(f"   You can now run: python src/index_documents.py")

