# src/ingest_pdf.py
import os
from pathlib import Path
from utils import extract_text_from_pdf, chunk_text_by_words

# Resolve project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
OUT_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"

OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

def process_one(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    base = os.path.splitext(os.path.basename(str(pdf_path)))[0]
    txt_path = OUT_TEXT_DIR / f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    chunks = chunk_text_by_words(text)
    # optional: save chunk files
    for i, c in enumerate(chunks):
        with open(OUT_TEXT_DIR / f"{base}_chunk_{i}.txt", "w", encoding="utf-8") as cf:
            cf.write(c)
    print(f"Saved text + {len(chunks)} chunks for {pdf_path}")

if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    for p in pdfs:
        process_one(p)