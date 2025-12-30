# src/utils.py
import fitz   # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        text = "\n\n".join(pages)
        # basic cleaning
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    finally:
        doc.close()

def chunk_text_by_words(text, chunk_size_words=700, overlap_words=100):
    words = text.split()
    if chunk_size_words <= 0:
        return [text]
    # Ensure overlap is smaller than chunk size to guarantee forward progress
    overlap_words = max(0, min(overlap_words, max(0, chunk_size_words - 1)))

    chunks = []
    start = 0
    total_words = len(words)

    while start < total_words:
        end = min(start + chunk_size_words, total_words)
        chunks.append(" ".join(words[start:end]))

        if end >= total_words:
            break

        # Advance start with overlap
        start = max(0, end - overlap_words)

    return chunks