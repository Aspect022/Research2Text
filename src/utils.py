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
    """Chunk text by words with overlap.
    
    Args:
        text: The text to chunk
        chunk_size_words: Number of words per chunk
        overlap_words: Number of words to overlap between chunks
    
    Returns:
        List of text chunks (never empty - returns at least one chunk with full text)
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")
    
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        raise ValueError("Text contains no words after splitting")
    
    if chunk_size_words <= 0:
        return [text]
    
    # If text is shorter than chunk size, return as single chunk
    if total_words <= chunk_size_words:
        return [text]
    
    # Ensure overlap is smaller than chunk size to guarantee forward progress
    overlap_words = max(0, min(overlap_words, max(0, chunk_size_words - 1)))

    chunks = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size_words, total_words)
        chunk_text = " ".join(words[start:end])
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= total_words:
            break

        # Advance start with overlap
        start = max(0, end - overlap_words)
        
        # Safety check to prevent infinite loop
        if start >= end:
            break

    # Ensure we return at least one chunk
    if not chunks:
        return [text]
    
    return chunks