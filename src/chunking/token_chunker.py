"""
Token-aware text chunking with sentence boundary detection.

Ported from NewResearcher (CrewAI) with improvements:
- Sentence boundary detection using NLTK
- Token counting with tiktoken (OpenAI's tokenizer)
- Configurable chunk size and overlap
- Preserves context across chunks
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import tiktoken, fallback to approximate counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    _encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5 encoding
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")

# Try to import NLTK for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    # Download punkt if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not available, using simple sentence splitting")


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    index: int
    token_count: int
    start_char: int
    end_char: int
    sentences: List[str]


class TokenChunker:
    """
    Token-aware text chunker with sentence boundary preservation.

    Features:
        - Sentence-aware chunking (won't split mid-sentence)
        - Token counting with tiktoken
        - Configurable overlap between chunks
        - Metadata for each chunk

    Usage:
        chunker = TokenChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_text(long_text)
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        model_name: Optional[str] = None
    ):
        """
        Initialize token chunker.

        Args:
            chunk_size: Target token count per chunk (default: 800)
            chunk_overlap: Token overlap between chunks (default: 100)
            model_name: Optional model name for tokenizer selection
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE and model_name:
            try:
                self.encoder = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoder = _encoder
        elif TIKTOKEN_AVAILABLE:
            self.encoder = _encoder
        else:
            self.encoder = None

        logger.info(f"Initialized TokenChunker (size={chunk_size}, overlap={chunk_overlap})")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Approximate: ~4 characters per token on average
            return len(text) // 4

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple fallback: split on sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text with sentence boundary preservation.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0
        char_position = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_token_count + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    token_count=current_token_count,
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    sentences=list(current_chunk_sentences)
                ))

                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_token_count = overlap_tokens
                chunk_index += 1
                char_position += len(chunk_text) - len(" ".join(overlap_sentences))

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=chunk_index,
                token_count=current_token_count,
                start_char=char_position,
                end_char=char_position + len(chunk_text),
                sentences=list(current_chunk_sentences)
            ))

        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks

    def chunk_with_context(
        self,
        text: str,
        context_prefix: str = "",
        context_suffix: str = ""
    ) -> List[Tuple[str, Chunk]]:
        """
        Chunk text with optional context prefix/suffix for each chunk.

        Returns:
            List of (full_text, chunk_metadata) tuples
        """
        chunks = self.chunk_text(text)
        result = []

        for chunk in chunks:
            full_text = f"{context_prefix}{chunk.text}{context_suffix}".strip()
            result.append((full_text, chunk))

        return result


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Convenience function to chunk text by tokens.

    Args:
        text: Input text
        chunk_size: Target tokens per chunk
        chunk_overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
    chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_text(text)
    return [c.text for c in chunks]


# Integration with existing chunking agent
class TokenChunkingAgent:
    """Agent wrapper for token-aware chunking."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process(self, text: str, paper_base: Optional[str] = None) -> dict:
        """Process text and return chunks with metadata."""
        chunks = self.chunker.chunk_text(text)

        return {
            "chunks": [c.text for c in chunks],
            "chunk_metadata": [
                {
                    "index": c.index,
                    "token_count": c.token_count,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "num_sentences": len(c.sentences),
                }
                for c in chunks
            ],
            "total_chunks": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "avg_chunk_size": sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
        }
