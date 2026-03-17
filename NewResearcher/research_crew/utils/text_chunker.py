"""
text_chunker.py
---------------
Splits large documents into overlapping token-bounded chunks so that
individual chunks can be summarised before being sent to an LLM.

Design goals
------------
* Chunks never exceed MAX_CHUNK_TOKENS (default 800).
* Consecutive chunks share OVERLAP_TOKENS (default 100) for context continuity.
* Splitting happens at sentence/paragraph boundaries when possible.
"""

import logging
import re
from typing import List

from research_crew.utils.token_utils import count_tokens, truncate_text

logger = logging.getLogger(__name__)

# Defaults (can be overridden per call)
MAX_CHUNK_TOKENS: int = 800
OVERLAP_TOKENS: int = 100
HARD_DOC_LIMIT: int = 3000   # chars, applied before chunking


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex heuristic."""
    # Split on '.', '!', '?' followed by whitespace or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_text(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    hard_limit_chars: int = HARD_DOC_LIMIT,
) -> List[str]:
    """Chunk *text* into token-bounded segments with optional overlap.

    Parameters
    ----------
    text:
        The raw document text.
    max_tokens:
        Maximum tokens allowed per chunk.
    overlap_tokens:
        Number of tokens from the previous chunk to prepend to the next,
        ensuring context continuity.
    hard_limit_chars:
        Character hard-limit applied to the whole document *before* chunking.
        Prevents feeding enormous PDFs into the pipeline at all.

    Returns
    -------
    List[str]
        Ordered list of text chunks, each within *max_tokens*.
    """
    if not text:
        return []

    # Hard character limit first
    if len(text) > hard_limit_chars:
        text = text[:hard_limit_chars]
        logger.debug("Document truncated to %d chars before chunking.", hard_limit_chars)

    sentences = _split_into_sentences(text)
    if not sentences:
        return [truncate_text(text, max_tokens)]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens: int = 0
    overlap_buffer: str = ""

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Single sentence exceeds limit — truncate it
        if sentence_tokens > max_tokens:
            sentence = truncate_text(sentence, max_tokens - overlap_tokens)
            sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens:
            # Flush current chunk
            chunk_text_value = overlap_buffer + " ".join(current_sentences)
            chunks.append(chunk_text_value.strip())
            logger.debug("Chunk %d: %d tokens.", len(chunks), count_tokens(chunk_text_value))

            # Build overlap buffer from the tail of the current chunk
            overlap_sentences: List[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                tok = count_tokens(s)
                if overlap_count + tok <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += tok
                else:
                    break
            overlap_buffer = " ".join(overlap_sentences) + " " if overlap_sentences else ""

            current_sentences = [sentence]
            current_tokens = count_tokens(overlap_buffer) + sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Flush the last partial chunk
    if current_sentences:
        chunk_text_value = overlap_buffer + " ".join(current_sentences)
        chunks.append(chunk_text_value.strip())

    logger.info("Document split into %d chunks.", len(chunks))
    return chunks


def summarise_chunks_placeholder(chunks: List[str]) -> str:
    """Join chunk summaries into a single condensed representation.

    In the full pipeline the Evidence Extraction Agent calls the LLM to
    summarise each chunk individually; this helper merges those summaries.
    Here we simply join them — real summarisation happens inside the agent.
    """
    return "\n\n---\n\n".join(chunks)
