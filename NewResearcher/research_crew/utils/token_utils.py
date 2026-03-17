"""
token_utils.py
--------------
Provides token counting, text truncation, and safety checks to prevent
context-window overflow and excessive LLM costs.

Uses tiktoken (OpenAI's tokenizer) as the counting backend.
Falls back to a simple word-based estimate when tiktoken is unavailable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to load tiktoken; fall back to naive estimation
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-3.5 encoding
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not found – falling back to word-count token estimation.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the number of tokens in *text* for the given *model*.

    When tiktoken is available the count is exact; otherwise we use the
    rule-of-thumb that 1 token ≈ 0.75 words.
    """
    if not text:
        return 0

    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = _ENCODING
        return len(enc.encode(text))

    # Naive fallback: count words and scale
    words = len(text.split())
    return int(words / 0.75)


def truncate_text(text: str, max_tokens: int = 3000, model: str = "gpt-4o") -> str:
    """Truncate *text* so that it does not exceed *max_tokens* tokens.

    The function works at the character level for speed, then verifies the
    token count and trims further if needed.
    """
    if not text:
        return ""

    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    # Estimate ratio and do a first rough cut
    ratio = max_tokens / current_tokens
    cutoff = int(len(text) * ratio * 0.95)  # 5 % safety margin
    text = text[:cutoff]

    # Fine-trim token by token (only needed when estimates are off)
    while count_tokens(text, model) > max_tokens and len(text) > 0:
        text = text[: int(len(text) * 0.95)]

    logger.debug("Text truncated to ~%d tokens.", count_tokens(text, model))
    return text


def is_within_token_limit(text: str, limit: int, model: str = "gpt-4o") -> bool:
    """Return True if *text* fits within *limit* tokens."""
    return count_tokens(text, model) <= limit


def estimate_cost(tokens: int, model: str = "gpt-4o") -> float:
    """Rough USD cost estimate for a given token count.

    Prices as of early 2025; update as needed.
    """
    prices_per_1k = {
        "gpt-4o": 0.005,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.0005,
    }
    rate = prices_per_1k.get(model, 0.005)
    return (tokens / 1000) * rate
