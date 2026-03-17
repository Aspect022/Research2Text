"""Unit tests for token chunker."""

import pytest
from chunking.token_chunker import TokenChunker, chunk_text_by_tokens


class TestTokenChunker:
    """Test cases for TokenChunker."""

    def test_initialization(self):
        """Test chunker initialization."""
        chunker = TokenChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_count_tokens_approximate(self):
        """Test token counting (approximate)."""
        chunker = TokenChunker()
        text = "This is a test sentence."
        count = chunker.count_tokens(text)
        # Approximate: ~4 chars per token
        assert count > 0
        assert count < len(text)

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        chunker = TokenChunker()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) >= 3

    def test_chunk_text_basic(self):
        """Test basic chunking."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"Sentence number {i}." for i in range(20)])
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all(len(c.text) > 0 for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = TokenChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_text_single_sentence(self):
        """Test chunking single sentence."""
        chunker = TokenChunker(chunk_size=1000)
        text = "This is a single sentence."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert len(chunk.sentences) > 0

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TokenChunker(chunk_size=50, chunk_overlap=20)
        text = " ".join([f"Sentence {i}." for i in range(10)])
        chunks = chunker.chunk_text(text)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                current_text = chunks[i].text
                next_text = chunks[i + 1].text
                # There should be some overlap
                assert len(set(current_text.split()) & set(next_text.split())) > 0

    def test_chunk_with_context(self):
        """Test chunking with context prefix/suffix."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        text = "Sentence one. Sentence two. Sentence three."
        prefix = "[START] "
        suffix = " [END]"

        results = chunker.chunk_with_context(text, prefix, suffix)

        for full_text, chunk in results:
            assert full_text.startswith(prefix)
            assert full_text.endswith(suffix)
            assert chunk.text in full_text


class TestChunkTextByTokens:
    """Test cases for chunk_text_by_tokens convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        text = " ".join([f"Sentence {i}." for i in range(10)])
        chunks = chunk_text_by_tokens(text, chunk_size=50, chunk_overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text(self):
        """Test with empty text."""
        chunks = chunk_text_by_tokens("")
        assert chunks == []


class TestTokenChunkingAgent:
    """Test cases for TokenChunkingAgent."""

    def test_process(self):
        """Test agent processing."""
        from chunking.token_chunker import TokenChunkingAgent

        agent = TokenChunkingAgent(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"Sentence {i}." for i in range(10)])
        result = agent.process(text, paper_base="test_paper")

        assert "chunks" in result
        assert "chunk_metadata" in result
        assert "total_chunks" in result
        assert result["total_chunks"] > 0

    def test_process_empty(self):
        """Test agent with empty text."""
        from chunking.token_chunker import TokenChunkingAgent

        agent = TokenChunkingAgent()
        result = agent.process("")

        assert result["total_chunks"] == 0
        assert result["chunks"] == []
