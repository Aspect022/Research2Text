"""
Agent 3: Chunking Agent
Responsibility: Create processable units with semantic representations
"""

from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import chunk_text_by_words
from .base import BaseAgent, AgentMessage, AgentResponse


class ChunkingAgent(BaseAgent):
    """Agent for creating semantic chunks with embeddings."""
    
    def __init__(self):
        super().__init__("chunking", "Chunking Agent")
        self._embedding_fn = None
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Create chunks and generate embeddings."""
        payload = message.payload
        text = payload.get("text")
        paper_base = payload.get("paper_base", "unknown")
        
        if not text:
            return AgentResponse(
                success=False,
                error="Missing 'text' in payload"
            )
        
        try:
            # Create chunks
            chunks = chunk_text_by_words(text, chunk_size_words=750, overlap_words=100)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            return AgentResponse(
                success=True,
                data={
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "chunk_count": len(chunks),
                    "paper_base": paper_base
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Chunking failed: {str(e)}"
            )
    
    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks using Sentence Transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._embedding_fn is None:
                self._embedding_fn = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            embeddings = self._embedding_fn.encode(chunks, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            # Return empty embeddings if model not available
            return [[] for _ in chunks]
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Create semantic chunks with embeddings",
            "chunk_size": 750,
            "overlap": 100,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384
        }

