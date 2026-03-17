"""
Gemini Embedding integration for Research2Text.

Provides high-quality embeddings with 8K context window.
Falls back to local MiniLM if Gemini not available.
"""

import os
import logging
from typing import List, Union, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed. Gemini embeddings unavailable.")

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. MiniLM embeddings unavailable.")


class GeminiEmbedder:
    """
    Google Gemini embedding model wrapper.

    Supports:
    - Text embeddings (gemini-embedding-001)
    - Multimodal embeddings (gemini-embedding-2-preview)
    - Configurable output dimensions (768, 1536, 3072)
    - Task-specific optimization (RETRIEVAL_DOCUMENT, etc.)

    Context window: 8,192 tokens (vs MiniLM's 512)
    MTEB Score: ~68 (vs MiniLM's ~62)
    """

    MODELS = {
        "text": "gemini-embedding-001",
        "multimodal": "gemini-embedding-2-preview"
    }

    # Recommended dimensions from Gemini docs
    DIMENSIONS = {
        "small": 768,
        "medium": 1536,
        "large": 3072
    }

    def __init__(
        self,
        model_type: str = "text",
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini embedder.

        Args:
            model_type: "text" or "multimodal"
            output_dimensionality: 768, 1536, or 3072
            task_type: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.
            api_key: Gemini API key (or from env)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google GenAI package not installed. "
                "Install with: pip install google-genai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = self.MODELS.get(model_type, "gemini-embedding-001")
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type

        logger.info(
            f"Initialized GeminiEmbedder: model={self.model}, "
            f"dimensions={output_dimensionality}, task={task_type}"
        )

    def embed(
        self,
        contents: Union[str, List[str]],
        is_document: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for text.

        Args:
            contents: Text string or list of strings
            is_document: True for documents, False for queries

        Returns:
            List of embedding vectors (each is list of floats)
        """
        if isinstance(contents, str):
            contents = [contents]

        # Determine task type
        task = self.task_type
        if not is_document and self.task_type == "RETRIEVAL_DOCUMENT":
            task = "RETRIEVAL_QUERY"

        # Batch processing (max 100 per request)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]

            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task,
                        output_dimensionality=self.output_dimensionality
                    )
                )

                for embedding in result.embeddings:
                    all_embeddings.append(embedding.values)

            except Exception as e:
                logger.error(f"Gemini embedding error: {e}")
                raise

        return all_embeddings

    def embed_pdf(self, pdf_path: Union[str, Path]) -> List[float]:
        """
        Embed a PDF document directly (multimodal model only).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Embedding vector
        """
        if self.model != "gemini-embedding-2-preview":
            raise ValueError("PDF embedding requires multimodal model (gemini-embedding-2-preview)")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf'
                )
            ]
        )

        return result.embeddings[0].values

    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """
        Embed an image (multimodal model only).

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector
        """
        if self.model != "gemini-embedding-2-preview":
            raise ValueError("Image embedding requires multimodal model")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type = f"image/{image_path.suffix.lstrip('.')}"
        if mime_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            ]
        )

        return result.embeddings[0].values

    @property
    def dimensionality(self) -> int:
        """Return embedding dimension."""
        return self.output_dimensionality


class MiniLMEmbedder:
    """
    Local MiniLM embedder (fallback when Gemini unavailable).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize MiniLM embedder.

        Args:
            model_name: Sentence transformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self._dimensionality = self.model.get_sentence_embedding_dimension()

        logger.info(f"Initialized MiniLMEmbedder: model={model_name}, dim={self._dimensionality}")

    def embed(self, contents: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """
        Generate embeddings using MiniLM.

        Args:
            contents: Text or list of texts

        Returns:
            List of embedding vectors
        """
        if isinstance(contents, str):
            contents = [contents]

        embeddings = self.model.encode(contents, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimensionality(self) -> int:
        """Return embedding dimension."""
        return self._dimensionality


class HybridEmbedder:
    """
    Hybrid embedder that uses Gemini when available, falls back to MiniLM.

    This is the recommended embedder for Research2Text - it provides
    high-quality embeddings when API is available, but works offline
    with local MiniLM as fallback.
    """

    def __init__(
        self,
        gemini_dimensionality: int = 768,
        prefer_gemini: bool = True
    ):
        """
        Initialize hybrid embedder.

        Args:
            gemini_dimensionality: Dimensions for Gemini (768, 1536, or 3072)
            prefer_gemini: Whether to prefer Gemini over MiniLM
        """
        self.gemini = None
        self.minilm = None
        self._gemini_dimensionality = gemini_dimensionality
        self._minilm_dimensionality = 384  # MiniLM-L6-v2

        # Try Gemini first if preferred
        if prefer_gemini:
            self._try_gemini()

        # If Gemini not available, use MiniLM
        if self.gemini is None:
            self._try_minilm()

        if self.gemini is None and self.minilm is None:
            raise RuntimeError(
                "No embedder available. Install either:\n"
                "  - google-genai for Gemini (pip install google-genai)\n"
                "  - sentence-transformers for MiniLM (pip install sentence-transformers)"
            )

    def _try_gemini(self):
        """Try to initialize Gemini embedder."""
        if not GEMINI_AVAILABLE:
            logger.info("Gemini package not available")
            return

        if not os.getenv("GEMINI_API_KEY"):
            logger.info("GEMINI_API_KEY not set, skipping Gemini")
            return

        try:
            self.gemini = GeminiEmbedder(
                model_type="text",
                output_dimensionality=self._gemini_dimensionality
            )
            logger.info(f"Using Gemini embeddings ({self._gemini_dimensionality}d)")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            self.gemini = None

    def _try_minilm(self):
        """Try to initialize MiniLM embedder."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("sentence-transformers not available")
            return

        try:
            self.minilm = MiniLMEmbedder()
            logger.info(f"Using MiniLM embeddings ({self._minilm_dimensionality}d)")
        except Exception as e:
            logger.warning(f"Failed to initialize MiniLM: {e}")
            self.minilm = None

    def embed(
        self,
        contents: Union[str, List[str]],
        is_document: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings using best available embedder.

        Args:
            contents: Text or list of texts
            is_document: Whether content is document (for task type)

        Returns:
            List of embedding vectors
        """
        if self.gemini:
            return self.gemini.embed(contents, is_document)
        elif self.minilm:
            return self.minilm.embed(contents)
        else:
            raise RuntimeError("No embedder available")

    @property
    def dimensionality(self) -> int:
        """Return current embedding dimension."""
        if self.gemini:
            return self.gemini.dimensionality
        elif self.minilm:
            return self.minilm.dimensionality
        return 0

    @property
    def using_gemini(self) -> bool:
        """Whether currently using Gemini."""
        return self.gemini is not None

    def __repr__(self) -> str:
        if self.gemini:
            return f"HybridEmbedder(Gemini, dim={self.dimensionality})"
        elif self.minilm:
            return f"HybridEmbedder(MiniLM, dim={self.dimensionality})"
        return "HybridEmbedder(None)"


# Convenience function
def get_embedder(prefer_gemini: bool = True) -> HybridEmbedder:
    """
    Get the default hybrid embedder.

    Args:
        prefer_gemini: Whether to prefer Gemini over MiniLM

    Returns:
        HybridEmbedder instance
    """
    return HybridEmbedder(prefer_gemini=prefer_gemini)
