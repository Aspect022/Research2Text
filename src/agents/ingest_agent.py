"""
Agent 1: Ingest Agent (Upgraded - Phase 2)
Responsibility: Accept PDF uploads, extract textual and visual content.

Extraction Chain (best-to-fallback):
  1. MinerU  — Preserves structure, LaTeX equations, tables
  2. olmOCR  — VLM-based fallback for visually complex / scanned pages
  3. PyMuPDF — Lightweight fallback if neither MinerU nor olmOCR is installed
"""

import json
import logging
import subprocess
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLMOCR_TIMEOUT
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)


class IngestAgent(BaseAgent):
    """Agent for PDF processing and content extraction with multi-backend support."""

    def __init__(self):
        super().__init__("ingest", "Ingest Agent")
        self._backend = self._detect_backend()
        logger.info(f"[IngestAgent] Using extraction backend: {self._backend}")

    def _detect_backend(self) -> str:
        """Detect the best available extraction backend."""
        try:
            import mineru  # noqa: F401
            return "mineru"
        except ImportError:
            pass

        try:
            import olmocr  # noqa: F401
            return "olmocr"
        except ImportError:
            pass

        try:
            import fitz  # noqa: F401
            return "pymupdf"
        except ImportError:
            pass

        return "none"

    def process(self, message: AgentMessage) -> AgentResponse:
        """Extract text and images from PDF or use provided text."""
        payload = message.payload
        pdf_path = payload.get("pdf_path")
        text = payload.get("text")
        paper_base = payload.get("paper_base")

        # If text is provided directly, use it
        if text:
            return AgentResponse(
                success=True,
                data={
                    "text": text,
                    "images": [],
                    "tables": [],
                    "equations": [],
                    "metadata": {
                        "paper_base": paper_base or "unknown",
                        "source": "pre-extracted_text",
                        "backend": "direct",
                    },
                    "paper_base": paper_base or "unknown",
                },
            )

        if not pdf_path:
            return AgentResponse(
                success=False,
                error="Missing 'pdf_path' or 'text' in payload",
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return AgentResponse(
                success=False,
                error=f"PDF file not found: {pdf_path}",
            )

        # Run the extraction chain
        try:
            result = self._extract(pdf_path, paper_base)
            return AgentResponse(success=True, data=result)
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Failed to process PDF: {str(e)}",
            )

    # ─── Extraction Chain ─────────────────────────────

    def _extract(self, pdf_path: Path, paper_base: Optional[str]) -> Dict[str, Any]:
        """Try the extraction chain: MinerU → olmOCR → PyMuPDF."""
        paper_base = paper_base or pdf_path.stem

        if self._backend == "mineru":
            try:
                return self._extract_with_mineru(pdf_path, paper_base)
            except Exception as e:
                logger.warning(f"[IngestAgent] MinerU extraction failed: {e}. Falling back...")

        if self._backend in ("mineru", "olmocr"):
            try:
                return self._extract_with_olmocr(pdf_path, paper_base)
            except Exception as e:
                logger.warning(f"[IngestAgent] olmOCR extraction failed: {e}. Falling back...")

        return self._extract_with_pymupdf(pdf_path, paper_base)

    def _extract_with_mineru(self, pdf_path: Path, paper_base: str) -> Dict[str, Any]:
        """
        Extract using MinerU — best quality for structured documents.
        Preserves LaTeX equations, tables, and reading order.
        """
        logger.info(f"[IngestAgent] Extracting with MinerU: {pdf_path.name}")

        from mineru import pdf_to_markdown, pdf_to_json

        # Get structured JSON for programmatic access
        json_result = pdf_to_json(str(pdf_path))

        # Get markdown for human-readable text
        md_result = pdf_to_markdown(str(pdf_path))
        text = md_result if isinstance(md_result, str) else str(md_result)

        # Parse structured elements from the JSON result
        images = []
        tables = []
        equations = []

        if isinstance(json_result, (list, dict)):
            self._parse_mineru_json(json_result, images, tables, equations)

        return {
            "text": text,
            "images": images,
            "tables": tables,
            "equations": equations,
            "metadata": {
                "paper_base": paper_base,
                "backend": "mineru",
                "table_count": len(tables),
                "equation_count": len(equations),
                "image_count": len(images),
            },
            "paper_base": paper_base,
        }

    def _parse_mineru_json(
        self,
        data: Any,
        images: List[Dict],
        tables: List[Dict],
        equations: List[str],
    ) -> None:
        """Recursively parse MinerU JSON output for structured elements."""
        if isinstance(data, dict):
            element_type = data.get("type", "")

            if element_type == "image" or "image" in element_type.lower():
                images.append({
                    "page": data.get("page", 0),
                    "type": "image",
                    "path": data.get("path"),
                    "caption": data.get("caption"),
                })
            elif element_type == "table" or "table" in element_type.lower():
                tables.append({
                    "page": data.get("page", 0),
                    "content": data.get("content", data.get("text", "")),
                    "html": data.get("html"),
                })
            elif element_type == "equation" or "equation" in element_type.lower():
                eq_text = data.get("content", data.get("text", ""))
                if eq_text:
                    equations.append(eq_text)

            # Recurse into nested structures
            for value in data.values():
                if isinstance(value, (dict, list)):
                    self._parse_mineru_json(value, images, tables, equations)

        elif isinstance(data, list):
            for item in data:
                self._parse_mineru_json(item, images, tables, equations)

    def _extract_with_olmocr(self, pdf_path: Path, paper_base: str) -> Dict[str, Any]:
        """
        Extract using olmOCR — VLM-based fallback for complex/scanned pages.
        Uses CLI pipeline and reads the markdown output.
        """
        logger.info(f"[IngestAgent] Extracting with olmOCR: {pdf_path.name}")

        with tempfile.TemporaryDirectory(prefix="olmocr_") as workspace:
            workspace_path = Path(workspace)

            # Run olmOCR pipeline
            cmd = [
                "python", "-m", "olmocr.pipeline",
                str(workspace_path),
                "--markdown",
                "--pdfs", str(pdf_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=OLMOCR_TIMEOUT,
            )

            if result.returncode != 0:
                raise RuntimeError(f"olmOCR failed: {result.stderr[:500]}")

            # Find generated markdown files
            md_files = list(workspace_path.rglob("*.md"))
            if not md_files:
                raise RuntimeError("olmOCR produced no markdown output")

            text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in md_files)

        return {
            "text": text,
            "images": [],
            "tables": [],
            "equations": [],
            "metadata": {
                "paper_base": paper_base,
                "backend": "olmocr",
            },
            "paper_base": paper_base,
        }

    def _extract_with_pymupdf(self, pdf_path: Path, paper_base: str) -> Dict[str, Any]:
        """
        Extract using PyMuPDF — lightweight fallback.
        Does not preserve LaTeX or complex table structures.
        """
        logger.info(f"[IngestAgent] Extracting with PyMuPDF: {pdf_path.name}")
        import re

        try:
            import fitz
        except ImportError:
            raise RuntimeError("No extraction backend available. Install mineru, olmocr, or PyMuPDF.")

        doc = fitz.open(str(pdf_path))
        try:
            pages = []
            images = []

            for page_num, page in enumerate(doc):
                pages.append(page.get_text("text"))

                # Collect image metadata
                for img_idx, img in enumerate(page.get_images()):
                    images.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "type": "image",
                        "path": None,
                    })

            text = "\n\n".join(pages)
            text = re.sub(r"\n{3,}", "\n\n", text)
        finally:
            doc.close()

        return {
            "text": text,
            "images": images,
            "tables": [],
            "equations": [],
            "metadata": {
                "paper_base": paper_base,
                "backend": "pymupdf",
                "file_size": pdf_path.stat().st_size,
            },
            "paper_base": paper_base,
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Extract text, equations, tables, and images from PDF files",
            "backend": self._backend,
            "extraction_chain": ["mineru", "olmocr", "pymupdf"],
            "operations": [
                "text_extraction",
                "image_extraction",
                "table_extraction",
                "equation_extraction",
                "metadata_extraction",
            ],
            "output_format": "JSON with text, images, tables, equations, metadata",
        }
