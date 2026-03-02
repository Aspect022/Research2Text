"""
Agent 2: Vision Agent (Upgraded - Phase 2)
Responsibility: Extract information from figures, tables, diagrams.

Backends (auto-detected):
  1. MinerU  — Already extracts tables/figures during ingestion; this agent
               performs post-processing (caption extraction, classification).
  2. Tesseract OCR — Fallback for raw image files.
  3. BLIP / Captioning models — Optional future hook.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """Agent for processing visual content (figures, tables, equations in images)."""

    def __init__(self):
        super().__init__("vision", "Vision Agent")
        self._ocr_available = self._check_ocr()

    def _check_ocr(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process images to extract text, captions, and table data.

        Accepts two payload formats:
          1. Single image: {"image_path": "...", "image_type": "..."}
          2. Batch from ingestion: {"images": [...], "tables": [...]}
        """
        payload = message.payload

        # ── Batch mode (from orchestrator pipeline) ──
        images = payload.get("images", [])
        tables = payload.get("tables", [])

        if images or tables:
            return self._process_batch(images, tables)

        # ── Single image mode ──
        image_path = payload.get("image_path")
        image_type = payload.get("image_type", "unknown")

        if not image_path:
            return AgentResponse(
                success=True,
                data={
                    "processed_images": [],
                    "processed_tables": [],
                    "message": "No images or tables to process",
                },
            )

        path = Path(image_path)
        if not path.exists():
            return AgentResponse(
                success=False,
                error=f"Image file not found: {path}",
            )

        try:
            result = self._process_single_image(path, image_type)
            return AgentResponse(success=True, data=result)
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Vision processing failed: {str(e)}",
            )

    def _process_batch(
        self,
        images: List[Dict],
        tables: List[Dict],
    ) -> AgentResponse:
        """Process a batch of images and tables from the ingestion stage."""
        processed_images = []
        processed_tables = []

        for img_info in images:
            img_path = img_info.get("path")
            if img_path and Path(img_path).exists():
                result = self._process_single_image(
                    Path(img_path),
                    img_info.get("type", "figure"),
                )
                result["caption"] = img_info.get("caption") or result.get("caption")
                processed_images.append(result)
            else:
                # Image metadata without file — store caption if available
                processed_images.append({
                    "page": img_info.get("page"),
                    "type": img_info.get("type", "image"),
                    "caption": img_info.get("caption"),
                    "ocr_text": None,
                })

        for table_info in tables:
            processed_tables.append({
                "page": table_info.get("page"),
                "content": table_info.get("content", ""),
                "html": table_info.get("html"),
                "type": "table",
            })

        return AgentResponse(
            success=True,
            data={
                "processed_images": processed_images,
                "processed_tables": processed_tables,
                "image_count": len(processed_images),
                "table_count": len(processed_tables),
            },
        )

    def _process_single_image(self, image_path: Path, image_type: str) -> Dict[str, Any]:
        """Process a single image file."""
        result: Dict[str, Any] = {
            "image_path": str(image_path),
            "image_type": image_type,
            "ocr_text": None,
            "caption": None,
            "table_data": None,
        }

        if self._ocr_available:
            result["ocr_text"] = self._extract_ocr(image_path)

        if image_type == "table":
            result["table_data"] = self._extract_table(image_path)

        return result

    def _extract_ocr(self, image_path: Path) -> Optional[str]:
        """Extract text using Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(str(image_path))
            text = pytesseract.image_to_string(image)
            return text.strip() if text else None
        except Exception:
            return None

    def _extract_table(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Extract table data using Camelot or similar."""
        try:
            import camelot

            tables = camelot.read_pdf(str(image_path), pages="1")
            if tables:
                return {
                    "data": tables[0].df.to_dict(),
                    "accuracy": tables[0].parsing_report.get("accuracy", 0),
                }
            return None
        except Exception:
            return None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Extract information from figures, tables, and diagrams",
            "modes": ["single_image", "batch_from_ingestion"],
            "tools": {
                "ocr": "Tesseract OCR" if self._ocr_available else "Not available",
                "table": "Camelot (when available)",
            },
        }
