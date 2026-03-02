"""
Agent 1: Ingest Agent
Responsibility: Accept PDF uploads, extract textual and visual content
"""

from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import extract_text_from_pdf
from .base import BaseAgent, AgentMessage, AgentResponse


class IngestAgent(BaseAgent):
    """Agent for PDF processing and content extraction."""
    
    def __init__(self):
        super().__init__("ingest", "Ingest Agent")
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Extract text and images from PDF or use provided text."""
        payload = message.payload
        pdf_path = payload.get("pdf_path")
        text = payload.get("text")  # Allow pre-extracted text
        paper_base = payload.get("paper_base")
        
        # If text is provided directly, use it
        if text:
            return AgentResponse(
                success=True,
                data={
                    "text": text,
                    "images": [],
                    "metadata": {
                        "paper_base": paper_base or "unknown",
                        "source": "pre-extracted_text"
                    },
                    "paper_base": paper_base or "unknown"
                }
            )
        
        if not pdf_path:
            return AgentResponse(
                success=False,
                error="Missing 'pdf_path' or 'text' in payload"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return AgentResponse(
                success=False,
                error=f"PDF file not found: {pdf_path}"
            )
        
        try:
            # Extract text
            text = extract_text_from_pdf(str(pdf_path))
            
            # Extract images (basic implementation - can be enhanced)
            images = self._extract_images(pdf_path)
            
            # Extract metadata
            metadata = {
                "filename": pdf_path.name,
                "paper_base": paper_base or pdf_path.stem,
                "file_size": pdf_path.stat().st_size,
            }
            
            return AgentResponse(
                success=True,
                data={
                    "text": text,
                    "images": images,
                    "metadata": metadata,
                    "paper_base": paper_base or pdf_path.stem
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Failed to process PDF: {str(e)}"
            )
    
    def _extract_images(self, pdf_path: Path) -> list:
        """Extract images from PDF pages."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            images = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                for img_idx, img in enumerate(image_list):
                    images.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "type": "image",  # Can be enhanced to detect equation/table/figure
                        "path": None  # Would need to save images to disk
                    })
            
            doc.close()
            return images
        except Exception as e:
            # If image extraction fails, return empty list
            return []
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Extract textual and visual content from PDF files",
            "operations": ["text_extraction", "image_extraction", "metadata_extraction"],
            "input_format": "PDF",
            "output_format": "JSON with text, images, metadata"
        }

