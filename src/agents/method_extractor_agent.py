"""
Agent 4: Method Extractor Agent
Responsibility: Extract structured method information using LLM
"""

from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from method_extractor import find_method_sections, extract_method_entities
from schemas import MethodStruct
from .base import BaseAgent, AgentMessage, AgentResponse


class MethodExtractorAgent(BaseAgent):
    """Agent for extracting structured method information."""
    
    def __init__(self):
        super().__init__("method_extractor", "Method Extractor Agent")
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """Extract method information from text."""
        payload = message.payload
        text = payload.get("text", "")
        chunks = payload.get("chunks", [])
        
        if not text and not chunks:
            return AgentResponse(
                success=False,
                error="Missing 'text' or 'chunks' in payload"
            )
        
        try:
            # Use full text if available, otherwise concatenate chunks
            if text:
                method_text = text
            else:
                method_text = "\n\n".join(chunks)
            
            # Find method sections
            method_sections = find_method_sections(method_text)
            if method_sections:
                method_text = "\n\n".join(method_sections)
            
            # Extract method entities using LLM if available, otherwise use heuristics
            method_struct = self._extract_with_llm(method_text)
            
            if not method_struct:
                # Fallback to heuristic extraction
                method_struct = extract_method_entities(method_text)
            
            return AgentResponse(
                success=True,
                data={
                    "method_struct": method_struct.model_dump() if isinstance(method_struct, MethodStruct) else method_struct,
                    "method_sections": method_sections
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Method extraction failed: {str(e)}"
            )
    
    def _extract_with_llm(self, method_text: str) -> Any:
        """Extract method information using LLM."""
        try:
            import ollama
            from schemas import MethodStruct
            import json
            
            prompt = f"""Extract structured information from this research method section.

Text:
{method_text[:4000]}

Return a JSON object with:
- algorithm_name: name of the algorithm/method
- equations: list of mathematical equations mentioned
- datasets: list of datasets mentioned
- training: object with optimizer, loss, epochs, learning_rate, batch_size
- inputs: dict describing input specifications
- outputs: dict describing output specifications
- references: list of citation markers like [1], [2], etc.

Return ONLY valid JSON, no markdown or explanation."""

            response = ollama.chat(
                model="gpt-oss:120b-cloud",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.get("message", {}).get("content", "")
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join([l for l in lines if not l.startswith("```")])
            
            data = json.loads(content)
            return MethodStruct(**data)
        except Exception:
            # Fallback to heuristic extraction
            return None
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Extract structured method information from research papers",
            "extraction_method": "LLM-based with heuristic fallback",
            "target_accuracy": "85-92%",
            "output_format": "MethodStruct JSON"
        }

