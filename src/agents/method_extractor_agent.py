"""
Agent 4: Method Extractor Agent (Upgraded - Phase 3)
Responsibility: Extract structured method information with confidence scoring.

Implements a simplified Conformal Prediction framework:
  - Each extracted field gets a confidence score (0.0–1.0)
  - Fields explicitly stated in the paper get high scores
  - Fields inferred by the LLM (not directly quoted) get lower scores
  - Low-confidence fields are flagged for downstream agents
  
LLM Strategy:
  - Primary: Configurable model (DeepSeek-V3 / Qwen2.5-Coder / any Ollama model)
  - Fallback: Heuristic regex-based extraction
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_CHAIN, EXTRACTION_TEMPERATURE, MAX_TEXT_LENGTH_FOR_LLM, LOW_CONFIDENCE_THRESHOLD
from method_extractor import find_method_sections, extract_method_entities
from schemas import ConfidenceScore, MethodStruct
from .base import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)

# Models imported from config.py
DEFAULT_MODEL_CHAIN = MODEL_CHAIN


class MethodExtractorAgent(BaseAgent):
    """Agent for extracting structured method information with confidence scoring."""

    def __init__(self, model: Optional[str] = None):
        super().__init__("method_extractor", "Method Extractor Agent")
        self._model = model  # None = auto-detect from chain
        self._available_model: Optional[str] = None

    def process(self, message: AgentMessage) -> AgentResponse:
        """Extract method information with per-field confidence scores."""
        payload = message.payload
        text = payload.get("text", "")
        chunks = payload.get("chunks", [])

        if not text and not chunks:
            return AgentResponse(
                success=False,
                error="Missing 'text' or 'chunks' in payload",
            )

        try:
            method_text = text if text else "\n\n".join(chunks)

            # Find method-specific sections
            method_sections = find_method_sections(method_text)
            focused_text = "\n\n".join(method_sections) if method_sections else method_text

            # Try LLM extraction with confidence scoring
            method_struct = self._extract_with_confidence(focused_text, method_text)

            if method_struct is None:
                # Fallback to heuristic extraction
                logger.info("[MethodExtractor] LLM unavailable, using heuristic extraction")
                method_struct = self._heuristic_with_confidence(method_text)

            return AgentResponse(
                success=True,
                data={
                    "method_struct": method_struct.model_dump(),
                    "method_sections": method_sections,
                    "overall_confidence": method_struct.overall_confidence(),
                    "low_confidence_fields": method_struct.low_confidence_fields(),
                },
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Method extraction failed: {str(e)}",
            )

    # ─── LLM Extraction with Confidence ──────────────

    def _extract_with_confidence(self, focused_text: str, full_text: str) -> Optional[MethodStruct]:
        """Extract method information using LLM with confidence scoring."""
        try:
            import ollama
        except ImportError:
            return None

        model = self._resolve_model(ollama)
        if not model:
            return None

        logger.info(f"[MethodExtractor] Using model: {model}")

        # Step 1: Extract structured data
        raw_data = self._llm_extract(ollama, model, focused_text)
        if not raw_data:
            return None

        # Step 2: Compute confidence scores by cross-referencing with source text
        confidence = self._compute_confidence(raw_data, full_text)

        # Step 3: Build MethodStruct with confidence
        try:
            method_struct = MethodStruct(**raw_data, confidence=confidence)
            return method_struct
        except Exception as e:
            logger.warning(f"[MethodExtractor] Failed to build MethodStruct: {e}")
            return None

    def _resolve_model(self, ollama_module: Any) -> Optional[str]:
        """Find the first available model from the preference chain."""
        if self._available_model:
            return self._available_model

        if self._model:
            self._available_model = self._model
            return self._model

        try:
            available = ollama_module.list()
            available_names = set()
            if isinstance(available, dict):
                for m in available.get("models", []):
                    available_names.add(m.get("name", ""))
                    # Also match without tag
                    name = m.get("name", "")
                    if ":" in name:
                        available_names.add(name.split(":")[0])

            for preferred in DEFAULT_MODEL_CHAIN:
                if preferred in available_names or preferred.split(":")[0] in available_names:
                    self._available_model = preferred
                    logger.info(f"[MethodExtractor] Auto-selected model: {preferred}")
                    return preferred

            # If none from chain found, use first available
            if available_names:
                first = list(available_names)[0]
                self._available_model = first
                logger.info(f"[MethodExtractor] Using first available model: {first}")
                return first

        except Exception as e:
            logger.warning(f"[MethodExtractor] Failed to list models: {e}")

        return None

    def _llm_extract(self, ollama_module: Any, model: str, text: str) -> Optional[Dict[str, Any]]:
        """Run the LLM extraction with an optimized prompt."""
        prompt = f"""You are a precise scientific paper analyzer. Extract ONLY information that is EXPLICITLY stated in the text below. Do NOT guess or infer values that are not mentioned.

TEXT:
{text[:MAX_TEXT_LENGTH_FOR_LLM]}

Extract a JSON object with these fields. For each field, ONLY include values you can directly quote from the text. Leave fields as null/empty if not explicitly stated:

{{
  "algorithm_name": "Name of the primary algorithm/method (string or null)",
  "equations": ["List of mathematical equations mentioned, in LaTeX or text form"],
  "datasets": ["List of dataset names explicitly mentioned"],
  "training": {{
    "optimizer": "Optimizer name or null",
    "loss": "Loss function name or null",
    "epochs": null,
    "learning_rate": null,
    "batch_size": null
  }},
  "inputs": {{"description": "Input specification or empty dict"}},
  "outputs": {{"description": "Output specification or empty dict"}},
  "references": ["Citation markers like [1], [2]"]
}}

CRITICAL RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no code fences.
2. If a value is NOT explicitly stated in the text, use null (not a guess).
3. For numerical values (epochs, learning_rate, batch_size), only include if a specific number is given.
4. For equations, include the exact mathematical expression as written."""

        try:
            response = ollama_module.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": EXTRACTION_TEMPERATURE},
            )

            content = response.get("message", {}).get("content", "")
            return self._parse_json_response(content)
        except Exception as e:
            logger.warning(f"[MethodExtractor] LLM call failed: {e}")
            return None

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Robustly parse JSON from LLM response."""
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(l for l in lines if not l.strip().startswith("```"))
            content = content.strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                return json.loads(content[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    # ─── Conformal Prediction: Confidence Scoring ─────

    def _compute_confidence(self, data: Dict[str, Any], source_text: str) -> Dict[str, ConfidenceScore]:
        """
        Compute per-field confidence using source text cross-referencing.

        Scoring rules:
          1.0  — Value is directly quoted from the text (exact match)
          0.85 — Value substring found in text (partial match)
          0.6  — Field type is mentioned but exact value unclear
          0.3  — Value appears to be inferred/hallucinated (no match in text)
          0.0  — Field is null/empty (no extraction attempted)
        """
        confidence: Dict[str, ConfidenceScore] = {}
        text_lower = source_text.lower()

        # Score algorithm_name
        algo = data.get("algorithm_name")
        if algo:
            score, evidence = self._score_string_field(algo, source_text)
            confidence["algorithm_name"] = ConfidenceScore(
                score=score,
                source="explicit" if score >= 0.85 else "inferred",
                evidence=evidence,
            )

        # Score datasets
        datasets = data.get("datasets", [])
        if datasets:
            dataset_scores = []
            for ds in datasets:
                score, evidence = self._score_string_field(ds, source_text)
                dataset_scores.append(score)
            avg = sum(dataset_scores) / len(dataset_scores) if dataset_scores else 0.0
            confidence["datasets"] = ConfidenceScore(
                score=avg,
                source="explicit" if avg >= 0.85 else "inferred",
                evidence=f"Found {len([s for s in dataset_scores if s >= 0.85])}/{len(datasets)} datasets in text",
            )

        # Score training parameters
        training = data.get("training", {})
        if isinstance(training, dict):
            for param in ["optimizer", "loss", "epochs", "learning_rate", "batch_size"]:
                value = training.get(param)
                if value is not None:
                    score, evidence = self._score_training_param(param, value, source_text)
                    confidence[f"training.{param}"] = ConfidenceScore(
                        score=score,
                        source="explicit" if score >= 0.85 else "inferred",
                        evidence=evidence,
                    )

        # Score equations
        equations = data.get("equations", [])
        if equations:
            eq_scores = []
            for eq in equations:
                score, _ = self._score_string_field(eq, source_text)
                eq_scores.append(score)
            avg = sum(eq_scores) / len(eq_scores) if eq_scores else 0.0
            confidence["equations"] = ConfidenceScore(
                score=avg,
                source="explicit" if avg >= 0.85 else "inferred",
                evidence=f"Found {len([s for s in eq_scores if s >= 0.85])}/{len(equations)} equations in text",
            )

        return confidence

    def _score_string_field(self, value: str, source_text: str) -> Tuple[float, Optional[str]]:
        """Score a string field by checking if it appears in the source text."""
        if not value:
            return 0.0, None

        value_lower = value.lower().strip()
        text_lower = source_text.lower()

        # Exact substring match
        if value_lower in text_lower:
            idx = text_lower.index(value_lower)
            start = max(0, idx - 30)
            end = min(len(source_text), idx + len(value) + 30)
            evidence = source_text[start:end].strip()
            return 1.0, f"...{evidence}..."

        # Fuzzy: check if all significant words appear
        words = [w for w in value_lower.split() if len(w) > 2]
        if words:
            found = sum(1 for w in words if w in text_lower)
            ratio = found / len(words)
            if ratio >= 0.8:
                return 0.85, f"Most words found ({found}/{len(words)})"
            elif ratio >= 0.5:
                return 0.6, f"Some words found ({found}/{len(words)})"

        return 0.3, "Not found in source text — possibly inferred"

    def _score_training_param(self, param: str, value: Any, source_text: str) -> Tuple[float, Optional[str]]:
        """Score a training parameter by searching for its value in text."""
        if value is None:
            return 0.0, None

        text_lower = source_text.lower()
        value_str = str(value)

        # Look for the number directly
        if value_str in source_text:
            return 1.0, f"Value '{value_str}' found directly in text"

        # Look for common patterns
        patterns = {
            "learning_rate": [rf"learning[\s_-]*rate[\s:=of]*{re.escape(value_str)}", rf"lr[\s:=]*{re.escape(value_str)}"],
            "batch_size": [rf"batch[\s_-]*size[\s:=of]*{re.escape(value_str)}"],
            "epochs": [rf"{re.escape(value_str)}[\s]*epoch", rf"epoch[s]?[\s:=of]*{re.escape(value_str)}"],
            "optimizer": [rf"{re.escape(value_str.lower())}"],
            "loss": [rf"{re.escape(value_str.lower())}"],
        }

        for pattern in patterns.get(param, []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 1.0, f"Pattern match for {param}={value_str}"

        # Numeric value found anywhere
        if re.search(re.escape(value_str), source_text):
            return 0.7, f"Value '{value_str}' found but context unclear"

        return 0.3, f"Value '{value_str}' not found in text"

    # ─── Heuristic Fallback with Confidence ───────────

    def _heuristic_with_confidence(self, text: str) -> MethodStruct:
        """Heuristic extraction with confidence scores."""
        base = extract_method_entities(text)

        confidence: Dict[str, ConfidenceScore] = {}

        if base.algorithm_name:
            score, evidence = self._score_string_field(base.algorithm_name, text)
            confidence["algorithm_name"] = ConfidenceScore(
                score=score, source="heuristic", evidence=evidence,
            )

        if base.datasets:
            confidence["datasets"] = ConfidenceScore(
                score=1.0, source="heuristic", evidence="Found via regex pattern matching",
            )

        if base.equations:
            confidence["equations"] = ConfidenceScore(
                score=0.7, source="heuristic", evidence="Found via regex (may be incomplete)",
            )

        base.confidence = confidence
        return base

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Extract structured method information with conformal prediction confidence scoring",
            "extraction_method": "LLM-based (multi-model chain) with heuristic fallback",
            "confidence_scoring": "Per-field conformal prediction (0.0-1.0)",
            "model_preference": DEFAULT_MODEL_CHAIN,
            "output_format": "MethodStruct JSON with confidence scores",
        }
