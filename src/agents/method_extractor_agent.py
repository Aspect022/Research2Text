"""
Agent 4: Method Extractor Agent (Upgraded - Phase 3 + Dual Model + Rules)
Responsibility: Extract structured method information with confidence scoring
              and detailed architecture breakdown for code generation.

Uses LLMRouter with role="reasoner" for extraction tasks.
Loads comprehensive rules from data/rules/ for high-quality extraction.

Implements a simplified Conformal Prediction framework:
  - Each extracted field gets a confidence score (0.0–1.0)
  - Fields explicitly stated in the paper get high scores
  - Fields inferred by the LLM (not directly quoted) get lower scores
  - Low-confidence fields are flagged for downstream agents

LLM Strategy:
  - Primary: LLMRouter with role="reasoner" (auto-selects best model)
  - Rules: Comprehensive 10k+ line rule files for pattern matching
  - Fallback: Heuristic regex-based extraction
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXTRACTION_TEMPERATURE, MAX_TEXT_LENGTH_FOR_LLM, LOW_CONFIDENCE_THRESHOLD
from llm_router import LLMRouter
from method_extractor import find_method_sections, extract_method_entities
from schemas import ArchitectureDetail, ConfidenceScore, MethodStruct
from rules import RuleLoader, load_rules_for_agent
from .base import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)

# ── Enriched extraction prompt ────────────────────────

EXTRACTION_PROMPT = """You are a precise scientific paper analyzer specializing in deep learning architectures. Extract ONLY information that is EXPLICITLY stated in the text below. Do NOT guess or infer values that are not mentioned.

TEXT:
{text}

Extract a JSON object with these fields. For each field, ONLY include values you can directly find in the text. Leave fields as null/empty if not explicitly stated:

{{
  "algorithm_name": "EXACT name of the primary algorithm/method as stated in the paper (e.g. 'CNN-BiLSTM with Attention Mechanism', NOT just 'Transformer')",
  "paper_summary": "2-3 sentence summary of what the paper proposes and how it works",
  "equations": ["List of mathematical equations mentioned, in LaTeX or text form"],
  "datasets": ["List of dataset names explicitly mentioned (e.g. 'CIFAR-10', 'FER2013', 'ImageNet')"],
  "training": {{
    "optimizer": "Optimizer name (e.g. 'Adam', 'SGD') or null",
    "loss": "Loss function name (e.g. 'CrossEntropyLoss', 'MSE') or null",
    "epochs": "Number of training epochs (integer) or null",
    "learning_rate": "Learning rate value (float) or null",
    "batch_size": "Batch size (integer) or null"
  }},
  "architecture": {{
    "layer_types": ["ORDERED list of all layer types in the model, e.g. ['Conv2D', 'BatchNorm', 'ReLU', 'MaxPool', 'BiLSTM', 'Attention', 'Dense']"],
    "input_shape": "Input tensor shape if mentioned, e.g. '(batch, 3, 48, 48)' or null",
    "output_shape": "Output tensor shape if mentioned, e.g. '(batch, 7)' or null",
    "num_classes": "Number of output classes (integer) or null",
    "hidden_dims": [128, 256],
    "attention_type": "Type of attention mechanism ('self', 'cross', 'multi-head', 'additive', 'bahdanau') or null",
    "preprocessing": ["Data preprocessing steps mentioned, e.g. 'resize to 48x48', 'grayscale conversion', 'normalization'"],
    "key_components": ["Named architectural components, e.g. 'ResidualBlock', 'SqueezeExcitation', 'DropoutLayer', 'BatchNorm']"]
  }},
  "inputs": {{"description": "What the model takes as input (e.g. 'facial expression images 48x48 grayscale')"}},
  "outputs": {{"description": "What the model produces (e.g. '7-class emotion prediction')"}},
  "references": ["Citation markers like [1], [2]"]
}}

CRITICAL RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no code fences.
2. If a value is NOT explicitly stated in the text, use null (not a guess).
3. For algorithm_name: use the FULL name as the paper describes it (e.g. "CNN-BiLSTM with Attention" not just "CNN").
4. For architecture.layer_types: list EVERY distinct layer type mentioned, in the order they appear in the model.
5. For equations: include the exact mathematical expression as written.
6. For numerical values (epochs, learning_rate, batch_size), only include if a specific number is given."""


class MethodExtractorAgent(BaseAgent):
    """Agent for extracting structured method information with confidence scoring."""

    def __init__(self, model: Optional[str] = None):
        super().__init__("method_extractor", "Method Extractor Agent")
        self._router = LLMRouter()
        self._rules = self._load_rules()

    def _load_rules(self) -> Optional[Dict[str, Any]]:
        """Load comprehensive extraction rules."""
        try:
            rules = load_rules_for_agent("method_extraction")
            logger.info(f"[MethodExtractor] Loaded rules version {rules.version}")
            return rules
        except FileNotFoundError:
            logger.warning("[MethodExtractor] Rules not found, using default prompt")
            return None
        except Exception as e:
            logger.error(f"[MethodExtractor] Error loading rules: {e}")
            return None

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
        # Step 1: Extract structured data via LLMRouter
        raw_data = self._llm_extract(focused_text)
        if not raw_data:
            return None

        # Step 2: Compute confidence scores by cross-referencing with source text
        confidence = self._compute_confidence(raw_data, full_text)

        # Step 3: Build architecture detail
        arch_data = raw_data.pop("architecture", {})
        if isinstance(arch_data, dict):
            architecture = ArchitectureDetail(**arch_data)
        else:
            architecture = ArchitectureDetail()

        # Step 4: Build MethodStruct with confidence and architecture
        try:
            method_struct = MethodStruct(
                **raw_data,
                architecture=architecture,
                confidence=confidence,
            )
            return method_struct
        except Exception as e:
            logger.warning(f"[MethodExtractor] Failed to build MethodStruct: {e}")
            # Try building with just the basic fields
            try:
                safe_data = {
                    "algorithm_name": raw_data.get("algorithm_name"),
                    "equations": raw_data.get("equations", []),
                    "datasets": raw_data.get("datasets", []),
                    "paper_summary": raw_data.get("paper_summary"),
                    "architecture": architecture,
                    "confidence": confidence,
                }
                training = raw_data.get("training", {})
                if isinstance(training, dict):
                    safe_data["training"] = training
                return MethodStruct(**safe_data)
            except Exception as e2:
                logger.warning(f"[MethodExtractor] Fallback build also failed: {e2}")
                return None

    def _llm_extract(self, text: str) -> Optional[Dict[str, Any]]:
        """Run the LLM extraction with rules-enhanced prompt via LLMRouter."""
        # Build prompt with rules if available
        if self._rules:
            rules_context = RuleLoader.format_for_llm(self._rules, max_chars=6000)
            prompt = f"""{rules_context}

---

Now extract method information from this research paper text:

TEXT:
{text[:MAX_TEXT_LENGTH_FOR_LLM]}

Extract a JSON object following the patterns and rules above. For each field, indicate your confidence based on evidence strength in the text.

Return ONLY valid JSON with these fields:
- algorithm_name
- paper_summary
- equations
- datasets
- training (optimizer, loss, epochs, learning_rate, batch_size)
- architecture (layer_types, input_shape, output_shape, num_classes, hidden_dims, attention_type, preprocessing, key_components)
- inputs
- outputs
- references

CRITICAL:
1. Use patterns from the rules above
2. Only extract explicitly stated information
3. Use null for values not mentioned
4. Return ONLY JSON, no markdown or explanation
"""
        else:
            # Fallback to default prompt
            prompt = EXTRACTION_PROMPT.format(text=text[:MAX_TEXT_LENGTH_FOR_LLM])

        result = self._router.chat_json(
            role="reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=EXTRACTION_TEMPERATURE,
        )

        if result:
            logger.info(
                f"[MethodExtractor] LLM extracted: algorithm={result.get('algorithm_name')}, "
                f"layers={result.get('architecture', {}).get('layer_types', [])}, "
                f"datasets={result.get('datasets', [])}"
            )
            return result

        logger.warning("[MethodExtractor] LLM returned empty/unparseable result")
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

        # Score architecture details
        arch = data.get("architecture", {})
        if isinstance(arch, dict) and arch.get("layer_types"):
            layer_scores = []
            for layer in arch["layer_types"]:
                score, _ = self._score_string_field(layer, source_text)
                layer_scores.append(score)
            avg = sum(layer_scores) / len(layer_scores) if layer_scores else 0.0
            confidence["architecture.layer_types"] = ConfidenceScore(
                score=avg,
                source="explicit" if avg >= 0.85 else "inferred",
                evidence=f"Found {len([s for s in layer_scores if s >= 0.85])}/{len(arch['layer_types'])} layers in text",
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
            "description": "Extract structured method information with architecture details and conformal prediction confidence scoring",
            "extraction_method": "LLMRouter (role=reasoner) with heuristic fallback",
            "confidence_scoring": "Per-field conformal prediction (0.0-1.0)",
            "architecture_extraction": "Layer types, dims, attention, preprocessing, key components",
            "output_format": "MethodStruct JSON with ArchitectureDetail and confidence scores",
        }
