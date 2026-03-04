"""
LLM Router — Unified model routing with role-based selection.
=============================================================
Every agent calls LLMRouter instead of reimplementing model resolution.

Roles:
  - "reasoner" → REASONER_MODEL_CHAIN  (method extraction, analysis, Q&A)
  - "coder"    → CODER_MODEL_CHAIN     (code generation, self-heal, code review)

All models are local via Ollama. Zero API costs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

from config import (
    REASONER_MODEL_CHAIN,
    CODER_MODEL_CHAIN,
    DEFAULT_OLLAMA_MODEL,
    CODEGEN_RULES_PATH,
)

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM calls to the right Ollama model based on role."""

    _instance: Optional["LLMRouter"] = None
    _codegen_rules_cache: Optional[str] = None

    def __init__(self) -> None:
        self._available_models: Optional[List[str]] = None

    def __new__(cls) -> "LLMRouter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Model Discovery ──────────────────────────────────────

    def _detect_available_models(self) -> List[str]:
        """Query Ollama for installed models. Cache the result."""
        if self._available_models is not None:
            return self._available_models

        if ollama is None:
            logger.warning("ollama package not installed — using DEFAULT_OLLAMA_MODEL")
            self._available_models = [DEFAULT_OLLAMA_MODEL]
            return self._available_models

        try:
            response = ollama.list()
            # Handle both old dict format and new object format
            if isinstance(response, dict):
                models_list = response.get("models", [])
            elif hasattr(response, "models"):
                models_list = response.models or []
            else:
                models_list = []

            names: List[str] = []
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get("name", "")
                elif hasattr(m, "model"):
                    name = m.model or ""
                elif hasattr(m, "name"):
                    name = m.name or ""
                else:
                    name = str(m)
                if name:
                    names.append(name)
                    # Also match without tag  (e.g. 'qwen3' matches 'qwen3:latest')
                    if ":" in name:
                        names.append(name.split(":")[0])

            self._available_models = names
            logger.info(f"Detected {len(names)} Ollama model entries: {names[:5]}")
        except Exception as exc:
            logger.warning(f"Could not list Ollama models: {exc}")
            self._available_models = [DEFAULT_OLLAMA_MODEL]

        return self._available_models

    def _pick_model(self, role: str) -> str:
        """Pick the best available model for the given role."""
        chain = CODER_MODEL_CHAIN if role == "coder" else REASONER_MODEL_CHAIN
        available = self._detect_available_models()

        for model in chain:
            if model in available:
                logger.debug(f"Role '{role}' → model '{model}'")
                return model

        # Fallback: try any model from either chain
        for model in REASONER_MODEL_CHAIN + CODER_MODEL_CHAIN:
            if model in available:
                logger.warning(f"Role '{role}' fallback → '{model}'")
                return model

        # Last resort
        fallback = available[0] if available else DEFAULT_OLLAMA_MODEL
        logger.warning(f"Role '{role}' last-resort → '{fallback}'")
        return fallback

    # ── Reference Knowledge ──────────────────────────────────

    @classmethod
    def load_codegen_rules(cls) -> str:
        """Load the codegen_rules.md reference file. Cached after first read."""
        if cls._codegen_rules_cache is not None:
            return cls._codegen_rules_cache

        rules_path = Path(CODEGEN_RULES_PATH)
        if rules_path.exists():
            cls._codegen_rules_cache = rules_path.read_text(encoding="utf-8")
            logger.info(
                f"Loaded codegen rules ({len(cls._codegen_rules_cache)} chars)"
            )
        else:
            logger.warning(f"codegen_rules.md not found at {rules_path}")
            cls._codegen_rules_cache = ""

        return cls._codegen_rules_cache

    # ── Main Chat Interface ──────────────────────────────────

    def chat(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        inject_codegen_rules: bool = False,
    ) -> str:
        """
        Send a chat request to the appropriate Ollama model.

        Args:
            role: "reasoner" or "coder"
            messages: List of {"role": "system"|"user"|"assistant", "content": ...}
            temperature: Sampling temperature
            inject_codegen_rules: If True, prepend codegen_rules.md to system message

        Returns:
            The model's response text.
        """
        model = self._pick_model(role)

        # Inject codegen rules into the system message if requested
        if inject_codegen_rules:
            rules = self.load_codegen_rules()
            if rules:
                messages = self._inject_rules(messages, rules)

        if ollama is None:
            logger.error("ollama package not available")
            return ""

        start = time.time()
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            elapsed = time.time() - start
            content: str = resp.get("message", {}).get("content", "")
            logger.info(
                f"LLM [{role}:{model}] responded in {elapsed:.1f}s "
                f"({len(content)} chars)"
            )
            return content
        except Exception as exc:
            logger.error(f"LLM [{role}:{model}] call failed: {exc}")

            # Try fallback if the primary model failed
            fallback = self._try_fallback(role, model, messages, temperature)
            if fallback is not None:
                return fallback

            return ""

    def _try_fallback(
        self,
        role: str,
        failed_model: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Optional[str]:
        """Attempt the next model in chain after a failure."""
        chain = CODER_MODEL_CHAIN if role == "coder" else REASONER_MODEL_CHAIN
        available = self._detect_available_models()

        for model in chain:
            if model == failed_model or model not in available:
                continue
            try:
                logger.info(f"Fallback [{role}] → '{model}'")
                resp = ollama.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": temperature},
                )
                return resp.get("message", {}).get("content", "")
            except Exception:
                continue

        return None

    @staticmethod
    def _inject_rules(
        messages: List[Dict[str, str]], rules: str
    ) -> List[Dict[str, str]]:
        """Prepend codegen rules to the system message."""
        messages = list(messages)  # shallow copy
        rules_block = (
            "# Code Generation Reference Rules\n"
            "Follow these patterns and practices for all generated code:\n\n"
            f"{rules}\n\n"
            "---\n"
        )

        # Either prepend to existing system message or insert a new one
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": rules_block + messages[0]["content"],
            }
        else:
            messages.insert(0, {"role": "system", "content": rules_block})

        return messages

    # ── JSON Extraction Helper ───────────────────────────────

    def chat_json(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Chat and parse the response as JSON. Returns {} on parse failure."""
        raw = self.chat(role, messages, temperature)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON block in markdown fences
        for marker in ["```json", "```"]:
            idx = text.find(marker)
            if idx != -1:
                start = idx + len(marker)
                end_idx = text.find("```", start)
                end = end_idx if end_idx != -1 else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try to find first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Could not extract JSON from LLM response")
        return {}
