"""
Code Generator (Upgraded - Phase 3)
Generates executable Python files from a MethodStruct.

Upgrades:
  - Confidence-aware: uses MethodStruct.confidence to mark low-confidence
    parameters with TODO/WARNING comments in generated code.
  - Multi-model support: auto-detects available Ollama models.
  - Improved prompting optimized for coder-class models.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from config import MODEL_CHAIN, CODE_GEN_TEMPERATURE
from schemas import GeneratedFile, MethodStruct

logger = logging.getLogger(__name__)

# Models to try in order of preference for code generation
CODE_MODEL_CHAIN = MODEL_CHAIN


def _resolve_model(ollama_module: Any) -> Optional[str]:
    """Find the first available code-generation model."""
    try:
        available = ollama_module.list()
        available_names = set()
        if isinstance(available, dict):
            for m in available.get("models", []):
                name = m.get("name", "")
                available_names.add(name)
                if ":" in name:
                    available_names.add(name.split(":")[0])

        for preferred in CODE_MODEL_CHAIN:
            if preferred in available_names or preferred.split(":")[0] in available_names:
                return preferred

        if available_names:
            return list(available_names)[0]
    except Exception:
        pass
    return None


def _pytorch_stub(method: MethodStruct) -> List[GeneratedFile]:
    """Generate a basic PyTorch template when LLM is unavailable."""
    algo_name = method.algorithm_name or "SimpleModel"
    class_name = "".join(w.capitalize() for w in algo_name.split()) or "SimpleModel"

    # Check confidence for training parameters
    warnings = []
    if method.confidence:
        for field, conf in method.confidence.items():
            if conf.score < 0.5:
                warnings.append(f"# ⚠️ LOW CONFIDENCE ({conf.score:.1f}): {field} — {conf.evidence or 'possibly inferred'}")

    warning_block = "\n".join(warnings) + "\n" if warnings else ""

    lr = method.training.learning_rate or 1e-3
    bs = method.training.batch_size or 32
    epochs = method.training.epochs or 10
    optimizer = method.training.optimizer or "Adam"
    loss = method.training.loss or "CrossEntropyLoss"

    model_code = f'''import torch
import torch.nn as nn


class {class_name}(nn.Module):
    """Generated model for: {algo_name}"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
'''

    train_code = f'''{warning_block}import torch
from torch.utils.data import DataLoader, TensorDataset
from model import {class_name}


def main():
    # Synthetic placeholder data — replace with actual dataset loader
    X = torch.randn(256, 128)
    y = torch.randint(0, 10, (256,))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size={bs}, shuffle=True)

    model = {class_name}()
    opt = torch.optim.{optimizer}(model.parameters(), lr={lr})
    loss_fn = torch.nn.{loss}()

    model.train()
    for epoch in range({epochs}):
        total_loss = 0.0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {{epoch + 1}}/{epochs} — Loss: {{total_loss / len(dl):.4f}}")
    print("Training complete.")


if __name__ == "__main__":
    main()
'''

    return [
        GeneratedFile(path="model.py", content=model_code),
        GeneratedFile(path="train.py", content=train_code),
    ]


def _parse_files_json(text: str) -> List[GeneratedFile]:
    """Robustly parse JSON array of file objects from LLM output."""
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            raise

    files: List[GeneratedFile] = []
    for item in data:
        p = item.get("path")
        c = item.get("content")
        if p and c is not None:
            files.append(GeneratedFile(path=p, content=c))
    return files


def generate_code(
    method: MethodStruct,
    framework: str = "pytorch",
    use_llm: bool = True,
) -> List[GeneratedFile]:
    """
    Generate executable code from a MethodStruct.
    
    Uses an auto-detected LLM model, with confidence-aware prompting
    that marks low-confidence parameters as TODOs.
    """
    if not use_llm:
        return _pytorch_stub(method)

    try:
        import ollama
    except ImportError:
        return _pytorch_stub(method)

    model = _resolve_model(ollama)
    if not model:
        logger.warning("[CodeGenerator] No Ollama model available, using template stub")
        return _pytorch_stub(method)

    logger.info(f"[CodeGenerator] Generating code with model: {model}")

    # Build confidence warnings for the prompt
    confidence_notes = ""
    if method.confidence:
        low_conf = method.low_confidence_fields(threshold=0.5)
        if low_conf:
            confidence_notes = (
                "\n\nIMPORTANT — Low-confidence fields (mark these with # TODO in comments):\n"
                + "\n".join(f"  - {f}: possibly inferred, not explicitly in paper" for f in low_conf)
            )

    sys_prompt = (
        "You are an expert ML engineer. Generate clean, runnable Python files implementing "
        f"the described method. Target framework: {framework}. "
        "Keep files minimal, well-documented, and self-contained. "
        "Do NOT include markdown fences or explanations. "
        "Return ONLY a JSON array of objects with keys 'path' and 'content'."
    )

    user_prompt = (
        "Method specification:\n"
        + method.model_dump_json(indent=2, exclude={"confidence"})
        + confidence_notes
        + "\n\nConstraints:\n"
        "- Include all necessary imports.\n"
        "- Provide a training loop with real hyperparameters from the spec.\n"
        "- Use placeholder data if no dataset loader is available.\n"
        "- Mark any uncertain values with # TODO: verify this value.\n"
        "- Generate: model.py, train.py, and optionally utils.py.\n\n"
        'Output format: [{"path": "model.py", "content": "..."}, ...]'
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": CODE_GEN_TEMPERATURE},
        )
        content = resp.get("message", {}).get("content", "")
        return _parse_files_json(content)
    except Exception as e:
        logger.warning(f"[CodeGenerator] LLM generation failed: {e}. Using template.")
        return _pytorch_stub(method)
