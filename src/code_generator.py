"""
Code Generator (Upgraded - Dual Model Architecture)
====================================================
Generates executable Python files from a MethodStruct.

Key improvements over previous version:
  - Uses LLMRouter with role="coder" for coding-specialized models
  - Injects codegen_rules.md reference knowledge into every prompt
  - Accepts raw paper text for full context (not just sparse JSON)
  - Multi-step generation: architecture plan → model → train → utils
  - Confidence-aware: flags low-confidence parameters with TODOs
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config import CODE_GEN_TEMPERATURE
from llm_router import LLMRouter
from schemas import GeneratedFile, MethodStruct

logger = logging.getLogger(__name__)


# ── System prompt template ────────────────────────────

SYSTEM_PROMPT = """You are an expert ML research engineer. Your task is to implement a research paper's methodology as clean, runnable PyTorch code.

CRITICAL RULES:
1. Implement the EXACT architecture described — never substitute with a generic model.
2. Use proper PyTorch patterns: nn.Module with __init__ and forward, device management, gradient clipping.
3. Include shape comments in forward pass: # (B, C, H, W) -> (B, C', H', W')
4. Set random seeds for reproducibility.
5. Include both training AND validation in the training loop.
6. Save the best model checkpoint.
7. IMPORTANT: Use synthetic/placeholder data for training — do NOT hardcode dataset paths. The code must be runnable without external data files. Use TensorDataset with random tensors or create a synthetic data generator.
8. DO NOT download pretrained models or weights from Hugging Face, Torch Hub, or any external source. Use randomly initialized weights only.
9. Add type hints and docstrings.
10. Do NOT leave TODO placeholders or incomplete code.
11. Use ONLY ASCII characters in code and print statements (e.g., use '->' instead of '→') to prevent UnicodeEncodeError in Windows.

OUTPUT FORMAT:
Return ONLY a JSON array of file objects. No markdown, no explanation.
[{{"path": "model.py", "content": "..."}}, {{"path": "train.py", "content": "..."}}, ...]"""


# ── User prompt template ─────────────────────────────

USER_PROMPT_TEMPLATE = """## Paper: {algorithm_name}

### Method Summary
{paper_summary}

### Architecture Specification
- Layer types (in order): {layer_types}
- Input shape: {input_shape}
- Output shape: {output_shape}
- Number of classes: {num_classes}
- Hidden dimensions: {hidden_dims}
- Attention type: {attention_type}
- Key components: {key_components}
- Preprocessing: {preprocessing}

### Training Configuration
- Optimizer: {optimizer}
- Loss function: {loss}
- Learning rate: {learning_rate}
- Batch size: {batch_size}
- Epochs: {epochs}

### Mathematical Equations
{equations}

### Datasets
{datasets}
{confidence_notes}
### Raw Method Section from Paper
{paper_text}

### Required Output Files
Generate these files:
1. `model.py` — Complete model class implementing the EXACT architecture above
2. `train.py` — Full training + validation loop with proper metrics
3. `utils.py` — Data loading, preprocessing, and helper functions

Return ONLY a JSON array: [{{"path": "model.py", "content": "..."}}, ...]"""


def generate_code(
    method: MethodStruct,
    paper_text: str = "",
    framework: str = "pytorch",
    use_llm: bool = True,
) -> List[GeneratedFile]:
    """
    Generate executable code from a MethodStruct.

    Args:
        method: Structured method information from the paper
        paper_text: Raw method section text for full context
        framework: Target framework (currently 'pytorch')
        use_llm: Whether to use LLM (False = template stub only)

    Returns:
        List of GeneratedFile objects
    """
    if not use_llm:
        return _pytorch_stub(method)

    router = LLMRouter()

    # Build the enriched prompt
    arch = method.architecture
    training = method.training

    # Build confidence warning notes
    confidence_notes = ""
    if method.confidence:
        low_conf = method.low_confidence_fields(threshold=0.5)
        if low_conf:
            confidence_notes = (
                "\n### ⚠️ Low-Confidence Fields (use sensible defaults, mark with comments)\n"
                + "\n".join(f"- {f}: possibly inferred, not explicitly stated" for f in low_conf)
                + "\n"
            )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        algorithm_name=method.algorithm_name or "Unknown Model",
        paper_summary=method.paper_summary or "No summary available.",
        layer_types=", ".join(arch.layer_types) if arch.layer_types else "Not specified",
        input_shape=arch.input_shape or "Not specified",
        output_shape=arch.output_shape or "Not specified",
        num_classes=arch.num_classes or "Not specified",
        hidden_dims=", ".join(str(d) for d in arch.hidden_dims) if arch.hidden_dims else "Not specified",
        attention_type=arch.attention_type or "None",
        key_components=", ".join(arch.key_components) if arch.key_components else "None",
        preprocessing=", ".join(arch.preprocessing) if arch.preprocessing else "Not specified",
        optimizer=training.optimizer or "Adam (default)",
        loss=training.loss or "CrossEntropyLoss (default)",
        learning_rate=training.learning_rate or "0.001 (default)",
        batch_size=training.batch_size or "32 (default)",
        epochs=training.epochs or "10 (default)",
        equations="\n".join(f"- {eq}" for eq in method.equations) if method.equations else "None specified",
        datasets=", ".join(method.datasets) if method.datasets else "No specific dataset mentioned (use synthetic data)",
        confidence_notes=confidence_notes,
        paper_text=paper_text[:4000] if paper_text else "No raw text available.",
    )

    logger.info(
        f"[CodeGenerator] Generating code for: {method.algorithm_name} | "
        f"layers={arch.layer_types} | datasets={method.datasets}"
    )

    try:
        # Use LLMRouter with role="coder" and inject codegen_rules.md
        content = router.chat(
            role="coder",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=CODE_GEN_TEMPERATURE,
            inject_codegen_rules=True,
        )

        if content:
            files = _parse_files_json(content)
            if files:
                logger.info(f"[CodeGenerator] Generated {len(files)} files: {[f.path for f in files]}")
                return files

        logger.warning("[CodeGenerator] LLM returned empty/unparseable response. Using stub.")
    except Exception as e:
        logger.warning(f"[CodeGenerator] LLM generation failed: {e}. Using template.")

    return _pytorch_stub(method)


def _pytorch_stub(method: MethodStruct) -> List[GeneratedFile]:
    """Generate an architecture-aware PyTorch template when LLM is unavailable.

    Unlike the old version which always generated a 3-layer MLP, this stub
    attempts to match the extracted architecture type.
    """
    algo_name = method.algorithm_name or "SimpleModel"
    class_name = "".join(w.capitalize() for w in algo_name.replace("-", " ").split()) or "SimpleModel"
    arch = method.architecture

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
    num_classes = arch.num_classes or 10

    # Determine architecture type from layer_types
    layer_types_lower = [l.lower() for l in arch.layer_types]
    has_cnn = any("conv" in l for l in layer_types_lower)
    has_lstm = any("lstm" in l or "rnn" in l or "gru" in l for l in layer_types_lower)
    has_attention = any("attention" in l for l in layer_types_lower) or arch.attention_type
    has_transformer = any("transformer" in l for l in layer_types_lower)

    # Build model code based on detected architecture
    if has_cnn and has_lstm:
        model_code = _stub_cnn_lstm(class_name, algo_name, num_classes, has_attention)
    elif has_transformer or (has_attention and not has_cnn):
        model_code = _stub_transformer(class_name, algo_name, num_classes)
    elif has_cnn:
        model_code = _stub_cnn(class_name, algo_name, num_classes)
    else:
        model_code = _stub_mlp(class_name, algo_name, num_classes)

    train_code = f'''{warning_block}import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import {class_name}

# ─── Reproducibility ───
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic placeholder data — replace with actual dataset
X = torch.randn(1000, 3, 48, 48) if {has_cnn or has_transformer} else torch.randn(1000, 128)
y = torch.randint(0, {num_classes}, (1000,))
dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size={bs}, shuffle=True)
val_loader = DataLoader(val_set, batch_size={bs}, shuffle=False)

model = {class_name}(num_classes={num_classes}).to(device)
criterion = nn.{loss}()
optimizer = optim.{optimizer}(model.parameters(), lr={lr}, weight_decay=1e-4)

best_val_acc = 0.0
for epoch in range(1, {epochs} + 1):
    # ─── Train ───
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        correct += out.argmax(1).eq(yb).sum().item()
        total += xb.size(0)

    # ─── Validate ───
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += out.argmax(1).eq(yb).sum().item()
            val_total += xb.size(0)

    train_acc = 100.0 * correct / total
    val_acc = 100.0 * val_correct / val_total
    print(f"Epoch {{epoch}}/{epochs} | Train Loss: {{train_loss/total:.4f}} Acc: {{train_acc:.1f}}% | Val Acc: {{val_acc:.1f}}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"Training complete. Best val accuracy: {{best_val_acc:.1f}}%")
'''

    utils_code = '''"""Utility functions for data loading and preprocessing."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Compute accuracy@k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomDataset(Dataset):
    """Custom dataset — replace with actual data loading."""

    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        # Load your data here

    def __len__(self) -> int:
        return 0  # Replace with actual length

    def __getitem__(self, idx: int):
        # Replace with actual data loading
        raise NotImplementedError("Implement data loading")
'''

    return [
        GeneratedFile(path="model.py", content=model_code),
        GeneratedFile(path="train.py", content=train_code),
        GeneratedFile(path="utils.py", content=utils_code),
    ]


# ── Architecture-specific stubs ─────────────────────

def _stub_cnn_lstm(class_name: str, algo_name: str, num_classes: int, has_attention: bool) -> str:
    """CNN + LSTM/BiLSTM stub with optional attention."""
    attn_layer = """
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=4,
            batch_first=True,
        )""" if has_attention else ""

    attn_forward = """
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = attn_out[:, -1, :]  # (B, hidden_dim*2)""" if has_attention else """
        x = lstm_out[:, -1, :]  # (B, hidden_dim*2)"""

    return f'''import torch
import torch.nn as nn


class {class_name}(nn.Module):
    """Architecture: {algo_name}
    CNN feature extraction → BiLSTM sequence modeling {'→ Attention' if has_attention else ''} → Classification
    """

    def __init__(self, num_classes: int = {num_classes}, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
{attn_layer}
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        features = self.features(x)              # (B, 256, 4, 4)
        b, c, h, w = features.size()
        seq = features.view(b, c, h * w)          # (B, 256, 16)
        seq = seq.permute(0, 2, 1)                # (B, 16, 256)

        lstm_out, _ = self.bilstm(seq)            # (B, 16, hidden_dim*2)
{attn_forward}
        out = self.classifier(x)                  # (B, num_classes)
        return out
'''


def _stub_transformer(class_name: str, algo_name: str, num_classes: int) -> str:
    return f'''import torch
import torch.nn as nn
import math


class {class_name}(nn.Module):
    """Architecture: {algo_name}
    Transformer-based classification model.
    """

    def __init__(self, num_classes: int = {num_classes}, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, patch_size: int = 16, in_channels: int = 3, img_size: int = 224):
        super().__init__()
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = self.patch_embed(x)                      # (B, d_model, H', W')
        patches = patches.flatten(2).transpose(1, 2)       # (B, num_patches, d_model)
        b = patches.size(0)
        cls = self.cls_token.expand(b, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls, patches], dim=1)               # (B, num_patches+1, d_model)
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)                                # (B, num_patches+1, d_model)
        x = self.norm(x[:, 0])                             # (B, d_model) — CLS token
        return self.head(x)                                # (B, num_classes)
'''


def _stub_cnn(class_name: str, algo_name: str, num_classes: int) -> str:
    return f'''import torch
import torch.nn as nn


class {class_name}(nn.Module):
    """Architecture: {algo_name}
    CNN-based classification model.
    """

    def __init__(self, num_classes: int = {num_classes}, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # (B, 256, 1, 1)
        return self.classifier(x)  # (B, num_classes)
'''


def _stub_mlp(class_name: str, algo_name: str, num_classes: int) -> str:
    return f'''import torch
import torch.nn as nn


class {class_name}(nn.Module):
    """Architecture: {algo_name}"""

    def __init__(self, input_dim: int = 128, num_classes: int = {num_classes}):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''


def _parse_files_json(text: str) -> List[GeneratedFile]:
    """Robustly parse JSON array of file objects from LLM output."""
    text = text.strip()

    # Strip markdown fences
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
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    files: List[GeneratedFile] = []
    for item in data:
        if isinstance(item, dict):
            p = item.get("path")
            c = item.get("content")
            if p and c is not None:
                files.append(GeneratedFile(path=p, content=c))
    return files
