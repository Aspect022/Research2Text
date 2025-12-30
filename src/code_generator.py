from pathlib import Path
from typing import List
import json

from schemas import GeneratedFile, MethodStruct


def _pytorch_stub(method: MethodStruct) -> List[GeneratedFile]:
    model_code = """import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
"""

    train_code = """import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleModel


def main():
    X = torch.randn(128, 128)
    y = torch.randint(0, 10, (128,))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model = SimpleModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("OK")


if __name__ == "__main__":
    main()
"""

    return [
        GeneratedFile(path="model.py", content=model_code),
        GeneratedFile(path="train.py", content=train_code),
    ]


def _parse_files_json(text: str) -> List[GeneratedFile]:
    try:
        data = json.loads(text)
    except Exception:
        # try to extract a JSON array
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


def generate_code(method: MethodStruct, framework: str = "pytorch", use_llm: bool = True, model: str = "gpt-oss:120b-cloud") -> List[GeneratedFile]:
    if not use_llm:
        return _pytorch_stub(method)
    try:
        import ollama
    except Exception:
        # fallback if ollama not available
        return _pytorch_stub(method)

    sys_prompt = (
        "You are an AI research engineer. Generate runnable Python source files implementing the described method. "
        "Target framework: PyTorch. Keep files minimal and self-contained. Do not include markdown or code fences. "
        "Return ONLY a JSON array of objects with keys 'path' and 'content'."
    )
    user_prompt = (
        "Method JSON:\n" + method.model_dump_json(indent=2) + "\n\n"
        "Constraints:\n"
        "- Include all necessary imports.\n"
        "- Provide a small training loop if a dataset is mentioned (use random data as placeholder).\n"
        "- Suggested files: model.py, train.py, utils.py.\n\n"
        "Output format: A JSON array like [{\"path\": \"model.py\", \"content\": \"...\"}, ...]"
    )
    resp = ollama.chat(model=model, messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    content = resp.get("message", {}).get("content", "")
    try:
        return _parse_files_json(content)
    except Exception:
        # fallback to stub on parse failure
        return _pytorch_stub(method)


