from pathlib import Path
from typing import Dict, List

from schemas import MethodStruct


def build_markdown(paper_meta: Dict[str, str], method: MethodStruct, code_paths: List[str], run_logs_dir: Path) -> str:
    title = paper_meta.get("title", "Unknown Title")
    authors = paper_meta.get("authors", "?")

    md = []
    md.append(f"# 📘 Paper\n{title}")
    md.append(f"\n**Authors:** {authors}\n")
    md.append("\n# 🧮 Extracted Method\n")
    if method.algorithm_name:
        md.append(f"- Algorithm: {method.algorithm_name}")
    if method.equations:
        md.append(f"- Equations: {', '.join(method.equations)}")
    if method.datasets:
        md.append(f"- Datasets: {', '.join(method.datasets)}")
    if method.training:
        md.append(f"- Training: {method.training.model_dump(exclude_none=True)}")

    md.append("\n# 🧠 Generated Code\n")
    for p in code_paths:
        md.append(f"- `{p}`")

    md.append("\n# 📜 Run Logs\n")
    if run_logs_dir.exists():
        for f in sorted(run_logs_dir.glob("*.err")):
            md.append(f"- `{f.name}`")
        for f in sorted(run_logs_dir.glob("*.out")):
            md.append(f"- `{f.name}`")

    return "\n".join(md) + "\n"


