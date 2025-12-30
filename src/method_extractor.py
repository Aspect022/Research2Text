import re
from typing import List

from schemas import MethodStruct


SECTION_HEADERS = [
    r"^\s*methods?\b",
    r"^\s*methodolog(y|ies)\b",
    r"^\s*materials? and methods\b",
]


def find_method_sections(text: str) -> List[str]:
    # naive heuristic: split by headings and select method-like sections
    chunks = re.split(r"\n\s*\n", text)
    method_like: List[str] = []
    for block in chunks:
        head = block.strip().splitlines()[0].lower() if block.strip().splitlines() else ""
        if any(re.match(pat, head) for pat in SECTION_HEADERS):
            method_like.append(block.strip())
    return method_like


def extract_method_entities(method_text: str) -> MethodStruct:
    # Placeholder heuristic before LLM integration
    algo = None
    datasets: List[str] = []
    equations: List[str] = []

    # crude finds
    if "transformer" in method_text.lower():
        algo = "Transformer"
    if re.search(r"cifar[- ]?10", method_text, flags=re.I):
        datasets.append("CIFAR-10")
    if re.search(r"cora", method_text, flags=re.I):
        datasets.append("Cora")
    for m in re.finditer(r"\bQK\^?T\b|softmax|cross[- ]entropy", method_text, flags=re.I):
        equations.append(m.group(0))

    return MethodStruct(
        algorithm_name=algo,
        equations=equations,
        datasets=datasets,
    )


