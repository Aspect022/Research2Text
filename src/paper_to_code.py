from pathlib import Path
from typing import Dict

from method_extractor import find_method_sections, extract_method_entities
from code_generator import generate_code
from validator import self_heal_cycle
from linker import build_markdown
from schemas import MethodStruct


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


def run_paper_to_code(paper_base: str) -> Path:
    out_dir = OUTPUTS_DIR / paper_base
    out_dir.mkdir(parents=True, exist_ok=True)

    full_text_path = RAW_TEXT_DIR / f"{paper_base}.txt"
    if not full_text_path.exists():
        raise FileNotFoundError(f"Missing text file: {full_text_path}")
    text = full_text_path.read_text(encoding="utf-8", errors="ignore")

    method_sections = find_method_sections(text)
    method_text = "\n\n".join(method_sections) if method_sections else text
    method_struct: MethodStruct = extract_method_entities(method_text)

    # Save method.json
    (out_dir / "method.json").write_text(method_struct.model_dump_json(indent=2), encoding="utf-8")

    files = generate_code(method_struct)
    result = self_heal_cycle(out_dir, files, method_struct, max_attempts=1)

    code_paths = []
    sandbox = out_dir / "sandbox"
    for f in files:
        code_paths.append(str((sandbox / f.path).relative_to(out_dir)))

    paper_meta: Dict[str, str] = {"title": paper_base, "authors": ""}
    report = build_markdown(paper_meta, method_struct, code_paths, Path(result.logs_dir) if result.logs_dir else out_dir)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    return out_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-base", required=True, help="Base name of the paper (without extension)")
    args = parser.parse_args()
    out = run_paper_to_code(args.paper_base)
    print("Artifacts in:", out)


