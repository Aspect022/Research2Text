"""
Multi-agent version of paper_to_code pipeline.
Uses the orchestrator to coordinate all 9 agents.
Maintains compatibility with the original paper_to_code interface.
"""

from pathlib import Path
from typing import Dict, Optional

from agents.orchestrator import Orchestrator
from linker import build_markdown
from schemas import MethodStruct, GeneratedFile
from validator import write_files, self_heal_cycle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


def run_paper_to_code(paper_base: str, use_multiagent: bool = True) -> Path:
    """
    Run paper-to-code pipeline using multi-agent architecture.
    
    Args:
        paper_base: Base name of the paper (without extension)
        use_multiagent: If True, use multi-agent system; if False, use legacy pipeline
        
    Returns:
        Path to output directory
    """
    if not use_multiagent:
        # Fallback to legacy implementation
        from paper_to_code import run_paper_to_code as legacy_run
        return legacy_run(paper_base)
    
    out_dir = OUTPUTS_DIR / paper_base
    out_dir.mkdir(parents=True, exist_ok=True)

    full_text_path = RAW_TEXT_DIR / f"{paper_base}.txt"
    if not full_text_path.exists():
        raise FileNotFoundError(f"Missing text file: {full_text_path}")
    
    # Use orchestrator to process the paper
    orchestrator = Orchestrator()
    pdf_path = PROJECT_ROOT / "data" / "raw_pdfs" / f"{paper_base}.pdf"
    text = full_text_path.read_text(encoding="utf-8", errors="ignore")
    
    # Process using orchestrator (handles both PDF and text)
    if pdf_path.exists():
        results = orchestrator.process_paper(pdf_path=pdf_path, paper_base=paper_base)
    else:
        results = orchestrator.process_paper(text=text, paper_base=paper_base)
    
    # Extract method struct from results
    method_extraction = results.get("stages", {}).get("method_extraction", {})
    if method_extraction.get("success"):
        method_data = method_extraction.get("data", {}).get("method_struct", {})
        method_struct = MethodStruct(**method_data) if method_data else MethodStruct()
    else:
        # Fallback to legacy extraction
        text = full_text_path.read_text(encoding="utf-8", errors="ignore")
        from method_extractor import find_method_sections, extract_method_entities
        method_sections = find_method_sections(text)
        method_text = "\n\n".join(method_sections) if method_sections else text
        method_struct = extract_method_entities(method_text)

    # Save method.json
    (out_dir / "method.json").write_text(method_struct.model_dump_json(indent=2), encoding="utf-8")

    # Get generated files from code generation stage
    code_generation = results.get("stages", {}).get("code_generation", {})
    if code_generation.get("success"):
        files_data = code_generation.get("data", {}).get("files", [])
        files = [GeneratedFile(path=f["path"], content=f["content"]) for f in files_data]
    else:
        # Fallback to legacy code generation
        from code_generator import generate_code
        files = generate_code(method_struct)

    # Write files and validate
    result = self_heal_cycle(out_dir, files, method_struct, max_attempts=1)

    code_paths = []
    sandbox = out_dir / "sandbox"
    for f in files:
        code_paths.append(str((sandbox / f.path).relative_to(out_dir)))

    paper_meta: Dict[str, str] = {"title": paper_base, "authors": ""}
    report = build_markdown(paper_meta, method_struct, code_paths, Path(result.logs_dir) if result.logs_dir else out_dir)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    
    # Save knowledge graph if available
    graph_data = results.get("stages", {}).get("knowledge_graph", {})
    if graph_data.get("success"):
        import json
        graph_json = json.dumps({
            "nodes": graph_data.get("data", {}).get("nodes", []),
            "edges": graph_data.get("data", {}).get("edges", [])
        }, indent=2)
        (out_dir / "knowledge_graph.json").write_text(graph_json, encoding="utf-8")

    return out_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-base", required=True, help="Base name of the paper (without extension)")
    parser.add_argument("--multiagent", action="store_true", help="Use multi-agent architecture")
    args = parser.parse_args()
    out = run_paper_to_code(args.paper_base, use_multiagent=args.multiagent)
    print("Artifacts in:", out)

