"""
Multi-agent version of paper_to_code pipeline.
Uses the orchestrator to coordinate all agents.
Maintains compatibility with the original paper_to_code interface.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

# Ensure the src/ directory is on sys.path so bare module imports work
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agents.orchestrator import Orchestrator
from linker import build_markdown
from schemas import MethodStruct, GeneratedFile
from validator import write_files, self_heal_cycle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
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
    
    text = full_text_path.read_text(encoding="utf-8", errors="ignore")
    
    # Use orchestrator to process the paper
    orchestrator = Orchestrator()
    pdf_path = RAW_PDF_DIR / f"{paper_base}.pdf"
    
    # Process using orchestrator (handles both PDF and text)
    logger.info(f"Processing '{paper_base}' with multi-agent pipeline...")
    if pdf_path.exists():
        results = orchestrator.process_paper(pdf_path=pdf_path, paper_base=paper_base)
    else:
        results = orchestrator.process_paper(text=text, paper_base=paper_base)
    
    # ── Extract method struct from results ──
    method_extraction = results.get("stages", {}).get("method_extraction", {})
    if method_extraction.get("success"):
        method_data = method_extraction.get("data", {}).get("method_struct", {})
        try:
            method_struct = MethodStruct(**method_data) if method_data else MethodStruct()
        except Exception as e:
            logger.warning(f"Failed to parse method_struct from agent response: {e}. Using fallback.")
            method_struct = _fallback_method_extraction(text)
    else:
        logger.warning(f"Method extraction stage failed. Using fallback heuristic extraction.")
        method_struct = _fallback_method_extraction(text)

    # Save method.json
    (out_dir / "method.json").write_text(method_struct.model_dump_json(indent=2), encoding="utf-8")

    # ── Get generated files from code generation stage ──
    code_generation = results.get("stages", {}).get("code_generation", {})
    if code_generation.get("success"):
        files_data = code_generation.get("data", {}).get("files", [])
        try:
            files = [GeneratedFile(path=f["path"], content=f["content"]) for f in files_data]
        except Exception as e:
            logger.warning(f"Failed to parse generated files: {e}. Using fallback code generation.")
            files = _fallback_code_generation(method_struct)
    else:
        logger.warning(f"Code generation stage failed. Using fallback.")
        files = _fallback_code_generation(method_struct)

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

    logger.info(f"Pipeline complete. Artifacts in: {out_dir}")
    return out_dir


def _fallback_method_extraction(text: str) -> MethodStruct:
    """Fallback to heuristic method extraction."""
    from method_extractor import find_method_sections, extract_method_entities
    method_sections = find_method_sections(text)
    method_text = "\n\n".join(method_sections) if method_sections else text
    return extract_method_entities(method_text)


def _fallback_code_generation(method_struct: MethodStruct):
    """Fallback to legacy code generation."""
    from code_generator import generate_code
    return generate_code(method_struct)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Run paper-to-code pipeline with multi-agent architecture.")
    parser.add_argument("--paper-base", required=True, help="Base name of the paper (without extension)")
    parser.add_argument("--multiagent", action="store_true", default=True, help="Use multi-agent architecture (default: True)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy single-pipeline mode")
    args = parser.parse_args()
    
    out = run_paper_to_code(args.paper_base, use_multiagent=not args.legacy)
    print("Artifacts in:", out)
