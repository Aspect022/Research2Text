import io
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_texts"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


def _add_file_to_zip(zf: ZipFile, file_path: Path, arc_prefix: str = "") -> None:
    arcname = f"{arc_prefix}/{file_path.name}" if arc_prefix else file_path.name
    zf.write(file_path, arcname=arcname)


def build_artifacts_zip(base_name: str) -> bytes:
    """
    Create an in-memory zip containing available artifacts for a given paper base name.

    Priority:
    1) data/outputs/{base_name}/* (generated code, reports, logs)
    2) data/raw_texts/{base_name}* (full text + chunks)
    3) data/raw_pdfs/{base_name}.pdf (original pdf if exists)
    """
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        # outputs directory (if exists)
        out_dir = OUTPUTS_DIR / base_name
        if out_dir.exists():
            for p in out_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(out_dir)
                    zf.write(p, arcname=f"outputs/{base_name}/{rel.as_posix()}")

        # raw texts (full + chunks)
        for p in sorted(RAW_TEXT_DIR.glob(f"{base_name}*.txt")):
            _add_file_to_zip(zf, p, arc_prefix=f"raw_texts/{base_name}")

        # original pdf
        pdf_path = RAW_PDF_DIR / f"{base_name}.pdf"
        if pdf_path.exists():
            _add_file_to_zip(zf, pdf_path, arc_prefix="raw_pdfs")

    buf.seek(0)
    return buf.getvalue()


def list_known_bases() -> list[str]:
    bases = {p.stem.split("_chunk_")[0] for p in RAW_TEXT_DIR.glob("*_chunk_*.txt")}
    # also include outputs bases
    if OUTPUTS_DIR.exists():
        bases.update({p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir()})
    return sorted(bases)


def build_code_zip(base_name: str) -> bytes:
    """Create an in-memory zip containing only generated source code for a paper.

    Picks files from data/outputs/{base}/sandbox/* (e.g., model.py, train.py, utils.py)
    """
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        sandbox = OUTPUTS_DIR / base_name / "sandbox"
        if sandbox.exists():
            for p in sandbox.rglob("*.py"):
                rel = p.relative_to(sandbox)
                zf.write(p, arcname=f"{base_name}/{rel.as_posix()}")
    buf.seek(0)
    return buf.getvalue()


