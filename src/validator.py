import subprocess
from pathlib import Path
from typing import List

from config import SELF_HEAL_TEMPERATURE
from schemas import GeneratedFile, RunResult, ValidationResult, MethodStruct


def write_files(dst: Path, files: List[GeneratedFile]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = dst / f.path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f.content, encoding="utf-8")


def run_and_capture(cmd: List[str], cwd: Path, timeout: int = 60) -> RunResult:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return RunResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _llm_fix(files: List[GeneratedFile], error_text: str) -> List[GeneratedFile]:
    """Use LLMRouter (coder role) to fix broken code."""
    from llm_router import LLMRouter
    from config import SELF_HEAL_TEMPERATURE
    import json

    router = LLMRouter()
    sys_prompt = (
        "You are a senior Python engineer. The following files fail to run. "
        "Fix the errors and return the corrected files as a JSON array "
        "with 'path' and 'content' keys. Return ONLY valid JSON."
    )
    user_prompt = "ERROR:\n{err}\n\nFILES:\n{files}\n\nFix the code. Output JSON only.".format(
        err=error_text[:3000],
        files="\n\n".join(f"[{f.path}]\n{f.content}" for f in files),
    )
    try:
        content = router.chat(
            role="coder",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=SELF_HEAL_TEMPERATURE,
            inject_codegen_rules=True,
        )
        arr = json.loads(content) if content.strip().startswith("[") else []
        if not arr:
            # Try JSON extraction
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end > start:
                arr = json.loads(content[start:end + 1])

        fixed = []
        for it in arr:
            p = it.get("path")
            c = it.get("content")
            if p and c is not None:
                fixed.append(GeneratedFile(path=p, content=c))
        return fixed or files
    except Exception:
        return files


def self_heal_cycle(base_dir: Path, files: List[GeneratedFile], method: MethodStruct, max_attempts: int = 3) -> ValidationResult:
    sandbox = base_dir / "sandbox"
    write_files(sandbox, files)

    attempts = 0
    last_err = None
    logs_dir = (base_dir / "run_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    while attempts < max_attempts:
        attempts += 1
        res = run_and_capture(["python", "train.py"], cwd=sandbox, timeout=90)
        (logs_dir / f"attempt_{attempts}.out").write_text(res.stdout or "", encoding="utf-8")
        (logs_dir / f"attempt_{attempts}.err").write_text(res.stderr or "", encoding="utf-8")
        if res.returncode == 0:
            return ValidationResult(success=True, attempts=attempts, logs_dir=str(logs_dir))
        last_err = res.stderr or res.stdout
        # Attempt a single LLM-based fix iteration per attempt
        files = _llm_fix(files, last_err)
        # write fixed files back
        write_files(sandbox, files)
        # continue loop to retry

    return ValidationResult(success=False, attempts=attempts, last_error=last_err, logs_dir=str(logs_dir))


