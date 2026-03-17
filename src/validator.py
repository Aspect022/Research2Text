"""
Code validation with sandboxed execution and self-healing.

Uses platform-specific sandboxes for secure code execution:
- Windows: WindowsSandbox (Job Objects + restricted Python)
- Linux: LinuxSandbox (Firejail or restricted Python)
"""

import logging
from pathlib import Path
from typing import List, Dict

from config import SELF_HEAL_TEMPERATURE, MAX_HEAL_ATTEMPTS
from schemas import GeneratedFile, RunResult, ValidationResult, MethodStruct
from sandbox import DefaultSandbox, SandboxResult

logger = logging.getLogger(__name__)


def write_files(dst: Path, files: List[GeneratedFile]) -> None:
    """Write generated files to directory."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = dst / f.path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f.content, encoding="utf-8")


def run_and_capture(cmd: List[str], cwd: Path, timeout: int = 60) -> RunResult:
    """Legacy function - kept for backward compatibility."""
    import subprocess
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return RunResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _llm_fix(files: List[GeneratedFile], error_text: str) -> List[GeneratedFile]:
    """Use LLMRouter (coder role) to fix broken code."""
    from llm_router import LLMRouter
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
    except Exception as e:
        logger.error(f"LLM fix failed: {e}")
        return files


def self_heal_cycle(
    base_dir: Path,
    files: List[GeneratedFile],
    method: MethodStruct,
    max_attempts: int = MAX_HEAL_ATTEMPTS
) -> ValidationResult:
    """
    Self-healing validation with proper sandboxing.

    Uses platform-specific sandbox (Windows/Linux) for secure execution.
    Iteratively fixes errors using LLM until success or max attempts.

    Args:
        base_dir: Base directory for outputs
        files: List of generated files to validate
        method: Method structure (for context)
        max_attempts: Maximum number of fix attempts

    Returns:
        ValidationResult with success status and logs
    """
    logs_dir = base_dir / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Convert files to dict format for sandbox
    code_files = {f.path: f.content for f in files}

    attempts = 0
    last_err = None

    # Use sandbox for execution
    with DefaultSandbox(
        memory_limit_mb=1024,
        cpu_time_limit_sec=90,
        network_access=False
    ) as sandbox:
        while attempts < max_attempts:
            attempts += 1
            logger.info(f"Validation attempt {attempts}/{max_attempts}")

            # Run in sandbox
            result = sandbox.run(
                code_files=code_files,
                entry_point="train.py",
                timeout=90
            )

            # Save logs
            (logs_dir / f"attempt_{attempts}.out").write_text(
                result.stdout or "", encoding="utf-8"
            )
            (logs_dir / f"attempt_{attempts}.err").write_text(
                result.stderr or "", encoding="utf-8"
            )

            # Check success
            if result.success:
                logger.info(f"Validation succeeded on attempt {attempts}")
                return ValidationResult(
                    success=True,
                    attempts=attempts,
                    logs_dir=str(logs_dir),
                    stdout=result.stdout,
                    stderr=result.stderr
                )

            # Get error for fixing
            last_err = result.stderr or result.stdout
            logger.warning(f"Attempt {attempts} failed: {last_err[:200]}...")

            # Try to fix if not last attempt
            if attempts < max_attempts:
                logger.info("Attempting LLM fix...")
                fixed_files = _llm_fix(
                    [GeneratedFile(path=p, content=c) for p, c in code_files.items()],
                    last_err
                )
                code_files = {f.path: f.content for f in fixed_files}

    # Max attempts reached
    logger.error(f"Validation failed after {attempts} attempts")
    return ValidationResult(
        success=False,
        attempts=attempts,
        last_error=last_err,
        logs_dir=str(logs_dir),
        stdout=result.stdout if 'result' in locals() else "",
        stderr=result.stderr if 'result' in locals() else ""
    )


def validate_code_simple(
    files: List[GeneratedFile],
    timeout: int = 90
) -> SandboxResult:
    """
    Simple validation without self-healing.

    Uses sandbox for single execution.

    Args:
        files: Generated code files
        timeout: Execution timeout

    Returns:
        SandboxResult with execution details
    """
    code_files = {f.path: f.content for f in files}

    with DefaultSandbox(
        memory_limit_mb=1024,
        cpu_time_limit_sec=timeout,
        network_access=False
    ) as sandbox:
        return sandbox.run(
            code_files=code_files,
            entry_point="train.py",
            timeout=timeout
        )
