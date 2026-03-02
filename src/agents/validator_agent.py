"""
Agent 9: Validator Agent (Upgraded - Phase 4)
Responsibility: Validate generated code via static analysis AND sandbox execution.

Validation Chain:
  1. Static Analysis — AST syntax check, import resolution
  2. Sandbox Execution — Actually run the code in an isolated temp directory
  3. Self-Heal Loop — If execution fails, use LLM to fix the code and retry
  4. Docker Isolation (optional) — Run in a container if Docker is available
"""

import ast
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentMessage, AgentResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MAX_HEAL_ATTEMPTS, EXECUTION_TIMEOUT, DOCKER_BUILD_TIMEOUT,
    DOCKER_MEMORY_LIMIT, DOCKER_CPU_LIMIT, SELF_HEAL_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Add project src to path for schema/validator imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidatorAgent(BaseAgent):
    """Agent for validating generated code via static + dynamic analysis."""

    def __init__(self, max_heal_attempts: int = MAX_HEAL_ATTEMPTS, execution_timeout: int = EXECUTION_TIMEOUT):
        super().__init__("validator", "Validator Agent")
        self._max_heal_attempts = max_heal_attempts
        self._execution_timeout = execution_timeout
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available for sandboxed execution."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Validate generated code files.

        Validation levels (executed in order):
          1. static   — AST syntax + import check (always runs)
          2. sandbox  — Execute in temp dir (if payload has execute=True or default)
          3. docker   — Execute in Docker container (if available and requested)
        """
        payload = message.payload
        files = payload.get("files", [])
        execute = payload.get("execute", True)
        use_docker = payload.get("use_docker", False)

        if not files:
            return AgentResponse(
                success=False,
                error="Missing 'files' in payload",
            )

        try:
            # Phase 1: Static analysis (always)
            static_results = self._static_analysis(files)

            # Phase 2: Dynamic execution (if requested and syntax is valid)
            execution_result = None
            all_syntax_valid = all(r["syntax_valid"] for r in static_results)

            if execute and all_syntax_valid and files:
                if use_docker and self._docker_available:
                    execution_result = self._execute_in_docker(files)
                else:
                    execution_result = self._execute_in_sandbox(files)

            # Compute overall metrics
            syntax_score = (
                sum(1 for r in static_results if r["syntax_valid"]) / len(static_results)
                if static_results else 0.0
            )
            import_score = (
                sum(1 for r in static_results if r["imports_valid"]) / len(static_results)
                if static_results else 0.0
            )

            return AgentResponse(
                success=True,
                data={
                    "static_analysis": static_results,
                    "syntax_correctness": syntax_score,
                    "import_resolution": import_score,
                    "execution": execution_result,
                    "docker_available": self._docker_available,
                },
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Validation failed: {str(e)}",
            )

    # ─── Static Analysis ──────────────────────────────

    def _static_analysis(self, files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Run AST-based static analysis on all files."""
        results = []
        for file_info in files:
            file_path = file_info.get("path", "")
            content = file_info.get("content", "")

            result: Dict[str, Any] = {
                "file": file_path,
                "syntax_valid": False,
                "imports_valid": False,
                "imports": [],
                "errors": [],
                "warnings": [],
            }

            # Syntax check
            syntax = self._check_syntax(content)
            result["syntax_valid"] = syntax["valid"]
            if not syntax["valid"]:
                result["errors"].extend(syntax.get("errors", []))

            # Import check
            if syntax["valid"]:
                imports = self._check_imports(content)
                result["imports_valid"] = imports["valid"]
                result["imports"] = imports.get("imports", [])
                if not imports["valid"]:
                    result["errors"].extend(imports.get("errors", []))

                # Check for common issues
                warnings = self._check_code_quality(content, file_path)
                result["warnings"] = warnings

            results.append(result)
        return results

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check Python syntax via AST parsing."""
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"SyntaxError at line {e.lineno}: {e.msg}"],
            }
        except Exception as e:
            return {"valid": False, "errors": [f"ParseError: {str(e)}"]}

    def _check_imports(self, code: str) -> Dict[str, Any]:
        """Extract and validate import statements."""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            return {"valid": True, "imports": imports}
        except Exception as e:
            return {"valid": False, "errors": [f"Import analysis failed: {str(e)}"]}

    def _check_code_quality(self, code: str, filename: str) -> List[str]:
        """Basic code quality checks."""
        warnings = []
        lines = code.split("\n")

        # Check for very long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 200:
                warnings.append(f"Line {i} in {filename}: very long ({len(line)} chars)")

        # Check for bare except clauses
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "except:" or stripped == "except Exception:":
                warnings.append(f"Line {i} in {filename}: bare except clause (consider specific exceptions)")

        # Check for TODO/FIXME
        for i, line in enumerate(lines, 1):
            if "TODO" in line or "FIXME" in line:
                warnings.append(f"Line {i} in {filename}: {line.strip()[:80]}")

        return warnings

    # ─── Sandbox Execution ────────────────────────────

    def _execute_in_sandbox(self, files: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Execute generated code in an isolated temp directory.
        Implements a self-heal loop: run → fail → LLM fix → retry.
        """
        sandbox_dir = tempfile.mkdtemp(prefix="r2t_sandbox_")
        sandbox_path = Path(sandbox_dir)

        try:
            # Write files to sandbox
            self._write_files_to_sandbox(files, sandbox_path)

            # Find the entry point (train.py > main.py > first .py file)
            entry = self._find_entry_point(files)
            if not entry:
                return {
                    "success": False,
                    "error": "No Python entry point found",
                    "attempts": 0,
                }

            # Self-heal loop
            current_files = files.copy()
            attempts = 0
            last_error = None

            while attempts < self._max_heal_attempts:
                attempts += 1
                logger.info(f"[Validator] Sandbox execution attempt {attempts}/{self._max_heal_attempts}: {entry}")

                result = self._run_in_sandbox(sandbox_path, entry)

                if result["returncode"] == 0:
                    logger.info(f"[Validator] Execution succeeded on attempt {attempts}")
                    return {
                        "success": True,
                        "attempts": attempts,
                        "stdout": result["stdout"][:1000],
                        "entry_point": entry,
                    }

                last_error = result["stderr"] or result["stdout"]
                logger.warning(f"[Validator] Attempt {attempts} failed: {last_error[:200]}")

                # Try LLM-based fix (only if not last attempt)
                if attempts < self._max_heal_attempts:
                    fixed_files = self._llm_fix(current_files, last_error)
                    if fixed_files != current_files:
                        current_files = fixed_files
                        self._write_files_to_sandbox(current_files, sandbox_path)
                    else:
                        break  # LLM couldn't fix, no point retrying

            return {
                "success": False,
                "attempts": attempts,
                "last_error": last_error[:500] if last_error else None,
                "entry_point": entry,
            }
        finally:
            # Clean up sandbox
            try:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception:
                pass

    def _write_files_to_sandbox(self, files: List[Dict[str, str]], sandbox: Path) -> None:
        """Write code files to the sandbox directory."""
        for f in files:
            filepath = sandbox / f["path"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(f["content"], encoding="utf-8")

    def _find_entry_point(self, files: List[Dict[str, str]]) -> Optional[str]:
        """Find the best entry point file to execute."""
        paths = [f["path"] for f in files]
        for candidate in ["train.py", "main.py", "run.py"]:
            if candidate in paths:
                return candidate
        # Fall back to first .py file that's not model.py or utils.py
        for p in paths:
            if p.endswith(".py") and p not in ("model.py", "utils.py", "requirements.txt"):
                return p
        # Absolute last resort
        for p in paths:
            if p.endswith(".py"):
                return p
        return None

    def _run_in_sandbox(self, sandbox: Path, entry: str) -> Dict[str, Any]:
        """Execute a Python file in the sandbox."""
        try:
            result = subprocess.run(
                [sys.executable, entry],
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=self._execution_timeout,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self._execution_timeout}s",
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }

    # ─── Docker Execution ─────────────────────────────

    def _execute_in_docker(self, files: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute code in a Docker container for full isolation."""
        sandbox_dir = tempfile.mkdtemp(prefix="r2t_docker_")
        sandbox_path = Path(sandbox_dir)

        try:
            self._write_files_to_sandbox(files, sandbox_path)
            entry = self._find_entry_point(files)
            if not entry:
                return {"success": False, "error": "No entry point", "attempts": 0}

            # Write a Dockerfile
            dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true
CMD ["python", "{entry}"]
"""
            (sandbox_path / "Dockerfile").write_text(dockerfile, encoding="utf-8")

            # Build image
            tag = "r2t-sandbox:latest"
            build_result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                cwd=str(sandbox_path),
                capture_output=True, text=True, timeout=DOCKER_BUILD_TIMEOUT,
            )
            if build_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Docker build failed: {build_result.stderr[:300]}",
                    "attempts": 0,
                }

            # Run container
            run_result = subprocess.run(
                ["docker", "run", "--rm", "--network=none",
                 f"--memory={DOCKER_MEMORY_LIMIT}", f"--cpus={DOCKER_CPU_LIMIT}",
                 tag],
                capture_output=True, text=True,
                timeout=self._execution_timeout,
            )

            return {
                "success": run_result.returncode == 0,
                "attempts": 1,
                "stdout": run_result.stdout[:1000],
                "stderr": run_result.stderr[:500] if run_result.returncode != 0 else "",
                "entry_point": entry,
                "isolation": "docker",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Docker execution timed out after {self._execution_timeout}s",
                "attempts": 1,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "attempts": 0}
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            # Clean up docker image
            try:
                subprocess.run(["docker", "rmi", "r2t-sandbox:latest"],
                               capture_output=True, timeout=10)
            except Exception:
                pass

    # ─── LLM Self-Heal ────────────────────────────────

    def _llm_fix(self, files: List[Dict[str, str]], error_text: str) -> List[Dict[str, str]]:
        """Use LLM to fix code that failed execution."""
        try:
            import ollama
        except ImportError:
            return files

        try:
            # Resolve available model
            model = self._resolve_model(ollama)
            if not model:
                return files

            files_text = "\n\n".join(
                f"=== {f['path']} ===\n{f['content']}" for f in files
            )

            prompt = f"""The following Python files failed to execute. Fix the errors and return the corrected files.

ERROR:
{error_text[:2000]}

FILES:
{files_text[:6000]}

Return ONLY a JSON array of objects with 'path' and 'content' keys. No markdown, no explanation."""

            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": SELF_HEAL_TEMPERATURE},
            )

            import json
            content = resp.get("message", {}).get("content", "").strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))

            arr = json.loads(content)
            fixed = []
            for it in arr:
                p = it.get("path")
                c = it.get("content")
                if p and c is not None:
                    fixed.append({"path": p, "content": c})
            return fixed if fixed else files
        except Exception as e:
            logger.warning(f"[Validator] LLM fix failed: {e}")
            return files

    def _resolve_model(self, ollama_module: Any) -> Optional[str]:
        """Find an available Ollama model."""
        try:
            available = ollama_module.list()
            if isinstance(available, dict):
                models = available.get("models", [])
                if models:
                    return models[0].get("name")
        except Exception:
            pass
        return None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Validate generated code via static analysis and sandbox execution",
            "checks": [
                "AST syntax validation",
                "Import resolution",
                "Code quality warnings",
                "Sandbox execution",
                "Self-heal loop (LLM-based fix + retry)",
                "Docker isolation (when available)",
            ],
            "docker_available": self._docker_available,
            "max_heal_attempts": self._max_heal_attempts,
        }
