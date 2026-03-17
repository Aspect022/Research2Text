"""
Windows-specific sandbox implementation using:
1. Windows Job Objects (resource limits)
2. Restricted Python (limited builtins)
3. Temporary directories (filesystem isolation)
4. Subprocess with restricted environment
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List
import ctypes
from ctypes import wintypes

from .base import BaseSandbox, SandboxResult

logger = logging.getLogger(__name__)


class WindowsSandbox(BaseSandbox):
    """
    Windows-compatible code execution sandbox.

    Uses multiple isolation layers:
    - Temporary directory (filesystem isolation)
    - Restricted Python (limited builtins, controlled imports)
    - Windows Job Objects (memory/CPU limits)
    - Environment restrictions (no PYTHONPATH, isolated mode)
    """

    # Allowed Python modules for generated code
    ALLOWED_MODULES = {
        # ML frameworks
        'torch', 'torchvision', 'torch.nn', 'torch.optim', 'torch.utils',
        'torchvision.datasets', 'torchvision.transforms',
        'torchvision.models', 'torch.nn.functional',

        # Scientific computing
        'numpy', 'numpy as np', 'pandas', 'pandas as pd',
        'sklearn', 'sklearn.metrics', 'sklearn.model_selection',
        'sklearn.preprocessing', 'sklearn.datasets',

        # Visualization
        'matplotlib', 'matplotlib.pyplot', 'matplotlib.pyplot as plt',
        'seaborn', 'seaborn as sns',

        # Standard library
        'json', 'math', 'random', 'collections', 'itertools',
        'typing', 'dataclasses', 'pathlib', 'time', 'datetime',
        'os', 'sys', 're', 'string', 'hashlib', 'copy', 'pickle',
        'warnings', 'functools', 'inspect', 'types', 'enum',
        'abc', 'contextlib', 'io', 'csv', 'xml', 'html',

        # Utilities
        'tqdm', 'tqdm.auto', 'tqdm.auto as tqdm',
    }

    # Dangerous builtins to remove
    DANGEROUS_BUILTINS = ['eval', 'exec', 'compile', '__import__', 'open']

    def __init__(
        self,
        memory_limit_mb: int = 1024,
        cpu_time_limit_sec: Optional[int] = None,
        network_access: bool = False
    ):
        super().__init__(memory_limit_mb, cpu_time_limit_sec, network_access)
        self._job_handle = None

    def _create_job_object(self) -> Optional[int]:
        """
        Create Windows Job Object for resource limiting.

        Returns:
            Job handle or None if creation failed
        """
        try:
            kernel32 = ctypes.windll.kernel32

            # Create job object
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                logger.warning("Failed to create job object")
                return None

            # Set extended limits
            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_ulonglong),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", ctypes.c_byte * 48),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            # Configure limits
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.ProcessMemoryLimit = self.memory_limit_mb * 1024 * 1024
            info.JobMemoryLimit = self.memory_limit_mb * 1024 * 1024

            # Set limit flags
            # JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
            # JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
            # JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x8
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            info.BasicLimitInformation.LimitFlags = 0x3108
            info.BasicLimitInformation.ActiveProcessLimit = 1

            # Apply limits
            result = kernel32.SetInformationJobObject(
                job,
                9,  # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info)
            )

            if result == 0:
                logger.warning("Failed to set job object limits")
                kernel32.CloseHandle(job)
                return None

            return job

        except Exception as e:
            logger.error(f"Error creating job object: {e}")
            return None

    def _assign_process_to_job(self, process_handle: int, job_handle: int) -> bool:
        """Assign a process to a job object."""
        try:
            kernel32 = ctypes.windll.kernel32
            result = kernel32.AssignProcessToJobObject(job_handle, process_handle)
            return result != 0
        except Exception as e:
            logger.error(f"Failed to assign process to job: {e}")
            return False

    def _cleanup(self):
        """Clean up job object."""
        if self._job_handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self._job_handle)
            except Exception as e:
                logger.error(f"Error closing job handle: {e}")
            finally:
                self._job_handle = None

    def run(
        self,
        code_files: Dict[str, str],
        entry_point: str = "train.py",
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """
        Run code in Windows sandbox.

        Args:
            code_files: Dict of {filename: content}
            entry_point: Main file to run
            timeout: Execution timeout

        Returns:
            SandboxResult with execution details
        """
        import time
        start_time = time.time()

        # Use default timeout if not specified (None = no timeout)
        if timeout is None:
            timeout = self.cpu_time_limit_sec if self.cpu_time_limit_sec is not None else 3600 if self.cpu_time_limit_sec is not None else 0
            # If still None/0, use a very large timeout (effectively no limit)
            if timeout is None or timeout == 0:
                timeout = 3600  # 1 hour max

        # Create temporary directory for isolation
        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)

            try:
                # Write code files
                for filename, content in code_files.items():
                    file_path = sandbox_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")

                # Create restricted runner
                runner_path = self._create_restricted_runner(sandbox_path)

                # Execute with restrictions
                result = self._execute_restricted(
                    runner_path,
                    sandbox_path,
                    timeout
                )

                execution_time = time.time() - start_time

                return SandboxResult(
                    success=result.returncode == 0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    execution_time=execution_time
                )

            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    returncode=-1,
                    execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Sandbox execution error: {e}")
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Sandbox error: {str(e)}",
                    returncode=-1,
                    execution_time=execution_time
                )

    def _create_restricted_runner(self, sandbox_path: Path) -> Path:
        """
        Create a restricted Python runner script.

        This script runs in the sandbox and:
        1. Restricts imports to allowed modules
        2. Removes dangerous builtins
        3. Sets up safe execution environment
        """
        allowed_modules_str = ', '.join(f'"{m}"' for m in self.ALLOWED_MODULES)
        dangerous_builtins_str = ', '.join(f'"{b}"' for b in self.DANGEROUS_BUILTINS)

        runner_code = f'''"""
Restricted Python runner for sandboxed code execution.
"""
import sys
import builtins

# List of allowed modules
ALLOWED_MODULES = {{{allowed_modules_str}}}

# Dangerous builtins to remove
DANGEROUS_BUILTINS = [{dangerous_builtins_str}]

class RestrictedImportFinder:
    """Import hook to restrict module loading."""

    def find_module(self, fullname, path=None):
        # Check if module is allowed
        base_module = fullname.split('.')[0]
        if base_module not in ALLOWED_MODULES and fullname not in ALLOWED_MODULES:
            raise ImportError(
                f"Module '{{fullname}}' is not allowed in sandbox. "
                f"Allowed modules: torch, torchvision, numpy, pandas, sklearn, "
                f"matplotlib, json, math, random, collections, typing, etc."
            )
        return None

# Install import hook
sys.meta_path.insert(0, RestrictedImportFinder())

# Remove dangerous builtins
for name in DANGEROUS_BUILTINS:
    if hasattr(builtins, name):
        delattr(builtins, name)

# Restrict file operations
_original_open = open if hasattr(builtins, 'open') else None

def _restricted_open(file, mode='r', *args, **kwargs):
    """Restricted file open - only allow read mode."""
    if 'w' in mode or 'a' in mode or 'x' in mode:
        raise IOError(f"Write mode '{{mode}}' is not allowed in sandbox")
    if _original_open:
        return _original_open(file, mode, *args, **kwargs)
    raise IOError("File operations restricted")

if _original_open:
    builtins.open = _restricted_open

# Set resource limits if on Unix
if hasattr(sys, 'setrecursionlimit'):
    sys.setrecursionlimit(1000)  # Prevent stack overflow

# Run the actual code
if __name__ == "__main__":
    try:
        with open("{self._escape_string(entry_point)}", "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, {{"__name__": "__main__"}})
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
'''
        runner_path = sandbox_path / "_restricted_runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")
        return runner_path

    def _escape_string(self, s: str) -> str:
        """Escape string for embedding in code."""
        return s.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

    def _execute_restricted(
        self,
        runner_path: Path,
        sandbox_path: Path,
        timeout: Optional[int] = 3600
    ) -> subprocess.CompletedProcess:
        """
        Execute Python with restrictions.

        Uses:
        - Isolated mode (-I)
        - No site packages (-S)
        - Clean environment
        """
        # Build command
        cmd = [
            sys.executable,
            "-I",  # Isolated mode (no user site, no PYTHONPATH)
            "-S",  # No site module
            "-O",  # Optimized (no asserts)
            str(runner_path)
        ]

        # Set up restricted environment
        env = os.environ.copy()
        env["PYTHONPATH"] = ""  # No external packages
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        # Remove potentially dangerous env vars
        dangerous_env = [
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
            "GOOGLE_APPLICATION_CREDENTIALS", "GEMINI_API_KEY",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN", "DOCKER_AUTH_CONFIG"
        ]
        for key in dangerous_env:
            env.pop(key, None)

        # Create job object for resource limiting
        job_handle = self._create_job_object()

        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(sandbox_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows specific
            )

            # Assign to job object if created
            if job_handle:
                self._assign_process_to_job(process.pid, job_handle)

            # Wait for completion with timeout
            stdout, stderr = process.communicate(timeout=timeout)

            # Create result object
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )

            return result

        except subprocess.TimeoutExpired:
            # Kill the process
            process.kill()
            process.wait()
            raise

        finally:
            # Cleanup job object
            if job_handle:
                ctypes.windll.kernel32.CloseHandle(job_handle)


class LinuxSandbox(BaseSandbox):
    """
    Linux sandbox using Firejail (if available) or restricted Python.
    """

    def __init__(
        self,
        memory_limit_mb: int = 1024,
        cpu_time_limit_sec: Optional[int] = None,
        network_access: bool = False
    ):
        super().__init__(memory_limit_mb, cpu_time_limit_sec, network_access)
        self._use_firejail = shutil.which("firejail") is not None

        if self._use_firejail:
            logger.info("Using Firejail for sandboxing")
        else:
            logger.info("Firejail not available, using restricted Python")

    def run(
        self,
        code_files: Dict[str, str],
        entry_point: str = "train.py",
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """Run code in Linux sandbox."""
        import time
        start_time = time.time()

        if timeout is None:
            timeout = self.cpu_time_limit_sec if self.cpu_time_limit_sec is not None else 3600

        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)

            try:
                # Write code files
                for filename, content in code_files.items():
                    file_path = sandbox_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")

                if self._use_firejail:
                    result = self._run_firejail(sandbox_path, entry_point, timeout)
                else:
                    result = self._run_restricted(sandbox_path, entry_point, timeout)

                execution_time = time.time() - start_time

                return SandboxResult(
                    success=result.returncode == 0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    execution_time=execution_time
                )

            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    returncode=-1,
                    execution_time=execution_time
                )

    def _run_firejail(
        self,
        sandbox_path: Path,
        entry_point: str,
        timeout: Optional[int] = 3600
    ) -> subprocess.CompletedProcess:
        """Run with Firejail."""
        cmd = [
            "firejail",
            "--noprofile",
            f"--private={sandbox_path}",
            "--net=none" if not self.network_access else "",
            f"--rlimit-cpu={timeout}" if timeout else "",
            f"--rlimit-as={self.memory_limit_mb * 1024 * 1024}",
            sys.executable,
            "-I", "-S",
            entry_point
        ]

        # Remove empty strings
        cmd = [c for c in cmd if c]

        firejail_timeout = timeout + 10 if timeout else 3610
        return subprocess.run(
            cmd,
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=firejail_timeout
        )

    def _run_restricted(
        self,
        sandbox_path: Path,
        entry_point: str,
        timeout: Optional[int] = 3600
    ) -> subprocess.CompletedProcess:
        """Run with restricted Python (no Firejail)."""
        # Similar to Windows restricted runner
        cmd = [
            sys.executable,
            "-I", "-S", "-O",
            str(sandbox_path / entry_point)
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = ""

        return subprocess.run(
            cmd,
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
