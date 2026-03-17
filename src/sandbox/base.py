"""
Base sandbox interface for code execution isolation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    memory_used_mb: Optional[float] = None
    cpu_time_sec: Optional[float] = None


class BaseSandbox(ABC):
    """
    Abstract base class for code execution sandboxes.

    Provides isolation for running untrusted/generated code:
    - Filesystem isolation (temporary directories)
    - Resource limits (memory, CPU time)
    - Network restrictions
    - Import restrictions
    """

    def __init__(
        self,
        memory_limit_mb: int = 1024,
        cpu_time_limit_sec: int = 90,
        network_access: bool = False
    ):
        """
        Initialize sandbox with resource limits.

        Args:
            memory_limit_mb: Maximum memory allowed (MB)
            cpu_time_limit_sec: Maximum CPU time allowed (seconds)
            network_access: Whether to allow network access
        """
        self.memory_limit_mb = memory_limit_mb
        self.cpu_time_limit_sec = cpu_time_limit_sec
        self.network_access = network_access

    @abstractmethod
    def run(
        self,
        code_files: Dict[str, str],
        entry_point: str = "train.py",
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """
        Run code in sandbox.

        Args:
            code_files: Dict mapping filenames to file contents
            entry_point: Main file to execute
            timeout: Execution timeout (uses default if None)

        Returns:
            SandboxResult with execution details
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
        return False

    def _cleanup(self):
        """Clean up resources. Override in subclass."""
        pass
