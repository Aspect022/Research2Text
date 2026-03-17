"""
Sandbox module for secure code execution.

Provides platform-specific sandboxing:
- Windows: WindowsSandbox (Job Objects + restricted Python)
- Linux: LinuxSandbox (Firejail)
"""

import platform
from typing import Union

from .base import SandboxResult, BaseSandbox

# Platform-specific imports
if platform.system() == "Windows":
    from .windows_sandbox import WindowsSandbox
    DefaultSandbox = WindowsSandbox
else:
    from .linux_sandbox import LinuxSandbox
    DefaultSandbox = LinuxSandbox

__all__ = [
    "SandboxResult",
    "BaseSandbox",
    "DefaultSandbox",
    "WindowsSandbox",
    "LinuxSandbox",
]
