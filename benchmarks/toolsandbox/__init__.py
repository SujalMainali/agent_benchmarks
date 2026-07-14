"""ToolSandbox benchmark integration.

A sibling benchmark package to ``benchmarks/locomo`` that plugs the official
Apple ToolSandbox engine (stateful, conversational, milestone-scored) into the
shared AgentRuntime + BenchmarkEnvironment + Evaluator pattern. All coupling to
the vendored ``tool_sandbox`` package is isolated in ``worker.py``, which runs
under a separate interpreter; the main process talks to it via ``official_bridge``.
"""

from .adapter import ToolSandboxAdapter
from .environment import ToolSandboxEnvironment
from .evaluator import ToolSandboxEvaluator
from .loader import ToolSandboxLoader
from .runner import ToolSandboxRunner
from .user_simulator import ToolSandboxUserSimulator

__all__ = [
    "ToolSandboxLoader",
    "ToolSandboxAdapter",
    "ToolSandboxEnvironment",
    "ToolSandboxRunner",
    "ToolSandboxEvaluator",
    "ToolSandboxUserSimulator",
]
