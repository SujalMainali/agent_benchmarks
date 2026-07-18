"""Generic agent-driver interface — the construction seam for benchmarks.

See DriverInterface.md for the full design. The behavior contract for the
runtime a driver builds (reset/act/get_trajectory/get_raw_messages) lives in
AgentInterface.md; this module only answers "how does a benchmark obtain a
runtime bound to its prompt/tools" without importing any concrete agent.

Benchmarks must never import ``src.*`` (or any other agent package) directly —
the only bridge is the lazy string registry below, resolved at run time via
the ``AGENT_DRIVER`` environment variable.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import AgentRuntime


@dataclass
class RuntimeSpec:
    """Everything a benchmark tells a driver about the runtime it needs.

    Attributes:
        benchmark: "locomo" | "longmemeval" | "bfcl" | "tool_sandbox".
        system_prompt: when set, MUST replace the agent's default system
            prompt (same load-bearing rule as ``system_prompt_override``).
        tools: benchmark-supplied langchain ``StructuredTool``s. ``None``
            means "no benchmark tools — the agent may use its own toolset";
            a list (possibly empty) means "advertise exactly these".
        allow_tools: ``False`` disables the tool loop entirely regardless
            of ``tools`` (e.g. LongMemEval memory QA).
        max_tool_steps: per-turn tool-loop budget. ``None`` = driver default.
        metadata: benchmark-specific extras a driver may inspect.
    """

    benchmark: str
    system_prompt: Optional[str] = None
    tools: Optional[List[Any]] = None
    allow_tools: bool = True
    max_tool_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentDriver(ABC):
    """Factory + identity for one benchmarked agent.

    Drivers are cheap, stateless-across-runs objects; cache expensive
    resources (LLM clients, sessions) inside the instance so repeated
    ``create_runtime`` calls (per BFCL entry / ToolSandbox scenario) stay
    fast.
    """

    #: Slug recorded as ``agent_name`` in results/experiment_index.csv.
    name: str = "agent"

    @abstractmethod
    def create_runtime(self, spec: RuntimeSpec) -> AgentRuntime:
        """Build a runtime honoring ``spec``.

        Called once per binding: once per run (LoCoMo/LongMemEval), per
        entry (BFCL), or per scenario (ToolSandbox). The returned runtime
        must obey the AgentInterface.md behavior contract.
        """

    def describe(self) -> Dict[str, Any]:
        """Provenance recorded in summary.json ``run_metadata``.

        Override to expose model id, provider, agent version, etc.
        """
        return {"agent_name": self.name}


#: name -> "module.path:ClassName", imported lazily so benchmark import
#: graphs never pull in agent code. External agents can skip the registry
#: entirely by setting AGENT_DRIVER to a raw "pkg.module:ClassName" path.
_REGISTRY: Dict[str, str] = {
    "research_helper": "drivers.research_helper:ResearchHelperDriver",
}


def register_driver(name: str, target: str) -> None:
    """Register a driver import path ("pkg.module:ClassName") under a name."""
    _REGISTRY[name] = target


def resolve_driver(name: Optional[str] = None) -> AgentDriver:
    """Instantiate the driver selected by ``name`` or ``AGENT_DRIVER``.

    Accepts a registry key (e.g. "research_helper") or a raw
    "pkg.module:ClassName" import path, so an external agent's driver can
    live entirely outside this repository. Defaults to "research_helper".
    """
    key = (
        name
        or os.getenv("AGENT_DRIVER", "").split("#", 1)[0].strip()
        or "research_helper"
    )
    target = _REGISTRY.get(key, key)
    if ":" not in target:
        raise ValueError(
            f"Unknown agent driver '{key}'. Registered drivers: "
            f"{sorted(_REGISTRY)}; or set AGENT_DRIVER to a full "
            f"'pkg.module:ClassName' import path."
        )
    module_path, class_name = target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    driver = getattr(module, class_name)()
    # Duck-typed on purpose: external drivers need not subclass our ABC.
    if not callable(getattr(driver, "create_runtime", None)):
        raise TypeError(
            f"Driver '{target}' does not implement create_runtime(spec); "
            f"see DriverInterface.md."
        )
    return driver
