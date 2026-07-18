"""Reference AgentDriver for the in-repo ResearchHelperAgent.

Reuses ``src/agent.py`` and ``src/runtime.py`` unchanged — this module only
centralizes construction, replacing the four per-benchmark ``setup_agent``
duplicates.
"""

from __future__ import annotations

from typing import Any, Dict

from benchmarks.common.driver import AgentDriver, RuntimeSpec
from benchmarks.common.interfaces import AgentRuntime

from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.llm import build_provider
from src.runtime import ResearchHelperAgentRuntime
from src.tools.calculator import calculator
from src.tools.document_search import document_search
from src.tools.note_lookup import note_lookup
from src.tools.web_search import web_search

#: The agent's own toolset, used when a benchmark supplies no tools
#: (spec.tools is None — e.g. LoCoMo, LongMemEval).
DEFAULT_TOOLS = [calculator, document_search, note_lookup, web_search]


class ResearchHelperDriver(AgentDriver):
    """Builds ResearchHelperAgent runtimes for any benchmark spec."""

    name = "research_helper"

    def __init__(self) -> None:
        self._settings = load_settings()
        # One LLM provider for the whole run; reused across per-entry /
        # per-scenario create_runtime calls.
        self._llm = build_provider(self._settings)

    @property
    def llm(self):
        """Bare LLM provider — used by ToolSandbox ``llm_proxy`` mode."""
        return self._llm

    def create_runtime(self, spec: RuntimeSpec) -> AgentRuntime:
        tools = list(spec.tools) if spec.tools is not None else list(DEFAULT_TOOLS)
        max_steps = (
            spec.max_tool_steps
            if spec.max_tool_steps is not None
            else getattr(self._settings, "max_tool_steps", 5)
        )
        agent = ResearchHelperAgent(
            llm=self._llm,
            tools=tools,
            max_tool_steps=max_steps,
            system_prompt_override=spec.system_prompt,
            allow_tools=spec.allow_tools,
        )
        return ResearchHelperAgentRuntime(agent)

    def describe(self) -> Dict[str, Any]:
        return {
            "agent_name": self.name,
            "llm_provider": getattr(self._settings, "llm_provider", None),
            "model_id": getattr(self._settings, "model_id", None),
            "temperature": getattr(self._settings, "temperature", None),
        }
