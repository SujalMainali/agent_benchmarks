"""Parent-side glue for ToolSandbox runtime mode.

Runs in the MAIN interpreter. Never imports ``tool_sandbox``. Bridges the
worker's ``agent_turn_request`` protocol to our ``ResearchHelperAgentRuntime``:

* ``WorkerToolProxy`` implements the shared ``ToolExecutionEnvironment``
  interface (its first concrete implementation). Each ``execute`` call tunnels
  to the worker, where the official ``ExecutionEnvironment`` runs the tool
  against the sandbox world state (or the worker returns a synthetic transient
  fault). No real action ever happens in this process.
* ``build_proxy_structured_tools`` wraps the worker's OpenAI tool schemas as
  langchain ``StructuredTool``s (the pattern proven in ``benchmarks/bfcl``),
  so the agent's normal tool-binding path advertises exactly the scenario's
  allow-listed tools.
* ``ToolSandboxRuntimeSession`` owns one persistent runtime per scenario, so
  agent memory spans user turns, and exposes ``agent_turn_fn`` for the client.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from benchmarks.common.interfaces import ToolExecutionEnvironment
from benchmarks.common.models import (
    Action,
    EnvironmentState,
    Episode,
    Observation,
)

from src.agent import ResearchHelperAgent
from src.runtime import ResearchHelperAgentRuntime

# Callback the client injects per turn: (name, arguments) -> (result, exception, fault).
ExecuteTool = Callable[[str, Dict[str, Any]], Tuple[str, Optional[str], bool]]


class WorkerToolProxy(ToolExecutionEnvironment):
    """Executes tool calls in the worker's official sandbox (no local action).

    First concrete implementation of the shared ``ToolExecutionEnvironment``
    interface. The mutable world state lives in the worker's ``ExecutionContext``;
    this class only forwards the call and normalizes the result shape.
    """

    def __init__(self) -> None:
        self._execute_fn: Optional[ExecuteTool] = None
        self.fault_events: List[Dict[str, Any]] = []

    def bind(self, execute_fn: ExecuteTool) -> None:
        """Bind the client's per-turn tool-execution callback."""
        self._execute_fn = execute_fn

    def execute(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any] | None]:
        if self._execute_fn is None:
            return "", {"message": "Tool proxy is not bound to a worker turn."}
        result, exception, fault = self._execute_fn(tool_name, arguments)
        if fault:
            self.fault_events.append({"tool": tool_name, "arguments": arguments})
        return result, ({"message": exception} if exception else None)


def _make_proxy_func(tool_name: str, proxy: WorkerToolProxy):
    """Build an inert-looking executor that forwards to the worker sandbox."""

    def _proxy(**kwargs: Any) -> str:
        result, exception_info = proxy.execute(tool_name, kwargs)
        if exception_info:
            # The agent already treats error strings as tool feedback and can
            # retry, so surface the failure as the tool result content.
            return f"Error: {exception_info.get('message', 'tool call failed')}"
        return result

    _proxy.__name__ = re.sub(r"\W", "_", tool_name) or "tool_sandbox_tool"
    return _proxy


def build_proxy_structured_tools(
    tool_schemas: List[Dict[str, Any]], proxy: WorkerToolProxy
) -> List[StructuredTool]:
    """Wrap the worker's OpenAI tool schemas as langchain ``StructuredTool``s."""
    tools: List[StructuredTool] = []
    for schema in tool_schemas or []:
        function = schema.get("function", schema) or {}
        name = str(function.get("name", ""))
        if not name:
            continue
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        tools.append(
            StructuredTool(
                name=name,
                description=str(function.get("description", "")),
                args_schema=parameters,
                func=_make_proxy_func(name, proxy),
            )
        )
    return tools


def _openai_message_to_langchain(message: Dict[str, Any]) -> Optional[BaseMessage]:
    """Convert one worker OpenAI-format history dict to a langchain message."""
    role = message.get("role")
    content = message.get("content") or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(message.get("tool_call_id", "")),
            name=message.get("name"),
        )
    return None


class ToolSandboxRuntimeSession:
    """One per scenario: holds the persistent runtime so memory spans turns."""

    def __init__(
        self,
        llm: Any,
        system_prompt: str,
        max_tool_steps: int,
        episode: Episode,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_tool_steps = max_tool_steps
        self._episode = episode
        self._runtime: Optional[ResearchHelperAgentRuntime] = None
        self.proxy = WorkerToolProxy()

    @property
    def runtime(self) -> Optional[ResearchHelperAgentRuntime]:
        return self._runtime

    def agent_turn_fn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        execute_tool: ExecuteTool,
    ) -> str:
        """Drive one whole agent turn; return the final answer text."""
        self.proxy.bind(execute_tool)

        # First turn: build the agent + runtime once, seeding prior context.
        if self._runtime is None:
            structured = build_proxy_structured_tools(tools, self.proxy)
            agent = ResearchHelperAgent(
                llm=self._llm,
                tools=structured,
                max_tool_steps=self._max_tool_steps,
                system_prompt_override=self._system_prompt or None,
                allow_tools=True,
            )
            self._runtime = ResearchHelperAgentRuntime(agent)
            self._runtime.reset(
                self._episode,
                EnvironmentState(
                    episode_id=self._episode.episode_id,
                    messages=self._context_messages(messages),
                ),
            )

        observation = Observation(
            episode_id=self._episode.episode_id,
            text=self._last_user_content(messages),
            metadata={"benchmark_mode": "tool_sandbox"},
        )
        action: Action = self._runtime.act(observation)
        return action.text

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _last_user_index(messages: List[Dict[str, Any]]) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return None

    def _context_messages(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """All history except the final user utterance (that is the observation)."""
        last_user = self._last_user_index(messages)
        out: List[BaseMessage] = []
        for index, message in enumerate(messages):
            if index == last_user:
                continue
            converted = _openai_message_to_langchain(message)
            if converted is not None:
                out.append(converted)
        return out

    def _last_user_content(self, messages: List[Dict[str, Any]]) -> str:
        last_user = self._last_user_index(messages)
        if last_user is None:
            return ""
        return str(messages[last_user].get("content", "") or "")
