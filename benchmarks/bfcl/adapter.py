"""BFCL adapter — ``Episode`` -> runtime input (system_prompt, messages, tools).

Pure input transformation, mirroring the LoCoMo adapter's role in the shared
architecture. The adapter never executes tools, never evaluates, and never
parses model output.

Tools: BFCL function docs (Gorilla JSON-schema style) are wrapped as
langchain ``StructuredTool`` objects with inert stub executors, so the agent's
normal tool-binding mechanism advertises exactly the entry's functions. BFCL
single-turn scoring only inspects the *requested* tool calls, so the stub
result is never seen by the checker.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from benchmarks.common.interfaces import BenchmarkAdapter
from benchmarks.common.models import BenchmarkSample, Episode

# BFCL/Gorilla parameter types -> JSON-schema types accepted by tool binding.
# Same normalization direction as the official ``GORILLA_TO_OPENAPI`` mapping
# (which the official handlers apply in ``convert_to_tool``).
_GORILLA_TO_JSON_SCHEMA = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "bigint": "integer",
}

_TOOL_STUB_RESULT = "None"


def _make_stub(function_name: str):
    """Return an inert executor for a BFCL function doc."""

    def _stub(**_kwargs: Any) -> str:
        return _TOOL_STUB_RESULT

    _stub.__name__ = re.sub(r"\W", "_", function_name) or "bfcl_tool"
    return _stub


def _sanitize_schema(node: Any) -> Any:
    """Recursively map Gorilla types onto JSON-schema types."""
    if isinstance(node, dict):
        sanitized = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                sanitized[key] = _GORILLA_TO_JSON_SCHEMA.get(value, value)
            else:
                sanitized[key] = _sanitize_schema(value)
        return sanitized
    if isinstance(node, list):
        return [_sanitize_schema(item) for item in node]
    return node


class BFCLAdapter(BenchmarkAdapter):
    """Converts BFCL episodes into runtime inputs. Transformation only."""

    def load_sample(self, sample_data: Dict[str, Any]) -> BenchmarkSample:
        """Compatibility shim for the shared interface (loader owns loading)."""
        from .loader import BFCLLoader

        episode = BFCLLoader().load(sample_data)
        return BenchmarkSample.from_episode(episode)

    def build_context_messages(self, sample: BenchmarkSample | Episode) -> List[BaseMessage]:
        """Replay the entry's conversation turns as langchain messages.

        System messages embedded in the entry are returned in place so the
        runtime seeds them into memory; the final user utterance is NOT
        included here — it is delivered via the observation (same split the
        LoCoMo flow uses).
        """
        raw = self._raw_entry(sample)
        messages: List[BaseMessage] = []
        flattened = [
            message
            for turn in raw.get("question", [])
            if isinstance(turn, list)
            for message in turn
            if isinstance(message, dict)
        ]
        # Drop the final user message: it becomes the observation text.
        last_user_index = max(
            (i for i, m in enumerate(flattened) if m.get("role") == "user"),
            default=None,
        )
        for index, message in enumerate(flattened):
            if index == last_user_index:
                continue
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def build_agent_input(self, sample: BenchmarkSample | Episode) -> Dict[str, Any]:
        """Produce exactly ``system_prompt``, ``messages``, and ``tools``."""
        raw = self._raw_entry(sample)
        return {
            "system_prompt": self._system_prompt(raw),
            "messages": self.build_context_messages(sample),
            "tools": self.build_tools(sample),
        }

    def build_tools(self, sample: BenchmarkSample | Episode) -> List[StructuredTool]:
        """Wrap the entry's function docs as langchain tools.

        The tool names/schemas mirror the official docs so the model sees the
        same functions the official FC pipeline would advertise.
        """
        raw = self._raw_entry(sample)
        tools: List[StructuredTool] = []
        for function_doc in raw.get("function", []) or []:
            name = str(function_doc.get("name", ""))
            if not name:
                continue
            # Official ``convert_to_tool`` behavior: OpenAI-style APIs reject
            # dots in function names (^[a-zA-Z0-9_-]{1,64}$), so dots become
            # underscores. The official checker undoes/expects this via the
            # checker persona's ``underscore_to_dot`` flag, so generation and
            # evaluation stay consistent.
            name = re.sub(r"\.", "_", name)
            parameters = _sanitize_schema(function_doc.get("parameters", {})) or {
                "type": "object",
                "properties": {},
            }
            tools.append(
                StructuredTool(
                    name=name,
                    description=str(function_doc.get("description", "")),
                    args_schema=parameters,
                    func=_make_stub(name),
                )
            )
        return tools

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _raw_entry(sample: BenchmarkSample | Episode) -> Dict[str, Any]:
        if isinstance(sample, Episode):
            return sample.raw_data
        return sample.context.get("raw_fields", {})

    @staticmethod
    def _system_prompt(raw: Dict[str, Any]) -> str:
        """Extract the entry's own system prompt, if any.

        FC-mode entries usually carry no system prompt (tools are delivered
        natively); when one exists in the first turn, surface it so the
        runtime can seed the agent with it.
        """
        question = raw.get("question", [])
        if question and isinstance(question[0], list):
            for message in question[0]:
                if isinstance(message, dict) and message.get("role") == "system":
                    return str(message.get("content", ""))
        return ""
