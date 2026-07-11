"""Provider-agnostic LLM interface and normalized data structures.

This module is the boundary between the agent and any concrete LLM backend
(Hugging Face, OpenAI, or an OpenAI-compatible endpoint like Ollama). The agent
depends ONLY on the types defined here; it must never import a provider SDK.

The normalized types intentionally keep a `message` handle and a `raw` payload
so provider-specific objects can still flow through the agent/runtime/memory
stack (which speak `langchain_core` messages) without the agent knowing which
backend produced them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ToolCall:
    """A single tool invocation requested by the model, provider-normalized."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return the langchain-style tool-call dict the runtime expects.

        The benchmark runtime reads tool calls via ``.get("name")`` /
        ``.get("args")`` / ``.get("id")``; keep that shape stable here so the
        provider split never leaks into the benchmark contract.
        """
        return {"name": self.name, "args": dict(self.arguments), "id": self.id}


@dataclass
class LLMUsage:
    """Normalized token accounting across providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_metadata(cls, usage: Optional[Dict[str, Any]]) -> "LLMUsage":
        if not usage:
            return cls()
        input_tokens = int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        total = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
        )


@dataclass
class LLMResponse:
    """Provider-agnostic normalization of a single model response.

    Attributes:
        text: The assistant's textual content.
        tool_calls: Normalized tool calls the model requested (may be empty).
        provider: Provider name, e.g. "huggingface" or "openai".
        model: The model/repo id that produced the response.
        usage: Normalized token usage.
        message: The underlying provider message object (a langchain
            ``BaseMessage`` for the current providers). Kept opaque so the agent
            can thread it back into memory/trajectory without inspecting it.
        raw: The raw provider payload for debugging/telemetry.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    message: Any = None
    raw: Any = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def tool_calls_as_dicts(self) -> List[Dict[str, Any]]:
        """Tool calls in the langchain-style dict shape used by the runtime."""
        return [call.as_dict() for call in self.tool_calls]


@dataclass
class LLMStreamEvent:
    """Incremental streaming event (reserved for future streaming support)."""

    delta: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    done: bool = False
    raw: Any = None


class LLMProvider(ABC):
    """Abstract, provider-agnostic LLM backend.

    Concrete implementations live under ``src/llm/providers/`` and are the ONLY
    place allowed to import a provider SDK.
    """

    #: Human-readable provider name; overridden by subclasses.
    name: str = "base"

    @abstractmethod
    def invoke(
        self,
        messages: Iterable[Any],
        tools: Optional[Iterable[Any]] = None,
        response_format: Optional[Any] = None,
    ) -> LLMResponse:
        """Run one completion.

        Args:
            messages: Conversation messages (langchain ``BaseMessage`` objects).
            tools: Optional tool objects the model may call. When omitted, the
                provider must not request tool calls.
            response_format: Optional structured-output hint (provider-specific).

        Returns:
            A normalized :class:`LLMResponse`.
        """

    def stream(
        self,
        messages: Iterable[Any],
        tools: Optional[Iterable[Any]] = None,
    ) -> Iterable[LLMStreamEvent]:
        """Optional streaming interface. Default: single terminal event."""
        response = self.invoke(messages, tools=tools)
        yield LLMStreamEvent(
            delta=response.text,
            tool_calls=response.tool_calls,
            done=True,
            raw=response.raw,
        )

    def close(self) -> None:
        """Release any provider resources. Default: no-op."""
        return None
