"""Shared base for providers backed by a LangChain chat model.

Both the Hugging Face and OpenAI providers wrap a LangChain ``BaseChatModel``
(`ChatHuggingFace` / `ChatOpenAI`), which share the ``bind_tools`` +
``invoke`` -> ``AIMessage`` contract. This base centralizes:

* binding tools once and reusing the bound model,
* invoking with or without tools,
* normalizing the returned ``AIMessage`` into :class:`LLMResponse`.

Concrete providers only need to build the underlying chat model and declare
their name/model id. ``langchain_core`` is the repo's neutral message currency
(memory and the runtime both speak ``BaseMessage``), so importing it here does
NOT violate the "no provider SDK in the agent" rule — the provider *SDKs*
(`langchain_huggingface`, `langchain_openai`) are imported only by the concrete
provider subclasses.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import BaseMessage

from ..base import LLMProvider, LLMResponse, LLMUsage, ToolCall


class LangChainChatProvider(LLMProvider):
    """Provider adapter over a LangChain chat model."""

    def __init__(self, chat_model: Any, model: str, name: str) -> None:
        self.name = name
        self._model = model
        self._chat_model = chat_model
        # Bind tools lazily and cache per tool-set identity so we don't rebind
        # on every call.
        self._bound_cache: Dict[int, Any] = {}

    # -- public API ---------------------------------------------------------

    def invoke(
        self,
        messages: Iterable[Any],
        tools: Optional[Iterable[Any]] = None,
        response_format: Optional[Any] = None,
    ) -> LLMResponse:
        message_list = list(messages)
        model = self._resolve_model(tools)
        result = model.invoke(message_list)
        return self._normalize(result)

    def close(self) -> None:
        return None

    # -- internals ----------------------------------------------------------

    def _resolve_model(self, tools: Optional[Iterable[Any]]) -> Any:
        """Return the base chat model, or a tool-bound variant when tools given."""
        if not tools:
            return self._chat_model

        tool_list = list(tools)
        if not tool_list:
            return self._chat_model

        key = id(tool_list) if not _is_cacheable(tool_list) else _tool_key(tool_list)
        cached = self._bound_cache.get(key)
        if cached is None:
            cached = self._chat_model.bind_tools(tool_list)
            self._bound_cache[key] = cached
        return cached

    def _normalize(self, message: BaseMessage) -> LLMResponse:
        text = self._content_text(message)
        tool_calls = self._extract_tool_calls(message)
        usage = LLMUsage.from_metadata(getattr(message, "usage_metadata", None))
        model = self._resolve_model_name(message)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            provider=self.name,
            model=model,
            usage=usage,
            message=message,
            raw=message,
        )

    def _content_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def _extract_tool_calls(self, message: BaseMessage) -> List[ToolCall]:
        raw_calls = getattr(message, "tool_calls", None) or []
        normalized: List[ToolCall] = []
        for index, call in enumerate(raw_calls):
            if isinstance(call, dict):
                name = str(call.get("name", ""))
                args = call.get("args", {})
                call_id = call.get("id") or ""
            else:  # object-style tool call
                name = str(getattr(call, "name", ""))
                args = getattr(call, "args", {})
                call_id = getattr(call, "id", "") or ""
            if not isinstance(args, dict):
                args = {}
            normalized.append(
                ToolCall(name=name, arguments=args, id=str(call_id) or f"{name}_{index}")
            )
        return normalized

    def _resolve_model_name(self, message: BaseMessage) -> str:
        response_metadata = getattr(message, "response_metadata", None) or {}
        for key in ("model_name", "model", "model_id"):
            value = response_metadata.get(key)
            if value:
                return str(value)
        return self._model


def _is_cacheable(tool_list: List[Any]) -> bool:
    return all(hasattr(tool, "name") for tool in tool_list)


def _tool_key(tool_list: List[Any]) -> int:
    return hash(tuple(getattr(tool, "name", id(tool)) for tool in tool_list))
