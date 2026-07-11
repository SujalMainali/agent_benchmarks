"""OpenAI (and OpenAI-compatible) LLM provider.

This is the ONLY module (besides the Hugging Face provider) allowed to import
the OpenAI SDK path. It talks to OpenAI models — or any OpenAI-compatible
endpoint such as a local Ollama server — through LangChain's ``ChatOpenAI`` and
adapts the result to the provider-agnostic :class:`LLMProvider` interface.

Tool calls and token usage are normalized into :class:`ToolCall` / :class:`LLMUsage`
by the shared LangChain base, so the returned :class:`LLMResponse` has the same
shape as the Hugging Face provider.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI

from ._langchain_base import LangChainChatProvider


class OpenAIProvider(LangChainChatProvider):
    """Provider backed by OpenAI or an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        chat_model: Optional[Any] = None,
    ) -> None:
        if chat_model is None:
            kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            chat_model = ChatOpenAI(**kwargs)

        super().__init__(chat_model=chat_model, model=model, name="openai")
