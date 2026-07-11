"""Hugging Face LLM provider.

This is the ONLY module (besides the OpenAI provider) allowed to import the
Hugging Face SDK. It wraps the current ``HuggingFaceEndpoint`` +
``ChatHuggingFace`` path that previously lived in ``app.py`` and
``benchmarks/locomo/run.py`` and adapts it to the provider-agnostic
:class:`LLMProvider` interface.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from ._langchain_base import LangChainChatProvider


class HuggingFaceProvider(LangChainChatProvider):
    """Provider backed by a remote Hugging Face inference endpoint."""

    def __init__(
        self,
        model_id: str,
        hf_token: str,
        hf_provider: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        do_sample: bool = False,
        chat_model: Optional[Any] = None,
    ) -> None:
        if chat_model is None:
            endpoint = HuggingFaceEndpoint(
                repo_id=model_id,
                task="text-generation",
                huggingfacehub_api_token=hf_token,
                provider=hf_provider,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            chat_model = ChatHuggingFace(llm=endpoint)

        super().__init__(chat_model=chat_model, model=model_id, name="huggingface")
