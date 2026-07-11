"""Provider factory: config in, :class:`LLMProvider` out.

All provider-selection logic lives here. Callers (``app.py``, the benchmark
runner) pass a :class:`src.config.Settings` object and receive a ready
provider; they never import a provider SDK or branch on the backend
themselves.

Provider SDKs are imported lazily inside each branch so that, e.g., selecting
OpenAI does not require the Hugging Face SDK to be installed and vice versa.
"""

from __future__ import annotations

from .base import LLMProvider


def build_provider(settings) -> LLMProvider:
    """Construct the configured LLM provider from ``Settings``.

    Args:
        settings: A ``src.config.Settings`` instance carrying ``llm_provider``
            and the provider-specific fields.

    Returns:
        A concrete :class:`LLMProvider`.

    Raises:
        ValueError: If ``settings.llm_provider`` is not recognized.
    """
    provider = (getattr(settings, "llm_provider", "") or "huggingface").strip().lower()

    if provider in {"huggingface", "hf"}:
        from .providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider(
            model_id=settings.model_id,
            hf_token=settings.hf_token,
            hf_provider=settings.hf_inference_provider,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            do_sample=settings.do_sample,
        )

    if provider in {"openai", "ollama", "openai_compatible"}:
        from .providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=settings.model_id,
            api_key=settings.api_key,
            base_url=settings.openai_base_url or None,
            temperature=settings.temperature,
            max_output_tokens=settings.max_new_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'huggingface' or 'openai'."
    )
