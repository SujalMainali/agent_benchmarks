"""Provider-agnostic LLM layer for the agent.

Public surface: the normalized data types, the abstract provider, and the
factory. The agent imports from here; it must never import a concrete provider
or a provider SDK directly.
"""

from .base import (
    LLMProvider,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    ToolCall,
)
from .factory import build_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMStreamEvent",
    "LLMUsage",
    "ToolCall",
    "build_provider",
]
