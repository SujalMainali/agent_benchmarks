from .calculator import calculator
from .document_search import document_search
from .note_lookup import note_lookup
from .web_search import web_search
from .registry import (
    TOOL_PROFILES,
    TOOL_REGISTRY,
    get_default_tools,
    get_tool_profile,
    get_tools,
    get_tools_by_name,
    register_tool,
)

TOOLS = [calculator, document_search, note_lookup, web_search]

__all__ = [
    "calculator",
    "document_search",
    "note_lookup",
    "web_search",
    "TOOLS",
    "TOOL_REGISTRY",
    "TOOL_PROFILES",
    "get_default_tools",
    "get_tool_profile",
    "get_tools",
    "get_tools_by_name",
    "register_tool",
]
