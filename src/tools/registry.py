"""Central registry mapping tool names to tool objects (the ToolProfile layer).

This is the single source of truth for which tools exist and how a name
resolves to a concrete tool callable. It keeps the model from getting confused
about the "real" tool list: a benchmark runtime binds exactly the tools it asks
for by name, and the system prompt never advertises a different static set.

The interactive agent uses the full set (``get_default_tools``), while stateful
benchmarks such as ToolSandbox expose only a scenario-specific subset via
``get_tools_by_name(allow_list)``. Named bundles are available via
``get_tool_profile``.
"""

from __future__ import annotations

from typing import Dict, List

from .calculator import calculator
from .document_search import document_search
from .note_lookup import note_lookup
from .web_search import web_search

# Name -> tool object. Benchmarks (or any other caller) can register their own
# tools here via ``register_tool``, or maintain a separate registry and merge.
TOOL_REGISTRY: Dict[str, object] = {
    "calculator": calculator,
    "document_search": document_search,
    "note_lookup": note_lookup,
    "web_search": web_search,
}

# Named tool bundles. "default" is the full interactive set; add narrower
# profiles here as benchmarks or product surfaces need them.
TOOL_PROFILES: Dict[str, List[str]] = {
    "default": list(TOOL_REGISTRY.keys()),
    "research": ["web_search", "document_search", "note_lookup"],
    "math": ["calculator"],
}


def get_tools_by_name(names: List[str]) -> List[object]:
    """Resolve an allow-list of tool names into tool objects.

    Unknown names are skipped so a scenario referencing a tool this project
    does not implement degrades gracefully rather than raising.
    """
    return [TOOL_REGISTRY[name] for name in names if name in TOOL_REGISTRY]


def get_default_tools() -> List[object]:
    """Return every registered tool (the interactive agent's default set)."""
    return list(TOOL_REGISTRY.values())


def get_tool_profile(profile_name: str) -> List[object]:
    """Return the tool objects for a named profile.

    Falls back to the default profile when ``profile_name`` is unknown.
    """
    names = TOOL_PROFILES.get(profile_name, TOOL_PROFILES["default"])
    return get_tools_by_name(names)


def register_tool(name: str, tool: object) -> None:
    """Register (or replace) a tool by name.

    Lets benchmark packages add their own tools to the shared registry without
    editing this module.
    """
    TOOL_REGISTRY[name] = tool


# Backwards-compatible alias for the earlier name.
get_tools = get_tools_by_name
