"""Scripted user simulator for ToolSandbox.

Per the integration plan, the user simulator plays back the scenario's user
messages rather than becoming a second fully tool-using agent. It does not use
the benchmark tools and never mutates world state — its only job is to supply
the next scripted user utterance (and, in the official engine, to end the
conversation once the script is exhausted).

A scenario object only carries its *initial* user message; the official
benchmark generates follow-up turns with an LLM user simulator. This scripted
simulator therefore faithfully drives single-user-turn scenarios and yields no
further turns for multi-turn ones (the official engine then ends the
conversation). For full multi-turn fidelity, the runner can be pointed at the
official user simulator instead (see ``official_bridge.build_official_user_role``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmarks.common.models import ScenarioSpec


class ToolSandboxUserSimulator:
    """Replays scripted user turns for a scenario."""

    def __init__(self) -> None:
        self._utterances: List[str] = []
        self._index = 0

    def reset(self, scenario: ScenarioSpec | Dict[str, Any]) -> None:
        """Reset the simulator for a new scenario.

        The scenario's seed messages already seed the *first* user turn into the
        environment, so any additional scripted follow-ups are queued here. In
        practice ToolSandbox scenarios provide no scripted follow-ups, so this
        typically leaves the queue empty.
        """
        self._utterances = list(self._extract_followups(scenario))
        self._index = 0

    def next_message(self) -> Optional[str]:
        """Return the next scripted user message, or ``None`` when exhausted."""
        if self._index >= len(self._utterances):
            return None
        message = self._utterances[self._index]
        self._index += 1
        return message

    def is_done(self) -> bool:
        """True when there are no more scripted user turns to play."""
        return self._index >= len(self._utterances)

    def pending_utterances(self) -> List[str]:
        """Return the not-yet-played scripted utterances (for the role builder)."""
        return list(self._utterances[self._index :])

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _extract_followups(scenario: ScenarioSpec | Dict[str, Any]) -> List[str]:
        """Pull any explicitly scripted follow-up user turns from metadata.

        Recognizes an optional ``scripted_user_turns`` list in the scenario
        metadata, which lets a caller supply deterministic follow-ups. Absent
        that, returns an empty list.
        """
        if isinstance(scenario, ScenarioSpec):
            metadata = scenario.metadata
        elif isinstance(scenario, dict):
            metadata = scenario.get("metadata", scenario)
        else:
            metadata = {}
        followups = metadata.get("scripted_user_turns", []) if metadata else []
        return [str(turn) for turn in followups]
