"""ToolSandbox environment — a main-process snapshot holder.

Under the process-split architecture the real ToolSandbox execution and all
world-state mutation happen in the isolated worker process (see ``worker.py``).
The main process never imports ``tool_sandbox`` and never runs the official
engine, so this environment no longer owns a live ``ExecutionContext``.

What remains here is a thin wrapper over the shared
:class:`~benchmarks.common.interfaces.BenchmarkEnvironment` contract that:

* holds the last known :class:`EnvironmentState`,
* serializes a worker-returned world state into the shared format, and
* exposes ``observe`` / ``snapshot`` over that state.

``step`` intentionally raises — stepping the world happens in the worker, driven
by the runner via :class:`~benchmarks.toolsandbox.official_bridge.ToolSandboxClient`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmarks.common.interfaces import BenchmarkEnvironment
from benchmarks.common.models import (
    Action,
    EnvironmentState,
    Episode,
    Milestone,
    Minefield,
    Observation,
)


class ToolSandboxEnvironment(BenchmarkEnvironment):
    """Main-process snapshot wrapper around worker-returned world state."""

    def __init__(self, official_root: str = "third_party/ToolSandbox-official") -> None:
        self.official_root = official_root
        self._episode: Optional[Episode] = None
        self._state: Optional[EnvironmentState] = None

    # -- BenchmarkEnvironment ----------------------------------------------

    def reset(self, episode: Episode) -> EnvironmentState:
        """Initialize an empty snapshot for a scenario (no live execution)."""
        self._episode = episode
        self._state = EnvironmentState(
            episode_id=episode.episode_id,
            done=False,
            world_state={},
            allowed_tools=list(episode.metadata.get("tool_allow_list", [])),
            milestones=self._milestones(episode),
            minefields=self._minefields(episode),
            turn_index=0,
            metadata={
                "scenario_name": episode.metadata.get("scenario_name"),
                "categories": episode.metadata.get("categories", []),
            },
        )
        return self._state

    def observe(self) -> Observation:
        if self._episode is None:
            raise RuntimeError("Environment must be reset before observe().")
        world_state = self._state.world_state if self._state else {}
        return Observation(
            episode_id=self._episode.episode_id,
            text=self._episode.question,
            world_state_snapshot=world_state,
            available_tools=list(self._episode.metadata.get("tool_allow_list", [])),
            speaker="user",
            recipient="agent",
            metadata={
                "scenario_name": self._episode.metadata.get("scenario_name"),
                "mode": self._episode.mode,
            },
        )

    def step(self, action: Action) -> EnvironmentState:
        raise NotImplementedError(
            "ToolSandbox execution happens in the worker process; "
            "drive scenarios via ToolSandboxRunner/ToolSandboxClient."
        )

    def snapshot(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before snapshot().")
        return self._state

    def is_done(self) -> bool:
        return bool(self._state.done) if self._state else False

    # -- snapshot ingestion -------------------------------------------------

    def load_snapshot(
        self,
        episode: Episode,
        world_state: Dict[str, Any],
        official_eval: Optional[Dict[str, Any]] = None,
        turn_index: int = 0,
    ) -> EnvironmentState:
        """Wrap a worker-returned world state into an EnvironmentState."""
        official_eval = official_eval or {}
        self._episode = episode
        matched = set((official_eval.get("milestone_mapping") or {}).keys())
        tripped = set((official_eval.get("minefield_mapping") or {}).keys())
        self._state = EnvironmentState(
            episode_id=episode.episode_id,
            done=True,
            world_state=world_state or {},
            allowed_tools=list(episode.metadata.get("tool_allow_list", [])),
            milestones=self._milestones(episode, matched),
            minefields=self._minefields(episode, tripped),
            turn_index=turn_index,
            metadata={
                "scenario_name": episode.metadata.get("scenario_name"),
                "categories": episode.metadata.get("categories", []),
                "official_eval": official_eval,
            },
        )
        return self._state

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _milestones(episode: Episode, matched: Optional[set] = None) -> List[Milestone]:
        matched = matched or set()
        count = int(episode.metadata.get("milestone_count", 0))
        name = episode.episode_id
        return [
            Milestone(
                milestone_id=f"{name}::milestone::{i}",
                description="ToolSandbox milestone",
                kind="state_change",
                satisfied=str(i) in matched or i in matched,
            )
            for i in range(count)
        ]

    @staticmethod
    def _minefields(episode: Episode, tripped: Optional[set] = None) -> List[Minefield]:
        tripped = tripped or set()
        count = int(episode.metadata.get("minefield_count", 0))
        name = episode.episode_id
        return [
            Minefield(
                minefield_id=f"{name}::minefield::{i}",
                description="ToolSandbox minefield",
                kind="state_change",
                tripped=str(i) in tripped or i in tripped,
            )
            for i in range(count)
        ]
