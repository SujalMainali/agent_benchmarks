"""ToolSandbox scenario loader.

Reads normalized scenario specs from the ToolSandbox worker (via
:class:`~benchmarks.toolsandbox.official_bridge.ToolSandboxClient`) and turns
each into the shared :class:`~benchmarks.common.models.Episode` model with a
normalized :class:`~benchmarks.common.models.ScenarioSpec` attached.

The loader imports NO ``tool_sandbox`` code — it only consumes the plain-dict
specs the worker emits over stdio. Scenarios are driven later by name (the
worker owns the raw ``Scenario`` objects), so nothing non-serializable is stored
on an Episode.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from benchmarks.common.interfaces import BenchmarkLoader
from benchmarks.common.models import Episode, Milestone, Minefield, ScenarioSpec, Task

from .official_bridge import ToolSandboxClient, default_toolsandbox_python


class ToolSandboxLoader(BenchmarkLoader):
    """Loads and normalizes ToolSandbox scenario specs into shared Episodes."""

    def __init__(
        self,
        python_executable: Optional[str] = None,
        official_root: str = "third_party/ToolSandbox-official",
        client: Optional[ToolSandboxClient] = None,
    ) -> None:
        self.official_root = official_root
        self.client = client or ToolSandboxClient(
            python_executable=python_executable or default_toolsandbox_python(),
            official_root=official_root,
        )

    def load(self, raw_data: Dict[str, Any]) -> Episode:
        """Normalize one worker scenario-spec dict into an Episode."""
        return self._episode_from_spec(raw_data)

    def load_named_scenarios(self) -> List[Episode]:
        """Load every scenario spec exposed by the worker into Episodes."""
        specs = self.client.list_scenarios()
        return [self._episode_from_spec(spec) for spec in specs]

    def load_many(self, raw_items: Iterable[Dict[str, Any]]) -> List[Episode]:
        return [self.load(item) for item in raw_items]

    # -- internals ----------------------------------------------------------

    def _episode_from_spec(self, spec: Dict[str, Any]) -> Episode:
        """Build an Episode from a normalized scenario-spec dict."""
        name = spec["scenario_id"]
        milestones = [
            Milestone(
                milestone_id=f"{name}::milestone::{i}",
                description="ToolSandbox milestone",
                kind="state_change",
            )
            for i in range(spec.get("milestone_count", 0))
        ]
        minefields = [
            Minefield(
                minefield_id=f"{name}::minefield::{i}",
                description="ToolSandbox minefield",
                kind="state_change",
            )
            for i in range(spec.get("minefield_count", 0))
        ]
        scenario_spec = ScenarioSpec(
            scenario_id=name,
            name=name,
            messages=spec.get("seed_messages", []),
            tool_allow_list=spec.get("tool_allow_list", []),
            milestones=milestones,
            minefields=minefields,
            metadata={
                "categories": spec.get("categories", []),
                "tool_deny_list": spec.get("tool_deny_list", []),
                "max_messages": spec.get("max_messages", 30),
            },
        )

        question = spec.get("first_user_utterance") or name
        task = Task(
            task_id=name,
            question=question,
            gold_answer="",  # ToolSandbox is milestone-scored, not answer-matched.
            context={
                "scenario_spec": _scenario_spec_to_dict(scenario_spec),
                "tool_allow_list": spec.get("tool_allow_list", []),
                "seed_messages": spec.get("seed_messages", []),
                "categories": spec.get("categories", []),
            },
            mode="tool_sandbox",
            metadata={
                "scenario_name": name,
                "categories": spec.get("categories", []),
                "milestone_count": spec.get("milestone_count", 0),
                "minefield_count": spec.get("minefield_count", 0),
            },
        )
        return Episode(
            episode_id=name,
            task=task,
            metadata={
                "scenario_name": name,
                "categories": spec.get("categories", []),
                "max_messages": spec.get("max_messages", 30),
                "tool_allow_list": spec.get("tool_allow_list", []),
                "milestone_count": spec.get("milestone_count", 0),
                "minefield_count": spec.get("minefield_count", 0),
            },
            raw_data={"scenario_spec": _scenario_spec_to_dict(scenario_spec)},
        )


def _scenario_spec_to_dict(spec: ScenarioSpec) -> Dict[str, Any]:
    """Serialize a ScenarioSpec (with its dataclass milestones) to plain dicts."""
    return {
        "scenario_id": spec.scenario_id,
        "name": spec.name,
        "tool_allow_list": spec.tool_allow_list,
        "messages": spec.messages,
        "milestones": [
            {"milestone_id": m.milestone_id, "kind": m.kind, "description": m.description}
            for m in spec.milestones
        ],
        "minefields": [
            {"minefield_id": m.minefield_id, "kind": m.kind, "description": m.description}
            for m in spec.minefields
        ],
        "metadata": spec.metadata,
    }
