"""Shared data structures for all benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


@dataclass
class BenchmarkSpec:
    """Descriptor for a benchmark suite."""

    name: str
    version: str = "v1"
    description: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Single task inside an episode."""

    task_id: str
    question: str
    gold_answer: str
    context: Dict[str, Any] = field(default_factory=dict)
    mode: str = "plain_qa"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Milestone:
    """A checkpoint the agent is expected to reach in a stateful benchmark.

    Modeled after ToolSandbox milestones: a milestone is satisfied when the
    world reaches an expected state (or a specific tool call / response occurs),
    possibly after its dependencies have already been satisfied.
    """

    milestone_id: str
    description: str = ""
    kind: str = "state_change"  # state_change | tool_call | response | constraint
    expected_state: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    satisfied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Minefield:
    """A state the agent must NOT reach (an anti-milestone).

    Tripping a minefield marks the trajectory as having entered a forbidden
    state, mirroring ToolSandbox's negative-evaluation semantics.
    """

    minefield_id: str
    description: str = ""
    kind: str = "state_change"  # state_change | tool_call | response | constraint
    forbidden_state: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tripped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """A point-in-time capture of the environment world state."""

    turn_index: int = 0
    world_state: Dict[str, Any] = field(default_factory=dict)
    actor: Optional[str] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    """A stateful, conversational benchmark scenario.

    Mirrors the ToolSandbox scenario structure: an initial world state, a seed
    message list, the set of tools the agent is allowed to call, and the
    milestones (and minefields) that define success.
    """

    scenario_id: str
    name: str = ""
    base_scenario: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_allow_list: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    minefields: List[Minefield] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """Mutable environment state for benchmark execution."""

    episode_id: str
    messages: List[BaseMessage] = field(default_factory=list)
    done: bool = False
    latest_observation: Dict[str, Any] = field(default_factory=dict)
    latest_action: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    minefields: List[Minefield] = field(default_factory=list)
    turn_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """What the runtime sees at a given step."""

    episode_id: str
    text: str
    messages: List[BaseMessage] = field(default_factory=list)
    world_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    available_tools: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    recipient: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """What the runtime returns to the environment."""

    action_type: str
    text: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    recipient: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEvent:
    """Record of a single tool call and its result."""

    tool_name: str
    arguments: Dict[str, Any]
    result: str
    latency_ms: float = 0.0
    token_count: Optional[int] = None


@dataclass
class TrajectoryEvent:
    """Single event inside a trajectory."""

    event_type: str = "turn"
    turn_number: int = 0
    user_input: str = ""
    system_prompt: str = ""
    agent_message: str = ""
    actor: Optional[str] = None
    recipient: Optional[str] = None
    observation: Optional[Observation] = None
    action: Optional[Action] = None
    tool_calls: List[ToolEvent] = field(default_factory=list)
    memory_state: Dict[str, Any] = field(default_factory=dict)
    environment_state_before: Dict[str, Any] = field(default_factory=dict)
    environment_state_after: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    milestone_checks: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Ordered collection of trajectory events."""

    events: List[TrajectoryEvent] = field(default_factory=list)

    def append(self, event: TrajectoryEvent) -> None:
        self.events.append(event)

    def extend(self, events: List[TrajectoryEvent]) -> None:
        self.events.extend(events)

    def __len__(self) -> int:
        return len(self.events)


@dataclass
class Episode:
    """A benchmark episode containing one or more tasks."""

    episode_id: str
    task: Task
    metadata: Dict[str, Any] = field(default_factory=dict)
    environment_state: EnvironmentState | None = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        return self.task.question

    @property
    def gold_answer(self) -> str:
        return self.task.gold_answer

    @property
    def context(self) -> Dict[str, Any]:
        return self.task.context

    @property
    def mode(self) -> str:
        return self.task.mode

    @property
    def sample_id(self) -> str:
        return self.episode_id


@dataclass
class EvaluationContext:
    """Structured input to evaluators."""

    episode: Episode
    trajectory: List[TrajectoryEvent] = field(default_factory=list)
    environment_state: EnvironmentState | None = None
    predicted_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    official_metadata: Dict[str, Any] = field(default_factory=dict)
    run_result: RunResult | None = None  # type: ignore[name-defined]


@dataclass
class RunResult:
    """Complete result of running one benchmark episode or sample."""

    sample_id: str
    question: str = ""
    predicted_answer: str = ""
    gold_answer: str = ""
    trajectory: List[TrajectoryEvent] = field(default_factory=list)
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    benchmark_mode: str = "plain_qa"  # e.g., "plain_qa", "retrieval_qa", "strict"
    context_turn_count: int = 0
    episode_id: str = ""
    episode: Optional[Episode] = None
    final_state: Optional[EnvironmentState] = None
    official_eval: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Evaluation scores and diagnostics for one sample."""

    sample_id: str
    is_correct: bool
    score: float  # typically 0.0 to 1.0
    correctness_reason: str = ""
    evidence_hits: List[str] = field(default_factory=list)
    failure_mode: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """Compatibility wrapper around the newer Task / Episode model."""

    sample_id: str
    question: str
    gold_answer: str
    context: Dict[str, Any] = field(default_factory=dict)
    mode: str = "plain_qa"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def episode_id(self) -> str:
        return self.sample_id

    def to_task(self) -> Task:
        return Task(
            task_id=self.sample_id,
            question=self.question,
            gold_answer=self.gold_answer,
            context=self.context,
            mode=self.mode,
            metadata=self.metadata,
        )

    def to_episode(self) -> Episode:
        return Episode(
            episode_id=self.sample_id,
            task=self.to_task(),
            metadata=self.metadata,
            raw_data=self.context.get("raw_fields", {}),
        )

    @classmethod
    def from_episode(cls, episode: Episode) -> "BenchmarkSample":
        return cls(
            sample_id=episode.episode_id,
            question=episode.question,
            gold_answer=episode.gold_answer,
            context=episode.context,
            mode=episode.mode,
            metadata={**episode.task.metadata, **episode.metadata},
        )


TrajectoryStep = TrajectoryEvent
