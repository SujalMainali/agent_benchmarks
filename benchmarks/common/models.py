"""Shared data structures for all benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolEvent:
    """Record of a single tool call and its result."""

    tool_name: str
    arguments: Dict[str, Any]
    result: str
    latency_ms: float = 0.0
    token_count: Optional[int] = None


@dataclass
class TrajectoryStep:
    """One turn of agent interaction."""

    turn_number: int
    user_input: str
    system_prompt: str
    agent_message: str
    tool_calls: List[ToolEvent] = field(default_factory=list)
    memory_state: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RunResult:
    """Complete result of running one benchmark sample."""

    sample_id: str
    predicted_answer: str
    gold_answer: str
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
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
    """Normalized sample from any benchmark dataset."""

    sample_id: str
    question: str
    gold_answer: str
    context: Dict[str, Any] = field(default_factory=dict)  # benchmark-specific data
    mode: str = "plain_qa"  # e.g., "plain_qa", "retrieval_qa", "multi_turn"
    metadata: Dict[str, Any] = field(default_factory=dict)
