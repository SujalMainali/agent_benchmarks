"""LongMemEval reporting on the standardized ResultFormat layout.

Mirrors ``print_qa_metrics.py`` aggregation (per-type accuracy, task-averaged
accuracy, overall accuracy, abstention accuracy) as the summary metrics block —
the QA scoring itself stays inside the official judge. Unless full-trajectory
mode is on, the heavy history-replay events are collapsed to one-line
summaries before landing in ``raw/trajectories/`` (the final-question turn is
always kept verbatim).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from benchmarks.common.base_reporter import StandardReporter
from benchmarks.common.models import EvaluationResult, RunResult, TrajectoryEvent
from benchmarks.common.result_writer import ExperimentRunWriter

_QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


def _compact_replay_events(
    run_result: RunResult, events: List[TrajectoryEvent]
) -> List[Dict[str, Any]]:
    """Collapse history-replay events to one-line summaries; keep question.

    The frozen runtime tags trajectory events with stream-step metadata, not
    the observation's ``phase`` — so the final-question turn is identified as
    the group of events sharing the highest ``turn_number`` (each ``act()``
    call shares one turn_number; the question is the last act).
    """
    question_start: Optional[int] = None
    if events:
        max_turn = max(event.turn_number for event in events)
        for i, event in enumerate(events):
            if event.turn_number == max_turn:
                question_start = i
                break

    compact: List[Dict[str, Any]] = []
    for i, event in enumerate(events):
        if question_start is not None and i >= question_start:
            # Final-question turn: keep the full event record.
            compact.append(ExperimentRunWriter._event_dict(event))
        else:
            compact.append(
                {
                    "event_type": event.event_type,
                    "turn_number": event.turn_number,
                    "session_id": event.metadata.get("session_id"),
                    "chars": len(event.agent_message or "")
                    + len(event.user_input or ""),
                    "replay_summary": True,
                }
            )
    return compact


class LongMemEvalReporter(StandardReporter):
    """Writes LongMemEval runs in the standardized immutable run layout."""

    benchmark = "longmemeval"

    def __init__(
        self,
        *,
        dataset: str = "longmemeval",
        results_root: str = "results",
        benchmark_version: str = "v1",
        full_trajectory: bool = False,
        memory_architecture: Optional[str] = None,
        agent_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.full_trajectory = full_trajectory
        super().__init__(
            dataset=dataset,
            results_root=results_root,
            benchmark_version=benchmark_version,
            memory_architecture=memory_architecture,
            agent_name=agent_name,
            run_metadata=run_metadata,
            event_transform=None if full_trajectory else _compact_replay_events,
            include_raw_messages=full_trajectory,
        )

    def case_fields(
        self, run_result: RunResult, eval_result: EvaluationResult
    ) -> Dict[str, Any]:
        question_type = str(
            eval_result.diagnostics.get(
                "question_type",
                (run_result.metadata or {}).get("question_type", "unknown"),
            )
        )
        return {
            "task_family": "long_memory_qa",
            "task_type": question_type,
            "benchmark_metrics": run_result.metrics,
            "benchmark_specific": {
                "question_type": question_type,
                "is_abstention": bool(eval_result.diagnostics.get("is_abstention")),
                "official_eval": run_result.official_eval,
                "num_sessions": (run_result.metadata or {}).get("num_sessions"),
                "sessions_replayed": (run_result.metrics or {}).get(
                    "sessions_replayed"
                ),
            },
        }

    def aggregate_metrics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        """Per-type / task-averaged / abstention accuracy + run metrics."""
        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.is_correct)

        by_type: Dict[str, Dict[str, int]] = {}
        abst_total = abst_correct = 0
        for r in eval_results:
            qtype = str(r.diagnostics.get("question_type", "unknown"))
            bucket = by_type.setdefault(qtype, {"correct": 0, "total": 0})
            bucket["total"] += 1
            if r.is_correct:
                bucket["correct"] += 1
            if r.diagnostics.get("is_abstention"):
                abst_total += 1
                if r.is_correct:
                    abst_correct += 1

        per_type_accuracy = {
            qtype: (b["correct"] / b["total"] if b["total"] else 0.0)
            for qtype, b in by_type.items()
        }
        # Task-averaged accuracy = mean of per-type accuracies (present types).
        present = [per_type_accuracy[t] for t in _QUESTION_TYPES if t in per_type_accuracy]
        task_averaged = sum(present) / len(present) if present else 0.0

        errors = sum(1 for r in run_results if r.error)
        sessions = [
            r.metrics.get("sessions_replayed", 0) for r in run_results if r.metrics
        ]
        latencies = [r.total_latency_ms for r in run_results]

        return {
            "overall_accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total_samples": total,
            "per_type_accuracy": per_type_accuracy,
            "per_type_counts": by_type,
            "task_averaged_accuracy": task_averaged,
            "abstention_accuracy": (abst_correct / abst_total if abst_total else 0.0),
            "abstention_total": abst_total,
            "abstention_correct": abst_correct,
            "error_count": errors,
            "mean_sessions_replayed": (sum(sessions) / len(sessions) if sessions else 0.0),
            "mean_latency_ms": (sum(latencies) / len(latencies) if latencies else 0.0),
        }
