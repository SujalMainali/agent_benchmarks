"""Generic report writer for benchmark results."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    EnvironmentState,
    EvaluationContext,
    EvaluationResult,
    Episode,
    RunResult,
    Trajectory,
    TrajectoryEvent,
)


def _as_serializable(value: Any) -> Any:
    """Coerce dataclasses (and lists of them) into JSON-friendly dicts."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_as_serializable(v) for v in value]
    return value


class ReportWriter:
    """Writes benchmark results in multiple formats (JSON, CSV, Markdown)."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _run_dict(self, run_result: RunResult) -> Dict[str, Any]:
        return {
            "episode_id": run_result.episode_id or run_result.sample_id,
            "sample_id": run_result.sample_id,
            "question": run_result.question,
            "benchmark_mode": run_result.benchmark_mode,
            "context_turn_count": run_result.context_turn_count,
            "predicted_answer": run_result.predicted_answer,
            "gold_answer": run_result.gold_answer,
            "metrics": run_result.metrics,
            "metadata": run_result.metadata,
            "total_latency_ms": run_result.total_latency_ms,
            "official_eval": run_result.official_eval,
            "error": run_result.error,
        }

    def _episode_dict(self, episode: Episode) -> Dict[str, Any]:
        return {
            "episode_id": episode.episode_id,
            "task": {
                "task_id": episode.task.task_id,
                "question": episode.task.question,
                "gold_answer": episode.task.gold_answer,
                "context": episode.task.context,
                "mode": episode.task.mode,
                "metadata": episode.task.metadata,
            },
            "metadata": episode.metadata,
            "raw_data": episode.raw_data,
        }

    def _trajectory_dict(self, trajectory: list[TrajectoryEvent] | Trajectory) -> Dict[str, Any]:
        events = trajectory.events if isinstance(trajectory, Trajectory) else trajectory
        return {
            "events": [
                {
                    "event_type": event.event_type,
                    "turn_number": event.turn_number,
                    "user_input": event.user_input,
                    "system_prompt": event.system_prompt,
                    "agent_message": event.agent_message,
                    "actor": event.actor,
                    "recipient": event.recipient,
                    "exception": event.exception,
                    "milestone_checks": event.milestone_checks,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "result": tc.result,
                            "latency_ms": tc.latency_ms,
                            "token_count": tc.token_count,
                        }
                        for tc in event.tool_calls
                    ],
                    "memory_state": event.memory_state,
                    "environment_state_before": event.environment_state_before,
                    "environment_state_after": event.environment_state_after,
                    "latency_ms": event.latency_ms,
                    "token_count": event.token_count,
                    "metadata": event.metadata,
                }
                for event in events
            ]
        }

    def write_episode(self, episode: Episode | RunResult, subdir: str = "") -> None:
        """Write the canonical episode artifact."""
        output_path = os.path.join(self.output_dir, subdir, "episode.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if isinstance(episode, RunResult):
            data = self._run_dict(episode)
        else:
            data = self._episode_dict(episode)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_trajectory(self, run_result: RunResult, subdir: str = "") -> None:
        """Write the canonical trajectory artifact."""
        output_path = os.path.join(self.output_dir, subdir, "trajectory.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "question": run_result.question,
            "benchmark_mode": run_result.benchmark_mode,
            "context_turn_count": run_result.context_turn_count,
            "official_eval": run_result.official_eval,
            **self._trajectory_dict(run_result.trajectory),
            "raw_messages": run_result.raw_messages,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_evaluation(self, eval_result: EvaluationResult, subdir: str = "") -> None:
        """Write the canonical evaluation artifact."""
        output_path = os.path.join(self.output_dir, subdir, "evaluation.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "sample_id": eval_result.sample_id,
            "is_correct": eval_result.is_correct,
            "score": eval_result.score,
            "correctness_reason": eval_result.correctness_reason,
            "evidence_hits": eval_result.evidence_hits,
            "failure_mode": eval_result.failure_mode,
            "diagnostics": eval_result.diagnostics,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _resolve_state(source: RunResult | EnvironmentState | None) -> Optional[EnvironmentState]:
        """Return the EnvironmentState from a RunResult or a state passed directly."""
        if source is None:
            return None
        if isinstance(source, EnvironmentState):
            return source
        return source.final_state

    def write_milestones(self, source: RunResult | EnvironmentState, subdir: str = "") -> None:
        """Write milestone progress for stateful benchmarks (milestone.json)."""
        state = self._resolve_state(source)
        milestones = getattr(state, "milestones", []) if state else []
        serialized = [_as_serializable(m) for m in milestones]
        data = {
            "episode_id": getattr(state, "episode_id", "") if state else "",
            "total": len(serialized),
            "satisfied": sum(1 for m in serialized if m.get("satisfied")),
            "milestones": serialized,
        }
        output_path = os.path.join(self.output_dir, subdir, "milestone.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_minefields(self, source: RunResult | EnvironmentState, subdir: str = "") -> None:
        """Write minefield (forbidden-state) status for stateful benchmarks (minefields.json)."""
        state = self._resolve_state(source)
        minefields = getattr(state, "minefields", []) if state else []
        serialized = [_as_serializable(m) for m in minefields]
        data = {
            "episode_id": getattr(state, "episode_id", "") if state else "",
            "total": len(serialized),
            "tripped": sum(1 for m in serialized if m.get("tripped")),
            "minefields": serialized,
        }
        output_path = os.path.join(self.output_dir, subdir, "minefields.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_world_state(self, source: RunResult | EnvironmentState, subdir: str = "") -> None:
        """Write the final world state for stateful benchmarks (world_state.json)."""
        state = self._resolve_state(source)
        data = {
            "episode_id": getattr(state, "episode_id", "") if state else "",
            "turn_index": getattr(state, "turn_index", 0) if state else 0,
            "allowed_tools": getattr(state, "allowed_tools", []) if state else [],
            "world_state": getattr(state, "world_state", {}) if state else {},
        }
        output_path = os.path.join(self.output_dir, subdir, "world_state.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_state_trace(self, run_result: RunResult, subdir: str = "") -> None:
        """Write the per-turn world-state evolution for stateful benchmarks (state_trace.json).

        Reconstructs the trace from each trajectory event's before/after
        environment snapshots so state changes can be inspected turn by turn.
        """
        events = (
            run_result.trajectory.events
            if isinstance(run_result.trajectory, Trajectory)
            else run_result.trajectory
        )
        steps = [
            {
                "turn_number": event.turn_number,
                "actor": event.actor,
                "recipient": event.recipient,
                "state_before": event.environment_state_before,
                "state_after": event.environment_state_after,
                "exception": event.exception,
                "milestone_checks": event.milestone_checks,
            }
            for event in events
        ]
        final_state = self._resolve_state(run_result)
        data = {
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "final_world_state": getattr(final_state, "world_state", {}) if final_state else {},
            "steps": steps,
        }
        output_path = os.path.join(self.output_dir, subdir, "state_trace.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    def write_run_results(self, run_result: RunResult, subdir: str = "") -> None:
        """
        Write a single run result as JSON.

        Args:
            run_result: The RunResult to write.
            subdir: Optional subdirectory for organization.
        """
        output_path = os.path.join(self.output_dir, subdir, "output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self._run_dict(run_result), f, indent=2, default=str)

    def write_trace(self, run_result: RunResult, subdir: str = "") -> None:
        """
        Write the full interaction trace as JSON.

        Args:
            run_result: The RunResult containing trajectory.
            subdir: Optional subdirectory for organization.
        """
        output_path = os.path.join(self.output_dir, subdir, "trace.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = {
            "sample_id": run_result.sample_id,
            "question": run_result.question,
            "benchmark_mode": run_result.benchmark_mode,
            "context_turn_count": run_result.context_turn_count,
            "official_eval": run_result.official_eval,
            "trajectory": self._trajectory_dict(run_result.trajectory)["events"],
            "raw_messages": run_result.raw_messages,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_evaluation_results(self, eval_results: List[EvaluationResult], filename: str = "results.json") -> None:
        """
        Write evaluation results as JSON.

        Args:
            eval_results: List of EvaluationResult objects.
            filename: Output filename.
        """
        output_path = os.path.join(self.output_dir, filename)
        data = [
            {
                "sample_id": r.sample_id,
                "is_correct": r.is_correct,
                "score": r.score,
                "correctness_reason": r.correctness_reason,
                "evidence_hits": r.evidence_hits,
                "failure_mode": r.failure_mode,
                "diagnostics": r.diagnostics,
            }
            for r in eval_results
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_csv_summary(self, eval_results: List[EvaluationResult], filename: str = "summary.csv") -> None:
        """
        Write a CSV summary of evaluation results.

        Args:
            eval_results: List of EvaluationResult objects.
            filename: Output filename.
        """
        output_path = os.path.join(self.output_dir, filename)
        if not eval_results:
            return

        fieldnames = [
            "sample_id",
            "is_correct",
            "score",
            "correctness_reason",
            "failure_mode",
            "category",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in eval_results:
                writer.writerow(
                    {
                        "sample_id": result.sample_id,
                        "is_correct": result.is_correct,
                        "score": result.score,
                        "correctness_reason": result.correctness_reason,
                        "failure_mode": result.failure_mode,
                        "category": result.diagnostics.get("category", "unknown"),
                    }
                )

    def write_markdown_report(
        self,
        title: str,
        eval_results: List[EvaluationResult],
        metrics: Optional[Dict[str, Any]] = None,
        filename: str = "report.md",
    ) -> None:
        """
        Write a human-readable markdown report.

        Args:
            title: Report title.
            eval_results: List of EvaluationResult objects.
            metrics: Optional summary metrics dictionary.
            filename: Output filename.
        """
        output_path = os.path.join(self.output_dir, filename)

        lines = [
            f"# {title}",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\n## Summary",
        ]

        if metrics:
            lines.append(f"\n- **Total Samples**: {metrics.get('total_samples', 0)}")
            lines.append(f"- **Correct**: {metrics.get('correct', 0)}")
            lines.append(f"- **Accuracy**: {metrics.get('accuracy', 0):.2%}")
            lines.append(f"- **Average Score**: {metrics.get('average_score', 0):.3f}")

        lines.append("\n## Result Breakdown\n")

        for result in eval_results:
            status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
            lines.append(f"### {result.sample_id} - {status}")
            lines.append(f"\n**Score**: {result.score:.3f}")
            lines.append(f"\n**Reason**: {result.correctness_reason}")

            if result.failure_mode:
                lines.append(f"\n**Failure Mode**: {result.failure_mode}")

            if result.evidence_hits:
                lines.append(f"\n**Evidence Hits**: {', '.join(result.evidence_hits)}")

            lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
