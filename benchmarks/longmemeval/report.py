"""LongMemEval report generation.

Mirrors ``print_qa_metrics.py`` aggregation (per-type accuracy, task-averaged
accuracy, overall accuracy, abstention accuracy) as report conversion — the QA
scoring itself stays inside the official judge. Also truncates the heavy
history-replay trajectory/raw_messages unless full-trajectory mode is on.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from benchmarks.common.models import EvaluationResult, RunResult, Trajectory
from benchmarks.common.report_writer import ReportWriter

_QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


class LongMemEvalReporter:
    """Generates LongMemEval batch + per-sample reports."""

    def __init__(self, output_dir: str, full_trajectory: bool = False) -> None:
        self.output_dir = output_dir
        self.full_trajectory = full_trajectory
        self.report_writer = ReportWriter(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def write_full_report(
        self,
        run_results: List[RunResult],
        eval_results: List[EvaluationResult],
    ) -> None:
        metrics = self._aggregate_metrics(run_results, eval_results)
        self.report_writer.write_metrics(metrics)
        self.report_writer.write_evaluation_results(eval_results)
        self.report_writer.write_csv_summary(eval_results)

        summary_metrics = {
            "total_samples": len(eval_results),
            "correct": sum(1 for r in eval_results if r.is_correct),
            "accuracy": metrics["overall_accuracy"],
            "average_score": (
                sum(r.score for r in eval_results) / len(eval_results)
                if eval_results
                else 0.0
            ),
        }
        self.report_writer.write_markdown_report(
            title="LongMemEval Benchmark Report",
            eval_results=eval_results,
            metrics=summary_metrics,
        )

    def _aggregate_metrics(
        self,
        run_results: List[RunResult],
        eval_results: List[EvaluationResult],
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

    def write_per_sample_report(
        self, run_result: RunResult, eval_result: EvaluationResult
    ) -> None:
        sample_dir = os.path.join(self.output_dir, run_result.sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        if self.full_trajectory:
            self.report_writer.write_trajectory(run_result, subdir=run_result.sample_id)
        else:
            self._write_truncated_trajectory(run_result, sample_dir)

        self.report_writer.write_evaluation(eval_result, subdir=run_result.sample_id)

        analysis = {
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "question": run_result.question,
            "gold_answer": run_result.gold_answer,
            "predicted_answer": run_result.predicted_answer,
            "benchmark_mode": run_result.benchmark_mode,
            "context_turn_count": run_result.context_turn_count,
            "official_eval": run_result.official_eval,
            "is_correct": eval_result.is_correct,
            "score": eval_result.score,
            "correctness_reason": eval_result.correctness_reason,
            "failure_mode": eval_result.failure_mode,
            "metrics": run_result.metrics,
            "diagnostics": eval_result.diagnostics,
            "error": run_result.error,
        }
        with open(os.path.join(sample_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    def _write_truncated_trajectory(self, run_result: RunResult, sample_dir: str) -> None:
        """Collapse history-replay events to one-line summaries; keep question.

        The frozen runtime tags trajectory events with stream-step metadata, not
        the observation's ``phase`` — so we identify the final-question turn as
        the group of events sharing the highest ``turn_number`` (each ``act()``
        call shares one turn_number; the question is the last act). This is the
        plan's documented fallback ("last replay session + everything after it").
        """
        events = (
            run_result.trajectory.events
            if isinstance(run_result.trajectory, Trajectory)
            else run_result.trajectory
        )

        question_start = None
        if events:
            max_turn = max(event.turn_number for event in events)
            for i, event in enumerate(events):
                if event.turn_number == max_turn:
                    question_start = i
                    break

        compact: List[Dict[str, Any]] = []
        for i, event in enumerate(events):
            in_question = question_start is not None and i >= question_start
            if in_question:
                compact.append(
                    {
                        "event_type": event.event_type,
                        "turn_number": event.turn_number,
                        "agent_message": event.agent_message,
                        "phase": event.metadata.get("phase"),
                    }
                )
            else:
                compact.append(
                    {
                        "event_type": event.event_type,
                        "turn_number": event.turn_number,
                        "session_id": event.metadata.get("session_id"),
                        "chars": len(event.agent_message or "")
                        + len(event.user_input or ""),
                    }
                )

        raw = run_result.raw_messages or []
        # Cap raw_messages to the final-question slice (heuristic: keep the last 4).
        tail = raw[-4:] if len(raw) > 4 else raw
        data = {
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "question": run_result.question,
            "benchmark_mode": run_result.benchmark_mode,
            "context_turn_count": run_result.context_turn_count,
            "official_eval": run_result.official_eval,
            "events": compact,
            "raw_messages": tail,
            "raw_messages_truncated": max(len(raw) - len(tail), 0),
        }
        with open(os.path.join(sample_dir, "trajectory.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)
