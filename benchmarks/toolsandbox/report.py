"""ToolSandbox-specific report generation.

Emphasizes state transitions and milestone progress rather than a single text
answer. Reuses the shared :class:`~benchmarks.common.report_writer.ReportWriter`
(including its ToolSandbox-oriented ``milestone.json`` / ``minefields.json`` /
``world_state.json`` / ``state_trace.json`` writers).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from benchmarks.common.models import EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter

from .metrics import ToolSandboxMetrics


class ToolSandboxReporter:
    """Generates detailed ToolSandbox benchmark reports."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.report_writer = ReportWriter(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def write_full_report(
        self,
        run_results: List[RunResult],
        eval_results: List[EvaluationResult],
    ) -> None:
        """Write batch-level artifacts: metrics, results, csv, markdown."""
        metrics = {
            "run_metrics": [
                {"sample_id": r.sample_id, "metrics": ToolSandboxMetrics.compute_metrics(r)}
                for r in run_results
            ],
            "batch_metrics": ToolSandboxMetrics.compute_batch_metrics(run_results),
            "evaluation_summary": self._evaluation_summary(eval_results),
        }
        self.report_writer.write_metrics(metrics)
        self.report_writer.write_evaluation_results(eval_results)
        self.report_writer.write_csv_summary(eval_results)

        summary_metrics = {
            "total_samples": len(eval_results),
            "correct": sum(1 for r in eval_results if r.is_correct),
            "accuracy": (
                sum(1 for r in eval_results if r.is_correct) / len(eval_results)
                if eval_results
                else 0
            ),
            "average_score": (
                sum(r.score for r in eval_results) / len(eval_results)
                if eval_results
                else 0
            ),
        }
        self.report_writer.write_markdown_report(
            title="ToolSandbox Benchmark Report",
            eval_results=eval_results,
            metrics=summary_metrics,
        )

    def write_per_sample_report(self, run_result: RunResult, eval_result: EvaluationResult) -> None:
        """Write per-scenario artifacts, focused on state and milestones."""
        sample_dir = os.path.join(self.output_dir, run_result.sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        subdir = run_result.sample_id

        # Canonical episode / trajectory / evaluation artifacts.
        self.report_writer.write_episode(run_result, subdir=subdir)
        self.report_writer.write_trajectory(run_result, subdir=subdir)
        self.report_writer.write_evaluation(eval_result, subdir=subdir)
        self.report_writer.write_run_results(run_result, subdir=subdir)
        self.report_writer.write_trace(run_result, subdir=subdir)

        # ToolSandbox state-oriented artifacts.
        self.report_writer.write_milestones(run_result, subdir=subdir)
        self.report_writer.write_minefields(run_result, subdir=subdir)
        self.report_writer.write_world_state(run_result, subdir=subdir)
        self._write_state_trace(run_result, sample_dir)

        analysis = {
            "sample_id": run_result.sample_id,
            "scenario_name": run_result.metadata.get("scenario_name", run_result.sample_id),
            "question": run_result.question,
            "predicted_answer": run_result.predicted_answer,
            "is_correct": eval_result.is_correct,
            "score": eval_result.score,
            "correctness_reason": eval_result.correctness_reason,
            "failure_mode": eval_result.failure_mode,
            "official_eval": run_result.official_eval,
            "metrics": ToolSandboxMetrics.compute_metrics(run_result),
            "diagnostics": eval_result.diagnostics,
            "error": run_result.error,
        }
        with open(os.path.join(sample_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    def _write_state_trace(self, run_result: RunResult, sample_dir: str) -> None:
        """Write the official per-snapshot world-state trace (state_trace.json).

        The runner captures the real ToolSandbox snapshot trace in run metadata;
        we persist it directly so the file reflects actual state transitions.
        """
        final_state = run_result.final_state
        data = {
            "sample_id": run_result.sample_id,
            "scenario_name": run_result.metadata.get("scenario_name", run_result.sample_id),
            "final_world_state": getattr(final_state, "world_state", {}) if final_state else {},
            "steps": run_result.metadata.get("state_trace", []),
        }
        with open(os.path.join(sample_dir, "state_trace.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        failure_modes: Dict[str, int] = {}
        for result in results:
            if result.failure_mode:
                failure_modes[result.failure_mode] = failure_modes.get(result.failure_mode, 0) + 1
        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0,
            "average_score": sum(r.score for r in results) / total if total > 0 else 0,
            "failure_modes": failure_modes,
            "categories": self._categorize(results),
        }

    def _categorize(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        by_category: Dict[str, Dict[str, Any]] = {}
        for result in results:
            category = result.diagnostics.get("category", "uncategorized")
            bucket = by_category.setdefault(category, {"correct": 0, "total": 0, "scores": []})
            bucket["total"] += 1
            if result.is_correct:
                bucket["correct"] += 1
            bucket["scores"].append(result.score)
        for category, data in by_category.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] else 0
            data["average_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            del data["scores"]
        return by_category
