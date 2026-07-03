"""LoCoMo-specific report generation."""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict, List

from benchmarks.common.models import EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter

from .metrics import LoCoMoMetrics


class LoCoMoReporter:
    """Generates detailed LoCoMo benchmark reports."""

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the reporter.

        Args:
            output_dir: Directory to write reports.
        """
        self.output_dir = output_dir
        self.report_writer = ReportWriter(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def write_full_report(
        self,
        run_results: List[RunResult],
        eval_results: List[EvaluationResult],
    ) -> None:
        """
        Write a comprehensive report for a batch of runs.

        Writes:
        - results.json: All evaluation results
        - summary.csv: Quick summary
        - report.md: Human-readable report
        - metrics.json: Detailed metrics

        Args:
            run_results: List of RunResult objects.
            eval_results: List of EvaluationResult objects.
        """
        # Compute and write metrics
        metrics = {
            "run_metrics": [
                {
                    "sample_id": r.sample_id,
                    "metrics": LoCoMoMetrics.compute_metrics(r),
                }
                for r in run_results
            ],
            "batch_metrics": LoCoMoMetrics.compute_batch_metrics(run_results),
            "evaluation_summary": self._compute_evaluation_summary(eval_results),
        }

        self.report_writer.write_metrics(metrics)
        self.report_writer.write_evaluation_results(eval_results)
        self.report_writer.write_csv_summary(eval_results)

        # Write markdown report
        summary_metrics = {
            "total_samples": len(eval_results),
            "correct": sum(1 for r in eval_results if r.is_correct),
            "accuracy": sum(1 for r in eval_results if r.is_correct) / len(eval_results)
            if eval_results
            else 0,
            "average_score": sum(r.score for r in eval_results) / len(eval_results)
            if eval_results
            else 0,
        }

        self.report_writer.write_markdown_report(
            title="LoCoMo Benchmark Report",
            eval_results=eval_results,
            metrics=summary_metrics,
        )

    def write_per_sample_report(self, run_result: RunResult, eval_result: EvaluationResult) -> None:
        """
        Write detailed report for a single sample.

        Creates a subdirectory for each sample with:
        - output.json: The predicted answer with question
        - trace.json: Full interaction trace
        - analysis.json: Evaluation analysis

        Args:
            run_result: The RunResult.
            eval_result: The EvaluationResult.
        """
        sample_dir = os.path.join(self.output_dir, run_result.sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Write canonical and legacy outputs for compatibility.
        self.report_writer.write_episode(run_result, subdir=run_result.sample_id)
        self.report_writer.write_trajectory(run_result, subdir=run_result.sample_id)
        self.report_writer.write_evaluation(eval_result, subdir=run_result.sample_id)
        self.report_writer.write_run_results(run_result, subdir=run_result.sample_id)
        self.report_writer.write_trace(run_result, subdir=run_result.sample_id)

        # Write analysis (using run_result.question directly, not from raw_messages)
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
            "metrics": LoCoMoMetrics.compute_metrics(run_result),
            "diagnostics": eval_result.diagnostics,
        }

        with open(os.path.join(sample_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    def _compute_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute summary statistics from evaluation results."""
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)

        failure_modes = {}
        for r in results:
            if r.failure_mode:
                failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0,
            "average_score": sum(r.score for r in results) / total if total > 0 else 0,
            "failure_modes": failure_modes,
            "categories": self._categorize_results(results),
        }

    def _categorize_results(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Group results by category and compute category-wise metrics."""
        by_category = {}

        for result in results:
            category = result.diagnostics.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"correct": 0, "total": 0, "scores": []}

            by_category[category]["total"] += 1
            if result.is_correct:
                by_category[category]["correct"] += 1
            by_category[category]["scores"].append(result.score)

        # Compute accuracy per category
        for category in by_category:
            data = by_category[category]
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
            data["average_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            del data["scores"]  # Remove raw scores for cleaner JSON

        return by_category
