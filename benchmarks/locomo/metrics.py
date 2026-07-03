"""Metrics computation for LoCoMo benchmark runs."""

from __future__ import annotations

from typing import Any, Dict

from benchmarks.common.models import EvaluationContext, RunResult


class LoCoMoMetrics:
    """Computes metrics for LoCoMo runs."""

    @staticmethod
    def compute_metrics(run_result: RunResult | EvaluationContext) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a single run.

        Metrics include:
        - answer_length: number of tokens/words in the answer
        - turn_count: number of turns in the trajectory
        - tool_count: number of tool calls made
        - latency_ms: total execution time
        - tool_latency_ms: time spent in tool calls
        - model_latency_ms: time spent in model calls

        Args:
            run_result: The RunResult or EvaluationContext from agent execution.

        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}

        # Answer length metrics
        if isinstance(run_result, EvaluationContext):
            predicted_answer = run_result.predicted_output
            trajectory = run_result.trajectory
            total_latency_ms = run_result.run_result.total_latency_ms if run_result.run_result else 0.0
        else:
            predicted_answer = run_result.predicted_answer
            trajectory = run_result.trajectory
            total_latency_ms = run_result.total_latency_ms

        answer = predicted_answer.strip()
        metrics["answer_length_chars"] = len(answer)
        metrics["answer_length_words"] = len(answer.split())

        # Trajectory metrics
        metrics["turn_count"] = len(trajectory)

        # Tool usage metrics
        total_tool_calls = sum(len(step.tool_calls) for step in trajectory)
        metrics["tool_call_count"] = total_tool_calls

        # Latency metrics
        metrics["total_latency_ms"] = total_latency_ms
        metrics["average_turn_latency_ms"] = (
            total_latency_ms / len(trajectory) if trajectory else 0
        )

        # Tool latency
        tool_latency = sum(
            sum(tc.latency_ms for tc in step.tool_calls) for step in trajectory
        )
        metrics["tool_latency_ms"] = tool_latency
        metrics["model_latency_ms"] = total_latency_ms - tool_latency

        # Tool breakdown by type
        tool_types: Dict[str, int] = {}
        for step in trajectory:
            for tool_call in step.tool_calls:
                tool_name = tool_call.tool_name
                tool_types[tool_name] = tool_types.get(tool_name, 0) + 1
        metrics["tool_breakdown"] = tool_types

        return metrics

    @staticmethod
    def compute_batch_metrics(run_results: list[RunResult]) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple runs.

        Args:
            run_results: List of RunResult objects.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not run_results:
            return {}

        total = len(run_results)
        all_metrics = [LoCoMoMetrics.compute_metrics(r) for r in run_results]

        avg_answer_length = (
            sum(m["answer_length_chars"] for m in all_metrics) / total
        )
        avg_answer_words = (
            sum(m["answer_length_words"] for m in all_metrics) / total
        )
        avg_turn_count = sum(m["turn_count"] for m in all_metrics) / total
        avg_tool_calls = sum(m["tool_call_count"] for m in all_metrics) / total
        avg_latency = sum(m["total_latency_ms"] for m in all_metrics) / total

        return {
            "total_runs": total,
            "avg_answer_length_chars": avg_answer_length,
            "avg_answer_length_words": avg_answer_words,
            "avg_turn_count": avg_turn_count,
            "avg_tool_calls": avg_tool_calls,
            "avg_latency_ms": avg_latency,
        }
