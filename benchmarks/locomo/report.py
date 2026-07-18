"""LoCoMo-specific reporting on the standardized ResultFormat layout."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from benchmarks.common.base_reporter import StandardReporter
from benchmarks.common.models import EvaluationResult, RunResult

from .metrics import LoCoMoMetrics


class LoCoMoReporter(StandardReporter):
    """Writes LoCoMo runs in the standardized immutable run layout."""

    benchmark = "locomo"

    def __init__(
        self,
        *,
        dataset: str = "locomo",
        results_root: str = "results",
        benchmark_version: str = "v1",
        memory_architecture: Optional[str] = None,
        agent_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            results_root=results_root,
            benchmark_version=benchmark_version,
            memory_architecture=memory_architecture,
            agent_name=agent_name,
            run_metadata=run_metadata,
        )

    def case_fields(
        self, run_result: RunResult, eval_result: EvaluationResult
    ) -> Dict[str, Any]:
        category = eval_result.diagnostics.get(
            "category",
            (run_result.metadata or {}).get("category", "unknown"),
        )
        return {
            "task_family": "long_conversation_qa",
            "task_type": run_result.benchmark_mode,
            "benchmark_metrics": LoCoMoMetrics.compute_metrics(run_result),
            "benchmark_specific": {
                "category": category,
                "official_eval": run_result.official_eval,
                "source_sample_id": (run_result.metadata or {}).get(
                    "source_sample_id"
                ),
            },
        }

    def aggregate_metrics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        return {
            "batch_metrics": LoCoMoMetrics.compute_batch_metrics(list(run_results)),
            "by_category": self._by_category(eval_results),
        }

    def aggregate_diagnostics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        failure_modes: Dict[str, int] = {}
        for r in eval_results:
            if r.failure_mode:
                failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1
        return {"failure_modes": failure_modes}

    @staticmethod
    def _by_category(
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        by_category: Dict[str, Dict[str, Any]] = {}
        for result in eval_results:
            category = result.diagnostics.get("category", "unknown")
            bucket = by_category.setdefault(
                category, {"correct": 0, "total": 0, "score_sum": 0.0}
            )
            bucket["total"] += 1
            bucket["score_sum"] += result.score
            if result.is_correct:
                bucket["correct"] += 1
        for bucket in by_category.values():
            total = bucket["total"]
            bucket["accuracy"] = bucket["correct"] / total if total else 0.0
            bucket["average_score"] = bucket["score_sum"] / total if total else 0.0
            del bucket["score_sum"]
        return by_category
