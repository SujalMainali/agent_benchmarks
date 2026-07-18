"""ToolSandbox reporting on the standardized ResultFormat layout.

Emphasizes state transitions and milestone progress rather than a single text
answer: milestone/minefield outcomes and the official similarity scores land in
each case's ``benchmark_specific`` block, and the per-snapshot world-state
trace is captured in the sample's ``raw/environments/`` artifact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from benchmarks.common.base_reporter import StandardReporter
from benchmarks.common.models import EvaluationResult, RunResult

from .metrics import ToolSandboxMetrics


class ToolSandboxReporter(StandardReporter):
    """Writes ToolSandbox runs in the standardized immutable run layout."""

    benchmark = "toolsandbox"

    def __init__(
        self,
        *,
        dataset: str = "toolsandbox",
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
        metadata = run_result.metadata or {}
        official = run_result.official_eval or {}
        state = run_result.final_state
        return {
            "task_family": "stateful_tool_use",
            "task_type": str(
                eval_result.diagnostics.get(
                    "category",
                    (metadata.get("categories") or ["uncategorized"])[0],
                )
            ),
            "benchmark_metrics": ToolSandboxMetrics.compute_metrics(run_result),
            "expected_state": {
                "milestones": [
                    getattr(m, "milestone_id", None)
                    for m in (getattr(state, "milestones", []) or [])
                ],
                "minefields": [
                    getattr(m, "minefield_id", None)
                    for m in (getattr(state, "minefields", []) or [])
                ],
            },
            "benchmark_specific": {
                "scenario_name": metadata.get("scenario_name", run_result.sample_id),
                "categories": metadata.get("categories", []),
                "official_eval": official,
                "milestone_similarity": official.get("milestone_similarity"),
                "minefield_similarity": official.get("minefield_similarity"),
                "milestones_matched": len(official.get("milestone_mapping", {}) or {}),
                "milestone_count": official.get("milestone_count", 0),
            },
        }

    def aggregate_metrics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        return {
            "batch_metrics": ToolSandboxMetrics.compute_batch_metrics(
                list(run_results)
            ),
            "by_category": self._by_category(eval_results),
        }

    def aggregate_diagnostics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        failure_modes: Dict[str, int] = {}
        for result in eval_results:
            if result.failure_mode:
                failure_modes[result.failure_mode] = (
                    failure_modes.get(result.failure_mode, 0) + 1
                )
        return {"failure_modes": failure_modes}

    @staticmethod
    def _by_category(
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        by_category: Dict[str, Dict[str, Any]] = {}
        for result in eval_results:
            category = result.diagnostics.get("category", "uncategorized")
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
