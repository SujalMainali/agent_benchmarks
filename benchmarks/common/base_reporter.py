"""Shared reporter that drives :class:`ExperimentRunWriter`.

All four benchmarks produce the identical standardized layout (see
ResultFormat.md). The only differences between benchmarks are:

- which benchmark-specific fields decorate each case record, and
- how the batch-level aggregate metrics are computed.

Subclasses override the small hooks below; the write/finalize orchestration
lives here once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import EvaluationResult, RunResult
from .result_writer import ExperimentRunWriter


class StandardReporter:
    """Base reporter writing the standardized ResultFormat layout."""

    #: Benchmark identifier (e.g. "locomo"). Subclasses must set this.
    benchmark: str = "benchmark"

    def __init__(
        self,
        *,
        dataset: str = "",
        results_root: str = "results",
        benchmark_version: str = "v1",
        memory_architecture: Optional[str] = None,
        agent_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        event_transform: Optional[Any] = None,
        include_raw_messages: bool = True,
    ) -> None:
        self.writer = ExperimentRunWriter(
            benchmark=self.benchmark,
            dataset=dataset,
            results_root=results_root,
            benchmark_version=benchmark_version,
            memory_architecture=memory_architecture,
            agent_name=agent_name,
            run_metadata=run_metadata,
            event_transform=event_transform,
            include_raw_messages=include_raw_messages,
        )

    @property
    def run_dir(self) -> str:
        return self.writer.run_dir

    # -- per-sample hooks (override in subclasses) -------------------------

    def case_fields(
        self, run_result: RunResult, eval_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Benchmark-specific keyword args passed to ``append_case``.

        Return a subset of: task_family, task_type, expected_tool_behavior,
        expected_state, benchmark_metrics, benchmark_specific, routing,
        memory_actions.
        """
        return {}

    def aggregate_metrics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        """Benchmark-specific metrics block for ``summary.json``."""
        return {}

    def aggregate_diagnostics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> Dict[str, Any]:
        """Benchmark-specific diagnostics block for ``summary.json``."""
        return {}

    # -- active per-sample writing -----------------------------------------

    def record_sample(
        self, run_result: RunResult, eval_result: EvaluationResult, index: int
    ) -> None:
        """Write one sample's raw artifacts + processed case record.

        Call this from the run loop as each sample finishes so raw artifacts
        land on disk actively rather than at the end of the batch.
        """
        self.writer.write_raw(run_result, index)
        self.writer.append_case(
            run_result,
            eval_result,
            index,
            **self.case_fields(run_result, eval_result),
        )

    def record_all(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> None:
        """Convenience: record every sample in order."""
        for index, (run_result, eval_result) in enumerate(
            zip(run_results, eval_results)
        ):
            self.record_sample(run_result, eval_result, index)

    # -- finalization ------------------------------------------------------

    @staticmethod
    def _accuracy_and_score(
        eval_results: Sequence[EvaluationResult],
    ) -> Tuple[int, float, float]:
        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.is_correct)
        accuracy = correct / total if total else 0.0
        avg_score = (
            sum(r.score for r in eval_results) / total if total else 0.0
        )
        return correct, accuracy, avg_score

    def finalize(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> str:
        """Write summary.json + aggregates + index row. Returns run dir."""
        correct, accuracy, avg_score = self._accuracy_and_score(eval_results)
        errors = sum(1 for r in run_results if getattr(r, "error", None))
        return self.writer.finalize(
            sample_count=len(eval_results),
            accuracy=accuracy,
            average_score=avg_score,
            correct=correct,
            errors=errors,
            metrics=self.aggregate_metrics(run_results, eval_results),
            diagnostics=self.aggregate_diagnostics(run_results, eval_results),
        )

    def write_batch(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> str:
        """Record all samples then finalize — for non-streaming benchmarks."""
        self.record_all(run_results, eval_results)
        return self.finalize(run_results, eval_results)
