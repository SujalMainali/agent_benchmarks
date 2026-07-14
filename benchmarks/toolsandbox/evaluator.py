"""ToolSandbox evaluator - milestone-based, trajectory-level scoring.

ToolSandbox scores the *whole trajectory* against a milestone DAG (with
minefields that nullify the score), not a final answer. The heavy lifting is
done by the official scorer via ``official_bridge`` during the run; this
evaluator reads those official results off the :class:`RunResult` and maps them
into the shared :class:`EvaluationResult`.

Score semantics (from the official engine):
* ``milestone_similarity`` in [0, 1] = fraction/quality of milestones reached,
* ``minefield_similarity`` > 0 means a forbidden state was entered,
* ``similarity`` = ``milestone_similarity`` gated to 0 by any minefield hit.
"""

from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.common.evaluator_base import EvaluatorBase
from benchmarks.common.models import EvaluationContext, EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter

from .metrics import ToolSandboxMetrics

# A run counts as "correct" when milestones are fully satisfied and no minefield
# was tripped. Partial progress is still reported via ``score``.
CORRECT_THRESHOLD = 1.0


class ToolSandboxEvaluator(EvaluatorBase):
    """Evaluates ToolSandbox runs using official milestone results."""

    def __init__(self, name: str = "ToolSandbox Evaluator", correct_threshold: float = CORRECT_THRESHOLD) -> None:
        super().__init__(name)
        self.correct_threshold = correct_threshold
        self.report_writer: ReportWriter | None = None

    def evaluate(self, context: EvaluationContext | RunResult) -> EvaluationResult:
        if isinstance(context, RunResult):
            context = self._coerce_context(context)

        result = context.run_result
        official = (result.official_eval if result else None) or context.official_metadata or {}

        similarity = float(official.get("similarity", 0.0))
        milestone_similarity = float(official.get("milestone_similarity", 0.0))
        minefield_similarity = float(official.get("minefield_similarity", 0.0))

        is_correct = similarity >= self.correct_threshold
        if minefield_similarity > 0.0:
            failure_mode = "minefield_triggered"
            reason = (
                f"Minefield entered (minefield_similarity={minefield_similarity:.3f}); "
                f"trajectory nullified."
            )
        elif is_correct:
            failure_mode = None
            reason = "All milestones satisfied."
        else:
            failure_mode = "milestones_incomplete"
            reason = (
                f"Milestones partially satisfied "
                f"(milestone_similarity={milestone_similarity:.3f})."
            )
        if result is not None and result.error:
            failure_mode = failure_mode or "rollout_error"
            reason = f"{reason} Rollout error: {result.error}"

        metrics = ToolSandboxMetrics.compute_metrics(result) if result else {}

        diagnostics: Dict[str, Any] = {
            "scenario_name": context.episode.metadata.get("scenario_name", context.episode.episode_id),
            "categories": context.episode.metadata.get("categories", []),
            "milestone_similarity": milestone_similarity,
            "minefield_similarity": minefield_similarity,
            "similarity": similarity,
            "milestone_count": official.get("milestone_count", 0),
            "milestones_matched": len(official.get("milestone_mapping", {}) or {}),
            "minefield_violations": len(official.get("minefield_mapping", {}) or {}),
            "turn_count": official.get("turn_count", 0),
            "metrics": metrics,
            # Category key drives per-category reporting groupings.
            "category": _primary_category(context.episode.metadata.get("categories", [])),
        }

        sample_id = result.sample_id if result else context.episode.episode_id
        return EvaluationResult(
            sample_id=sample_id,
            is_correct=is_correct,
            score=similarity,
            correctness_reason=reason,
            evidence_hits=[],
            failure_mode=failure_mode,
            diagnostics=diagnostics,
        )

    def evaluate_batch(self, run_results: List[RunResult | EvaluationContext]) -> List[EvaluationResult]:
        self.results = [self.evaluate(result) for result in run_results]
        return self.results

    def write_report(self, results: List[EvaluationResult], output_dir: str) -> None:
        self.report_writer = ReportWriter(output_dir)
        summary_metrics = self.compute_summary_metrics(results)
        self.report_writer.write_evaluation_results(results)
        self.report_writer.write_csv_summary(results)
        self.report_writer.write_markdown_report(
            title="ToolSandbox Benchmark Report",
            eval_results=results,
            metrics=summary_metrics,
        )


def _primary_category(categories: List[Any]) -> str:
    """Pick a single category label for grouping, or 'uncategorized'."""
    if not categories:
        return "uncategorized"
    return str(categories[0])
