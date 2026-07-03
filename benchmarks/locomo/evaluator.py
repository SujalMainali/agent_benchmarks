"""LoCoMo evaluator for scoring answers."""

from __future__ import annotations

import difflib
import os
from typing import Any, Dict, List

from benchmarks.common.evaluator_base import EvaluatorBase
from benchmarks.common.models import EvaluationContext, EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter

from .metrics import LoCoMoMetrics


class LoCoMoEvaluator(EvaluatorBase):
    """Evaluates LoCoMo benchmark runs."""

    def __init__(self, name: str = "LoCoMo Evaluator") -> None:
        super().__init__(name)
        self.report_writer: ReportWriter | None = None
        self.use_official_eval: bool = False
        self.locomo_data: List[Dict[str, Any]] | None = None

    def _normalize_answer(self, text: str) -> str:
        """Normalize an answer for comparison."""
        return text.strip().lower()

    def _exact_match(self, predicted: str, gold: str) -> bool:
        """Check for exact match (after normalization)."""
        return self._normalize_answer(predicted) == self._normalize_answer(gold)

    def _fuzzy_match(self, predicted: str, gold: str, threshold: float = 0.85) -> bool:
        """Check for fuzzy match using sequence matching."""
        ratio = difflib.SequenceMatcher(None, 
                                       self._normalize_answer(predicted), 
                                       self._normalize_answer(gold)).ratio()
        return ratio >= threshold

    def _contains_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted contains gold answer as a substring."""
        pred_norm = self._normalize_answer(predicted)
        gold_norm = self._normalize_answer(gold)
        return gold_norm in pred_norm

    def evaluate(self, context: EvaluationContext | RunResult) -> EvaluationResult:
        """
        Evaluate a single run result or evaluation context.

        Scoring logic:
        1. Exact match (after normalization) = 1.0
        2. Fuzzy match (>85% similarity) = 0.8
        3. Contains answer = 0.5
        4. No match = 0.0

        Args:
            context: EvaluationContext or RunResult from agent execution.

        Returns:
            EvaluationResult with correctness score.
        """
        if isinstance(context, RunResult):
            context = self._coerce_context(context)

        result = context.run_result
        predicted = context.predicted_output or (result.predicted_answer if result else "")
        gold = context.episode.gold_answer

        # Compute score
        if self._exact_match(predicted, gold):
            score = 1.0
            is_correct = True
            reason = "Exact match"
        elif self._fuzzy_match(predicted, gold):
            score = 0.8
            is_correct = True
            reason = "Fuzzy match (high similarity)"
        elif self._contains_answer(predicted, gold):
            score = 0.5
            is_correct = False
            reason = "Partial match (contains gold answer)"
        else:
            score = 0.0
            is_correct = False
            reason = "No match"

        # Compute metrics
        metrics = LoCoMoMetrics.compute_metrics(result) if result else {}

        # Build diagnostics
        diagnostics = {
            "category": context.episode.metadata.get("category", context.episode.task.metadata.get("category", "unknown")),
            "episode_id": context.episode.episode_id,
            "metrics": metrics,
            "predicted_length_chars": len(predicted),
            "gold_length_chars": len(gold),
            "trajectory_length": len(context.trajectory),
            "benchmark_mode": context.episode.mode,
        }

        sample_id = result.sample_id if result else context.episode.episode_id

        return EvaluationResult(
            sample_id=sample_id,
            is_correct=is_correct,
            score=score,
            correctness_reason=reason,
            evidence_hits=[],
            failure_mode=None if is_correct else "answer_mismatch",
            diagnostics=diagnostics,
        )

    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """
        Write evaluation results to disk.

        Writes:
        - results.json: Detailed evaluation results
        - summary.csv: Quick summary as CSV
        - report.md: Human-readable markdown report

        Args:
            results: List of EvaluationResult objects.
            output_dir: Directory to write reports.
        """
        self.report_writer = ReportWriter(output_dir)

        # Compute summary metrics
        summary_metrics = self.compute_summary_metrics(results)

        # Write results
        self.report_writer.write_evaluation_results(results)
        self.report_writer.write_csv_summary(results)
        self.report_writer.write_markdown_report(
            title="LoCoMo Benchmark Report",
            eval_results=results,
            metrics=summary_metrics,
        )

    def evaluate_batch_official(
        self, results: List[RunResult], locomo_data: List[Dict[str, Any]] | None = None
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of results using the official LoCoMo evaluator.

        This calls the official eval_question_answering function which provides
        category-aware F1 scoring aligned with the benchmark.

        Args:
            results: List of RunResult objects to evaluate.
            locomo_data: Optional original LoCoMo dataset for context and categories.

        Returns:
            List of EvaluationResult objects with official scores.

        Raises:
            ImportError: If the official evaluator cannot be imported.
        """
        from .official_bridge import run_official_evaluation

        return run_official_evaluation(results, locomo_data)
