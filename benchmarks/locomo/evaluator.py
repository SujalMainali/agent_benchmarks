"""LoCoMo evaluator for scoring answers."""

from __future__ import annotations

import difflib
import os
from typing import Any, Dict, List

from benchmarks.common.evaluator_base import EvaluatorBase
from benchmarks.common.models import EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter

from .metrics import LoCoMoMetrics


class LoCoMoEvaluator(EvaluatorBase):
    """Evaluates LoCoMo benchmark runs."""

    def __init__(self, name: str = "LoCoMo Evaluator") -> None:
        super().__init__(name)
        self.report_writer: ReportWriter | None = None

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

    def evaluate(self, result: RunResult) -> EvaluationResult:
        """
        Evaluate a single run result.

        Scoring logic:
        1. Exact match (after normalization) = 1.0
        2. Fuzzy match (>85% similarity) = 0.8
        3. Contains answer = 0.5
        4. No match = 0.0

        Args:
            result: RunResult from agent execution.

        Returns:
            EvaluationResult with correctness score.
        """
        predicted = result.predicted_answer
        gold = result.gold_answer

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
        metrics = LoCoMoMetrics.compute_metrics(result)

        # Build diagnostics
        diagnostics = {
            "category": result.metadata.get("category", "unknown"),
            "metrics": metrics,
            "predicted_length_chars": len(predicted),
            "gold_length_chars": len(gold),
            "trajectory_length": len(result.trajectory),
        }

        return EvaluationResult(
            sample_id=result.sample_id,
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
