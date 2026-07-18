"""LongMemEval evaluator.

Two scoring paths:
- ``evaluate`` — a LOCAL, offline heuristic (normalized substring/fuzzy match,
  plus refusal detection for abstention questions). Clearly labeled non-official.
- ``evaluate_batch_official`` — the official LLM judge via ``evaluate_qa.py``.

The official QA judge prompts are never reimplemented or tweaked here.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import List

from benchmarks.common.evaluator_base import EvaluatorBase
from benchmarks.common.models import (
    EvaluationContext,
    EvaluationResult,
    RunResult,
)
from benchmarks.common.report_writer import ReportWriter

from .official_bridge import (
    parse_official_results,
    run_official_evaluation,
    write_hypotheses_jsonl,
)

_REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "no information",
    "not mentioned",
    "wasn't discussed",
    "was not discussed",
    "cannot find",
    "can't find",
    "unable to find",
    "never mentioned",
    "don't recall",
    "do not recall",
)


class LongMemEvalEvaluator(EvaluatorBase):
    """Evaluates LongMemEval runs (local heuristic + official judge bridge)."""

    def __init__(self, name: str = "LongMemEval Evaluator") -> None:
        super().__init__(name)
        self.report_writer: ReportWriter | None = None

    def _normalize(self, text: str) -> str:
        return (text or "").strip().lower()

    def _exact_match(self, predicted: str, gold: str) -> bool:
        return self._normalize(predicted) == self._normalize(gold)

    def _fuzzy_match(self, predicted: str, gold: str, threshold: float = 0.85) -> bool:
        ratio = difflib.SequenceMatcher(
            None, self._normalize(predicted), self._normalize(gold)
        ).ratio()
        return ratio >= threshold

    def _contains_answer(self, predicted: str, gold: str) -> bool:
        gold_norm = self._normalize(gold)
        return bool(gold_norm) and gold_norm in self._normalize(predicted)

    def evaluate(self, context: EvaluationContext | RunResult) -> EvaluationResult:
        """LOCAL heuristic fallback (no API key). Labeled non-official."""
        if isinstance(context, RunResult):
            context = self._coerce_context(context)

        result = context.run_result
        predicted = context.predicted_output or (result.predicted_answer if result else "")
        gold = context.episode.gold_answer
        meta = context.episode.metadata
        is_abstention = bool(meta.get("is_abstention"))

        if is_abstention:
            pred_norm = self._normalize(predicted)
            refused = any(marker in pred_norm for marker in _REFUSAL_MARKERS)
            score = 1.0 if refused else 0.0
            is_correct = refused
            reason = "Abstention: refusal detected" if refused else "Abstention: no refusal"
        elif self._exact_match(predicted, gold):
            score, is_correct, reason = 1.0, True, "Exact match"
        elif self._fuzzy_match(predicted, gold):
            score, is_correct, reason = 0.8, True, "Fuzzy match (high similarity)"
        elif self._contains_answer(predicted, gold):
            score, is_correct, reason = 0.5, False, "Partial match (contains gold answer)"
        else:
            score, is_correct, reason = 0.0, False, "No match"

        diagnostics = {
            "question_type": meta.get("question_type", "unknown"),
            "question_date": meta.get("question_date", ""),
            "is_abstention": is_abstention,
            "num_sessions": meta.get("num_sessions", 0),
            "evaluator": "local-heuristic",
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

    def evaluate_batch_official(
        self,
        run_results: List[RunResult],
        ref_file: str,
        metric_model: str,
        official_root: str,
        output_dir: str,
        judge_api_key: str | None = None,
        judge_base_url: str | None = None,
    ) -> List[EvaluationResult]:
        """Score a batch with the official ``evaluate_qa.py`` LLM judge."""
        hyp = write_hypotheses_jsonl(
            run_results, str(Path(output_dir) / "hypotheses.jsonl")
        )
        result_file = run_official_evaluation(
            hyp,
            ref_file,
            metric_model,
            official_root,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
        )
        labels = parse_official_results(result_file)

        eval_results: List[EvaluationResult] = []
        for r in run_results:
            meta = r.metadata or {}
            diagnostics = {
                "question_type": meta.get("question_type", "unknown"),
                "question_date": meta.get("question_date", ""),
                "is_abstention": bool(meta.get("is_abstention")),
                "num_sessions": meta.get("num_sessions", 0),
                "metric_model": metric_model,
                "evaluator": "official",
                "sessions_truncated": meta.get("sessions_truncated", False),
            }
            if r.sample_id not in labels:
                eval_results.append(
                    EvaluationResult(
                        sample_id=r.sample_id,
                        is_correct=False,
                        score=0.0,
                        correctness_reason=f"official autoeval ({metric_model}) — missing",
                        failure_mode="official_eval_missing",
                        diagnostics=diagnostics,
                    )
                )
                continue

            label = labels[r.sample_id]
            eval_results.append(
                EvaluationResult(
                    sample_id=r.sample_id,
                    is_correct=label,
                    score=1.0 if label else 0.0,
                    correctness_reason=f"official autoeval ({metric_model})",
                    failure_mode=None if label else "answer_mismatch",
                    diagnostics=diagnostics,
                )
            )
        return eval_results

    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """Write results.json / summary.csv / report.md via ReportWriter."""
        self.report_writer = ReportWriter(output_dir)
        summary_metrics = self.compute_summary_metrics(results)
        self.report_writer.write_evaluation_results(results)
        self.report_writer.write_csv_summary(results)
        self.report_writer.write_markdown_report(
            title="LongMemEval Benchmark Report",
            eval_results=results,
            metrics=summary_metrics,
        )
