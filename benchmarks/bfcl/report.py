"""BFCL report generation — official verdicts -> repository report artifacts.

Reuses the shared ``ReportWriter`` (same layout as LoCoMo/ToolSandbox
reports) plus a BFCL result file that mirrors the official
``result/.../*_result.json`` JSON-lines layout (id + result + latency +
token counts + inference log per LOG_GUIDE.md).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from benchmarks.common.models import EvaluationResult, RunResult
from benchmarks.common.report_writer import ReportWriter


class BFCLReporter:
    """Writes BFCL benchmark artifacts using the shared report classes."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.report_writer = ReportWriter(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def write_full_report(
        self,
        run_results: List[RunResult],
        eval_results: List[EvaluationResult],
    ) -> None:
        """Write batch-level artifacts (results, summary, markdown, metrics)."""
        metrics = {
            "batch_metrics": self._batch_metrics(run_results, eval_results),
            "per_category": self._per_category_metrics(eval_results),
        }
        self.report_writer.write_metrics(metrics)
        self.report_writer.write_evaluation_results(eval_results)
        self.report_writer.write_csv_summary(eval_results)

        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.is_correct)
        self.report_writer.write_markdown_report(
            title="BFCL Benchmark Report",
            eval_results=eval_results,
            metrics={
                "total_samples": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "average_score": (
                    sum(r.score for r in eval_results) / total if total else 0.0
                ),
            },
        )

        self._write_official_style_results(run_results)

    def write_per_sample_report(
        self, run_result: RunResult, eval_result: EvaluationResult
    ) -> None:
        """Write per-entry artifacts under ``<output_dir>/<entry_id>/``."""
        subdir = run_result.sample_id
        self.report_writer.write_run_results(run_result, subdir=subdir)
        self.report_writer.write_trace(run_result, subdir=subdir)
        self.report_writer.write_evaluation(eval_result, subdir=subdir)

    def write_evaluation_results(self, eval_results: List[EvaluationResult]) -> None:
        self.report_writer.write_evaluation_results(eval_results)

    # -- internals ----------------------------------------------------------

    def _write_official_style_results(self, run_results: List[RunResult]) -> None:
        """Mirror the official result-file layout (JSON lines per category).

        Each line matches what an official handler's ``write`` would store:
        id, result (raw model response), latency, token counts, and the
        LOG_GUIDE-style inference log.
        """
        by_category: Dict[str, List[RunResult]] = defaultdict(list)
        for run_result in run_results:
            category = str(run_result.metadata.get("test_category", "unknown"))
            by_category[category].append(run_result)

        result_dir = os.path.join(self.output_dir, "official_format")
        os.makedirs(result_dir, exist_ok=True)
        for category, results in by_category.items():
            path = os.path.join(result_dir, f"BFCL_{category}_result.json")
            with open(path, "w") as f:
                for run_result in results:
                    entry: Dict[str, Any] = {
                        "id": run_result.sample_id,
                        "result": run_result.metadata.get("raw_response", ""),
                        "latency": run_result.total_latency_ms / 1000.0,
                        "input_token_count": run_result.metrics.get(
                            "input_token_count", 0
                        ),
                        "output_token_count": run_result.metrics.get(
                            "output_token_count", 0
                        ),
                        "inference_log": run_result.metadata.get("inference_log", []),
                    }
                    f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _batch_metrics(
        run_results: List[RunResult], eval_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        total = len(eval_results)
        correct = sum(1 for r in eval_results if r.is_correct)
        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "total_latency_ms": sum(r.total_latency_ms for r in run_results),
            "average_latency_ms": (
                sum(r.total_latency_ms for r in run_results) / len(run_results)
                if run_results
                else 0.0
            ),
            "errors": sum(1 for r in run_results if r.error),
        }

    @staticmethod
    def _per_category_metrics(
        eval_results: List[EvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        by_category: Dict[str, List[EvaluationResult]] = defaultdict(list)
        for result in eval_results:
            by_category[result.diagnostics.get("category", "unknown")].append(result)
        return {
            category: {
                "total": len(results),
                "correct": sum(1 for r in results if r.is_correct),
                "accuracy": (
                    sum(1 for r in results if r.is_correct) / len(results)
                    if results
                    else 0.0
                ),
            }
            for category, results in by_category.items()
        }
