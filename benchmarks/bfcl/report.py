"""BFCL reporting on the standardized ResultFormat layout.

Official checker verdicts land in each case's ``benchmark_specific`` block; a
mirror of the official ``result/.../*_result.json`` JSON-lines layout (id +
result + latency + token counts + inference log per LOG_GUIDE.md) is written
under the run's ``raw/logs/official_format/``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from benchmarks.common.base_reporter import StandardReporter
from benchmarks.common.models import EvaluationResult, RunResult


class BFCLReporter(StandardReporter):
    """Writes BFCL runs in the standardized immutable run layout."""

    benchmark = "bfcl"

    def __init__(
        self,
        *,
        dataset: str = "bfcl",
        results_root: str = "results",
        benchmark_version: str = "v4",
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
        category = str(
            eval_result.diagnostics.get(
                "category", metadata.get("test_category", "unknown")
            )
        )
        return {
            "task_family": "function_calling",
            "task_type": category,
            "benchmark_metrics": run_result.metrics,
            "expected_tool_behavior": metadata.get("possible_answer"),
            "benchmark_specific": {
                "test_category": category,
                "official_eval": run_result.official_eval,
                "raw_response": metadata.get("raw_response", ""),
            },
        }

    def aggregate_metrics(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
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
            "per_category": self._per_category_metrics(eval_results),
        }

    def finalize(
        self,
        run_results: Sequence[RunResult],
        eval_results: Sequence[EvaluationResult],
    ) -> str:
        self._write_official_style_results(list(run_results))
        return super().finalize(run_results, eval_results)

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

        result_dir = os.path.join(self.writer.logs_dir, "official_format")
        os.makedirs(result_dir, exist_ok=True)
        for category, results in by_category.items():
            path = os.path.join(result_dir, f"BFCL_{category}_result.json")
            with open(path, "w", encoding="utf-8") as f:
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
    def _per_category_metrics(
        eval_results: Sequence[EvaluationResult],
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
