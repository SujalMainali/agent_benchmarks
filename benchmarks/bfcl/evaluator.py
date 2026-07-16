"""BFCL evaluator — scores bridge outputs with the official BFCL checkers.

No AST matching, parameter matching, type coercion, or multiple-call
comparison is implemented here. Each entry is scored by the official
per-entry helpers from ``bfcl_eval.eval_checker.eval_runner``:

- ``_evaluate_single_ast_entry``       (AST categories)
- ``_evaluate_single_relevance_entry`` (relevance / irrelevance categories)

Those helpers drive a ``handler.decode_ast`` call; we hand them a minimal
handler shim that returns the bridge's already-decoded output (built from the
agent's structured tool calls), so decoding never re-parses text and checker
logic stays 100% official.

Ground truth is loaded from the official ``possible_answer`` files via the
official ``load_ground_truth_entry`` helper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmarks.common.interfaces import BenchmarkEvaluator
from benchmarks.common.models import EvaluationContext, EvaluationResult

from .official import bootstrap_official

bootstrap_official()

from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.eval_checker.eval_runner import (  # noqa: E402
    _evaluate_single_ast_entry,
    _evaluate_single_relevance_entry,
)
from bfcl_eval.utils import (  # noqa: E402
    is_java,
    is_js,
    is_relevance_or_irrelevance,
    load_ground_truth_entry,
)


class _BridgeHandlerShim:
    """Quacks like a BFCL handler for the official per-entry evaluators.

    The official helpers only call ``decode_ast`` / ``decode_execute``. The
    bridge already produced both from structured ``Action.tool_calls``, so the
    shim simply returns them. An empty decode (no tool calls) is returned as
    ``[]``, which the official relevance logic already treats as
    "no function call" via ``is_empty_output``.
    """

    def __init__(
        self, decoded_ast: List[Dict[str, Any]], decoded_execute: List[str]
    ) -> None:
        self._decoded_ast = decoded_ast
        self._decoded_execute = decoded_execute

    def decode_ast(self, result: Any, language: Any = None, has_tool_call_tag: bool = False):
        return self._decoded_ast

    def decode_execute(self, result: Any, has_tool_call_tag: bool = False):
        return self._decoded_execute


class BFCLEvaluator(BenchmarkEvaluator):
    """Scores one bridge result per episode using the official evaluator."""

    def __init__(self, checker_model_name: str) -> None:
        """
        Args:
            checker_model_name: A model name registered in the official
                ``MODEL_CONFIG_MAPPING``. Used only for the checker's
                function-name normalization (``underscore_to_dot``); must be
                a model style whose tools cannot contain dots (e.g. an OpenAI
                FC model) because our tool binding applies the same renaming.
        """
        self.checker_model_name = checker_model_name
        self._ground_truth_cache: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Score one episode. Expects bridge output in ``context.metadata``.

        Required metadata keys: ``test_category``, ``decoded_ast``,
        ``decoded_execute``, ``raw_response``.
        """
        episode = context.episode
        metadata = context.metadata
        test_category = str(metadata.get("test_category", ""))
        entry_id = episode.episode_id

        shim = _BridgeHandlerShim(
            decoded_ast=list(metadata.get("decoded_ast", [])),
            decoded_execute=list(metadata.get("decoded_execute", [])),
        )
        raw_response = metadata.get("raw_response", "")
        prompt_entry = episode.raw_data

        if is_relevance_or_irrelevance(test_category):
            official_result = _evaluate_single_relevance_entry(
                shim,
                entry_id,
                raw_response,
                prompt_entry,
                self.checker_model_name,
                test_category,
            )
        else:
            possible_answer = self._ground_truth_for(test_category, entry_id)
            if possible_answer is None:
                return EvaluationResult(
                    sample_id=entry_id,
                    is_correct=False,
                    score=0.0,
                    correctness_reason=(
                        f"No official ground truth found for {entry_id} "
                        f"in category {test_category}."
                    ),
                    failure_mode="missing_ground_truth",
                    diagnostics={"category": test_category},
                )
            language, return_format = self._language_for(test_category)
            official_result = _evaluate_single_ast_entry(
                shim,
                entry_id,
                raw_response,
                possible_answer,
                prompt_entry,
                self.checker_model_name,
                test_category,
                language=language,
                return_format=return_format,
                has_tool_call_tag=False,
            )

        return self._to_evaluation_result(entry_id, test_category, official_result)

    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """Reports are handled by ``BFCLReporter``; kept for interface parity."""
        from .report import BFCLReporter

        BFCLReporter(output_dir).write_evaluation_results(results)

    # -- internals ----------------------------------------------------------

    def _ground_truth_for(
        self, test_category: str, entry_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch one entry's ground truth from the official answer files."""
        if test_category not in self._ground_truth_cache:
            entries = load_ground_truth_entry(test_category)
            self._ground_truth_cache[test_category] = {
                str(entry["id"]): entry for entry in entries
            }
        entry = self._ground_truth_cache[test_category].get(entry_id)
        if entry is None:
            return None
        return entry["ground_truth"]

    @staticmethod
    def _language_for(test_category: str) -> tuple[Any, Any]:
        """Language/return format via official predicates (no hardcoding)."""
        if is_java(test_category):
            return Language.JAVA, ReturnFormat.JAVA
        if is_js(test_category):
            return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
        return Language.PYTHON, ReturnFormat.PYTHON

    @staticmethod
    def _to_evaluation_result(
        entry_id: str, test_category: str, official_result: Dict[str, Any]
    ) -> EvaluationResult:
        """Convert the official per-entry verdict to the shared result type."""
        valid = bool(official_result.get("valid", False))
        errors = official_result.get("error", [])
        error_type = official_result.get("error_type")
        return EvaluationResult(
            sample_id=entry_id,
            is_correct=valid,
            score=1.0 if valid else 0.0,
            correctness_reason=(
                "Official BFCL checker: valid."
                if valid
                else "; ".join(str(e) for e in errors) or "Official BFCL checker: invalid."
            ),
            failure_mode=None if valid else str(error_type or "checker_invalid"),
            diagnostics={
                "category": test_category,
                "official_eval": official_result,
            },
        )
