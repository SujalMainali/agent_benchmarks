"""Bridge to the official LoCoMo QA evaluator.

This module provides integration with the vendored official LoCoMo benchmark
evaluation code to score predictions using the official metrics.

The official evaluator provides category-aware scoring using F1 metrics that
vary by question category (1: multi-hop, 2: single-hop, 3: temporal,
4: open-domain, 5: adversarial).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.common.models import EvaluationResult, RunResult


def export_predictions_for_official_eval(
    run_results: List[RunResult],
    locomo_data: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Export predictions in the format expected by the official LoCoMo evaluator.

    Args:
        run_results: List of RunResult objects from our benchmark.
        locomo_data: The original LoCoMo dataset entries (contains category, answer, evidence, etc).

    Returns:
        List of dicts with structure:
        {
            'sample_id': str,
            'qa': [
                {
                    'category': int,
                    'answer': str,
                    'evidence': list,
                    'prediction': str,  # our predicted answer
                    ...
                },
                ...
            ]
        }
    """
    # Index the original data by sample_id for quick lookup when available
    data_by_id = {d.get("sample_id"): d for d in locomo_data} if locomo_data else {}

    export = []
    for run_result in run_results:
        sample_id = run_result.sample_id
        original = data_by_id.get(sample_id)
        if original is None and "_" in sample_id:
            original = data_by_id.get(sample_id.rsplit("_", 1)[0])

        if original and isinstance(original.get("qa"), list):
            original_qa = original.get("qa", [])
            qa_items = [
                {
                    "question": original_qa_item.get("question", run_result.question),
                    "category": _normalize_category(original_qa_item.get("category", run_result.metadata.get("category", 4))),
                    "answer": original_qa_item.get("answer", run_result.gold_answer),
                    "evidence": original_qa_item.get("evidence", []),
                    "prediction": run_result.predicted_answer,
                    "prediction_context": original.get("conversation", {}),
                }
                for original_qa_item in original_qa
            ]
        else:
            qa_items = [
                {
                    "question": run_result.question,
                    "category": _normalize_category(run_result.metadata.get("category", 4)),
                    "answer": run_result.gold_answer,
                    "evidence": run_result.metadata.get("evidence", []),
                    "prediction": run_result.predicted_answer,
                }
            ]

        export.append(
            {
                "sample_id": sample_id,
                "qa": qa_items,
            }
        )

    return export


def _normalize_category(category: Any) -> int:
    """Normalize category values to the official LoCoMo integer labels."""
    if isinstance(category, int):
        return category

    if isinstance(category, str):
        cleaned = category.strip().lower()
        mapping = {
            "multi-hop": 1,
            "multihop": 1,
            "single-hop": 2,
            "singlehop": 2,
            "temporal": 3,
            "open-domain": 4,
            "opendomain": 4,
            "adversarial": 5,
        }
        if cleaned in mapping:
            return mapping[cleaned]
        try:
            return int(cleaned)
        except ValueError:
            return 4

    return 4


def run_official_evaluation(
    run_results: List[RunResult],
    locomo_data: Optional[List[Dict[str, Any]]] = None,
) -> List[EvaluationResult]:
    """
    Score predictions using the official LoCoMo QA evaluator.

    This function:
    1. Exports predictions in the official format
    2. Calls the official eval_question_answering function
    3. Parses results back into our EvaluationResult format

    Args:
        run_results: List of RunResult objects.
        locomo_data: Optional original LoCoMo dataset.

    Returns:
        List of EvaluationResult objects with official scores.
    """
    # Add the official LoCoMo repo to path
    locomo_repo = Path(__file__).parents[2] / "third_party" / "locomo-official"
    if str(locomo_repo) not in sys.path:
        sys.path.insert(0, str(locomo_repo))

    # Import the official evaluator
    try:
        from task_eval.evaluation import eval_question_answering
    except ImportError as e:
        missing_details = []
        for module_name in ("bert_score", "rouge", "nltk"):
            try:
                __import__(module_name)
            except Exception:
                missing_details.append(module_name)

        dependency_hint = (
            f"Missing official evaluator dependencies: {', '.join(missing_details)}. "
            if missing_details
            else ""
        )
        raise ImportError(
            f"Could not import official LoCoMo evaluator from {locomo_repo}. "
            f"Make sure third_party/locomo-official is present and its Python dependencies are installed. "
            f"{dependency_hint}Error: {e}"
        )

    # Export predictions
    export_data = export_predictions_for_official_eval(run_results, locomo_data)

    # Score predictions using official evaluator
    results = []
    for export_item in export_data:
        sample_id = export_item["sample_id"]
        qa_list = export_item["qa"]

        # Run official evaluator on this sample's QA items
        scores, _, recall = eval_question_answering(qa_list, eval_key="prediction", metric="f1")

        # Find the matching RunResult and create EvaluationResult
        run_result = next((r for r in run_results if r.sample_id == sample_id), None)
        if not run_result:
            continue

        # Aggregate scores (take max if multiple QA items)
        official_score = float(max(scores)) if scores else 0.0
        official_recall = float(max(recall)) if recall else 0.0
        is_correct = official_score >= 0.5  # threshold for "correct"

        # Extract category from the original QA item
        category = qa_list[0].get("category", 4) if qa_list else 4

        evaluation_result = EvaluationResult(
            sample_id=sample_id,
            is_correct=is_correct,
            score=official_score,
            correctness_reason=f"Official F1 score: {official_score:.3f}",
            diagnostics={
                "official_score": official_score,
                "official_recall": official_recall,
                "category": category,
                "num_qa_items": len(qa_list),
            },
        )

        results.append(evaluation_result)

    return results


def parse_official_output(
    output_dir: str,
) -> Dict[str, Any]:
    """
    Parse the official evaluation output directory.

    This is a placeholder for future use if we write official output to disk.

    Args:
        output_dir: Directory containing official evaluation output.

    Returns:
        Dictionary of parsed results.
    """
    # TODO: implement if official scripts write to disk
    return {}
