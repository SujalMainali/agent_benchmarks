"""Main entry point for BFCL benchmark execution.

Usage:
    ./AgentEnv/bin/python -m benchmarks.bfcl.run

Configuration is environment-driven (BFCL_* variables; see config.py).
Categories are resolved through the official ``TEST_COLLECTION_MAPPING`` so
collection names (``single_turn``, ``live``, ...) and future categories work
without code changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.llm import build_provider

from benchmarks.bfcl.config import BFCLSettings, load_bfcl_settings
from benchmarks.bfcl.official import bootstrap_official


def _resolve_categories(settings: BFCLSettings) -> List[str]:
    """Expand category/collection names via official constants."""
    from bfcl_eval.constants.category_mapping import (
        ALL_CATEGORIES,
        TEST_COLLECTION_MAPPING,
    )
    from bfcl_eval.utils import is_agentic, is_multi_turn, is_format_sensitivity

    resolved: List[str] = []
    for name in settings.test_categories:
        if name in TEST_COLLECTION_MAPPING:
            resolved.extend(TEST_COLLECTION_MAPPING[name])
        elif name in ALL_CATEGORIES:
            resolved.append(name)
        else:
            print(f"Warning: unknown BFCL test category '{name}' - skipped.")

    # First integration iteration scores single-turn categories (AST +
    # relevance). Multi-turn/agentic/format-sensitivity require the official
    # stateful execution loop and are deferred; skip with a notice instead of
    # producing bogus scores.
    supported: List[str] = []
    for category in dict.fromkeys(resolved):  # dedupe, keep order
        if is_multi_turn(category) or is_agentic(category) or is_format_sensitivity(category):
            print(
                f"Notice: category '{category}' needs the official multi-turn/"
                "agentic loop, which this integration does not run yet - skipped."
            )
        else:
            supported.append(category)
    return supported


def main() -> None:
    settings = load_bfcl_settings()
    bootstrap_official(settings.official_root)

    # Imports that require the official package on sys.path.
    from benchmarks.bfcl.evaluator import BFCLEvaluator
    from benchmarks.bfcl.loader import BFCLLoader
    from benchmarks.bfcl.report import BFCLReporter
    from benchmarks.bfcl.runner import BFCLRunner

    print("BFCL benchmark configuration")
    print(f"  official_root: {settings.official_root}")
    print(f"  test_categories: {settings.test_categories}")
    print(f"  run_ids: {settings.run_ids or 'None'}")
    print(f"  max_samples: {settings.max_samples if settings.max_samples else 'None'}")
    print(f"  max_tool_steps: {settings.max_tool_steps}")
    print(f"  output_dir: {settings.output_dir}")
    print(f"  checker_model_name: {settings.checker_model_name}")

    categories = _resolve_categories(settings)
    if not categories:
        print("Error: no runnable BFCL test categories selected.")
        sys.exit(1)
    print(f"  resolved categories: {categories}")

    # Load episodes from the official datasets.
    loader = BFCLLoader()
    episodes = loader.load_categories(
        categories,
        max_samples=settings.max_samples,
        run_ids=settings.run_ids or None,
    )
    if not episodes:
        print("Error: no test entries matched the requested categories/ids.")
        sys.exit(1)
    print(f"Loaded {len(episodes)} test entries.")

    # Build the LLM through the provider factory (never a raw SDK).
    print("Setting up agent LLM provider...")
    agent_settings = load_settings()
    llm = build_provider(agent_settings)

    # Standardized immutable run layout (see ResultFormat.md). Raw artifacts
    # are written ACTIVELY as each entry finishes.
    reporter = BFCLReporter(
        results_root=settings.output_dir,
        dataset=",".join(settings.test_categories),
        run_metadata={
            "llm_provider": getattr(agent_settings, "llm_provider", None),
            "model_id": getattr(agent_settings, "model_id", None),
            "temperature": getattr(agent_settings, "temperature", None),
            "checker_model_name": settings.checker_model_name,
            "max_tool_steps": settings.max_tool_steps,
            "resolved_categories": categories,
        },
    )

    # Run.
    runner = BFCLRunner(llm=llm, max_tool_steps=settings.max_tool_steps)
    run_results = runner.run_batch(
        episodes, verbose=settings.verbose, on_result=reporter.writer.write_raw
    )

    # Evaluate with the official checkers.
    print("Evaluating with the official BFCL evaluator...")
    evaluator = BFCLEvaluator(checker_model_name=settings.checker_model_name)
    eval_results = [evaluator.evaluate_run_result(r) for r in run_results]
    for run_result, eval_result in zip(run_results, eval_results):
        run_result.official_eval = eval_result.diagnostics.get("official_eval", {})

    # Report.
    print(f"Writing reports under {settings.output_dir}...")
    for index, (run_result, eval_result) in enumerate(zip(run_results, eval_results)):
        reporter.writer.append_case(
            run_result,
            eval_result,
            index,
            **reporter.case_fields(run_result, eval_result),
        )
    run_dir = reporter.finalize(run_results, eval_results)

    # Summary.
    total = len(eval_results)
    correct = sum(1 for r in eval_results if r.is_correct)
    print(f"\n{'=' * 50}")
    print("BFCL Summary")
    print(f"Total entries: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total:.2%}" if total else "Accuracy: n/a")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
