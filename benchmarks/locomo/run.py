"""Main entry point for LoCoMo benchmark execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common.driver import RuntimeSpec, resolve_driver
from benchmarks.locomo.config import load_locomo_settings
from benchmarks.locomo.prompts import (
    LOCOMO_EVIDENCE_AWARE_PROMPT,
    LOCOMO_QA_PROMPT,
    LOCOMO_STRICT_FORMAT_PROMPT,
    LOCOMO_SYSTEM_PROMPT,
)
from benchmarks.locomo.loader import LoCoMoLoader
from benchmarks.locomo.runner import LoCoMoRunner
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.locomo.report import LoCoMoReporter


def _select_locomo_prompt(prompt_mode: str) -> str:
    """Select the benchmark prompt for the requested mode."""
    if prompt_mode == "strict":
        return LOCOMO_STRICT_FORMAT_PROMPT
    if prompt_mode == "rag":
        return LOCOMO_EVIDENCE_AWARE_PROMPT
    if prompt_mode == "qa":
        return LOCOMO_QA_PROMPT
    return LOCOMO_SYSTEM_PROMPT


def _runtime_spec(benchmark_settings) -> RuntimeSpec:
    """The runtime binding LoCoMo asks of any agent driver.

    ``tools=None`` means the agent may bring its own toolset — LoCoMo
    supplies no benchmark tools of its own.
    """
    return RuntimeSpec(
        benchmark="locomo",
        system_prompt=_select_locomo_prompt(benchmark_settings.prompt_mode),
        tools=None,
        allow_tools=benchmark_settings.allow_tools,
    )


def _run_metadata(driver, benchmark_settings) -> dict:
    """Collect run-level provenance recorded in summary.json."""
    return {
        **driver.describe(),
        "prompt_mode": benchmark_settings.prompt_mode,
        "use_official_eval": benchmark_settings.use_official_eval,
        "allow_tools": benchmark_settings.allow_tools,
    }


def run_single_sample(
    data_file: str,
    sample_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    question_count: Optional[int] = None,
) -> None:
    """
    Run a single LoCoMo sample.

    Args:
        data_file: Path to LoCoMo data file (JSON or JSONL).
        sample_id: Optional sample ID to run. If None, runs first sample.
        output_dir: Directory to save results. Defaults to ./results/.
        question_count: Optional cap on the number of QA items to run for the
            targeted sample. If None, runs all of the sample's QA items.
    """
    if output_dir is None:
        output_dir = "results"

    # Resolve the agent driver (AGENT_DRIVER env var; see DriverInterface.md).
    print("Setting up agent...")
    benchmark_settings = load_locomo_settings()
    driver = resolve_driver()

    # Load sample
    print(f"Loading samples from {data_file}...")
    loader = LoCoMoLoader()
    episodes = loader.load_episodes_from_json(data_file) if data_file.endswith(".json") else loader.load_episodes_from_jsonl(data_file)

    # Filter by sample_id if provided.
    # A single raw sample (e.g. "conv-30") is expanded by the loader into one
    # episode per QA item, each with episode_id "<sample_id>_<index>" and the
    # original sample id preserved in metadata["source_sample_id"]. Match on the
    # source sample id so LOCOMO_SAMPLE_ID targets the whole sample; also allow
    # an exact episode_id match to target a single QA item.
    if sample_id:
        episodes = [
            episode
            for episode in episodes
            if episode.metadata.get("source_sample_id") == sample_id
            or episode.episode_id == sample_id
        ]
        if not episodes:
            print(f"Sample {sample_id} not found")
            return

    if not episodes:
        print("No samples found")
        return

    # When targeting a sample, run all of its QA items; otherwise run the first.
    if not sample_id:
        episodes = episodes[:1]
    elif question_count is not None and question_count > 0:
        # Cap the number of QA items asked for the targeted sample.
        episodes = episodes[:question_count]

    print(f"Running {len(episodes)} QA item(s) for sample: {sample_id or episodes[0].episode_id}")

    runner = LoCoMoRunner(driver, _runtime_spec(benchmark_settings))

    # Standardized immutable run layout (see ResultFormat.md). Raw artifacts
    # for each QA item are flushed to disk the moment it finishes, so a long
    # batch can be inspected sample-by-sample while it is still running.
    reporter = LoCoMoReporter(
        results_root=output_dir,
        agent_name=getattr(driver, "name", None),
        run_metadata=_run_metadata(driver, benchmark_settings),
    )

    def _flush_raw(run_result, index):
        run_result.benchmark_mode = benchmark_settings.prompt_mode
        reporter.writer.write_raw(run_result, index)

    run_results = runner.run_batch(episodes, verbose=True, on_result=_flush_raw)

    # Evaluate
    evaluator = LoCoMoEvaluator()
    if benchmark_settings.use_official_eval:
        eval_results = evaluator.evaluate_batch_official(run_results)
        for run_result, eval_result in zip(run_results, eval_results):
            run_result.official_eval = eval_result.diagnostics
    else:
        eval_results = [evaluator.evaluate(r) for r in run_results]

    # Processed case records + batch summary (raw already streamed above).
    for index, (run_result, eval_result) in enumerate(zip(run_results, eval_results)):
        reporter.writer.append_case(
            run_result,
            eval_result,
            index,
            **reporter.case_fields(run_result, eval_result),
        )
    run_dir = reporter.finalize(run_results, eval_results)

    # Print per-QA-item results
    for episode, run_result, eval_result in zip(episodes, run_results, eval_results):
        print(f"\n{'='*50}")
        print(f"Sample: {episode.episode_id}")
        print(f"Question: {episode.question}")
        print(f"Gold Answer: {episode.gold_answer}")
        print(f"Predicted Answer: {run_result.predicted_answer}")
        print(f"Score: {eval_result.score:.2f}")
        print(f"Correct: {eval_result.is_correct}")
        print(f"Reason: {eval_result.correctness_reason}")
        print(f"Total Latency: {run_result.total_latency_ms:.2f}ms")
        print(f"Turns: {len(run_result.trajectory)}")
    print(f"\nResults saved to: {run_dir}")

    # Print aggregate summary across the sample's QA items
    correct = sum(1 for r in eval_results if r.is_correct)
    total = len(eval_results)
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Sample Summary: {sample_id or episodes[0].episode_id}")
    print(f"Total QA items: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")


def run_batch(
    data_file: str,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Run a batch of LoCoMo samples.

    Args:
        data_file: Path to LoCoMo data file (JSON or JSONL).
        output_dir: Directory to save results. Defaults to ./results/.
        max_samples: Maximum number of samples to run. If None, runs all.
    """
    if output_dir is None:
        output_dir = "results"

    # Resolve the agent driver (AGENT_DRIVER env var; see DriverInterface.md).
    print("Setting up agent...")
    benchmark_settings = load_locomo_settings()
    driver = resolve_driver()

    # Load samples
    print(f"Loading samples from {data_file}...")
    loader = LoCoMoLoader()
    episodes = loader.load_episodes_from_json(data_file) if data_file.endswith(".json") else loader.load_episodes_from_jsonl(data_file)

    if max_samples:
        episodes = episodes[:max_samples]

    print(f"Running {len(episodes)} samples...")

    # Run batch. Standardized immutable run layout (see ResultFormat.md): raw
    # artifacts are flushed per-sample so a long batch is inspectable while it
    # is still running.
    reporter = LoCoMoReporter(
        results_root=output_dir,
        agent_name=getattr(driver, "name", None),
        run_metadata=_run_metadata(driver, benchmark_settings),
    )

    def _flush_raw(run_result, index):
        run_result.benchmark_mode = benchmark_settings.prompt_mode
        reporter.writer.write_raw(run_result, index)

    runner = LoCoMoRunner(driver, _runtime_spec(benchmark_settings))
    run_results = runner.run_batch(episodes, verbose=verbose, on_result=_flush_raw)

    # Evaluate
    print("Evaluating results...")
    evaluator = LoCoMoEvaluator()
    if benchmark_settings.use_official_eval:
        eval_results = evaluator.evaluate_batch_official(run_results)
        for run_result, eval_result in zip(run_results, eval_results):
            run_result.official_eval = eval_result.diagnostics
    else:
        eval_results = [evaluator.evaluate(r) for r in run_results]

    # Processed case records + batch summary (raw already streamed above).
    print(f"Writing reports under {output_dir}...")
    for index, (run_result, eval_result) in enumerate(zip(run_results, eval_results)):
        reporter.writer.append_case(
            run_result,
            eval_result,
            index,
            **reporter.case_fields(run_result, eval_result),
        )
    run_dir = reporter.finalize(run_results, eval_results)

    # Print summary
    correct = sum(1 for r in eval_results if r.is_correct)
    total = len(eval_results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Batch Summary")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {run_dir}")


def main() -> None:
    settings = load_locomo_settings()

    print("LoCoMo benchmark configuration")
    print(f"  run_mode: {settings.run_mode}")
    print(f"  data_file: {settings.data_file}")
    print(f"  output_dir: {settings.output_dir}")
    print(f"  official_root: {settings.official_root}")
    print(f"  use_official_eval: {settings.use_official_eval}")
    print(f"  allow_tools: {settings.allow_tools}")
    print(f"  prompt_mode: {settings.prompt_mode}")
    print(f"  sample_id: {settings.sample_id or 'None'}")
    print(f"  question_count: {settings.question_count if settings.question_count is not None else 'None'}")
    print(f"  max_samples: {settings.max_samples if settings.max_samples is not None else 'None'}")

    if not os.path.exists(settings.data_file):
        print(f"Error: Data file not found: {settings.data_file}")
        sys.exit(1)

    if settings.run_mode == "batch":
        run_batch(
            settings.data_file,
            settings.output_dir,
            settings.max_samples,
            verbose=settings.verbose,
        )
        return

    if settings.run_mode != "single":
        print(f"Error: Unknown LOCOMO_RUN_MODE '{settings.run_mode}'. Use 'single' or 'batch'.")
        sys.exit(1)

    run_single_sample(
        settings.data_file,
        settings.sample_id,
        settings.output_dir,
        settings.question_count,
    )


if __name__ == "__main__":
    main()
