"""Main entry point for LongMemEval benchmark execution.

Runnable as ``./AgentEnv/bin/python -m benchmarks.longmemeval.run``.
Mirrors ``benchmarks/locomo/run.py`` in structure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.llm import build_provider
from src.tools.web_search import web_search
from src.tools.document_search import document_search
from src.tools.note_lookup import note_lookup
from src.tools.calculator import calculator

from benchmarks.longmemeval.adapter import LongMemEvalAdapter
from benchmarks.longmemeval.config import LongMemEvalSettings, load_longmemeval_settings
from benchmarks.longmemeval.evaluator import LongMemEvalEvaluator
from benchmarks.longmemeval.loader import LongMemEvalLoader
from benchmarks.longmemeval.prompts import LONGMEMEVAL_SYSTEM_PROMPT
from benchmarks.longmemeval.report import LongMemEvalReporter
from benchmarks.longmemeval.runner import LongMemEvalRunner


def setup_agent(settings, bench: LongMemEvalSettings) -> ResearchHelperAgent:
    """Build the ResearchHelperAgent with the LongMemEval system prompt."""
    llm = build_provider(settings)
    tools = [calculator, document_search, note_lookup, web_search]
    return ResearchHelperAgent(
        llm=llm,
        tools=tools,
        max_tool_steps=settings.max_tool_steps,
        system_prompt_override=LONGMEMEVAL_SYSTEM_PROMPT,
        allow_tools=bench.allow_tools,
    )


def _evaluate(run_results, bench: LongMemEvalSettings, evaluator: LongMemEvalEvaluator):
    """Score run results with the official judge or the local heuristic."""
    if bench.use_official_eval:
        eval_results = evaluator.evaluate_batch_official(
            run_results,
            ref_file=bench.data_file,
            metric_model=bench.metric_model,
            official_root=bench.official_root,
            output_dir=bench.output_dir,
        )
    else:
        eval_results = [evaluator.evaluate(r) for r in run_results]
    for run_result, eval_result in zip(run_results, eval_results):
        run_result.official_eval = eval_result.diagnostics
    return eval_results


def run_single(bench: LongMemEvalSettings) -> None:
    print("Setting up agent...")
    settings = load_settings()
    agent = setup_agent(settings, bench)

    print(f"Loading episode from {bench.data_file}...")
    loader = LongMemEvalLoader()
    episodes = list(
        loader.iter_episodes(
            bench.data_file,
            question_id=bench.question_id,
            question_types=bench.question_types,
            max_samples=1,
        )
    )
    if not episodes:
        print(f"No episode matched: {bench.question_id or '(first entry)'}")
        return

    runner = LongMemEvalRunner(
        agent, LongMemEvalAdapter(), max_sessions=bench.max_sessions, verbose=bench.verbose
    )
    episode = episodes[0]
    print(f"Running question {episode.episode_id} ({episode.metadata['num_sessions']} sessions)...")
    run_result = runner.run_episode(episode)

    evaluator = LongMemEvalEvaluator()
    eval_results = _evaluate([run_result], bench, evaluator)
    eval_result = eval_results[0]

    reporter = LongMemEvalReporter(bench.output_dir, full_trajectory=bench.full_trajectory)
    reporter.write_per_sample_report(run_result, eval_result)

    print(f"\n{'=' * 50}")
    print(f"Question ({episode.metadata.get('question_type')}): {run_result.question}")
    print(f"Gold: {run_result.gold_answer}")
    print(f"Predicted: {run_result.predicted_answer}")
    print(f"Correct: {eval_result.is_correct}  Score: {eval_result.score:.3f}")
    print(f"Reason: {eval_result.correctness_reason}")
    if run_result.error:
        print(f"Error: {run_result.error}")
    print(f"Results saved to: {os.path.join(bench.output_dir, run_result.sample_id)}")


def run_batch(bench: LongMemEvalSettings) -> None:
    print("Setting up agent...")
    settings = load_settings()
    agent = setup_agent(settings, bench)

    loader = LongMemEvalLoader()
    episodes_iter = loader.iter_episodes(
        bench.data_file,
        question_types=bench.question_types,
        max_samples=bench.max_samples,
    )

    runner = LongMemEvalRunner(
        agent, LongMemEvalAdapter(), max_sessions=bench.max_sessions, verbose=bench.verbose
    )
    print("Running batch (streaming)...")
    run_results = runner.run_batch(episodes_iter, verbose=bench.verbose)
    if not run_results:
        print("No episodes ran.")
        return

    evaluator = LongMemEvalEvaluator()
    eval_results = _evaluate(run_results, bench, evaluator)

    reporter = LongMemEvalReporter(bench.output_dir, full_trajectory=bench.full_trajectory)
    reporter.write_full_report(run_results, eval_results)
    for run_result, eval_result in zip(run_results, eval_results):
        reporter.write_per_sample_report(run_result, eval_result)

    agg = reporter._aggregate_metrics(run_results, eval_results)
    print(f"\n{'=' * 50}")
    print("LongMemEval Summary")
    print(f"Total questions: {agg['total_samples']}")
    print(f"Overall accuracy: {agg['overall_accuracy']:.2%}")
    print(f"Task-averaged accuracy: {agg['task_averaged_accuracy']:.2%}")
    print(f"Abstention accuracy: {agg['abstention_accuracy']:.2%} ({agg['abstention_total']} qs)")
    print("Per-type accuracy:")
    for qtype, acc in agg["per_type_accuracy"].items():
        counts = agg["per_type_counts"][qtype]
        print(f"  {qtype}: {acc:.2%} ({counts['correct']}/{counts['total']})")
    print(f"Errors: {agg['error_count']}")
    print(f"Results saved to: {bench.output_dir}")


def main() -> None:
    bench = load_longmemeval_settings()

    print("LongMemEval benchmark configuration")
    print(f"  run_mode: {bench.run_mode}")
    print(f"  data_file: {bench.data_file}")
    print(f"  output_dir: {bench.output_dir}")
    print(f"  official_root: {bench.official_root}")
    print(f"  use_official_eval: {bench.use_official_eval}")
    print(f"  metric_model: {bench.metric_model}")
    print(f"  question_id: {bench.question_id or 'None'}")
    print(f"  question_types: {bench.question_types or 'None'}")
    print(f"  max_samples: {bench.max_samples if bench.max_samples is not None else 'None'}")
    print(f"  max_sessions: {bench.max_sessions}")
    print(f"  allow_tools: {bench.allow_tools}")
    print(f"  full_trajectory: {bench.full_trajectory}")

    if not os.path.exists(bench.data_file):
        print(f"Error: Data file not found: {bench.data_file}")
        sys.exit(1)

    if bench.run_mode == "batch":
        run_batch(bench)
        return
    if bench.run_mode != "single":
        print(f"Error: Unknown LONGMEMEVAL_RUN_MODE '{bench.run_mode}'. Use 'single' or 'batch'.")
        sys.exit(1)
    run_single(bench)


if __name__ == "__main__":
    main()
