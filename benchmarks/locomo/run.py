"""Main entry point for LoCoMo benchmark execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.tools.web_search import web_search
from src.tools.document_search import document_search
from src.tools.note_lookup import note_lookup
from src.tools.calculator import calculator

from benchmarks.locomo.config import load_locomo_settings
from benchmarks.locomo.loader import LoCoMoLoader
from benchmarks.locomo.runner import LoCoMoRunner
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.locomo.report import LoCoMoReporter


def setup_agent(settings) -> ResearchHelperAgent:
    """Initialize the ResearchHelperAgent with configured tools."""
    llm = HuggingFaceEndpoint(
        repo_id=settings.model_id,
        task="text-generation",
        huggingfacehub_api_token=settings.hf_token,
        provider=settings.provider,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        do_sample=settings.do_sample,
    )

    chat_model = ChatHuggingFace(llm=llm)

    tools = [
        calculator,
        document_search,
        note_lookup,
        web_search,
    ]

    return ResearchHelperAgent(
        chat_model=chat_model,
        tools=tools,
        max_tool_steps=settings.max_tool_steps,
    )


def run_single_sample(
    data_file: str,
    sample_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run a single LoCoMo sample.

    Args:
        data_file: Path to LoCoMo data file (JSON or JSONL).
        sample_id: Optional sample ID to run. If None, runs first sample.
        output_dir: Directory to save results. Defaults to ./results/.
    """
    if output_dir is None:
        output_dir = "results"

    # Load settings and setup agent
    print("Setting up agent...")
    settings = load_settings()
    agent = setup_agent(settings)

    # Load sample
    print(f"Loading samples from {data_file}...")
    loader = LoCoMoLoader()
    samples = loader.load_from_jsonl(data_file)

    # Filter by sample_id if provided
    if sample_id:
        samples = [s for s in samples if s.sample_id == sample_id]
        if not samples:
            print(f"Sample {sample_id} not found")
            return

    if not samples:
        print("No samples found")
        return

    # Run first sample
    sample = samples[0]
    print(f"Running sample: {sample.sample_id}")

    runner = LoCoMoRunner(agent)
    run_result = runner.run_sample(sample)

    # Evaluate
    evaluator = LoCoMoEvaluator()
    eval_result = evaluator.evaluate(run_result)

    # Report
    sample_output_dir = os.path.join(output_dir, sample.sample_id)
    reporter = LoCoMoReporter(sample_output_dir)
    reporter.write_per_sample_report(run_result, eval_result)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Sample: {sample.sample_id}")
    print(f"Question: {sample.question}")
    print(f"Gold Answer: {sample.gold_answer}")
    print(f"Predicted Answer: {run_result.predicted_answer}")
    print(f"Score: {eval_result.score:.2f}")
    print(f"Correct: {eval_result.is_correct}")
    print(f"Reason: {eval_result.correctness_reason}")
    print(f"Total Latency: {run_result.total_latency_ms:.2f}ms")
    print(f"Turns: {len(run_result.trajectory)}")
    print(f"Results saved to: {sample_output_dir}")


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

    # Load settings and setup agent
    print("Setting up agent...")
    settings = load_settings()
    agent = setup_agent(settings)

    # Load samples
    print(f"Loading samples from {data_file}...")
    loader = LoCoMoLoader()
    samples = loader.load_from_jsonl(data_file)

    if max_samples:
        samples = samples[:max_samples]

    print(f"Running {len(samples)} samples...")

    # Run batch
    runner = LoCoMoRunner(agent)
    run_results = runner.run_batch(samples, verbose=verbose)

    # Evaluate
    print("Evaluating results...")
    evaluator = LoCoMoEvaluator()
    eval_results = [evaluator.evaluate(r) for r in run_results]

    # Report
    print(f"Writing reports to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    reporter = LoCoMoReporter(output_dir)

    # Write overall report
    reporter.write_full_report(run_results, eval_results)

    # Write per-sample reports
    for run_result, eval_result in zip(run_results, eval_results):
        reporter.write_per_sample_report(run_result, eval_result)

    # Print summary
    correct = sum(1 for r in eval_results if r.is_correct)
    total = len(eval_results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Batch Summary")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_dir}")


def main() -> None:
    settings = load_locomo_settings()

    print("LoCoMo benchmark configuration")
    print(f"  run_mode: {settings.run_mode}")
    print(f"  data_file: {settings.data_file}")
    print(f"  output_dir: {settings.output_dir}")
    print(f"  sample_id: {settings.sample_id or 'None'}")
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

    run_single_sample(settings.data_file, settings.sample_id, settings.output_dir)


if __name__ == "__main__":
    main()
