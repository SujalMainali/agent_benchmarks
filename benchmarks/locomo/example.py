"""
Example usage of the LoCoMo benchmark.

This script demonstrates how to:
1. Load LoCoMo samples
2. Run them through the agent
3. Evaluate results
4. Generate reports
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.tools.web_search import web_search
from src.tools.document_search import document_search
from src.tools.note_lookup import note_lookup
from src.tools.calculator import calculator
from benchmarks.locomo.config import load_locomo_settings
from benchmarks.locomo.loader import LoCoMoLoader
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.runner import LoCoMoRunner
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.locomo.report import LoCoMoReporter


def create_sample_data(demo_file: str):
    """Create a sample LoCoMo-style JSON file for testing."""
    samples = [
        {
            "sample_id": "locomo_001",
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
            "category": "factual",
            "sessions": [
                [
                    {
                        "user": "I'm planning a trip to Europe",
                        "assistant": "That sounds exciting! Where are you thinking of going?",
                    },
                    {
                        "user": "I want to visit France",
                        "assistant": "France is beautiful! The capital is Paris.",
                    },
                ]
            ],
            "evidence": ["Paris is the capital of France"],
        },
        {
            "sample_id": "locomo_002",
            "question": "What is 15 + 27?",
            "gold_answer": "42",
            "category": "arithmetic",
            "sessions": [
                [
                    {
                        "user": "I need to calculate something",
                        "assistant": "Sure, what do you need to calculate?",
                    },
                    {
                        "user": "What is 15 + 27?",
                        "assistant": "Let me calculate that for you.",
                    },
                ]
            ],
            "evidence": [],
        },
    ]

    # Write to a demo file
    demo_dir = os.path.dirname(demo_file)
    if demo_dir:
        os.makedirs(demo_dir, exist_ok=True)

    with open(demo_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created demo data file: {demo_file}")
    return demo_file


def example_basic_usage(demo_file: str, create_demo_data: bool):
    """Basic example of loading and inspecting a sample."""
    print("=" * 60)
    print("Example 1: Basic Usage - Load and Inspect Samples")
    print("=" * 60)

    # Create or reuse demo data depending on config.
    if create_demo_data:
        demo_file = create_sample_data(demo_file)
    elif not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo data file not found: {demo_file}")

    # Load samples
    loader = LoCoMoLoader()
    samples = loader.load_from_jsonl(demo_file)

    print(f"\nLoaded {len(samples)} samples:\n")
    for sample in samples:
        print(f"  ID: {sample.sample_id}")
        print(f"  Question: {sample.question}")
        print(f"  Gold Answer: {sample.gold_answer}")
        print(f"  Category: {sample.metadata.get('category', 'unknown')}")
        print()


def example_adapter_usage(demo_file: str, create_demo_data: bool):
    """Example of using the adapter to prepare agent input."""
    print("=" * 60)
    print("Example 2: Adapter Usage - Convert to Agent Input")
    print("=" * 60)

    # Create or reuse demo data depending on config.
    if create_demo_data:
        demo_file = create_sample_data(demo_file)
    elif not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demo data file not found: {demo_file}")

    # Load and adapt
    loader = LoCoMoLoader()
    samples = loader.load_from_jsonl(demo_file)

    adapter = LoCoMoAdapter()
    sample = samples[0]

    agent_input = adapter.build_agent_input(sample)

    print(f"\nOriginal Sample:")
    print(f"  ID: {sample.sample_id}")
    print(f"  Question: {sample.question}")

    print(f"\nAgent Input:")
    print(f"  Sample ID: {agent_input['sample_id']}")
    print(f"  Question: {agent_input['question']}")
    print(f"  Mode: {agent_input['mode']}")
    print(f"  Context Messages: {len(agent_input['context_messages'])} messages")
    for i, msg in enumerate(agent_input["context_messages"]):
        msg_type = type(msg).__name__
        content = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
        print(f"    [{i}] {msg_type}: {content}")


def example_evaluator_usage():
    """Example of evaluation without running the full agent."""
    print("=" * 60)
    print("Example 3: Evaluator Usage - Score Predictions")
    print("=" * 60)

    from benchmarks.common.models import RunResult

    # Create a mock run result
    run_result = RunResult(
        sample_id="locomo_001",
        question="What is the capital of France?",
        predicted_answer="Paris",
        gold_answer="Paris",
        trajectory=[],
        raw_messages=[],
        metadata={"category": "factual"},
    )

    # Evaluate
    evaluator = LoCoMoEvaluator()
    eval_result = evaluator.evaluate(run_result)

    print(f"\nRun Result:")
    print(f"  Sample ID: {run_result.sample_id}")
    print(f"  Question: {run_result.question}")
    print(f"  Gold Answer: {run_result.gold_answer}")
    print(f"  Predicted Answer: {run_result.predicted_answer}")

    print(f"\nEvaluation:")
    print(f"  Correct: {eval_result.is_correct}")
    print(f"  Score: {eval_result.score}")
    print(f"  Reason: {eval_result.correctness_reason}")


def main():
    settings = load_locomo_settings()

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  LoCoMo Benchmark - Example Usage".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run examples
    try:
        example_basic_usage(settings.data_file, settings.create_demo_data)
        print()
        example_adapter_usage(settings.data_file, settings.create_demo_data)
        print()
        example_evaluator_usage()
        print()

        print("=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        print()
        print("To run the full benchmark with your agent:")
        print("  python -m benchmarks.locomo.run")
        print()
        print("To run a single sample:")
        print("  set LOCOMO_RUN_MODE=single and LOCOMO_SAMPLE_ID=locomo_001")
        print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
