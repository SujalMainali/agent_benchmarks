"""
Quick Testing and Validation Guide

This file explains how to validate the benchmark implementation.
"""

import subprocess
import sys
from pathlib import Path


def test_imports():
    """Test that all benchmark modules can be imported."""
    print("Testing imports...")
    try:
        from benchmarks.common import (
            BenchmarkAdapter,
            BenchmarkEvaluator,
            BenchmarkLogger,
            BenchmarkSample,
            EvaluationResult,
            EvaluatorBase,
            ReportWriter,
            RunResult,
            ToolEvent,
            TrajectoryStep,
        )
        print("✓ Common module imports OK")

        from benchmarks.locomo import (
            LoCoMoAdapter,
            LoCoMoEvaluator,
            LoCoMoLoader,
            LoCoMoRunner,
        )
        print("✓ LoCoMo module imports OK")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_structures():
    """Test that data structures work correctly."""
    print("\nTesting data structures...")
    try:
        from benchmarks.common.models import (
            BenchmarkSample,
            TrajectoryStep,
            ToolEvent,
            RunResult,
            EvaluationResult,
        )

        # Create sample instances
        sample = BenchmarkSample(
            sample_id="test_001",
            question="What is 2+2?",
            gold_answer="4",
        )
        print("✓ BenchmarkSample creation OK")

        tool_event = ToolEvent(
            tool_name="calculator",
            arguments={"expression": "2+2"},
            result="4",
        )
        print("✓ ToolEvent creation OK")

        trajectory_step = TrajectoryStep(
            turn_number=1,
            user_input="What is 2+2?",
            system_prompt="You are helpful.",
            agent_message="The answer is 4",
            tool_calls=[tool_event],
        )
        print("✓ TrajectoryStep creation OK")

        run_result = RunResult(
            sample_id="test_001",
            predicted_answer="4",
            gold_answer="4",
            trajectory=[trajectory_step],
        )
        print("✓ RunResult creation OK")

        eval_result = EvaluationResult(
            sample_id="test_001",
            is_correct=True,
            score=1.0,
            correctness_reason="Exact match",
        )
        print("✓ EvaluationResult creation OK")

        return True
    except Exception as e:
        print(f"✗ Data structure error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """Test the benchmark logger."""
    print("\nTesting logger...")
    try:
        from benchmarks.common.logger import BenchmarkLogger

        log = BenchmarkLogger("test_sample")
        log.log_turn_start("What is the capital of France?", ["System prompt"])
        log.log_message("user", "What is the capital of France?")
        log.log_agent_message("The capital of France is Paris.")
        log.log_tool_call("web_search", {"query": "capital of France"}, "Paris, France")
        log.log_memory_state({"summary": "About France", "facts": []})
        log.finalize_turn()

        # Check that logging worked
        assert len(log.trajectory) == 1
        assert log.trajectory[0].turn_number == 1
        assert len(log.raw_messages) > 0

        # Test JSON serialization
        json_str = log.to_json()
        assert "test_sample" in json_str
        assert "trajectory" in json_str

        print("✓ BenchmarkLogger OK")
        return True
    except Exception as e:
        print(f"✗ Logger error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter():
    """Test the LoCoMo adapter."""
    print("\nTesting adapter...")
    try:
        from benchmarks.locomo.adapter import LoCoMoAdapter
        from benchmarks.common.models import BenchmarkSample

        adapter = LoCoMoAdapter()

        sample = BenchmarkSample(
            sample_id="test_001",
            question="What is this about?",
            gold_answer="It's about something",
            context={
                "sessions": [
                    [
                        {
                            "user": "Tell me something",
                            "assistant": "Here's something",
                        }
                    ]
                ]
            },
        )

        agent_input = adapter.build_agent_input(sample)

        assert agent_input["question"] == "What is this about?"
        assert agent_input["gold_answer"] == "It's about something"
        assert "context_messages" in agent_input

        print("✓ LoCoMoAdapter OK")
        return True
    except Exception as e:
        print(f"✗ Adapter error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test the LoCoMo evaluator."""
    print("\nTesting evaluator...")
    try:
        from benchmarks.locomo.evaluator import LoCoMoEvaluator
        from benchmarks.common.models import RunResult

        evaluator = LoCoMoEvaluator()

        result = RunResult(
            sample_id="test_001",
            predicted_answer="Paris",
            gold_answer="Paris",
            metadata={"category": "factual"},
        )

        eval_result = evaluator.evaluate(result)

        assert eval_result.is_correct
        assert eval_result.score == 1.0
        assert eval_result.correctness_reason == "Exact match"

        print("✓ LoCoMoEvaluator OK")
        return True
    except Exception as e:
        print(f"✗ Evaluator error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    try:
        from benchmarks.locomo.metrics import LoCoMoMetrics
        from benchmarks.common.models import RunResult, TrajectoryStep

        step = TrajectoryStep(
            turn_number=1,
            user_input="Test",
            system_prompt="Test",
            agent_message="This is a test answer",
        )

        result = RunResult(
            sample_id="test_001",
            predicted_answer="Test answer",
            gold_answer="Test",
            trajectory=[step],
        )

        metrics = LoCoMoMetrics.compute_metrics(result)

        assert "answer_length_chars" in metrics
        assert "answer_length_words" in metrics
        assert "turn_count" in metrics
        assert "tool_call_count" in metrics

        print("✓ LoCoMoMetrics OK")
        return True
    except Exception as e:
        print(f"✗ Metrics error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("BENCHMARK IMPLEMENTATION VALIDATION")
    print("=" * 60 + "\n")

    tests = [
        test_imports,
        test_data_structures,
        test_logger,
        test_adapter,
        test_evaluator,
        test_metrics,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    if all(results):
        print("✓ All tests passed! Implementation is valid.")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
