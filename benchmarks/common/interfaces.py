"""Abstract interfaces for all benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import BenchmarkSample, EvaluationResult, RunResult


class BenchmarkAdapter(ABC):
    """Abstract adapter to convert benchmark samples into agent inputs."""

    @abstractmethod
    def load_sample(self, sample_data: Dict[str, Any]) -> BenchmarkSample:
        """
        Load and normalize a raw sample from the benchmark dataset.

        Args:
            sample_data: Raw sample dictionary from the benchmark.

        Returns:
            BenchmarkSample: Normalized sample with question, gold_answer, and context.
        """

    @abstractmethod
    def build_agent_input(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """
        Convert a BenchmarkSample into agent-specific input format.

        For ResearchHelperAgent, this returns something like:
        {
            "context_messages": [...],
            "question": "...",
            "gold_answer": "...",
            "mode": "plain_qa"
        }

        Args:
            sample: Normalized BenchmarkSample.

        Returns:
            Dict with "question" and "gold_answer" at minimum.
        """


class BenchmarkEvaluator(ABC):
    """Abstract evaluator to score agent outputs."""

    @abstractmethod
    def evaluate(self, result: RunResult) -> EvaluationResult:
        """
        Compare predicted_answer against gold_answer and return scores.

        Args:
            result: RunResult from agent execution.

        Returns:
            EvaluationResult with correctness score and diagnostics.
        """

    @abstractmethod
    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """
        Write evaluation results to human-readable and JSON formats.

        Args:
            results: List of EvaluationResult objects.
            output_dir: Directory to write reports.
        """
