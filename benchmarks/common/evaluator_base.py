"""Base evaluator class for benchmark implementations."""

from abc import abstractmethod
from typing import Dict, List, Optional

from .interfaces import BenchmarkEvaluator
from .models import EvaluationResult, RunResult


class EvaluatorBase(BenchmarkEvaluator):
    """Base class for benchmark evaluators.
    
    Provides common evaluation infrastructure that specific benchmarks can extend.
    """

    def __init__(self, name: str = "BenchmarkEvaluator") -> None:
        self.name = name
        self.results: List[EvaluationResult] = []

    def evaluate_batch(self, run_results: List[RunResult]) -> List[EvaluationResult]:
        """Evaluate multiple run results and store internally.
        
        Args:
            run_results: List of RunResult objects from agent runs.
            
        Returns:
            List of EvaluationResult objects.
        """
        self.results = [self.evaluate(result) for result in run_results]
        return self.results

    def compute_summary_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate metrics across results.
        
        Args:
            results: List of EvaluationResult objects.
            
        Returns:
            Dictionary with summary statistics (accuracy, etc).
        """
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "average_score": avg_score,
        }

    def group_results_by_category(self, results: List[EvaluationResult]) -> Dict[str, List[EvaluationResult]]:
        """Group results by question category if available.
        
        Args:
            results: List of EvaluationResult objects.
            
        Returns:
            Dictionary mapping category names to lists of results.
        """
        grouped: Dict[str, List[EvaluationResult]] = {}
        for result in results:
            category = result.diagnostics.get("category", "unknown")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(result)
        return grouped

    @abstractmethod
    def evaluate(self, result: RunResult) -> EvaluationResult:
        """Subclasses must implement the actual evaluation logic."""

    @abstractmethod
    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """Subclasses should implement report writing."""
