"""Base evaluator class for benchmark implementations."""

from abc import abstractmethod
from typing import Dict, List

from .interfaces import BenchmarkEvaluator
from .models import EvaluationContext, EvaluationResult, RunResult, TrajectoryEvent


class EvaluatorBase(BenchmarkEvaluator):
    """Base class for benchmark evaluators.
    
    Provides common evaluation infrastructure that specific benchmarks can extend.
    """

    def __init__(self, name: str = "BenchmarkEvaluator") -> None:
        self.name = name
        self.results: List[EvaluationResult] = []

    def evaluate_batch(self, run_results: List[RunResult | EvaluationContext]) -> List[EvaluationResult]:
        """Evaluate multiple results or contexts and store internally.
        
        Args:
            run_results: List of RunResult objects or EvaluationContext objects.
            
        Returns:
            List of EvaluationResult objects.
        """
        self.results = [self.evaluate(self._coerce_context(result)) for result in run_results]
        return self.results

    def _coerce_context(self, result: RunResult | EvaluationContext) -> EvaluationContext:
        if isinstance(result, EvaluationContext):
            return result

        episode = result.episode
        if episode is None:
            from .models import BenchmarkSample

            episode = BenchmarkSample(
                sample_id=result.sample_id,
                question=result.question,
                gold_answer=result.gold_answer,
                context={"metadata": result.metadata},
                mode=result.benchmark_mode,
                metadata=result.metadata,
            ).to_episode()

        return EvaluationContext(
            episode=episode,
            trajectory=list(result.trajectory),
            environment_state=result.final_state,
            predicted_output=result.predicted_answer,
            metadata=result.metadata,
            official_metadata=result.official_eval or {},
            run_result=result,
        )

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
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Subclasses must implement the actual evaluation logic."""

    @abstractmethod
    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """Subclasses should implement report writing."""
