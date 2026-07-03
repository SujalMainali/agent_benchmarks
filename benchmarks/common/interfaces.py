"""Abstract interfaces for all benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

from .models import (
    Action,
    BenchmarkSample,
    Episode,
    EvaluationContext,
    EvaluationResult,
    EnvironmentState,
    Observation,
    RunResult,
    Trajectory,
)


class BenchmarkLoader(ABC):
    """Abstract loader that normalizes raw benchmark data."""

    @abstractmethod
    def load(self, raw_data: Dict[str, Any]) -> Episode:
        """Normalize one raw benchmark record into an episode."""

    def load_many(self, raw_items: Iterable[Dict[str, Any]]) -> List[Episode]:
        return [self.load(item) for item in raw_items]


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
    def build_context_messages(self, sample: BenchmarkSample | Episode) -> list[Any]:
        """
        Build role-preserving context messages for benchmark replay.

        Args:
            sample: Normalized benchmark sample or episode.

        Returns:
            List of BaseMessage-compatible objects that preserve conversation roles.
        """

    @abstractmethod
    def build_agent_input(self, sample: BenchmarkSample | Episode) -> Dict[str, Any]:
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


class BenchmarkEnvironment(ABC):
    """Abstract environment that owns benchmark state and transitions."""

    @abstractmethod
    def reset(self, episode: Episode) -> EnvironmentState:
        """Reset the environment for a new episode."""

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current observation for the runtime."""

    @abstractmethod
    def step(self, action: Action) -> EnvironmentState:
        """Apply an action to the environment and return the updated state."""

    @abstractmethod
    def snapshot(self) -> EnvironmentState:
        """Return a serializable snapshot of the environment."""

    @abstractmethod
    def is_done(self) -> bool:
        """Indicate whether the environment has finished."""


class AgentRuntime(ABC):
    """Abstract runtime wrapper around the concrete agent implementation."""

    @abstractmethod
    def reset(self, episode: Episode, initial_state: EnvironmentState) -> None:
        """Prepare the runtime for a new episode."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Produce one action for the current observation."""

    @abstractmethod
    def get_trajectory(self) -> Trajectory:
        """Return the collected trajectory for the current episode."""

    def get_metrics(self) -> Dict[str, Any]:
        return {}


class BenchmarkReporter(ABC):
    """Abstract reporter that persists benchmark artifacts."""

    @abstractmethod
    def write(self, run: RunResult, eval_result: EvaluationResult) -> None:
        """Write the episode/run and evaluation artifacts."""


class BenchmarkEvaluator(ABC):
    """Abstract evaluator to score agent outputs."""

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """
        Compare the predicted output against the benchmark context and return scores.

        Args:
            context: EvaluationContext from agent execution.

        Returns:
            EvaluationResult with correctness score and diagnostics.
        """

    def evaluate_run_result(self, result: RunResult) -> EvaluationResult:
        """Compatibility helper for older run-result-based call sites."""
        episode = result.episode or Episode(
            episode_id=result.episode_id or result.sample_id,
            task=BenchmarkSample(
                sample_id=result.sample_id,
                question=result.question,
                gold_answer=result.gold_answer,
                context={"metadata": result.metadata},
                mode=result.benchmark_mode,
                metadata=result.metadata,
            ).to_task(),
            metadata=result.metadata,
        )
        return self.evaluate(
            EvaluationContext(
                episode=episode,
                trajectory=result.trajectory,
                environment_state=result.final_state,
                predicted_output=result.predicted_answer,
                metadata=result.metadata,
                official_metadata=result.official_eval or {},
                run_result=result,
            )
        )

    @abstractmethod
    def write_report(self, results: list[EvaluationResult], output_dir: str) -> None:
        """
        Write evaluation results to human-readable and JSON formats.

        Args:
            results: List of EvaluationResult objects.
            output_dir: Directory to write reports.
        """
