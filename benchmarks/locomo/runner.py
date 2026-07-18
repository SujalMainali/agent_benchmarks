"""LoCoMo runner - orchestrates benchmark execution."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from benchmarks.common.driver import RuntimeSpec
from benchmarks.common.logger import BenchmarkLogger
from benchmarks.common.models import Action, BenchmarkSample, Episode, RunResult, TrajectoryEvent
from benchmarks.locomo.environment import LoCoMoEnvironment


class LoCoMoRunner:
    """Runs LoCoMo samples through any AgentRuntime (see AgentInterface.md)."""

    def __init__(self, runtime_or_driver: Any, spec: Optional[RuntimeSpec] = None) -> None:
        """
        Initialize the runner.

        Args:
            runtime_or_driver: either a prebuilt ``AgentRuntime`` (anything
                duck-typed with ``reset``/``act``), or an ``AgentDriver``
                whose ``create_runtime`` will be called once with ``spec``.
            spec: the RuntimeSpec used when a driver is given. Defaults to a
                bare ``RuntimeSpec(benchmark="locomo")``.
        """
        if hasattr(runtime_or_driver, "create_runtime"):
            self.runtime = runtime_or_driver.create_runtime(
                spec or RuntimeSpec(benchmark="locomo")
            )
        elif hasattr(runtime_or_driver, "act"):
            self.runtime = runtime_or_driver
        else:
            raise TypeError(
                "LoCoMoRunner needs an AgentRuntime (reset/act) or an "
                "AgentDriver (create_runtime); got "
                f"{type(runtime_or_driver).__name__}. See DriverInterface.md."
            )
        self.environment = LoCoMoEnvironment()

    def run_sample(self, sample: BenchmarkSample | Episode, log: Optional[BenchmarkLogger] = None) -> RunResult:
        """
        Run a single LoCoMo sample through the runtime/environment pair.

        The flow:
        1. Reset the environment
        2. Reset the runtime with the initial environment snapshot
        3. Observe the final benchmark question
        4. Act once
        5. Collect the trajectory and raw trace

        Args:
            sample: BenchmarkSample to run.
            log: Optional BenchmarkLogger for tracing. If None, creates one.

        Returns:
            RunResult with predicted answer and full trajectory.
        """
        if log is None:
            log = BenchmarkLogger(getattr(sample, "sample_id", getattr(sample, "episode_id", "unknown")))

        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        initial_state = self.environment.reset(episode)
        self.runtime.reset(episode, initial_state)

        observation = self.environment.observe()
        context_turn_count = sum(1 for msg in observation.messages if getattr(msg, "type", "") == "human")

        log.log_environment_snapshot(initial_state)
        log.log_observation(observation)
        log.log_question(observation.text, metadata={"sample_id": episode.episode_id, "benchmark_mode": episode.mode})

        try:
            start_time = time.time()
            action = self.runtime.act(observation)
            latency_ms = (time.time() - start_time) * 1000
            final_state = self.environment.step(action)

            predicted_answer = action.text
            log.log_action(action)
            log.log_environment_snapshot(final_state)

            trajectory = self.runtime.get_trajectory().events
            raw_messages = self.runtime.get_raw_messages()

            # Build the run result with all required fields
            run_result = RunResult(
                sample_id=episode.episode_id,
                episode_id=episode.episode_id,
                question=episode.question,
                predicted_answer=predicted_answer,
                gold_answer=episode.gold_answer,
                trajectory=trajectory,
                raw_messages=raw_messages,
                benchmark_mode=episode.mode,
                context_turn_count=context_turn_count,
                metrics=self._compute_metrics(latency_ms, predicted_answer, context_turn_count, trajectory),
                metadata=episode.metadata,
                total_latency_ms=latency_ms,
                episode=episode,
                final_state=final_state,
            )

            return run_result

        except Exception as e:
            # Log error
            error_msg = f"Error during agent execution: {str(e)}"
            return RunResult(
                sample_id=episode.episode_id,
                episode_id=episode.episode_id,
                question=episode.question,
                predicted_answer="",
                gold_answer=episode.gold_answer,
                trajectory=self.runtime.get_trajectory().events,
                raw_messages=self.runtime.get_raw_messages(),
                benchmark_mode=episode.mode,
                context_turn_count=context_turn_count,
                metadata=episode.metadata,
                total_latency_ms=0.0,
                episode=episode,
                error=error_msg,
            )

    def run_batch(
        self,
        samples: List[BenchmarkSample | Episode],
        verbose: bool = True,
        on_result: Optional[Any] = None,
    ) -> List[RunResult]:
        """
        Run multiple samples and collect results.

        Args:
            samples: List of BenchmarkSample objects.
            verbose: Whether to print progress.
            on_result: Optional ``(run_result, index) -> None`` callback fired
                as each sample finishes — used to write raw artifacts actively
                during the run rather than at the end of the batch.

        Returns:
            List of RunResult objects.
        """
        results = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"Running sample {i+1}/{len(samples)}: {sample.sample_id}")

            result = self.run_sample(sample)
            results.append(result)
            if on_result is not None:
                on_result(result, i)

        return results

    def _capture_memory_state(self) -> Dict[str, Any]:
        """
        Capture the current state of the agent's memory.

        Returns:
            Dictionary representation of memory state.
        """
        return {
            "summary": getattr(self.runtime.agent.memory, "summary", ""),
            "facts": getattr(self.runtime.agent.memory, "facts", []),
            "tool_results_count": len(getattr(self.runtime.agent.memory, "tool_results", [])),
            "recent_messages_count": len(getattr(self.runtime.agent.memory, "recent_messages", [])),
            "turn_count": getattr(self.runtime.agent.memory, "turn_count", 0),
        }

    def _compute_metrics(
        self,
        latency_ms: float,
        predicted_answer: str,
        context_turn_count: int,
        trajectory: List[TrajectoryEvent],
    ) -> Dict[str, Any]:
        """
        Compute per-sample metrics.

        Args:
            latency_ms: Time taken to generate the answer.
            predicted_answer: The generated answer text.
            context_turn_count: Number of turns in the context history.

        Returns:
            Dictionary of metrics for this sample.
        """
        return {
            "latency_ms": latency_ms,
            "answer_length": len(predicted_answer.split()),
            "answer_char_count": len(predicted_answer),
            "context_turn_count": context_turn_count,
            "turn_count": len(trajectory),
            "tool_call_count": sum(len(event.tool_calls) for event in trajectory),
        }
