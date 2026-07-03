"""LoCoMo runner - orchestrates benchmark execution."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from benchmarks.common.logger import BenchmarkLogger
from benchmarks.common.models import BenchmarkSample, RunResult
from src.memory import TemporaryMemory

if TYPE_CHECKING:
    from src.agent import ResearchHelperAgent


class LoCoMoRunner:
    """Runs LoCoMo samples through the ResearchHelperAgent."""

    def __init__(self, agent: ResearchHelperAgent) -> None:
        """
        Initialize the runner.

        Args:
            agent: ResearchHelperAgent instance to use for inference.
        """
        self.agent = agent

    def run_sample(self, sample: BenchmarkSample, log: Optional[BenchmarkLogger] = None) -> RunResult:
        """
        Run a single LoCoMo sample through the agent.

        The flow:
        1. Reset the agent's memory
        2. Load context messages from the sample's conversation history
        3. Replay all context messages with role-preserving add_message
        4. Ask the final LoCoMo question
        5. Collect the trajectory

        Args:
            sample: BenchmarkSample to run.
            log: Optional BenchmarkLogger for tracing. If None, creates one.

        Returns:
            RunResult with predicted answer and full trajectory.
        """
        if log is None:
            log = BenchmarkLogger(sample.sample_id)

        # Reset memory for clean state
        self.agent.memory = TemporaryMemory()

        # Reconstruct agent input using the adapter
        from .adapter import LoCoMoAdapter

        adapter = LoCoMoAdapter()
        agent_input = adapter.build_agent_input(sample)

        # Get context messages (with role-preserved HumanMessage and AIMessage)
        context_messages = agent_input.get("context_messages", [])
        context_turn_count = agent_input.get("context_turn_count", 0)
        
        # Replay context messages into memory with correct roles
        for msg in context_messages:
            self.agent.memory.add_message(msg)
            # Also log them to the trace
            log.log_context_message(msg)

        # Run the turn with the final question
        question = agent_input["question"]
        gold_answer = agent_input["gold_answer"]
        benchmark_mode = agent_input["mode"]

        log.log_question(question, metadata={"sample_id": sample.sample_id, "benchmark_mode": benchmark_mode})

        # Log the turn start
        system_prompts = [
            self.agent.memory.summary if self.agent.memory.summary.strip() else "No summary",
        ]
        if self.agent.memory.facts:
            system_prompts.append(f"Facts: {self.agent.memory.format_facts()}")

        log.log_turn_start(question, system_prompts)

        # Execute the agent turn
        try:
            start_time = time.time()
            predicted_answer = self.agent.run_turn(question)
            latency_ms = (time.time() - start_time) * 1000

            # Log the result
            log.log_agent_message(predicted_answer)
            log.log_memory_state(self._capture_memory_state())
            log.finalize_turn()

            # Build the run result with all required fields
            run_result = RunResult(
                sample_id=sample.sample_id,
                question=question,  # Now stored directly
                predicted_answer=predicted_answer,
                gold_answer=gold_answer,
                trajectory=log.trajectory,
                raw_messages=log.raw_messages,
                benchmark_mode=benchmark_mode,
                context_turn_count=context_turn_count,
                metrics=self._compute_metrics(latency_ms, predicted_answer, context_turn_count),
                metadata=sample.metadata,
                total_latency_ms=log.get_total_latency_ms(),
            )

            return run_result

        except Exception as e:
            # Log error
            error_msg = f"Error during agent execution: {str(e)}"
            return RunResult(
                sample_id=sample.sample_id,
                question=question,
                predicted_answer="",
                gold_answer=gold_answer,
                trajectory=log.trajectory,
                raw_messages=log.raw_messages,
                benchmark_mode=benchmark_mode,
                context_turn_count=context_turn_count,
                metadata=sample.metadata,
                total_latency_ms=log.get_total_latency_ms(),
                error=error_msg,
            )

    def run_batch(
        self, samples: List[BenchmarkSample], verbose: bool = True
    ) -> List[RunResult]:
        """
        Run multiple samples and collect results.

        Args:
            samples: List of BenchmarkSample objects.
            verbose: Whether to print progress.

        Returns:
            List of RunResult objects.
        """
        results = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"Running sample {i+1}/{len(samples)}: {sample.sample_id}")

            result = self.run_sample(sample)
            results.append(result)

        return results

    def _feed_context(self, log: BenchmarkLogger, context_messages: List[BaseMessage]) -> None:
        """
        Log context messages that are fed to the agent.

        Args:
            log: BenchmarkLogger instance.
            context_messages: List of context messages.
        """
        for msg in context_messages:
            if isinstance(msg, HumanMessage):
                log.log_message("user", msg.content)
            elif isinstance(msg, AIMessage):
                log.log_message("assistant", msg.content)

    def _capture_memory_state(self) -> Dict[str, Any]:
        """
        Capture the current state of the agent's memory.

        Returns:
            Dictionary representation of memory state.
        """
        return {
            "summary": self.agent.memory.summary,
            "facts": self.agent.memory.facts,
            "tool_results_count": len(self.agent.memory.tool_results),
            "recent_messages_count": len(self.agent.memory.recent_messages),
            "turn_count": self.agent.memory.turn_count,
        }

    def _compute_metrics(self, latency_ms: float, predicted_answer: str, context_turn_count: int) -> Dict[str, Any]:
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
        }
