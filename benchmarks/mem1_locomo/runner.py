"""
MEM1 LoCoMo Runner - Orchestrates MEM1 evaluation on LoCoMo.

Reuses existing LoCoMo components (loader, adapter, evaluator)
but runs Mem1AgentRuntime instead of the default agent.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import time

from benchmarks.mem1_locomo.config import Mem1LoCoMoSettings, load_mem1_locomo_settings
from src.mem1.runtime import Mem1AgentRuntime
from src.mem1.retriever import Mem1Retriever


@dataclass
class Mem1RunResult:
    """Result from running MEM1 on a single sample."""
    sample_id: str
    question: str
    expected: str
    prediction: str
    final_think: str
    reasoning_steps: int
    search_count: int
    latency_ms: float
    trajectory: Any = None
    think_history: list[str] = field(default_factory=list)


class Mem1LoCoMoRunner:
    """
    Runs MEM1 agent on LoCoMo benchmark samples.
    
    Uses:
    - LoCoMoLoader to load samples
    - LoCoMoAdapter to prepare context
    - Mem1AgentRuntime to run the agent
    - LoCoMoEvaluator to score results
    """
    
    def __init__(self, settings: Optional[Mem1LoCoMoSettings] = None):
        self.settings = settings or load_mem1_locomo_settings()
        
        # Create retriever with corpus support for LoCoMo
        self.retriever = Mem1Retriever(
            retriever_type="corpus",  # Use conversation as corpus
            top_k=self.settings.mem1.top_k_results,
            max_chars=self.settings.mem1.max_information_chars,
        )
        
        # Create runtime
        self.runtime = Mem1AgentRuntime(
            settings=self.settings.mem1,
            retriever=self.retriever,
        )
    
    def run_sample(
        self,
        sample_id: str,
        question: str,
        expected: str,
        context: list[str],
    ) -> Mem1RunResult:
        """
        Run MEM1 on a single sample.
        
        Args:
            sample_id: Unique identifier for the sample
            question: The question to answer
            expected: Expected answer
            context: Conversation context (list of messages)
            
        Returns:
            Mem1RunResult with prediction and metadata
        """
        # Set corpus for retrieval
        self.retriever.set_corpus(context)
        
        # Create a simple episode-like object for reset
        class SimpleEpisode:
            pass
        
        episode = SimpleEpisode()
        episode.context = context
        
        # Reset runtime
        self.runtime.reset(episode=episode)
        
        # Run agent
        action = self.runtime.act(question)
        
        # Collect results
        metrics = self.runtime.get_metrics()
        trajectory = self.runtime.get_trajectory()
        last_result = self.runtime.get_last_result()
        
        return Mem1RunResult(
            sample_id=sample_id,
            question=question,
            expected=expected,
            prediction=action.get("response", ""),
            final_think=action.get("final_think", ""),
            reasoning_steps=metrics.get("reasoning_steps", 0),
            search_count=metrics.get("search_count", 0),
            latency_ms=metrics.get("latency_ms", 0.0),
            trajectory=trajectory if self.settings.save_trajectories else None,
            think_history=last_result.steps if last_result and self.settings.save_think_history else [],
        )
    
    def run_batch(
        self,
        samples: list[dict],
        progress_callback=None,
    ) -> list[Mem1RunResult]:
        """
        Run MEM1 on a batch of samples.
        
        Args:
            samples: List of sample dicts with id, question, expected, context
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of Mem1RunResult
        """
        results = []
        total = len(samples)
        
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i + 1, total)
            
            result = self.run_sample(
                sample_id=sample.get("id", f"sample_{i}"),
                question=sample.get("question", ""),
                expected=sample.get("expected", sample.get("answer", "")),
                context=sample.get("context", sample.get("conversation", [])),
            )
            results.append(result)
        
        return results