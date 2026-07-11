"""
MEM1 Agent Runtime - Wraps Mem1Agent for benchmark interface.

Implements the AgentRuntime interface from benchmarks/common/interfaces.py
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import time

from src.mem1.config import Mem1Settings, load_mem1_settings
from src.mem1.agent import Mem1Agent, AgentResult
from src.mem1.retriever import Mem1Retriever


@dataclass
class TrajectoryEvent:
    """A single event in the agent's trajectory."""
    event_type: str  # "input", "think", "search", "information", "output"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass 
class Trajectory:
    """Complete trajectory of an agent run."""
    events: list[TrajectoryEvent] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class RuntimeMetrics:
    """Metrics from a runtime execution."""
    latency_ms: float = 0.0
    reasoning_steps: int = 0
    search_count: int = 0
    final_think_length: int = 0


class Mem1AgentRuntime:
    """
    Runtime wrapper for Mem1Agent.
    
    Implements the AgentRuntime interface for benchmark integration.
    """
    
    def __init__(
        self,
        settings: Optional[Mem1Settings] = None,
        retriever: Optional[Mem1Retriever] = None,
    ):
        self.settings = settings or load_mem1_settings()
        self.agent = Mem1Agent(settings=self.settings, retriever=retriever)
        
        self._trajectory: Trajectory = Trajectory()
        self._metrics: RuntimeMetrics = RuntimeMetrics()
        self._last_result: Optional[AgentResult] = None
    
    def reset(self, episode: Any = None, initial_state: Any = None) -> None:
        """
        Reset the runtime for a new episode.
        
        Args:
            episode: Episode object with context/conversation
            initial_state: Optional initial state
        """
        self._trajectory = Trajectory()
        self._metrics = RuntimeMetrics()
        self._last_result = None
        
        # Extract corpus from episode if available (for LoCoMo)
        corpus = None
        if episode and hasattr(episode, "context"):
            corpus = episode.context if isinstance(episode.context, list) else [episode.context]
        elif episode and hasattr(episode, "conversation"):
            corpus = episode.conversation
        
        self.agent.reset(corpus=corpus)
    
    def act(self, observation: Any) -> dict:
        """
        Process an observation and return an action.
        
        Args:
            observation: The input (question) to process
            
        Returns:
            Action dict with response and metadata
        """
        # Extract question from observation
        if isinstance(observation, str):
            question = observation
        elif hasattr(observation, "content"):
            question = observation.content
        elif hasattr(observation, "question"):
            question = observation.question
        else:
            question = str(observation)
        
        # Record start
        self._trajectory.start_time = time.time()
        self._trajectory.events.append(TrajectoryEvent(
            event_type="input",
            content=question,
        ))
        
        # Run agent
        result = self.agent.chat(question)
        self._last_result = result
        
        # Record trajectory events from reasoning steps
        for step in result.steps:
            if step.parsed.think:
                self._trajectory.events.append(TrajectoryEvent(
                    event_type="think",
                    content=step.parsed.think,
                    metadata={"step": step.step_num},
                ))
            if step.parsed.search:
                self._trajectory.events.append(TrajectoryEvent(
                    event_type="search",
                    content=step.parsed.search,
                    metadata={"step": step.step_num},
                ))
            if step.information:
                self._trajectory.events.append(TrajectoryEvent(
                    event_type="information",
                    content=step.information,
                    metadata={"step": step.step_num},
                ))
        
        # Record output
        self._trajectory.events.append(TrajectoryEvent(
            event_type="output",
            content=result.answer,
        ))
        self._trajectory.end_time = time.time()
        
        # Update metrics
        self._metrics.latency_ms = self._trajectory.duration_ms
        self._metrics.reasoning_steps = len(result.steps)
        self._metrics.search_count = sum(
            1 for s in result.steps if s.parsed.search
        )
        self._metrics.final_think_length = len(result.final_think)
        
        return {
            "response": result.answer,
            "final_think": result.final_think,
            "steps": len(result.steps),
        }
    
    def get_trajectory(self) -> Trajectory:
        """Get the trajectory from the last run."""
        return self._trajectory
    
    def get_metrics(self) -> dict:
        """Get metrics from the last run."""
        return {
            "latency_ms": self._metrics.latency_ms,
            "reasoning_steps": self._metrics.reasoning_steps,
            "search_count": self._metrics.search_count,
            "final_think_length": self._metrics.final_think_length,
        }
    
    def get_last_result(self) -> Optional[AgentResult]:
        """Get the full result from the last run."""
        return self._last_result