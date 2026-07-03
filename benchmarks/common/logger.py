"""Logging and tracing utilities for benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from .models import Action, EnvironmentState, Observation, ToolEvent, TrajectoryEvent, TrajectoryStep


class BenchmarkLogger:
    """Captures full interaction traces including prompts, tool calls, and memory updates."""

    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.trajectory: List[TrajectoryEvent] = []
        self.raw_messages: List[Dict[str, Any]] = []
        self.questions: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.environment_snapshots: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.turn_number = 0

    def log_turn_start(self, user_input: str, system_prompts: List[str]) -> None:
        """Log the beginning of a turn."""
        self.turn_number += 1
        self.current_turn_start = time.time()
        self.current_user_input = user_input
        self.current_system_prompt = "\n---\n".join(system_prompts)
        self.current_tool_events: List[ToolEvent] = []

    def log_agent_message(self, message: str) -> None:
        """Log the agent's response."""
        self.current_agent_message = message

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Log a tool execution."""
        self.current_tool_events.append(
            ToolEvent(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                latency_ms=latency_ms,
            )
        )

    def log_memory_state(self, state: Dict[str, Any]) -> None:
        """Log the current memory state."""
        self.current_memory_state = state

    def log_observation(self, observation: Observation) -> None:
        """Log a benchmark observation."""
        self.observations.append(
            {
                "episode_id": observation.episode_id,
                "text": observation.text,
                "metadata": observation.metadata,
                "timestamp": time.time(),
            }
        )

    def log_action(self, action: Action) -> None:
        """Log a benchmark action."""
        self.actions.append(
            {
                "action_type": action.action_type,
                "text": action.text,
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "metadata": action.metadata,
                "timestamp": time.time(),
            }
        )

    def log_environment_snapshot(self, snapshot: EnvironmentState) -> None:
        """Log a serializable environment snapshot."""
        self.environment_snapshots.append(
            {
                "episode_id": snapshot.episode_id,
                "done": snapshot.done,
                "latest_observation": snapshot.latest_observation,
                "latest_action": snapshot.latest_action,
                "metadata": snapshot.metadata,
                "timestamp": time.time(),
            }
        )

    def log_event(self, event: TrajectoryEvent) -> None:
        """Log a generic trajectory event."""
        self.trajectory.append(event)

    def finalize_turn(self) -> None:
        """Finalize the current turn and add to trajectory."""
        latency_ms = (time.time() - self.current_turn_start) * 1000
        step = TrajectoryEvent(
            event_type="turn",
            turn_number=self.turn_number,
            user_input=self.current_user_input,
            system_prompt=self.current_system_prompt,
            agent_message=self.current_agent_message,
            tool_calls=self.current_tool_events,
            memory_state=getattr(self, "current_memory_state", {}),
            latency_ms=latency_ms,
        )
        self.trajectory.append(step)

    def log_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a raw message (for message history)."""
        self.raw_messages.append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )

    def log_context_message(self, message: BaseMessage, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a context message with the original role preserved."""
        role = getattr(message, "type", None) or message.__class__.__name__.replace("Message", "").lower()
        self.log_message(role, str(message.content), metadata=metadata)

    def log_question(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the benchmark question separately from the conversation trace."""
        self.questions.append({"question": question, "metadata": metadata or {}, "timestamp": time.time()})
        self.log_message("question", question, metadata=metadata)

    def get_total_latency_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full trace to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "trajectory": [asdict(step) for step in self.trajectory],
            "raw_messages": self.raw_messages,
            "questions": self.questions,
            "observations": self.observations,
            "actions": self.actions,
            "environment_snapshots": self.environment_snapshots,
            "total_latency_ms": self.get_total_latency_ms(),
        }

    def to_json(self) -> str:
        """Serialize the full trace to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)
