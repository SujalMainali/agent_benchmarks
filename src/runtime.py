from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from benchmarks.common.interfaces import AgentRuntime
from benchmarks.common.models import Action, EnvironmentState, Episode, Observation, Trajectory, TrajectoryEvent, ToolEvent

from .agent import ResearchHelperAgent
from .memory import TemporaryMemory


class ResearchHelperAgentRuntime(AgentRuntime):
    """Benchmark-neutral runtime wrapper around ResearchHelperAgent."""

    def __init__(self, agent: ResearchHelperAgent) -> None:
        self.agent = agent
        self._episode: Episode | None = None
        self._trajectory = Trajectory()
        self._raw_messages: List[Dict[str, Any]] = []
        self._last_action: Action | None = None
        self._last_observation: Observation | None = None
        self._turn_count = 0

    def reset(self, episode: Episode, initial_state: EnvironmentState) -> None:
        self._episode = episode
        self._trajectory = Trajectory()
        self._raw_messages = []
        self._last_action = None
        self._last_observation = None
        self._turn_count = 0

        self.agent.memory = TemporaryMemory()
        for message in initial_state.messages:
            self.agent.memory.add_message(message)
            self._raw_messages.append(self._message_to_dict(message))

    def act(self, observation: Observation) -> Action:
        self._last_observation = observation
        self._raw_messages.append(
            {
                "role": "user",
                "content": observation.text,
                "metadata": observation.metadata,
            }
        )

        # One act() == one turn (an act may emit several events: user/model/
        # tools/final/done — they all share this turn number).
        self._turn_count += 1
        turn_index = self._turn_count
        collected_answer = ""

        for update in self.agent.stream_turn_updates(observation.text):
            step_name = str(update.get("step", "event"))
            event_messages = update.get("messages", [])
            is_user_step = step_name == "user"
            event = TrajectoryEvent(
                event_type=step_name,
                turn_number=turn_index,
                # The "user" step's message is the user's own text — record it
                # only as user_input, never as agent_message.
                user_input=observation.text if is_user_step else "",
                agent_message="" if is_user_step else self._first_message_text(event_messages),
                actor="user" if is_user_step else ("tool" if step_name == "tools" else "agent"),
                recipient="agent" if is_user_step or step_name == "tools" else "user",
                tool_calls=self._tool_events_from_update(update),
                metadata={k: v for k, v in update.items() if k not in {"messages", "tool_calls", "answer"}},
            )
            self._trajectory.append(event)

            for message in event_messages:
                self._raw_messages.append(self._message_to_dict(message))

            if step_name == "done":
                collected_answer = str(update.get("answer", ""))
            elif step_name == "final" and not collected_answer:
                collected_answer = self._first_message_text(event_messages)

        action = Action(
            action_type="final_answer",
            text=collected_answer,
            metadata={
                "episode_id": observation.episode_id,
                "benchmark_mode": observation.metadata.get("benchmark_mode", "plain_qa"),
            },
        )
        self._last_action = action
        self._raw_messages.append(
            {
                "role": "assistant",
                "content": collected_answer,
                "metadata": action.metadata,
            }
        )
        return action

    def get_trajectory(self) -> Trajectory:
        return self._trajectory

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "event_count": len(self._trajectory.events),
            "message_count": len(self._raw_messages),
        }

    def get_raw_messages(self) -> List[Dict[str, Any]]:
        return list(self._raw_messages)

    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        role = getattr(message, "type", None) or message.__class__.__name__.replace("Message", "").lower()
        return {
            "role": role,
            "content": message.content if isinstance(message.content, str) else str(message.content),
            "metadata": self._message_metadata(message),
        }

    def _message_metadata(self, message: BaseMessage) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key in ["name", "tool_call_id", "status", "tool_calls", "invalid_tool_calls", "usage_metadata", "response_metadata"]:
            value = getattr(message, key, None)
            if value is not None:
                metadata[key] = value

        # Preserve adapter-provided metadata (e.g. session/timestamp info) attached
        # to replayed benchmark messages.
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if additional_kwargs:
            metadata["additional_kwargs"] = additional_kwargs
        return metadata

    def _first_message_text(self, messages: List[BaseMessage]) -> str:
        if not messages:
            return ""
        first = messages[0]
        return first.content if isinstance(first.content, str) else str(first.content)

    def _tool_events_from_update(self, update: Dict[str, Any]) -> List[ToolEvent]:
        tool_events: List[ToolEvent] = []
        for tool_call in update.get("tool_calls", []) or []:
            tool_events.append(
                ToolEvent(
                    tool_name=str(tool_call.get("name", "")),
                    arguments=tool_call.get("args", {}) or {},
                    result=str(tool_call.get("result", "")),
                    latency_ms=float(tool_call.get("latency_ms", 0.0) or 0.0),
                )
            )
        return tool_events
