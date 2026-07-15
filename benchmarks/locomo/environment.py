from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage

from benchmarks.common.interfaces import BenchmarkEnvironment
from benchmarks.common.models import Action, EnvironmentState, Episode, Observation

from .adapter import LoCoMoAdapter


class LoCoMoEnvironment(BenchmarkEnvironment):
    """Simple environment wrapper for official and demo LoCoMo episodes."""

    def __init__(self, adapter: Optional[LoCoMoAdapter] = None) -> None:
        self.adapter = adapter or LoCoMoAdapter()
        self._episode: Episode | None = None
        self._state: EnvironmentState | None = None
        self._context_messages: List[BaseMessage] = []
        self._last_observation: Observation | None = None
        self._last_action: Action | None = None

    def reset(self, episode: Episode) -> EnvironmentState:
        self._episode = episode
        self._context_messages = self.adapter.build_context_messages(episode)
        self._last_action = None
        self._last_observation = None

        sessions = episode.context.get("sessions", [])
        session_metadata = [
            {
                key: session.get(key)
                for key in ("session_index", "session_id", "session_key", "timestamp", "date", "time", "speaker_a", "speaker_b")
                if isinstance(session, dict) and session.get(key) is not None
            }
            for session in sessions
            if isinstance(session, dict)
        ]
        session_headers = [
            str(msg.content)
            for msg in self._context_messages
            if getattr(msg, "type", "") == "system"
        ]
        context_turn_count = sum(1 for msg in self._context_messages if getattr(msg, "type", "") == "human")
        has_timestamps = any(meta.get("timestamp") for meta in session_metadata)

        self._state = EnvironmentState(
            episode_id=episode.episode_id,
            messages=list(self._context_messages),
            done=False,
            latest_observation={
                "question": episode.question,
                "mode": episode.mode,
                "session_headers": session_headers,
            },
            latest_action={},
            metadata={
                **episode.metadata,
                "question": episode.question,
                "gold_answer": episode.gold_answer,
                "mode": episode.mode,
                "session_count": len(sessions),
                "session_metadata": session_metadata,
                "context_turn_count": context_turn_count,
                "has_timestamps": has_timestamps,
            },
        )
        return self._state

    def observe(self) -> Observation:
        if self._episode is None:
            raise RuntimeError("Environment must be reset before observe().")

        observation = Observation(
            episode_id=self._episode.episode_id,
            text=self._episode.question,
            messages=list(self._context_messages),
            metadata={
                "benchmark_mode": self._episode.mode,
                "context_turn_count": sum(1 for msg in self._context_messages if getattr(msg, "type", "") == "human"),
                "gold_answer": self._episode.gold_answer,
            },
        )
        self._last_observation = observation
        if self._state:
            self._state.latest_observation = {
                "text": observation.text,
                "metadata": observation.metadata,
            }
        return observation

    def step(self, action: Action) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step().")

        self._last_action = action
        self._state.done = True
        self._state.latest_action = {
            "action_type": action.action_type,
            "text": action.text,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "metadata": action.metadata,
        }
        self._state.metadata["predicted_answer"] = action.text
        return self._state

    def snapshot(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before snapshot().")
        return self._state

    def is_done(self) -> bool:
        return bool(self._state.done) if self._state else False
