"""LoCoMo adapter to convert samples into ResearchHelperAgent input."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from benchmarks.common.models import BenchmarkSample, Episode
from benchmarks.common.interfaces import BenchmarkAdapter


class LoCoMoAdapter(BenchmarkAdapter):
    """Converts LoCoMo samples into ResearchHelperAgent input."""

    def __init__(self, system_prompt_override: str = "") -> None:
        """
        Initialize the adapter.

        Args:
            system_prompt_override: Optional custom system prompt for LoCoMo.
                                   If empty, uses the agent's default.
        """
        self.system_prompt_override = system_prompt_override

    def load_sample(self, sample_data: Dict[str, Any]) -> BenchmarkSample:
        """Load and normalize a LoCoMo sample.

        This is typically delegated to LoCoMoLoader, but provided here
        for the BenchmarkAdapter interface.

        Args:
            sample_data: Raw sample dictionary.

        Returns:
            BenchmarkSample.
        """
        from .loader import LoCoMoLoader

        loader = LoCoMoLoader()
        return loader.load_sample_from_dict(sample_data)

    def build_context_messages(self, sample: BenchmarkSample | Episode) -> List[BaseMessage]:
        """
        Build the initial context messages from a LoCoMo sample.

        This converts the conversation history from sessions into a series of
        role-preserving messages: HumanMessage for user turns, AIMessage for assistant turns.

        When a session carries timestamp metadata, a SystemMessage session header
        (session number, date, time, participants) is inserted before that
        session's turns so the timestamps are visible in the final prompt.

        Args:
            sample: The BenchmarkSample or Episode.
        Returns:
            List of messages to prepend to the agent's context with correct roles.
        """
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        messages: List[BaseMessage] = []

        sessions = episode.context.get("sessions", [])
        conversation = episode.context.get("conversation", {})
        speaker_a = conversation.get("speaker_a") or episode.context.get("speaker_a")
        speaker_b = conversation.get("speaker_b") or episode.context.get("speaker_b")

        for session_index, session in enumerate(sessions):
            turns = session.get("turns", []) if isinstance(session, dict) else session

            session_turn_messages: List[BaseMessage] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue

                text = str(turn.get("text", turn.get("content", ""))).strip()
                if not text:
                    continue

                speaker = turn.get("speaker") or turn.get("role") or turn.get("author")

                if speaker is None:
                    raise ValueError(f"Unable to determine speaker role for turn: {turn!r}")

                turn_metadata = self._turn_metadata(turn, session, session_index)

                speaker_name = str(speaker)
                if speaker_b and speaker_name == speaker_b:
                    session_turn_messages.append(AIMessage(content=text, additional_kwargs=turn_metadata))
                elif speaker_a and speaker_name == speaker_a:
                    session_turn_messages.append(HumanMessage(content=text, additional_kwargs=turn_metadata))
                elif speaker_name.lower() in {"assistant", "ai", "bot", "system", "speaker_b"}:
                    session_turn_messages.append(AIMessage(content=text, additional_kwargs=turn_metadata))
                elif speaker_name.lower() in {"user", "human", "speaker_a"}:
                    session_turn_messages.append(HumanMessage(content=text, additional_kwargs=turn_metadata))
                else:
                    raise ValueError(f"Unsupported speaker role '{speaker_name}' in turn: {turn!r}")

            if not session_turn_messages:
                continue

            header = self.format_session_header(session, session_index, speaker_a, speaker_b)
            if header:
                messages.append(
                    SystemMessage(
                        content=header,
                        additional_kwargs=self._session_metadata(session, session_index),
                    )
                )
            messages.extend(session_turn_messages)

        return messages

    def format_session_header(
        self,
        session: Any,
        session_index: int,
        speaker_a: str | None = None,
        speaker_b: str | None = None,
    ) -> str:
        """Format a prompt-visible header for one session (number, date, time, participants)."""
        if not isinstance(session, dict):
            session = {}

        session_number = session.get("session_index", session_index) + 1
        lines = [f"SESSION {session_number}"]

        timestamp = session.get("timestamp") or session.get("date_time")
        date = session.get("date")
        time = session.get("time")
        if date:
            lines.append(f"Date: {date}")
        if time:
            lines.append(f"Time: {time}")
        if timestamp and not (date or time):
            lines.append(f"Timestamp: {timestamp}")

        participant_a = session.get("speaker_a") or speaker_a
        participant_b = session.get("speaker_b") or speaker_b
        participants = [p for p in (participant_a, participant_b) if p]
        if participants:
            lines.append(f"Participants: {', '.join(str(p) for p in participants)}")

        return "\n".join(lines)

    def _session_metadata(self, session: Any, session_index: int) -> Dict[str, Any]:
        """Collect session-level metadata for attachment to the header message."""
        if not isinstance(session, dict):
            return {"session_index": session_index}
        metadata: Dict[str, Any] = {"session_index": session.get("session_index", session_index)}
        for key in ("session_id", "session_key", "timestamp", "date_time", "date", "time", "speaker_a", "speaker_b"):
            value = session.get(key)
            if value:
                metadata[key] = value
        return metadata

    def _turn_metadata(self, turn: Dict[str, Any], session: Any, session_index: int) -> Dict[str, Any]:
        """Collect per-turn metadata (timestamps, ids) for the message's additional_kwargs."""
        metadata: Dict[str, Any] = {}
        for key in ("dia_id", "turn_index", "session_index", "session_timestamp", "timestamp", "date_time"):
            value = turn.get(key)
            if value is not None:
                metadata[key] = value
        metadata.setdefault("session_index", session.get("session_index", session_index) if isinstance(session, dict) else session_index)
        if isinstance(session, dict):
            session_timestamp = session.get("timestamp") or session.get("date_time")
            if session_timestamp:
                metadata.setdefault("session_timestamp", session_timestamp)
        return metadata

    def build_agent_input(self, sample: BenchmarkSample | Episode) -> Dict[str, Any]:
        """
        Convert a BenchmarkSample into ResearchHelperAgent input.

        This includes the question, context history (with correct roles),
        and benchmark metadata. The runner will use this to set up the agent.

        Args:
            sample: The normalized BenchmarkSample.

        Returns:
            Dictionary with:
            - question: the QA question to answer
            - gold_answer: the expected answer
            - context_messages: prior conversation history with role-preserved messages
            - mode: benchmark mode (e.g., "plain_qa", "retrieval_qa")
            - context_turn_count: number of turns in the context history
            - metadata: additional metadata
            - sample_id: the sample identifier
        """
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        context_messages = self.build_context_messages(episode)

        context_turn_count = sum(1 for msg in context_messages if isinstance(msg, HumanMessage))

        return {
            "question": episode.question,
            "gold_answer": episode.gold_answer,
            "context_messages": context_messages,
            "mode": episode.mode,
            "context_turn_count": context_turn_count,
            "evidence": episode.context.get("evidence", []),
            "raw_fields": episode.context.get("raw_fields", {}),
            "metadata": episode.metadata,
            "sample_id": episode.episode_id,
            "episode": episode,
        }
