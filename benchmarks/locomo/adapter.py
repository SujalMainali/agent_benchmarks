"""LoCoMo adapter to convert samples into ResearchHelperAgent input."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

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

        Args:
            sample: The BenchmarkSample or Episode.
        Returns:
            List of messages to prepend to the agent's context with correct roles.
        """
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        messages: List[BaseMessage] = []

        sessions = episode.context.get("sessions", [])
        conversation = episode.context.get("conversation", {})
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")

        for session in sessions:
            turns = session.get("turns", []) if isinstance(session, dict) else session
            for turn in turns:
                if not isinstance(turn, dict):
                    continue

                text = str(turn.get("text", turn.get("content", ""))).strip()
                if not text:
                    continue

                speaker = turn.get("speaker") or turn.get("role") or turn.get("author")

                if speaker is None:
                    raise ValueError(f"Unable to determine speaker role for turn: {turn!r}")

                speaker_name = str(speaker)
                if speaker_b and speaker_name == speaker_b:
                    messages.append(AIMessage(content=text))
                elif speaker_a and speaker_name == speaker_a:
                    messages.append(HumanMessage(content=text))
                elif speaker_name.lower() in {"assistant", "ai", "bot", "system", "speaker_b"}:
                    messages.append(AIMessage(content=text))
                elif speaker_name.lower() in {"user", "human", "speaker_a"}:
                    messages.append(HumanMessage(content=text))
                else:
                    raise ValueError(f"Unsupported speaker role '{speaker_name}' in turn: {turn!r}")

        return messages

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
