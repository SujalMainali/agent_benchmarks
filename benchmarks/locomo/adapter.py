"""LoCoMo adapter to convert samples into ResearchHelperAgent input."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from benchmarks.common.models import BenchmarkSample
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

    def build_context_messages(self, sample: BenchmarkSample, mode: str = "plain_qa") -> List[BaseMessage]:
        """
        Build the initial context messages from a LoCoMo sample.

        For the first version, this converts the conversation history from sessions
        into a series of HumanMessage + AI response pairs that set context.

        Args:
            sample: The BenchmarkSample.
            mode: The benchmark mode (e.g., "plain_qa", "retrieval_qa").

        Returns:
            List of messages to prepend to the agent's context.
        """
        messages: List[BaseMessage] = []

        # Extract sessions and convert to message history
        sessions = sample.context.get("sessions", [])

        # Flatten conversation context from sessions
        for session in sessions:
            if isinstance(session, dict):
                turns = session.get("turns", [])
                for turn in turns:
                    user_text = turn.get("user", "")
                    assistant_text = turn.get("assistant", "")

                    if user_text:
                        messages.append(HumanMessage(content=user_text))
                    if assistant_text:
                        messages.append(HumanMessage(content=assistant_text))
            elif isinstance(session, list):
                # If sessions is a direct list of turns
                for turn in session:
                    if isinstance(turn, dict):
                        user_text = turn.get("user", "") or turn.get("text", "")
                        assistant_text = turn.get("assistant", "")
                        if user_text:
                            messages.append(HumanMessage(content=user_text))
                        if assistant_text:
                            messages.append(HumanMessage(content=assistant_text))

        return messages

    def build_agent_input(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """
        Convert a BenchmarkSample into ResearchHelperAgent input.

        Args:
            sample: The normalized BenchmarkSample.

        Returns:
            Dictionary with:
            - question: the QA question
            - gold_answer: the expected answer
            - context_messages: prior conversation context
            - mode: benchmark mode
            - metadata: any additional metadata
        """
        context_messages = self.build_context_messages(sample)

        return {
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "context_messages": context_messages,
            "mode": sample.mode,
            "metadata": sample.metadata,
            "sample_id": sample.sample_id,
        }
