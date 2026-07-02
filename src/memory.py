from __future__ import annotations

import re
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class TemporaryMemory:
    recent_messages: List[BaseMessage] = field(default_factory=list)
    summary: str = ""
    facts: Dict[str, str] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    summary_every_n_turns: int = 4
    recent_window_turns: int = 6

    def add_tool_result(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        self.tool_results.append(
            {
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.recent_messages.append(HumanMessage(content=user_text))
        self.recent_messages.append(AIMessage(content=assistant_text))
        self.turn_count += 1

    def extract_stable_facts(self, user_text: str) -> None:
        """
        Very small rule-based extractor for learning.
        Later, we can replace this with an LLM-based fact extractor.
        """
        patterns = [
            (r"\bmy name is ([^.!\n]+)", "name"),
            (r"\bi am ([^.!\n]+)", "identity"),
            (r"\bi like ([^.!\n]+)", "preference"),
            (r"\bi prefer ([^.!\n]+)", "preference"),
            (r"\bmy goal is ([^.!\n]+)", "goal"),
            (r"\bi study ([^.!\n]+)", "study"),
            (r"\bi live in ([^.!\n]+)", "location"),
        ]

        for pattern, key in patterns:
            match = re.search(pattern, user_text, flags=re.IGNORECASE)
            if match:
                self.facts[key] = match.group(1).strip()

    def format_facts(self) -> str:
        if not self.facts:
            return "None"
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items())

    def format_tool_results(self, limit: int = 5) -> str:
        if not self.tool_results:
            return "None"

        lines = []
        for item in self.tool_results[-limit:]:
            lines.append(
                f"- tool={item['tool']} args={item['args']} result={item['result']}"
            )
        return "\n".join(lines)

    def recent_context_messages(self) -> List[BaseMessage]:
        # Keep tool-call request/result pairs intact for the next model call.
        return self.recent_messages

    def _content_text(self, msg: BaseMessage) -> str:
        content = msg.content
        if isinstance(content, str):
            return content
        return pformat(content, width=100)

    def _format_message_trace(self, index: int, msg: BaseMessage) -> str:
        role = getattr(msg, "type", "message")
        content = self._content_text(msg).strip() or "<empty>"
        lines = [f"{index:02d}. {role}: {content}"]

        metadata = {}
        if getattr(msg, "id", None):
            metadata["id"] = msg.id
        if getattr(msg, "name", None):
            metadata["name"] = msg.name
        if getattr(msg, "tool_call_id", None):
            metadata["tool_call_id"] = msg.tool_call_id
        if getattr(msg, "status", None):
            metadata["status"] = msg.status
        if getattr(msg, "tool_calls", None):
            metadata["tool_calls"] = msg.tool_calls
        if getattr(msg, "invalid_tool_calls", None):
            metadata["invalid_tool_calls"] = msg.invalid_tool_calls
        if getattr(msg, "usage_metadata", None):
            metadata["usage_metadata"] = msg.usage_metadata
        if getattr(msg, "response_metadata", None):
            metadata["response_metadata"] = msg.response_metadata

        for key, value in metadata.items():
            lines.append(f"    {key}: {pformat(value, width=100)}")

        return "\n".join(lines)

    def format_recent_dialogue(self, window_messages: int | None = None) -> str:
        if not self.recent_messages:
            return "No recent dialogue."

        if window_messages is None:
            window_messages = self.recent_window_turns * 4

        selected = self.recent_messages[-window_messages:]
        lines = []
        for msg in selected:
            role = getattr(msg, "type", "message").upper()
            content = self._content_text(msg).strip() or "<empty>"
            lines.append(f"{role}: {content}")
            if getattr(msg, "tool_calls", None):
                lines.append(f"TOOL_CALLS: {pformat(msg.tool_calls, width=100)}")
            if getattr(msg, "tool_call_id", None):
                lines.append(f"TOOL_CALL_ID: {msg.tool_call_id}")
        return "\n".join(lines)

    def should_refresh_summary(self) -> bool:
        return self.turn_count > 0 and self.turn_count % self.summary_every_n_turns == 0

    def print_state(self) -> None:
        print("\n========== TEMPORARY MEMORY ==========")
        print("SUMMARY:")
        print(self.summary if self.summary else "None")
        print("\nFACTS:")
        print(self.format_facts())
        print("\nTOOL RESULTS:")
        print(self.format_tool_results())
        print("\nRECENT MESSAGES:")
        if not self.recent_messages:
            print("None")
        else:
            for i, msg in enumerate(self.recent_context_messages(), start=1):
                print(self._format_message_trace(i, msg))
        print("======================================\n")
