from __future__ import annotations

import json
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .prompts import FACT_EXTRACTOR_PROMPT


@dataclass
class TemporaryMemory:
    recent_messages: List[BaseMessage] = field(default_factory=list)
    summary: str = ""
    facts: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    summary_every_n_turns: int = 4
    recent_window_turns: int = 6

    def add_tool_result(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        source_type = self._infer_tool_source_type(tool_name, result)
        self.tool_results.append(
            {
                "tool": tool_name,
                "source_type": source_type,
                "args": args,
                "result": result,
            }
        )

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.recent_messages.append(HumanMessage(content=user_text))
        self.recent_messages.append(AIMessage(content=assistant_text))
        self.turn_count += 1

    def add_message(self, msg: BaseMessage) -> None:
        """
        Add an arbitrary BaseMessage to the recent messages.

        This is used for benchmark replay where messages have already been
        constructed with correct roles (HumanMessage, AIMessage, ToolMessage).

        Args:
            msg: A BaseMessage to append (HumanMessage, AIMessage, ToolMessage, etc).
        """
        self.recent_messages.append(msg)

    def _json_from_text(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(cleaned):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                return value if isinstance(value, dict) else {}
        return {}

    def _normalize_fact(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        fact = str(item.get("fact", "")).strip()
        value = str(item.get("value", "")).strip()
        category = str(item.get("category", "other")).strip().lower() or "other"

        if not fact or not value:
            return None

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return {
            "fact": fact,
            "value": value,
            "confidence": confidence,
            "category": category,
            "source_type": "local_memory",
            "source": "llm_fact_extractor",
            "last_seen_turn": self.turn_count + 1,
        }

    def _upsert_fact(self, fact: Dict[str, Any]) -> None:
        new_key = (fact["category"].lower(), fact["fact"].lower())
        for index, existing in enumerate(self.facts):
            existing_key = (
                str(existing.get("category", "")).lower(),
                str(existing.get("fact", "")).lower(),
            )
            if existing_key == new_key:
                self.facts[index] = fact
                return
        self.facts.append(fact)

    def extract_stable_facts(self, user_text: str, chat_model=None) -> List[Dict[str, Any]]:
        """
        Use the LLM to extract durable local-memory facts from a user message.
        """
        if chat_model is None:
            return []

        prompt = FACT_EXTRACTOR_PROMPT.format(user_text=user_text)

        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
        except Exception:
            return []

        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = self._json_from_text(content)
        raw_facts = parsed.get("facts", [])

        if isinstance(raw_facts, dict):
            raw_facts = [raw_facts]
        if not isinstance(raw_facts, list):
            return []

        extracted = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact = self._normalize_fact(item)
            if fact is None:
                continue
            self._upsert_fact(fact)
            extracted.append(fact)

        return extracted

    def format_facts(self) -> str:
        if not self.facts:
            return "None"
        lines = []
        for item in self.facts:
            lines.append(
                "- source=LOCAL MEMORY "
                f"category={item.get('category')} "
                f"fact={item.get('fact')} "
                f"value={item.get('value')} "
                f"confidence={item.get('confidence')}"
            )
        return "\n".join(lines)

    def _infer_tool_source_type(self, tool_name: str, result: str) -> str:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and parsed.get("source_type"):
                return str(parsed["source_type"])
        except json.JSONDecodeError:
            pass

        return {
            "document_search": "document",
            "note_lookup": "local_memory",
            "web_search": "web",
            "calculator": "calculation",
        }.get(tool_name, "tool")

    def format_tool_results(self, limit: int = 5) -> str:
        if not self.tool_results:
            return "None"

        lines = []
        for item in self.tool_results[-limit:]:
            lines.append(
                "- "
                f"source={item.get('source_type', 'tool')} "
                f"tool={item['tool']} "
                f"args={item['args']} "
                f"result={item['result']}"
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
