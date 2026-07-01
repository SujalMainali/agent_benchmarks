from __future__ import annotations

from typing import Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace

from .memory import TemporaryMemory
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT, TOOL_PROTOCOL_PROMPT
from .utils.json_utils import extract_json_object, normalize_response_object


class ResearchHelperAgent:
    def __init__(
        self,
        chat_model: ChatHuggingFace,
        tools: Sequence,
        memory: TemporaryMemory | None = None,
        max_tool_steps: int = 3,
    ) -> None:
        self.chat_model = chat_model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory or TemporaryMemory()
        self.max_tool_steps = max_tool_steps

    def build_context_messages(self, user_text: str) -> List[BaseMessage]:
        messages: List[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=TOOL_PROTOCOL_PROMPT),
        ]

        if self.memory.summary.strip():
            messages.append(SystemMessage(content=f"Working summary:\n{self.memory.summary}"))

        if self.memory.facts:
            messages.append(SystemMessage(content=f"Stable facts:\n{self.memory.format_facts()}"))

        if self.memory.tool_results:
            messages.append(
                SystemMessage(content=f"Recent tool outputs:\n{self.memory.format_tool_results()}")
            )

        messages.extend(self.memory.recent_messages[-self.memory.recent_window_turns * 2 :])
        messages.append(HumanMessage(content=user_text))
        return messages

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"Error: unknown tool '{tool_name}'."

        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a JSON object."

        try:
            return str(tool.invoke(arguments))
        except Exception as exc:
            return f"Tool execution error: {exc}"

    def _refresh_summary(self) -> None:
        prompt = SUMMARY_PROMPT.format(
            old_summary=self.memory.summary.strip() or "None",
            recent_dialogue=self.memory.format_recent_dialogue(),
        )
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        self.memory.summary = response.content.strip()

    def run_turn(self, user_text: str) -> str:
        self.memory.extract_stable_facts(user_text)

        working_messages = self.build_context_messages(user_text)
        turn_log: List[BaseMessage] = [HumanMessage(content=user_text)]

        final_answer = ""

        for step in range(self.max_tool_steps):
            response = self.chat_model.invoke(working_messages)
            raw_text = response.content.strip()

            parsed = extract_json_object(raw_text)
            if parsed is None:
                # Fallback: treat the model output as the final answer.
                final_answer = raw_text
                break

            normalized = normalize_response_object(parsed)

            if normalized["type"] == "final":
                final_answer = normalized["answer"].strip()
                break

            if normalized["type"] == "tool_call":
                tool_name = normalized["tool_name"]
                arguments = normalized["arguments"]

                tool_call_message = AIMessage(content=raw_text)
                working_messages.append(tool_call_message)

                result = self._execute_tool(tool_name, arguments)
                self.memory.add_tool_result(tool_name, arguments, result)

                tool_message = ToolMessage(
                    content=result,
                    name=tool_name,
                    tool_call_id=f"{tool_name}_{self.memory.turn_count + 1}_{step + 1}",
                )
                working_messages.append(tool_message)

                turn_log.append(tool_message)
                continue

            final_answer = raw_text
            break

        if not final_answer:
            final_answer = "I could not produce a final answer."

        turn_log.append(AIMessage(content=final_answer))
        self.memory.recent_messages.extend(turn_log)
        self.memory.turn_count += 1

        if self.memory.should_refresh_summary():
            self._refresh_summary()

        return final_answer