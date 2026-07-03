from __future__ import annotations

from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace

from .memory import TemporaryMemory
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT


class ResearchHelperAgent:
    def __init__(
        self,
        chat_model: ChatHuggingFace,
        tools: Sequence,
        memory: TemporaryMemory | None = None,
        max_tool_steps: int = 3,
        system_prompt_override: str | None = None,
        allow_tools: bool = True,
    ) -> None:
        tool_list = list(tools)
        self.chat_model = chat_model
        self.tool_model = chat_model.bind_tools(tool_list)
        self.tools = {tool.name: tool for tool in tool_list}
        self.memory = memory or TemporaryMemory()
        self.max_tool_steps = max_tool_steps
        self.system_prompt_override = system_prompt_override
        self.allow_tools = allow_tools

    def build_context_messages(self, user_text: str) -> List[BaseMessage]:
        # Use the benchmark override if provided, otherwise use the default system prompt
        system_prompt = self.system_prompt_override if self.system_prompt_override else SYSTEM_PROMPT
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        if self.memory.summary.strip():
            messages.append(SystemMessage(content=f"Working summary:\n{self.memory.summary}"))

        if self.memory.facts:
            messages.append(SystemMessage(content=f"LOCAL MEMORY facts:\n{self.memory.format_facts()}"))

        messages.extend(self.memory.recent_context_messages())
        messages.append(HumanMessage(content=user_text))
        return messages

    def _message_content_text(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"Error: unknown tool '{tool_name}'."

        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a dictionary."

        try:
            return str(tool.invoke(arguments))
        except Exception as exc:
            return f"Tool execution error: {exc}"

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolMessage:
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("args", {})
        tool_call_id = tool_call.get("id") or f"{tool_name}_{self.memory.turn_count + 1}"

        result = self._execute_tool(tool_name, arguments)
        self.memory.add_tool_result(tool_name, arguments, result)

        return ToolMessage(
            content=result,
            name=tool_name,
            tool_call_id=tool_call_id,
        )

    def _refresh_summary(self) -> None:
        prompt = SUMMARY_PROMPT.format(
            old_summary=self.memory.summary.strip() or "None",
            recent_dialogue=self.memory.format_recent_dialogue(),
        )
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        self.memory.summary = self._message_content_text(response)

    def run_turn(self, user_text: str) -> str:
        self.memory.extract_stable_facts(user_text, self.chat_model)

        working_messages = self.build_context_messages(user_text)
        turn_log: List[BaseMessage] = [HumanMessage(content=user_text)]

        final_answer = ""

        # Benchmark mode: no tool loop
        if not self.allow_tools:
            response = self.chat_model.invoke(working_messages)
            working_messages.append(response)
            turn_log.append(response)
            final_answer = self._message_content_text(response)
        else:
            # Normal mode: tool loop
            for _ in range(self.max_tool_steps):
                response = self.tool_model.invoke(working_messages)
                working_messages.append(response)
                turn_log.append(response)

                tool_calls = getattr(response, "tool_calls", [])
                if not tool_calls:
                    final_answer = self._message_content_text(response)
                    break

                tool_messages = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
                working_messages.extend(tool_messages)
                turn_log.extend(tool_messages)
            else:
                response = self.chat_model.invoke(working_messages)
                working_messages.append(response)
                turn_log.append(response)
                final_answer = self._message_content_text(response)

        if not final_answer:
            final_answer = "I could not produce a final answer."
            turn_log.append(AIMessage(content=final_answer))

        self.memory.recent_messages.extend(turn_log)
        self.memory.turn_count += 1

        if self.memory.should_refresh_summary():
            self._refresh_summary()

        return final_answer

    def stream_turn_updates(self, user_text: str):
        """
        Yield step-wise updates for debugging the native tool-calling loop.
        
        In benchmark mode (allow_tools=False), skips the tool loop entirely.
        """
        self.memory.extract_stable_facts(user_text, self.chat_model)

        working_messages = self.build_context_messages(user_text)
        turn_log: List[BaseMessage] = [HumanMessage(content=user_text)]

        yield {"step": "user", "messages": [turn_log[0]]}

        final_answer = ""

        # Benchmark mode: no tool loop
        if not self.allow_tools:
            response = self.chat_model.invoke(working_messages)
            working_messages.append(response)
            turn_log.append(response)
            final_answer = self._message_content_text(response)
            yield {"step": "final", "messages": [response]}
        else:
            # Normal mode: tool loop
            for _ in range(self.max_tool_steps):
                response = self.tool_model.invoke(working_messages)
                working_messages.append(response)
                turn_log.append(response)

                tool_calls = getattr(response, "tool_calls", [])
                yield {"step": "model", "messages": [response], "tool_calls": tool_calls}

                if not tool_calls:
                    final_answer = self._message_content_text(response)
                    break

                tool_messages = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
                working_messages.extend(tool_messages)
                turn_log.extend(tool_messages)
                yield {"step": "tools", "messages": tool_messages}
            else:
                response = self.chat_model.invoke(working_messages)
                working_messages.append(response)
                turn_log.append(response)
                final_answer = self._message_content_text(response)
                yield {"step": "final", "messages": [response]}

        if not final_answer:
            final_answer = "I could not produce a final answer."
            final_message = AIMessage(content=final_answer)
            turn_log.append(final_message)
            yield {"step": "final", "messages": [final_message]}

        self.memory.recent_messages.extend(turn_log)
        self.memory.turn_count += 1

        if self.memory.should_refresh_summary():
            self._refresh_summary()

        yield {"step": "done", "answer": final_answer}
