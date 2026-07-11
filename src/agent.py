from __future__ import annotations

from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .llm import LLMProvider, LLMResponse, ToolCall
from .memory import TemporaryMemory
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT


class ResearchHelperAgent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: Sequence,
        memory: TemporaryMemory | None = None,
        max_tool_steps: int = 3,
        system_prompt_override: str | None = None,
        allow_tools: bool = True,
    ) -> None:
        tool_list = list(tools)
        self.llm = llm
        self.tool_list = tool_list
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

    def _response_message(self, response: LLMResponse) -> BaseMessage:
        """Return the underlying provider message, or synthesize one.

        The runtime and memory speak ``langchain_core`` messages, so we thread
        the provider's own message object (an ``AIMessage`` for the current
        providers) back into the conversation. This keeps tool-call ids aligned
        for the next turn regardless of which backend produced the response.
        """
        if isinstance(response.message, BaseMessage):
            return response.message
        return AIMessage(content=response.text)

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

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        tool_name = tool_call.name
        arguments = tool_call.arguments
        tool_call_id = tool_call.id or f"{tool_name}_{self.memory.turn_count + 1}"

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
        response = self.llm.invoke([HumanMessage(content=prompt)])
        self.memory.summary = response.text

    def run_turn(self, user_text: str) -> str:
        self.memory.extract_stable_facts(user_text, self.llm)

        working_messages = self.build_context_messages(user_text)
        turn_log: List[BaseMessage] = [HumanMessage(content=user_text)]

        final_answer = ""

        # Benchmark mode: no tool loop
        if not self.allow_tools:
            response = self.llm.invoke(working_messages)
            message = self._response_message(response)
            working_messages.append(message)
            turn_log.append(message)
            final_answer = response.text
        else:
            # Normal mode: tool loop
            for _ in range(self.max_tool_steps):
                response = self.llm.invoke(working_messages, tools=self.tool_list)
                message = self._response_message(response)
                working_messages.append(message)
                turn_log.append(message)

                if not response.has_tool_calls():
                    final_answer = response.text
                    break

                tool_messages = [self._execute_tool_call(tc) for tc in response.tool_calls]
                working_messages.extend(tool_messages)
                turn_log.extend(tool_messages)
            else:
                response = self.llm.invoke(working_messages)
                message = self._response_message(response)
                working_messages.append(message)
                turn_log.append(message)
                final_answer = response.text

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

        Update contract (consumed by the runtime): ``messages`` holds
        ``langchain_core`` messages and ``tool_calls`` holds langchain-style
        dicts. Both shapes are provider-agnostic.
        """
        self.memory.extract_stable_facts(user_text, self.llm)

        working_messages = self.build_context_messages(user_text)
        turn_log: List[BaseMessage] = [HumanMessage(content=user_text)]

        yield {"step": "user", "messages": [turn_log[0]]}

        final_answer = ""

        # Benchmark mode: no tool loop
        if not self.allow_tools:
            response = self.llm.invoke(working_messages)
            message = self._response_message(response)
            working_messages.append(message)
            turn_log.append(message)
            final_answer = response.text
            yield {"step": "final", "messages": [message]}
        else:
            # Normal mode: tool loop
            for _ in range(self.max_tool_steps):
                response = self.llm.invoke(working_messages, tools=self.tool_list)
                message = self._response_message(response)
                working_messages.append(message)
                turn_log.append(message)

                tool_call_dicts = response.tool_calls_as_dicts()
                yield {"step": "model", "messages": [message], "tool_calls": tool_call_dicts}

                if not response.has_tool_calls():
                    final_answer = response.text
                    break

                tool_messages = [self._execute_tool_call(tc) for tc in response.tool_calls]
                working_messages.extend(tool_messages)
                turn_log.extend(tool_messages)
                yield {"step": "tools", "messages": tool_messages}
            else:
                response = self.llm.invoke(working_messages)
                message = self._response_message(response)
                working_messages.append(message)
                turn_log.append(message)
                final_answer = response.text
                yield {"step": "final", "messages": [message]}

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
