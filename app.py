import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not TOKEN:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a research helper agent. "
    "Use the working memory carefully. "
    "Be concise, accurate, and consistent across turns."
)

SUMMARY_PROMPT = """You are updating the short working summary for a multi-turn research helper.
Keep only the most important facts, decisions, constraints, goals, and unresolved questions.
Write 3 to 6 short bullet points.
Do not repeat unimportant text.
Do not add anything not supported by the conversation.

Current summary:
{old_summary}

Recent conversation:
{recent_dialogue}

Updated summary:
"""


@dataclass
class TemporaryMemory:
    recent_messages: List[BaseMessage] = field(default_factory=list)
    summary: str = ""
    facts: Dict[str, str] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    summary_every_n_turns: int = 4
    recent_window_turns: int = 6

    def add_user_message(self, text: str) -> None:
        self.recent_messages.append(HumanMessage(content=text))

    def add_assistant_message(self, text: str) -> None:
        self.recent_messages.append(AIMessage(content=text))

    def add_tool_result(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        self.tool_results.append(
            {
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )

    def extract_stable_facts(self, user_text: str) -> None:
        """
        Very small rule-based extractor for learning.
        Later, you can replace this with an LLM-based fact extractor.
        """
        patterns = [
            (r"\bmy name is ([^.!\n]+)", "name"),
            (r"\bi am ([^.!\n]+)", "identity"),
            (r"\bi like ([^.!\n]+)", "preference"),
            (r"\bi prefer ([^.!\n]+)", "preference"),
            (r"\bmy goal is ([^.!\n]+)", "goal"),
            (r"\bi live in ([^.!\n]+)", "location"),
            (r"\bi study ([^.!\n]+)", "study"),
        ]

        lower_text = user_text.lower()
        for pattern, key in patterns:
            match = re.search(pattern, lower_text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                self.facts[key] = value

    def format_facts(self) -> str:
        if not self.facts:
            return "None"
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items())

    def format_tool_results(self, limit: int = 3) -> str:
        if not self.tool_results:
            return "None"
        lines = []
        for item in self.tool_results[-limit:]:
            lines.append(
                f"- tool={item['tool']} args={item['args']} result={item['result']}"
            )
        return "\n".join(lines)

    def build_prompt(self, current_user_text: str) -> List[BaseMessage]:
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        if self.summary.strip():
            messages.append(SystemMessage(content=f"Working summary:\n{self.summary}"))

        if self.facts:
            messages.append(SystemMessage(content=f"Stable facts:\n{self.format_facts()}"))

        if self.tool_results:
            messages.append(
                SystemMessage(content=f"Recent tool outputs:\n{self.format_tool_results()}")
            )

        # Keep only the recent part of the conversation as raw turns.
        messages.extend(self.recent_messages[-self.recent_window_turns * 2 :])

        # Add the current user turn at the end.
        messages.append(HumanMessage(content=current_user_text))
        return messages

    def recent_dialogue_text(self) -> str:
        if not self.recent_messages:
            return "No recent dialogue."
        lines = []
        for msg in self.recent_messages[-self.recent_window_turns * 2 :]:
            role = getattr(msg, "type", "message")
            lines.append(f"{role.upper()}: {msg.content}")
        return "\n".join(lines)

    def print_memory_state(self) -> None:
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
            for i, msg in enumerate(self.recent_messages[-self.recent_window_turns * 2 :], start=1):
                print(f"{i:02d}. {msg.type}: {msg.content}")
        print("======================================\n")

    def should_refresh_summary(self) -> bool:
        return self.turn_count > 0 and self.turn_count % self.summary_every_n_turns == 0

    def refresh_summary(self, chat_model: ChatHuggingFace) -> None:
        prompt = SUMMARY_PROMPT.format(
            old_summary=self.summary.strip() or "None",
            recent_dialogue=self.recent_dialogue_text(),
        )
        summary_messages = [HumanMessage(content=prompt)]
        response = chat_model.invoke(summary_messages)
        self.summary = response.content.strip()


def main() -> None:
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=TOKEN,
        provider="auto",
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False,
    )

    chat_model = ChatHuggingFace(llm=llm)
    memory = TemporaryMemory()

    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        memory.extract_stable_facts(user_text)

        prompt_messages = memory.build_prompt(user_text)
        response = chat_model.invoke(prompt_messages)

        assistant_text = response.content
        print("\nAssistant:", assistant_text)

        memory.add_user_message(user_text)
        memory.add_assistant_message(assistant_text)
        memory.turn_count += 1

        if memory.should_refresh_summary():
            memory.refresh_summary(chat_model)

        memory.print_memory_state()


if __name__ == "__main__":
    main()