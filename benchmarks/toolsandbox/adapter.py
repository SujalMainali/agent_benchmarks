"""ToolSandbox adapter: scenario Episodes -> agent-ready inputs.

Responsibilities (per the integration plan):
* preserve the scenario's role-based seed messages,
* build the initial dialogue context (role-preserving langchain messages),
* surface the scenario's allow-listed tool names,
* provide the benchmark overlay system prompt.

The actual binding of tools and the system prompt happens in the runner/agent
role; this adapter only shapes the inputs. It never enumerates tools in the
prompt — the runtime passes the real allow-listed schemas to the provider.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from benchmarks.common.interfaces import BenchmarkAdapter
from benchmarks.common.models import BenchmarkSample, Episode

from .prompts import (
    TOOLSANDBOX_STATEFUL_PROMPT,
    TOOLSANDBOX_STRICT_RESPONSE_PROMPT,
    TOOLSANDBOX_SYSTEM_PROMPT,
    TOOLSANDBOX_TOOL_USE_PROMPT,
)


class ToolSandboxAdapter(BenchmarkAdapter):
    """Converts ToolSandbox Episodes into agent inputs."""

    def __init__(self, system_prompt_override: str = "") -> None:
        self.system_prompt_override = system_prompt_override

    def load_sample(self, sample_data: Dict[str, Any]) -> BenchmarkSample:
        """Load a raw ``{"name", "scenario"}`` record via the loader."""
        from .loader import ToolSandboxLoader

        episode = ToolSandboxLoader().load(sample_data)
        return BenchmarkSample.from_episode(episode)

    def build_system_prompt(self) -> str:
        """Assemble the benchmark overlay system prompt (behavior only)."""
        if self.system_prompt_override:
            return self.system_prompt_override
        return "\n".join(
            [
                TOOLSANDBOX_SYSTEM_PROMPT.strip(),
                TOOLSANDBOX_TOOL_USE_PROMPT.strip(),
                TOOLSANDBOX_STATEFUL_PROMPT.strip(),
                TOOLSANDBOX_STRICT_RESPONSE_PROMPT.strip(),
            ]
        )

    def build_context_messages(self, sample: BenchmarkSample | Episode) -> List[BaseMessage]:
        """Build role-preserving langchain messages from the scenario seeds.

        SYSTEM->USER seeds are user-simulator few-shots and are omitted from the
        agent's view. USER->AGENT seeds become HumanMessages; AGENT->USER seeds
        become AIMessages. The behavioral overlay is prepended as a SystemMessage.
        """
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        messages: List[BaseMessage] = [SystemMessage(content=self.build_system_prompt())]

        for seed in episode.context.get("seed_messages", []):
            sender = str(seed.get("sender", "")).lower()
            recipient = str(seed.get("recipient", "")).lower()
            content = str(seed.get("content", "")).strip()
            if not content:
                continue
            if sender.endswith("user") and recipient.endswith("agent"):
                messages.append(HumanMessage(content=content))
            elif sender.endswith("agent") and recipient.endswith("user"):
                messages.append(AIMessage(content=content))
            # System->User few-shot instructions are intentionally skipped.
        return messages

    def build_agent_input(self, sample: BenchmarkSample | Episode) -> Dict[str, Any]:
        """Convert an Episode into ToolSandbox agent input."""
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        return {
            "question": episode.question,
            "scenario_name": episode.metadata.get("scenario_name", episode.episode_id),
            "tool_allow_list": episode.context.get("tool_allow_list", []),
            "system_prompt": self.build_system_prompt(),
            "context_messages": self.build_context_messages(episode),
            "categories": episode.context.get("categories", []),
            "mode": episode.mode,
            "metadata": episode.metadata,
            "episode": episode,
        }
