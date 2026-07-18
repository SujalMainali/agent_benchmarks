"""BFCL runner — orchestrates episode -> adapter -> runtime -> bridge -> RunResult.

Mirrors the LoCoMo/ToolSandbox runner role. The runner owns the LLM provider
and builds one fresh agent + runtime per episode (each entry advertises its
own tool set), then delegates execution to the runtime bridge. It never talks
to the model directly and contains no evaluation logic.
"""

from __future__ import annotations

from typing import Any, List, Optional

from benchmarks.common.models import Episode, RunResult

from src.agent import ResearchHelperAgent
from src.runtime import ResearchHelperAgentRuntime

from .adapter import BFCLAdapter
from .runtime_bridge import BFCLRuntimeBridge


class BFCLRunner:
    """Runs BFCL episodes through the shared agent runtime."""

    def __init__(
        self,
        llm: Any,
        max_tool_steps: int = 1,
        adapter: Optional[BFCLAdapter] = None,
    ) -> None:
        """
        Args:
            llm: LLMProvider built by the provider factory (never a raw SDK).
            max_tool_steps: Agent tool-loop budget per entry. BFCL single-turn
                entries are scored on the first tool-calling response, so 1 is
                the faithful setting.
            adapter: Input transformation layer (defaults to BFCLAdapter).
        """
        self.llm = llm
        self.max_tool_steps = max_tool_steps
        self.adapter = adapter or BFCLAdapter()
        self.bridge = BFCLRuntimeBridge(
            runtime_factory=self._build_runtime,
            adapter=self.adapter,
        )

    def run_episode(self, episode: Episode) -> RunResult:
        """Run one entry and package the bridge output as a shared RunResult."""
        bridge_result = self.bridge.run(episode)

        tool_events = bridge_result["tool_events"]
        token_usage = bridge_result["token_usage"]

        return RunResult(
            sample_id=episode.episode_id,
            episode_id=episode.episode_id,
            question=episode.question,
            predicted_answer=str(bridge_result["raw_response"]),
            gold_answer=episode.gold_answer,
            trajectory=bridge_result["trajectory"],
            raw_messages=bridge_result["raw_messages"],
            benchmark_mode="bfcl",
            context_turn_count=len(episode.raw_data.get("question", [])),
            metrics={
                "latency_ms": bridge_result["latency_ms"],
                "tool_call_count": len(tool_events),
                **token_usage,
            },
            metadata={
                **episode.metadata,
                "raw_response": bridge_result["raw_response"],
                "decoded_ast": bridge_result["decoded_ast"],
                "decoded_execute": bridge_result["decoded_execute"],
                "inference_log": bridge_result["logs"],
            },
            total_latency_ms=bridge_result["latency_ms"],
            episode=episode,
            error=bridge_result["error"],
        )

    def run_batch(
        self,
        episodes: List[Episode],
        verbose: bool = True,
        on_result: Optional[Any] = None,
    ) -> List[RunResult]:
        """Run multiple entries sequentially.

        Args:
            on_result: Optional ``(run_result, index) -> None`` callback fired
                as each entry finishes — used to write raw artifacts actively
                during the run.
        """
        results: List[RunResult] = []
        for index, episode in enumerate(episodes):
            if verbose:
                print(f"Running entry {index + 1}/{len(episodes)}: {episode.episode_id}")
            result = self.run_episode(episode)
            results.append(result)
            if on_result is not None:
                on_result(result, index)
        return results

    # -- internals ----------------------------------------------------------

    def _build_runtime(
        self, system_prompt: str, tools: List[Any]
    ) -> ResearchHelperAgentRuntime:
        """Fresh agent + runtime bound to one entry's tools and system prompt.

        Uses the agent's normal tool-list initialization so the entry's
        functions are advertised through the same binding path as any other
        run (Phase 4 requirement).
        """
        agent = ResearchHelperAgent(
            llm=self.llm,
            tools=tools,
            max_tool_steps=self.max_tool_steps,
            system_prompt_override=system_prompt or None,
            allow_tools=True,
        )
        return ResearchHelperAgentRuntime(agent)
