"""ToolSandbox runner - orchestrates stateful scenario rollouts.

The runner lives in the MAIN interpreter (research-helper's env) and never
imports ``tool_sandbox``. It drives scenarios through the isolated worker
process via :class:`~benchmarks.toolsandbox.official_bridge.ToolSandboxClient`:

    episode -> benchmark prompt overlay (adapter)
            -> client.run_scenario(name, system_prompt, ...)  # worker plays it
            -> terminal result dict (conversation, world_state, state_trace,
               predicted, evaluation, error)
            -> shared RunResult + metrics

The worker owns all official-engine execution and world-state mutation. The
runner only converts the worker's plain-dict result into the shared
:class:`~benchmarks.common.models.RunResult` and computes metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from benchmarks.common.models import (
    EnvironmentState,
    Episode,
    Milestone,
    Minefield,
    RunResult,
    ToolEvent,
    Trajectory,
    TrajectoryEvent,
)

from .adapter import ToolSandboxAdapter
from .metrics import ToolSandboxMetrics
from .official_bridge import (
    ToolSandboxClient,
    ToolSandboxWorkerError,
    make_inference_fn,
)
from .runtime_bridge import ToolSandboxRuntimeSession


class ToolSandboxRunner:
    """Runs ToolSandbox scenarios through the isolated worker + our LLM."""

    def __init__(
        self,
        llm: Any,
        python_executable: str,
        official_root: str = "third_party/ToolSandbox-official",
        user_mode: str = "scripted",
        max_turns: int = 0,
        adapter: Optional[ToolSandboxAdapter] = None,
        agent_mode: str = "runtime",
        max_tool_steps: int = 8,
        fault_rate: float = 0.0,
        fault_seed: int = 13,
        real_search_tools: bool = False,
        rapid_api_key: Optional[str] = None,
        user_api_key: Optional[str] = None,
        user_base_url: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.python_executable = python_executable
        self.official_root = official_root
        self.user_mode = user_mode
        self.max_turns = max_turns
        self.adapter = adapter or ToolSandboxAdapter()
        self.agent_mode = agent_mode
        self.max_tool_steps = max_tool_steps
        self.fault_rate = fault_rate
        self.fault_seed = fault_seed
        self.client = ToolSandboxClient(
            python_executable=python_executable,
            official_root=official_root,
            inference_fn=make_inference_fn(llm),
            real_search_tools=real_search_tools,
            rapid_api_key=rapid_api_key,
            user_api_key=user_api_key,
            user_base_url=user_base_url,
        )

    def run_episode(self, episode: Episode) -> RunResult:
        """Play out and score a single scenario in the worker."""
        scenario_name = episode.metadata.get("scenario_name", episode.episode_id)

        agent_input = self.adapter.build_agent_input(episode)
        system_prompt = agent_input["system_prompt"]

        # In runtime mode, our agent (ResearchHelperAgentRuntime) drives the
        # whole turn; the client tunnels each of its tool calls to the worker.
        session = None
        if self.agent_mode == "runtime":
            session = ToolSandboxRuntimeSession(
                llm=self.llm,
                system_prompt=system_prompt,
                max_tool_steps=self.max_tool_steps,
                episode=episode,
            )
            self.client.agent_turn_fn = session.agent_turn_fn

        start_time = time.time()
        try:
            result = self.client.run_scenario(
                name=scenario_name,
                system_prompt=system_prompt,
                max_turns=self.max_turns,
                user_mode=self.user_mode,
                agent_mode=self.agent_mode,
                fault_rate=self.fault_rate,
                fault_seed=self.fault_seed,
            )
        except ToolSandboxWorkerError as exc:
            latency_ms = (time.time() - start_time) * 1000
            return self._error_result(
                episode, f"ToolSandbox worker failed: {exc}", latency_ms
            )

        latency_ms = (time.time() - start_time) * 1000
        return self._build_run_result(episode, result, latency_ms, session)

    def run_batch(
        self,
        episodes: List[Episode],
        verbose: bool = True,
    ) -> List[RunResult]:
        """Run multiple scenarios sequentially (each in a fresh worker)."""
        results: List[RunResult] = []
        for index, episode in enumerate(episodes):
            name = episode.metadata.get("scenario_name", episode.episode_id)
            if verbose:
                print(f"Running scenario {index + 1}/{len(episodes)}: {name}")
            results.append(self.run_episode(episode))
        return results

    # -- internals ----------------------------------------------------------

    def _build_run_result(
        self,
        episode: Episode,
        result: Dict[str, Any],
        latency_ms: float,
        session: Optional[ToolSandboxRuntimeSession] = None,
    ) -> RunResult:
        conversation: List[Dict[str, Any]] = result.get("conversation", []) or []
        world_state: Dict[str, Any] = result.get("world_state", {}) or {}
        state_trace: List[Dict[str, Any]] = result.get("state_trace", []) or []
        official_eval: Dict[str, Any] = result.get("evaluation", {}) or {}
        predicted = result.get("predicted", "") or ""
        error = result.get("error")
        fault_injections: List[Dict[str, Any]] = result.get("fault_injections", []) or []

        trajectory = self._build_trajectory(conversation, state_trace)
        final_state = self._build_final_state(
            episode, world_state, official_eval, state_trace
        )

        metadata: Dict[str, Any] = {
            **episode.metadata,
            "state_change_count": len(state_trace),
            "world_state": world_state,
            "state_trace": state_trace,
            "agent_mode": result.get("agent_mode", self.agent_mode),
            "fault_injections": fault_injections,
            "parallel_batching": "sequential",
        }

        # In runtime mode, attach the parent-side runtime view (agent-internal
        # steps) alongside the worker-conversation trajectory used by scoring.
        if session is not None and session.runtime is not None:
            metadata["runtime_trajectory"] = [
                event.__dict__ for event in session.runtime.get_trajectory().events
            ]
            metadata["runtime_fault_events"] = list(session.proxy.fault_events)

        run_result = RunResult(
            sample_id=episode.episode_id,
            episode_id=episode.episode_id,
            question=episode.question,
            predicted_answer=predicted,
            gold_answer=episode.gold_answer,
            trajectory=trajectory.events,
            raw_messages=conversation,
            benchmark_mode="tool_sandbox",
            context_turn_count=sum(
                1 for m in conversation if str(m.get("sender", "")).endswith("user")
            ),
            metadata=metadata,
            total_latency_ms=latency_ms,
            episode=episode,
            final_state=final_state,
            official_eval=official_eval,
            error=error,
        )
        run_result.metrics = ToolSandboxMetrics.compute_metrics(run_result)
        return run_result

    def _error_result(
        self, episode: Episode, error: str, latency_ms: float
    ) -> RunResult:
        """Wrap a worker/transport failure into a RunResult without crashing."""
        final_state = self._build_final_state(episode, {}, {}, [])
        run_result = RunResult(
            sample_id=episode.episode_id,
            episode_id=episode.episode_id,
            question=episode.question,
            predicted_answer="",
            gold_answer=episode.gold_answer,
            benchmark_mode="tool_sandbox",
            metadata={
                **episode.metadata,
                "state_change_count": 0,
                "world_state": {},
                "state_trace": [],
            },
            total_latency_ms=latency_ms,
            episode=episode,
            final_state=final_state,
            official_eval={},
            error=error,
        )
        run_result.metrics = ToolSandboxMetrics.compute_metrics(run_result)
        return run_result

    def _build_trajectory(
        self, conversation: List[Dict[str, Any]], state_trace: List[Dict[str, Any]]
    ) -> Trajectory:
        trajectory = Trajectory()
        turn = 0
        for message in conversation:
            turn += 1
            sender = str(message.get("sender", ""))
            recipient = str(message.get("recipient", ""))
            content = message.get("content", "")
            exception = message.get("tool_call_exception")

            tool_calls: List[ToolEvent] = []
            event_type = "message"
            agent_message = ""
            if sender.endswith("agent") and recipient.endswith("execution_environment"):
                event_type = "tool_call"
                tool_calls.append(
                    ToolEvent(
                        tool_name=str(message.get("openai_function_name") or ""),
                        arguments={},
                        result="",
                    )
                )
            elif sender.endswith("execution_environment") and recipient.endswith("agent"):
                event_type = "tool_result"
                if tool_calls_from := message.get("openai_function_name"):
                    tool_calls.append(
                        ToolEvent(
                            tool_name=str(tool_calls_from),
                            arguments={},
                            result=content,
                        )
                    )
            elif sender.endswith("agent") and recipient.endswith("user"):
                event_type = "final"
                agent_message = content

            trajectory.append(
                TrajectoryEvent(
                    event_type=event_type,
                    turn_number=turn,
                    agent_message=agent_message,
                    actor=sender,
                    recipient=recipient,
                    tool_calls=tool_calls,
                    exception=str(exception) if exception else None,
                    metadata={"content": content},
                )
            )
        return trajectory

    def _build_final_state(
        self,
        episode: Episode,
        world_state: Dict[str, Any],
        official_eval: Dict[str, Any],
        state_trace: List[Dict[str, Any]],
    ) -> EnvironmentState:
        matched = set((official_eval.get("milestone_mapping") or {}).keys())
        tripped = set((official_eval.get("minefield_mapping") or {}).keys())
        name = episode.episode_id

        milestones = [
            Milestone(
                milestone_id=f"{name}::milestone::{i}",
                description="ToolSandbox milestone",
                kind="state_change",
                satisfied=str(i) in matched or i in matched,
            )
            for i in range(
                int(
                    official_eval.get(
                        "milestone_count",
                        episode.metadata.get("milestone_count", 0),
                    )
                )
            )
        ]
        minefields = [
            Minefield(
                minefield_id=f"{name}::minefield::{i}",
                description="ToolSandbox minefield",
                kind="state_change",
                tripped=str(i) in tripped or i in tripped,
            )
            for i in range(
                int(
                    official_eval.get(
                        "minefield_count",
                        episode.metadata.get("minefield_count", 0),
                    )
                )
            )
        ]

        return EnvironmentState(
            episode_id=name,
            done=True,
            world_state=world_state,
            allowed_tools=list(episode.metadata.get("tool_allow_list", [])),
            milestones=milestones,
            minefields=minefields,
            turn_index=len(state_trace),
            metadata={
                "scenario_name": episode.metadata.get("scenario_name"),
                "categories": episode.metadata.get("categories", []),
                "official_eval": official_eval,
            },
        )
