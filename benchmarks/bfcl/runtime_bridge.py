"""BFCL runtime bridge — makes the repo's AgentRuntime look like a BFCL handler.

This is the only BFCL-specific execution layer. It drives one episode through
the shared runtime (reset -> observe -> act) and converts the structured
``Action``/trajectory output into exactly what the official BFCL evaluator
consumes:

    raw_response      official FC result format: [{func_name: '{json args}'}]
                      (or the assistant's text when no tool call was made)
    decoded_ast       [{func_name: {param: value}}]
    decoded_execute   ["func_name(param=value, ...)"]
    logs              LOG_GUIDE.md-style inference log entries

Decoding always reads the agent's structured tool calls (``ToolEvent``s the
runtime collected from ``Action``/trajectory) — assistant text is never
re-parsed. The executable strings are produced by the official
``convert_to_function_call`` helper so the semantics can never drift from the
official FC handlers.

The bridge talks to the model exclusively through the injected AgentRuntime;
it never instantiates LLMs or imports provider handlers.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from benchmarks.common.interfaces import AgentRuntime
from benchmarks.common.models import (
    EnvironmentState,
    Episode,
    Observation,
    ToolEvent,
    TrajectoryEvent,
)

from .adapter import BFCLAdapter
from .official import bootstrap_official

bootstrap_official()

from bfcl_eval.model_handler.utils import convert_to_function_call  # noqa: E402

# Builds a fresh runtime for one episode, bound to that episode's tools and
# system prompt. Constructed by the runner (which owns the LLM); the bridge
# only calls it.
RuntimeFactory = Callable[[str, List[Any]], AgentRuntime]


def decode_ast(tool_events: List[ToolEvent]) -> List[Dict[str, Any]]:
    """Structured tool calls -> official AST-checker input.

    Output format (per the official Contributing Guide):
        [{"tool_name": {"parameter": value, ...}}, ...]
    """
    return [{event.tool_name: dict(event.arguments)} for event in tool_events]


def decode_execute(tool_events: List[ToolEvent]) -> List[str]:
    """Structured tool calls -> official executable-checker strings.

    Delegates to the official ``convert_to_function_call`` so quoting/repr
    behavior is identical to the official FC handlers:
        ["tool_name(param=value, ...)"]
    """
    return convert_to_function_call(
        [{event.tool_name: dict(event.arguments)} for event in tool_events]
    )


def to_official_raw_response(
    tool_events: List[ToolEvent], assistant_text: str
) -> Any:
    """Mirror the official FC handlers' stored ``model_responses`` format.

    With tool calls:  [{func_name: '<json-serialized args>'}]
    Without:          the assistant's plain text (so official relevance /
                      irrelevance decoding behaves exactly as for FC models).
    """
    if tool_events:
        return [
            {event.tool_name: json.dumps(event.arguments)} for event in tool_events
        ]
    return assistant_text


class BFCLRuntimeBridge:
    """Runs one BFCL episode through the shared AgentRuntime."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        adapter: Optional[BFCLAdapter] = None,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.adapter = adapter or BFCLAdapter()

    def run(self, episode: Episode) -> Dict[str, Any]:
        """Play one episode and return handler-equivalent outputs.

        Returns a dict with ``raw_response``, ``decoded_ast``,
        ``decoded_execute``, ``logs`` plus supporting fields
        (``assistant_text``, ``tool_events``, ``trajectory``,
        ``raw_messages``, ``latency_ms``, ``token_usage``, ``error``).
        """
        agent_input = self.adapter.build_agent_input(episode)
        runtime = self.runtime_factory(
            agent_input["system_prompt"], agent_input["tools"]
        )

        initial_state = EnvironmentState(
            episode_id=episode.episode_id,
            messages=agent_input["messages"],
            allowed_tools=[tool.name for tool in agent_input["tools"]],
            metadata={"test_category": episode.metadata.get("test_category", "")},
        )
        observation = Observation(
            episode_id=episode.episode_id,
            text=episode.question,
            messages=agent_input["messages"],
            available_tools=initial_state.allowed_tools,
            metadata={"benchmark_mode": "bfcl"},
        )

        runtime.reset(episode, initial_state)

        start_time = time.time()
        error: Optional[str] = None
        try:
            action = runtime.act(observation)
            assistant_text = action.text
        except Exception as exc:  # noqa: BLE001 - runner surfaces the error
            action = None
            assistant_text = ""
            error = f"Runtime failure: {exc}"
        latency_ms = (time.time() - start_time) * 1000

        trajectory = runtime.get_trajectory().events
        raw_messages = (
            runtime.get_raw_messages() if hasattr(runtime, "get_raw_messages") else []
        )

        tool_events = self._model_tool_events(trajectory)
        raw_response = to_official_raw_response(tool_events, assistant_text)
        decoded_ast_output = decode_ast(tool_events)
        decoded_execute_output = decode_execute(tool_events)

        return {
            "raw_response": raw_response,
            "decoded_ast": decoded_ast_output,
            "decoded_execute": decoded_execute_output,
            "logs": self._build_logs(
                raw_response, decoded_ast_output, tool_events, error
            ),
            "assistant_text": assistant_text,
            "tool_events": tool_events,
            "action": action,
            "trajectory": trajectory,
            "raw_messages": raw_messages,
            "latency_ms": latency_ms,
            "token_usage": self._collect_token_usage(raw_messages),
            "error": error,
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _model_tool_events(trajectory: List[TrajectoryEvent]) -> List[ToolEvent]:
        """Tool calls from the model's first tool-calling response.

        BFCL single-turn scoring inspects the model's answer to the question:
        the first model step that requested tools (parallel calls arrive
        together in that one step). Later steps — the post-tool wrap-up the
        agent loop produces — are not part of the scored response.
        """
        for event in trajectory:
            if event.event_type == "model" and event.tool_calls:
                return list(event.tool_calls)
        return []

    @staticmethod
    def _build_logs(
        raw_response: Any,
        decoded_ast_output: List[Dict[str, Any]],
        tool_events: List[ToolEvent],
        error: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Per-entry inference log following the official LOG_GUIDE roles."""
        logs: List[Dict[str, Any]] = [
            {"role": "assistant", "content": raw_response}
        ]
        if error:
            logs.append(
                {
                    "role": "handler_log",
                    "content": "Runtime error during inference.",
                    "error": error,
                }
            )
        elif tool_events:
            logs.append(
                {
                    "role": "handler_log",
                    "content": "Successfully decoded model response.",
                    "model_response_decoded": decoded_ast_output,
                }
            )
        else:
            logs.append(
                {
                    "role": "handler_log",
                    "content": "Empty response from the model (no tool calls).",
                    "model_response_decoded": [],
                }
            )
        return logs

    @staticmethod
    def _collect_token_usage(raw_messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Best-effort token accounting from provider usage metadata."""
        input_tokens = 0
        output_tokens = 0
        for message in raw_messages:
            usage = (message.get("metadata") or {}).get("usage_metadata") or {}
            if isinstance(usage, dict):
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
        return {
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
        }
