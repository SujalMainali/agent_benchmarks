"""ToolSandbox worker — runs under the isolated ToolSandbox interpreter.

This is the ONLY module that imports ``tool_sandbox``. It is never imported by
the main process; instead it is spawned as a subprocess with the ToolSandbox
virtualenv's Python (``TOOLSANDBOX_PYTHON``) and speaks JSON-lines over stdio to
``official_bridge`` in the main process.

Subcommands
-----------
``list-scenarios``
    Print a single JSON array of normalized scenario specs, then exit.

``run-scenario --name NAME [--max-turns N] [--user-mode scripted]``
    Play out one scenario against the official engine. Whenever the agent needs
    a model completion, emit an ``inference_request`` line and block on stdin for
    the matching ``inference_response``. When the rollout finishes, emit a single
    ``result`` line.

Protocol (one JSON object per line)
-----------------------------------
worker -> main : {"type":"inference_request","id":N,"messages":[...],"tools":[...]|null}
main -> worker : {"type":"inference_response","id":N,"text":str,"tool_calls":[{id,name,arguments}]}
worker -> main : {"type":"result", ...}          # terminal, run-scenario
worker -> main : {"type":"error","message":str}  # terminal, on failure

All protocol traffic goes over a private duplicated stdout fd. The process's
real stdout is redirected to stderr so that tqdm/library prints can never
corrupt the JSON stream.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Protocol channel setup: isolate the JSON stream from stray stdout writes.
# ---------------------------------------------------------------------------

# Duplicate the real stdout to a private fd for protocol messages, then point
# fd 1 (and sys.stdout) at stderr so nothing else pollutes the channel.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_PROTOCOL = os.fdopen(_PROTOCOL_FD, "w", buffering=1, encoding="utf-8")
_STDIN = sys.stdin


def _send(obj: Dict[str, Any]) -> None:
    """Write one JSON object as a line on the protocol channel."""
    _PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def _recv() -> Optional[Dict[str, Any]]:
    """Read one JSON object (a line) from stdin, or None on EOF."""
    line = _STDIN.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line)


# ---------------------------------------------------------------------------
# tool_sandbox imports (only valid inside the ToolSandbox interpreter)
# ---------------------------------------------------------------------------


def _ensure_importable() -> None:
    """Make ``tool_sandbox`` importable from an explicit root if provided.

    Normally the worker runs with the ToolSandbox venv's Python where the
    package is installed. ``TOOLSANDBOX_OFFICIAL_ROOT`` allows a sys.path
    fallback for a source checkout.
    """
    root = os.environ.get("TOOLSANDBOX_OFFICIAL_ROOT", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _role_types():
    from tool_sandbox.common.execution_context import RoleType

    return RoleType


def _database_namespaces():
    from tool_sandbox.common.execution_context import DatabaseNamespace

    return DatabaseNamespace


# ---------------------------------------------------------------------------
# Value / row serialization helpers
# ---------------------------------------------------------------------------


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(getattr(value, "value", value))


def _value_to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_value_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _value_to_jsonable(v) for k, v in value.items()}
    return str(value)


def _rows_to_jsonable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _value_to_jsonable(v) for k, v in row.items()} for row in rows]


def _mapping_to_jsonable(mapping: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        items = mapping.items()
    except AttributeError:
        return result
    for key, value in items:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            result[str(key)] = [int(value[0]), float(value[1])]
        else:
            result[str(key)] = _value_to_jsonable(value)
    return result


# ---------------------------------------------------------------------------
# Scenario normalization
# ---------------------------------------------------------------------------


def _seed_messages(scenario: Any) -> List[Dict[str, Any]]:
    DatabaseNamespace = _database_namespaces()
    context = scenario.starting_context
    try:
        df = context.get_database(
            namespace=DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=True,
        )
    except Exception:
        return []
    messages: List[Dict[str, Any]] = []
    for row in df.to_dicts():
        messages.append(
            {
                "sender": _to_str(row.get("sender")),
                "recipient": _to_str(row.get("recipient")),
                "content": row.get("content", "") or "",
                "visible_to": [_to_str(v) for v in (row.get("visible_to") or [])],
            }
        )
    return messages


def _first_user_utterance(seed_messages: List[Dict[str, Any]]) -> str:
    RoleType = _role_types()
    for message in seed_messages:
        if message["sender"] == str(RoleType.USER) and message["recipient"] == str(
            RoleType.AGENT
        ):
            return message["content"]
    return ""


def _scenario_spec(name: str, scenario: Any) -> Dict[str, Any]:
    context = scenario.starting_context
    tool_allow_list = list(getattr(context, "tool_allow_list", None) or [])
    tool_deny_list = list(getattr(context, "tool_deny_list", None) or [])
    categories = [_to_str(c) for c in getattr(scenario, "categories", []) or []]

    evaluation = getattr(scenario, "evaluation", None)
    milestone_count = 0
    minefield_count = 0
    if evaluation is not None:
        milestone_count = len(getattr(evaluation.milestone_matcher, "milestones", []) or [])
        minefield_count = len(getattr(evaluation.minefield_matcher, "milestones", []) or [])

    seed_messages = _seed_messages(scenario)
    return {
        "scenario_id": name,
        "name": name,
        "tool_allow_list": tool_allow_list,
        "tool_deny_list": tool_deny_list,
        "categories": categories,
        "max_messages": int(getattr(scenario, "max_messages", 30)),
        "milestone_count": milestone_count,
        "minefield_count": minefield_count,
        "seed_messages": seed_messages,
        "first_user_utterance": _first_user_utterance(seed_messages),
    }


def _load_named_scenarios() -> Dict[str, Any]:
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    return named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)


# ---------------------------------------------------------------------------
# World-state / conversation / trace serialization from an ExecutionContext
# ---------------------------------------------------------------------------


def _serialize_world_state(context: Any) -> Dict[str, List[Dict[str, Any]]]:
    DatabaseNamespace = _database_namespaces()
    state: Dict[str, List[Dict[str, Any]]] = {}
    for namespace in DatabaseNamespace:
        if namespace == DatabaseNamespace.SANDBOX:
            continue
        try:
            df = context.get_database(namespace=namespace)
            state[_to_str(namespace)] = _rows_to_jsonable(df.to_dicts())
        except Exception as exc:  # pragma: no cover - defensive
            state[_to_str(namespace)] = [{"error": str(exc)}]
    return state


def _extract_conversation(context: Any) -> List[Dict[str, Any]]:
    DatabaseNamespace = _database_namespaces()
    RoleType = _role_types()
    try:
        df = context.get_database(
            namespace=DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=True,
        )
    except Exception:
        return []

    conversation: List[Dict[str, Any]] = []
    for row in df.to_dicts():
        sender = _to_str(row.get("sender"))
        recipient = _to_str(row.get("recipient"))
        visible_to = [_to_str(v) for v in (row.get("visible_to") or [])]
        if visible_to == [str(RoleType.USER)]:
            continue
        if sender != str(RoleType.AGENT) and recipient != str(RoleType.AGENT):
            continue
        conversation.append(
            {
                "sender": sender,
                "recipient": recipient,
                "content": row.get("content", "") or "",
                "tool_call_exception": row.get("tool_call_exception"),
                "openai_function_name": row.get("openai_function_name"),
                "tool_trace": _value_to_jsonable(row.get("tool_trace")),
            }
        )
    return conversation


def _last_agent_message_to_user(conversation: List[Dict[str, Any]]) -> str:
    RoleType = _role_types()
    for message in reversed(conversation):
        if message["sender"] == str(RoleType.AGENT) and message["recipient"] == str(
            RoleType.USER
        ):
            return message["content"]
    return ""


def _build_state_trace(context: Any) -> List[Dict[str, Any]]:
    from tool_sandbox.common.message_conversion import (
        get_snapshot_indices_to_databases,
    )

    try:
        snapshot_map = get_snapshot_indices_to_databases(context)
    except Exception:
        return []
    steps: List[Dict[str, Any]] = []
    for snapshot_index in sorted(snapshot_map.keys()):
        databases = snapshot_map[snapshot_index]
        steps.append(
            {
                "snapshot_index": int(snapshot_index),
                "database_updates": {
                    _to_str(name): _rows_to_jsonable(df.to_dicts())
                    for name, df in databases.items()
                },
            }
        )
    return steps


# ---------------------------------------------------------------------------
# Roles: LLM-proxy agent (over stdio) + scripted user
# ---------------------------------------------------------------------------


def _build_proxy_agent(system_prompt: str, model_name: str = "research-helper") -> Any:
    """Official OpenAI agent whose model_inference proxies to the parent."""
    from openai import NOT_GIVEN
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

    class ProxyAgent(OpenAIAPIAgent):
        def __init__(self, prompt: str, name: str) -> None:
            # Do NOT call super().__init__(): avoid constructing an OpenAI client.
            self._system_prompt = prompt
            self.model_name = name
            self._request_id = 0

        def model_inference(self, openai_messages, openai_tools):  # type: ignore[override]
            tools = None if openai_tools is NOT_GIVEN else list(openai_tools)
            messages = _inject_system_prompt(list(openai_messages), self._system_prompt)
            self._request_id += 1
            _send(
                {
                    "type": "inference_request",
                    "id": self._request_id,
                    "messages": messages,
                    "tools": tools,
                }
            )
            response = _recv()
            if response is None:
                raise RuntimeError("Parent closed the stream during inference.")
            if response.get("type") == "error":
                raise RuntimeError(f"Parent inference error: {response.get('message')}")
            return _response_to_chat_completion(response, self.model_name)

    return ProxyAgent(system_prompt, model_name)


# ---------------------------------------------------------------------------
# Remote-runtime agent: delegates the whole turn to the parent's runtime,
# and services the parent's tool calls against the official environment.
# ---------------------------------------------------------------------------

# Synthetic transient error returned to the agent when a fault is injected.
# The call never touches the sandbox, so it consumes no message budget and
# mutates no world state — it only exercises the agent's recovery behavior.
FAULT_MESSAGE = (
    "TransientToolError: the tool backend timed out. "
    "The call had no effect. Please retry."
)


def _history_payload(messages: List[Any]) -> List[Dict[str, Any]]:
    """Official message history -> JSON-safe OpenAI-format dicts for the parent."""
    from tool_sandbox.common.message_conversion import to_openai_messages

    openai_messages, _ = to_openai_messages(messages)
    return [
        {k: _value_to_jsonable(v) for k, v in message.items()}
        for message in openai_messages
    ]


def _tool_schemas(role: Any) -> List[Dict[str, Any]]:
    """Allow-list + scrambling-aware OpenAI tool schemas for the agent's tools."""
    from tool_sandbox.common.tool_conversion import convert_to_openai_tools

    schemas = convert_to_openai_tools(role.get_available_tools())
    return [{k: _value_to_jsonable(v) for k, v in schema.items()} for schema in schemas]


def _build_runtime_agent(fault_rate: float, fault_seed: int) -> Any:
    """Agent role whose whole turn is driven by the parent's runtime over stdio."""
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )
    from tool_sandbox.common.execution_context import RoleType, get_current_context
    from tool_sandbox.common.message_conversion import (
        Message,
        openai_tool_call_to_python_code,
    )
    from tool_sandbox.roles.base_role import BaseRole
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment

    class RemoteRuntimeAgent(BaseRole):
        role_type: Any = RoleType.AGENT

        def __init__(self, rate: float, seed: int) -> None:
            self._turn_id = 0
            self._fault_rate = rate
            self._rng = random.Random(seed)
            self._fault_log: List[Dict[str, Any]] = []
            self._environment = ExecutionEnvironment()

        def respond(self, ending_index: Optional[int] = None) -> None:
            messages = self.get_messages(ending_index=ending_index)
            self.messages_validation(messages=messages)
            messages = self.filter_messages(messages=messages)
            # System is a special role: roles do not respond back to System.
            if messages[-1].sender == RoleType.SYSTEM:
                return
            self._turn_id += 1
            _send(
                {
                    "type": "agent_turn_request",
                    "id": self._turn_id,
                    "messages": _history_payload(messages),
                    "tools": _tool_schemas(self),
                }
            )
            while True:
                reply = _recv()
                if reply is None:
                    raise RuntimeError("Parent closed the stream mid-turn.")
                rtype = reply.get("type")
                if rtype == "error":
                    raise RuntimeError(f"Parent error: {reply.get('message')}")
                if rtype == "tool_call_request":
                    _send(self._handle_tool_call(reply))
                elif rtype == "agent_turn_done":
                    self.add_messages(
                        [
                            Message(
                                sender=self.role_type,
                                recipient=RoleType.USER,
                                content=str(reply.get("text", "")),
                            )
                        ]
                    )
                    return
                else:
                    raise RuntimeError(f"Unexpected message in turn: {rtype}")

        def _handle_tool_call(self, reply: Dict[str, Any]) -> Dict[str, Any]:
            call_id = reply.get("call_id")
            name = str(reply.get("name", ""))
            arguments = reply.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                arguments = {}

            # Fault gate: BEFORE any sandbox interaction. No message appended,
            # no snapshot, no tool_trace, no max_messages consumption.
            if self._fault_rate > 0.0 and self._rng.random() < self._fault_rate:
                self._fault_log.append(
                    {
                        "turn": self._turn_id,
                        "call_id": call_id,
                        "tool": name,
                        "arguments": _value_to_jsonable(arguments),
                    }
                )
                return {
                    "type": "tool_call_response",
                    "id": self._turn_id,
                    "call_id": call_id,
                    "result": "",
                    "exception": FAULT_MESSAGE,
                    "fault_injected": True,
                }

            # Real execution against the official sandbox world state.
            current_context = get_current_context()
            available_tool_names = set(self.get_available_tools().keys())
            openai_tool_call = ChatCompletionMessageToolCall(
                id=f"tool_call_{self._turn_id}_{call_id}",
                type="function",
                function=Function(name=name, arguments=json.dumps(arguments)),
            )
            try:
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(name)
                )
                code = openai_tool_call_to_python_code(
                    openai_tool_call,
                    available_tool_names,
                    execution_facing_tool_name=execution_facing_tool_name,
                )
            except Exception as exc:  # noqa: BLE001 - unknown tool / bad args
                return {
                    "type": "tool_call_response",
                    "id": self._turn_id,
                    "call_id": call_id,
                    "result": "",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "fault_injected": False,
                }

            self.add_messages(
                [
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=code,
                        openai_tool_call_id=openai_tool_call.id,
                        openai_function_name=name,
                    )
                ]
            )
            self._environment.respond()

            # Read the environment's reply (last EXECUTION_ENVIRONMENT->AGENT row).
            last = self.get_messages()[-1]
            exception = last.tool_call_exception
            return {
                "type": "tool_call_response",
                "id": self._turn_id,
                "call_id": call_id,
                "result": str(last.content or ""),
                "exception": str(exception) if exception else None,
                "fault_injected": False,
            }

    return RemoteRuntimeAgent(fault_rate, fault_seed)


def _build_scripted_user(utterances: List[str]) -> Any:
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.roles.base_role import BaseRole

    class ScriptedUser(BaseRole):
        role_type: Any = RoleType.USER

        def __init__(self, scripted: List[str]) -> None:
            self._scripted = list(scripted)
            self._index = 0

        def respond(self, ending_index: Optional[int] = None) -> None:
            messages = self.get_messages(ending_index=ending_index)
            self.messages_validation(messages=messages)
            messages = self.filter_messages(messages=messages)
            if messages[-1].sender == RoleType.SYSTEM:
                return
            if self._index < len(self._scripted):
                text = self._scripted[self._index]
                self._index += 1
                self.add_messages(
                    [Message(sender=RoleType.USER, recipient=RoleType.AGENT, content=text)]
                )
                return
            end_code = (
                "end_conversation_response = end_conversation()\n"
                "print(repr(end_conversation_response))"
            )
            self.add_messages(
                [
                    Message(
                        sender=RoleType.USER,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=end_code,
                    )
                ]
            )

    return ScriptedUser(utterances)


def _build_official_user(user_impl: str) -> Any:
    from tool_sandbox.roles.openai_api_user import (
        GPT_3_5_0125_User,
        GPT_4_0125_User,
        GPT_4_o_2024_05_13_User,
    )

    mapping = {
        "gpt-4o": GPT_4_o_2024_05_13_User,
        "gpt-4": GPT_4_0125_User,
        "gpt-3.5": GPT_3_5_0125_User,
    }
    return mapping.get(user_impl.lower(), GPT_4_o_2024_05_13_User)()


def _build_roles(
    system_prompt: str,
    user_mode: str,
    scripted: List[str],
    agent_mode: str = "llm_proxy",
    fault_rate: float = 0.0,
    fault_seed: int = 0,
) -> Dict[Any, Any]:
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment

    user_role = (
        _build_scripted_user(scripted)
        if user_mode == "scripted"
        else _build_official_user(user_mode)
    )
    if agent_mode == "runtime":
        agent_role = _build_runtime_agent(fault_rate, fault_seed)
    else:
        agent_role = _build_proxy_agent(system_prompt)
    return {
        RoleType.AGENT: agent_role,
        RoleType.USER: user_role,
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
    }


# ---------------------------------------------------------------------------
# OpenAI message/response conversion (worker side)
# ---------------------------------------------------------------------------


def _inject_system_prompt(openai_messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    """Prepend the benchmark overlay system prompt if none is present."""
    if not system_prompt:
        return openai_messages
    if any(m.get("role") == "system" for m in openai_messages):
        return openai_messages
    return [{"role": "system", "content": system_prompt}, *openai_messages]


def _response_to_chat_completion(response: Dict[str, Any], model_name: str):
    """Rebuild an OpenAI ``ChatCompletion`` from the parent's inference reply."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    tool_calls = []
    for index, call in enumerate(response.get("tool_calls", []) or []):
        arguments = call.get("arguments", {}) or {}
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(
            ChatCompletionMessageToolCall(
                id=call.get("id") or f"call_{index}",
                type="function",
                function=Function(name=call.get("name", ""), arguments=arguments),
            )
        )
    message = ChatCompletionMessage(
        role="assistant",
        content=response.get("text", "") or "",
        tool_calls=tool_calls or None,
    )
    choice = Choice(
        index=0,
        finish_reason="tool_calls" if tool_calls else "stop",
        message=message,
    )
    return ChatCompletion(
        id="research-helper-completion",
        choices=[choice],
        created=0,
        model=model_name,
        object="chat.completion",
    )


# ---------------------------------------------------------------------------
# Rollout & evaluation
# ---------------------------------------------------------------------------


def _play_and_evaluate(
    scenario: Any, roles: Dict[Any, Any], scenario_name: str
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """Play the scenario, returning (context, evaluation_dict, error)."""
    from tool_sandbox.common.execution_context import get_current_context

    error: Optional[str] = None
    try:
        context = scenario.play(roles=roles, scenario_name=scenario_name)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        try:
            context = get_current_context()
        except Exception:
            return None, {}, error

    evaluation: Dict[str, Any] = {}
    try:
        result = scenario.evaluation.evaluate(
            execution_context=context, max_turn_count=scenario.max_messages
        )
        evaluation = {
            "milestone_similarity": float(result.milestone_similarity),
            "minefield_similarity": float(result.minefield_similarity),
            "similarity": float(result.similarity),
            "turn_count": int(result.turn_count),
            "milestone_mapping": _mapping_to_jsonable(result.milestone_mapping),
            "minefield_mapping": _mapping_to_jsonable(result.minefield_mapping),
            "milestone_count": len(
                getattr(scenario.evaluation.milestone_matcher, "milestones", []) or []
            ),
            "minefield_count": len(
                getattr(scenario.evaluation.minefield_matcher, "milestones", []) or []
            ),
        }
    except Exception as exc:  # noqa: BLE001
        error = error or f"Evaluation error: {type(exc).__name__}: {exc}"
    return context, evaluation, error


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_list_scenarios() -> None:
    scenarios = _load_named_scenarios()
    specs = [_scenario_spec(name, scenario) for name, scenario in scenarios.items()]
    _send({"type": "scenarios", "scenarios": specs})


def _cmd_run_scenario(
    name: str,
    max_turns: int,
    user_mode: str,
    system_prompt: str,
    agent_mode: str = "llm_proxy",
    fault_rate: float = 0.0,
    fault_seed: int = 0,
) -> None:
    from tool_sandbox.common.execution_context import RoleType, set_current_context

    scenarios = _load_named_scenarios()
    scenario = scenarios.get(name)
    if scenario is None:
        _send({"type": "error", "message": f"Scenario '{name}' not found."})
        return

    scenario = copy.deepcopy(scenario)
    if max_turns and max_turns > 0:
        try:
            scenario.max_messages = min(int(scenario.max_messages), int(max_turns))
        except Exception:
            pass

    # Reset global context to this scenario's starting state.
    set_current_context(copy.deepcopy(scenario.starting_context))

    roles = _build_roles(
        system_prompt,
        user_mode,
        scripted=[],
        agent_mode=agent_mode,
        fault_rate=fault_rate,
        fault_seed=fault_seed,
    )
    context, evaluation, error = _play_and_evaluate(scenario, roles, name)

    fault_injections = list(getattr(roles[RoleType.AGENT], "_fault_log", []) or [])

    conversation = _extract_conversation(context) if context is not None else []
    result = {
        "type": "result",
        "scenario_name": name,
        "conversation": conversation,
        "world_state": _serialize_world_state(context) if context is not None else {},
        "state_trace": _build_state_trace(context) if context is not None else [],
        "predicted": _last_agent_message_to_user(conversation),
        "evaluation": evaluation,
        "agent_mode": agent_mode,
        "fault_injections": fault_injections,
        "error": error,
    }
    _send(result)


def main() -> None:
    _ensure_importable()

    parser = argparse.ArgumentParser(description="ToolSandbox stdio worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-scenarios")

    run_parser = subparsers.add_parser("run-scenario")
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--max-turns", type=int, default=0)
    run_parser.add_argument("--user-mode", default="scripted")
    run_parser.add_argument("--system-prompt", default="")
    run_parser.add_argument(
        "--agent-mode", choices=["llm_proxy", "runtime"], default="llm_proxy"
    )
    run_parser.add_argument("--fault-rate", type=float, default=0.0)
    run_parser.add_argument("--fault-seed", type=int, default=0)

    args = parser.parse_args()

    try:
        if args.command == "list-scenarios":
            _cmd_list_scenarios()
        elif args.command == "run-scenario":
            _cmd_run_scenario(
                name=args.name,
                max_turns=args.max_turns,
                user_mode=args.user_mode,
                system_prompt=args.system_prompt,
                agent_mode=args.agent_mode,
                fault_rate=args.fault_rate,
                fault_seed=args.fault_seed,
            )
        else:  # pragma: no cover - argparse enforces choices
            _send({"type": "error", "message": f"Unknown command {args.command!r}"})
    except Exception as exc:  # noqa: BLE001 - report any worker failure
        _send(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
