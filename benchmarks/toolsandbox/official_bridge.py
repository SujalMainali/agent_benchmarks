"""Main-process client for the isolated ToolSandbox worker.

This module runs in the MAIN interpreter (research-helper's env). It does NOT
import ``tool_sandbox`` — that package lives in a separate virtualenv with
incompatible pins (polars 0.20, numpy 1.26, ...). Instead it spawns
``worker.py`` with the ToolSandbox interpreter (``TOOLSANDBOX_PYTHON``) and
communicates over stdio using JSON-lines.

The one subtlety: the official engine calls the model *inside* its rollout loop.
So ``run_scenario`` is not a single request/response — it is an interactive loop
where the worker emits ``inference_request`` lines and this client services each
one by invoking our provider-agnostic :class:`~src.llm.base.LLMProvider`, then
writes back an ``inference_response``. When the rollout finishes the worker emits
a terminal ``result`` (or ``error``) line.

Everything ToolSandbox-specific (scenario normalization, roles, play, milestone
scoring, world-state serialization) lives in ``worker.py``; this file only owns
the transport and the model call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "benchmarks.toolsandbox.worker"

# Default protocol timeout (seconds) for a single line from the worker. Model
# calls are serviced by us, so the worker should never be silent for long except
# while it waits on our response (which does not count against readline here).
DEFAULT_TIMEOUT = float(os.environ.get("TOOLSANDBOX_WORKER_TIMEOUT", "600"))


class ToolSandboxWorkerError(RuntimeError):
    """Raised when the worker fails, exits, or violates the protocol."""


class ToolSandboxClient:
    """Spawns and talks to the ToolSandbox worker subprocess.

    Args:
        python_executable: Path to the ToolSandbox venv's Python.
        official_root: Path to the vendored repo (passed to the worker for an
            optional sys.path fallback when the package is a source checkout).
        inference_fn: Callable ``(messages, tools) -> {"text", "tool_calls"}``
            that runs one model completion in the main process. ``messages`` are
            OpenAI-format dicts and ``tools`` is an OpenAI tools list or ``None``.
    """

    def __init__(
        self,
        python_executable: str,
        official_root: str = "third_party/ToolSandbox-official",
        inference_fn: Optional[Any] = None,
    ) -> None:
        self.python_executable = python_executable
        self.official_root = official_root
        self.inference_fn = inference_fn

    # -- public API ---------------------------------------------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Return normalized scenario specs from the worker."""
        message = self._run_worker(["list-scenarios"], inference=False)
        if message.get("type") != "scenarios":
            raise ToolSandboxWorkerError(
                f"Expected 'scenarios' from worker, got: {message!r}"
            )
        return message.get("scenarios", [])

    def run_scenario(
        self,
        name: str,
        system_prompt: str = "",
        max_turns: int = 0,
        user_mode: str = "scripted",
    ) -> Dict[str, Any]:
        """Play out and score one scenario, servicing inference over stdio.

        Returns the worker's terminal ``result`` dict (with ``conversation``,
        ``world_state``, ``state_trace``, ``predicted``, ``evaluation``,
        ``error``). Requires ``inference_fn`` to have been provided.
        """
        if self.inference_fn is None:
            raise ToolSandboxWorkerError(
                "run_scenario requires an inference_fn to service model calls."
            )
        args = [
            "run-scenario",
            "--name",
            name,
            "--max-turns",
            str(max_turns),
            "--user-mode",
            user_mode,
            "--system-prompt",
            system_prompt,
        ]
        return self._run_worker(args, inference=True)

    # -- internals ----------------------------------------------------------

    def _worker_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        root = Path(self.official_root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        env["TOOLSANDBOX_OFFICIAL_ROOT"] = str(root)
        # Ensure the worker can import the benchmarks package by module path.
        existing = env.get("PYTHONPATH", "")
        parts = [str(PROJECT_ROOT)] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _spawn(self, args: List[str]) -> subprocess.Popen:
        cmd = [self.python_executable, "-m", WORKER_MODULE, *args]
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                env=self._worker_env(),
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise ToolSandboxWorkerError(
                f"Could not launch ToolSandbox worker with "
                f"'{self.python_executable}'. Set TOOLSANDBOX_PYTHON to the "
                f"ToolSandbox venv's python (e.g. ./ToolSandboxEnv/bin/python). "
                f"Original error: {exc}"
            ) from exc

    def _run_worker(self, args: List[str], inference: bool) -> Dict[str, Any]:
        """Drive the worker to completion, returning its terminal message.

        When ``inference`` is True, ``inference_request`` lines are serviced via
        ``inference_fn`` until a ``result``/``error`` line arrives. Otherwise the
        first well-formed message is returned.
        """
        process = self._spawn(args)
        assert process.stdout is not None and process.stdin is not None
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    # Stream closed without a terminal message.
                    return self._handle_eof(process)
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # Non-protocol noise on the channel; ignore defensively.
                    continue

                msg_type = message.get("type")
                if msg_type == "inference_request":
                    if not inference:
                        continue
                    self._service_inference(process, message)
                    continue
                if msg_type == "error":
                    raise ToolSandboxWorkerError(
                        message.get("message", "Unknown worker error")
                    )
                # Any other typed message is terminal for this call.
                return message
        finally:
            self._cleanup(process)

    def _service_inference(self, process: subprocess.Popen, request: Dict[str, Any]) -> None:
        """Run one model completion and write the response back to the worker."""
        assert process.stdin is not None
        request_id = request.get("id")
        try:
            result = self.inference_fn(
                request.get("messages", []), request.get("tools")
            )
            response = {
                "type": "inference_response",
                "id": request_id,
                "text": result.get("text", ""),
                "tool_calls": result.get("tool_calls", []),
            }
        except Exception as exc:  # noqa: BLE001 - forward failures to the worker
            response = {
                "type": "error",
                "id": request_id,
                "message": f"Main-process inference failed: {type(exc).__name__}: {exc}",
            }
        process.stdin.write(json.dumps(response, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _handle_eof(self, process: subprocess.Popen) -> Dict[str, Any]:
        stderr = self._drain_stderr(process)
        code = process.poll()
        raise ToolSandboxWorkerError(
            f"ToolSandbox worker exited (code={code}) without a result. "
            f"Stderr:\n{stderr}"
        )

    def _drain_stderr(self, process: subprocess.Popen) -> str:
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read() or ""
        except Exception:
            return ""

    def _cleanup(self, process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()


# ---------------------------------------------------------------------------
# Inference servicer: turn an OpenAI-format request into an LLMProvider call
# ---------------------------------------------------------------------------


def make_inference_fn(llm: Any, model_name: str = "research-helper") -> Any:
    """Build an ``inference_fn`` that routes worker requests through ``llm``.

    Converts the worker's OpenAI-format messages/tools into langchain messages,
    calls ``llm.invoke``, and serializes the response back into the minimal
    ``{"text", "tool_calls"}`` shape the worker expects.
    """

    def inference_fn(openai_messages: List[Dict[str, Any]], openai_tools: Optional[List[Any]]):
        messages = _openai_messages_to_langchain(openai_messages)
        tools = openai_tools if openai_tools else None
        response = llm.invoke(messages, tools=tools)
        return {
            "text": getattr(response, "text", "") or "",
            "tool_calls": [
                {
                    "id": getattr(call, "id", "") or f"call_{index}",
                    "name": getattr(call, "name", ""),
                    "arguments": getattr(call, "arguments", {}) or {},
                }
                for index, call in enumerate(getattr(response, "tool_calls", []) or [])
            ],
        }

    return inference_fn


def _openai_messages_to_langchain(openai_messages: List[Dict[str, Any]]):
    """Convert OpenAI-format messages (incl. system/tool) to langchain messages."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    out: List[Any] = []
    for message in openai_messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls = []
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {}) or {}
                arguments = function.get("arguments")
                try:
                    parsed = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else (arguments or {})
                    )
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                tool_calls.append(
                    {
                        "name": function.get("name", ""),
                        "args": parsed if isinstance(parsed, dict) else {},
                        "id": tool_call.get("id", ""),
                        "type": "tool_call",
                    }
                )
            out.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=content,
                    tool_call_id=message.get("tool_call_id", ""),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------


def default_toolsandbox_python() -> str:
    """Best-effort guess for the ToolSandbox interpreter path."""
    candidate = PROJECT_ROOT / "ToolSandboxEnv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return "python3"
