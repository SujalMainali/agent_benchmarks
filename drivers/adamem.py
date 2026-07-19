"""AgentDriver for the external AdaMem memory-strategy agents.

Adapts the three agents under ``Agents/AdaMemAgents/`` (flat_rag, episodic,
mem1) to the harness's ``AgentRuntime`` contract **without importing or
modifying any code inside ``Agents/``** beyond a lazy ``sys.path`` bootstrap.

The AdaMem agents speak an ingest/ask interface (write-only history load +
read-and-generate QA) rather than our reset/act loop; ``AdaMemRuntime`` bridges
the two. They have no tool support, so only LoCoMo and LongMemEval are
meaningful — any other benchmark (or a spec carrying tools) fails fast.

See AdaMemDrivers.md for the design and Agents/CLAUDE.md for the agents.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage

from benchmarks.common.driver import AgentDriver, RuntimeSpec
from benchmarks.common.interfaces import AgentRuntime
from benchmarks.common.models import (
    Action,
    EnvironmentState,
    Episode,
    Observation,
    Trajectory,
    TrajectoryEvent,
)

#: Benchmarks these memory-QA agents can actually run (no tool support).
_SUPPORTED_BENCHMARKS = {"locomo", "longmemeval"}

#: A replayed LongMemEval turn begins with ``ROLE: `` (USER:/ASSISTANT:/…),
#: matching the adapter's stable ``build_session_observation_texts`` format.
#: Continuation lines (wrapped content) don't match and stay with their turn.
_TURN_LINE = re.compile(r"^([A-Z][A-Z_]*): (.*)$")


class UnsupportedBenchmarkError(RuntimeError):
    """Raised when an AdaMem driver is pointed at a benchmark it can't serve."""


def _bootstrap_adamem_path() -> None:
    """Put ``<repo>/Agents/AdaMemAgents`` on ``sys.path`` once, lazily.

    The AdaMem package uses top-level imports (``from agents...``,
    ``from shared...``); it must be importable as a source root. Done here
    rather than at module import so merely importing this driver stays cheap
    and the heavy deps (faiss, sentence-transformers) load only on
    ``create_runtime``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    adamem_root = repo_root / "Agents" / "AdaMemAgents"
    path_str = str(adamem_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


class AdaMemRuntime(AgentRuntime):
    """Adapts a ``BaseMemoryAgent`` (ingest/ask) to the AgentRuntime contract.

    - ``reset`` wipes memory and ingests any pre-seeded history (LoCoMo path).
    - ``act`` routes on ``observation.metadata["phase"]``:
        * ``"history_replay"`` (LongMemEval session) → ingest, no generation,
          reply ``"Noted."`` (the runner discards it).
        * otherwise (LoCoMo question / LongMemEval final question) → flush any
          buffered ingests, then ``ask()`` and return the answer.
    """

    def __init__(self, agent: Any, strategy: str, mem1_session_ingest: bool) -> None:
        self.agent = agent
        self.strategy = strategy
        self._mem1_session_ingest = mem1_session_ingest
        self._episode: Episode | None = None
        self._trajectory = Trajectory()
        self._raw_messages: List[Dict[str, Any]] = []
        self._last_action: Action | None = None
        self._act_count = 0
        self._ingest_turn = 0
        self._ingested_count = 0
        self._ask_count = 0

    # ------------------------------------------------------------- lifecycle
    def reset(self, episode: Episode, initial_state: EnvironmentState) -> None:
        self._episode = episode
        self._trajectory = Trajectory()
        self._raw_messages = []
        self._last_action = None
        self._act_count = 0
        self._ingest_turn = 0
        self._ingested_count = 0
        self._ask_count = 0

        self.agent.reset_session()

        # LoCoMo seeds the whole conversation through initial_state.messages;
        # LongMemEval seeds nothing here (history flows via replay act()s).
        carried_date = ""
        for message in initial_state.messages:
            mtype = getattr(message, "type", "")
            kwargs = getattr(message, "additional_kwargs", {}) or {}
            if mtype == "system":
                carried_date = self._message_date(kwargs) or carried_date
                continue  # session headers aren't memories
            text = self._message_text(message)
            if not text:
                continue
            role = "assistant" if mtype == "ai" else "user"
            ts = self._message_date(kwargs) or carried_date
            self._ingest_turn += 1
            self.agent.ingest(
                text=self._prefix(text, ts),
                role=role,
                turn=self._ingest_turn,
            )
            self._ingested_count += 1

    # ------------------------------------------------------------------- act
    def act(self, observation: Observation) -> Action:
        self._act_count += 1
        turn = self._act_count
        phase = observation.metadata.get("phase")

        if phase == "history_replay":
            return self._act_replay(observation, turn)
        return self._act_question(observation, turn)

    def _act_replay(self, observation: Observation, turn: int) -> Action:
        session_date = str(observation.metadata.get("session_date", "") or "")
        session_id = observation.metadata.get("session_id")
        turns = self._split_session_turns(observation.text)

        if self.strategy == "mem1" and self._mem1_session_ingest:
            # Fold the whole session into <IS> with ONE LLM call instead of one
            # per turn — mem1 rewrites state on every ingest, so per-turn on a
            # 500-session batch is punishingly slow (AdaMemDrivers.md §6.2).
            body = "\n".join(f"{role.upper()}: {text}" for role, text in turns) or observation.text
            self._ingest_turn += 1
            self.agent.ingest(
                text=self._prefix(body, session_date),
                role="session",
                turn=self._ingest_turn,
            )
            ingested = 1
        else:
            for role, text in turns:
                self._ingest_turn += 1
                self.agent.ingest(
                    text=self._prefix(text, session_date),
                    role="assistant" if role.lower() in _ASSISTANT_ROLES else "user",
                    turn=self._ingest_turn,
                )
            ingested = len(turns)

        self._ingested_count += ingested
        self._trajectory.append(
            TrajectoryEvent(
                event_type="history_replay",
                turn_number=turn,
                user_input=observation.text,
                actor="user",
                recipient="agent",
                metadata={
                    "phase": "history_replay",
                    "session_id": session_id,
                    "session_date": session_date,
                    "ingested_turns": ingested,
                },
            )
        )
        self._raw_messages.append(
            {"role": "user", "content": observation.text, "metadata": {"phase": "history_replay"}}
        )
        self._raw_messages.append(
            {"role": "assistant", "content": "Noted.", "metadata": {"ingested_turns": ingested}}
        )
        action = Action(action_type="final_answer", text="Noted.")
        self._last_action = action
        return action

    def _act_question(self, observation: Observation, turn: int) -> Action:
        # Flush buffered ingests so retrieval sees the full history (episodic).
        finalize = getattr(self.agent, "finalize_ingest", None)
        if callable(finalize):
            finalize()

        log = self.agent.ask(observation.text)
        answer = str(log.get("response", ""))
        self._ask_count += 1

        diagnostics = {k: v for k, v in log.items() if k != "response"}
        self._trajectory.append(
            TrajectoryEvent(
                event_type="user",
                turn_number=turn,
                user_input=observation.text,
                actor="user",
                recipient="agent",
                metadata={"phase": observation.metadata.get("phase", "question")},
            )
        )
        self._trajectory.append(
            TrajectoryEvent(
                event_type="final",
                turn_number=turn,
                agent_message=answer,
                actor="agent",
                recipient="user",
                latency_ms=float(log.get("latency_s", 0.0) or 0.0) * 1000.0,
                metadata=diagnostics,
            )
        )
        self._trajectory.append(
            TrajectoryEvent(
                event_type="done",
                turn_number=turn,
                actor="agent",
                recipient="user",
                metadata={"answer": answer},
            )
        )
        self._raw_messages.append(
            {"role": "user", "content": observation.text, "metadata": observation.metadata}
        )
        self._raw_messages.append(
            {"role": "assistant", "content": answer, "metadata": diagnostics}
        )
        action = Action(
            action_type="final_answer",
            text=answer,
            metadata={
                "episode_id": observation.episode_id,
                "benchmark_mode": observation.metadata.get("benchmark_mode", "plain_qa"),
            },
        )
        self._last_action = action
        return action

    # ------------------------------------------------------------- accessors
    def get_trajectory(self) -> Trajectory:
        return self._trajectory

    def get_raw_messages(self) -> List[Dict[str, Any]]:
        return list(self._raw_messages)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "ingested_count": self._ingested_count,
            "ask_count": self._ask_count,
            "event_count": len(self._trajectory.events),
            "agent_logs": self._safe_export_logs(),
        }

    def _safe_export_logs(self) -> List[Dict[str, Any]]:
        export = getattr(self.agent, "export_logs", None)
        if not callable(export):
            return []
        try:
            return list(export())
        except Exception:
            return []

    # --------------------------------------------------------------- helpers
    def _split_session_turns(self, text: str) -> List[Tuple[str, str]]:
        """Parse a packed LongMemEval session back into ``(role, text)`` turns.

        Splits on ``^ROLE: `` lines; continuation lines join the current turn.
        The ``[Past chat session — …]`` header line is skipped. If nothing
        matches (unexpected format), returns the whole body as a single
        user turn so no history is silently dropped.
        """
        turns: List[Tuple[str, str]] = []
        role: Optional[str] = None
        buf: List[str] = []
        stray: List[str] = []

        def _flush() -> None:
            if role is not None:
                turns.append((role, "\n".join(buf).strip()))

        for line in text.splitlines():
            if line.startswith("[Past chat session"):
                continue
            m = _TURN_LINE.match(line)
            if m:
                _flush()
                role, buf = m.group(1), [m.group(2)]
            elif role is not None:
                buf.append(line)
            elif line.strip():
                stray.append(line)
        _flush()

        if not turns:
            body = "\n".join(stray).strip() or text.strip()
            return [("user", body)] if body else []
        return turns

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        return content.strip() if isinstance(content, str) else str(content).strip()

    @staticmethod
    def _message_date(kwargs: Dict[str, Any]) -> str:
        for key in ("session_timestamp", "timestamp", "date_time", "date"):
            value = kwargs.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _prefix(text: str, date_str: str) -> str:
        """Prefix a memory with its date so the agents' date-reasoning QA
        prompt can resolve relative time expressions (``[date] text``)."""
        date_str = (date_str or "").strip()
        return f"[{date_str}] {text}" if date_str else text


#: Role tokens (from the packed replay format) that map to the assistant side.
_ASSISTANT_ROLES = {"assistant", "ai", "bot", "system"}


class AdaMemDriver(AgentDriver):
    """Builds an AdaMem memory agent (one of three strategies) as a runtime.

    Instantiate via the three no-arg subclasses (registered in
    ``benchmarks/common/driver.py``); each pins one strategy.
    """

    #: Overridden by subclasses: "flat_rag" | "episodic" | "mem1".
    strategy: str = ""

    def __init__(self) -> None:
        if not self.strategy:
            raise ValueError("AdaMemDriver must be subclassed with a concrete `strategy`.")
        self.name = f"adamem_{self.strategy}"
        self.prompt_mode = (_env("ADAMEM_PROMPT_MODE", "native") or "native").lower()
        self._mem1_session_ingest = (
            _env("ADAMEM_MEM1_INGEST_GRANULARITY", "session") or "session"
        ).lower() != "turn"
        # Results bucket under results/<strategy>/... unless overridden.
        os.environ.setdefault("MEMORY_ARCHITECTURE", self.strategy)

    def create_runtime(self, spec: RuntimeSpec) -> AgentRuntime:
        if spec.benchmark not in _SUPPORTED_BENCHMARKS or spec.tools is not None:
            raise UnsupportedBenchmarkError(
                f"AdaMem agent '{self.name}' has no tool support; it can only run "
                f"{sorted(_SUPPORTED_BENCHMARKS)}. Got benchmark='{spec.benchmark}', "
                f"tools={'provided' if spec.tools is not None else 'none'}."
            )

        _bootstrap_adamem_path()
        from agents import get_agent_entry  # AdaMem registry (on sys.path now)

        entry = get_agent_entry(self.strategy)
        config = self._build_config(entry["config_cls"], spec)
        agent = entry["agent_cls"](config)
        return AdaMemRuntime(
            agent,
            strategy=self.strategy,
            mem1_session_ingest=self._mem1_session_ingest,
        )

    def _build_config(self, config_cls: Any, spec: RuntimeSpec) -> Any:
        config = config_cls()

        model = _env("ADAMEM_OLLAMA_MODEL")
        if model:
            config.ollama_model = model
        host = _env("ADAMEM_OLLAMA_HOST")
        if host:
            config.ollama_host = host

        max_tokens = _env("ADAMEM_MAX_TOKENS")
        if max_tokens and hasattr(config, "max_tokens"):
            config.max_tokens = int(max_tokens)
        temperature = _env("ADAMEM_TEMPERATURE")
        if temperature and hasattr(config, "temperature"):
            config.temperature = float(temperature)
        timeout = _env("ADAMEM_TIMEOUT")
        if timeout and hasattr(config, "request_timeout"):
            config.request_timeout = int(timeout)

        if self.strategy == "episodic" and hasattr(config, "db_path"):
            db_path = _env("ADAMEM_EPISODIC_DB")
            if db_path:
                config.db_path = db_path

        # Prompt policy (AdaMemDrivers.md §4). `native` keeps each agent's own
        # memory-format-aware QA prompt; `benchmark` swaps in the harness prompt.
        if self.prompt_mode == "benchmark" and spec.system_prompt and hasattr(config, "qa_system_prompt"):
            config.qa_system_prompt = spec.system_prompt

        return config

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "agent_name": self.name,
            "strategy": self.strategy,
            "prompt_mode": self.prompt_mode,
            "ollama_model": _env("ADAMEM_OLLAMA_MODEL", "llama3.2"),
            "ollama_host": _env("ADAMEM_OLLAMA_HOST", "http://localhost:11434"),
            "system_prompt_override_honored": self.prompt_mode == "benchmark",
        }
        if self.strategy == "mem1":
            info["mem1_ingest_granularity"] = "session" if self._mem1_session_ingest else "turn"
        return info


class AdaMemFlatRAGDriver(AdaMemDriver):
    strategy = "flat_rag"


class AdaMemEpisodicDriver(AdaMemDriver):
    strategy = "episodic"


class AdaMemMem1Driver(AdaMemDriver):
    strategy = "mem1"
