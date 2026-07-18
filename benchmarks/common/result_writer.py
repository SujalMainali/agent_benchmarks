"""Standardized, immutable experiment output writer (see ResultFormat.md).

Every benchmark run produces the SAME on-disk layout regardless of benchmark:

    results/
    ├── experiment_index.csv                 # master catalogue (append-only)
    └── <memory_architecture>/
        └── <YYYY-MM-DD_HH-MM-SS>/            # one immutable batch run
            ├── raw/
            │   ├── trajectories/sample_0001.jsonl
            │   ├── tool_calls/sample_0001.jsonl
            │   ├── environments/sample_0001.json
            │   └── logs/
            ├── reports/
            │   ├── cases.jsonl               # one processed record per sample
            │   ├── summary.json              # whole-batch summary
            │   └── aggregates_long.csv       # tidy long-format metrics
            └── figures/                      # reserved (populated later)

Design contract:
- ``raw/`` artifacts are written ACTIVELY as each sample finishes (call
  :meth:`write_raw` from the run loop), not deferred to the end of the batch.
- ``reports/cases.jsonl`` is appended per sample once evaluation is known.
- ``finalize`` writes the batch summary + aggregates and appends one row to the
  root ``experiment_index.csv``.
- A run directory is created once and never overwritten (immutability). If the
  timestamp collides with an existing run a numeric suffix is added.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import EvaluationResult, RunResult, Trajectory, TrajectoryEvent

_INDEX_COLUMNS = [
    "run_id",
    "timestamp",
    "agent_name",
    "memory_architecture",
    "benchmark",
    "benchmark_version",
    "dataset",
    "sample_count",
    "accuracy",
    "average_score",
    "result_directory",
]


def resolve_memory_architecture() -> str:
    """Top-level results bucket for the agent's memory design.

    Configurable via ``MEMORY_ARCHITECTURE`` so runs of different memory
    designs (adamem, flat_rag, episodic, ...) land in sibling directories.
    """
    return (os.getenv("MEMORY_ARCHITECTURE", "").split("#", 1)[0].strip() or "default")


def resolve_agent_name(default: str = "research_helper") -> str:
    """Human-facing agent label recorded in the experiment index.

    Precedence: explicit ``AGENT_NAME`` env var, then the ``AGENT_DRIVER``
    selection (registry key, or class name of a raw import path), then the
    default. Callers that hold a resolved driver should pass
    ``agent_name=driver.name`` instead of relying on this fallback.
    """
    explicit = os.getenv("AGENT_NAME", "").split("#", 1)[0].strip()
    if explicit:
        return explicit
    driver_key = os.getenv("AGENT_DRIVER", "").split("#", 1)[0].strip()
    if driver_key:
        return driver_key.rsplit(":", 1)[-1] if ":" in driver_key else driver_key
    return default


def _as_serializable(value: Any) -> Any:
    """Coerce dataclasses (and nested lists) into JSON-friendly structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_as_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_serializable(v) for k, v in value.items()}
    return value


def _events(run_result: RunResult) -> List[TrajectoryEvent]:
    traj = run_result.trajectory
    return traj.events if isinstance(traj, Trajectory) else list(traj or [])


class ExperimentRunWriter:
    """Writes one immutable batch run in the standardized ResultFormat layout."""

    def __init__(
        self,
        *,
        benchmark: str,
        dataset: str,
        results_root: str = "results",
        benchmark_version: str = "v1",
        memory_architecture: Optional[str] = None,
        agent_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        event_transform: Optional[Any] = None,
        include_raw_messages: bool = True,
    ) -> None:
        self.benchmark = benchmark
        self.dataset = dataset
        self.benchmark_version = benchmark_version
        self.memory_architecture = memory_architecture or resolve_memory_architecture()
        self.agent_name = agent_name or resolve_agent_name()
        self.run_metadata = run_metadata or {}
        # Optional hook: ``(run_result, events) -> list`` to trim/annotate the
        # per-event stream before it is written (e.g. LongMemEval collapses its
        # huge history-replay events). Defaults to writing events verbatim.
        self.event_transform = event_transform
        # When False, the trajectory header omits the (potentially huge) raw
        # message list — used by benchmarks that replay long histories.
        self.include_raw_messages = include_raw_messages

        self.results_root = results_root
        self.started_at = datetime.now()
        self.run_id = self._reserve_run_dir()

        # Directory skeleton (created once; never reused).
        self.raw_dir = os.path.join(self.run_dir, "raw")
        self.trajectories_dir = os.path.join(self.raw_dir, "trajectories")
        self.tool_calls_dir = os.path.join(self.raw_dir, "tool_calls")
        self.environments_dir = os.path.join(self.raw_dir, "environments")
        self.logs_dir = os.path.join(self.raw_dir, "logs")
        self.reports_dir = os.path.join(self.run_dir, "reports")
        self.figures_dir = os.path.join(self.run_dir, "figures")
        for directory in (
            self.trajectories_dir,
            self.tool_calls_dir,
            self.environments_dir,
            self.logs_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            os.makedirs(directory, exist_ok=True)

        self.cases_path = os.path.join(self.reports_dir, "cases.jsonl")
        # Truncate any stale (shouldn't exist for a fresh run dir) cases file.
        open(self.cases_path, "w", encoding="utf-8").close()

        self._raw_written: set[int] = set()
        self._case_count = 0
        # Streaming state: index -> number of events already appended live to
        # that sample's trajectory/tool-call files (see ``open_stream``).
        self._streamed: Dict[int, int] = {}
        self._stream_open: set[int] = set()

    # -- directory reservation ---------------------------------------------

    def _reserve_run_dir(self) -> str:
        base = os.path.join(self.results_root, self.memory_architecture)
        os.makedirs(base, exist_ok=True)
        stamp = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        candidate = os.path.join(base, stamp)
        suffix = 1
        # Immutability: never reuse/overwrite an existing timestamp directory.
        while os.path.exists(candidate):
            candidate = os.path.join(base, f"{stamp}_{suffix}")
            suffix += 1
        os.makedirs(candidate)
        self.run_dir = candidate
        return os.path.basename(candidate)

    @staticmethod
    def sample_stem(index: int) -> str:
        """1-based ``sample_0001`` stem for a sample's raw artifacts."""
        return f"sample_{index + 1:04d}"

    # -- raw artifacts (written actively during the run) -------------------

    def write_raw(self, run_result: RunResult, index: int) -> Dict[str, str]:
        """Write this sample's trajectory / tool-call / environment artifacts.

        Returns a dict of run-dir-relative paths (as stored inside cases.jsonl).
        Idempotent per index.
        """
        stem = self.sample_stem(index)
        if index in self._raw_written:
            # Already written (e.g. a streamed sample closed via ``close_stream``).
            # Preserve the existing artifacts rather than clobbering them.
            return {
                "trajectory": f"../raw/trajectories/{stem}.jsonl",
                "tool_calls": f"../raw/tool_calls/{stem}.jsonl",
                "environment": f"../raw/environments/{stem}.json",
            }
        events = _events(run_result)

        self._write_trajectory_jsonl(run_result, events, stem)
        self._write_tool_calls_jsonl(events, stem)
        self._write_environment_json(run_result, stem)

        self._raw_written.add(index)
        return {
            "trajectory": f"../raw/trajectories/{stem}.jsonl",
            "tool_calls": f"../raw/tool_calls/{stem}.jsonl",
            "environment": f"../raw/environments/{stem}.json",
        }

    def _write_trajectory_jsonl(
        self, run_result: RunResult, events: List[TrajectoryEvent], stem: str
    ) -> None:
        path = os.path.join(self.trajectories_dir, f"{stem}.jsonl")
        # Optional per-benchmark trimming of the event stream (returns dicts).
        if self.event_transform is not None:
            event_dicts = self.event_transform(run_result, events)
        else:
            event_dicts = [self._event_dict(event) for event in events]
        with open(path, "w", encoding="utf-8") as f:
            # First line is a header event carrying sample identity + raw msgs.
            f.write(json.dumps(self._trajectory_header(run_result), default=str) + "\n")
            for event_dict in event_dicts:
                f.write(json.dumps(event_dict, default=str) + "\n")

    def _trajectory_header(self, run_result: RunResult) -> Dict[str, Any]:
        """The ``meta`` first line of a sample's trajectory JSONL file."""
        return {
            "event_type": "meta",
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "benchmark": self.benchmark,
            "benchmark_mode": run_result.benchmark_mode,
            "question": run_result.question,
            "gold_answer": run_result.gold_answer,
            "predicted_answer": run_result.predicted_answer,
            "total_latency_ms": run_result.total_latency_ms,
            "error": run_result.error,
            "raw_messages": run_result.raw_messages if self.include_raw_messages else None,
        }

    @staticmethod
    def _tool_call_records(event: TrajectoryEvent) -> List[Dict[str, Any]]:
        """Flat per-tool-call records for the tool_calls JSONL file."""
        return [
            {
                "turn_number": event.turn_number,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result": tc.result,
                "latency_ms": tc.latency_ms,
                "token_count": tc.token_count,
                "success": event.exception is None,
                "exception": event.exception,
            }
            for tc in event.tool_calls
        ]

    # -- live intra-sample streaming ---------------------------------------

    def open_stream(self, run_result: RunResult, index: int) -> None:
        """Begin streaming a sample whose events arrive incrementally.

        Writes a provisional trajectory header immediately so the file exists
        and is inspectable, then :meth:`stream_events` appends events as the
        agent produces them (e.g. LongMemEval's per-session replay). Use this
        instead of :meth:`write_raw` for samples that take a long time to
        finish — the on-disk artifact grows live rather than appearing only at
        the end.
        """
        stem = self.sample_stem(index)
        traj_path = os.path.join(self.trajectories_dir, f"{stem}.jsonl")
        tool_path = os.path.join(self.tool_calls_dir, f"{stem}.jsonl")
        with open(traj_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._trajectory_header(run_result), default=str) + "\n")
        # Create/truncate the tool-calls file so it exists from the start.
        open(tool_path, "w", encoding="utf-8").close()
        self._streamed[index] = 0
        self._stream_open.add(index)

    def stream_events(self, index: int, events: List[TrajectoryEvent]) -> None:
        """Append newly-produced trajectory events for an open stream.

        Idempotent w.r.t. already-streamed events: pass the sample's full
        event list each time and only the unwritten tail is flushed. Bytes are
        fsync-flushed so a tail-follow sees them promptly.
        """
        if index not in self._stream_open:
            return
        already = self._streamed.get(index, 0)
        if len(events) <= already:
            return
        new_events = events[already:]
        stem = self.sample_stem(index)
        traj_path = os.path.join(self.trajectories_dir, f"{stem}.jsonl")
        tool_path = os.path.join(self.tool_calls_dir, f"{stem}.jsonl")
        with open(traj_path, "a", encoding="utf-8") as tf, open(
            tool_path, "a", encoding="utf-8"
        ) as cf:
            for event in new_events:
                tf.write(json.dumps(self._event_dict(event), default=str) + "\n")
                for record in self._tool_call_records(event):
                    cf.write(json.dumps(record, default=str) + "\n")
            tf.flush()
            cf.flush()
        self._streamed[index] = len(events)

    def close_stream(self, run_result: RunResult, index: int) -> Dict[str, str]:
        """Finish a streamed sample: flush any remaining events + environment.

        After this the sample is treated as fully written (``write_raw`` is a
        no-op for it) and :meth:`append_case` can reference its artifacts.

        The final on-disk artifact is reconciled to match a one-shot
        :meth:`write_raw`: if an ``event_transform`` is configured (e.g.
        LongMemEval collapses replay events — a decision that needs the whole
        event list, so it can't be applied live), the trajectory file is
        rewritten canonically now. Otherwise the streamed full events are
        already canonical and only the header is refreshed with final values.
        """
        stem = self.sample_stem(index)
        events = _events(run_result)
        if index in self._stream_open:
            self.stream_events(index, events)
            self._stream_open.discard(index)
            if self.event_transform is not None:
                # Live file holds full events; replace it with the canonical
                # (transformed) form so it matches write_raw exactly.
                self._write_trajectory_jsonl(run_result, events, stem)
                self._write_tool_calls_jsonl(events, stem)
            else:
                # Rewrite the header line with final values now that the sample
                # is complete (predicted answer / latency / error were unknown
                # at open time).
                self._rewrite_stream_header(run_result, stem)
        else:
            # Never opened as a stream — fall back to a one-shot raw write.
            self._write_trajectory_jsonl(run_result, events, stem)
            self._write_tool_calls_jsonl(events, stem)
        self._write_environment_json(run_result, stem)
        self._raw_written.add(index)
        return {
            "trajectory": f"../raw/trajectories/{stem}.jsonl",
            "tool_calls": f"../raw/tool_calls/{stem}.jsonl",
            "environment": f"../raw/environments/{stem}.json",
        }

    def _rewrite_stream_header(self, run_result: RunResult, stem: str) -> None:
        """Replace the provisional header (first line) with final values."""
        path = os.path.join(self.trajectories_dir, f"{stem}.jsonl")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return
        if not lines:
            return
        lines[0] = json.dumps(self._trajectory_header(run_result), default=str) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def _event_dict(event: TrajectoryEvent) -> Dict[str, Any]:
        return {
            "event_type": event.event_type,
            "turn_number": event.turn_number,
            "actor": event.actor,
            "recipient": event.recipient,
            "user_input": event.user_input,
            "agent_message": event.agent_message,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "latency_ms": tc.latency_ms,
                    "token_count": tc.token_count,
                }
                for tc in event.tool_calls
            ],
            "exception": event.exception,
            "milestone_checks": event.milestone_checks,
            "environment_state_before": event.environment_state_before,
            "environment_state_after": event.environment_state_after,
            "latency_ms": event.latency_ms,
            "token_count": event.token_count,
            "metadata": event.metadata,
        }

    def _write_tool_calls_jsonl(self, events: List[TrajectoryEvent], stem: str) -> None:
        path = os.path.join(self.tool_calls_dir, f"{stem}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for event in events:
                for record in self._tool_call_records(event):
                    f.write(json.dumps(record, default=str) + "\n")

    def _write_environment_json(self, run_result: RunResult, stem: str) -> None:
        state = run_result.final_state
        data: Dict[str, Any] = {
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "benchmark": self.benchmark,
            "done": bool(getattr(state, "done", False)) if state else False,
            "turn_index": int(getattr(state, "turn_index", 0)) if state else 0,
            "allowed_tools": list(getattr(state, "allowed_tools", []) or []) if state else [],
            "world_state": getattr(state, "world_state", {}) if state else {},
            "milestones": [_as_serializable(m) for m in getattr(state, "milestones", [])] if state else [],
            "minefields": [_as_serializable(m) for m in getattr(state, "minefields", [])] if state else [],
            # Per-snapshot world-state trace, when the runner recorded one
            # (e.g. ToolSandbox's official snapshot trace).
            "state_trace": (run_result.metadata or {}).get("state_trace", []),
            "metadata": getattr(state, "metadata", {}) if state else {},
        }
        path = os.path.join(self.environments_dir, f"{stem}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # -- processed per-sample record ---------------------------------------

    def append_case(
        self,
        run_result: RunResult,
        eval_result: EvaluationResult,
        index: int,
        *,
        task_family: str = "",
        task_type: str = "",
        expected_tool_behavior: Any = None,
        expected_state: Any = None,
        benchmark_metrics: Optional[Dict[str, Any]] = None,
        benchmark_specific: Optional[Dict[str, Any]] = None,
        routing: Any = None,
        memory_actions: Any = None,
    ) -> None:
        """Append one processed sample record to ``reports/cases.jsonl``.

        Ensures the sample's raw artifacts exist (writes them if the run loop
        did not already), then references them from the case record.
        """
        if index not in self._raw_written:
            raw_refs = self.write_raw(run_result, index)
        else:
            stem = self.sample_stem(index)
            raw_refs = {
                "trajectory": f"../raw/trajectories/{stem}.jsonl",
                "tool_calls": f"../raw/tool_calls/{stem}.jsonl",
                "environment": f"../raw/environments/{stem}.json",
            }

        case = {
            # Identification
            "sample_id": run_result.sample_id,
            "episode_id": run_result.episode_id or run_result.sample_id,
            "benchmark": self.benchmark,
            "benchmark_version": self.benchmark_version,
            "task_family": task_family,
            "task_type": task_type or run_result.benchmark_mode,
            # Inputs
            "inputs": {
                "question": run_result.question,
                "context_metadata": self._context_metadata(run_result),
                "conversation_stats": self._conversation_stats(run_result),
            },
            # Prediction
            "prediction": {
                "predicted_answer": run_result.predicted_answer,
                "tool_usage": self._tool_usage_summary(run_result),
                "routing": routing,
                "memory_actions": memory_actions,
            },
            # Reference
            "reference": {
                "gold_answer": run_result.gold_answer,
                "expected_tool_behavior": expected_tool_behavior,
                "expected_state": expected_state,
            },
            # Evaluation
            "evaluation": {
                "is_correct": eval_result.is_correct,
                "score": eval_result.score,
                "benchmark_metrics": benchmark_metrics or run_result.metrics,
                "evidence_hits": eval_result.evidence_hits,
                "failure_mode": eval_result.failure_mode,
                "correctness_reason": eval_result.correctness_reason,
            },
            # Diagnostics + benchmark-specific extension points
            "diagnostics": eval_result.diagnostics,
            "benchmark_specific": benchmark_specific or {},
            "error": run_result.error,
            # Raw artifact references
            "raw": raw_refs,
        }

        with open(self.cases_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(case, default=str) + "\n")
        self._case_count += 1

    def _context_metadata(self, run_result: RunResult) -> Dict[str, Any]:
        meta = run_result.metadata or {}
        # Keep the context metadata compact: drop bulky replay payloads.
        skip = {"haystack_sessions", "state_trace", "world_state", "runtime_trajectory"}
        return {k: v for k, v in meta.items() if k not in skip}

    @staticmethod
    def _conversation_stats(run_result: RunResult) -> Dict[str, Any]:
        events = _events(run_result)
        return {
            "context_turn_count": run_result.context_turn_count,
            "trajectory_events": len(events),
            "raw_message_count": len(run_result.raw_messages or []),
            "total_latency_ms": run_result.total_latency_ms,
        }

    @staticmethod
    def _tool_usage_summary(run_result: RunResult) -> Dict[str, Any]:
        events = _events(run_result)
        by_tool: Dict[str, int] = {}
        total = 0
        for event in events:
            for tc in event.tool_calls:
                total += 1
                name = tc.tool_name or "unknown"
                by_tool[name] = by_tool.get(name, 0) + 1
        return {"total_tool_calls": total, "by_tool": by_tool}

    # -- batch summary + master index --------------------------------------

    def finalize(
        self,
        *,
        sample_count: int,
        accuracy: float,
        average_score: float,
        correct: int,
        errors: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write ``summary.json`` + ``aggregates_long.csv`` and index the run.

        Returns the run directory path.
        """
        summary = {
            "run_id": self.run_id,
            "timestamp": self.started_at.isoformat(),
            "benchmark": self.benchmark,
            "benchmark_version": self.benchmark_version,
            "memory_architecture": self.memory_architecture,
            "agent_name": self.agent_name,
            "dataset": self.dataset,
            "sample_count": sample_count,
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "average_score": average_score,
            "metrics": metrics or {},
            "diagnostics": diagnostics or {},
            "run_metadata": self.run_metadata,
        }
        with open(os.path.join(self.reports_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self._write_aggregates_long(summary)
        self._append_index_row(sample_count, accuracy, average_score)
        return self.run_dir

    def _write_aggregates_long(self, summary: Dict[str, Any]) -> None:
        """Flatten summary numerics into a tidy long table for plotting."""
        rows: List[Dict[str, Any]] = []

        def add(group_type: str, group_key: str, metric: str, value: Any) -> None:
            rows.append(
                {
                    "group_type": group_type,
                    "group_key": group_key,
                    "metric": metric,
                    "value": value,
                }
            )

        for key in ("sample_count", "correct", "errors", "accuracy", "average_score"):
            add("overall", "", key, summary[key])

        self._flatten_into(summary.get("metrics", {}), add, prefix="")
        return self._dump_long_csv(rows)

    @staticmethod
    def _flatten_into(obj: Any, add, prefix: str) -> None:
        """Recurse dict-of-numerics; dict-of-dicts becomes grouped rows."""
        if not isinstance(obj, dict):
            return

        def is_number(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        for key, value in obj.items():
            name = f"{prefix}{key}"
            if is_number(value):
                add("metric", "", name, value)
            elif isinstance(value, dict):
                # A dict of scalars becomes group rows keyed by the parent name;
                # nested dicts recurse one more level under that group.
                for sub_key, sub_val in value.items():
                    if is_number(sub_val):
                        add(name, str(sub_key), sub_key, sub_val)
                    elif isinstance(sub_val, dict):
                        for leaf_k, leaf_v in sub_val.items():
                            if is_number(leaf_v):
                                add(name, str(sub_key), str(leaf_k), leaf_v)

    def _dump_long_csv(self, rows: List[Dict[str, Any]]) -> None:
        path = os.path.join(self.reports_dir, "aggregates_long.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["group_type", "group_key", "metric", "value"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _append_index_row(
        self, sample_count: int, accuracy: float, average_score: float
    ) -> None:
        index_path = os.path.join(self.results_root, "experiment_index.csv")
        exists = os.path.exists(index_path)
        result_directory = os.path.relpath(self.run_dir, self.results_root)
        row = {
            "run_id": self.run_id,
            "timestamp": self.started_at.isoformat(),
            "agent_name": self.agent_name,
            "memory_architecture": self.memory_architecture,
            "benchmark": self.benchmark,
            "benchmark_version": self.benchmark_version,
            "dataset": self.dataset,
            "sample_count": sample_count,
            "accuracy": round(accuracy, 6),
            "average_score": round(average_score, 6),
            "result_directory": result_directory,
        }
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
