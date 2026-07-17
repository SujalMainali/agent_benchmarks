"""Metrics computation for ToolSandbox runs."""

from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.common.models import RunResult, TrajectoryEvent


class ToolSandboxMetrics:
    """Computes state- and tool-oriented metrics for ToolSandbox runs."""

    @staticmethod
    def compute_metrics(run_result: RunResult) -> Dict[str, Any]:
        """Compute per-run metrics emphasizing tool use and state progress."""
        trajectory: List[TrajectoryEvent] = list(run_result.trajectory)
        official = run_result.official_eval or {}

        tool_calls = sum(len(event.tool_calls) for event in trajectory)
        invalid_tool_calls = sum(
            1 for event in trajectory if event.exception
        )
        valid_tool_calls = max(tool_calls - invalid_tool_calls, 0)

        state_change_count = int(
            run_result.metadata.get("state_change_count", 0)
        )

        fault_metrics = ToolSandboxMetrics._fault_metrics(run_result, trajectory)

        return {
            "total_tool_calls": tool_calls,
            "valid_tool_calls": valid_tool_calls,
            "invalid_tool_calls": invalid_tool_calls,
            "milestone_similarity": float(official.get("milestone_similarity", 0.0)),
            "minefield_similarity": float(official.get("minefield_similarity", 0.0)),
            "milestone_count": int(official.get("milestone_count", 0)),
            "milestones_matched": len(official.get("milestone_mapping", {}) or {}),
            "minefield_violations": len(official.get("minefield_mapping", {}) or {}),
            "turn_count": int(official.get("turn_count", len(trajectory))),
            "trajectory_length": len(trajectory),
            "state_change_count": state_change_count,
            "total_latency_ms": run_result.total_latency_ms,
            **fault_metrics,
        }

    @staticmethod
    def _fault_metrics(
        run_result: RunResult, trajectory: List[TrajectoryEvent]
    ) -> Dict[str, Any]:
        """Fault-injection recovery metrics.

        A fault is "recovered" when the agent issues a later successful
        (executed, non-erroring) tool call for the same tool name. Tool
        execution is not recorded in the worker conversation for injected
        faults (they never touch the sandbox), so we count successful executed
        calls per tool from the trajectory and match them against the logged
        faults.
        """
        faults: List[Dict[str, Any]] = list(
            run_result.metadata.get("fault_injections", []) or []
        )
        injected = len(faults)
        if injected == 0:
            return {
                "injected_fault_count": 0,
                "post_fault_retry_count": 0,
                "fault_recovery_rate": 0.0,
            }

        # Count executed, non-erroring tool calls per tool name (retry evidence).
        successful_by_tool: Dict[str, int] = {}
        for event in trajectory:
            if event.event_type != "tool_result" or event.exception:
                continue
            for call in event.tool_calls:
                name = call.tool_name
                if name:
                    successful_by_tool[name] = successful_by_tool.get(name, 0) + 1

        remaining = dict(successful_by_tool)
        recovered = 0
        for fault in faults:
            name = str(fault.get("tool", ""))
            if remaining.get(name, 0) > 0:
                remaining[name] -= 1
                recovered += 1

        return {
            "injected_fault_count": injected,
            "post_fault_retry_count": recovered,
            "fault_recovery_rate": recovered / injected,
        }

    @staticmethod
    def compute_batch_metrics(run_results: List[RunResult]) -> Dict[str, float]:
        """Aggregate metrics across a batch of runs."""
        if not run_results:
            return {}

        total = len(run_results)
        per_run = [ToolSandboxMetrics.compute_metrics(r) for r in run_results]

        def _avg(key: str) -> float:
            return sum(m[key] for m in per_run) / total

        return {
            "total_runs": total,
            "avg_total_tool_calls": _avg("total_tool_calls"),
            "avg_valid_tool_calls": _avg("valid_tool_calls"),
            "avg_invalid_tool_calls": _avg("invalid_tool_calls"),
            "avg_milestone_similarity": _avg("milestone_similarity"),
            "avg_minefield_similarity": _avg("minefield_similarity"),
            "avg_turn_count": _avg("turn_count"),
            "avg_state_change_count": _avg("state_change_count"),
            "avg_latency_ms": _avg("total_latency_ms"),
            "avg_injected_fault_count": _avg("injected_fault_count"),
            "avg_post_fault_retry_count": _avg("post_fault_retry_count"),
            "avg_fault_recovery_rate": _avg("fault_recovery_rate"),
            "minefield_violation_rate": sum(
                1 for m in per_run if m["minefield_violations"] > 0
            )
            / total,
        }
