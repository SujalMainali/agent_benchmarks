"""Benchmark-only settings for ToolSandbox runs.

Kept separate from ``src.config.Settings`` so the agent core stays generic
while stateful-benchmark behavior stays easy to tune from environment
variables, mirroring ``benchmarks/locomo/config.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _get_int(name: str, default: str) -> int:
    return int(_get_str(name, default) or default)


def _get_float(name: str, default: str) -> float:
    return float(_get_str(name, default) or default)


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.split("#", 1)[0].strip()


def _get_list(name: str, default: str = "") -> List[str]:
    """Parse a comma-separated env var into a list of trimmed tool names."""
    raw = _get_str(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class ToolSandboxSettings:
    """Settings that control a ToolSandbox benchmark run."""

    official_root: str
    toolsandbox_python: str
    scenario: str | None
    scenario_category: str | None
    use_official_eval: bool
    allow_tools: List[str]
    max_turns: int
    run_mode: str
    max_scenarios: int | None
    output_dir: str
    verbose: bool
    agent_mode: str
    max_tool_steps: int
    fault_rate: float
    fault_seed: int


def load_toolsandbox_settings() -> ToolSandboxSettings:
    """Load ToolSandbox benchmark settings from environment variables.

    Environment variables:
    - TOOLSANDBOX_OFFICIAL_ROOT: Path to the vendored official ToolSandbox repo.
    - TOOLSANDBOX_PYTHON: Path to the isolated ToolSandbox venv interpreter used
      to spawn the worker process (e.g. ./ToolSandboxEnv/bin/python).
    - TOOLSANDBOX_SCENARIO: Optional scenario name/id to target in single mode.
    - TOOLSANDBOX_SCENARIO_CATEGORY: Optional category filter for batch runs.
    - TOOLSANDBOX_USE_OFFICIAL_EVAL: Use the official milestone scoring path.
    - TOOLSANDBOX_ALLOW_TOOLS: Optional comma-separated tool allow-list override.
      When empty, each scenario's own ``tool_allow_list`` is used.
    - TOOLSANDBOX_MAX_TURNS: Hard cap on conversation turns per scenario.
    - TOOLSANDBOX_RUN_MODE: `single` runs one scenario, `batch` runs many.
    - TOOLSANDBOX_MAX_SCENARIOS: Optional cap for batch runs; `0` means no cap.
    - TOOLSANDBOX_OUTPUT_DIR: Directory where benchmark reports are written.
    - TOOLSANDBOX_VERBOSE: If true, runs print progress to the terminal.
    - TOOLSANDBOX_AGENT_MODE: `runtime` (evaluate our ResearchHelperAgentRuntime)
      or `llm_proxy` (legacy: official agent loop driving only our LLM).
    - TOOLSANDBOX_MAX_TOOL_STEPS: Agent tool-loop budget per user turn in
      runtime mode.
    - TOOLSANDBOX_FAULT_RATE: Probability [0,1] that a tool call is answered
      with a synthetic transient error instead of executing (recovery testing).
    - TOOLSANDBOX_FAULT_SEED: RNG seed for reproducible fault injection.
    """

    max_scenarios = _get_int("TOOLSANDBOX_MAX_SCENARIOS", "0")

    return ToolSandboxSettings(
        official_root=_get_str("TOOLSANDBOX_OFFICIAL_ROOT", "third_party/ToolSandbox-official"),
        toolsandbox_python=_get_str("TOOLSANDBOX_PYTHON", "./ToolSandboxEnv/bin/python"),
        scenario=_get_str("TOOLSANDBOX_SCENARIO", "") or None,
        scenario_category=_get_str("TOOLSANDBOX_SCENARIO_CATEGORY", "") or None,
        use_official_eval=_get_bool("TOOLSANDBOX_USE_OFFICIAL_EVAL", "true"),
        allow_tools=_get_list("TOOLSANDBOX_ALLOW_TOOLS", ""),
        max_turns=_get_int("TOOLSANDBOX_MAX_TURNS", "20"),
        run_mode=_get_str("TOOLSANDBOX_RUN_MODE", "single").lower(),
        max_scenarios=max_scenarios if max_scenarios > 0 else None,
        output_dir=_get_str("TOOLSANDBOX_OUTPUT_DIR", "results/toolsandbox"),
        verbose=_get_bool("TOOLSANDBOX_VERBOSE", "true"),
        agent_mode=_get_str("TOOLSANDBOX_AGENT_MODE", "runtime").lower(),
        max_tool_steps=_get_int("TOOLSANDBOX_MAX_TOOL_STEPS", "8"),
        fault_rate=_get_float("TOOLSANDBOX_FAULT_RATE", "0.0"),
        fault_seed=_get_int("TOOLSANDBOX_FAULT_SEED", "13"),
    )
