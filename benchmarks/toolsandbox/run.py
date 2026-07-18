"""Main entry point for ToolSandbox benchmark execution.

Parallel to ``benchmarks/locomo/run.py``. Loads settings, resolves scenarios
from the official ToolSandbox repo, builds the configured LLM provider, runs the
scenario(s) through the official engine, scores milestones, and writes reports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.llm import build_provider

from benchmarks.common.models import Episode
from benchmarks.toolsandbox.config import ToolSandboxSettings, load_toolsandbox_settings
from benchmarks.toolsandbox.evaluator import ToolSandboxEvaluator
from benchmarks.toolsandbox.loader import ToolSandboxLoader
from benchmarks.toolsandbox.report import ToolSandboxReporter
from benchmarks.toolsandbox.runner import ToolSandboxRunner


def _resolve_user_mode(settings: ToolSandboxSettings) -> str:
    """Map config to the runner's user-simulator mode.

    ``scripted`` (default) replays the scenario and needs no external creds. An
    official OpenAI user simulator is used when TOOLSANDBOX_USER_MODE names one,
    which requires a real OpenAI key in TOOLSANDBOX_USER_API_KEY (the simulator
    talks to api.openai.com directly; the project-level OPENAI_API_KEY is the
    local agent model's placeholder).
    """
    mode = os.getenv("TOOLSANDBOX_USER_MODE", "scripted").strip().lower()
    mode = mode or "scripted"
    if mode != "scripted" and not settings.user_api_key:
        raise SystemExit(
            f"TOOLSANDBOX_USER_MODE={mode!r} needs TOOLSANDBOX_USER_API_KEY "
            "(a real OpenAI key for the official user simulator). Set it or "
            "use TOOLSANDBOX_USER_MODE=scripted."
        )
    return mode


def _filter_episodes(
    episodes: List[Episode],
    scenario: Optional[str],
    category: Optional[str],
) -> List[Episode]:
    """Filter loaded episodes by an exact scenario name and/or category."""
    filtered = episodes
    if scenario:
        filtered = [
            e
            for e in filtered
            if e.metadata.get("scenario_name") == scenario or e.episode_id == scenario
        ]
    if category:
        filtered = [
            e
            for e in filtered
            if category in [str(c) for c in e.metadata.get("categories", [])]
        ]
    return filtered


def _setup_runner(settings: ToolSandboxSettings) -> ToolSandboxRunner:
    """Build the LLM-backed runner from project + benchmark settings."""
    llm = build_provider(load_settings())
    return ToolSandboxRunner(
        llm=llm,
        python_executable=settings.toolsandbox_python,
        official_root=settings.official_root,
        user_mode=_resolve_user_mode(settings),
        max_turns=settings.max_turns,
        agent_mode=settings.agent_mode,
        max_tool_steps=settings.max_tool_steps,
        fault_rate=settings.fault_rate,
        fault_seed=settings.fault_seed,
        real_search_tools=settings.real_search_tools,
        rapid_api_key=settings.rapid_api_key,
        user_api_key=settings.user_api_key,
        user_base_url=settings.user_base_url,
    )


def _evaluate_and_report(
    run_results,
    output_dir: str,
) -> None:
    """Score runs and write per-scenario + batch reports."""
    evaluator = ToolSandboxEvaluator()
    eval_results = evaluator.evaluate_batch(run_results)

    reporter = ToolSandboxReporter(output_dir)
    reporter.write_full_report(run_results, eval_results)
    for run_result, eval_result in zip(run_results, eval_results):
        reporter.write_per_sample_report(run_result, eval_result)

    for run_result, eval_result in zip(run_results, eval_results):
        official = run_result.official_eval or {}
        matched = len(official.get("milestone_mapping", {}) or {})
        total_milestones = official.get("milestone_count", 0)
        print(f"\n{'=' * 50}")
        print(f"Scenario: {run_result.sample_id}")
        print(
            f"Milestones matched: {matched}/{total_milestones} "
            f"(similarity={official.get('milestone_similarity', 0.0):.3f})"
        )
        print(f"Minefield similarity: {official.get('minefield_similarity', 0.0):.3f}")
        print(f"Score: {eval_result.score:.3f}  Correct: {eval_result.is_correct}")
        print(f"Reason: {eval_result.correctness_reason}")
        if run_result.error:
            print(f"Error: {run_result.error}")
        print(f"Results saved to: {os.path.join(output_dir, run_result.sample_id)}")

    correct = sum(1 for r in eval_results if r.is_correct)
    total = len(eval_results)
    print(f"\n{'=' * 50}")
    print("ToolSandbox Summary")
    print(f"Total scenarios: {total}")
    print(f"Correct (fully solved): {correct}")
    print(f"Accuracy: {(correct / total) if total else 0:.2%}")
    print(f"Avg score: {(sum(r.score for r in eval_results) / total) if total else 0:.3f}")
    print(f"Results saved to: {output_dir}")


def run_single(settings: ToolSandboxSettings) -> None:
    """Run one scenario (TOOLSANDBOX_SCENARIO, or the first available)."""
    print("Loading scenarios...")
    loader = ToolSandboxLoader(
        python_executable=settings.toolsandbox_python,
        official_root=settings.official_root,
    )
    episodes = loader.load_named_scenarios()

    episodes = _filter_episodes(episodes, settings.scenario, settings.scenario_category)
    if not episodes:
        print(f"No scenario matched: {settings.scenario or settings.scenario_category}")
        return
    episodes = episodes[:1]
    print(f"Running scenario: {episodes[0].metadata.get('scenario_name')}")

    runner = _setup_runner(settings)
    run_results = runner.run_batch(episodes, verbose=settings.verbose)
    _evaluate_and_report(run_results, settings.output_dir)


def run_batch(settings: ToolSandboxSettings) -> None:
    """Run a batch of scenarios (optionally capped / category-filtered)."""
    print("Loading scenarios...")
    loader = ToolSandboxLoader(
        python_executable=settings.toolsandbox_python,
        official_root=settings.official_root,
    )
    episodes = loader.load_named_scenarios()

    episodes = _filter_episodes(episodes, None, settings.scenario_category)
    if settings.max_scenarios:
        episodes = episodes[: settings.max_scenarios]
    print(f"Running {len(episodes)} scenario(s)...")

    runner = _setup_runner(settings)
    run_results = runner.run_batch(episodes, verbose=settings.verbose)
    _evaluate_and_report(run_results, settings.output_dir)


def main() -> None:
    settings = load_toolsandbox_settings()

    print("ToolSandbox benchmark configuration")
    print(f"  run_mode: {settings.run_mode}")
    print(f"  official_root: {settings.official_root}")
    print(f"  toolsandbox_python: {settings.toolsandbox_python}")
    print(f"  output_dir: {settings.output_dir}")
    print(f"  use_official_eval: {settings.use_official_eval}")
    print(f"  scenario: {settings.scenario or 'None'}")
    print(f"  scenario_category: {settings.scenario_category or 'None'}")
    print(f"  allow_tools_override: {settings.allow_tools or 'scenario-defined'}")
    print(f"  max_turns: {settings.max_turns}")
    print(f"  max_scenarios: {settings.max_scenarios if settings.max_scenarios is not None else 'None'}")
    print(f"  user_mode: {_resolve_user_mode(settings)}")
    print(f"  agent_mode: {settings.agent_mode}")
    print(f"  max_tool_steps: {settings.max_tool_steps}")
    print(f"  fault_rate: {settings.fault_rate}")
    print(f"  fault_seed: {settings.fault_seed}")

    official_root = Path(settings.official_root)
    if not official_root.is_absolute():
        official_root = PROJECT_ROOT / official_root
    if not official_root.exists():
        print(f"Error: Official ToolSandbox repo not found: {official_root}")
        sys.exit(1)

    if settings.run_mode == "batch":
        run_batch(settings)
        return
    if settings.run_mode != "single":
        print(f"Error: Unknown TOOLSANDBOX_RUN_MODE '{settings.run_mode}'. Use 'single' or 'batch'.")
        sys.exit(1)
    run_single(settings)


if __name__ == "__main__":
    main()
