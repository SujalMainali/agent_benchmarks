"""LongMemEval runner — drives the session-replay loop over the frozen runtime.

The replay loop lives HERE (not in the adapter). For each episode we reset the
runtime with an EMPTY message list, replay each history session as a separate
``act()`` call (discarding the "Noted." reply), then ask the final question and
capture the answer. Memory accumulates across the replay ``act()`` calls and is
cleared only by the next ``reset()``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from benchmarks.common.logger import BenchmarkLogger
from benchmarks.common.models import (
    Episode,
    EnvironmentState,
    Observation,
    RunResult,
)
from src.runtime import ResearchHelperAgentRuntime

from .adapter import LongMemEvalAdapter

if TYPE_CHECKING:
    from src.agent import ResearchHelperAgent


class LongMemEvalRunner:
    """Runs LongMemEval episodes through the ResearchHelperAgent runtime."""

    def __init__(
        self,
        agent_or_runtime: "ResearchHelperAgent | ResearchHelperAgentRuntime",
        adapter: LongMemEvalAdapter | None = None,
        max_sessions: int = 0,
        verbose: bool = True,
    ) -> None:
        if isinstance(agent_or_runtime, ResearchHelperAgentRuntime):
            self.runtime = agent_or_runtime
        else:
            self.runtime = ResearchHelperAgentRuntime(agent_or_runtime)
        self.adapter = adapter or LongMemEvalAdapter()
        self.max_sessions = max_sessions
        self.verbose = verbose

    def _apply_session_cap(
        self, episode: Episode, session_items: List
    ) -> tuple[List, bool]:
        """Debug-only cap: keep the last N sessions by date + ALL evidence.

        Returns (possibly-truncated items, truncated_flag). Evidence sessions
        (``answer_session_ids``) are always retained so the answer stays
        reachable. Any capped run is flagged so scores are marked non-official.
        """
        if not self.max_sessions or len(session_items) <= self.max_sessions:
            return session_items, False

        evidence = set(episode.metadata.get("answer_session_ids", []) or [])
        tail = session_items[-self.max_sessions :]
        tail_ids = {meta.get("session_id") for _, meta in tail}
        kept = list(tail)
        for text, meta in session_items[: -self.max_sessions]:
            if meta.get("session_id") in evidence and meta.get("session_id") not in tail_ids:
                kept.append((text, meta))
        return kept, True

    def run_episode(
        self, episode: Episode, log: Optional[BenchmarkLogger] = None
    ) -> RunResult:
        if log is None:
            log = BenchmarkLogger(episode.episode_id)

        session_items = self.adapter.build_session_observation_texts(episode)
        session_items, truncated = self._apply_session_cap(episode, session_items)
        if truncated:
            episode.metadata["sessions_truncated"] = True

        total_sessions = len(session_items)
        run_metadata: Dict[str, Any] = dict(episode.metadata)

        self.runtime.reset(
            episode,
            EnvironmentState(episode_id=episode.episode_id, messages=[]),
        )

        start = time.time()
        replay_turns = 0

        for i, (text, smeta) in enumerate(session_items):
            observation = Observation(
                episode_id=episode.episode_id,
                text=text,
                metadata={"benchmark_mode": "longmemeval", **smeta},
            )
            try:
                self.runtime.act(observation)  # discard the "Noted." reply
            except Exception as exc:  # a failed replay invalidates memory state
                error_msg = f"Error replaying session {i + 1}/{total_sessions}: {exc}"
                return self._error_result(episode, run_metadata, total_sessions, error_msg)
            replay_turns += smeta.get("turn_count", 0)
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  replayed {i + 1}/{total_sessions} sessions")

        replay_latency_ms = (time.time() - start) * 1000

        final_text = self.adapter.build_final_question_text(episode)
        q_start = time.time()
        try:
            action = self.runtime.act(
                Observation(
                    episode_id=episode.episode_id,
                    text=final_text,
                    metadata={"benchmark_mode": "longmemeval", "phase": "question"},
                )
            )
        except Exception as exc:
            error_msg = f"Error answering final question: {exc}"
            return self._error_result(episode, run_metadata, total_sessions, error_msg)
        question_latency_ms = (time.time() - q_start) * 1000

        predicted_answer = action.text
        trajectory = self.runtime.get_trajectory().events
        raw_messages = self.runtime.get_raw_messages()
        total_latency_ms = replay_latency_ms + question_latency_ms

        metrics = {
            "latency_ms": total_latency_ms,
            "sessions_replayed": total_sessions,
            "history_turns_replayed": replay_turns,
            "answer_length": len(predicted_answer.split()),
            "trajectory_events": len(trajectory),
            "replay_latency_ms": replay_latency_ms,
            "question_latency_ms": question_latency_ms,
        }

        return RunResult(
            sample_id=episode.episode_id,
            episode_id=episode.episode_id,
            question=episode.question,
            predicted_answer=predicted_answer,
            gold_answer=episode.gold_answer,
            trajectory=trajectory,
            raw_messages=raw_messages,
            benchmark_mode="longmemeval",
            context_turn_count=total_sessions,
            metrics=metrics,
            metadata=run_metadata,
            total_latency_ms=total_latency_ms,
            episode=episode,
        )

    def _error_result(
        self,
        episode: Episode,
        run_metadata: Dict[str, Any],
        total_sessions: int,
        error_msg: str,
    ) -> RunResult:
        return RunResult(
            sample_id=episode.episode_id,
            episode_id=episode.episode_id,
            question=episode.question,
            predicted_answer="",
            gold_answer=episode.gold_answer,
            trajectory=self.runtime.get_trajectory().events,
            raw_messages=self.runtime.get_raw_messages(),
            benchmark_mode="longmemeval",
            context_turn_count=total_sessions,
            metadata=run_metadata,
            total_latency_ms=0.0,
            episode=episode,
            error=error_msg,
        )

    def run_batch(
        self,
        episodes_iter: Iterator[Episode],
        verbose: bool | None = None,
    ) -> List[RunResult]:
        """Consume an episode ITERATOR (streaming) and collect RunResults.

        After each episode the episode's sessions are dropped to keep memory
        flat across a 500-entry ``_m`` batch; metadata retains everything the
        evaluator/report need.
        """
        show = self.verbose if verbose is None else verbose
        results: List[RunResult] = []
        for i, episode in enumerate(episodes_iter):
            if show:
                print(f"Running episode {i + 1}: {episode.episode_id}")
            result = self.run_episode(episode)
            results.append(result)
            if result.episode is not None:
                result.episode.task.context.pop("haystack_sessions", None)
        return results
