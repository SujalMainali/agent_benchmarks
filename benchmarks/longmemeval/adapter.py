"""LongMemEval adapter — pure conversion from Episode to agent-visible text.

No execution, no evaluation. Implements the BenchmarkAdapter ABC plus the
session-replay builder the runner drives.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from benchmarks.common.interfaces import BenchmarkAdapter
from benchmarks.common.models import BenchmarkSample, Episode

_DATE_FMT = "%Y/%m/%d %H:%M"


def _parse_session_date(date_str: str) -> datetime | None:
    """Parse ``"2023/04/10 (Mon) 23:07"`` into a datetime; None on failure."""
    try:
        cleaned = re.sub(r"\s*\(\w+\)\s*", " ", str(date_str)).strip()
        return datetime.strptime(cleaned, _DATE_FMT)
    except (ValueError, TypeError):
        return None


class LongMemEvalAdapter(BenchmarkAdapter):
    """Converts LongMemEval samples into ResearchHelperAgent input text."""

    def load_sample(self, sample_data: Dict[str, Any]) -> BenchmarkSample:
        """Delegate to the loader, then wrap as a BenchmarkSample."""
        from .loader import LongMemEvalLoader

        episode = LongMemEvalLoader().load(sample_data)
        return BenchmarkSample.from_episode(episode)

    def build_context_messages(self, sample: BenchmarkSample | Episode) -> List[Any]:
        """Return [] — LongMemEval seeds nothing through reset().

        History deliberately flows through repeated act() calls (one per
        session) so the agent's memory accumulates naturally; nothing is
        pre-seeded into the initial environment state. The method exists only
        to satisfy the BenchmarkAdapter ABC.
        """
        return []

    def build_session_observation_texts(
        self, episode: Episode
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Render one replay text (+ per-session metadata) per history session.

        Sessions are ordered by parsed ``haystack_dates`` (``_s``/``_m`` are
        already sorted; oracle is not — sorting makes all three consistent). On
        a date parse failure the original order is preserved. ``has_answer`` and
        any key other than role/content are NEVER rendered into the text.
        """
        context = episode.task.context
        sessions = context.get("haystack_sessions", []) or []
        session_ids = context.get("haystack_session_ids", []) or []
        dates = context.get("haystack_dates", []) or []

        indexed = []
        for i, session in enumerate(sessions):
            sid = session_ids[i] if i < len(session_ids) else f"session_{i}"
            date_str = str(dates[i]) if i < len(dates) else ""
            indexed.append((i, sid, date_str, session))

        parsed_all = all(_parse_session_date(d) is not None for _, _, d, _ in indexed if d)
        if indexed and parsed_all:
            indexed.sort(
                key=lambda item: _parse_session_date(item[2]) or datetime.min
            )

        results: List[Tuple[str, Dict[str, Any]]] = []
        for order, (orig_index, sid, date_str, session) in enumerate(indexed):
            lines = [f"[Past chat session — {date_str}]"]
            turn_count = 0
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                role = str(turn.get("role", "")).strip().upper() or "MESSAGE"
                lines.append(f"{role}: {content}")
                turn_count += 1

            text = "\n".join(lines)
            meta = {
                "session_id": sid,
                "session_date": date_str,
                "session_index": order,
                "turn_count": turn_count,
                "phase": "history_replay",
            }
            results.append((text, meta))

        return results

    def build_final_question_text(self, episode: Episode) -> str:
        """The final user turn: current date header + the question."""
        question_date = episode.metadata.get("question_date", "")
        question = episode.question
        return f"Current date: {question_date}\n\n{question}"

    def build_agent_input(self, sample: BenchmarkSample | Episode) -> Dict[str, Any]:
        """Assemble the conversion bundle (runner may call builders directly)."""
        episode = sample.to_episode() if isinstance(sample, BenchmarkSample) else sample
        return {
            "question": episode.question,
            "gold_answer": episode.gold_answer,
            "session_texts": self.build_session_observation_texts(episode),
            "final_question": self.build_final_question_text(episode),
            "mode": "longmemeval",
            "metadata": episode.metadata,
            "sample_id": episode.episode_id,
            "episode": episode,
        }
