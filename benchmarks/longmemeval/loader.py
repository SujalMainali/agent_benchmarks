"""LongMemEval loader with a streaming top-level-array JSON parser.

``longmemeval_m_cleaned.json`` is 1.6 GB; ``json.load`` would materialize
~2-3 GB and every retained episode keeps ~5 MB of sessions alive. The loader
therefore parses the top-level array incrementally, yields one Episode at a
time, and applies filters on the raw dict BEFORE building the Episode.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from benchmarks.common.interfaces import BenchmarkLoader
from benchmarks.common.models import Episode, Task

_CHUNK = 32 * 1024 * 1024


def iter_raw_entries(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one entry dict at a time from a top-level JSON array file."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(_CHUNK)
        idx = buf.index("[") + 1
        while True:
            # skip whitespace / commas between entries
            while idx < len(buf) and buf[idx] in " \t\r\n,":
                idx += 1
            if idx < len(buf) and buf[idx] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                chunk = f.read(_CHUNK)  # entry spans the buffer edge
                if not chunk:
                    raise
                buf = buf[idx:] + chunk  # drop consumed prefix, retry
                idx = 0
                continue
            yield obj
            buf = buf[end:]  # free consumed text
            idx = 0


class LongMemEvalLoader(BenchmarkLoader):
    """Normalizes raw LongMemEval entries into Episodes."""

    def load(self, raw_data: Dict[str, Any]) -> Episode:
        """Map ONE raw entry to an Episode, preserving everything."""
        raw = raw_data
        question_id = str(raw["question_id"])
        haystack_sessions = raw["haystack_sessions"]
        haystack_session_ids = list(raw["haystack_session_ids"])
        haystack_dates = list(raw["haystack_dates"])

        has_answer_turns = {
            sid: [i for i, t in enumerate(sess) if isinstance(t, dict) and t.get("has_answer")]
            for sid, sess in zip(haystack_session_ids, haystack_sessions)
            if any(isinstance(t, dict) and t.get("has_answer") for t in sess)
        }

        metadata: Dict[str, Any] = {
            "question_id": question_id,
            "question_type": str(raw["question_type"]),
            "question_date": str(raw["question_date"]),
            "answer_session_ids": list(raw.get("answer_session_ids", [])),
            "haystack_session_ids": haystack_session_ids,
            "haystack_dates": haystack_dates,
            "is_abstention": "_abs" in question_id,
            "num_sessions": len(haystack_sessions),
            "num_turns": sum(len(s) for s in haystack_sessions),
            "has_answer_turns": has_answer_turns,
        }

        return Episode(
            episode_id=question_id,
            task=Task(
                task_id=question_id,
                question=str(raw["question"]),
                gold_answer=str(raw["answer"]),
                context={
                    "haystack_sessions": haystack_sessions,  # untouched, has_answer intact
                    "haystack_dates": haystack_dates,
                    "haystack_session_ids": haystack_session_ids,
                },
                mode="longmemeval",
                metadata=dict(metadata),
            ),
            metadata=metadata,
            raw_data={},  # do NOT stash the full raw entry again (memory)
        )

    def iter_episodes(
        self,
        path: str,
        question_id: Optional[str] = None,
        question_types: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Iterator[Episode]:
        """Stream Episodes from ``path``, filtering on the raw dict first.

        Filters are applied to the raw entry (cheaply, before Episode
        construction) and the generator early-stops once ``question_id`` is
        found or ``max_samples`` episodes have been yielded.
        """
        type_filter = set(question_types) if question_types else None
        yielded = 0
        for raw in iter_raw_entries(path):
            rid = str(raw.get("question_id", ""))
            if question_id is not None and rid != question_id:
                continue
            if type_filter is not None and str(raw.get("question_type", "")) not in type_filter:
                continue

            yield self.load(raw)
            yielded += 1

            if question_id is not None:
                return
            if max_samples is not None and yielded >= max_samples:
                return
