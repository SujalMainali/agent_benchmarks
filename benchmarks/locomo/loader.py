"""LoCoMo dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from benchmarks.common.models import BenchmarkSample


class LoCoMoLoader:
    """Loads and normalizes LoCoMo benchmark samples."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        """
        Initialize the LoCoMo loader.

        Args:
            data_dir: Optional path to LoCoMo data directory.
                     If None, looks for data/locomo/ in the project root.
        """
        if data_dir is None:
            data_dir = "data/locomo"
        self.data_dir = Path(data_dir)

    def load_sample_from_dict(self, raw_sample: Dict[str, Any]) -> BenchmarkSample:
        """
        Load a raw LoCoMo sample and normalize it.

        The raw sample is expected to have keys like:
        - sample_id: unique identifier
        - sessions: list of conversation sessions or turns
        - question: the QA question to answer
        - gold_answer: expected answer
        - evidence: optional list of evidence items
        - category: question type (e.g., "multi-hop", "factual")

        Args:
            raw_sample: Raw sample from LoCoMo dataset.

        Returns:
            BenchmarkSample with normalized structure.
        """
        sample_id = str(raw_sample.get("sample_id", "unknown"))
        question = raw_sample.get("question") or ""
        gold_answer = raw_sample.get("gold_answer", raw_sample.get("answer", ""))
        category = raw_sample.get("category", raw_sample.get("qa_category", "general"))

        sessions = raw_sample.get("sessions") or raw_sample.get("conversation") or []
        evidence = raw_sample.get("evidence", [])

        context = {
            "sessions": self._normalize_sessions(sessions),
            "conversation": raw_sample.get("conversation", {}),
            "evidence": evidence,
            "category": category,
            "raw_fields": raw_sample.get("raw_fields", raw_sample),
            "official": raw_sample.get("official", {}),
        }

        return BenchmarkSample(
            sample_id=sample_id,
            question=question,
            gold_answer=str(gold_answer),
            context=context,
            mode=raw_sample.get("mode", "plain_qa"),
            metadata={
                "category": category,
                "official": bool(raw_sample.get("qa")),
            },
        )

    def _normalize_sessions(self, sessions: Any) -> List[Dict[str, Any]]:
        """Normalize official and demo conversation layouts into session dictionaries."""
        if isinstance(sessions, list):
            if sessions and all(isinstance(item, dict) and "speaker" in item for item in sessions):
                return [{"turns": sessions}]
            return sessions

        if isinstance(sessions, dict):
            turn_keys = [key for key in sessions.keys() if key.startswith("session_") and isinstance(sessions.get(key), list)]
            if turn_keys:
                return [{"turns": sessions[key]} for key in sorted(turn_keys)]

        return []

    def load_from_jsonl(self, filepath: str) -> List[BenchmarkSample]:
        """
        Load samples from a JSONL file.

        Args:
            filepath: Path to JSONL file.

        Returns:
            List of BenchmarkSample objects.
        """
        samples = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    samples.extend(self._load_raw_sample(raw))
        return samples

    def load_from_json(self, filepath: str) -> List[BenchmarkSample]:
        """
        Load samples from a JSON file (list of objects).

        Args:
            filepath: Path to JSON file.

        Returns:
            List of BenchmarkSample objects.
        """
        with open(filepath) as f:
            raw_samples = json.load(f)

        # Handle both list and dict formats
        if isinstance(raw_samples, dict):
            raw_samples = raw_samples.get("samples", [])

        samples: List[BenchmarkSample] = []
        for raw in raw_samples:
            samples.extend(self._load_raw_sample(raw))
        return samples

    def iter_samples(self, filepath: str) -> Generator[BenchmarkSample, None, None]:
        """
        Iterate over samples from a file (memory-efficient for large datasets).

        Args:
            filepath: Path to JSONL or JSON file.

        Yields:
            BenchmarkSample objects one at a time.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".jsonl":
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        raw = json.loads(line)
                        for sample in self._load_raw_sample(raw):
                            yield sample
        else:
            # JSON file
            with open(filepath) as f:
                raw_samples = json.load(f)
            if isinstance(raw_samples, dict):
                raw_samples = raw_samples.get("samples", [])
            for raw in raw_samples:
                for sample in self._load_raw_sample(raw):
                    yield sample

    def _load_raw_sample(self, raw_sample: Dict[str, Any]) -> List[BenchmarkSample]:
        """Load one raw record, expanding official QA lists into individual benchmark samples."""
        if isinstance(raw_sample.get("qa"), list) and raw_sample.get("conversation"):
            sample_id = str(raw_sample.get("sample_id", "unknown"))
            conversation = raw_sample.get("conversation", {})
            sessions = self._conversation_to_sessions(conversation)
            samples: List[BenchmarkSample] = []
            for index, qa_item in enumerate(raw_sample["qa"]):
                merged = {
                    "sample_id": f"{sample_id}_{index}",
                    "question": qa_item.get("question", ""),
                    "answer": qa_item.get("answer", ""),
                    "gold_answer": qa_item.get("answer", ""),
                    "category": qa_item.get("category", 4),
                    "evidence": qa_item.get("evidence", []),
                    "conversation": conversation,
                    "sessions": sessions,
                    "qa": [qa_item],
                    "raw_fields": raw_sample,
                    "official": {
                        "conversation": conversation,
                        "qa_item": qa_item,
                        "qa_index": index,
                        "source_sample_id": sample_id,
                    },
                }
                samples.append(self.load_sample_from_dict(merged))
            return samples

        return [self.load_sample_from_dict(raw_sample)]

    def _conversation_to_sessions(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert official conversation dict into a session list."""
        sessions: List[Dict[str, Any]] = []
        for key in sorted(conversation.keys()):
            if not key.startswith("session_") or not isinstance(conversation[key], list):
                continue
            sessions.append({"turns": conversation[key], "session_key": key})
        return sessions
