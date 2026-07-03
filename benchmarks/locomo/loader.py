"""LoCoMo dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from benchmarks.common.models import BenchmarkSample, Episode, Task


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

    def load_episode_from_dict(self, raw_sample: Dict[str, Any]) -> Episode:
        """Load a raw LoCoMo record into the shared Episode model."""
        sample = self.load_sample_from_dict(raw_sample)
        return sample.to_episode()

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
        samples: List[BenchmarkSample] = []
        for raw in self._read_json_records(filepath):
            samples.extend(self._load_raw_sample(raw))
        return samples

    def load_episodes_from_jsonl(self, filepath: str) -> List[Episode]:
        """Load episodes from a JSONL file."""
        episodes: List[Episode] = []
        for raw in self._read_json_records(filepath):
            episodes.extend(self._load_raw_episodes(raw))
        return episodes

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

    def load_episodes_from_json(self, filepath: str) -> List[Episode]:
        """Load episodes from a JSON file."""
        with open(filepath) as f:
            raw_samples = json.load(f)

        if isinstance(raw_samples, dict):
            raw_samples = raw_samples.get("samples", [])

        episodes: List[Episode] = []
        for raw in raw_samples:
            episodes.extend(self._load_raw_episodes(raw))
        return episodes

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
            for raw in self._read_json_records(filepath):
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

    def iter_episodes(self, filepath: str) -> Generator[Episode, None, None]:
        """Iterate over episodes from a file."""
        filepath = Path(filepath)
        if filepath.suffix == ".jsonl":
            for raw in self._read_json_records(filepath):
                for episode in self._load_raw_episodes(raw):
                    yield episode
        else:
            with open(filepath) as f:
                raw_samples = json.load(f)
            if isinstance(raw_samples, dict):
                raw_samples = raw_samples.get("samples", [])
            for raw in raw_samples:
                for episode in self._load_raw_episodes(raw):
                    yield episode

    def _read_json_records(self, filepath: str) -> List[Dict[str, Any]]:
        """Read JSONL or concatenated JSON objects from a file."""
        text = Path(filepath).read_text()
        stripped = text.strip()
        if not stripped:
            return []

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return self._parse_json_stream(stripped)

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            samples = parsed.get("samples")
            if isinstance(samples, list):
                return samples
            return [parsed]
        return [parsed]

    def _parse_json_stream(self, text: str) -> List[Dict[str, Any]]:
        """Parse a stream of JSON objects separated by whitespace or newlines."""
        decoder = json.JSONDecoder()
        index = 0
        length = len(text)
        records: List[Dict[str, Any]] = []

        while index < length:
            while index < length and text[index].isspace():
                index += 1
            if index >= length:
                break

            record, end = decoder.raw_decode(text, index)
            records.append(record)
            index = end

            while index < length and (text[index].isspace() or text[index] == ","):
                index += 1

        return records

    def _load_raw_sample(self, raw_sample: Dict[str, Any]) -> List[BenchmarkSample]:
        """Load one raw record, expanding official QA lists into individual benchmark samples."""
        return [BenchmarkSample.from_episode(episode) for episode in self._load_raw_episodes(raw_sample)]

    def _load_raw_episodes(self, raw_sample: Dict[str, Any]) -> List[Episode]:
        """Load one raw record, expanding official QA lists into individual episodes."""
        if isinstance(raw_sample.get("qa"), list) and raw_sample.get("conversation"):
            sample_id = str(raw_sample.get("sample_id", "unknown"))
            conversation = raw_sample.get("conversation", {})
            sessions = self._conversation_to_sessions(conversation)
            episodes: List[Episode] = []
            for index, qa_item in enumerate(raw_sample["qa"]):
                task = Task(
                    task_id=f"{sample_id}_{index}",
                    question=str(qa_item.get("question", "")),
                    gold_answer=str(qa_item.get("answer", "")),
                    context={
                        "conversation": conversation,
                        "sessions": sessions,
                        "evidence": qa_item.get("evidence", []),
                        "category": qa_item.get("category", 4),
                        "raw_fields": raw_sample,
                        "official": {
                            "conversation": conversation,
                            "qa_item": qa_item,
                            "qa_index": index,
                            "source_sample_id": sample_id,
                        },
                    },
                    mode=str(raw_sample.get("mode", "plain_qa")),
                    metadata={
                        "category": qa_item.get("category", 4),
                        "official": True,
                    },
                )
                episodes.append(
                    Episode(
                        episode_id=f"{sample_id}_{index}",
                        task=task,
                        metadata={
                            "category": qa_item.get("category", 4),
                            "official": True,
                            "source_sample_id": sample_id,
                            "qa_index": index,
                        },
                        raw_data=raw_sample,
                    )
                )
            return episodes

        return [self.load_episode_from_dict(raw_sample)]

    def _conversation_to_sessions(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert official conversation dict into a session list."""
        sessions: List[Dict[str, Any]] = []
        for key in sorted(conversation.keys()):
            if not key.startswith("session_") or not isinstance(conversation[key], list):
                continue
            sessions.append({"turns": conversation[key], "session_key": key})
        return sessions
