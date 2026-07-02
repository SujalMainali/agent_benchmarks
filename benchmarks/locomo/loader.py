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
        sample_id = raw_sample.get("sample_id", "unknown")
        question = raw_sample.get("question", "")
        gold_answer = raw_sample.get("gold_answer", "")
        category = raw_sample.get("category", "general")
        sessions = raw_sample.get("sessions", [])
        evidence = raw_sample.get("evidence", [])

        # Build context from sessions
        context = {
            "sessions": sessions,
            "evidence": evidence,
            "category": category,
            "raw_fields": raw_sample,
        }

        return BenchmarkSample(
            sample_id=sample_id,
            question=question,
            gold_answer=gold_answer,
            context=context,
            mode="plain_qa",
            metadata={"category": category},
        )

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
                    sample = self.load_sample_from_dict(raw)
                    samples.append(sample)
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

        samples = [self.load_sample_from_dict(raw) for raw in raw_samples]
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
                        yield self.load_sample_from_dict(raw)
        else:
            # JSON file
            with open(filepath) as f:
                raw_samples = json.load(f)
            if isinstance(raw_samples, dict):
                raw_samples = raw_samples.get("samples", [])
            for raw in raw_samples:
                yield self.load_sample_from_dict(raw)
