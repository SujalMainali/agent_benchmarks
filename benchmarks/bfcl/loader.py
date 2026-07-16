"""BFCL dataset loader — official data files -> shared ``Episode`` objects.

Loads test entries straight from the vendored official datasets
(``third_party/bfcl-official/bfcl_eval/data/``) using the official
``load_dataset_entry`` helper, so category-specific preprocessing (memory,
web-search, language hints) stays official. No prompt construction, no tool
building, no conversation merging happens here — that is the adapter's job.

Every raw field of an entry is preserved verbatim in ``Episode.raw_data``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from benchmarks.common.interfaces import BenchmarkLoader
from benchmarks.common.models import Episode, Task

from .official import bootstrap_official

bootstrap_official()

from bfcl_eval.utils import (  # noqa: E402
    extract_test_category_from_id,
    load_dataset_entry,
)


class BFCLLoader(BenchmarkLoader):
    """Normalizes official BFCL test entries into shared ``Episode`` objects."""

    def load(self, raw_data: Dict[str, Any]) -> Episode:
        """Normalize one official test entry into an ``Episode``.

        The entry is stored untouched in ``raw_data`` so downstream layers
        (adapter, bridge, evaluator) always work from the official record and
        unknown/future fields survive the round trip.
        """
        entry_id = str(raw_data["id"])
        test_category = extract_test_category_from_id(entry_id)

        question_text = self._last_user_content(raw_data.get("question", []))

        task = Task(
            task_id=entry_id,
            question=question_text,
            # Ground truth lives in the official possible_answer files and is
            # loaded by the evaluator via the official helper; it is not part
            # of the generation-side record (mirrors the official pipeline).
            gold_answer="",
            context={},
            mode="bfcl",
            metadata={"test_category": test_category},
        )

        return Episode(
            episode_id=entry_id,
            task=task,
            metadata={
                "test_category": test_category,
                "benchmark": "bfcl",
            },
            raw_data=dict(raw_data),
        )

    def load_category(self, test_category: str) -> List[Episode]:
        """Load every entry of one official test category.

        Uses the official ``load_dataset_entry`` (with language-specific
        hints, exactly like the official generation pipeline) so datasets are
        read from the vendored repo and never duplicated.
        """
        entries = load_dataset_entry(
            test_category,
            include_prereq=True,
            include_language_specific_hint=True,
        )
        return [self.load(entry) for entry in entries]

    def load_categories(
        self,
        test_categories: Iterable[str],
        max_samples: Optional[int] = None,
        run_ids: Optional[List[str]] = None,
    ) -> List[Episode]:
        """Load several categories, optionally capping or filtering by id."""
        episodes: List[Episode] = []
        run_id_set = set(run_ids or [])
        for category in test_categories:
            category_episodes = self.load_category(category)
            if run_id_set:
                category_episodes = [
                    episode
                    for episode in category_episodes
                    if episode.episode_id in run_id_set
                ]
            if max_samples is not None:
                category_episodes = category_episodes[:max_samples]
            episodes.extend(category_episodes)
        return episodes

    @staticmethod
    def _last_user_content(question: Any) -> str:
        """Extract the final user utterance for ``Episode.question``.

        ``question`` is the official list-of-turns structure; each turn is a
        list of role dicts. Only used for display/observation purposes — the
        full structure stays in ``raw_data``.
        """
        if not isinstance(question, list):
            return ""
        for turn in reversed(question):
            if not isinstance(turn, list):
                continue
            for message in reversed(turn):
                if isinstance(message, dict) and message.get("role") == "user":
                    return str(message.get("content", ""))
        return ""
