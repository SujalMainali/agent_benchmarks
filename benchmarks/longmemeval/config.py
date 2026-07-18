from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _get_int(name: str, default: str) -> int:
    return int(str(os.getenv(name, default)).split("#", 1)[0].strip())


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.split("#", 1)[0].strip()


@dataclass(frozen=True)
class LongMemEvalSettings:
    """Benchmark-only settings for LongMemEval runs.

    Kept separate from ``src.config.Settings`` so the agent core stays generic
    while benchmark behavior stays configurable from env vars.
    """

    data_file: str
    output_dir: str
    official_root: str
    run_mode: str
    question_id: str | None
    max_samples: int | None
    question_types: list[str] | None
    use_official_eval: bool
    metric_model: str
    allow_tools: bool
    max_sessions: int
    full_trajectory: bool
    verbose: bool


def load_longmemeval_settings() -> LongMemEvalSettings:
    """Load LongMemEval benchmark settings from environment variables.

    Environment variables (see .env.example for the documented block):
    - LONGMEMEVAL_DATA_FILE: dataset to run AND the official eval ref_file.
    - LONGMEMEVAL_OUTPUT_DIR: directory where reports are written.
    - LONGMEMEVAL_OFFICIAL_ROOT: vendored official repo root.
    - LONGMEMEVAL_RUN_MODE: `single` runs one question, `batch` runs many.
    - LONGMEMEVAL_QUESTION_ID: target one question in single mode.
    - LONGMEMEVAL_MAX_SAMPLES: batch cap; `0` means no cap.
    - LONGMEMEVAL_QUESTION_TYPES: optional comma-separated question_type filter.
    - LONGMEMEVAL_USE_OFFICIAL_EVAL: official judge vs local heuristic.
    - LONGMEMEVAL_METRIC_MODEL: judge model key for evaluate_qa.py.
    - LONGMEMEVAL_ALLOW_TOOLS: passed to the agent (memory QA needs no tools).
    - LONGMEMEVAL_MAX_SESSIONS: DEBUG-ONLY cap on replayed sessions; `0` = all.
    - LONGMEMEVAL_FULL_TRAJECTORY: if false, per-sample reports truncate replay.
    - LONGMEMEVAL_VERBOSE: progress printing.
    """

    max_samples = _get_int("LONGMEMEVAL_MAX_SAMPLES", "0")
    raw_types = _get_str("LONGMEMEVAL_QUESTION_TYPES", "")
    question_types = (
        [t.strip() for t in raw_types.split(",") if t.strip()] if raw_types else None
    )

    return LongMemEvalSettings(
        data_file=_get_str(
            "LONGMEMEVAL_DATA_FILE",
            "third_party/longmemeval-official/data/longmemeval_m_cleaned.json",
        ),
        output_dir=_get_str("LONGMEMEVAL_OUTPUT_DIR", "results/longmemeval"),
        official_root=_get_str("LONGMEMEVAL_OFFICIAL_ROOT", "third_party/longmemeval-official"),
        run_mode=_get_str("LONGMEMEVAL_RUN_MODE", "single").lower(),
        question_id=_get_str("LONGMEMEVAL_QUESTION_ID", "") or None,
        max_samples=max_samples if max_samples > 0 else None,
        question_types=question_types,
        use_official_eval=_get_bool("LONGMEMEVAL_USE_OFFICIAL_EVAL", "true"),
        metric_model=_get_str("LONGMEMEVAL_METRIC_MODEL", "gpt-4o"),
        allow_tools=_get_bool("LONGMEMEVAL_ALLOW_TOOLS", "false"),
        max_sessions=_get_int("LONGMEMEVAL_MAX_SESSIONS", "0"),
        full_trajectory=_get_bool("LONGMEMEVAL_FULL_TRAJECTORY", "false"),
        verbose=_get_bool("LONGMEMEVAL_VERBOSE", "true"),
    )
