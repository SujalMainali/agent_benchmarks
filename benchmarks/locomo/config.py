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
    return int(os.getenv(name, default))


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.split("#", 1)[0].strip()


@dataclass(frozen=True)
class LoCoMoSettings:
    """Benchmark-only settings for LoCoMo runs.

    Keep these values separate from `src.config.Settings` so the agent core
    stays generic while benchmark behavior stays easy to edit from env vars.
    """

    data_file: str
    output_dir: str
    official_root: str
    use_official_eval: bool
    allow_tools: bool
    prompt_mode: str
    run_mode: str
    sample_id: str | None
    question_count: int | None
    max_samples: int | None
    create_demo_data: bool
    verbose: bool


def load_locomo_settings() -> LoCoMoSettings:
    """Load LoCoMo benchmark settings from environment variables.

    Environment variables:
    - LOCOMO_DATA_FILE: Input dataset path for LoCoMo JSON or JSONL samples.
    - LOCOMO_OUTPUT_DIR: Directory where benchmark reports are written.
    - LOCOMO_RUN_MODE: `single` runs one sample, `batch` runs all or many.
    - LOCOMO_SAMPLE_ID: Optional sample id to target in single mode.
    - LOCOMO_QUESTION_COUNT: Optional cap on QA items to run for the targeted
      sample in single mode; `0` means run all questions.
    - LOCOMO_MAX_SAMPLES: Optional cap for batch runs; `0` means no cap.
    - LOCOMO_CREATE_DEMO_DATA: If true, example.py creates demo data first.
    - LOCOMO_VERBOSE: If true, batch runs print progress.
    """

    max_samples = _get_int("LOCOMO_MAX_SAMPLES", "0")
    question_count = _get_int("LOCOMO_QUESTION_COUNT", "0")

    return LoCoMoSettings(
        data_file=_get_str("LOCOMO_DATA_FILE", "data/locomo/locomo10.jsonl"),
        output_dir=_get_str("LOCOMO_OUTPUT_DIR", "results/locomo"),
        official_root=_get_str("LOCOMO_OFFICIAL_ROOT", "third_party/locomo-official"),
        use_official_eval=_get_bool("LOCOMO_USE_OFFICIAL_EVAL", "true"),
        allow_tools=_get_bool("LOCOMO_ALLOW_TOOLS", "false"),
        prompt_mode=_get_str("LOCOMO_PROMPT_MODE", "qa").lower(),
        run_mode=_get_str("LOCOMO_RUN_MODE", "single").lower(),
        sample_id=_get_str("LOCOMO_SAMPLE_ID", "") or None,
        question_count=question_count if question_count > 0 else None,
        max_samples=max_samples if max_samples > 0 else None,
        create_demo_data=_get_bool("LOCOMO_CREATE_DEMO_DATA", "true"),
        verbose=_get_bool("LOCOMO_VERBOSE", "true"),
    )