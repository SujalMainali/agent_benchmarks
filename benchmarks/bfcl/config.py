"""Benchmark-only settings for BFCL runs.

Kept separate from ``src.config.Settings`` so the agent core stays generic,
mirroring ``benchmarks/locomo/config.py`` and ``benchmarks/toolsandbox/config.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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


def _get_list(name: str, default: str = "") -> List[str]:
    raw = _get_str(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class BFCLSettings:
    """Settings that control a BFCL benchmark run."""

    official_root: str
    test_categories: List[str] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    max_samples: Optional[int] = None
    max_tool_steps: int = 1
    output_dir: str = "results/bfcl"
    checker_model_name: str = "gpt-4o-2024-11-20-FC"
    include_input_log: bool = False
    verbose: bool = True


def load_bfcl_settings() -> BFCLSettings:
    """Load BFCL benchmark settings from environment variables.

    Environment variables:
    - BFCL_OFFICIAL_ROOT: Path to the vendored official BFCL repo.
    - BFCL_TEST_CATEGORY: Comma-separated categories or collections (e.g.
      ``simple_python`` or ``single_turn``). Resolved through the official
      ``TEST_COLLECTION_MAPPING`` so new BFCL categories work unchanged.
    - BFCL_RUN_IDS: Optional comma-separated exact test-entry ids to run.
    - BFCL_MAX_SAMPLES: Optional cap on entries per category; ``0`` = no cap.
    - BFCL_MAX_TOOL_STEPS: Agent tool-loop budget per entry (single-turn BFCL
      entries expect exactly one model call; keep at 1 for faithful scoring).
    - BFCL_OUTPUT_DIR: Directory where benchmark reports are written.
    - BFCL_CHECKER_MODEL_NAME: Registered BFCL model name used as the checker
      persona (controls only official function-name normalization); must exist
      in the official ``MODEL_CONFIG_MAPPING``.
    - BFCL_INCLUDE_INPUT_LOG: Include the fully-transformed model input in the
      per-entry inference log (official ``--include-input-log`` equivalent).
    - BFCL_VERBOSE: Print progress to the terminal.
    """
    max_samples = _get_int("BFCL_MAX_SAMPLES", "0")

    return BFCLSettings(
        official_root=_get_str("BFCL_OFFICIAL_ROOT", "third_party/bfcl-official"),
        test_categories=_get_list("BFCL_TEST_CATEGORY", "simple_python"),
        run_ids=_get_list("BFCL_RUN_IDS", ""),
        max_samples=max_samples if max_samples > 0 else None,
        max_tool_steps=_get_int("BFCL_MAX_TOOL_STEPS", "1"),
        output_dir=_get_str("BFCL_OUTPUT_DIR", "results/bfcl"),
        checker_model_name=_get_str("BFCL_CHECKER_MODEL_NAME", "gpt-4o-2024-11-20-FC"),
        include_input_log=_get_bool("BFCL_INCLUDE_INPUT_LOG", "false"),
        verbose=_get_bool("BFCL_VERBOSE", "true"),
    )
