"""Subprocess bridge to the official LongMemEval QA judge (evaluate_qa.py).

The official script is invoked UNCHANGED. We only write the hypotheses file it
consumes and parse the ``.eval-results-*`` file it produces. QA scoring stays
entirely inside ``evaluate_qa.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from benchmarks.common.models import RunResult


def write_hypotheses_jsonl(run_results: List[RunResult], out_path: str) -> str:
    """Write one ``{"question_id", "hypothesis"}`` line per run result."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in run_results:
            f.write(
                json.dumps(
                    {"question_id": r.sample_id, "hypothesis": r.predicted_answer or ""}
                )
                + "\n"
            )
    return str(out)


def run_official_evaluation(
    hyp_file: str,
    ref_file: str,
    metric_model: str,
    official_root: str,
    python_executable: str = sys.executable,
    timeout: int = 3600,
) -> str:
    """Invoke ``evaluate_qa.py`` via subprocess and return the result-file path.

    NOTE (§1.1): ``ref_file`` MUST be the same data file the run used —
    ``question_date`` differs across the three dataset files.
    """
    if metric_model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is not set but metric_model='{metric_model}' requires it. "
            "Set OPENAI_API_KEY or use LONGMEMEVAL_USE_OFFICIAL_EVAL=false for the "
            "local heuristic evaluator."
        )

    official = Path(official_root).resolve()
    script = official / "src/evaluation/evaluate_qa.py"
    if not script.exists():
        raise FileNotFoundError(f"Official evaluator not found: {script}")

    hyp_resolved = str(Path(hyp_file).resolve())
    ref_resolved = str(Path(ref_file).resolve())

    subprocess.run(
        [python_executable, str(script), metric_model, hyp_resolved, ref_resolved],
        cwd=str(official / "src/evaluation"),
        env={**os.environ},
        check=True,
        timeout=timeout,
    )
    return f"{hyp_file}.eval-results-{metric_model}"


def parse_official_results(result_file: str) -> Dict[str, bool]:
    """Read the ``.eval-results-*`` JSONL, return {question_id: label}."""
    labels: Dict[str, bool] = {}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = str(entry.get("question_id", ""))
            autoeval = entry.get("autoeval_label", {}) or {}
            labels[qid] = bool(autoeval.get("label", False))
    return labels
