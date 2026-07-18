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
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
) -> str:
    """Invoke ``evaluate_qa.py`` via subprocess and return the result-file path.

    NOTE (§1.1): ``ref_file`` MUST be the same data file the run used —
    ``question_date`` differs across the three dataset files.

    ``judge_api_key`` (LONGMEMEVAL_JUDGE_API_KEY) is the key the judge uses,
    kept separate from the project-level OPENAI_API_KEY which points at the
    local agent model. ``judge_base_url`` (LONGMEMEVAL_JUDGE_BASE_URL) is an
    optional OpenAI-compatible endpoint for the judge (DeepSeek, LiteLLM,
    vLLM, ...). evaluate_qa.py builds its client with ``base_url=None`` for
    gpt-* models, so the OpenAI SDK resolves the endpoint from the
    OPENAI_BASE_URL env var — we set it to ``judge_base_url``, or strip it so
    the SDK defaults to the real OpenAI API. Without this the judge would
    inherit the project-level OPENAI_BASE_URL (local Ollama) and send gpt-4o
    requests to a server that cannot serve them.

    Caveat for custom endpoints: the official script pins concrete model
    names (gpt-4o -> gpt-4o-2024-08-06), so the endpoint must serve or alias
    that exact name.
    """
    env = {**os.environ}
    if metric_model.startswith("gpt"):
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_BASE", None)
        if judge_base_url:
            env["OPENAI_BASE_URL"] = judge_base_url
        if judge_api_key:
            env["OPENAI_API_KEY"] = judge_api_key
        else:
            raise RuntimeError(
                f"metric_model='{metric_model}' needs an API key for the "
                "judge. Set LONGMEMEVAL_JUDGE_API_KEY (the project-level "
                "OPENAI_API_KEY is reserved for the local agent model), or use "
                "LONGMEMEVAL_USE_OFFICIAL_EVAL=false for the local heuristic "
                "evaluator."
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
        env=env,
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
