# Benchmark Evaluators — How Each One Actually Scores

Verified 2026-07-18 by reading the wired evaluation paths in `benchmarks/` and the
vendored official code in `third_party/`. Answers the question: *which evaluators
call an LLM judge, and which credentials do they need?*

## TL;DR

| Benchmark | LLM judge? | Scoring mechanism | API key needed for scoring |
|---|---|---|---|
| **LongMemEval** | **YES** (gpt-4o via `evaluate_qa.py`) | LLM yes/no correctness prompts | `OPENAI_API_KEY` (real OpenAI) — see caveat below |
| **LoCoMo** | **NO** | Token-level F1 (stemmed), category-aware | None |
| **BFCL** | **NO** | Deterministic AST/structural checking | None |
| **ToolSandbox** | **NO** | Deterministic milestone/minefield matching + ROUGE-L | None |

**Why you didn't remember providing a judge token:** only LongMemEval actually uses an
LLM judge. LoCoMo does not (that was a misremembering — its official evaluator is pure
F1). And LongMemEval's judge *appears* to work without a real OpenAI token because your
`.env` sets `OPENAI_API_KEY` (a placeholder for local Ollama) plus `OPENAI_BASE_URL`
pointing at `http://127.0.0.1:11434/v1` — see the LongMemEval caveat.

---

## 1. LongMemEval — LLM judge (the only one)

**Path:** `benchmarks/longmemeval/official_bridge.py` → subprocess →
`third_party/longmemeval-official/src/evaluation/evaluate_qa.py` (invoked unchanged).

**What the judge does** (`evaluate_qa.py`):
- Supported metric models (`evaluate_qa.py:11-15`): `gpt-4o` (→ `gpt-4o-2024-08-06`),
  `gpt-4o-mini`, or a local `llama-3.1-70b-instruct` served at `http://localhost:8001/v1`.
- For each question it builds a task-specific yes/no prompt (`get_anscheck_prompt`,
  `evaluate_qa.py:24-43`) — different templates per question type (single-session,
  temporal-reasoning with off-by-one forgiveness, knowledge-update, preference rubric,
  abstention detection for `_abs` question ids).
- Calls `client.chat.completions.create` with `temperature=0, max_tokens=10`
  (`evaluate_qa.py:102-111`) and labels correct iff "yes" appears in the reply
  (`evaluate_qa.py:113`).
- For `gpt-*` models it reads `OPENAI_API_KEY` / `OPENAI_ORGANIZATION` from the
  environment (`evaluate_qa.py:62-65`).

**Our bridge** (`benchmarks/longmemeval/official_bridge.py`):
- Writes `hypotheses.jsonl` ({question_id, hypothesis} lines), runs the script via
  subprocess with `env={**os.environ}`, parses the `.eval-results-<model>` JSONL back.
- Fail-fast guard: raises if `metric_model` starts with `gpt` and `OPENAI_API_KEY` is
  unset (`official_bridge.py:48-53`).
- Toggle: `LONGMEMEVAL_USE_OFFICIAL_EVAL` (default true). When false, a **local
  heuristic** in `benchmarks/longmemeval/evaluator.py` is used instead
  (exact/fuzzy-0.85/contains + refusal-marker detection for abstention) — clearly
  labeled non-official, no network calls.

### ⚠️ Caveat (RESOLVED 2026-07-19): judge key is now separate

`evaluate_qa.py:70-73` constructs `OpenAI(api_key=..., base_url=None)`. With
`base_url=None`, the OpenAI SDK falls back to the **`OPENAI_BASE_URL`** env var —
which in this project points at local Ollama for the agent model. Previously the
judge subprocess inherited that and would have sent `gpt-4o` requests to Ollama.

**Fix now in place:** the judge has its own env vars:
- `LONGMEMEVAL_JUDGE_API_KEY` — the key the judge subprocess uses (fail-fast
  with a clear error if unset for gpt-* metric models — the project-level
  `OPENAI_API_KEY` is the local agent placeholder and is deliberately NOT used).
- `LONGMEMEVAL_JUDGE_BASE_URL` — optional OpenAI-compatible endpoint (DeepSeek,
  LiteLLM, vLLM, ...). When set, the bridge exports it as `OPENAI_BASE_URL` in
  the judge subprocess; when unset, the bridge strips `OPENAI_BASE_URL` /
  `OPENAI_API_BASE` so the SDK talks to the real OpenAI API. ⚠️ Caveat:
  `evaluate_qa.py` pins concrete model names (`gpt-4o` → `gpt-4o-2024-08-06`),
  so a custom endpoint must serve or alias that exact name.

Non-`gpt` judges (e.g. `llama-3.1-70b-instruct` served locally on :8001) need no
key and the env is left untouched. Alternatively set
`LONGMEMEVAL_USE_OFFICIAL_EVAL=false` for the offline heuristic.

---

## 2. LoCoMo — NO LLM judge (pure metrics)

**Path:** `benchmarks/locomo/official_bridge.py` — in-process import (no subprocess):
`sys.path.insert` of `third_party/locomo-official` (`official_bridge.py:138-140`), then
`from task_eval.evaluation import eval_question_answering` (`official_bridge.py:144`).

**Official scoring** (`third_party/locomo-official/task_eval/evaluation.py`),
`eval_question_answering` at `evaluation.py:189-241` — category-aware, all local math:
- Categories 2/3/4 (single-hop, temporal, open-domain): token-level **F1 with Porter
  stemming** (`f1_score`, `evaluation.py:126-138`); temporal answers truncated at `;`.
- Category 1 (multi-hop): multi-answer F1 — comma-split both sides, mean-of-max
  (`f1`, `evaluation.py:141-145`).
- Category 5 (adversarial): string check for "no information available" /
  "not mentioned" → 1 else 0 (`evaluation.py:217-221`).
- Evidence recall when a prediction context is present (`evaluation.py:228-237`).

Our bridge aggregates: `official_score = max(scores)`, `is_correct = score >= 0.5`
(`official_bridge.py:182-184`).

**Toggle:** `LOCOMO_USE_OFFICIAL_EVAL` (default true; `config.py:70`). When false, the
built-in exact/fuzzy(difflib ≥0.85)/contains heuristic in
`benchmarks/locomo/evaluator.py:29-85` runs instead. **Both paths are metric-based —
there is no LLM-judge mode at all.**

**Why the confusion:** the official repo *does* contain LLM code —
`task_eval/evaluate_qa.py` + `global_methods.py` read `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` — but those scripts **generate answers** with
GPT/Claude/Gemini as systems under test. They are never imported or invoked by
`benchmarks/locomo/`. Dependencies pulled in at import time (`bert_score`, `nltk`,
`rouge`) are local libraries; the wired path doesn't even use BERTScore (hard-wired to
F1 by category — the `metric='f1'` argument is never read inside the function).

---

## 3. BFCL — NO LLM judge (deterministic AST checking)

**Path:** in-process import, no subprocess. `benchmarks/bfcl/official.py:28-61`
(`bootstrap_official`) does `sys.path.insert` of `third_party/bfcl-official`, then
`benchmarks/bfcl/evaluator.py:30-40` imports the official per-entry helpers:
- `_evaluate_single_ast_entry` (`bfcl_eval/eval_checker/eval_runner.py:319`)
- `_evaluate_single_relevance_entry` (`eval_runner.py:263`)

**Checks performed (all deterministic, offline):**
1. **AST match** — decode check → function-calling-format check → `ast_checker`
   (`bfcl_eval/eval_checker/ast_eval/ast_checker.py:33`) comparing function name,
   required/unexpected params, param types (Java/JS converters), and values against
   `possible_answer` ground truth. Variants: simple / `multiple` /
   `parallel` (order-insensitive).
2. **Relevance/irrelevance** — irrelevance passes iff *no* valid call was decoded;
   relevance passes iff one was (`eval_runner.py:291-294`).

Multi-turn state-based execution, agentic, and format-sensitivity checks exist in the
official code but are **explicitly skipped** by this integration
(`benchmarks/bfcl/run.py:46-59`).

**The `gpt-4o` red herring:** `BFCL_CHECKER_MODEL_NAME=gpt-4o-2024-11-20-FC` in `.env`
looks like a judge but is **not an LLM call**. It's a "checker persona" looked up in
`MODEL_CONFIG_MAPPING` solely to control function-name normalization
(`underscore_to_dot` — whether `.` in expected names is rewritten to `_`;
`ast_checker.py:83-90`). No API client is ever instantiated for it.

**Keys:** scoring needs none. Only env vars read are path plumbing
(`BFCL_PROJECT_ROOT`, `BFCL_OFFICIAL_ROOT`, `BFCL_*` run settings). The agent under
test uses whatever key `src/config.py` provides — that's inference, not evaluation.

---

## 4. ToolSandbox — NO LLM judge (deterministic milestone/minefield)

**Path:** subprocess over JSON-lines stdio. Main process spawns
`ToolSandboxEnv/bin/python -m benchmarks.toolsandbox.worker`
(`official_bridge.py:142`); only the worker imports `tool_sandbox`. After the rollout
the worker calls `scenario.evaluation.evaluate(...)` (`worker.py:682`) →
`Evaluation.evaluate` in
`third_party/ToolSandbox-official/tool_sandbox/common/evaluation.py:1226`.

**Similarity measures (all local — no embeddings, no LLM, no network):**
- Column-level: exact match, numeric `is_close` tolerance, substring contains,
  tool-trace name+argument matching, and **ROUGE-L f-measure** for free-text columns
  (`column_rouge_l_similarity`, `evaluation.py:225` — the `rouge_score` package, a
  purely local n-gram/LCS metric, *not* a model).
- Snapshot-level: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) over
  −log column similarities (`snapshot_similarity`, `evaluation.py:283,338`).

**Final score:**
1. Milestones form a DAG; a pruned DFS finds the milestone→snapshot mapping that
   maximizes total similarity (`MilestoneMatcher._dfs`, `evaluation.py:1056-1163`).
   Per-milestone similarity = geometric mean of its constraint similarities.
2. `milestone_similarity` = arithmetic mean over milestones (`evaluation.py:1211`).
3. **Minefield nullification** (`evaluation.py:978-982`):
   `similarity = int(minefield_similarity == 0) * milestone_similarity` — touching any
   minefield zeroes the score.
4. Turn count is reported as a diagnostic only — no score penalty.

Our side: `score = similarity`, `is_correct = similarity >= 1.0`
(`benchmarks/toolsandbox/evaluator.py:27,49,90`); failure modes
`minefield_triggered` / `milestones_incomplete` / `rollout_error`.

**Keys:** scoring needs none. Two *rollout* (not scoring) paths can use keys
(both wired as of 2026-07-19):
- `TOOLSANDBOX_USER_MODE=gpt-4o|gpt-4|gpt-3.5` builds the official OpenAI user
  simulator (default `base_url="https://api.openai.com/v1"`,
  `roles/openai_api_user.py:38`) → requires a key in
  **`TOOLSANDBOX_USER_API_KEY`** (the worker rebuilds the simulator's client
  with it; the project-level `OPENAI_API_KEY` is the local-model placeholder
  and will not authenticate). **`TOOLSANDBOX_USER_BASE_URL`** optionally
  points the simulator at any OpenAI-compatible endpoint (DeepSeek, LiteLLM,
  vLLM, ...) — note the simulator classes pin concrete model names (e.g.
  `gpt-4o-2024-05-13`), so the endpoint must serve or alias the pinned name.
  Default remains `scripted` — no key, no network.
- RapidAPI search tools (`search_lat_lon`, weather, stock, currency, etc.):
  by default the worker **replaces `rapid_api_get_request` with a deterministic
  offline simulation** (`benchmarks/toolsandbox/worker.py`,
  `_simulated_rapid_api_get_request`) whose canned payloads match the official
  scenario milestones (Apple Park address/phone, AAPL, etc.) and preserve the
  WiFi gate (`ConnectionError` when wifi is off). Set
  **`TOOLSANDBOX_REAL_SEARCH_TOOLS=true`** plus **`RAPID_API_KEY`** to execute
  real web requests instead; the bridge only forwards `RAPID_API_KEY` to the
  worker in real mode, so simulated runs can never leak a live request.

---

## Summary of credentials actually required

| Env var | Needed by | When |
|---|---|---|
| `LONGMEMEVAL_JUDGE_API_KEY` (+ optional `LONGMEMEVAL_JUDGE_BASE_URL`) | LongMemEval judge | Only if `LONGMEMEVAL_USE_OFFICIAL_EVAL=true` with a `gpt-*` metric model. Base URL empty = real OpenAI API; set it to use any OpenAI-compatible endpoint (must serve/alias the pinned model name) |
| `TOOLSANDBOX_USER_API_KEY` (+ optional `TOOLSANDBOX_USER_BASE_URL`) | ToolSandbox user simulator | Only if `TOOLSANDBOX_USER_MODE` ≠ `scripted` (default is scripted). Base URL empty = api.openai.com; custom endpoint must serve/alias the pinned model name |
| `RAPID_API_KEY` + `TOOLSANDBOX_REAL_SEARCH_TOOLS=true` | ToolSandbox search tools | Only for real web-backed tool execution; default is an offline deterministic simulation (no key, no network) |
| — | LoCoMo, BFCL evaluation | Never — fully offline/deterministic |
