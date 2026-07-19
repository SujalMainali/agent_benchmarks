# LongMemEval — Environment Configuration

How to configure and target a LongMemEval run. Every field is read by
[benchmarks/longmemeval/config.py](benchmarks/longmemeval/config.py) from `.env`
(loaded with `override=True`) and consumed in
[benchmarks/longmemeval/run.py](benchmarks/longmemeval/run.py).

Run it with:

```bash
./AgentEnv/bin/python -m benchmarks.longmemeval.run
```

The agent under test is chosen by **`AGENT_DRIVER`** (see
[DriverInterface.md](DriverInterface.md) / [AdaMemDrivers.md](AdaMemDrivers.md)).
LongMemEval is memory-QA with no tools — the AdaMem drivers
(`adamem_flat_rag` / `adamem_episodic` / `adamem_mem1`) fit it directly.

> **How a question runs:** the runner resets the agent, then replays each past
> chat session as its own `act()` (ingest-only, memory accumulates), then asks
> the final question. See [AgentInterface.md](AgentInterface.md).

---

## Config fields

| Env var | Type | Default | Effect |
|---|---|---|---|
| `LONGMEMEVAL_DATA_FILE` | path | `third_party/longmemeval-official/data/longmemeval_m_cleaned.json` | Dataset to run **and** the official evaluator's `ref_file`. `question_date` differs across files, so the run file and eval ref must be the same file. See datasets below. |
| `LONGMEMEVAL_OUTPUT_DIR` | path | `results/longmemeval` | Root of the immutable run dir. Results also bucket by `MEMORY_ARCHITECTURE`. |
| `LONGMEMEVAL_OFFICIAL_ROOT` | path | `third_party/longmemeval-official` | Vendored official repo; the official judge subprocess runs from here. |
| `LONGMEMEVAL_RUN_MODE` | enum | `single` | `single` runs one question, `batch` streams many. Any other value → error + exit. |
| `LONGMEMEVAL_QUESTION_ID` | string | *(unset)* | **single mode.** Target one question by `question_id`. Unset → first entry (after any type filter). |
| `LONGMEMEVAL_MAX_SAMPLES` | int | `0` (all) | **batch mode.** Cap number of questions. `0`/unset = no cap. |
| `LONGMEMEVAL_QUESTION_TYPES` | csv | *(unset)* | Filter by `question_type` (comma-separated). Applies in both modes (in single mode it narrows which question is "first"). Values below. |
| `LONGMEMEVAL_USE_OFFICIAL_EVAL` | bool | `true` | `true` → official LLM judge (`evaluate_qa.py`). `false` → local offline heuristic (no API key needed). |
| `LONGMEMEVAL_METRIC_MODEL` | enum | `gpt-4o` | Judge model key for `evaluate_qa.py`: `gpt-4o` \| `gpt-4o-mini` \| `llama-3.1-70b-instruct`. Only used when official eval is on. |
| `LONGMEMEVAL_JUDGE_API_KEY` | string | *(unset)* | API key for the judge subprocess. Required for `gpt-*` judges. Kept separate from the project-level `OPENAI_API_KEY`/`OPENAI_BASE_URL` (those point at the local agent model, e.g. Ollama). |
| `LONGMEMEVAL_JUDGE_BASE_URL` | url | *(unset)* | Optional OpenAI-compatible endpoint for the judge (DeepSeek/LiteLLM/vLLM). Unset → real OpenAI API (the bridge strips the local base url). The script pins concrete model names (`gpt-4o` → `gpt-4o-2024-08-06`), so a custom endpoint must serve/alias that name. |
| `LONGMEMEVAL_ALLOW_TOOLS` | bool | `false` | Passed to the agent. Memory QA needs no tools — keep `false`. |
| `LONGMEMEVAL_MAX_SESSIONS` | int | `0` (all) | **DEBUG-ONLY** cap on replayed history sessions per question. `0` = replay all. When `>0`, the run is flagged `sessions_truncated` and must **not** be reported as an official score (it keeps the last N sessions by date plus all evidence sessions). |
| `LONGMEMEVAL_FULL_TRAJECTORY` | bool | `false` | If `false`, per-sample reports collapse the (huge) history-replay events. `true` keeps every replay event in the processed report. Raw streamed trajectory is unaffected. |
| `LONGMEMEVAL_VERBOSE` | bool | `true` | Print progress (replay counters, per-question result). |

Booleans are truthy for `1/true/yes/y`.

### Datasets (`LONGMEMEVAL_DATA_FILE`)

Shipped under `third_party/longmemeval-official/data/`:

| File | Size | Notes |
|---|---|---|
| `longmemeval_oracle.json` | ~15 MB | 500 questions, only evidence sessions — cheap dev/smoke dataset. |
| `longmemeval_s_cleaned.json` | ~277 MB | "short" haystack; realistic long-context. |
| `longmemeval_m_cleaned.json` | ~2.7 GB | "medium" haystack; hundreds of sessions per question (slow; loader streams it). |

### `LONGMEMEVAL_QUESTION_TYPES` values

From the dataset's `question_type` field (counts shown for the 500-question
oracle set):

`temporal-reasoning` (133), `multi-session` (133), `knowledge-update` (78),
`single-session-user` (70), `single-session-assistant` (56),
`single-session-preference` (30).

Abstention variants have `_abs` in the `question_id` (flagged
`is_abstention` in metadata).

---

## Listing available questions / types

No CLI flag — read from the dataset. Use the small `oracle` file for speed.

**Question-type distribution:**

```bash
./AgentEnv/bin/python - <<'PY'
import json
from collections import Counter
path = "third_party/longmemeval-official/data/longmemeval_oracle.json"
data = json.load(open(path))
c = Counter(str(e.get("question_type", "")) for e in data)
print(f"{len(data)} questions")
for t, n in c.most_common():
    print(f"  {t:30} {n}")
PY
```

**Question ids** (optionally filtered by type), for `LONGMEMEVAL_QUESTION_ID`:

```bash
./AgentEnv/bin/python - <<'PY'
import json
path = "third_party/longmemeval-official/data/longmemeval_s_cleaned.json"
WANT = "temporal-reasoning"          # <-- set to "" for all types
data = json.load(open(path))
for e in data:
    if not WANT or str(e.get("question_type")) == WANT:
        print(e["question_id"], "|", str(e.get("question_type")), "|", e["question"][:60])
PY
```

For the giant `_m` file, stream instead of `json.load` (it's 2.7 GB):

```bash
./AgentEnv/bin/python - <<'PY'
from benchmarks.longmemeval.loader import iter_raw_entries
path = "third_party/longmemeval-official/data/longmemeval_m_cleaned.json"
for i, e in enumerate(iter_raw_entries(path)):
    print(e["question_id"], "|", e.get("question_type"))
    if i >= 20: break
PY
```

---

## Common recipes

Single question, official judge (needs a judge key):

```bash
LONGMEMEVAL_RUN_MODE=single \
LONGMEMEVAL_DATA_FILE=third_party/longmemeval-official/data/longmemeval_oracle.json \
LONGMEMEVAL_QUESTION_ID=0bb5a684 \
LONGMEMEVAL_JUDGE_API_KEY=sk-... \
AGENT_DRIVER=adamem_mem1 \
./AgentEnv/bin/python -m benchmarks.longmemeval.run
```

Offline smoke — first `temporal-reasoning` question on oracle, no judge key,
capped replay:

```bash
LONGMEMEVAL_RUN_MODE=single \
LONGMEMEVAL_DATA_FILE=third_party/longmemeval-official/data/longmemeval_oracle.json \
LONGMEMEVAL_QUESTION_TYPES=temporal-reasoning \
LONGMEMEVAL_USE_OFFICIAL_EVAL=false \
LONGMEMEVAL_MAX_SESSIONS=5 \
AGENT_DRIVER=adamem_flat_rag \
./AgentEnv/bin/python -m benchmarks.longmemeval.run
```

Batch of 20 multi-session questions, official judge via a custom endpoint:

```bash
LONGMEMEVAL_RUN_MODE=batch LONGMEMEVAL_MAX_SAMPLES=20 \
LONGMEMEVAL_QUESTION_TYPES=multi-session \
LONGMEMEVAL_DATA_FILE=third_party/longmemeval-official/data/longmemeval_s_cleaned.json \
LONGMEMEVAL_METRIC_MODEL=gpt-4o \
LONGMEMEVAL_JUDGE_API_KEY=sk-... \
LONGMEMEVAL_JUDGE_BASE_URL=https://your-gateway/v1 \
AGENT_DRIVER=adamem_episodic \
./AgentEnv/bin/python -m benchmarks.longmemeval.run
```

> Any run with `LONGMEMEVAL_MAX_SESSIONS>0` is truncated and marked
> non-official — use it only for debugging, never for reported numbers.
