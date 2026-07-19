# LoCoMo — Environment Configuration

How to configure and target a LoCoMo run. Every field below is read by
[benchmarks/locomo/config.py](benchmarks/locomo/config.py) from `.env` (loaded
with `override=True`, so `.env` wins over the shell). Values are consumed in
[benchmarks/locomo/run.py](benchmarks/locomo/run.py).

Run it with:

```bash
./AgentEnv/bin/python -m benchmarks.locomo.run
```

The agent under test is chosen by **`AGENT_DRIVER`** (see
[DriverInterface.md](DriverInterface.md) / [AdaMemDrivers.md](AdaMemDrivers.md)),
not by any `LOCOMO_*` var.

---

## Config fields

| Env var | Type | Default | Effect |
|---|---|---|---|
| `LOCOMO_DATA_FILE` | path | `data/locomo/locomo10.jsonl` | Dataset to load. `.json` → `load_episodes_from_json`; anything else → JSONL reader. The repo ships `data/locomo/locomo10.json` (10 full conversations) and `data/locomo/demo.jsonl` (tiny smoke sample). |
| `LOCOMO_OUTPUT_DIR` | path | `results/locomo` | Root under which the immutable run directory is written (see ResultFormat.md). Actual results also bucket by `MEMORY_ARCHITECTURE`. |
| `LOCOMO_OFFICIAL_ROOT` | path | `third_party/locomo-official` | Vendored official LoCoMo repo, used by the official evaluator. |
| `LOCOMO_USE_OFFICIAL_EVAL` | bool | `true` | `true` → score with the official LoCoMo evaluator (`evaluate_batch_official`). `false` → local heuristic evaluator. |
| `LOCOMO_ALLOW_TOOLS` | bool | `false` | Passed into `RuntimeSpec.allow_tools`. `false` disables the agent tool loop (pure memory QA). `true` lets a tool-capable agent call its own tools. AdaMem agents ignore this (no tools). |
| `LOCOMO_PROMPT_MODE` | enum | `qa` | Selects the benchmark system prompt (see table below). |
| `LOCOMO_RUN_MODE` | enum | `single` | `single` runs one sample (all its QA items), `batch` runs many/all samples. Any other value → error + exit. |
| `LOCOMO_SAMPLE_ID` | string | *(unset)* | **single mode only.** Which sample to target (e.g. `conv-30`). Matches on `source_sample_id` (whole sample) OR an exact `episode_id` (one QA item, `conv-30_7`). Unset → the first sample only. |
| `LOCOMO_QUESTION_COUNT` | int | `0` (all) | **single mode only.** Caps QA items run for the targeted sample. `0`/unset = all questions of that sample. Only applies when `LOCOMO_SAMPLE_ID` is set. |
| `LOCOMO_MAX_SAMPLES` | int | `0` (all) | **batch mode only.** Caps number of samples. `0`/unset = all. |
| `LOCOMO_CREATE_DEMO_DATA` | bool | `true` | Only used by [benchmarks/locomo/example.py](benchmarks/locomo/example.py): create demo data before loading. No effect on `run.py`. |
| `LOCOMO_VERBOSE` | bool | `true` | Print per-sample progress during batch runs. |

Booleans are truthy for `1/true/yes/y` (case-insensitive); everything else is false.

### `LOCOMO_PROMPT_MODE` values

Resolved by `_select_locomo_prompt` in
[benchmarks/locomo/run.py](benchmarks/locomo/run.py); prompt bodies live in
[benchmarks/locomo/prompts.py](benchmarks/locomo/prompts.py). This becomes
`RuntimeSpec.system_prompt`.

| Value | Prompt | Use it for |
|---|---|---|
| `qa` (default) | `LOCOMO_QA_PROMPT` | Standard QA: answer from the conversation, cite context, tools only as fallback. |
| `strict` | `LOCOMO_STRICT_FORMAT_PROMPT` | One-sentence terse answers (tighter token-overlap scoring). |
| `rag` | `LOCOMO_EVIDENCE_AWARE_PROMPT` | Evidence-aware: combine conversation history with provided evidence. |
| *anything else* | `LOCOMO_SYSTEM_PROMPT` | The general system prompt (fallback for any unrecognized value). |

> Note for AdaMem drivers: with `ADAMEM_PROMPT_MODE=native` (default) the agent
> keeps its own memory-format QA prompt and this benchmark prompt is recorded
> but not applied. Set `ADAMEM_PROMPT_MODE=benchmark` to force it.

---

## Listing available samples / question ids

There's no dedicated CLI flag; enumerate straight from the dataset file.

**Sample ids (and per-sample QA counts):**

```bash
./AgentEnv/bin/python - <<'PY'
import json
data = json.load(open("data/locomo/locomo10.json"))
samples = data.get("samples", data) if isinstance(data, dict) else data
print(f"{len(samples)} samples:")
for s in samples:
    print(f"  {s.get('sample_id'):10}  {len(s.get('qa', []))} QA items")
PY
```

Current `locomo10.json`: `conv-26, conv-30, conv-41, conv-42, conv-43,
conv-44, conv-47, conv-48, conv-49, conv-50` (105–260 QA items each).

**Individual QA item ids** for a sample (the `episode_id` form
`<sample_id>_<index>` you can pass to `LOCOMO_SAMPLE_ID` to target one item):

```bash
./AgentEnv/bin/python - <<'PY'
from benchmarks.locomo.loader import LoCoMoLoader
eps = LoCoMoLoader().load_episodes_from_json("data/locomo/locomo10.json")
for e in eps:
    if e.metadata.get("source_sample_id") == "conv-30":   # <-- change target
        print(e.episode_id, "|", e.question[:70])
PY
```

**Question categories** present in the dataset (LoCoMo encodes category as an
integer per QA item):

```bash
./AgentEnv/bin/python - <<'PY'
import json
from collections import Counter
data = json.load(open("data/locomo/locomo10.json"))
samples = data.get("samples", data) if isinstance(data, dict) else data
c = Counter(q.get("category") for s in samples for q in s.get("qa", []))
print("category -> count:", dict(sorted(c.items(), key=lambda x: str(x[0]))))
PY
```

---

## Common recipes

Run one whole sample with the official evaluator:

```bash
LOCOMO_RUN_MODE=single LOCOMO_SAMPLE_ID=conv-30 \
LOCOMO_DATA_FILE=data/locomo/locomo10.json \
AGENT_DRIVER=adamem_flat_rag \
./AgentEnv/bin/python -m benchmarks.locomo.run
```

Quick 5-question smoke of one sample:

```bash
LOCOMO_RUN_MODE=single LOCOMO_SAMPLE_ID=conv-30 LOCOMO_QUESTION_COUNT=5 \
LOCOMO_DATA_FILE=data/locomo/locomo10.json \
./AgentEnv/bin/python -m benchmarks.locomo.run
```

Target a single QA item:

```bash
LOCOMO_RUN_MODE=single LOCOMO_SAMPLE_ID=conv-30_7 \
LOCOMO_DATA_FILE=data/locomo/locomo10.json \
./AgentEnv/bin/python -m benchmarks.locomo.run
```

Full batch, capped at 3 samples, local eval, verbose:

```bash
LOCOMO_RUN_MODE=batch LOCOMO_MAX_SAMPLES=3 LOCOMO_USE_OFFICIAL_EVAL=false \
LOCOMO_DATA_FILE=data/locomo/locomo10.json \
./AgentEnv/bin/python -m benchmarks.locomo.run
```
