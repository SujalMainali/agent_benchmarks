# BFCL — Environment Configuration

How to configure and target a BFCL (Berkeley Function-Calling Leaderboard) run.
Every field is read by [benchmarks/bfcl/config.py](benchmarks/bfcl/config.py)
from `.env` (loaded with `override=True`) and consumed in
[benchmarks/bfcl/run.py](benchmarks/bfcl/run.py).

Run it with:

```bash
./AgentEnv/bin/python -m benchmarks.bfcl.run
```

The agent under test is chosen by **`AGENT_DRIVER`** (see
[DriverInterface.md](DriverInterface.md)).

> **Tool support required.** BFCL is function-calling: each entry advertises
> tools and the driver must build a tool-capable runtime. The reference
> `research_helper` driver works. The AdaMem memory agents do **not** support
> tools — pointing `AGENT_DRIVER=adamem_*` at BFCL fails fast with
> `UnsupportedBenchmarkError`.

---

## Config fields

| Env var | Type | Default | Effect |
|---|---|---|---|
| `BFCL_OFFICIAL_ROOT` | path | `third_party/bfcl-official` | Vendored official BFCL repo. Datasets, category mapping, and checkers all load from here (`bootstrap_official` puts it on `sys.path`). |
| `BFCL_TEST_CATEGORY` | csv | `simple_python` | Which categories/collections to run. Accepts collection names (`single_turn`, `live`, …) and individual categories (`simple_python`, `multiple`, …), comma-separated. Resolved via the official `TEST_COLLECTION_MAPPING` + `ALL_CATEGORIES`. Unknown names are warned and skipped. See values below. |
| `BFCL_RUN_IDS` | csv | *(unset)* | Exact test-entry ids to run (e.g. `simple_python_12`). When set, only entries whose `id` matches are kept — this is how you target individual cases. Applies within the selected categories. |
| `BFCL_MAX_SAMPLES` | int | `0` (all) | Cap entries **per category**. `0`/unset = no cap. |
| `BFCL_MAX_TOOL_STEPS` | int | `1` | Agent tool-loop budget per entry. Single-turn BFCL expects exactly one model call — **keep at 1** for faithful scoring. |
| `BFCL_OUTPUT_DIR` | path | `results/bfcl` | Root of the immutable run dir. |
| `BFCL_CHECKER_MODEL_NAME` | string | `gpt-4o-2024-11-20-FC` | Registered BFCL model name used as the checker persona. Controls only official function-name normalization; must exist in the official `MODEL_CONFIG_MAPPING`. Not an LLM you pay for here. |
| `BFCL_INCLUDE_INPUT_LOG` | bool | `false` | Include the fully-transformed model input in each entry's inference log (official `--include-input-log`). |
| `BFCL_VERBOSE` | bool | `true` | Print progress to the terminal. |

Booleans are truthy for `1/true/yes/y`.

### `BFCL_TEST_CATEGORY` values

**Collections** (each expands to several categories via
`TEST_COLLECTION_MAPPING`):

| Collection | # categories |
|---|---|
| `all` | 23 |
| `all_scoring` | 22 |
| `single_turn` | 13 |
| `multi_turn` | 4 |
| `live` | 6 |
| `non_live` | 7 |
| `python` | 11 |
| `non_python` | 2 |
| `memory` | 3 |
| `web_search` | 2 |
| `agentic` | 5 |

**Individual categories** (`ALL_CATEGORIES`, 23 total):

`simple_python, simple_java, simple_javascript, multiple, parallel,
parallel_multiple, irrelevance, live_simple, live_multiple, live_parallel,
live_parallel_multiple, live_irrelevance, live_relevance, multi_turn_base,
multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context,
memory_kv, memory_vector, memory_rec_sum, web_search_base,
web_search_no_snippet, format_sensitivity`

> **Runnable vs deferred.** This integration scores **single-turn** categories
> only (AST + relevance). Multi-turn / agentic / format-sensitivity categories
> need the official stateful execution loop and are **skipped with a notice**
> (see `_resolve_categories` in [benchmarks/bfcl/run.py](benchmarks/bfcl/run.py)).
> If your selection resolves to *only* deferred categories, the run exits with
> "no runnable BFCL test categories selected."
>
> - **Runnable now:** `simple_python, simple_java, simple_javascript, multiple,
>   parallel, parallel_multiple, irrelevance, live_simple, live_multiple,
>   live_parallel, live_parallel_multiple, live_irrelevance, live_relevance`
>   (i.e. the `single_turn` collection).
> - **Deferred (skipped):** all `multi_turn_*`, `memory_*`, `web_search_*`,
>   `format_sensitivity`.

---

## Listing available categories / entry ids

**Collections and categories** (authoritative, from the official constants):

```bash
./AgentEnv/bin/python - <<'PY'
from benchmarks.bfcl.official import bootstrap_official
bootstrap_official()
from bfcl_eval.constants.category_mapping import ALL_CATEGORIES, TEST_COLLECTION_MAPPING
print("COLLECTIONS:")
for k, v in TEST_COLLECTION_MAPPING.items():
    print(f"  {k:14} -> {len(v)} categories")
print("\nALL_CATEGORIES:")
print(" ", ", ".join(ALL_CATEGORIES))
PY
```

**Which categories actually run** in this integration (runnable vs deferred):

```bash
./AgentEnv/bin/python - <<'PY'
from benchmarks.bfcl.official import bootstrap_official
bootstrap_official()
from bfcl_eval.constants.category_mapping import ALL_CATEGORIES
from bfcl_eval.utils import is_agentic, is_multi_turn, is_format_sensitivity
run, skip = [], []
for c in ALL_CATEGORIES:
    (skip if (is_multi_turn(c) or is_agentic(c) or is_format_sensitivity(c)) else run).append(c)
print("RUNNABLE:", run)
print("DEFERRED:", skip)
PY
```

**Entry ids within a category** (for `BFCL_RUN_IDS`):

```bash
./AgentEnv/bin/python - <<'PY'
from benchmarks.bfcl.loader import BFCLLoader
eps = BFCLLoader().load_category("simple_python")   # <-- change category
print(f"{len(eps)} entries")
for e in eps[:20]:
    print(" ", e.episode_id, "|", e.question[:60])
PY
```

You can also see the raw dataset files directly:

```bash
ls third_party/bfcl-official/bfcl_eval/data/BFCL_v4_*.json
```

---

## Common recipes

Default smoke — a few `simple_python` entries:

```bash
BFCL_TEST_CATEGORY=simple_python BFCL_MAX_SAMPLES=5 \
AGENT_DRIVER=research_helper \
./AgentEnv/bin/python -m benchmarks.bfcl.run
```

Target specific entries by id:

```bash
BFCL_TEST_CATEGORY=simple_python \
BFCL_RUN_IDS=simple_python_0,simple_python_1 \
AGENT_DRIVER=research_helper \
./AgentEnv/bin/python -m benchmarks.bfcl.run
```

The whole runnable single-turn suite:

```bash
BFCL_TEST_CATEGORY=single_turn \
AGENT_DRIVER=research_helper \
./AgentEnv/bin/python -m benchmarks.bfcl.run
```

Multiple explicit categories, capped, with input logging:

```bash
BFCL_TEST_CATEGORY=simple_python,multiple,live_simple \
BFCL_MAX_SAMPLES=10 BFCL_INCLUDE_INPUT_LOG=true \
AGENT_DRIVER=research_helper \
./AgentEnv/bin/python -m benchmarks.bfcl.run
```
