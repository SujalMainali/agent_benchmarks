# BFCL (Berkeley Function Calling Leaderboard) — vendored repo notes

Notes from the Phase-0 read-only analysis of `third_party/bfcl-official/`,
gathered while integrating BFCL as `benchmarks/bfcl/`. This file documents how
the official code works and which pieces we reuse vs. bridge.

## Repo layout

```
third_party/bfcl-official/
    README.md               # official pipeline docs
    LOG_GUIDE.md            # inference-log format
    TEST_CATEGORIES.md      # category groups + individual categories
    CONTRIBUTING.md
    pyproject.toml          # package `bfcl_eval`, heavy dependency list
    bfcl_eval/
        constants/          # category_mapping.py, enums.py, model_config.py, ...
        data/               # ALL datasets (BFCL_v4_*.json) + possible_answer/
        eval_checker/       # ast_eval/, multi_turn_eval/, agentic_eval/, eval_runner.py
        model_handler/      # base_handler.py + per-vendor handlers
        utils.py            # dataset loading, category predicates, id helpers
```

## Official execution pipeline

```
Dataset (bfcl_eval/data/BFCL_v4_<category>.json, JSON-lines)
  -> Handler (subclass of model_handler/base_handler.py:BaseHandler)
       inference() -> model_responses ("raw result" written to result file)
  -> decode_ast(result)      -> [{func_name: {param: value}}]   -> AST checker
  -> decode_execute(result)  -> ["func(param=value)"]           -> executable checker
  -> eval_runner.py computes accuracy, writes score/ files + leaderboard CSVs
```

### Dataset entry shape (single-turn)

Each line of `data/BFCL_v4_<category>.json` is one JSON object:
- `id` — e.g. `simple_python_0`; category is recovered by
  `utils.extract_test_category_from_id` (rsplit on last `_`).
- `question` — list of turns; each turn is a list of role dicts
  (`{"role": "user"|"system"|"assistant", "content": ...}`). Single-turn
  entries have exactly one turn: `question[0]`.
- `function` — list of function docs in BFCL/Gorilla JSON-schema style
  (NOT OpenAI tool format; conversion happens in handlers via
  `model_handler/utils.convert_to_tool`).
- Multi-turn entries add `initial_config`, `involved_classes`,
  `missed_function`, etc.

Ground truth lives in `data/possible_answer/BFCL_v4_<category>.json`
(`{"id": ..., "ground_truth": [...]}`), loaded by
`utils.load_ground_truth_entry(test_category)`. Relevance/irrelevance
categories have NO ground-truth files.

### BaseHandler contract (`model_handler/base_handler.py`)

Key methods a handler exposes (what our runtime bridge must mirror):
- `inference(test_entry, include_input_log, exclude_state_log)` — dispatches
  to single/multi-turn × FC/prompting variants. Returns
  `(model_responses, metadata)`; metadata has `input_token_count`,
  `output_token_count`, `latency`, optional `inference_log`.
- `decode_ast(result, language, has_tool_call_tag)` — raw result →
  `[{name: {param: value}}]` for the AST checker.
- `decode_execute(result, has_tool_call_tag)` — raw result →
  `["func(arg=value)"]` executable strings.
- For FC models (e.g. `OpenAICompletionsHandler`), the stored raw result is
  a list like `[{func_name: '{"json": "args"}'}]` (arguments as a JSON
  string), and `decode_ast` simply `json.loads` the values;
  `decode_execute` uses `model_handler/utils.convert_to_function_call`.

The result files handlers write are JSON-lines:
`{"id": ..., "result": <model_responses>, ...metadata}` under
`result/<model>/<group>/BFCL_v4_<category>_result.json`.

### Evaluator (`eval_checker/`)

- `eval_runner.py` orchestrates per-category scoring. The per-entry helpers we
  reuse directly:
  - `_evaluate_single_ast_entry(handler, index, model_result_item,
    possible_answer_item, prompt_entry, model_name, test_category, language,
    return_format, has_tool_call_tag)` → `{"valid": bool, ...error details}`.
    Internally calls `handler.decode_ast` then `ast_eval/ast_checker.ast_checker`.
  - `_evaluate_single_relevance_entry(handler, index, model_result_item,
    prompt_entry, model_name, test_category)` — relevance/irrelevance logic
    (irrelevance passes when decode fails/empty; relevance the opposite).
- `ast_eval/ast_checker.py::ast_checker(func_description, model_output,
  possible_answer, language, test_category, model_name)` — dispatches to
  simple/multiple/parallel checkers. IMPORTANT: `convert_func_name` looks up
  `MODEL_CONFIG_MAPPING[model_name]` — the `model_name` passed to the checker
  MUST be a registered BFCL model name (we pass a registered FC model, e.g.
  `gpt-4o-2024-11-20-FC`, as a "checker persona"; it only affects the
  underscore-to-dot function-name normalization for models that can't emit
  dots in function names).
- `multi_turn_eval/`, `agentic_eval/` — state-based checkers for multi-turn /
  memory / web-search categories (not wired in the first iteration of our
  integration; single-turn AST + relevance categories are).

### Categories (`constants/category_mapping.py`)

Official constants (REUSE, never hardcode):
- `VERSION_PREFIX = "BFCL_v4"` — dataset file prefix.
- `NON_LIVE_CATEGORY`, `LIVE_CATEGORY`, `MULTI_TURN_CATEGORY`,
  `MEMORY_CATEGORY`, `WEB_SEARCH_CATEGORY`, `AGENTIC_CATEGORY`,
  `SINGLE_TURN_CATEGORY`, `ALL_SCORING_CATEGORIES`, `ALL_CATEGORIES`,
  `TEST_COLLECTION_MAPPING` (e.g. `"single_turn"` → list of categories).
- `utils.py` predicates: `is_multi_turn`, `is_live`, `is_java`, `is_js`,
  `is_relevance_or_irrelevance`, `is_agentic`, `extract_test_category_from_id`...
- Languages/return formats: `constants/enums.py` → `Language`, `ReturnFormat`.

### Log format (LOG_GUIDE.md)

Inference log = list of role entries: `user`, `assistant`, `tool`,
`state_info`, `inference_input`, `handler_log`. Single-turn entries only
carry `inference_input`/`handler_log`. We mirror this shape in our per-entry
logs (`assistant` + `handler_log` entries with `model_response_decoded`).

## Import constraints in our AgentEnv (important)

`bfcl_eval` is importable from `third_party/bfcl-official` by adding that dir
to `sys.path` (no install needed — it's a plain package), BUT some modules
pull heavy/vendor deps not present in AgentEnv:

| Import | Status in AgentEnv |
|---|---|
| `bfcl_eval.constants.category_mapping` | OK |
| `bfcl_eval.utils` (dataset loading) | OK |
| `bfcl_eval.constants.enums` | OK |
| `bfcl_eval.eval_checker.ast_eval.ast_checker` | pulls `constants.model_config` → ALL vendor handlers → needs `anthropic`, `cohere`, `mistralai`, `overrides`, `google-genai`, `boto3`, `writerai`, `qwen_agent`, `tree_sitter` (java/js type converters) |
| `bfcl_eval.model_handler.utils` | needs `tree_sitter` |
| `bfcl_eval.eval_checker.eval_runner` | same heavy closure |

Consequence for the integration: the evaluator layer imports the official
checker lazily and requires those extras to be installed in AgentEnv
(`pip install anthropic cohere mistralai overrides google-genai boto3
writer-sdk qwen-agent tree_sitter==0.21.3 tree-sitter-java==0.21.0
tree-sitter-javascript==0.21.4`) — we do NOT copy/modify any official code.
`benchmarks/bfcl/official.py` centralizes the `sys.path` setup.

## How our integration maps onto this

```
benchmarks/bfcl/
    config.py         # env-driven settings (BFCL_* vars), official_root path
    official.py       # sys.path bootstrap + lazy imports of official symbols
    loader.py         # data/*.json -> Episode (raw fields preserved)
    adapter.py        # Episode -> {system_prompt, messages, tools} only
    runtime_bridge.py # ResearchRuntime.run() facade; decode_ast/decode_execute
                      #   from structured Action.tool_calls (never re-parse text)
    evaluator.py      # calls official _evaluate_single_ast_entry / relevance entry
    report.py         # official result -> common EvaluationResult / reports
    runner.py         # orchestrates episode -> bridge -> evaluator -> RunResult
    run.py            # entry point (python -m benchmarks.bfcl.run)
```

Decode formats produced by the bridge (official contracts):
- `decode_ast` → `[{tool_name: {param: value}}]`
- `decode_execute` → `["tool_name(param=value, ...)"]` via official
  `convert_to_function_call` semantics (values `repr()`-ed like the official
  FC handlers).
