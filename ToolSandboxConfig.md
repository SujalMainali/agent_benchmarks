# ToolSandbox Configuration Guide

How to configure and run the ToolSandbox benchmark against our agent.
All configuration is done through environment variables (read from `.env` at
the repo root, or exported inline to override for a single run).

The benchmark entry point is:

```bash
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

Every setting below is parsed in
[benchmarks/toolsandbox/config.py](benchmarks/toolsandbox/config.py) by
`load_toolsandbox_settings()`. Values may include a trailing `# comment`
(stripped automatically) except booleans.

---

## Quick answer: selecting a scenario

To run **one specific scenario** (e.g. `cellular_off`), set run mode to
`single` and name the scenario:

```bash
TOOLSANDBOX_RUN_MODE=single \
TOOLSANDBOX_SCENARIO=cellular_off \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

`TOOLSANDBOX_SCENARIO` matches a scenario by its **scenario name** (the
`name="..."` in the official scenario files) or by episode id — they are the
same string. See [Finding scenario names](#finding-scenario-names) below.

To run **a group of scenarios by category** (e.g. all single-tool-call ones):

```bash
TOOLSANDBOX_RUN_MODE=batch \
TOOLSANDBOX_SCENARIO_CATEGORY=single_tool_call \
TOOLSANDBOX_MAX_SCENARIOS=5 \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

---

## Selection & run-flow settings

| Env var | Default | Values | Purpose |
|---|---|---|---|
| `TOOLSANDBOX_RUN_MODE` | `single` | `single` \| `batch` | `single` runs exactly one scenario (the first match of the filters). `batch` runs many. |
| `TOOLSANDBOX_SCENARIO` | *(empty)* | a scenario name | Targets one scenario by name/id. Empty = no name filter. In `single` mode with no name set, the first available scenario is run. |
| `TOOLSANDBOX_SCENARIO_CATEGORY` | *(empty)* | a category string (see [Categories](#scenario-categories)) | Keeps only scenarios tagged with this category. Applies in both modes. Empty = no category filter. |
| `TOOLSANDBOX_MAX_SCENARIOS` | `0` | integer; `0` = no cap | Caps how many scenarios a `batch` run executes (after filtering). Ignored in `single` mode (always 1). |

**How filtering composes** (from
[benchmarks/toolsandbox/run.py](benchmarks/toolsandbox/run.py)
`_filter_episodes`):

- `single` mode: applies `TOOLSANDBOX_SCENARIO` **and**
  `TOOLSANDBOX_SCENARIO_CATEGORY`, then takes the first result.
- `batch` mode: applies **only** `TOOLSANDBOX_SCENARIO_CATEGORY` (the exact
  scenario name is ignored in batch), then caps at `TOOLSANDBOX_MAX_SCENARIOS`.

> To run one named scenario, always use `single` mode — a scenario name in
> `batch` mode has no effect.

---

## Agent & evaluation settings

| Env var | Default | Values | Purpose |
|---|---|---|---|
| `TOOLSANDBOX_AGENT_MODE` | `runtime` | `runtime` \| `llm_proxy` | `runtime` evaluates **our** `ResearchHelperAgentRuntime` (our agent drives the whole turn; its tool calls tunnel to the worker's official sandbox). `llm_proxy` is the legacy path: the official ToolSandbox agent loop runs and only borrows our LLM for inference. |
| `TOOLSANDBOX_MAX_TOOL_STEPS` | `8` | integer | Tool-loop budget per user turn, **runtime mode only**. How many tool round-trips our agent may take before it must answer. |
| `TOOLSANDBOX_MAX_TURNS` | `20` | integer | Hard cap on conversation turns per scenario (applies in both modes). |
| `TOOLSANDBOX_USER_MODE` | `scripted` | `scripted` \| `gpt-4o` \| `gpt-4` \| `gpt-3.5` | User simulator. `scripted` replays the scenario's canned user turns (no external creds needed). The `gpt-*` values use the official OpenAI-backed user simulator and require OpenAI credentials. |
| `TOOLSANDBOX_USE_OFFICIAL_EVAL` | `true` | bool | Use the official milestone/minefield similarity scoring. |
| `TOOLSANDBOX_ALLOW_TOOLS` | *(empty)* | comma-separated tool names | Overrides the tool allow-list for every scenario. Empty = each scenario uses its own `tool_allow_list`. |

---

## Fault injection (recovery testing)

Faults make a tool call return a synthetic transient error **without touching
the sandbox**, so the agent can be scored on recovery. Fault handling lives in
the worker; metrics (`injected_fault_count`, `post_fault_retry_count`,
`fault_recovery_rate`) are computed in
[benchmarks/toolsandbox/metrics.py](benchmarks/toolsandbox/metrics.py).

| Env var | Default | Values | Purpose |
|---|---|---|---|
| `TOOLSANDBOX_FAULT_RATE` | `0.0` | float `[0.0, 1.0]` | Probability that any given tool call is answered with a transient fault instead of executing. `0.0` disables faults; `1.0` faults every call. |
| `TOOLSANDBOX_FAULT_SEED` | `13` | integer | RNG seed, for reproducible fault sequences. |

> Fault injection is only meaningful in `runtime` mode (it is applied to our
> agent's tunneled tool calls).

Example — force a fault on the first call, then let retries succeed:

```bash
TOOLSANDBOX_RUN_MODE=single \
TOOLSANDBOX_SCENARIO=cellular_off \
TOOLSANDBOX_AGENT_MODE=runtime \
TOOLSANDBOX_FAULT_RATE=0.3 \
TOOLSANDBOX_FAULT_SEED=13 \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

---

## Environment & I/O settings

| Env var | Default | Purpose |
|---|---|---|
| `TOOLSANDBOX_OFFICIAL_ROOT` | `third_party/ToolSandbox-official` | Path to the vendored official ToolSandbox repo. |
| `TOOLSANDBOX_PYTHON` | `./ToolSandboxEnv/bin/python` | Isolated interpreter used to spawn the worker (pinned polars 0.20 / numpy 1.26). **Never** the main `AgentEnv` interpreter. |
| `TOOLSANDBOX_OUTPUT_DIR` | `results/toolsandbox` | Where per-scenario + batch reports are written. |
| `TOOLSANDBOX_VERBOSE` | `true` | Print progress to the terminal during a run. |

The LLM backend itself is configured separately in `.env` via the project
settings (`LLM_PROVIDER`, `OPENAI_MODEL_ID`, `OPENAI_BASE_URL`, etc.), consumed
by `src/llm`. ToolSandbox reuses whatever provider the project is set to.

---

## Scenario categories

Set `TOOLSANDBOX_SCENARIO_CATEGORY` to one of these (each scenario may belong to
several; a scenario matches if the category is in its list). Source:
`ScenarioCategories` in
`third_party/ToolSandbox-official/tool_sandbox/common/execution_context.py`.

| Category | Meaning |
|---|---|
| `single_tool_call` | Completable with one tool call. |
| `multiple_tool_call` | Requires multiple tool calls. |
| `single_user_turn` | Solvable within a single user turn. |
| `multiple_user_turn` | Requires multiple user turns. |
| `state_dependency` | Tools depend on expected world state and error if unmet. |
| `canonicalization` | Requires surface text → canonical form. |
| `coreference` | Requires co-reference resolution. |
| `disambiguation` | Requires disambiguating between entities/tools. |
| `insufficient_information` | Cannot be completed with the given tools/prompts. |
| `no_distraction_tools` | Only the necessary tools are provided. |
| `three_distraction_tools` | 3 distractor tools added. |
| `ten_distraction_tools` | 10 distractor tools added. |
| `all_tools_available` | Every sandbox tool is provided. |
| `tool_name_scrambled` | Tool names replaced with generic ones. |
| `tool_description_scrambled` | Tool descriptions removed. |
| `arg_description_scrambled` | Argument descriptions removed. |
| `arg_name_scrambled` | Argument names scrambled. |
| `arg_type_scrambled` | Argument types scrambled. |

---

## Finding scenario names

Scenario names are the `name="..."` values in the official scenario modules:

```bash
grep -rhoE 'name="[a-z0-9_]+"' \
  third_party/ToolSandbox-official/tool_sandbox/scenarios/ | sort -u
```

Scenario files:
- `single_tool_call_scenarios.py` — e.g. `cellular_off`, `get_wifi`,
  `send_message_with_phone_number_and_content`, `convert_currency`.
- `multiple_tool_call_scenarios.py`
- `multiple_user_turn_scenarios.py`
- `insufficient_information_scenarios.py`
- `base_scenarios.py`

---

## Common recipes

**One simple scenario, our agent, no faults:**
```bash
TOOLSANDBOX_RUN_MODE=single TOOLSANDBOX_SCENARIO=cellular_off \
TOOLSANDBOX_AGENT_MODE=runtime TOOLSANDBOX_FAULT_RATE=0.0 \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

**Legacy official-loop baseline (for comparison):**
```bash
TOOLSANDBOX_RUN_MODE=single TOOLSANDBOX_SCENARIO=cellular_off \
TOOLSANDBOX_AGENT_MODE=llm_proxy \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

**Batch of single-tool scenarios, capped at 10:**
```bash
TOOLSANDBOX_RUN_MODE=batch TOOLSANDBOX_SCENARIO_CATEGORY=single_tool_call \
TOOLSANDBOX_MAX_SCENARIOS=10 \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

**Recovery stress test (every call faults once):**
```bash
TOOLSANDBOX_RUN_MODE=single TOOLSANDBOX_SCENARIO=cellular_off \
TOOLSANDBOX_AGENT_MODE=runtime TOOLSANDBOX_FAULT_RATE=1.0 \
./AgentEnv/bin/python -m benchmarks.toolsandbox.run
```

Results (per-scenario reports + a batch summary) are written under
`TOOLSANDBOX_OUTPUT_DIR`.
