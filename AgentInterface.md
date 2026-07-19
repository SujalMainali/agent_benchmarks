# Agent Interface Contract (`AgentRuntime`)

How to plug **any agent** into this repo's benchmarks (LoCoMo, LongMemEval,
BFCL, ToolSandbox). The benchmarks never talk to an agent directly — they talk
to an **`AgentRuntime`** wrapper. If your agent's runtime honors the contract
below, all benchmarks can drive and score it.

> **Construction is handled separately — see [DriverInterface.md](DriverInterface.md).**
> Benchmarks obtain runtimes through an `AgentDriver`
> (`create_runtime(RuntimeSpec) -> AgentRuntime`), selected via the
> `AGENT_DRIVER` env var. To benchmark a new agent: leave the agent untouched,
> write one driver class, point `AGENT_DRIVER` at it. This document specifies
> the *behavior* the returned runtime must honor.

- Abstract interface: [benchmarks/common/interfaces.py](benchmarks/common/interfaces.py) (`AgentRuntime`, line 121)
- Driver interface: [benchmarks/common/driver.py](benchmarks/common/driver.py) (`AgentDriver`, `RuntimeSpec`)
- Shared data models: [benchmarks/common/models.py](benchmarks/common/models.py)
- Reference implementation: [src/runtime.py](src/runtime.py) (`ResearchHelperAgentRuntime`), built by [drivers/research_helper.py](drivers/research_helper.py)

```
Benchmark loader ─▶ Episode ─▶ adapter (system prompt / tools / context)
                                   │
                                   ▼
                    runtime.reset(episode, initial_state)
                    action = runtime.act(observation)      ◀── may repeat (multi-turn)
                    runtime.get_trajectory() / get_raw_messages()
                                   │
                                   ▼
                    RunResult ─▶ evaluator ─▶ metrics/report
```

---

## 1. Required methods

Your runtime must subclass (or duck-type) `benchmarks.common.interfaces.AgentRuntime`
and implement:

### 1.1 `reset(episode: Episode, initial_state: EnvironmentState) -> None`

Called **once per episode**, before any `act()`. Must:

1. **Discard all state from previous episodes** — memory, trajectory, message
   log. Each episode must start clean (some benchmarks reuse one runtime
   object across samples; LoCoMo does).
2. **Ingest `initial_state.messages` as prior conversation context.** This is
   a `List[langchain_core.messages.BaseMessage]` (`SystemMessage` /
   `HumanMessage` / `AIMessage` / `ToolMessage`). The agent must treat these
   as history it has "already seen" (e.g. seed them into its memory), NOT as
   a new user request. LoCoMo puts the whole long conversation here;
   ToolSandbox puts earlier turns of the scenario; BFCL usually passes few
   or none.
3. Return nothing. Raising here fails the episode.

**Inputs you may read:**

| Param | Type | Relevant fields |
|---|---|---|
| `episode` | `Episode` | `episode_id: str`, `question: str`, `gold_answer: str`, `metadata: dict`, `task.mode: str` |
| `initial_state` | `EnvironmentState` | `episode_id: str`, `messages: List[BaseMessage]`, `allowed_tools: List[str]`, `metadata: dict` |

Do **not** answer `episode.question` during `reset` — the question arrives via
`act()`.

### 1.2 `act(observation: Observation) -> Action`

Called **one or more times** per episode. Each call is one full user turn:
receive the user text, run your complete internal reasoning/tool loop
**synchronously**, and return the final answer for that turn.

**Input — `Observation`:**

| Field | Type | Meaning |
|---|---|---|
| `episode_id` | `str` | Current episode id. |
| `text` | `str` | **The user utterance to answer.** This is the primary input. |
| `messages` | `List[BaseMessage]` | Optional extra context (LoCoMo/BFCL set it; ToolSandbox does not — context came via `reset`). |
| `available_tools` | `List[str]` | Tool names allowed this turn (informational; the actual tool objects were bound at construction, see §3). |
| `metadata` | `dict` | Includes `"benchmark_mode"`: `"locomo"` \| `"bfcl"` \| `"tool_sandbox"`. |

**Output — `Action`.** The benchmarks consume exactly one field for scoring:

| Field | Type | Required value |
|---|---|---|
| `action_type` | `str` | Use `"final_answer"`. |
| `text` | `str` | **The agent's final answer text for this turn.** This becomes `RunResult.predicted_answer` (LoCoMo/BFCL) and, in ToolSandbox, the AGENT→USER message the official engine records and the user simulator reacts to. Must be plain text — no tool-call syntax. |
| `metadata` | `dict` | Optional; the reference impl stores `episode_id` and `benchmark_mode`. |

**Behavioral requirements inside `act()`:**

- **Run the whole tool loop inside the call.** If the model requests a tool,
  execute it (see §2), feed the result back, and iterate until you have a
  final text answer (bounded by your own max-steps budget). `act()` must not
  return "pending tool call" states.
- **Never crash on tool errors.** A tool returning an error string (e.g.
  `"Error: ..."`, `"TransientToolError: ... Please retry."`) is normal
  benchmark feedback — give it to the model as the tool result so it can
  retry or work around it. ToolSandbox *deliberately injects* such transient
  faults to score recovery.
- **Persist memory across `act()` calls within one episode.** ToolSandbox
  scenarios have multiple user turns; turn 2 must remember turn 1. (Reset
  only on `reset()`.)
- Exceptions escaping `act()` are caught by the runners and recorded as a
  failed episode (`error` in `RunResult`) — allowed, but you get scored 0.

### 1.3 `get_trajectory() -> Trajectory`

Called after (or between) `act()` calls. Returns a
`benchmarks.common.models.Trajectory` — a dataclass with an ordered
`events: List[TrajectoryEvent]`.

You must append `TrajectoryEvent`s as the turn unfolds. **Minimum event
contract the benchmarks depend on:**

| Requirement | Who needs it |
|---|---|
| One event with `event_type="model"` per LLM response, carrying that response's tool calls in `tool_calls: List[ToolEvent]` (empty list if the model produced plain text). | **BFCL scoring reads the FIRST `"model"` event that has non-empty `tool_calls`** and decodes those calls as the model's answer ([benchmarks/bfcl/runtime_bridge.py](benchmarks/bfcl/runtime_bridge.py) `_model_tool_events`). Miss this shape and BFCL scores 0. |
| Each `ToolEvent` filled as `ToolEvent(tool_name=str, arguments=dict, result=str)`. Arguments must be the **parsed dict**, not a JSON string. | BFCL AST/execute decoding; ToolSandbox `runtime_trajectory` metadata. |
| Events must be plain dataclasses (serializable via `__dict__`/`asdict`). Don't stash live objects in them. | ToolSandbox serializes `event.__dict__` into report metadata; LoCoMo `asdict()`s events. |
| `event_type` values used by the reference impl: `"user"`, `"model"`, `"tools"`, `"final"`, `"done"`. Only `"model"` (above) is load-bearing for scoring; the rest are for reports/debugging. | reports |

`TrajectoryEvent` fields you should populate: `event_type`, `turn_number`,
`user_input` (on the user event), `agent_message` (text of that step),
`tool_calls`, and optionally `metadata`. Everything else can stay default.

### 1.4 `get_metrics() -> Dict[str, Any]` *(optional)*

Has a default (`{}`) in the ABC. Purely informational.

### 1.5 `get_raw_messages() -> List[Dict[str, Any]]` *(strongly recommended)*

Not in the ABC, but:

- **LoCoMo calls it unconditionally** ([benchmarks/locomo/runner.py](benchmarks/locomo/runner.py:76)) — without it LoCoMo crashes.
- BFCL calls it behind a `hasattr` guard and mines it for `usage_metadata`
  token accounting.

Return the full message log as plain dicts:
`{"role": "system"|"user"|"assistant"|"tool", "content": str, "metadata": dict}`.
Include provider `usage_metadata` in `metadata` when available if you want
token metrics.

---

## 2. Tool execution contract

Tools are handed to your **agent at construction time** (not per-`act()`) as
**langchain `StructuredTool` objects**: each has `.name`, `.description`,
`.args_schema` (a JSON-schema dict), and is executed via
`tool.invoke(arguments_dict) -> str`.

Your agent loop must:

1. Advertise exactly those tools to the model (names + schemas).
2. When the model requests a call, run `str(tool.invoke(args))`
   **synchronously** and feed the string back as the tool result
   (`ToolMessage` or your equivalent).
3. Treat any returned string as data — including error strings. Never raise.
4. Unknown tool name / non-dict args → feed back an error string (reference
   impl: `f"Error: unknown tool '{name}'."`), don't crash.

**Why synchronous matters (ToolSandbox):** in runtime mode each
`tool.invoke()` is a proxy that tunnels over stdio to a worker process holding
the sandbox world state, *while the worker blocks waiting inside your turn*
([benchmarks/toolsandbox/runtime_bridge.py](benchmarks/toolsandbox/runtime_bridge.py)).
Calls must happen sequentially on the calling thread, only between
`reset`/`act` entry and `act` return. No background threads, no deferred
execution, no caching of tool callables beyond the turn.

The tool result string may be:
- a normal result (repr of the sandbox tool's return value),
- an execution error from the sandbox (agent should adapt), or
- an injected transient fault: `"Error: TransientToolError: the tool backend
  timed out. The call had no effect. Please retry."` — the correct behavior
  is to **retry the same call**; recovery is scored
  (`fault_recovery_rate` in [benchmarks/toolsandbox/metrics.py](benchmarks/toolsandbox/metrics.py)).

---

## 3. How each benchmark constructs the runtime

Construction is uniform now: every benchmark calls
`driver.create_runtime(RuntimeSpec(...))` on the driver resolved from
`AGENT_DRIVER` (see [DriverInterface.md](DriverInterface.md)). What differs is
the **binding** each benchmark requests and how often it asks:

| Benchmark | Frequency | RuntimeSpec shape |
|---|---|---|
| LoCoMo | once per run ([benchmarks/locomo/runner.py](benchmarks/locomo/runner.py)) | `benchmark="locomo"`, prompt-mode system prompt, `tools=None` (agent's own tools) |
| LongMemEval | once per run ([benchmarks/longmemeval/runner.py](benchmarks/longmemeval/runner.py)) | `benchmark="longmemeval"`, LongMemEval system prompt, `tools=None`, `allow_tools` from config |
| BFCL | fresh per entry ([benchmarks/bfcl/runner.py](benchmarks/bfcl/runner.py) `_build_runtime`) | `benchmark="bfcl"`, category system prompt, `tools=<entry's functions>` |
| ToolSandbox | fresh per scenario, lazily on first worker turn ([benchmarks/toolsandbox/runtime_bridge.py](benchmarks/toolsandbox/runtime_bridge.py)) | `benchmark="tool_sandbox"`, scenario system prompt, `tools=<sandbox proxy tools>` |

The LoCoMo/LongMemEval runners also still accept a prebuilt `AgentRuntime`
directly (duck-typed on `reset`/`act`) for tests and ad-hoc use.

A driver may legitimately be **QA-only**: an agent with no tool support can
serve just LoCoMo and LongMemEval and raise on any other benchmark (or on a
spec carrying `tools`). The AdaMem memory agents do exactly this — see
[AdaMemDrivers.md](AdaMemDrivers.md) — failing fast with a clear message rather
than pretending to support tools.

`spec.system_prompt` is load-bearing: benchmarks build scenario-specific
system prompts (ToolSandbox scenario instructions, BFCL category prompts) and
your agent must actually use the override, not merely store it.

`spec.format_contract` (optional) is a **separate, additive** channel: the
scorer's answer-format requirements (terseness, abstention phrasing, date
format) written to be architecture-neutral — valid for a full-context, RAG,
or state-rewriting agent alike. Unlike `system_prompt` (full replace), a
driver SHOULD deliver it *additively* — append it to the agent's own prompt
so the agent's identity/architecture guidance survives. It defaults to `None`
(the benchmark authored none). The AdaMem drivers use it for their default
`merge` prompt mode; see [Analysis.md](Analysis.md) §6.2 and
[PromptPlan.md](PromptPlan.md).

Your driver may build the agent's LLM however it likes. The reference driver
uses the project-level provider factory (`src.llm.build_provider`, configured
by `LLM_PROVIDER`, `OPENAI_MODEL_ID`, … in `.env`); an external agent's driver
can bring its own client. One extra: ToolSandbox's `llm_proxy` agent mode
(official baseline agent, not your runtime) needs a bare LLM — a driver used
in that mode must expose an `llm` attribute with
`invoke(messages, tools=None) -> response(.text, .tool_calls)`.

---

## 4. Per-benchmark call sequences (what your runtime will observe)

**LoCoMo** (long-context QA, single shot):
```
reset(episode, state(messages=<entire long conversation>))
act(obs(text=<final question>))            # exactly once
get_trajectory(); get_raw_messages()       # required
```

**BFCL** (function-calling, single shot, fresh runtime per entry):
```
runtime = factory(system_prompt, tools)    # entry-specific tool set
reset(episode, state(messages=<category context>, allowed_tools=[...]))
act(obs(text=<question>, metadata={"benchmark_mode": "bfcl"}))
get_trajectory()                           # first "model" event w/ tool_calls is scored
```
Note: in BFCL the *scored output* is the tool calls in the trajectory, not
`action.text`. Emitting correct `ToolEvent`s is the whole game.

**ToolSandbox** (stateful, multi-turn, proxied tools, optional faults):
```
session created per scenario
# first user turn (from worker):
reset(episode, state(messages=<turns so far, minus current user msg>))
act(obs(text=<user turn 1>, metadata={"benchmark_mode": "tool_sandbox"}))
    └─ each tool.invoke() blocks-and-tunnels to the sandbox worker
# each later user turn:
act(obs(text=<user turn N>))               # same runtime — memory must persist
get_trajectory()                           # serialized into report metadata
```
Scoring is milestone/minefield-based on the **worker-side** conversation and
world state — your `action.text` and tool calls are what create that record.

---

## 5. Compliance checklist

- [ ] `reset()` clears all prior-episode state and seeds `initial_state.messages` as history
- [ ] `act()` returns `Action(action_type="final_answer", text=<plain answer>)`
- [ ] Full tool loop runs synchronously inside `act()`; tools called via `tool.invoke(dict) -> str`
- [ ] Tool error strings (incl. `TransientToolError`) are fed back to the model, never raised
- [ ] Memory persists across `act()` calls within an episode; cleared only by `reset()`
- [ ] `get_trajectory()` emits a `"model"` event with parsed-dict `ToolEvent`s for every LLM response
- [ ] Trajectory events are plain serializable dataclasses
- [ ] `get_raw_messages()` implemented (list of role/content/metadata dicts)
- [ ] `system_prompt_override` actually replaces the agent's system prompt
- [ ] No background threads / async tool execution / cross-turn tool caching
