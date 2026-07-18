# Agent Interface Contract (`AgentRuntime`)

How to plug **any agent** into this repo's benchmarks (LoCoMo, BFCL,
ToolSandbox). The benchmarks never talk to an agent directly — they talk to an
**`AgentRuntime`** wrapper. If your agent's runtime honors the contract below,
all three benchmarks can drive and score it.

- Abstract interface: [benchmarks/common/interfaces.py](benchmarks/common/interfaces.py) (`AgentRuntime`, line 121)
- Shared data models: [benchmarks/common/models.py](benchmarks/common/models.py)
- Reference implementation: [src/runtime.py](src/runtime.py) (`ResearchHelperAgentRuntime`)

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

The ABC covers behavior; construction is per-benchmark. Today all three
instantiate the concrete classes directly, so a new agent needs either
(a) a drop-in agent class matching `ResearchHelperAgent`'s constructor, or
(b) small edits at these three seams:

| Benchmark | Construction site | Shape |
|---|---|---|
| LoCoMo | [benchmarks/locomo/runner.py](benchmarks/locomo/runner.py:20) | `LoCoMoRunner(agent_or_runtime)` — accepts an already-built `AgentRuntime` directly. **Easiest entry point: pass your own runtime.** |
| BFCL | [benchmarks/bfcl/runner.py](benchmarks/bfcl/runner.py:92) `_build_runtime` | `runtime_factory(system_prompt: str, tools: List[StructuredTool]) -> AgentRuntime` — a fresh runtime per entry. Swap the factory to return yours. |
| ToolSandbox | [benchmarks/toolsandbox/runtime_bridge.py](benchmarks/toolsandbox/runtime_bridge.py) `ToolSandboxRuntimeSession.agent_turn_fn` | Builds the agent lazily on the first worker turn with `(llm, tools, max_tool_steps, system_prompt_override, allow_tools=True)`, wraps it in the runtime, `reset()`s with prior context, then calls `act()` once per user turn. Replace the two constructor calls to plug a different agent. |

To stay drop-in compatible with all three **without editing benchmark code**,
your agent class should accept this constructor (the shape the factories use):

```python
YourAgent(
    llm,                          # src.llm.LLMProvider (provider-agnostic)
    tools,                        # Sequence[StructuredTool]
    max_tool_steps: int = ...,    # tool-loop budget per turn
    system_prompt_override: str | None = None,  # MUST replace your default system prompt
    allow_tools: bool = True,
)
```

and your runtime should accept `YourRuntime(agent)`.

`system_prompt_override` is load-bearing: benchmarks build scenario-specific
system prompts (ToolSandbox scenario instructions, BFCL category prompts) and
your agent must actually use the override, not merely store it.

The `llm` object is a `src.llm.base.LLMProvider`:
`invoke(messages, tools=None, response_format=None) -> LLMResponse`, where
`LLMResponse` has `.text: str`, `.tool_calls: List[ToolCall(name, arguments, id)]`,
`.message` (the underlying langchain `AIMessage`). If your agent brings its own
LLM client you can ignore the passed provider — but you lose the project-level
model configuration (`LLM_PROVIDER`, `OPENAI_MODEL_ID`, … in `.env`).

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
