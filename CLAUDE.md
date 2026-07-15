# research-helper

## Codebase navigation — USE THE KNOWLEDGE GRAPH FIRST

This repo is indexed in the codebase-memory MCP server
(project: `home-sujalmainali727-Projects-Major-Project-research-helper`).

Before grepping or reading files to understand structure, query the graph:
- `search_graph` — find symbols, modules, classes, functions
- `get_architecture` — structural overviews of the codebase
- `trace_path` — call/dependency chains between components

Only fall back to Grep/Glob for content the graph doesn't hold (string
literals, config values, comments). Re-run `index_repository` after large
refactors to keep the graph fresh.

## Environments (important — two interpreters)

- **Main interpreter:** `./AgentEnv/bin/python` — research-helper deps
  (langchain, src/, benchmarks/ main-process code). Use this for
  `py_compile`, tests, and `python -m benchmarks.<name>.run`.
- **Isolated ToolSandbox interpreter:** `./ToolSandboxEnv/bin/python` —
  pinned to polars 0.20 / numpy 1.26, incompatible with AgentEnv.

## Architecture rules (enforced)

- `tool_sandbox` is importable ONLY in `benchmarks/toolsandbox/worker.py`,
  which runs as a subprocess under ToolSandboxEnv. The main process never
  imports it.
- The main process talks to the worker over stdio JSON-lines via
  `benchmarks/toolsandbox/official_bridge.py` (`ToolSandboxClient`). The
  worker emits `inference_request` lines during rollout; the client services
  them through `src.llm` and writes back `inference_response`; rollout ends
  with a terminal `result`/`error` line.
- `src/*` must never import `benchmarks/*`.
- Benchmarks use the shared common models in `benchmarks/common/models.py`
  (`Episode`, `RunResult`, `EnvironmentState`, `Trajectory`, ...).

## Configuration

Settings load from `.env` (see `.env.example`). Key ToolSandbox vars:
`TOOLSANDBOX_PYTHON` (worker interpreter), `TOOLSANDBOX_OFFICIAL_ROOT`,
`TOOLSANDBOX_SCENARIO`, `TOOLSANDBOX_RUN_MODE`, `TOOLSANDBOX_MAX_TURNS`.

## Quick commands

```bash
# Compile check (main env)
./AgentEnv/bin/python -m py_compile benchmarks/toolsandbox/*.py

# Worker smoke test (ToolSandbox env)
PYTHONPATH="$PWD" ./ToolSandboxEnv/bin/python -m benchmarks.toolsandbox.worker list-scenarios

# Run the ToolSandbox benchmark (main env)
PYTHONPATH="$PWD" ./AgentEnv/bin/python -m benchmarks.toolsandbox.run

# Validation tests (main env)
PYTHONPATH="$PWD" ./AgentEnv/bin/python benchmarks/test_implementation.py
```
