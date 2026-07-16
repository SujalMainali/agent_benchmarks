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

## Active work — BFCL integration

An in-progress task integrates the official BFCL benchmark into
`benchmarks/bfcl/`. Progress, design decisions, and remaining steps are
tracked in [checkpoint.md](checkpoint.md); vendored-repo analysis notes are
in [BFCL.md](BFCL.md). Read both before touching `benchmarks/bfcl/`.

## Environments (important — two interpreters)

- **Main interpreter:** `./AgentEnv/bin/python` — research-helper deps
  (langchain, src/, benchmarks/ main-process code). Use this for
  `py_compile`, tests, and `python -m benchmarks.<name>.run`.
- **Isolated ToolSandbox interpreter:** `./ToolSandboxEnv/bin/python` —
  pinned to polars 0.20 / numpy 1.26, incompatible with AgentEnv.
