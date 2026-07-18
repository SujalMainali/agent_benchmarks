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

## Existing Contracts
The AgentInterface.md contains the contracts between agent runtime interface and benchmarks