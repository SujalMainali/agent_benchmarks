# AI Agent Memory Evaluation Framework

A modular research framework for building, running, and evaluating LangChain-based AI agents with different memory behaviors.

The repository began as a **Research Helper Agent** and now centers on a **unified benchmark interface** for evaluating agent behavior across conversation-heavy tasks. The current implementation includes:

- a LangChain + Hugging Face research agent,
- temporary working memory with structured fact extraction,
- tool use for calculator, local document search, note lookup, and web search,
- a benchmark core with shared models and abstract interfaces,
- a **LoCoMo** benchmark integration,
- benchmark reporting, trajectory logging, and evaluation artifacts.

The long-term research goal is to provide a uniform interface for different benchmarks to apply to a model.

---

## What this repository contains

### Core agent
The current interactive agent is implemented in `src/` and uses:

- remote Hugging Face inference,
- a temporary memory layer,
- a short summary refresh loop,
- structured fact extraction,
- native tool calling,
- local and web retrieval tools.

### Benchmark framework
The `benchmarks/` package provides a reusable interface for benchmark execution and evaluation.

### Current benchmark support
- **LoCoMo** — integrated
- **ToolSandbox** — planned
- **LongMemEval** — planned
- **Custom adaptive benchmark** — planned

---

## Architecture

```text
Dataset / Scenario
    │
    ▼
Benchmark Loader
    │
    ▼
Benchmark Adapter
    │
    ▼
Benchmark Environment
    │
    ▼
Agent Runtime
    │
    ▼
Trajectory Logger
    │
    ▼
Benchmark Evaluator
    │
    ▼
Report Writer
```

The benchmark layer is independent from the agent implementation.

---

## Repository structure

```text
project/
├── app.py
├── requirements.txt
├── Readme.md
├── .env
├── .env.example
├── data/
│   ├── documents/
│   ├── notes/
│   └── locomo/
├── results/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── memory.py
│   ├── prompts.py
│   ├── runtime.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── document_search.py
│   │   ├── note_lookup.py
│   │   └── web_search.py
│   └── utils/
│       ├── __init__.py
│       └── corpus_search.py
└── benchmarks/
    ├── __init__.py
    ├── README.md
    ├── test_implementation.py
    ├── common/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── interfaces.py
    │   ├── environment.py
    │   ├── logger.py
    │   ├── evaluator_base.py
    │   └── report_writer.py
    └── locomo/
        ├── __init__.py
        ├── adapter.py
        ├── config.py
        ├── environment.py
        ├── evaluator.py
        ├── example.py
        ├── loader.py
        ├── metrics.py
        ├── official_bridge.py
        ├── prompts.py
        ├── report.py
        ├── run.py
        └── runner.py
```

---

## Current research agent

The interactive baseline agent is `src/agent.py`.

It uses:

- `ChatHuggingFace` with `HuggingFaceEndpoint`,
- `TemporaryMemory`,
- a system prompt and summary refresh loop,
- native tool calling,
- an optional benchmark prompt override,
- a tool loop with a configurable maximum number of tool steps.

### Current tools
The agent currently has four tools:

- `calculator`
- `document_search`
- `note_lookup`
- `web_search`

The tools are collected in `src/tools/__init__.py` as `TOOLS`.

---

## Temporary memory

The working memory is implemented in `src/memory.py`.

It keeps:

- `recent_messages`
- `summary`
- `facts`
- `tool_results`
- `turn_count`

### Important memory methods

- `add_message(msg)`
- `add_turn(user_text, assistant_text)`
- `add_tool_result(tool_name, args, result)`
- `extract_stable_facts(user_text, chat_model)`
- `format_facts()`
- `format_tool_results(limit=...)`
- `recent_context_messages()`
- `format_recent_dialogue(window_messages=...)`
- `should_refresh_summary()`
- `print_state()`

The memory layer is designed to be simple enough for learning, but structured enough to support later benchmark work.

---

## Unified benchmark interface

The `benchmarks/common/` package defines the shared structures and abstract contracts used by all benchmarks.

### Shared models
Defined in `benchmarks/common/models.py`:

- `BenchmarkSpec`
- `Task`
- `EnvironmentState`
- `Observation`
- `Action`
- `ToolEvent`
- `TrajectoryEvent`
- `Trajectory`
- `Episode`
- `EvaluationContext`
- `RunResult`
- `EvaluationResult`
- `BenchmarkSample`

### Compatibility notes
- `BenchmarkSample` is a compatibility wrapper around the newer `Task` / `Episode` model.
- `TrajectoryStep` remains available as an alias for `TrajectoryEvent`.

### Abstract interfaces
Defined in `benchmarks/common/interfaces.py`:

- `BenchmarkLoader`
- `BenchmarkAdapter`
- `BenchmarkEnvironment`
- `AgentRuntime`
- `BenchmarkReporter`
- `BenchmarkEvaluator`

### Base utilities
- `BenchmarkLogger`
- `EvaluatorBase`
- `ReportWriter`

---

## Unified agent runtime interface

Any agent that should run in this framework should be wrapped behind an `AgentRuntime`.

### Required methods

#### `reset(episode, initial_state) -> None`
Prepare the runtime for a new benchmark episode.

Parameters:
- `episode`: normalized `Episode`
- `initial_state`: initial `EnvironmentState`

#### `act(observation) -> Action`
Produce one action for the current observation.

Parameters:
- `observation`: current `Observation`

Returns:
- `Action`

#### `get_trajectory() -> Trajectory`
Return the collected trajectory for the current episode.

#### `get_metrics() -> Dict[str, Any]`
Return runtime metrics such as:
- message count
- event count
- token usage
- latency
- memory statistics

### Current runtime implementation
The concrete runtime wrapper is `src/runtime.py`:

- `ResearchHelperAgentRuntime`

It adapts the current `ResearchHelperAgent` to the benchmark interface.

---

## Benchmark interface responsibilities

Every benchmark should provide the following components.

### Loader
Reads raw benchmark data and normalizes it into `Episode` objects.

### Adapter
Converts benchmark data into agent-ready context and messages.

### Environment
Owns the benchmark state and exposes:

- `reset()`
- `observe()`
- `step(action)`
- `snapshot()`
- `is_done()`

### Evaluator
Scores the agent output or trajectory using benchmark-specific rules.

### Report writer
Writes benchmark artifacts to disk in multiple formats.

---

## LoCoMo benchmark integration

LoCoMo is currently the integrated benchmark in the repository.

It is implemented under `benchmarks/locomo/` and uses the shared benchmark interface.

### LoCoMo loader
`benchmarks/locomo/loader.py`

The loader can:
- load LoCoMo samples from JSON or JSONL,
- normalize raw records into `BenchmarkSample` or `Episode`,
- iterate over records,
- normalize conversation sessions.

### LoCoMo adapter
`benchmarks/locomo/adapter.py`

The adapter:
- preserves conversation roles,
- converts LoCoMo context into LangChain messages,
- prepares benchmark input for the agent,
- supports benchmark-specific prompt modes.

### LoCoMo environment
`benchmarks/locomo/environment.py`

The LoCoMo environment provides a simple replay-style benchmark environment. It:
- resets to the episode,
- returns an observation for the final benchmark question,
- marks the episode done after the answer is produced,
- stores state snapshots for the trajectory.

### LoCoMo runner
`benchmarks/locomo/runner.py`

The runner:
- resets the environment,
- resets the runtime,
- obtains the final observation,
- executes the agent,
- collects the trajectory,
- returns a `RunResult`.

### LoCoMo evaluator
`benchmarks/locomo/evaluator.py`

The evaluator supports:
- local fuzzy/exact scoring,
- run-result-based scoring,
- batch evaluation,
- optional official-evaluator bridging when configured.

### LoCoMo metrics
`benchmarks/locomo/metrics.py`

The benchmark computes:
- answer length,
- turn count,
- tool-call count,
- latency,
- tool latency,
- model latency,
- tool breakdown.

### LoCoMo report generation
`benchmarks/locomo/report.py`

Reports include per-sample and batch outputs such as:

- `episode.json`
- `trajectory.json`
- `evaluation.json`
- `output.json`
- `trace.json`
- `analysis.json`
- `results.json`
- `summary.csv`
- `metrics.json`
- `report.md`

### LoCoMo configuration
`benchmarks/locomo/config.py`

LoCoMo is controlled through environment variables.

---

## Setup

### 1. Create a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create `.env`

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

### 4. Add your Hugging Face token

The agent requires:

- `HUGGINGFACEHUB_API_TOKEN`

Optional agent settings include:

- `HF_MODEL_ID`
- `HF_PROVIDER`
- `MAX_NEW_TOKENS`
- `TEMPERATURE`
- `DO_SAMPLE`
- `SUMMARY_EVERY_N_TURNS`
- `RECENT_WINDOW_TURNS`
- `MAX_TOOL_STEPS`

---

## Running the interactive agent

Start the baseline agent with:

```bash
python app.py
```

The agent will:
- load the configured Hugging Face model,
- initialize temporary memory,
- start a chat loop,
- use the available tools when needed.

Type `exit` or `quit` to stop.

---

## Running LoCoMo

Run the benchmark entry point with:

```bash
python -m benchmarks.locomo.run
```

The LoCoMo runner reads its configuration from environment variables.

### Common LoCoMo environment variables

#### Data and output
- `LOCOMO_DATA_FILE`
- `LOCOMO_OUTPUT_DIR`

#### Official evaluation
- `LOCOMO_OFFICIAL_ROOT`
- `LOCOMO_USE_OFFICIAL_EVAL`

#### Runtime behavior
- `LOCOMO_ALLOW_TOOLS`
- `LOCOMO_PROMPT_MODE`
- `LOCOMO_RUN_MODE`
- `LOCOMO_SAMPLE_ID`
- `LOCOMO_MAX_SAMPLES`
- `LOCOMO_CREATE_DEMO_DATA`
- `LOCOMO_VERBOSE`

### Example LoCoMo configuration

```env
LOCOMO_DATA_FILE=data/locomo/locomo10.jsonl
LOCOMO_OUTPUT_DIR=results/locomo
LOCOMO_OFFICIAL_ROOT=third_party/locomo-official
LOCOMO_USE_OFFICIAL_EVAL=true
LOCOMO_ALLOW_TOOLS=false
LOCOMO_PROMPT_MODE=qa
LOCOMO_RUN_MODE=single
LOCOMO_SAMPLE_ID=locomo_001
LOCOMO_MAX_SAMPLES=0
LOCOMO_CREATE_DEMO_DATA=true
LOCOMO_VERBOSE=true
```

### LoCoMo run modes

- `single` — run one sample
- `batch` — run multiple samples

---

## LoCoMo example script

The repository includes `benchmarks/locomo/example.py` for quick inspection of the benchmark components.

It demonstrates:

- loading benchmark data,
- converting a sample through the adapter,
- evaluating a mock answer,
- generating simple benchmark reports.

Run it with:

```bash
python -m benchmarks.locomo.example
```

---

## Benchmark validation script

`benchmarks/test_implementation.py` provides a simple import and structure sanity check.

It verifies:

- common module imports,
- shared dataclass creation,
- logger behavior,
- adapter behavior,
- evaluator behavior,
- metrics computation.

Run it with:

```bash
python benchmarks/test_implementation.py
```

---

## Benchmark artifacts

Per-sample output is written under the configured benchmark output directory.

A typical sample folder contains:

- `episode.json`
- `trajectory.json`
- `evaluation.json`
- `output.json`
- `trace.json`
- `analysis.json`

Batch runs also produce:

- `results.json`
- `summary.csv`
- `metrics.json`
- `report.md`

---

## Adding a new agent runtime

To integrate a different agent, wrap it in a class that satisfies the `AgentRuntime` interface.

The runtime should expose:

- `reset(episode, initial_state)`
- `act(observation)`
- `get_trajectory()`
- `get_metrics()`

This allows the benchmark framework to stay independent from the agent implementation.

---

## Adding a new benchmark

To add another benchmark, create a new package under `benchmarks/` with the same pattern used by LoCoMo.

Recommended files:

- `loader.py`
- `adapter.py`
- `environment.py`
- `runner.py`
- `evaluator.py`
- `metrics.py`
- `report.py`
- `prompts.py`
- `official_bridge.py` if needed

The benchmark should normalize its raw data into the shared models and use the same interface contracts as LoCoMo.

---

## Research direction

The long-term research goal is to develop an **Adaptive Memory Agent** that can choose among different memory structures based on the current task.

Future custom benchmarks are intended to combine ideas from:

- long-context conversation recall,
- multi-session memory,
- stateful tool execution,
- trajectory-based evaluation,
- and adaptive memory routing.

---

## Design principles

This repository follows these design principles:

- benchmark-independent core interfaces
- modular benchmark implementations
- reusable agent runtime wrappers
- explicit environment state
- detailed trajectory logging
- reproducible evaluation
- benchmark-specific reporting
- research-friendly code organization

---

## Notes on current implementation

- The current interactive baseline is still available through `app.py`.
- The current benchmark implementation is centered on LoCoMo.
- The framework is already structured to support more benchmarks without rewriting the agent core.
- The benchmark layer uses shared abstractions so that future benchmark integrations can reuse the same interfaces.

---
