# AdaMem Evaluation Output Specification

**Version:** 1.0

## Purpose

This document defines the standard output structure for all benchmark evaluations in the AdaMem project.

The objective is to ensure that every evaluation run is:

* Reproducible
* Easy to inspect
* Comparable across memory architectures
* Compatible with all supported benchmarks
* Easy to visualize and analyse later

This specification applies to:

* LongMemEval
* LoCoMo
* ToolSandbox
* BFCL
* Future heterogeneous-session benchmark
* Any future benchmark integrated into the evaluation framework

---

# 1. General Principles

The evaluation system shall produce **immutable experiment outputs**.

A completed evaluation run **must never overwrite** any previous run.

Every execution of a benchmark creates a completely new directory containing every artifact produced during that run.

The reporting pipeline must remain benchmark-independent.

Benchmark-specific metrics are allowed, but the file structure and reporting format must remain identical across all benchmarks.

---

# 2. Result Directory Structure

```text
results/

│
├── experiment_index.csv
│
├── adamem/
│   ├── 2026-07-18_09-32-14/
│   ├── 2026-07-20_14-10-42/
│   └── ...
│
├── flat_rag/
│   ├── 2026-07-18_10-55-22/
│   └── ...
│
├── episodic/
│   └── ...
│
├── learned_consolidation/
│   └── ...
│
└── ...
```

Each timestamp directory represents **one complete batch evaluation run**.

A batch run is defined as a single execution of one benchmark over an entire dataset (or a selected subset of that dataset). A batch therefore consists of multiple evaluation cases (episodes/samples), each executed independently but aggregated into a single benchmark result.

Examples:

* LongMemEval test set (500 samples)
* LoCoMo validation split
* ToolSandbox evaluation scenarios
* BFCL benchmark suite
* Future heterogeneous-session benchmark

All individual sample executions belonging to the same benchmark execution must be stored inside the same timestamp directory.

A timestamp directory must never be reused or overwritten.

---

# 3. Run Directory Layout

Every run directory shall contain the following structure.

```
YYYY-MM-DD_HH-MM-SS/

│
├── raw/
│
├── reports/
│
└── figures/
```

---

# 4. raw/

Write these raw files actively during the ongoing run not waiting till the whole batch runs.
The `raw` directory stores execution artifacts exactly as produced during evaluation.

Unlike the processed reports, the contents of this directory are organized **per evaluation case**, since a single batch run may contain hundreds or thousands of independent episodes.

These files are intended for debugging, reproduction, execution replay, and detailed trajectory inspection.

Nothing inside this directory should be modified after the run finishes.

Recommended layout:

```text
raw/

├── trajectories/
│     sample_0001.jsonl
│     sample_0002.jsonl
│     ...
│
├── tool_calls/
│     sample_0001.jsonl
│     sample_0002.jsonl
│     ...
│
├── environments/
│     sample_0001.json
│     sample_0002.json
│     ...
│
└── logs/
```

Each file corresponds to **one evaluated sample**.

---

## trajectories/

Stores the complete interaction trajectory for a single evaluation case.

Each JSONL file contains one trajectory event per line in chronological order.

Typical contents include:

* user messages
* assistant responses
* observations
* actions
* tool calls
* memory operations
* state transitions
* milestone checks
* latency
* token usage
* exceptions

Keeping one trajectory file per sample avoids extremely large monolithic files and makes debugging individual failures significantly easier.

---

## tool_calls/

Stores only tool invocation records for a single evaluation case.

Each entry should include:

* tool name
* arguments
* result
* latency
* token count
* success/failure

Separating tool calls from trajectories enables dedicated analysis of tool usage without parsing complete conversations.

---

## environments/

Stores the final (or checkpointed) environment state for one evaluation case.

Examples include:

* ToolSandbox world state
* milestone completion
* minefield status
* environment metadata

Benchmarks without explicit environment state may leave benchmark-specific fields empty while preserving the file.

---

# 5. reports/

This directory contains processed evaluation outputs representing the **entire batch run**.

Unlike the `raw` directory, these files summarize all evaluated samples rather than storing execution traces.

---

## cases.jsonl

The primary per-sample evaluation report.

There is one JSON object per evaluated sample.

Each record summarizes the outcome of a single evaluation case while referencing its corresponding raw execution artifacts.

Each record should contain:

### Identification

* sample id
* episode id
* benchmark
* task family
* task type

### Inputs

* question
* context metadata
* conversation statistics

### Prediction

* predicted answer
* tool usage summary
* routing decision (if available)
* memory actions summary (if available)

### Reference

* gold answer
* expected tool behaviour
* expected state

### Evaluation

* overall correctness
* overall score
* benchmark-specific metrics
* evidence hits
* failure mode
* correctness explanation

### Diagnostics

Evaluator-specific diagnostics.

### Raw Artifact References

Each case should include references to its corresponding raw execution artifacts.

Example:

```json
{
  "raw": {
    "trajectory": "../raw/trajectories/sample_0001.jsonl",
    "tool_calls": "../raw/tool_calls/sample_0001.jsonl",
    "environment": "../raw/environments/sample_0001.json"
  }
}
```

This allows `cases.jsonl` to remain compact while preserving direct access to the complete execution history for every evaluated sample.

---

# 6. figures/

The code for generating figures will be written later.

Reserved for generated visualisations.

Examples:

* bar charts
* radar plots
* confusion matrices
* routing distributions
* memory usage plots
* benchmark comparison plots

Evaluation code should not depend on these files.

They are considered derived artifacts.

---

# 7. experiment_index.csv

Update this actively after the run. 
The root directory contains a global index of all evaluation runs.

One row per execution.

Suggested columns:

```
run_id

timestamp

agent_name

memory_architecture

benchmark

benchmark_version

dataset

sample_count

accuracy

average_score

result_directory
```

This file serves as the master experiment catalogue.

It allows quick lookup without traversing every run directory.

---

# 8. Benchmark Independence Rules

Every benchmark must produce the same directory structure.

Benchmark-specific metrics are allowed.

Benchmark-specific files are not.

Additional information should always be placed inside:

* diagnostics
* benchmark_specific
* metadata

rather than introducing new report files.

---

# 9. Extensibility Rules

Future benchmarks must reuse this reporting structure.

Only the contents of:

* metrics
* diagnostics
* benchmark_specific

should change.

No modifications to the directory layout should be required when integrating additional benchmarks.

---

# 10. Overwrite Policy

Evaluation runs are immutable.

The evaluation framework shall never overwrite an existing timestamp directory.

Every execution must create a new timestamped run directory.

Existing runs must remain unchanged.

---

# 11. Serialization Rules

All report files must use UTF-8 encoding.

JSON files should be human-readable (pretty-printed where practical, except JSONL).

JSONL files must contain exactly one JSON object per line.

CSV files should include headers.

Timestamps should follow ISO-8601 where stored inside JSON metadata.

Directory names should use:

```
YYYY-MM-DD_HH-MM-SS
```

using the local execution time.

---

# 12. Compatibility with Existing Framework

The current evaluation framework already contains sufficient information to generate this reporting format.

Primary sources are:

* `RunResult`
* `EvaluationResult`
* `TrajectoryEvent`
* `ToolEvent`
* `EnvironmentState`
* `EvaluationContext`

No major redesign of the evaluation pipeline is required.

Implementation should focus on exporting these structures into the standardized reporting format defined in this document.

---

# 13. Design Philosophy

The reporting system is organized into three hierarchical layers:

### Layer 1 — Raw Execution Artifacts

Stores the exact execution history for every individual evaluation case.

Each sample has its own trajectory, tool-call log, and environment snapshot.

These files exist for reproducibility, debugging, and execution replay.

---

### Layer 2 — Processed Evaluation Reports

Summarizes every evaluated sample within a single batch run.

`cases.jsonl` acts as the bridge between raw execution data and aggregated benchmark statistics by recording the evaluation outcome of each sample together with references to its raw artifacts.

`summary.json` and `aggregates_long.csv` summarize the entire batch.

---

### Layer 3 — Derived Visualizations

Contains figures generated from the processed reports.

These are derived artifacts only and should never be treated as the source of truth.

The raw execution artifacts and processed reports remain the canonical evaluation outputs.
