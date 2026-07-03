"""
Benchmarks Module - Overview and Usage

This module provides a comprehensive benchmarking framework for evaluating
the ResearchHelperAgent on various benchmark datasets. Currently implemented:

- LoCoMo: Long Conversation Memory benchmark

## Architecture

### Common Module (benchmarks/common/)

Shared infrastructure used by all benchmarks:

- **models.py**: Core data structures
  - BenchmarkSample: Normalized benchmark sample
  - RunResult: Complete run with trajectory and answer
  - EvaluationResult: Evaluation scores and diagnostics
  - TrajectoryStep: Single interaction turn
  - ToolEvent: Tool call and result

- **interfaces.py**: Abstract interfaces
  - BenchmarkAdapter: Converts benchmark samples to agent input
  - BenchmarkEvaluator: Scores agent outputs

- **evaluator_base.py**: Base evaluator class
  - Provides common evaluation utilities
  - Batch evaluation and aggregation
  - Category-based grouping

- **logger.py**: Interaction tracing
  - BenchmarkLogger: Captures full traces
  - Includes prompts, tool calls, memory state
  - Exportable to JSON

- **report_writer.py**: Report generation
  - Writes JSON, CSV, and Markdown reports
  - Per-sample and batch-level reports

### LoCoMo Module (benchmarks/locomo/)

Full implementation for the LoCoMo benchmark:

- **loader.py**: Dataset loading
  - LoCoMoLoader: Loads from JSON/JSONL files
  - Flattens official conversation-plus-QA records into per-question samples
  - Normalizes to BenchmarkSample format

- **adapter.py**: Sample adaptation
  - LoCoMoAdapter: Converts samples to agent input
  - Reconstructs conversation context with role-preserving replay
  - Keeps the final benchmark question separate from the history

- **prompts.py**: LoCoMo-specific prompts
  - System prompt for conversational QA
  - Evidence-aware QA prompt
  - Strict format prompt variants

- **runner.py**: Orchestration engine
  - LoCoMoRunner: Executes samples through the agent
  - Resets memory per sample
  - Replays the long context once and asks the final question once
  - Captures full trajectory and raw messages
  - Handles errors gracefully

- **evaluator.py**: Answer evaluation
  - LoCoMoEvaluator: Scores predictions
  - Exact match, fuzzy match, partial match logic for local fallback
  - Optional bridge to the official LoCoMo QA evaluator
  - Generates diagnostics and failure analysis

- **metrics.py**: Metrics computation
  - Per-run metrics: answer length, turns, latency, tool usage
  - Batch metrics: averages and aggregates
  - Tool breakdown analysis

- **report.py**: Report generation
  - LoCoMoReporter: Generates full reports
  - Per-sample analysis and trace logs
  - Category-wise breakdown

- **run.py**: Command-line entry point
  - Env-driven benchmark wrapper
  - Supports benchmark prompt modes and tool disabling
  - Can route scoring through the official LoCoMo evaluator

- **example.py**: Usage examples
  - Basic loading and inspection
  - Adapter usage
  - Evaluator usage without full agent

## Usage Guide

### 1. Running Examples

First, check out the example usage:

```bash
cd ~/Projects/Major\ Project/research-helper
python -m benchmarks.locomo.example
```

This will:
- Create a demo LoCoMo dataset
- Show how to load and inspect samples
- Demonstrate the adapter and evaluator
- Print information about how to run full benchmarks

### 2. Preparing Your Data

LoCoMo data can be provided in JSON or JSONL format. The loader supports both
simple demo samples and the official conversation-plus-QA format.

Simple sample shape:

```json
{
  "sample_id": "unique_id",
  "question": "The question to answer",
  "gold_answer": "The expected answer",
  "category": "question_type",
  "sessions": [
    [
      {
        "user": "User's message",
        "assistant": "Assistant's response"
      }
    ]
  ],
  "evidence": ["Optional evidence items"]
}
```

Official sample shape:

```json
{
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1": [
      {"speaker": "Caroline", "text": "Hey Mel!"},
      {"speaker": "Melanie", "text": "Hey Caroline!"}
    ]
  },
  "qa": [
    {
      "question": "What did Caroline research?",
      "answer": "Adoption agencies",
      "evidence": ["D2:8"],
      "category": 1
    }
  ]
}
```

### 3. Running Single Sample

Set the environment variables and run `python -m benchmarks.locomo.run`.

This will:
- Load the specific sample
- Run it through the agent
- Evaluate the output
- Generate detailed report with trace

### 4. Running Batch

Set `LOCOMO_RUN_MODE=batch` and run `python -m benchmarks.locomo.run`.

Options:
- `LOCOMO_RUN_MODE=batch`: Run all samples or the configured limit
- `LOCOMO_MAX_SAMPLES=N`: Limit to N samples
- `LOCOMO_OUTPUT_DIR=DIR`: Save results to DIR

### 5. Output Reports

Single sample produces:
- `output.json`: Predicted answer, question, metrics, and benchmark metadata
- `trace.json`: Full interaction trace with raw replayed messages
- `analysis.json`: Evaluation analysis and diagnostics

Batch produces:
- `results.json`: All evaluation results
- `summary.csv`: Quick summary as CSV
- `metrics.json`: Detailed metrics (per-sample and batch)
- `report.md`: Human-readable markdown report
- `{sample_id}/`: Subdirectory for each sample with same files

When `LOCOMO_USE_OFFICIAL_EVAL=true`, the final scores come from the vendored
official LoCoMo QA evaluator in `third_party/locomo-official/`.

### 6. Evaluating Results

Parse the evaluation results:

```python
import json

# Load results
with open("results/batch_run/results.json") as f:
    results = json.load(f)

# Check accuracy
correct = sum(1 for r in results if r["is_correct"])
total = len(results)
accuracy = correct / total

print(f"Accuracy: {accuracy:.2%}")
print(f"Correct: {correct}/{total}")

# Check by category
by_category = {}
for r in results:
    cat = r["diagnostics"].get("category", "unknown")
    if cat not in by_category:
        by_category[cat] = {"correct": 0, "total": 0}
    by_category[cat]["total"] += 1
    if r["is_correct"]:
        by_category[cat]["correct"] += 1

for cat, stats in by_category.items():
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"{cat}: {acc:.2%}")
```

## Programmatic Usage

### Basic workflow

```python
from src.agent import ResearchHelperAgent
from src.config import load_settings
from src.tools import *
from langchain_huggingface import ChatHuggingFace

from benchmarks.locomo import LoCoMoLoader, LoCoMoRunner, LoCoMoEvaluator
from benchmarks.locomo.report import LoCoMoReporter

# Setup
settings = load_settings()
model = ChatHuggingFace(model_id=settings.model_id, ...)
agent = ResearchHelperAgent(chat_model=model, tools=[...])

# Load
loader = LoCoMoLoader()
samples = loader.load_from_jsonl("data.jsonl")

# Run
runner = LoCoMoRunner(agent)
results = runner.run_batch(samples)

# Evaluate
evaluator = LoCoMoEvaluator()
evals = [evaluator.evaluate(r) for r in results]

# Report
reporter = LoCoMoReporter("results/")
reporter.write_full_report(results, evals)
```

### Custom adapter

For different benchmark formats, extend BenchmarkAdapter:

```python
from benchmarks.common import BenchmarkAdapter, BenchmarkSample

class MyBenchmarkAdapter(BenchmarkAdapter):
    def load_sample(self, data):
        # Your loading logic
        return BenchmarkSample(...)
    
    def build_agent_input(self, sample):
        # Your conversion logic
        return {...}
```

## Metrics and Scoring

### Correctness Scoring

- **Exact Match (1.0)**: Normalized strings match exactly
- **Fuzzy Match (0.8)**: >85% similarity by sequence matching
- **Partial Match (0.5)**: Predicted answer contains gold answer
- **No Match (0.0)**: No meaningful overlap

### Computed Metrics

Per-run:
- `answer_length_chars`: Character count of answer
- `answer_length_words`: Word count of answer
- `turn_count`: Number of interaction turns
- `tool_call_count`: Total tool invocations
- `total_latency_ms`: Execution time
- `tool_latency_ms`: Time in tool calls
- `model_latency_ms`: Time in model inference

Batch:
- `avg_answer_length_chars`: Average answer length
- `avg_turn_count`: Average turns per sample
- `avg_tool_calls`: Average tool calls per sample
- `avg_latency_ms`: Average execution time

### Evaluation Diagnostics

- `category`: Question category (from benchmark)
- `metrics`: Full metrics dict
- `predicted_length_chars`: Predicted answer length
- `gold_length_chars`: Gold answer length
- `trajectory_length`: Number of turns used

## Extending for New Benchmarks

To add a new benchmark (e.g., LongMemEval):

1. Create `benchmarks/longmemeval/` directory
2. Implement loader, adapter, evaluator, runner, metrics, report
3. Follow the same structure as LoCoMo
4. Extend from common/ classes and interfaces
5. Add to benchmarks/__init__.py

The common module ensures consistency across benchmarks.

## Integration with Agent

Key integration points:

- **Memory reset**: Each sample starts with clean TemporaryMemory
- **Tool binding**: Agent uses bound_tools from agent setup
- **Trajectory capture**: BenchmarkLogger captures all interactions
- **Error handling**: Errors are captured, not raised

The benchmark layer does NOT modify the core agent logic.

It simply:
1. Prepares input in the agent's expected format
2. Runs the agent through one or more samples
3. Collects output and traces
4. Evaluates and reports

This keeps the agent reusable for any application.
"""

# This module docstring serves as the README
# To view: python -c "import benchmarks; help(benchmarks)"
