# Research Helper Agent

A small LangChain + Hugging Face project for learning:

- message-based conversation state
- temporary memory
- structured tool calling
- tool result handling
- short summary refresh
- local document, note, and web lookup

## What this project does

- Calls a remote Qwen model through Hugging Face
- Keeps a temporary conversation memory with LLM-extracted structured facts
- Uses 4 tools:
  - `calculator`
  - `document_search`
  - `note_lookup`
  - `web_search`
- Labels retrieval sources as `local_memory`, `document`, or `web`
- Updates a short summary every few turns
- Logs the native message trace, including AI tool requests, `ToolMessage` results, and final AI answers

## Project structure

```text
research-helper/
├─ .env
├─ .env.example
├─ requirements.txt
├─ Readme.md
├─ app.py
├─ data/
│  ├─ documents/
│  └─ notes/
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ prompts.py
   ├─ memory.py
   ├─ agent.py
   ├─ utils/
   │  ├─ __init__.py
   │  └─ corpus_search.py
   └─ tools/
      ├─ __init__.py
      ├─ calculator.py
      ├─ document_search.py
      ├─ note_lookup.py
      └─ web_search.py


## LoCoMo benchmark configuration

The LoCoMo benchmark is configured from environment variables in `benchmarks/locomo/config.py`.

Use these keys in `.env` or `.env.example`:

- `LOCOMO_DATA_FILE`: input dataset path for LoCoMo JSON or JSONL samples
- `LOCOMO_OUTPUT_DIR`: where benchmark reports are written
- `LOCOMO_RUN_MODE`: `single` for one sample, `batch` for multiple samples
- `LOCOMO_SAMPLE_ID`: optional sample id to target in single mode
- `LOCOMO_MAX_SAMPLES`: cap for batch runs; `0` means no cap
- `LOCOMO_CREATE_DEMO_DATA`: if true, the example script creates demo data first
- `LOCOMO_VERBOSE`: if true, batch runs print progress

Example:

```env
LOCOMO_DATA_FILE=data/locomo/demo.jsonl
LOCOMO_OUTPUT_DIR=results/locomo
LOCOMO_RUN_MODE=single
LOCOMO_SAMPLE_ID=locomo_001
LOCOMO_MAX_SAMPLES=0
LOCOMO_CREATE_DEMO_DATA=true
LOCOMO_VERBOSE=true
```
