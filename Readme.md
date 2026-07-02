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
