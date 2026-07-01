# Research Helper Agent

A small LangChain + Hugging Face project for learning:

- message-based conversation state
- temporary memory
- structured tool calling
- tool result handling
- short summary refresh
- local document and note lookup

## What this project does

- Calls a remote Qwen model through Hugging Face
- Keeps a temporary conversation memory
- Uses 3 tools:
  - `calculator`
  - `document_search`
  - `note_lookup`
- Updates a short summary every few turns
- Logs tool outputs separately from chat history

## Project structure

```text
research-helper/
├─ .env
├─ .env.example
├─ requirements.txt
├─ README.md
├─ app.py
├─ data/
│  ├─ documents/
│  └─ notes/
└─ research_helper/
   ├─ __init__.py
   ├─ config.py
   ├─ prompts.py
   ├─ memory.py
   ├─ agent.py
   ├─ utils/
   │  ├─ __init__.py
   │  ├─ json_utils.py
   │  └─ corpus_search.py
   └─ tools/
      ├─ __init__.py
      ├─ calculator.py
      ├─ document_search.py
      └─ note_lookup.py
