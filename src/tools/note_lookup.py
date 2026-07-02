import json
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.tools import tool

from ..utils.corpus_search import search_corpus_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = PROJECT_ROOT / "data" / "notes"


@tool
def note_lookup(query: str) -> str:
    """
    Search personal notes for relevant content.

    Returns structured JSON with source_type=local_memory, query, retrieved_at,
    retrieval_log, and ranked local note results.

    Put note text files inside:
    data/notes/
    """
    results = search_corpus_results(NOTES_DIR, query, top_k=3)
    payload = {
        "source_type": "local_memory",
        "query": query,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_log": {
            "tool": "note_lookup",
            "root": str(NOTES_DIR),
            "result_count": len(results),
        },
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False)
