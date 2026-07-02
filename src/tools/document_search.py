import json
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.tools import tool

from ..utils.corpus_search import search_corpus_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


@tool
def document_search(query: str) -> str:
    """
    Search local project documents for relevant text.

    Returns structured JSON with source_type=document, query, retrieved_at,
    retrieval_log, and ranked local document results.

    Put text files inside:
    data/documents/
    """
    results = search_corpus_results(DOCUMENTS_DIR, query, top_k=3)
    payload = {
        "source_type": "document",
        "query": query,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_log": {
            "tool": "document_search",
            "root": str(DOCUMENTS_DIR),
            "result_count": len(results),
        },
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False)
