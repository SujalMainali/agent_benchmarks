from pathlib import Path

from langchain_core.tools import tool

from ..utils.corpus_search import search_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


@tool
def document_search(query: str) -> str:
    """
    Search local project documents for relevant text.

    Put text files inside:
    data/documents/
    """
    return search_corpus(DOCUMENTS_DIR, query, top_k=3)