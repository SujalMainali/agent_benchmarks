from pathlib import Path

from langchain_core.tools import tool

from ..utils.corpus_search import search_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = PROJECT_ROOT / "data" / "notes"


@tool
def note_lookup(query: str) -> str:
    """
    Search personal notes for relevant content.

    Put note text files inside:
    data/notes/
    """
    return search_corpus(NOTES_DIR, query, top_k=3)