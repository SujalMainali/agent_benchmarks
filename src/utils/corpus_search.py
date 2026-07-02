from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def _excerpt(text: str, terms: List[str], width: int = 220) -> str:
    lower = text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) != -1]

    if not positions:
        snippet = text[:width].replace("\n", " ")
        return snippet + ("..." if len(text) > width else "")

    start = max(0, min(positions) - 80)
    end = min(len(text), start + width)
    snippet = text[start:end].replace("\n", " ")

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def search_corpus_results(root: Path, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search text files under a folder and return structured ranked results.
    """
    if not root.exists():
        return []

    terms = _tokenize(query)
    if not terms:
        return []

    scored: List[Tuple[int, Path, str]] = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lower = content.lower()
        score = sum(lower.count(term) for term in terms)

        if score > 0:
            scored.append((score, path, _excerpt(content, terms)))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, path, excerpt in scored[:top_k]:
        results.append(
            {
                "title": path.name,
                "path": str(path),
                "score": score,
                "snippet": excerpt,
            }
        )
    return results


def search_corpus(root: Path, query: str, top_k: int = 3) -> str:
    """
    Search text files under a folder and return a compact ranked result string.
    """
    if not root.exists():
        return f"Search folder does not exist: {root}"

    terms = _tokenize(query)
    if not terms:
        return "Please provide a more specific query."

    results = search_corpus_results(root, query, top_k=top_k)
    if not results:
        return "No relevant matches found."

    lines = []
    for item in results:
        lines.append(f"[{item['title']}] score={item['score']}\n{item['snippet']}")

    return "\n\n".join(lines)
