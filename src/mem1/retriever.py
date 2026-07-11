"""
MEM1 Retriever - Handles <search> queries and returns <information>.

Supports multiple retrieval backends:
- Web search (via Serper API or existing web_search tool)
- Corpus search (for LoCoMo conversation context)
- None (for testing without retrieval)
"""

from dataclasses import dataclass
from typing import Optional, Protocol
import json


class SearchBackend(Protocol):
    """Protocol for search backends."""
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Execute search and return results."""
        ...


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    content: str
    source: str = ""
    score: float = 0.0


class WebSearchBackend:
    """Web search using existing web_search tool or Serper API."""
    
    def __init__(self, serper_api_key: str = "", use_tool: bool = True):
        self.serper_api_key = serper_api_key
        self.use_tool = use_tool
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Execute web search."""
        if self.use_tool:
            return self._search_with_tool(query, top_k)
        elif self.serper_api_key:
            return self._search_with_serper(query, top_k)
        return []
    
    def _search_with_tool(self, query: str, top_k: int) -> list[dict]:
        """Use existing web_search tool."""
        try:
            from src.tools.web_search import web_search
            result = web_search(query)
            # Parse result and return as list of dicts
            return [{"title": "Web Result", "content": result, "source": "web"}]
        except Exception as e:
            return [{"title": "Error", "content": str(e), "source": "error"}]
    
    def _search_with_serper(self, query: str, top_k: int) -> list[dict]:
        """Use Serper API for Google search."""
        try:
            import requests
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_api_key},
                json={"q": query, "num": top_k},
                timeout=10,
            )
            data = response.json()
            results = []
            for item in data.get("organic", [])[:top_k]:
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("snippet", ""),
                    "source": item.get("link", ""),
                })
            return results
        except Exception as e:
            return [{"title": "Error", "content": str(e), "source": "error"}]


class CorpusSearchBackend:
    """Search within a provided corpus (e.g., LoCoMo conversation context)."""
    
    def __init__(self, corpus: list[str] = None):
        self.corpus = corpus or []
    
    def set_corpus(self, corpus: list[str]) -> None:
        """Set the corpus to search within."""
        self.corpus = corpus
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Simple keyword-based search in corpus."""
        if not self.corpus:
            return []
        
        query_terms = set(query.lower().split())
        scored = []
        
        for i, doc in enumerate(self.corpus):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                scored.append((overlap, i, doc))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, idx, doc in scored[:top_k]:
            results.append({
                "title": f"Context {idx + 1}",
                "content": doc[:500],  # Truncate long docs
                "source": f"corpus:{idx}",
                "score": score,
            })
        
        return results


class NoopSearchBackend:
    """No-op backend for testing without retrieval."""
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return []


class Mem1Retriever:
    """
    MEM1 Retriever - executes <search> queries and formats <information>.
    
    Usage:
        retriever = Mem1Retriever(retriever_type="web", serper_api_key="...")
        info_block = retriever.search("What is the capital of France?")
        # Returns: "<information>\n...\n</information>"
    """
    
    def __init__(
        self,
        retriever_type: str = "web",
        serper_api_key: str = "",
        top_k: int = 3,
        max_chars: int = 800,
        corpus: list[str] = None,
    ):
        self.top_k = top_k
        self.max_chars = max_chars
        
        # Initialize backend based on type
        if retriever_type == "web":
            self.backend = WebSearchBackend(serper_api_key=serper_api_key)
        elif retriever_type == "corpus":
            self.backend = CorpusSearchBackend(corpus=corpus)
        else:
            self.backend = NoopSearchBackend()
    
    def set_corpus(self, corpus: list[str]) -> None:
        """Set corpus for corpus-based retrieval."""
        if isinstance(self.backend, CorpusSearchBackend):
            self.backend.set_corpus(corpus)
    
    def search(self, query: str) -> str:
        """
        Execute search and return formatted <information> block.
        
        Args:
            query: The search query from <search> tag
            
        Returns:
            Formatted string: "<information>...</information>"
        """
        results = self.backend.search(query, self.top_k)
        
        if not results:
            return "<information>\nNo results found.\n</information>"
        
        # Format results
        formatted_parts = []
        total_chars = 0
        
        for r in results:
            entry = f"[{r.get('title', 'Result')}]\n{r.get('content', '')}"
            if r.get('source'):
                entry += f"\nSource: {r['source']}"
            
            # Check character limit
            if total_chars + len(entry) > self.max_chars:
                break
            
            formatted_parts.append(entry)
            total_chars += len(entry) + 2  # +2 for separator
        
        content = "\n\n".join(formatted_parts)
        return f"<information>\n{content}\n</information>"