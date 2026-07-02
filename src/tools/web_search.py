from __future__ import annotations

import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from langchain_core.tools import tool


SEARCH_URL = "https://duckduckgo.com/html/"
REQUEST_HEADERS = {
    "User-Agent": "research-helper/0.1 (+https://example.local)",
}


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _decode_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    return url


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._current_link: Dict[str, str] | None = None
        self._current_snippet: List[str] | None = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._current_link = {
                "title": "",
                "url": _decode_duckduckgo_url(attr_map.get("href", "")),
                "snippet": "",
            }

        if "result__snippet" in class_name:
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._current_link is not None:
            self._current_link["title"] += data
        if self._current_snippet is not None:
            self._current_snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_link is not None:
            self._current_link["title"] = " ".join(self._current_link["title"].split())
            self.results.append(self._current_link)
            self._current_link = None

        if self._current_snippet is not None and tag in {"a", "div"}:
            snippet = " ".join("".join(self._current_snippet).split())
            for result in reversed(self.results):
                if not result.get("snippet"):
                    result["snippet"] = snippet
                    break
            self._current_snippet = None


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _search_duckduckgo(query: str, max_results: int) -> tuple[List[Dict[str, str]], List[str]]:
    log = []
    url = f"{SEARCH_URL}?q={quote_plus(query)}"
    log.append(f"GET {url}")

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    log.append(f"search_status={response.status_code}")

    parser = _DuckDuckGoParser()
    parser.feed(response.text)

    results = []
    seen_urls = set()
    for item in parser.results:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(item)
        if len(results) >= max_results:
            break

    log.append(f"parsed_results={len(results)}")
    return results, log


def _fetch_page_content(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return "", f"skipped_content_type={content_type or 'unknown'}"

    parser = _PageTextParser()
    parser.feed(response.text)
    return _truncate(parser.text(), 1800), f"fetch_status={response.status_code}"


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for current external evidence.

    Returns structured JSON with query, top results, snippets, URLs, fetched
    page content, and a retrieval log.
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    retrieval_log: List[str] = []

    try:
        search_results, search_log = _search_duckduckgo(query, max(1, min(max_results, 5)))
        retrieval_log.extend(search_log)
    except Exception as exc:
        payload = {
            "source_type": "web",
            "query": query,
            "retrieved_at": retrieved_at,
            "retrieval_log": [f"search_error={exc}"],
            "results": [],
        }
        return json.dumps(payload, ensure_ascii=False)

    results: List[Dict[str, Any]] = []
    for rank, item in enumerate(search_results, start=1):
        url = item.get("url", "")
        content = ""
        content_log = "not_fetched"

        try:
            content, content_log = _fetch_page_content(url)
        except Exception as exc:
            content_log = f"fetch_error={exc}"

        retrieval_log.append(f"rank={rank} url={url} {content_log}")
        results.append(
            {
                "rank": rank,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "content": content,
            }
        )

    payload = {
        "source_type": "web",
        "query": query,
        "retrieved_at": retrieved_at,
        "retrieval_log": retrieval_log,
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False)
