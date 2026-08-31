"""Web search and page fetch.

Backends are tried in order of whichever credential exists, ending at a keyless
DuckDuckGo HTML scrape. The scrape is not as good as a real search API and it
will break when the page markup changes -- but it means the web tool works on a
fresh install with no signup, and it fails to a clear message rather than to a
missing capability.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse

from ..core import secrets
from ..core.errors import ProviderError
from ..core.logging import get_logger
from ..gateway.base import http_client

log = get_logger("hubs.websearch")

BACKENDS = [
    {"id": "tavily", "label": "Tavily", "env_hint": "TAVILY_API_KEY"},
    {"id": "brave", "label": "Brave Search", "env_hint": "BRAVE_API_KEY"},
    {"id": "serper", "label": "Serper (Google)", "env_hint": "SERPER_API_KEY"},
    {"id": "exa", "label": "Exa", "env_hint": "EXA_API_KEY"},
    {"id": "duckduckgo", "label": "DuckDuckGo (keyless)", "env_hint": ""},
]


def available_backends() -> list[dict[str, Any]]:
    return [
        {**b, "configured": b["id"] == "duckduckgo" or secrets.has(b["id"])}
        for b in BACKENDS
    ]


def _pick() -> str:
    for backend in BACKENDS:
        if backend["id"] == "duckduckgo" or secrets.has(backend["id"]):
            return str(backend["id"])
    return "duckduckgo"


def _result(title: str, url: str, snippet: str, source: str, published: str = "") -> dict[str, Any]:
    return {
        "title": (title or url)[:300].strip(),
        "url": url,
        "snippet": re.sub(r"\s+", " ", snippet or "")[:600].strip(),
        "source": source,
        "domain": urlparse(url).netloc,
        "published": published,
    }


async def _tavily(query: str, limit: int) -> list[dict[str, Any]]:
    response = await http_client("search", timeout=30).post(
        "https://api.tavily.com/search",
        json={
            "api_key": secrets.get("tavily"),
            "query": query,
            "max_results": limit,
            "search_depth": "advanced",
            "include_answer": False,
        },
    )
    response.raise_for_status()
    return [
        _result(r.get("title", ""), r.get("url", ""), r.get("content", ""), "tavily",
                r.get("published_date", ""))
        for r in response.json().get("results") or []
    ]


async def _brave(query: str, limit: int) -> list[dict[str, Any]]:
    response = await http_client("search", timeout=30).get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(limit, 20)},
        headers={
            "accept": "application/json",
            "x-subscription-token": secrets.get("brave") or "",
        },
    )
    response.raise_for_status()
    web = response.json().get("web") or {}
    return [
        _result(r.get("title", ""), r.get("url", ""), r.get("description", ""), "brave",
                r.get("age", ""))
        for r in web.get("results") or []
    ]


async def _serper(query: str, limit: int) -> list[dict[str, Any]]:
    response = await http_client("search", timeout=30).post(
        "https://google.serper.dev/search",
        json={"q": query, "num": min(limit, 20)},
        headers={"x-api-key": secrets.get("serper") or "", "content-type": "application/json"},
    )
    response.raise_for_status()
    return [
        _result(r.get("title", ""), r.get("link", ""), r.get("snippet", ""), "serper",
                r.get("date", ""))
        for r in response.json().get("organic") or []
    ]


async def _exa(query: str, limit: int) -> list[dict[str, Any]]:
    response = await http_client("search", timeout=30).post(
        "https://api.exa.ai/search",
        json={"query": query, "numResults": limit, "contents": {"text": {"maxCharacters": 600}}},
        headers={"x-api-key": secrets.get("exa") or "", "content-type": "application/json"},
    )
    response.raise_for_status()
    return [
        _result(r.get("title", ""), r.get("url", ""), (r.get("text") or "")[:600], "exa",
                r.get("publishedDate", ""))
        for r in response.json().get("results") or []
    ]


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _untag(html: str) -> str:
    text = _TAG_RE.sub("", html)
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
        ("&#x27;", "'"), ("&#39;", "'"), ("&nbsp;", " "),
    ):
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def _ddg_url(href: str) -> str:
    # DuckDuckGo wraps results in a redirector: /l/?uddg=<encoded>
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        return unquote(match.group(1))
    return href if href.startswith("http") else f"https:{href}"


async def _duckduckgo(query: str, limit: int) -> list[dict[str, Any]]:
    response = await http_client("search", timeout=30).post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
            "content-type": "application/x-www-form-urlencoded",
        },
    )
    response.raise_for_status()
    out: list[dict[str, Any]] = []
    for match in _DDG_RESULT_RE.finditer(response.text):
        url = _ddg_url(match.group("href"))
        if not url.startswith("http"):
            continue
        out.append(
            _result(_untag(match.group("title")), url, _untag(match.group("snippet")), "duckduckgo")
        )
        if len(out) >= limit:
            break
    return out


_DISPATCH = {
    "tavily": _tavily,
    "brave": _brave,
    "serper": _serper,
    "exa": _exa,
    "duckduckgo": _duckduckgo,
}


async def search(query: str, *, limit: int = 8, backend: str = "") -> list[dict[str, Any]]:
    """Search the web. Falls through to the next backend on failure."""
    if not query.strip():
        return []

    order = [backend] if backend and backend != "auto" else []
    if not order:
        chosen = _pick()
        order = [chosen] + [b["id"] for b in BACKENDS if b["id"] != chosen]

    errors: list[str] = []
    for name in order:
        fn = _DISPATCH.get(name)
        if fn is None:
            continue
        if name != "duckduckgo" and not secrets.has(name):
            continue
        try:
            results = await fn(query, limit)
            if results:
                return results[:limit]
            errors.append(f"{name}: no results")
        except Exception as exc:  # noqa: BLE001 - try the next backend
            log.debug("Search backend %s failed: %s", name, exc)
            errors.append(f"{name}: {str(exc)[:120]}")

    raise ProviderError(
        "websearch",
        "No search backend returned results. " + "; ".join(errors[:3]),
        retryable=True,
    )


async def fetch_page(url: str, *, max_chars: int = 20_000) -> dict[str, Any]:
    """Fetch a URL and return readable text."""
    from ..core.util import now
    from ..ingest import extractors

    extracted = await extractors.extract_url(url)
    return {
        "url": url,
        "title": extracted.title,
        "text": extracted.text[:max_chars],
        "truncated": len(extracted.text) > max_chars,
        "mime_type": extracted.mime_type,
        "fetched_at": now().isoformat(),
    }


def search_url(query: str) -> str:
    """A human-clickable search link, for citing where a result came from."""
    return f"https://duckduckgo.com/?q={quote_plus(query)}"
