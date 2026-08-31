"""Provider base class plus the plumbing every adapter needs.

Shared here so no adapter has to reinvent it: a pooled HTTP client, retry with
jittered backoff on the errors that are actually worth retrying, SSE line
parsing, and uniform error translation into :mod:`ciws.core.errors`.
"""

from __future__ import annotations

import abc
import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..core import secrets
from ..core.errors import MissingCredential, ProviderError, RateLimited
from ..core.logging import get_logger
from .types import (
    ChatRequest,
    ChatResponse,
    EmbeddingResult,
    HealthStatus,
    ModelInfo,
    StreamEvent,
    StreamEventType,
    Usage,
)

log = get_logger("gateway")

_clients: dict[str, httpx.AsyncClient] = {}

#: Status codes worth another attempt. 400/401/403/404 never are.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def http_client(key: str = "default", timeout: float = 300.0) -> httpx.AsyncClient:
    """One pooled client per provider so connections and HTTP/2 sessions are reused."""
    client = _clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0, read=timeout, write=60.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            follow_redirects=True,
            headers={"user-agent": "CIWS/0.1"},
        )
        _clients[key] = client
    return client


async def close_clients() -> None:
    for client in list(_clients.values()):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass
    _clients.clear()


async def with_retry(
    fn: Any,
    *,
    provider: str,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 12.0,
) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except RateLimited as exc:
            last = exc
            delay = exc.retry_after or min(max_delay, base_delay * (2**attempt))
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
            last = exc
            delay = min(max_delay, base_delay * (2**attempt))
        else:
            break
        if attempt == attempts - 1:
            break
        jitter = random.uniform(0, delay * 0.3)
        log.warning("%s: retry %d/%d in %.1fs (%s)", provider, attempt + 1, attempts, delay + jitter, last)
        await asyncio.sleep(delay + jitter)
    if last:
        if isinstance(last, ProviderError):
            raise last
        raise ProviderError(provider, str(last), retryable=True) from last
    raise ProviderError(provider, "Retry loop exhausted with no result")


def raise_for_status(provider: str, response: httpx.Response, body: str = "") -> None:
    if response.status_code < 400:
        return
    text = body or _safe_text(response)
    message = _extract_error_message(text) or f"HTTP {response.status_code}"
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        raise RateLimited(
            provider, message, retry_after=float(retry_after) if _isnum(retry_after) else None
        )
    if response.status_code in (401, 403):
        raise ProviderError(
            provider,
            f"Authentication failed ({response.status_code}): {message}",
            retryable=False,
            status=response.status_code,
        )
    raise ProviderError(
        provider,
        message,
        retryable=response.status_code in RETRY_STATUS,
        status=response.status_code,
    )


def _isnum(v: str | None) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text[:4000]
    except Exception:  # noqa: BLE001
        return ""


def _extract_error_message(text: str) -> str:
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    for path in (("error", "message"), ("error",), ("message",), ("detail",), ("msg",)):
        node: Any = data
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if isinstance(node, str) and node:
            return node[:500]
    return text[:500]


async def sse_lines(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed ``data:`` payloads from a Server-Sent Events stream.

    Tolerates the variations in the wild: ``event:`` lines that matter to some
    providers and not others, ``[DONE]`` sentinels, keep-alive comments, and
    multi-line data blocks.
    """
    event_name = ""
    buffer: list[str] = []

    async for raw in response.aiter_lines():
        line = raw.rstrip("\r")
        if not line:
            if buffer:
                payload = "\n".join(buffer)
                buffer.clear()
                if payload.strip() == "[DONE]":
                    event_name = ""
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    event_name = ""
                    continue
                if event_name and isinstance(data, dict):
                    data.setdefault("_event", event_name)
                yield data
            event_name = ""
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())

    if buffer:
        payload = "\n".join(buffer)
        if payload.strip() != "[DONE]":
            try:
                data = json.loads(payload)
                if event_name and isinstance(data, dict):
                    data.setdefault("_event", event_name)
                yield data
            except json.JSONDecodeError:
                pass


class Provider(abc.ABC):
    """Everything the gateway needs from a model backend."""

    #: Stable short id used in model ids as ``<id>/<model>``.
    id: str = "provider"
    label: str = "Provider"
    #: Human hint shown when the key is missing.
    env_hint: str = ""
    #: Local providers need no credential and report as offline rather than unauthorised.
    requires_key: bool = True
    local: bool = False
    website: str = ""
    supports_embeddings: bool = False

    def __init__(self) -> None:
        self._models_cache: list[ModelInfo] | None = None

    # -- credentials ------------------------------------------------------

    def api_key(self) -> str | None:
        return secrets.get(self.id)

    def require_key(self) -> str:
        key = self.api_key()
        if not key:
            raise MissingCredential(self.id, self.env_hint)
        return key

    def configured(self) -> bool:
        return (not self.requires_key) or bool(self.api_key())

    # -- chat -------------------------------------------------------------

    @abc.abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream a completion. Must always terminate with DONE or ERROR."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for type checkers

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming completion, assembled from the stream by default.

        Adapters with a cheaper unary endpoint should override this.
        """
        content: list[str] = []
        thinking: list[str] = []
        calls = []
        usage = Usage()
        finish = ""
        async for ev in self.stream_chat(request.model_copy(update={"stream": True})):
            if ev.type is StreamEventType.TEXT:
                content.append(ev.text)
            elif ev.type is StreamEventType.THINKING:
                thinking.append(ev.text)
            elif ev.type is StreamEventType.TOOL_CALL and ev.tool_call:
                calls.append(ev.tool_call)
            elif ev.type is StreamEventType.USAGE and ev.usage:
                usage = ev.usage
            elif ev.type is StreamEventType.DONE:
                finish = ev.finish_reason
            elif ev.type is StreamEventType.ERROR:
                raise ProviderError(self.id, ev.error)
        return ChatResponse(
            content="".join(content),
            thinking="".join(thinking),
            tool_calls=calls,
            usage=usage,
            model=request.model,
            provider=self.id,
            finish_reason=finish,
        )

    # -- models -----------------------------------------------------------

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Models this provider can serve right now."""

    async def models(self, refresh: bool = False) -> list[ModelInfo]:
        if self._models_cache is None or refresh:
            try:
                self._models_cache = await self.list_models()
            except Exception as exc:  # noqa: BLE001 - a dead provider must not break the catalog
                log.debug("%s: model list failed: %s", self.id, exc)
                self._models_cache = []
        return self._models_cache

    # -- embeddings -------------------------------------------------------

    async def embed(self, texts: list[str], model: str = "") -> EmbeddingResult:
        raise ProviderError(self.id, f"{self.label} does not provide embeddings", retryable=False)

    # -- health -----------------------------------------------------------

    async def health(self) -> HealthStatus:
        has_key = bool(self.api_key())
        if self.requires_key and not has_key:
            return HealthStatus(
                provider=self.id,
                ok=False,
                detail=f"No API key. {self.env_hint}".strip(),
                requires_key=True,
                has_key=False,
            )
        import time

        t0 = time.perf_counter()
        try:
            # Deliberately not self.models(): that wrapper swallows errors so a
            # dead provider cannot break the catalogue, which is exactly the
            # wrong behaviour for a health check -- an unreachable local server
            # would report "up" with zero models.
            models = await self.list_models()
            self._models_cache = models
            reachable = bool(models)
            return HealthStatus(
                provider=self.id,
                ok=reachable,
                detail=(
                    f"{len(models)} models available"
                    if reachable
                    else "Reachable, but it reports no models"
                ),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                model_count=len(models),
                requires_key=self.requires_key,
                has_key=has_key,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(
                provider=self.id,
                ok=False,
                detail=str(exc)[:300],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                requires_key=self.requires_key,
                has_key=has_key,
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Provider {self.id}>"
