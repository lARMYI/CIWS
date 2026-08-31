"""The gateway: one entry point for every model in the workspace.

Callers ask for ``"anthropic/claude-opus-5"``, or ``"deep"``, or just
``"opus"``, and get a stream back. The registry handles:

* resolving aliases and capability names against the routing policy
* picking a fallback when the first choice is unconfigured or failing
* recording usage, latency and cost per model
* keeping provider health fresh enough for the Models panel

Nothing above this module knows which provider serves a request.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select

from ..core.config import get_settings
from ..core.errors import ConfigError, MissingCredential, ProviderError
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..db.base import session_scope
from ..db.models import ModelRecord
from . import catalog
from .base import Provider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.ollama import OllamaProvider
from .providers.openai_compat import (
    DeepSeekProvider,
    GroqProvider,
    LMStudioProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
    TogetherProvider,
    VLLMProvider,
    XAIProvider,
)
from .types import (
    Capability,
    ChatRequest,
    ChatResponse,
    EmbeddingResult,
    HealthStatus,
    ModelInfo,
    StreamEvent,
    StreamEventType,
    Usage,
)

log = get_logger("gateway.registry")

PROVIDER_CLASSES: list[type[Provider]] = [
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    OllamaProvider,
    OpenRouterProvider,
    GroqProvider,
    XAIProvider,
    MistralProvider,
    DeepSeekProvider,
    TogetherProvider,
    PerplexityProvider,
    LMStudioProvider,
    VLLMProvider,
]

#: Capability names the routing policy understands, in place of a model id.
ROUTE_NAMES = {"fast", "balanced", "deep", "vision", "cheap", "local", "embed"}


class Gateway:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._models: dict[str, ModelInfo] = {}
        self._models_loaded_at: float = 0.0
        self._health: dict[str, HealthStatus] = {}
        self._lock = asyncio.Lock()

    # -- providers --------------------------------------------------------

    def providers(self) -> dict[str, Provider]:
        if not self._providers:
            for cls in PROVIDER_CLASSES:
                try:
                    inst = cls()
                    self._providers[inst.id] = inst
                except Exception as exc:  # noqa: BLE001 - one bad provider must not kill the hub
                    log.warning("Provider %s failed to initialise: %s", cls.__name__, exc)
        return self._providers

    def provider(self, provider_id: str) -> Provider:
        p = self.providers().get(provider_id)
        if p is None:
            raise ConfigError(f"Unknown provider '{provider_id}'")
        return p

    def configured_providers(self) -> list[Provider]:
        return [p for p in self.providers().values() if p.configured()]

    # -- model resolution -------------------------------------------------

    def resolve(self, model: str) -> str:
        """Turn a name, alias or capability into a concrete ``provider/model`` id."""
        if not model:
            model = "balanced"
        settings = get_settings()

        if model in ROUTE_NAMES:
            model = getattr(settings.routing, model, "") or settings.routing.balanced

        model = catalog.ALIASES.get(model, model)

        if "/" in model:
            return model

        # Bare model name: prefer a configured provider that actually lists it.
        matches = [mid for mid in self._models if mid.split("/", 1)[1] == model]
        for mid in matches:
            if self.providers().get(mid.split("/", 1)[0], None) and self.provider(
                mid.split("/", 1)[0]
            ).configured():
                return mid
        if matches:
            return matches[0]

        seeded = catalog.lookup(model)
        if seeded:
            return seeded.id
        raise ConfigError(
            f"Cannot resolve model '{model}'. Use 'provider/model', or one of: "
            f"{', '.join(sorted(ROUTE_NAMES))}."
        )

    async def models(self, refresh: bool = False) -> list[ModelInfo]:
        """Every model across every configured provider."""
        async with self._lock:
            stale = (time.time() - self._models_loaded_at) > 300
            if self._models and not refresh and not stale:
                return list(self._models.values())

            results = await asyncio.gather(
                *(p.models(refresh=refresh) for p in self.configured_providers()),
                return_exceptions=True,
            )
            merged: dict[str, ModelInfo] = {}
            for res in results:
                if isinstance(res, BaseException):
                    continue
                for m in res:
                    merged[m.id] = m
            # Always keep the seed catalogue visible so an unconfigured provider
            # still shows what it *would* offer once a key is added.
            for m in catalog.SEED_MODELS:
                merged.setdefault(m.id, m)
            self._models = merged
            self._models_loaded_at = time.time()

        await self._sync_records()
        return list(self._models.values())

    def model_info(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id) or catalog.lookup(model_id)

    async def _sync_records(self) -> None:
        """Mirror the catalogue into the DB so usage stats survive a restart."""
        try:
            async with session_scope() as s:
                existing = {
                    r.id: r for r in (await s.execute(select(ModelRecord))).scalars().all()
                }
                for m in self._models.values():
                    rec = existing.get(m.id)
                    if rec is None:
                        s.add(
                            ModelRecord(
                                id=m.id,
                                provider=m.provider,
                                name=m.name,
                                display_name=m.display_name or m.name,
                                family=m.family,
                                context_window=m.context_window,
                                max_output=m.max_output,
                                modalities=[x.value for x in m.modalities],
                                capabilities=[x.value for x in m.capabilities],
                                input_cost_per_mtok=m.input_cost_per_mtok,
                                output_cost_per_mtok=m.output_cost_per_mtok,
                                local=m.local,
                                meta=m.meta,
                            )
                        )
                    else:
                        rec.context_window = m.context_window or rec.context_window
                        rec.max_output = m.max_output or rec.max_output
                        rec.display_name = m.display_name or rec.display_name
                        rec.modalities = [x.value for x in m.modalities]
                        rec.capabilities = [x.value for x in m.capabilities]
                        # A price typed by the user in the UI wins over a re-seed.
                        if not rec.meta.get("price_overridden"):
                            rec.input_cost_per_mtok = m.input_cost_per_mtok
                            rec.output_cost_per_mtok = m.output_cost_per_mtok
        except Exception as exc:  # noqa: BLE001
            log.debug("Model record sync skipped: %s", exc)

    # -- chat -------------------------------------------------------------

    def _fallback_chain(self, model_id: str) -> list[str]:
        settings = get_settings()
        chain = [model_id]
        for candidate in settings.routing.fallbacks:
            try:
                resolved = self.resolve(candidate)
            except ConfigError:
                continue
            if resolved not in chain:
                chain.append(resolved)
        return chain

    async def stream(
        self, request: ChatRequest, *, allow_fallback: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion, falling back down the chain on hard failures.

        Fallback only fires before any token has been emitted. Once the caller
        has seen output, switching models mid-answer would splice two different
        voices together, so a late failure is surfaced as an error instead.
        """
        target = self.resolve(request.model)
        chain = self._fallback_chain(target) if allow_fallback else [target]
        last_error: Exception | None = None

        for attempt, model_id in enumerate(chain):
            provider_id, _ = catalog.split_id(model_id)
            try:
                provider = self.provider(provider_id)
            except ConfigError as exc:
                last_error = exc
                continue

            req = request.model_copy(update={"model": model_id})
            emitted = False
            started = time.perf_counter()
            usage = Usage()

            try:
                async for ev in provider.stream_chat(req):
                    if ev.type in (StreamEventType.TEXT, StreamEventType.THINKING, StreamEventType.TOOL_CALL):
                        emitted = True
                    if ev.type is StreamEventType.USAGE and ev.usage:
                        usage = ev.usage
                    if ev.type is StreamEventType.ERROR and not emitted and attempt < len(chain) - 1:
                        last_error = ProviderError(provider_id, ev.error)
                        break
                    ev.model = ev.model or model_id
                    yield ev
                else:
                    await self._record_usage(
                        model_id, usage, int((time.perf_counter() - started) * 1000), ok=True
                    )
                    return
            except (ProviderError, MissingCredential) as exc:
                last_error = exc
                await self._record_usage(model_id, usage, 0, ok=False)
                if emitted or attempt == len(chain) - 1:
                    yield StreamEvent(type=StreamEventType.ERROR, error=str(exc), model=model_id)
                    return
                log.warning("%s failed (%s); falling back", model_id, exc)
                bus.publish(
                    Topic.MODEL_STATUS, model=model_id, ok=False, detail=str(exc)[:200]
                )
                continue

        yield StreamEvent(
            type=StreamEventType.ERROR,
            error=str(last_error) if last_error else "No provider could serve this request.",
        )

    async def chat(self, request: ChatRequest, *, allow_fallback: bool = True) -> ChatResponse:
        content: list[str] = []
        thinking: list[str] = []
        calls = []
        usage = Usage()
        finish = ""
        model_used = request.model
        error = ""

        async for ev in self.stream(request, allow_fallback=allow_fallback):
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
                model_used = ev.model or model_used
            elif ev.type is StreamEventType.ERROR:
                error = ev.error

        if error and not content and not calls:
            raise ProviderError(catalog.split_id(model_used)[0] or "gateway", error)

        return ChatResponse(
            content="".join(content),
            thinking="".join(thinking),
            tool_calls=calls,
            usage=usage,
            model=model_used,
            provider=catalog.split_id(model_used)[0],
            finish_reason=finish,
        )

    async def complete(
        self, prompt: str, *, model: str = "fast", system: str = "", max_tokens: int = 2000, **kw: Any
    ) -> str:
        """One-shot text in, text out. Used by internal utilities."""
        from .types import ChatMessage

        res = await self.chat(
            ChatRequest(
                model=model,
                system=system,
                messages=[ChatMessage.user(prompt)],
                max_tokens=max_tokens,
                **kw,
            )
        )
        return res.content

    # -- embeddings -------------------------------------------------------

    async def embed(self, texts: list[str], model: str = "") -> EmbeddingResult:
        model_id = model or get_settings().routing.embed
        provider_id, model_name = catalog.split_id(model_id)
        provider = self.provider(provider_id)
        return await provider.embed(texts, model_name)

    # -- health & telemetry ----------------------------------------------

    async def health(self, refresh: bool = False) -> dict[str, HealthStatus]:
        if self._health and not refresh:
            return self._health
        results = await asyncio.gather(
            *(p.health() for p in self.providers().values()), return_exceptions=True
        )
        out: dict[str, HealthStatus] = {}
        for provider, res in zip(self.providers().values(), results):
            if isinstance(res, BaseException):
                out[provider.id] = HealthStatus(
                    provider=provider.id, ok=False, detail=str(res)[:200]
                )
            else:
                out[provider.id] = res
        self._health = out
        return out

    async def _record_usage(self, model_id: str, usage: Usage, latency_ms: int, ok: bool) -> None:
        try:
            async with session_scope() as s:
                rec = (
                    await s.execute(select(ModelRecord).where(ModelRecord.id == model_id))
                ).scalar_one_or_none()
                if rec is None:
                    return
                rec.call_count += 1
                if not ok:
                    rec.error_count += 1
                    rec.status = "error"
                else:
                    rec.status = "ok"
                    rec.total_input_tokens += usage.input_tokens
                    rec.total_output_tokens += usage.output_tokens
                    rec.total_cost_usd += usage.cost_usd
                    if latency_ms:
                        n = max(1, rec.call_count)
                        rec.avg_latency_ms = (
                            rec.avg_latency_ms * (n - 1) + latency_ms
                        ) / n
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a request
            log.debug("Usage record skipped: %s", exc)

    async def pick_by_capability(self, cap: Capability, prefer_local: bool = False) -> str | None:
        """First configured model advertising a capability. Used for graceful degradation."""
        models = await self.models()
        candidates = [m for m in models if cap in m.capabilities]
        if prefer_local:
            candidates.sort(key=lambda m: (not m.local, m.input_cost_per_mtok))
        else:
            candidates.sort(key=lambda m: (m.local, m.input_cost_per_mtok))
        for m in candidates:
            provider = self.providers().get(m.provider)
            if provider and provider.configured():
                return m.id
        return None


#: Process-wide gateway.
gateway = Gateway()
