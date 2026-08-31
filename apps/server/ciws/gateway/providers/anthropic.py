"""Anthropic adapter, built on the official ``anthropic`` SDK.

The SDK is used rather than raw HTTP because this API surface moves: adaptive
thinking, effort levels, refusal stop reasons and streaming block types all
changed shape recently, and the SDK tracks them.

The one thing this adapter must get exactly right is that the current models
*reject* parameters the older ones require -- ``temperature`` and
``budget_tokens`` return a 400 on Opus 5, Opus 4.7/4.8, Sonnet 5 and Fable 5.
Those flags live in :mod:`ciws.gateway.catalog` and are applied in
:meth:`AnthropicProvider._build_kwargs`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.errors import ProviderError, RateLimited
from ...core.logging import get_logger
from ...core.util import estimate_tokens
from .. import catalog
from ..base import Provider
from ..types import (
    Capability,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ContentPart,
    Modality,
    ModelInfo,
    PartType,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)

log = get_logger("gateway.anthropic")

#: Cache the system prompt only when it is long enough to beat the ~1024 token
#: minimum cacheable prefix; below that the breakpoint is wasted.
CACHE_SYSTEM_MIN_CHARS = 4000


class AnthropicProvider(Provider):
    id = "anthropic"
    label = "Anthropic"
    env_hint = "ANTHROPIC_API_KEY"
    website = "https://console.anthropic.com"

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._client_key: str | None = None

    # -- client -----------------------------------------------------------

    def _sdk(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                self.id,
                "The 'anthropic' package is not installed. Run: pip install anthropic",
                retryable=False,
            ) from exc
        return anthropic

    def client(self) -> Any:
        key = self.require_key()
        if self._client is None or self._client_key != key:
            self._client = self._sdk().AsyncAnthropic(api_key=key, max_retries=0)
            self._client_key = key
        return self._client

    # -- message translation ---------------------------------------------

    def _content_blocks(self, msg: ChatMessage) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []

        # Thinking must lead the assistant turn and be echoed back verbatim
        # (signature included) for the model to continue its own reasoning.
        if msg.role == "assistant" and msg.thinking and msg.signature:
            blocks.append(
                {"type": "thinking", "thinking": msg.thinking, "signature": msg.signature}
            )

        if msg.content:
            blocks.append({"type": "text", "text": msg.content})

        for part in msg.parts:
            if part.type is PartType.TEXT and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif part.type is PartType.IMAGE:
                if part.data:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.mime_type or "image/png",
                                "data": part.data,
                            },
                        }
                    )
                elif part.url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": part.url}})
            elif part.type is PartType.FILE and part.data:
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": part.mime_type or "application/pdf",
                            "data": part.data,
                        },
                    }
                )

        for call in msg.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )

        return blocks

    def _to_anthropic_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue  # hoisted into the top-level system parameter
            if msg.role == "tool":
                # Every tool_result for a turn goes back in ONE user message --
                # splitting them teaches the model to stop calling tools in parallel.
                results: list[dict[str, Any]] = []
                for res in msg.tool_results:
                    content: Any = res.content or "(no output)"
                    if res.parts:
                        content = [{"type": "text", "text": res.content or "(no output)"}]
                        for part in res.parts:
                            if part.type is PartType.IMAGE and part.data:
                                content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": part.mime_type or "image/png",
                                            "data": part.data,
                                        },
                                    }
                                )
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": res.call_id,
                        "content": content,
                    }
                    if res.is_error:
                        block["is_error"] = True
                    results.append(block)
                if results:
                    out.append({"role": "user", "content": results})
                continue

            blocks = self._content_blocks(msg)
            if not blocks:
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            # Anthropic rejects consecutive same-role turns; fold them together.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return out

    def _build_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        _, model_name = catalog.split_id(request.model)
        info = catalog.lookup(request.model)
        q = catalog.quirks(request.model)

        max_out = info.max_output if info and info.max_output else 32_000
        max_tokens = min(request.max_tokens or 32_000, max_out)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": self._to_anthropic_messages(request.messages),
        }

        system_text = request.system or "\n\n".join(
            m.content for m in request.messages if m.role == "system" and m.content
        )
        if system_text:
            if len(system_text) >= CACHE_SYSTEM_MIN_CHARS:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system_text

        # Sampling parameters were removed on the current generation.
        if q.get("sampling_allowed") and request.temperature is not None:
            kwargs["temperature"] = max(0.0, min(1.0, request.temperature))
        if q.get("sampling_allowed") and request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop:
            kwargs["stop_sequences"] = request.stop[:4]

        # Thinking: adaptive on current models, a token budget on older ones.
        if q.get("thinking_always_on"):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif request.thinking_budget > 0:
            if q.get("adaptive_thinking"):
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            else:
                budget = max(1024, min(request.thinking_budget, max_tokens - 1024))
                if budget >= 1024:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    kwargs.pop("temperature", None)  # incompatible with thinking on older models
                    kwargs.pop("top_p", None)

        if request.tools:
            tools = [t.anthropic_dict() for t in request.tools]
            # One cache breakpoint on the last tool covers the whole tool block,
            # which is the most stable prefix in an agent loop.
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools
            if request.tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}
            elif request.tool_choice not in ("auto", "none"):
                kwargs["tool_choice"] = {"type": "tool", "name": request.tool_choice}
            elif request.tool_choice == "auto":
                kwargs["tool_choice"] = {"type": "auto"}

        if request.json_schema:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.json_schema,
                }
            }
        return kwargs

    # -- streaming --------------------------------------------------------

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        anthropic = self._sdk()
        client = self.client()
        kwargs = self._build_kwargs(request)
        started = time.perf_counter()

        yield StreamEvent(type=StreamEventType.START, model=request.model)

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            yield StreamEvent(type=StreamEventType.TEXT, text=delta.text)
                        elif dtype == "thinking_delta":
                            yield StreamEvent(
                                type=StreamEventType.THINKING, text=getattr(delta, "thinking", "")
                            )
                        elif dtype == "input_json_delta":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_ARGS_DELTA,
                                text=getattr(delta, "partial_json", ""),
                                index=getattr(event, "index", 0),
                            )
                    elif etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block is not None and getattr(block, "type", "") == "tool_use":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_ARGS_DELTA,
                                text="",
                                index=getattr(event, "index", 0),
                                tool_call=ToolCall(
                                    id=getattr(block, "id", ""), name=getattr(block, "name", "")
                                ),
                            )

                final = await stream.get_final_message()

        except anthropic.RateLimitError as exc:
            retry_after = None
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            raw = headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except (TypeError, ValueError):
                    retry_after = None
            raise RateLimited(self.id, str(exc)[:300], retry_after=retry_after) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(self.id, f"Connection failed: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                self.id,
                f"{exc.status_code}: {str(exc)[:300]}",
                retryable=exc.status_code >= 500,
                status=exc.status_code,
            ) from exc

        stop_reason = getattr(final, "stop_reason", "") or ""

        # A refusal is an HTTP 200 with no usable content -- surface it plainly
        # rather than letting the agent loop spin on an empty assistant turn.
        if stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=f"The model declined this request (category: {category}).",
                finish_reason="refusal",
            )
            return

        for block in getattr(final, "content", []) or []:
            if getattr(block, "type", "") == "tool_use":
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    tool_call=ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    ),
                )

        usage = self._usage_from(final, request.model)
        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE,
            finish_reason=stop_reason,
            usage=usage,
            model=request.model,
            index=int((time.perf_counter() - started) * 1000),
        )

    def _usage_from(self, final: Any, model_id: str) -> Usage:
        raw = getattr(final, "usage", None)
        usage = Usage(
            input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
        )
        info = catalog.lookup(model_id)
        if info:
            # Cache reads bill at a discount and writes at a premium; without the
            # per-model multipliers we would rather under-report than invent a
            # number, so the plain input/output rate is used.
            usage.cost_usd = info.cost_for(usage)
        return usage

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Unary completion.

        Still streamed under the hood: the SDK refuses non-streaming requests
        whose ``max_tokens`` could exceed the HTTP timeout, and this hub uses
        large output budgets by default.
        """
        content: list[str] = []
        thinking: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        finish = ""
        started = time.perf_counter()

        async for ev in self.stream_chat(request):
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
                raise ProviderError(self.id, ev.error, retryable=False)

        return ChatResponse(
            content="".join(content),
            thinking="".join(thinking),
            tool_calls=calls,
            usage=usage,
            model=request.model,
            provider=self.id,
            finish_reason=finish,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # -- discovery --------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """Live model list, enriched with seeded pricing and request quirks."""
        client = self.client()
        live: list[ModelInfo] = []
        try:
            async for m in client.models.list(limit=100):
                live.append(self._model_from_api(m))
        except Exception as exc:  # noqa: BLE001 - fall back to the seed catalogue
            log.debug("anthropic: live model list unavailable (%s)", exc)
            return catalog.seed_for(self.id)
        return catalog.merge(live, self.id) if live else catalog.seed_for(self.id)

    def _model_from_api(self, m: Any) -> ModelInfo:
        caps_raw = getattr(m, "capabilities", None) or {}
        caps = [Capability.TOOLS, Capability.STREAMING, Capability.JSON, Capability.CACHING]
        mods = [Modality.TEXT]

        def supported(*path: str) -> bool:
            node: Any = caps_raw
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    return False
                node = node[key]
            return bool(node.get("supported")) if isinstance(node, dict) else bool(node)

        if supported("image_input"):
            caps.append(Capability.VISION)
            mods.append(Modality.IMAGE)
        if supported("thinking"):
            caps.append(Capability.THINKING)

        model_id = getattr(m, "id", "")
        return ModelInfo(
            id=f"{self.id}/{model_id}",
            provider=self.id,
            name=model_id,
            display_name=getattr(m, "display_name", "") or model_id,
            family="claude",
            context_window=int(getattr(m, "max_input_tokens", 0) or 0),
            max_output=int(getattr(m, "max_tokens", 0) or 0),
            modalities=mods,
            capabilities=caps,
            meta={
                "adaptive_thinking": supported("thinking", "types", "adaptive"),
                "sampling_allowed": not supported("thinking", "types", "adaptive"),
                "effort_supported": supported("effort"),
                "prefill_allowed": False,
            },
        )

    async def count_tokens(self, request: ChatRequest) -> int:
        """Exact token count from the API, falling back to an estimate."""
        try:
            client = self.client()
            kwargs = self._build_kwargs(request)
            result = await client.messages.count_tokens(
                model=kwargs["model"],
                messages=kwargs["messages"],
                **({"system": kwargs["system"]} if "system" in kwargs else {}),
                **({"tools": kwargs["tools"]} if "tools" in kwargs else {}),
            )
            return int(getattr(result, "input_tokens", 0) or 0)
        except Exception:  # noqa: BLE001
            return estimate_tokens(
                request.system + "".join(m.text() for m in request.messages)
            )
