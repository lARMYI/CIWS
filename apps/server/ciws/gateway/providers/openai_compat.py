"""One adapter for every provider that speaks the OpenAI chat-completions dialect.

OpenAI, Groq, Mistral, DeepSeek, xAI, OpenRouter, Together, Perplexity,
LM Studio and vLLM all expose ``/v1/chat/completions`` with the same request
and SSE shapes. Rather than ten near-identical files, they are subclasses that
change a base URL, a credential name and a few flags.

The variations that actually bite are handled here:

* ``stream_options.include_usage`` is an OpenAI extension. Servers that do not
  know it reject the whole request, so it is opt-in per provider.
* Streamed tool calls arrive as indexed fragments -- ``id`` and ``name`` on the
  first chunk, ``arguments`` dribbling in as partial JSON afterwards. They have
  to be reassembled by index, not by id.
* Reasoning models expose their chain of thought under three different keys
  (``reasoning_content``, ``reasoning``, ``thinking``) depending on vendor.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.logging import get_logger
from ...core.util import safe_json
from .. import catalog
from ..base import Provider, http_client, raise_for_status, sse_lines
from ..types import (
    Capability,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingResult,
    Modality,
    ModelInfo,
    PartType,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)

log = get_logger("gateway.openai_compat")


class OpenAICompatProvider(Provider):
    """Base for any OpenAI-dialect endpoint."""

    base_url = "https://api.openai.com/v1"
    #: Some gateways reject unknown request keys outright.
    supports_stream_usage = True
    supports_tools = True
    supports_json_schema = True
    supports_embeddings = False
    embedding_model = ""
    #: Extra headers (OpenRouter wants attribution headers, for example).
    extra_headers: dict[str, str] = {}
    #: Reasoning models use a different max-tokens key.
    max_tokens_key = "max_tokens"

    def headers(self) -> dict[str, str]:
        h = {"content-type": "application/json", **self.extra_headers}
        key = self.api_key()
        if key:
            h["authorization"] = f"Bearer {key}"
        elif self.requires_key:
            self.require_key()  # raises MissingCredential with the right hint
        return h

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    # -- message translation ---------------------------------------------

    def _content_for(self, msg: ChatMessage) -> Any:
        """Plain string when there is only text; block list when there is media."""
        if not msg.parts:
            return msg.content
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for part in msg.parts:
            if part.type is PartType.TEXT and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif part.type is PartType.IMAGE:
                url = part.url or f"data:{part.mime_type or 'image/png'};base64,{part.data}"
                blocks.append({"type": "image_url", "image_url": {"url": url}})
        return blocks or msg.content

    def to_wire_messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        system_text = request.system or "\n\n".join(
            m.content for m in request.messages if m.role == "system" and m.content
        )
        if system_text:
            out.append({"role": self.system_role(), "content": system_text})

        for msg in request.messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                for res in msg.tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": res.call_id,
                            "content": res.content or "(no output)",
                        }
                    )
                continue
            if msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}
                entry["content"] = msg.content or None
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.raw_arguments or json.dumps(c.arguments),
                            },
                        }
                        for c in msg.tool_calls
                    ]
                # An assistant turn with neither content nor tool calls is rejected.
                if entry["content"] is None and "tool_calls" not in entry:
                    continue
                out.append(entry)
                continue
            out.append({"role": "user", "content": self._content_for(msg)})
        return out

    def system_role(self) -> str:
        return "system"

    def build_payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        _, model_name = catalog.split_id(request.model)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": self.to_wire_messages(request),
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens:
            payload[self.max_tokens_key] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop[:4]
        if request.seed is not None:
            payload["seed"] = request.seed
        if stream and self.supports_stream_usage:
            payload["stream_options"] = {"include_usage": True}

        if request.tools and self.supports_tools:
            payload["tools"] = [t.openai_dict() for t in request.tools]
            if request.tool_choice in ("auto", "none", "required"):
                payload["tool_choice"] = request.tool_choice
            elif request.tool_choice:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": request.tool_choice},
                }

        if request.json_schema and self.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }
        elif request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    # -- streaming --------------------------------------------------------

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request, stream=True)
        client = http_client(self.id, timeout=request.timeout_s)
        started = time.perf_counter()

        yield StreamEvent(type=StreamEventType.START, model=request.model)

        # index -> partially assembled call
        pending: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        finish = ""

        async with client.stream(
            "POST", self.endpoint("chat/completions"), json=payload, headers=self.headers()
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise_for_status(self.id, response, body)

            async for chunk in sse_lines(response):
                if "error" in chunk and chunk.get("error"):
                    err = chunk["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    yield StreamEvent(type=StreamEventType.ERROR, error=str(msg)[:400])
                    return

                if chunk.get("usage"):
                    usage = self._usage_from(chunk["usage"], request.model)

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}

                    text = delta.get("content")
                    if isinstance(text, list):  # a few gateways send block lists
                        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
                    if text:
                        yield StreamEvent(type=StreamEventType.TEXT, text=text)

                    for key in ("reasoning_content", "reasoning", "thinking"):
                        reasoning = delta.get(key)
                        if isinstance(reasoning, str) and reasoning:
                            yield StreamEvent(type=StreamEventType.THINKING, text=reasoning)
                            break

                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                            yield StreamEvent(
                                type=StreamEventType.TOOL_ARGS_DELTA,
                                text=fn["arguments"],
                                index=idx,
                            )

                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]

        for idx in sorted(pending):
            slot = pending[idx]
            if not slot["name"]:
                continue
            args = safe_json(slot["args"], default=None)
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL,
                tool_call=ToolCall(
                    id=slot["id"] or f"call_{idx}",
                    name=slot["name"],
                    arguments=args if isinstance(args, dict) else {},
                    raw_arguments=slot["args"],
                ),
                index=idx,
            )

        if usage is None:
            usage = Usage()
        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE,
            finish_reason=finish,
            usage=usage,
            model=request.model,
            index=int((time.perf_counter() - started) * 1000),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self.build_payload(request, stream=False)
        client = http_client(self.id, timeout=request.timeout_s)
        started = time.perf_counter()
        response = await client.post(
            self.endpoint("chat/completions"), json=payload, headers=self.headers()
        )
        raise_for_status(self.id, response)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = safe_json(fn.get("arguments", "{}"), default={})
            calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {},
                    raw_arguments=fn.get("arguments", ""),
                )
            )

        thinking = ""
        for key in ("reasoning_content", "reasoning", "thinking"):
            if isinstance(message.get(key), str):
                thinking = message[key]
                break

        return ChatResponse(
            content=message.get("content") or "",
            thinking=thinking,
            tool_calls=calls,
            usage=self._usage_from(data.get("usage") or {}, request.model),
            model=request.model,
            provider=self.id,
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _usage_from(self, raw: dict[str, Any], model_id: str) -> Usage:
        details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        usage = Usage(
            input_tokens=int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or raw.get("output_tokens") or 0),
            cache_read_tokens=int(details.get("cached_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )
        # OpenRouter reports actual dollars spent; trust it over our table.
        if isinstance(raw.get("cost"), (int, float)):
            usage.cost_usd = float(raw["cost"])
        else:
            info = catalog.lookup(model_id)
            if info:
                usage.cost_usd = info.cost_for(usage)
        return usage

    # -- discovery --------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        client = http_client(self.id, timeout=30)
        response = await client.get(self.endpoint("models"), headers=self.headers())
        raise_for_status(self.id, response)
        data = response.json()
        rows = data.get("data") if isinstance(data, dict) else data
        models = [self.model_from_row(r) for r in (rows or []) if isinstance(r, dict)]
        models = [m for m in models if m is not None]
        return catalog.merge(models, self.id)

    def model_from_row(self, row: dict[str, Any]) -> ModelInfo | None:
        name = row.get("id") or row.get("name")
        if not name:
            return None
        ctx = int(
            row.get("context_length")
            or row.get("context_window")
            or (row.get("top_provider") or {}).get("context_length")
            or 0
        )
        caps = [Capability.STREAMING]
        if self.supports_tools:
            caps.append(Capability.TOOLS)
        if self.supports_json_schema:
            caps.append(Capability.JSON)
        mods = [Modality.TEXT]

        architecture = row.get("architecture") or {}
        inputs = architecture.get("input_modalities") or []
        if "image" in inputs or "vision" in str(row.get("id", "")).lower():
            caps.append(Capability.VISION)
            mods.append(Modality.IMAGE)

        pricing = row.get("pricing") or {}

        def per_mtok(key: str) -> float:
            # OpenRouter quotes dollars per token as a string.
            val = pricing.get(key)
            try:
                return float(val) * 1_000_000 if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        return ModelInfo(
            id=f"{self.id}/{name}",
            provider=self.id,
            name=str(name),
            display_name=row.get("name") if row.get("name") != name else str(name),
            family=str(row.get("owned_by") or self.id),
            context_window=ctx,
            max_output=int((row.get("top_provider") or {}).get("max_completion_tokens") or 0),
            modalities=mods,
            capabilities=caps,
            input_cost_per_mtok=per_mtok("prompt"),
            output_cost_per_mtok=per_mtok("completion"),
            local=self.local,
            description=str(row.get("description") or "")[:400],
        )

    # -- embeddings -------------------------------------------------------

    async def embed(self, texts: list[str], model: str = "") -> EmbeddingResult:
        if not self.supports_embeddings:
            return await super().embed(texts, model)
        model_name = model or self.embedding_model
        if "/" in model_name:
            _, model_name = catalog.split_id(model_name)
        client = http_client(self.id, timeout=120)
        response = await client.post(
            self.endpoint("embeddings"),
            json={"model": model_name, "input": texts},
            headers=self.headers(),
        )
        raise_for_status(self.id, response)
        data = response.json()
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = [r["embedding"] for r in rows]
        return EmbeddingResult(
            vectors=vectors,
            model=f"{self.id}/{model_name}",
            dim=len(vectors[0]) if vectors else 0,
            usage=self._usage_from(data.get("usage") or {}, f"{self.id}/{model_name}"),
        )


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class OpenAIProvider(OpenAICompatProvider):
    id = "openai"
    label = "OpenAI"
    env_hint = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"
    supports_embeddings = True
    embedding_model = "text-embedding-3-small"
    website = "https://platform.openai.com"

    def build_payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        payload = super().build_payload(request, stream)
        _, name = catalog.split_id(request.model)
        # The o-series and gpt-5 reasoning models rename max_tokens and reject sampling.
        if name.startswith(("o1", "o3", "o4", "gpt-5")):
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        return payload


class GroqProvider(OpenAICompatProvider):
    id = "groq"
    label = "Groq"
    env_hint = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"
    supports_json_schema = False
    website = "https://console.groq.com"


class MistralProvider(OpenAICompatProvider):
    id = "mistral"
    label = "Mistral"
    env_hint = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1"
    supports_stream_usage = False
    supports_json_schema = False
    supports_embeddings = True
    embedding_model = "mistral-embed"
    website = "https://console.mistral.ai"


class DeepSeekProvider(OpenAICompatProvider):
    id = "deepseek"
    label = "DeepSeek"
    env_hint = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com/v1"
    supports_json_schema = False
    website = "https://platform.deepseek.com"


class XAIProvider(OpenAICompatProvider):
    id = "xai"
    label = "xAI Grok"
    env_hint = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
    website = "https://console.x.ai"


class OpenRouterProvider(OpenAICompatProvider):
    id = "openrouter"
    label = "OpenRouter"
    env_hint = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"
    website = "https://openrouter.ai"
    extra_headers = {
        "HTTP-Referer": "https://github.com/lARMYI/CIWS",
        "X-Title": "CIWS",
    }


class TogetherProvider(OpenAICompatProvider):
    id = "together"
    label = "Together AI"
    env_hint = "TOGETHER_API_KEY"
    base_url = "https://api.together.xyz/v1"
    supports_embeddings = True
    embedding_model = "BAAI/bge-base-en-v1.5"
    website = "https://api.together.ai"


class PerplexityProvider(OpenAICompatProvider):
    id = "perplexity"
    label = "Perplexity"
    env_hint = "PERPLEXITY_API_KEY"
    base_url = "https://api.perplexity.ai"
    supports_tools = False
    supports_json_schema = False
    supports_stream_usage = False
    website = "https://docs.perplexity.ai"

    async def list_models(self) -> list[ModelInfo]:
        # Perplexity has no /models endpoint; its online models are documented only.
        return [
            ModelInfo(
                id=f"{self.id}/{name}", provider=self.id, name=name, display_name=display,
                family="sonar", context_window=127_072, modalities=[Modality.TEXT],
                capabilities=[Capability.STREAMING],
                description="Web-grounded answers with citations.",
            )
            for name, display in (
                ("sonar", "Sonar"),
                ("sonar-pro", "Sonar Pro"),
                ("sonar-reasoning", "Sonar Reasoning"),
            )
        ]


class LMStudioProvider(OpenAICompatProvider):
    id = "lmstudio"
    label = "LM Studio"
    requires_key = False
    local = True
    supports_stream_usage = False
    supports_json_schema = False
    website = "https://lmstudio.ai"

    def __init__(self) -> None:
        super().__init__()
        from ...core.config import get_settings

        self.base_url = get_settings().lmstudio_url


class VLLMProvider(OpenAICompatProvider):
    id = "vllm"
    label = "vLLM"
    requires_key = False
    local = True
    supports_stream_usage = False
    website = "https://docs.vllm.ai"

    def __init__(self) -> None:
        super().__init__()
        from ...core.config import get_settings

        self.base_url = get_settings().vllm_url or "http://127.0.0.1:8000/v1"

    def configured(self) -> bool:
        from ...core.config import get_settings

        return bool(get_settings().vllm_url)
