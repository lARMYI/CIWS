"""Ollama adapter -- models running on your own machine.

Ollama is the reason this hub can work with no API keys and no network. It
streams newline-delimited JSON rather than SSE, and reports token counts only
in the final ``done`` object.

Cost is always zero. That is not a placeholder: local inference has no
per-token price, and reporting it as such is what makes the cost panel useful
for deciding what to run locally.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.util import human_bytes, new_id
from .. import catalog
from ..base import Provider, http_client, raise_for_status
from ..types import (
    Capability,
    ChatRequest,
    EmbeddingResult,
    Modality,
    ModelInfo,
    PartType,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)

log = get_logger("gateway.ollama")


class OllamaProvider(Provider):
    id = "ollama"
    label = "Ollama (local)"
    requires_key = False
    local = True
    supports_embeddings = True
    website = "https://ollama.com"

    @property
    def base_url(self) -> str:
        return get_settings().ollama_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    # -- translation ------------------------------------------------------

    def _messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        system_text = request.system or "\n\n".join(
            m.content for m in request.messages if m.role == "system" and m.content
        )
        if system_text:
            out.append({"role": "system", "content": system_text})

        for msg in request.messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                for res in msg.tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "content": res.content or "(no output)",
                            "tool_name": res.name or "",
                        }
                    )
                continue

            entry: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            images = [
                p.data for p in msg.parts if p.type is PartType.IMAGE and p.data
            ]
            if images:
                entry["images"] = images
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in msg.tool_calls
                ]
            out.append(entry)
        return out

    def _payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        _, model_name = catalog.split_id(request.model)
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        if request.seed is not None:
            options["seed"] = request.seed

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._messages(request),
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = [t.openai_dict() for t in request.tools]
        if request.json_schema:
            payload["format"] = request.json_schema
        elif request.json_mode:
            payload["format"] = "json"
        return payload

    # -- streaming --------------------------------------------------------

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(request, stream=True)
        client = http_client(self.id, timeout=request.timeout_s)
        started = time.perf_counter()

        yield StreamEvent(type=StreamEventType.START, model=request.model)

        usage = Usage()
        finish = ""
        calls: list[ToolCall] = []

        async with client.stream("POST", self._url("/api/chat"), json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise_for_status(self.id, response, body)

            # Ollama streams newline-delimited JSON, not SSE.
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    yield StreamEvent(type=StreamEventType.ERROR, error=str(chunk["error"])[:400])
                    return

                message = chunk.get("message") or {}
                if message.get("thinking"):
                    yield StreamEvent(type=StreamEventType.THINKING, text=message["thinking"])
                if message.get("content"):
                    yield StreamEvent(type=StreamEventType.TEXT, text=message["content"])

                for tc in message.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        calls.append(
                            ToolCall(
                                id=new_id("call"),
                                name=fn["name"],
                                arguments=args if isinstance(args, dict) else {},
                            )
                        )

                if chunk.get("done"):
                    finish = str(chunk.get("done_reason") or "stop")
                    usage = Usage(
                        input_tokens=int(chunk.get("prompt_eval_count") or 0),
                        output_tokens=int(chunk.get("eval_count") or 0),
                        cost_usd=0.0,  # local inference is free
                    )

        for call in calls:
            yield StreamEvent(type=StreamEventType.TOOL_CALL, tool_call=call)

        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE,
            finish_reason=finish,
            usage=usage,
            model=request.model,
            index=int((time.perf_counter() - started) * 1000),
        )

    # -- discovery --------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        client = http_client(self.id, timeout=15)
        response = await client.get(self._url("/api/tags"))
        raise_for_status(self.id, response)
        rows = response.json().get("models") or []
        out: list[ModelInfo] = []
        for row in rows:
            name = row.get("name") or row.get("model")
            if not name:
                continue
            details = row.get("details") or {}
            family = str(details.get("family") or "")
            lowered = f"{name} {family}".lower()
            caps = [Capability.STREAMING, Capability.JSON, Capability.TOOLS]
            mods = [Modality.TEXT]
            if any(k in lowered for k in ("llava", "vision", "bakllava", "moondream", "minicpm-v")):
                caps.append(Capability.VISION)
                mods.append(Modality.IMAGE)
            if "embed" in lowered:
                caps = [Capability.EMBEDDING]
            size = int(row.get("size") or 0)
            out.append(
                ModelInfo(
                    id=f"{self.id}/{name}",
                    provider=self.id,
                    name=str(name),
                    display_name=str(name),
                    family=family or "local",
                    context_window=int(details.get("context_length") or 0),
                    modalities=mods,
                    capabilities=caps,
                    local=True,
                    description=(
                        f"{details.get('parameter_size', '?')} params, "
                        f"{details.get('quantization_level', '?')}, {human_bytes(size)} on disk"
                    ),
                    meta={"size_bytes": size},
                )
            )
        return out

    async def embed(self, texts: list[str], model: str = "") -> EmbeddingResult:
        model_name = model or "nomic-embed-text"
        if "/" in model_name:
            _, model_name = catalog.split_id(model_name)
        client = http_client(self.id, timeout=180)
        response = await client.post(
            self._url("/api/embed"), json={"model": model_name, "input": texts}
        )
        raise_for_status(self.id, response)
        vectors = response.json().get("embeddings") or []
        return EmbeddingResult(
            vectors=vectors,
            model=f"{self.id}/{model_name}",
            dim=len(vectors[0]) if vectors else 0,
        )

    async def pull(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Download a model, streaming progress so the UI can show a bar."""
        client = http_client(self.id, timeout=3600)
        async with client.stream(
            "POST", self._url("/api/pull"), json={"model": model, "stream": True}
        ) as response:
            raise_for_status(self.id, response)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def delete_model(self, model: str) -> bool:
        client = http_client(self.id, timeout=60)
        response = await client.request(
            "DELETE", self._url("/api/delete"), json={"model": model}
        )
        return response.status_code < 400
