"""Google Gemini adapter.

Gemini's wire format diverges more than most: roles are ``user``/``model``,
everything is a ``part``, the system prompt is a separate ``systemInstruction``
object, and tool results come back as ``functionResponse`` parts rather than a
distinct role. Streaming is SSE only when ``alt=sse`` is set -- without it the
endpoint returns a single JSON array and the stream appears to hang.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.logging import get_logger
from ...core.util import new_id
from .. import catalog
from ..base import Provider, http_client, raise_for_status, sse_lines
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

log = get_logger("gateway.google")


class GoogleProvider(Provider):
    id = "google"
    label = "Google Gemini"
    env_hint = "GOOGLE_API_KEY or GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    supports_embeddings = True
    website = "https://aistudio.google.com"

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json", "x-goog-api-key": self.require_key()}

    # -- translation ------------------------------------------------------

    def _contents(self, request: ChatRequest) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                continue

            if msg.role == "tool":
                parts = [
                    {
                        "functionResponse": {
                            "name": res.name or "tool",
                            "response": {"result": res.content or "(no output)"},
                        }
                    }
                    for res in msg.tool_results
                ]
                if parts:
                    contents.append({"role": "user", "parts": parts})
                continue

            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for part in msg.parts:
                if part.type is PartType.TEXT and part.text:
                    parts.append({"text": part.text})
                elif part.type is PartType.IMAGE and part.data:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": part.mime_type or "image/png",
                                "data": part.data,
                            }
                        }
                    )
                elif part.type is PartType.FILE and part.data:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": part.mime_type or "application/pdf",
                                "data": part.data,
                            }
                        }
                    )
            for call in msg.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.arguments}})

            if not parts:
                continue
            role = "model" if msg.role == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        gen: dict[str, Any] = {}
        if request.temperature is not None:
            gen["temperature"] = request.temperature
        if request.top_p is not None:
            gen["topP"] = request.top_p
        if request.max_tokens:
            gen["maxOutputTokens"] = request.max_tokens
        if request.stop:
            gen["stopSequences"] = request.stop[:5]
        if request.json_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = request.json_schema
        elif request.json_mode:
            gen["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {"contents": self._contents(request)}
        if gen:
            payload["generationConfig"] = gen

        system_text = request.system or "\n\n".join(
            m.content for m in request.messages if m.role == "system" and m.content
        )
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        if request.tools:
            payload["tools"] = [
                {"function_declarations": [t.google_dict() for t in request.tools]}
            ]
            if request.tool_choice == "required":
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif request.tool_choice == "none":
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        return payload

    # -- streaming --------------------------------------------------------

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        _, model_name = catalog.split_id(request.model)
        payload = self._payload(request)
        client = http_client(self.id, timeout=request.timeout_s)
        started = time.perf_counter()

        yield StreamEvent(type=StreamEventType.START, model=request.model)

        usage = Usage()
        finish = ""
        calls: list[ToolCall] = []

        url = self._url(f"models/{model_name}:streamGenerateContent")
        async with client.stream(
            "POST", url, json=payload, headers=self._headers(), params={"alt": "sse"}
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise_for_status(self.id, response, body)

            async for chunk in sse_lines(response):
                meta = chunk.get("usageMetadata") or {}
                if meta:
                    usage = Usage(
                        input_tokens=int(meta.get("promptTokenCount") or 0),
                        output_tokens=int(meta.get("candidatesTokenCount") or 0),
                        cache_read_tokens=int(meta.get("cachedContentTokenCount") or 0),
                        reasoning_tokens=int(meta.get("thoughtsTokenCount") or 0),
                    )
                for cand in chunk.get("candidates") or []:
                    if cand.get("finishReason"):
                        finish = str(cand["finishReason"])
                    for part in (cand.get("content") or {}).get("parts") or []:
                        if part.get("thought") and part.get("text"):
                            yield StreamEvent(type=StreamEventType.THINKING, text=part["text"])
                        elif part.get("text"):
                            yield StreamEvent(type=StreamEventType.TEXT, text=part["text"])
                        fc = part.get("functionCall")
                        if fc and fc.get("name"):
                            calls.append(
                                ToolCall(
                                    id=new_id("call"),
                                    name=fc["name"],
                                    arguments=fc.get("args") or {},
                                )
                            )

        for call in calls:
            yield StreamEvent(type=StreamEventType.TOOL_CALL, tool_call=call)

        info = catalog.lookup(request.model)
        if info:
            usage.cost_usd = info.cost_for(usage)
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
        client = http_client(self.id, timeout=30)
        response = await client.get(
            self._url("models"), headers=self._headers(), params={"pageSize": 200}
        )
        raise_for_status(self.id, response)
        rows = response.json().get("models") or []
        out: list[ModelInfo] = []
        for row in rows:
            raw_name = str(row.get("name", ""))
            name = raw_name.split("/")[-1]
            if not name:
                continue
            methods = row.get("supportedGenerationMethods") or []
            is_embedding = "embedContent" in methods
            if not is_embedding and not any(
                m in methods for m in ("generateContent", "streamGenerateContent")
            ):
                continue
            caps = (
                [Capability.EMBEDDING]
                if is_embedding
                else [Capability.STREAMING, Capability.TOOLS, Capability.JSON, Capability.VISION]
            )
            mods = [Modality.TEXT] if is_embedding else [Modality.TEXT, Modality.IMAGE]
            out.append(
                ModelInfo(
                    id=f"{self.id}/{name}",
                    provider=self.id,
                    name=name,
                    display_name=row.get("displayName") or name,
                    family="gemini",
                    context_window=int(row.get("inputTokenLimit") or 0),
                    max_output=int(row.get("outputTokenLimit") or 0),
                    modalities=mods,
                    capabilities=caps,
                    description=str(row.get("description") or "")[:300],
                )
            )
        return catalog.merge(out, self.id)

    # -- embeddings -------------------------------------------------------

    async def embed(self, texts: list[str], model: str = "") -> EmbeddingResult:
        model_name = model or "text-embedding-004"
        if "/" in model_name:
            _, model_name = catalog.split_id(model_name)
        client = http_client(self.id, timeout=120)
        response = await client.post(
            self._url(f"models/{model_name}:batchEmbedContents"),
            headers=self._headers(),
            json={
                "requests": [
                    {
                        "model": f"models/{model_name}",
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in texts
                ]
            },
        )
        raise_for_status(self.id, response)
        rows = response.json().get("embeddings") or []
        vectors = [r.get("values") or [] for r in rows]
        return EmbeddingResult(
            vectors=vectors,
            model=f"{self.id}/{model_name}",
            dim=len(vectors[0]) if vectors else 0,
        )
