"""Provider-neutral types for the model gateway.

Every provider speaks a different dialect. Anthropic wants ``system`` hoisted out
of the message list and tool results as user-turn content blocks; OpenAI wants a
``tool`` role; Google wants ``parts`` and ``functionResponse``. Rather than leak
that into the agent runtime, each adapter translates to and from the shapes in
this module.

The rule: nothing above :mod:`ciws.gateway` ever imports a provider SDK or
knows a provider's wire format.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.util import new_id

Role = Literal["system", "user", "assistant", "tool"]


class PartType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class ContentPart(BaseModel):
    """One block inside a multimodal message."""

    type: PartType = PartType.TEXT
    text: str = ""
    #: base64 payload for inline media, or a data URI.
    data: str = ""
    mime_type: str = ""
    url: str = ""
    #: For TOOL_USE / TOOL_RESULT blocks.
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    is_error: bool = False

    @classmethod
    def text_part(cls, text: str) -> "ContentPart":
        return cls(type=PartType.TEXT, text=text)

    @classmethod
    def image_b64(cls, data: str, mime_type: str = "image/png") -> "ContentPart":
        return cls(type=PartType.IMAGE, data=data, mime_type=mime_type)

    @classmethod
    def image_url(cls, url: str) -> "ContentPart":
        return cls(type=PartType.IMAGE, url=url)


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Kept when a provider streams arguments as partial JSON text.
    raw_arguments: str = ""


class ToolResult(BaseModel):
    call_id: str
    name: str = ""
    content: str = ""
    is_error: bool = False
    #: Media a tool produced (a rendered chart, a generated image) fed straight back to the model.
    parts: list[ContentPart] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Role = "user"
    content: str = ""
    parts: list[ContentPart] = Field(default_factory=list)
    name: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    thinking: str = ""
    #: Provider-specific opaque state (e.g. Anthropic thinking signatures).
    signature: str = ""

    @classmethod
    def user(cls, content: str, parts: list[ContentPart] | None = None) -> "ChatMessage":
        return cls(role="user", content=content, parts=parts or [])

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "ChatMessage":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, results: list[ToolResult]) -> "ChatMessage":
        return cls(role="tool", tool_results=results)

    def text(self) -> str:
        if self.content:
            return self.content
        return "\n".join(p.text for p in self.parts if p.type == PartType.TEXT and p.text)

    def has_media(self) -> bool:
        return any(p.type in (PartType.IMAGE, PartType.FILE, PartType.AUDIO) for p in self.parts)


class ToolSpec(BaseModel):
    """A tool as the model sees it -- JSON Schema in, string out."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def openai_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description[:1024],
                "parameters": self.parameters,
            },
        }

    def anthropic_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description[:2048],
            "input_schema": self.parameters,
        }

    def google_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description[:1024],
            "parameters": _strip_unsupported_schema(self.parameters),
        }


def _strip_unsupported_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Google's function declarations reject several JSON Schema keywords."""
    drop = {
        "$schema", "additionalProperties", "default", "examples", "const",
        "exclusiveMinimum", "exclusiveMaximum", "$ref", "$defs", "definitions",
        "oneOf", "anyOf", "allOf", "not", "patternProperties",
    }
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _strip_unsupported_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _strip_unsupported_schema(v)
        else:
            out[k] = v
    out.setdefault("type", "object")
    return out


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    system: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] = Field(default_factory=list)
    stream: bool = True
    #: Extended reasoning budget in tokens; 0 disables. Providers that lack it ignore it.
    thinking_budget: int = 0
    #: Ask for strict JSON back, optionally against a schema.
    json_mode: bool = False
    json_schema: dict[str, Any] | None = None
    seed: int | None = None
    timeout_s: float = 300.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def estimated_input_chars(self) -> int:
        total = len(self.system)
        for m in self.messages:
            total += len(m.text())
            for r in m.tool_results:
                total += len(r.content)
        return total


class StreamEventType(str, Enum):
    START = "start"
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_ARGS_DELTA = "tool_args_delta"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: str = ""
    error: str = ""
    model: str = ""
    index: int = 0


class ChatResponse(BaseModel):
    content: str = ""
    thinking: str = ""
    signature: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    latency_ms: int = 0
    parts: list[ContentPart] = Field(default_factory=list)

    def to_message(self) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
            thinking=self.thinking,
            signature=self.signature,
            parts=self.parts,
        )


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class Capability(str, Enum):
    TOOLS = "tools"
    VISION = "vision"
    JSON = "json"
    THINKING = "thinking"
    STREAMING = "streaming"
    EMBEDDING = "embedding"
    CACHING = "caching"


class ModelInfo(BaseModel):
    """What the hub knows about one model."""

    id: str  # "provider/model"
    provider: str
    name: str
    display_name: str = ""
    family: str = ""
    context_window: int = 0
    max_output: int = 0
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    capabilities: list[Capability] = Field(default_factory=list)
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    local: bool = False
    description: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def cost_for(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_cost_per_mtok
            + usage.output_tokens * self.output_cost_per_mtok
        ) / 1_000_000


class HealthStatus(BaseModel):
    provider: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0
    model_count: int = 0
    requires_key: bool = True
    has_key: bool = False


class EmbeddingResult(BaseModel):
    vectors: list[list[float]]
    model: str
    dim: int
    usage: Usage = Field(default_factory=Usage)
