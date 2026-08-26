"""Shared types for image and video generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MediaModel(BaseModel):
    id: str  # "provider/model"
    provider: str
    name: str
    display_name: str = ""
    kind: str = "image"  # image | video
    local: bool = False
    sizes: list[str] = Field(default_factory=list)
    supports_negative_prompt: bool = False
    supports_image_input: bool = False
    max_duration_s: float = 0.0
    description: str = ""
    params_schema: dict[str, Any] = Field(default_factory=dict)
    #: Populated by the router so the UI can grey out what is not usable yet.
    configured: bool = True
    env_hint: str = ""


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = ""
    size: str = "1024x1024"
    n: int = 1
    seed: int | None = None
    quality: str = ""
    style: str = ""
    #: base64 for image-to-image / edits.
    init_image_b64: str = ""
    strength: float = 0.6
    params: dict[str, Any] = Field(default_factory=dict)

    def dimensions(self) -> tuple[int, int]:
        try:
            w, _, h = self.size.partition("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return 1024, 1024


class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = ""
    duration_s: float = 5.0
    aspect_ratio: str = "16:9"
    seed: int | None = None
    init_image_b64: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    data: bytes | None = None
    remote_url: str = ""
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    seed: int = 0
    cost_usd: float = 0.0
    job_id: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
