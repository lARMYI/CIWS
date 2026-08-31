"""Image and video generation from inside an agent run."""

from __future__ import annotations

from typing import Any

from ...media import studio
from ...media.types import ImageRequest, VideoRequest
from ..registry import Risk, ToolOutput, registry
from ...gateway.types import ContentPart


@registry.tool(
    "image_generate",
    category="media",
    risk=Risk.NETWORK,
    timeout_s=300,
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Describe subject, composition, lighting and style. Concrete beats "
                    "flowery -- 'overhead shot, hard side light, muted palette' outperforms "
                    "'beautiful masterpiece'."
                ),
            },
            "negative_prompt": {"type": "string", "description": "What to avoid, if the model supports it."},
            "model": {"type": "string", "description": "Omit to use the configured default."},
            "size": {"type": "string", "default": "1024x1024"},
            "n": {"type": "integer", "description": "1-4.", "default": 1},
        },
        "required": ["prompt"],
    },
)
async def image_generate(
    prompt: str,
    negative_prompt: str = "",
    model: str = "",
    size: str = "1024x1024",
    n: int = 1,
    ctx: Any = None,
) -> ToolOutput:
    """Generate an image and save it to the asset library.

    The generated image is returned to you visually, so you can check whether
    it matches what was asked for and revise the prompt if not.
    """
    assets = await studio.generate_image(
        ImageRequest(prompt=prompt, negative_prompt=negative_prompt, model=model, size=size, n=n),
        project_id=getattr(ctx, "project_id", None),
        run_id=getattr(ctx, "run_id", None),
    )
    parts: list[ContentPart] = []
    lines = [f"Generated {len(assets)} image(s):"]
    for asset in assets:
        lines.append(f"- {asset.id}  {asset.model}  {asset.width}x{asset.height}")
        path = studio.asset_file(asset)
        if path and path.stat().st_size < 5_000_000:
            import base64

            parts.append(
                ContentPart.image_b64(
                    base64.b64encode(path.read_bytes()).decode(), asset.mime_type or "image/png"
                )
            )
    return ToolOutput(content="\n".join(lines), parts=parts, meta={"asset_ids": [a.id for a in assets]})


@registry.tool(
    "video_generate",
    category="media",
    risk=Risk.NETWORK,
    timeout_s=60,
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Describe the shot, motion and subject."},
            "model": {"type": "string"},
            "duration_s": {"type": "number", "default": 5},
            "aspect_ratio": {"type": "string", "default": "16:9"},
        },
        "required": ["prompt"],
    },
)
async def video_generate(
    prompt: str, model: str = "", duration_s: float = 5.0, aspect_ratio: str = "16:9", ctx: Any = None
) -> ToolOutput:
    """Start a video render.

    Returns immediately with an asset id -- video takes minutes. Tell the user
    it is rendering and that it will appear in the Studio; do not wait for it.
    """
    asset = await studio.generate_video_background(
        VideoRequest(prompt=prompt, model=model, duration_s=duration_s, aspect_ratio=aspect_ratio),
        project_id=getattr(ctx, "project_id", None),
        run_id=getattr(ctx, "run_id", None),
    )
    return ToolOutput.text(
        f"Video render started: asset {asset.id} using {asset.model}. "
        f"It will appear in the Studio when it finishes.",
        asset_id=asset.id,
    )


@registry.tool(
    "media_list",
    category="media",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["image", "video", "audio"]},
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
    },
)
async def media_list(kind: str = "", query: str = "", limit: int = 20, ctx: Any = None) -> ToolOutput:
    """List generated and imported media."""
    assets = await studio.list_assets(
        kind=kind or None, query=query, limit=limit,
        project_id=getattr(ctx, "project_id", None),
    )
    if not assets:
        return ToolOutput.text("No media assets yet.")
    lines = [f"{len(assets)} assets:"]
    lines += [
        f"- [{a.kind}] {a.id}  {a.status}  {a.model}  \"{a.prompt[:70]}\""
        for a in assets
    ]
    return ToolOutput.text("\n".join(lines))


@registry.tool(
    "media_models",
    category="media",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["image", "video"], "default": "image"}},
    },
)
async def media_models(kind: str = "image", ctx: Any = None) -> ToolOutput:
    """Which image or video models are usable right now."""
    models = await studio.available_models(kind, configured_only=False)
    lines = [f"{kind} models:"]
    for m in models:
        state = "ready" if m.configured else f"needs {m.env_hint or 'a key'}"
        lines.append(f"- {m.id}  ({state})  {m.description[:80]}")
    return ToolOutput.text("\n".join(lines))
