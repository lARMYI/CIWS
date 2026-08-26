"""The media studio: asset storage and the generation router.

Assets are created *before* generation starts, in ``queued`` state. That is
deliberate -- a video render can take minutes, and a grid of placeholder tiles
that fill in as results land is a far better experience than a spinner followed
by everything appearing at once. It also means a crashed render leaves a row
explaining what failed instead of nothing at all.

Files land under ``CIWS_HOME/assets/<yyyy-mm>/`` so a busy month does not turn
into a directory with ten thousand entries.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, or_, select

from ..core import paths
from ..core.config import get_settings
from ..core.errors import NotFound, ProviderError, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..db.base import session_scope
from ..db.models import Asset
from . import providers as media_providers
from .types import GenerationResult, ImageRequest, MediaModel, VideoRequest

log = get_logger("media")

EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
    "video/mp4": ".mp4", "video/webm": ".webm", "audio/mpeg": ".mp3", "audio/wav": ".wav",
}
THUMB_MAX = 512


# ---------------------------------------------------------------------------
# Asset store
# ---------------------------------------------------------------------------


def _asset_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m")
    target = paths.assets_dir() / stamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def asset_file(asset: Asset) -> Path | None:
    if not asset.file_path:
        return None
    path = Path(asset.file_path)
    return path if path.exists() else None


async def create_asset(
    kind: str,
    prompt: str,
    provider: str,
    model: str,
    *,
    params: dict[str, Any] | None = None,
    project_id: str | None = None,
    negative_prompt: str = "",
    run_id: str | None = None,
    parent_id: str | None = None,
) -> Asset:
    async with session_scope() as s:
        asset = Asset(
            kind=kind,
            status="queued",
            prompt=prompt,
            negative_prompt=negative_prompt,
            provider=provider,
            model=model,
            params=params or {},
            project_id=project_id,
            run_id=run_id,
            parent_id=parent_id,
        )
        s.add(asset)
        await s.flush()
    return asset


async def _set(asset_id: str, **fields: Any) -> Asset | None:
    async with session_scope() as s:
        asset = (await s.execute(select(Asset).where(Asset.id == asset_id))).scalar_one_or_none()
        if asset is None:
            return None
        for key, value in fields.items():
            if hasattr(asset, key):
                setattr(asset, key, value)
        await s.flush()
        return asset


def _make_thumbnail(source: Path, asset_id: str, kind: str) -> str:
    """Best-effort thumbnail. Pillow is optional, and video needs ffmpeg."""
    try:
        from PIL import Image
    except ImportError:
        return ""
    if kind != "image":
        return ""
    try:
        with Image.open(source) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_MAX, THUMB_MAX))
            target = source.parent / f"{asset_id}_thumb.jpg"
            img.save(target, "JPEG", quality=82)
            return str(target)
    except Exception as exc:  # noqa: BLE001 - a missing thumbnail is cosmetic
        log.debug("Thumbnail failed for %s: %s", asset_id, exc)
        return ""


def _probe_size(source: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(source) as img:
            return img.width, img.height
    except Exception:  # noqa: BLE001
        return 0, 0


async def attach_result(asset_id: str, result: GenerationResult) -> Asset:
    """Persist bytes to disk and mark the asset ready."""
    asset = await get_asset(asset_id)
    if asset is None:
        raise NotFound(f"No asset {asset_id}")

    data = result.data
    if data is None and result.remote_url:
        from ..gateway.base import http_client

        response = await http_client("media-fetch", timeout=300).get(result.remote_url)
        response.raise_for_status()
        data = response.content

    if not data:
        return await fail_asset(asset_id, "The provider returned no data")

    suffix = EXTENSIONS.get(result.mime_type, ".bin")
    target = _asset_dir() / f"{asset_id}{suffix}"
    target.write_bytes(data)

    width, height = result.width, result.height
    if asset.kind == "image" and not (width and height):
        width, height = _probe_size(target)

    thumb = _make_thumbnail(target, asset_id, asset.kind)

    updated = await _set(
        asset_id,
        status="ready",
        file_path=str(target),
        thumb_path=thumb,
        mime_type=result.mime_type,
        width=width,
        height=height,
        duration_s=result.duration_s,
        size_bytes=len(data),
        seed=result.seed,
        cost_usd=result.cost_usd,
        remote_url=result.remote_url,
        job_id=result.job_id,
        meta={**(asset.meta or {}), **result.meta},
    )
    bus.publish(
        Topic.MEDIA_DONE,
        asset_id=asset_id,
        kind=asset.kind,
        provider=asset.provider,
        model=asset.model,
        size_bytes=len(data),
    )
    assert updated is not None
    return updated


async def fail_asset(asset_id: str, error: str) -> Asset:
    asset = await _set(asset_id, status="failed", error=str(error)[:2000])
    bus.publish(Topic.MEDIA_ERROR, asset_id=asset_id, error=str(error)[:400])
    if asset is None:
        raise NotFound(f"No asset {asset_id}")
    return asset


async def get_asset(asset_id: str) -> Asset | None:
    async with session_scope() as s:
        return (await s.execute(select(Asset).where(Asset.id == asset_id))).scalar_one_or_none()


async def list_assets(
    *,
    kind: str | None = None,
    project_id: str | None = None,
    status: str | None = None,
    favorite: bool | None = None,
    query: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[Asset]:
    async with session_scope() as s:
        stmt = select(Asset)
        if kind:
            stmt = stmt.where(Asset.kind == kind)
        if project_id:
            stmt = stmt.where(Asset.project_id == project_id)
        if status:
            stmt = stmt.where(Asset.status == status)
        if favorite is not None:
            stmt = stmt.where(Asset.favorite.is_(favorite))
        if query:
            like = f"%{query}%"
            stmt = stmt.where(or_(Asset.prompt.ilike(like), Asset.model.ilike(like)))
        return list(
            (
                await s.execute(stmt.order_by(Asset.created_at.desc()).limit(limit).offset(offset))
            ).scalars().all()
        )


async def delete_asset(asset_id: str) -> bool:
    async with session_scope() as s:
        asset = (await s.execute(select(Asset).where(Asset.id == asset_id))).scalar_one_or_none()
        if asset is None:
            return False
        files = [asset.file_path, asset.thumb_path]
        await s.execute(sa_delete(Asset).where(Asset.id == asset_id))
    for path in files:
        if path:
            Path(path).unlink(missing_ok=True)
    return True


async def import_file(
    path: Path | str, *, kind: str = "", project_id: str | None = None, prompt: str = ""
) -> Asset:
    """Bring an existing file into the asset library."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValidationFailed(f"No such file: {path}")

    suffix = path.suffix.lower()
    mime = next((m for m, e in EXTENSIONS.items() if e == suffix), "application/octet-stream")
    if not kind:
        kind = mime.split("/", 1)[0] if mime != "application/octet-stream" else "image"
        kind = {"image": "image", "video": "video", "audio": "audio"}.get(kind, "image")

    asset = await create_asset(
        kind, prompt or path.stem, "import", "local/import", project_id=project_id
    )
    return await attach_result(
        asset.id,
        GenerationResult(data=path.read_bytes(), mime_type=mime),
    )


async def stats() -> dict[str, Any]:
    async with session_scope() as s:
        by_kind = {
            k: int(v)
            for k, v in (
                await s.execute(select(Asset.kind, func.count(Asset.id)).group_by(Asset.kind))
            ).all()
        }
        total_bytes = (await s.execute(select(func.sum(Asset.size_bytes)))).scalar() or 0
        total_cost = (await s.execute(select(func.sum(Asset.cost_usd)))).scalar() or 0.0
        by_status = {
            k: int(v)
            for k, v in (
                await s.execute(select(Asset.status, func.count(Asset.id)).group_by(Asset.status))
            ).all()
        }
    return {
        "by_kind": by_kind,
        "by_status": by_status,
        "bytes": int(total_bytes),
        "cost_usd": round(float(total_cost), 4),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def available_models(kind: str, *, configured_only: bool = True) -> list[MediaModel]:
    """Every model of a kind, annotated with whether it can actually run."""
    out: list[MediaModel] = []
    for provider in media_providers.for_kind(kind):
        usable = provider.configured()
        if configured_only and not usable:
            continue
        try:
            models = await provider.models()
        except Exception as exc:  # noqa: BLE001
            log.debug("%s model list failed: %s", provider.id, exc)
            continue
        for model in models:
            if model.kind != kind:
                continue
            model.configured = usable
            model.env_hint = provider.env_hint
            model.local = provider.local
            out.append(model)
    return out


def _resolve(model_id: str, kind: str) -> tuple[media_providers.MediaProvider, str]:
    """``"replicate/owner/name"`` -> (provider, ``"owner/name"``)."""
    settings = get_settings().media
    if not model_id:
        model_id = settings.default_image_model if kind == "image" else settings.default_video_model
    provider_id, _, rest = model_id.partition("/")
    if not rest:
        # Bare name: find the first provider of this kind that offers it.
        for provider in media_providers.for_kind(kind):
            return provider, model_id
        raise ValidationFailed(f"No {kind} provider available for '{model_id}'")
    return media_providers.get_provider(provider_id), rest


async def generate_image(
    req: ImageRequest, *, project_id: str | None = None, run_id: str | None = None
) -> list[Asset]:
    if not req.prompt.strip():
        raise ValidationFailed("An image needs a prompt")

    provider, model_name = _resolve(req.model, "image")
    if not provider.configured():
        raise ProviderError(
            provider.id,
            f"{provider.label} is not configured. Add its key in Settings -> Credentials"
            + (f" ({provider.env_hint})" if provider.env_hint else ""),
            retryable=False,
        )

    full_model = f"{provider.id}/{model_name}"
    count = max(1, min(req.n, 4))
    assets = [
        await create_asset(
            "image", req.prompt, provider.id, full_model,
            params=req.model_dump(exclude={"prompt", "negative_prompt", "init_image_b64"}),
            project_id=project_id, negative_prompt=req.negative_prompt, run_id=run_id,
        )
        for _ in range(count)
    ]

    bus.publish(
        Topic.MEDIA_START,
        kind="image",
        provider=provider.id,
        model=full_model,
        asset_ids=[a.id for a in assets],
        prompt=req.prompt[:300],
    )

    for asset in assets:
        await _set(asset.id, status="running")

    try:
        results = await provider.generate_image(req.model_copy(update={"model": full_model}))
    except Exception as exc:  # noqa: BLE001 - record the failure on every placeholder
        for asset in assets:
            await fail_asset(asset.id, str(exc))
        raise

    out: list[Asset] = []
    for asset, result in zip(assets, results):
        out.append(await attach_result(asset.id, result))
    # A provider that returned fewer images than requested leaves orphans.
    for asset in assets[len(results):]:
        await fail_asset(asset.id, "The provider returned fewer images than requested")
    return out


async def generate_video(
    req: VideoRequest, *, project_id: str | None = None, run_id: str | None = None
) -> Asset:
    if not req.prompt.strip() and not req.init_image_b64:
        raise ValidationFailed("A video needs a prompt or a source image")

    provider, model_name = _resolve(req.model, "video")
    if not provider.configured():
        raise ProviderError(
            provider.id,
            f"{provider.label} is not configured. Add its key in Settings -> Credentials"
            + (f" ({provider.env_hint})" if provider.env_hint else ""),
            retryable=False,
        )

    full_model = f"{provider.id}/{model_name}"
    asset = await create_asset(
        "video", req.prompt, provider.id, full_model,
        params=req.model_dump(exclude={"prompt", "negative_prompt", "init_image_b64"}),
        project_id=project_id, negative_prompt=req.negative_prompt, run_id=run_id,
    )
    bus.publish(
        Topic.MEDIA_START, kind="video", provider=provider.id, model=full_model,
        asset_ids=[asset.id], prompt=req.prompt[:300],
    )
    await _set(asset.id, status="running")

    try:
        result = await provider.generate_video(req.model_copy(update={"model": full_model}))
    except Exception as exc:  # noqa: BLE001
        await fail_asset(asset.id, str(exc))
        raise
    return await attach_result(asset.id, result)


async def generate_video_background(
    req: VideoRequest, *, project_id: str | None = None, run_id: str | None = None
) -> Asset:
    """Kick off a video render and return the queued asset immediately.

    Video takes minutes. The HTTP layer uses this so the request returns an
    asset id the UI can poll or watch on the event bus.
    """
    provider, model_name = _resolve(req.model, "video")
    asset = await create_asset(
        "video", req.prompt, provider.id, f"{provider.id}/{model_name}",
        params=req.model_dump(exclude={"prompt", "negative_prompt", "init_image_b64"}),
        project_id=project_id, negative_prompt=req.negative_prompt, run_id=run_id,
    )

    async def worker() -> None:
        await _set(asset.id, status="running")
        try:
            if not provider.configured():
                raise ProviderError(
                    provider.id, f"{provider.label} is not configured", retryable=False
                )
            result = await provider.generate_video(
                req.model_copy(update={"model": f"{provider.id}/{model_name}"})
            )
            await attach_result(asset.id, result)
        except Exception as exc:  # noqa: BLE001
            await fail_asset(asset.id, str(exc))

    task = asyncio.create_task(worker())
    _background.add(task)
    task.add_done_callback(_background.discard)
    return asset


#: Strong references so background renders are not garbage-collected mid-flight.
_background: set[asyncio.Task[None]] = set()
