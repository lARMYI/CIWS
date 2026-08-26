"""System, settings, credentials, models, tools, hubs, and projects."""

from __future__ import annotations

import platform
import sys
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from .. import __version__
from ..core import logging as ciws_logging
from ..core import paths, secrets
from ..core.config import get_settings, save_settings
from ..core.errors import NotFound
from ..db.base import db_stats, vacuum
from ..db.models import ModelRecord, Project
from ..db.base import session_scope
from ..db.vectors import gc_orphans, index
from ..gateway.registry import gateway
from ..hubs import registry as hub_registry
from ..memory import embeddings
from ..tools.registry import registry as tool_registry
from .deps import Auth, get_token, rotate_token

router = APIRouter(dependencies=[Auth])
public = APIRouter()  # unauthenticated: health only


@public.get("/health")
async def health() -> dict[str, Any]:
    """Liveness. Deliberately reveals nothing about the workspace."""
    return {"ok": True, "app": "CIWS", "version": __version__}


@router.get("/system")
async def system_info() -> dict[str, Any]:
    settings = get_settings()
    providers = gateway.providers()
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "paths": paths.describe(),
        "settings": settings.to_public_dict(),
        "providers": [
            {
                "id": p.id,
                "label": p.label,
                "configured": p.configured(),
                "local": p.local,
                "requires_key": p.requires_key,
                "env_hint": p.env_hint,
                "website": p.website,
            }
            for p in providers.values()
        ],
        "embeddings": await embeddings.backend_info(),
        "tools": len(tool_registry.list_tools()),
        "hubs_running": hub_registry.manager.running(),
    }


@router.get("/system/stats")
async def system_stats() -> dict[str, Any]:
    return {
        "database": await db_stats(),
        "vectors": index.stats(),
        "hubs": hub_registry.manager.status(),
    }


@router.get("/system/logs")
async def system_logs(lines: int = Query(200, le=2000)) -> dict[str, Any]:
    return {"lines": ciws_logging.tail(lines)}


@router.post("/system/vacuum")
async def system_vacuum() -> dict[str, Any]:
    removed = await gc_orphans()
    await vacuum()
    return {"ok": True, "orphaned_vectors_removed": removed}


@router.get("/system/token")
async def show_token() -> dict[str, Any]:
    return {"token": get_token(), "path": str(paths.home() / "token")}


@router.post("/system/token/rotate")
async def rotate() -> dict[str, Any]:
    return {"token": rotate_token(), "note": "Reload the UI to pick up the new token."}


# ---------------------------------------------------------------------------
# Settings and credentials
# ---------------------------------------------------------------------------


@router.get("/settings")
async def read_settings() -> dict[str, Any]:
    return get_settings().to_public_dict()


@router.patch("/settings")
async def patch_settings(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    updated = save_settings(patch)
    # An embedding-model change invalidates the cached backend choice.
    if "routing" in patch and "embed" in (patch.get("routing") or {}):
        embeddings.reset_backend()
    return updated.to_public_dict()


@router.get("/credentials")
async def list_credentials() -> dict[str, Any]:
    return {"credentials": secrets.status()}


class CredentialBody(BaseModel):
    name: str
    value: str


@router.put("/credentials")
async def set_credential(body: CredentialBody) -> dict[str, Any]:
    secrets.put(body.name, body.value)
    gateway._models_loaded_at = 0.0  # noqa: SLF001 - force a catalogue refresh
    return {"ok": True, "name": body.name, "source": secrets.source_of(body.name)}


@router.delete("/credentials/{name}")
async def delete_credential(name: str) -> dict[str, Any]:
    secrets.delete(name)
    return {"ok": True, "name": name}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models(refresh: bool = False) -> dict[str, Any]:
    models = await gateway.models(refresh=refresh)
    async with session_scope() as s:
        records = {
            r.id: r for r in (await s.execute(select(ModelRecord))).scalars().all()
        }
    rows = []
    for m in sorted(models, key=lambda x: (x.provider, x.name)):
        record = records.get(m.id)
        provider = gateway.providers().get(m.provider)
        rows.append(
            {
                **m.model_dump(mode="json"),
                "configured": bool(provider and provider.configured()),
                "usage": {
                    "calls": record.call_count if record else 0,
                    "errors": record.error_count if record else 0,
                    "input_tokens": record.total_input_tokens if record else 0,
                    "output_tokens": record.total_output_tokens if record else 0,
                    "cost_usd": round(record.total_cost_usd, 4) if record else 0.0,
                    "avg_latency_ms": round(record.avg_latency_ms) if record else 0,
                },
                "favorite": record.favorite if record else False,
            }
        )
    return {"models": rows, "count": len(rows)}


@router.get("/models/health")
async def models_health(refresh: bool = True) -> dict[str, Any]:
    health = await gateway.health(refresh=refresh)
    return {"providers": {k: v.model_dump() for k, v in health.items()}}


@router.patch("/models/{model_id:path}")
async def patch_model(model_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Override pricing or favourite a model. Price edits survive a re-seed."""
    async with session_scope() as s:
        record = (
            await s.execute(select(ModelRecord).where(ModelRecord.id == model_id))
        ).scalar_one_or_none()
        if record is None:
            raise HTTPException(404, f"Unknown model {model_id}")
        for key in ("favorite", "enabled", "input_cost_per_mtok", "output_cost_per_mtok"):
            if key in patch:
                setattr(record, key, patch[key])
        if "input_cost_per_mtok" in patch or "output_cost_per_mtok" in patch:
            record.meta = {**(record.meta or {}), "price_overridden": True}
        await s.flush()
        return record.to_dict()


@router.post("/models/pull")
async def pull_model(model: str = Body(..., embed=True)) -> dict[str, Any]:
    """Download a local model via Ollama."""
    from ..gateway.providers.ollama import OllamaProvider

    provider = OllamaProvider()
    last: dict[str, Any] = {}
    async for update in provider.pull(model):
        last = update
        if update.get("error"):
            raise HTTPException(502, str(update["error"]))
    gateway._models_loaded_at = 0.0  # noqa: SLF001
    return {"ok": True, "model": model, "status": last.get("status", "done")}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    tools = tool_registry.list_tools()
    categories: dict[str, int] = {}
    for tool in tools:
        categories[tool["category"]] = categories.get(tool["category"], 0) + 1
    return {"tools": tools, "categories": categories, "count": len(tools)}


@router.post("/tools/approve")
async def approve_tool(
    call_id: str = Body(...), approved: bool = Body(True)
) -> dict[str, Any]:
    resolved = tool_registry.resolve_approval(call_id, approved)
    return {"ok": resolved, "call_id": call_id, "approved": approved}


@router.get("/tools/pending")
async def pending_approvals() -> dict[str, Any]:
    return {"pending": tool_registry.pending_approvals()}


# ---------------------------------------------------------------------------
# Hubs
# ---------------------------------------------------------------------------


@router.get("/hubs")
async def list_hubs() -> dict[str, Any]:
    hubs = await hub_registry.list_hubs()
    running = set(hub_registry.manager.running())
    return {
        "hubs": [{**h.to_dict(), "running": h.slug in running} for h in hubs],
        "kinds": sorted(hub_registry.HUB_KINDS),
        "search_backends": hub_registry.websearch.available_backends()
        if hasattr(hub_registry, "websearch")
        else [],
    }


@router.post("/hubs")
async def create_hub(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    hub = await hub_registry.create_hub(
        name=str(body.get("name", "New hub")),
        kind=str(body.get("kind", "mcp_stdio")),
        config=body.get("config") or {},
        description=str(body.get("description", "")),
        enabled=bool(body.get("enabled", True)),
        autostart=bool(body.get("autostart", False)),
    )
    return hub.to_dict()


@router.patch("/hubs/{slug}")
async def update_hub(slug: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    hub = await hub_registry.update_hub(slug, **patch)
    return hub.to_dict()


@router.delete("/hubs/{slug}")
async def delete_hub(slug: str) -> dict[str, Any]:
    return {"ok": await hub_registry.delete_hub(slug)}


@router.post("/hubs/{slug}/start")
async def start_hub(slug: str) -> dict[str, Any]:
    hub = await hub_registry.manager.start(slug)
    return hub.to_dict()


@router.post("/hubs/{slug}/stop")
async def stop_hub(slug: str) -> dict[str, Any]:
    hub = await hub_registry.manager.stop(slug)
    return hub.to_dict() if hub else {"ok": True}


@router.post("/hubs/{slug}/test")
async def test_hub(slug: str) -> dict[str, Any]:
    return await hub_registry.test_hub(slug)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@router.get("/projects")
async def list_projects() -> dict[str, Any]:
    async with session_scope() as s:
        rows = (
            await s.execute(select(Project).where(Project.archived.is_(False)).order_by(Project.name))
        ).scalars().all()
    return {"projects": [p.to_dict() for p in rows]}


@router.post("/projects")
async def create_project(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    async with session_scope() as s:
        project = Project(
            name=str(body.get("name", "Untitled")),
            description=str(body.get("description", "")),
            color=str(body.get("color", "#22d3ee")),
            icon=str(body.get("icon", "layers")),
        )
        s.add(project)
        await s.flush()
        return project.to_dict()


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    async with session_scope() as s:
        project = (
            await s.execute(select(Project).where(Project.id == project_id))
        ).scalar_one_or_none()
        if project is None:
            raise NotFound(f"No project {project_id}")
        for key, value in patch.items():
            if key not in {"id", "created_at"} and hasattr(project, key):
                setattr(project, key, value)
        await s.flush()
        return project.to_dict()


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> dict[str, Any]:
    async with session_scope() as s:
        result = await s.execute(sa_delete(Project).where(Project.id == project_id))
        return {"ok": bool(result.rowcount)}
