"""Memory, ontology, corpus and media endpoints."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core import paths
from ..core.errors import NotFound, ValidationFailed
from ..ingest import pipeline
from ..media import studio
from ..media.types import ImageRequest, VideoRequest
from ..memory import store
from ..ontology import graph, schema
from .deps import Auth

router = APIRouter(dependencies=[Auth])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@router.get("/memory")
async def list_memory(
    query: str = "",
    project_id: str | None = None,
    kind: str | None = None,
    pinned: bool | None = None,
    archived: bool = False,
    limit: int = Query(100, le=500),
    offset: int = 0,
    order: str = "recent",
) -> dict[str, Any]:
    rows = await store.list_memories(
        project_id=project_id,
        kinds=[kind] if kind else None,
        pinned=pinned,
        archived=archived,
        query=query,
        limit=limit,
        offset=offset,
        order=order,
    )
    return {"memories": [m.to_dict() for m in rows], "stats": await store.stats(project_id)}


@router.post("/memory/search")
async def search_memory(
    query: str = Body(..., embed=True),
    limit: int = Body(12, embed=True),
    project_id: str | None = Body(None, embed=True),
) -> dict[str, Any]:
    hits = await store.recall(query, limit=limit, project_id=project_id)
    return {"hits": [h.to_dict() for h in hits], "count": len(hits)}


class MemoryBody(BaseModel):
    content: str
    kind: str = "fact"
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    project_id: str | None = None
    pinned: bool = False


@router.post("/memory")
async def create_memory(body: MemoryBody) -> dict[str, Any]:
    memory = await store.remember(
        body.content,
        kind=body.kind,
        importance=body.importance,
        tags=body.tags,
        project_id=body.project_id,
        pinned=body.pinned,
        source="user",
    )
    return memory.to_dict()


@router.patch("/memory/{memory_id}")
async def patch_memory(memory_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    memory = await store.update_memory(memory_id, **patch)
    if memory is None:
        raise NotFound(f"No memory {memory_id}")
    return memory.to_dict()


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, Any]:
    return {"ok": await store.forget(memory_id)}


@router.post("/memory/consolidate")
async def consolidate_memory(
    project_id: str | None = Body(None, embed=True), dry_run: bool = Body(False, embed=True)
) -> dict[str, Any]:
    return await store.consolidate(project_id=project_id, dry_run=dry_run)


@router.post("/memory/extract")
async def extract_memory(
    text: str = Body(..., embed=True), project_id: str | None = Body(None, embed=True)
) -> dict[str, Any]:
    created = await store.extract_memories(text, project_id=project_id, source_ref="manual")
    return {"created": [m.to_dict() for m in created], "count": len(created)}


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


@router.get("/graph/schema")
async def graph_schema() -> dict[str, Any]:
    return schema.describe()


@router.get("/graph")
async def get_graph(
    project_id: str | None = None,
    types: str = "",
    limit: int = Query(400, le=2000),
    min_salience: float = 0.0,
) -> dict[str, Any]:
    result = await graph.subgraph(
        project_id=project_id,
        types=[t for t in types.split(",") if t] or None,
        limit=limit,
        min_salience=min_salience,
    )
    return {**result, "stats": await graph.stats(project_id)}


@router.get("/graph/search")
async def search_graph(
    q: str, limit: int = Query(25, le=100), types: str = "", project_id: str | None = None
) -> dict[str, Any]:
    found = await graph.search_entities(
        q, limit=limit, types=[t for t in types.split(",") if t] or None, project_id=project_id
    )
    return {"entities": [graph.node_dict(e) for e in found]}


@router.get("/graph/entity/{entity_id}")
async def get_entity(entity_id: str, depth: int = 1, limit: int = Query(200, le=1000)) -> dict[str, Any]:
    entity = await graph.get_entity(entity_id)
    if entity is None:
        raise NotFound(f"No entity {entity_id}")
    neighbourhood = await graph.neighbors(entity_id, depth=depth, limit=limit)
    return {"entity": graph.node_dict(entity), **neighbourhood}


@router.post("/graph/entity")
async def upsert_entity(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    entity = await graph.upsert_entity(
        str(body.get("type", "concept")),
        str(body.get("name", "")),
        description=str(body.get("description", "")),
        properties=body.get("properties") or {},
        aliases=body.get("aliases") or [],
        project_id=body.get("project_id"),
        source="user",
    )
    return graph.node_dict(entity)


@router.patch("/graph/entity/{entity_id}")
async def patch_entity(entity_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    entity = await graph.update_entity(entity_id, **patch)
    if entity is None:
        raise NotFound(f"No entity {entity_id}")
    return graph.node_dict(entity)


@router.delete("/graph/entity/{entity_id}")
async def delete_entity(entity_id: str) -> dict[str, Any]:
    return {"ok": await graph.delete_entity(entity_id)}


@router.post("/graph/link")
async def create_link(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    edge = await graph.link(
        str(body["source_id"]),
        str(body["target_id"]),
        str(body.get("type", "related_to")),
        label=str(body.get("label", "")),
        confidence=float(body.get("confidence", 0.7)),
        project_id=body.get("project_id"),
        source="user",
    )
    return graph.edge_dict(edge)


@router.delete("/graph/link/{edge_id}")
async def delete_link(edge_id: str) -> dict[str, Any]:
    return {"ok": await graph.unlink(edge_id)}


@router.get("/graph/path")
async def graph_path(source: str, target: str, max_depth: int = 6) -> dict[str, Any]:
    return await graph.shortest_path(source, target, max_depth=max_depth)


@router.get("/graph/centrality")
async def graph_centrality(project_id: str | None = None, limit: int = Query(50, le=200)) -> dict[str, Any]:
    return {"rows": await graph.centrality(project_id=project_id, limit=limit)}


@router.get("/graph/duplicates")
async def graph_duplicates(threshold: float = 0.9, project_id: str | None = None) -> dict[str, Any]:
    return {"candidates": await graph.find_duplicates(threshold=threshold, project_id=project_id)}


@router.post("/graph/merge")
async def merge_entities(
    keep_id: str = Body(..., embed=True), merge_ids: list[str] = Body(..., embed=True)
) -> dict[str, Any]:
    entity = await graph.merge_entities(keep_id, merge_ids)
    return graph.node_dict(entity)


@router.post("/graph/extract")
async def extract_graph(
    text: str = Body(..., embed=True), project_id: str | None = Body(None, embed=True)
) -> dict[str, Any]:
    return await graph.extract_graph(text, project_id=project_id, source_ref="manual")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@router.get("/corpus")
async def list_corpus(
    query: str = "", project_id: str | None = None, status: str | None = None,
    limit: int = Query(100, le=500), offset: int = 0,
) -> dict[str, Any]:
    docs = await pipeline.list_documents(
        query=query, project_id=project_id, status=status, limit=limit, offset=offset
    )
    return {
        "documents": [d.to_dict(exclude={"text"}) for d in docs],
        "stats": await pipeline.stats(project_id),
        "supported": sorted(pipeline.extractors.supported_extensions()),
    }


@router.post("/corpus/search")
async def search_corpus(
    query: str = Body(..., embed=True),
    limit: int = Body(12, embed=True),
    project_id: str | None = Body(None, embed=True),
) -> dict[str, Any]:
    return {"results": await pipeline.search_corpus(query, limit=limit, project_id=project_id)}


@router.get("/corpus/{document_id}")
async def get_document(document_id: str, chunks: bool = False) -> dict[str, Any]:
    doc = await pipeline.get_document(document_id)
    if doc is None:
        raise NotFound(f"No document {document_id}")
    payload: dict[str, Any] = doc.to_dict()
    if chunks:
        payload["chunks"] = [c.to_dict() for c in await pipeline.get_chunks(document_id)]
    return payload


@router.post("/corpus/upload")
async def upload_document(
    file: UploadFile = File(...), project_id: str | None = None
) -> dict[str, Any]:
    """Accept a browser upload, stage it, and ingest it."""
    staging = paths.cache_dir() / "uploads"
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / (file.filename or "upload.bin")
    target.write_bytes(await file.read())
    try:
        doc = await pipeline.ingest_file(target, project_id=project_id)
    finally:
        target.unlink(missing_ok=True)
    return doc.to_dict(exclude={"text"})


@router.post("/corpus/ingest")
async def ingest(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project_id = body.get("project_id")
    if body.get("url"):
        doc = await pipeline.ingest_url(str(body["url"]), project_id=project_id)
        return doc.to_dict(exclude={"text"})
    if body.get("text"):
        doc = await pipeline.ingest_text(
            str(body["text"]), title=str(body.get("title") or "Pasted text"), project_id=project_id
        )
        return doc.to_dict(exclude={"text"})
    if body.get("path"):
        target = Path(str(body["path"])).expanduser()
        if target.is_dir():
            docs = await pipeline.ingest_directory(target, project_id=project_id)
            return {"documents": [d.to_dict(exclude={"text"}) for d in docs], "count": len(docs)}
        doc = await pipeline.ingest_file(target, project_id=project_id)
        return doc.to_dict(exclude={"text"})
    raise ValidationFailed("Give one of: url, text, or path")


@router.post("/corpus/{document_id}/summarize")
async def summarize(document_id: str, force: bool = False) -> dict[str, Any]:
    return {"summary": await pipeline.summarize_document(document_id, force=force)}


@router.post("/corpus/{document_id}/reindex")
async def reindex(document_id: str) -> dict[str, Any]:
    doc = await pipeline.reindex(document_id)
    return doc.to_dict(exclude={"text"})


@router.delete("/corpus/{document_id}")
async def delete_document(document_id: str) -> dict[str, Any]:
    return {"ok": await pipeline.delete_document(document_id)}


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@router.get("/media")
async def list_media(
    kind: str | None = None, query: str = "", project_id: str | None = None,
    status: str | None = None, favorite: bool | None = None,
    limit: int = Query(100, le=500), offset: int = 0,
) -> dict[str, Any]:
    assets = await studio.list_assets(
        kind=kind, query=query, project_id=project_id, status=status,
        favorite=favorite, limit=limit, offset=offset,
    )
    return {"assets": [a.to_dict() for a in assets], "stats": await studio.stats()}


@router.get("/media/models")
async def media_models(kind: str = "image") -> dict[str, Any]:
    models = await studio.available_models(kind, configured_only=False)
    return {"models": [m.model_dump() for m in models]}


@router.post("/media/image")
async def create_image(body: ImageRequest, project_id: str | None = None) -> dict[str, Any]:
    assets = await studio.generate_image(body, project_id=project_id)
    return {"assets": [a.to_dict() for a in assets]}


@router.post("/media/video")
async def create_video(body: VideoRequest, project_id: str | None = None) -> dict[str, Any]:
    asset = await studio.generate_video_background(body, project_id=project_id)
    return {"asset": asset.to_dict(), "note": "Rendering. Watch media.done on the event stream."}


@router.post("/media/upload")
async def upload_media(file: UploadFile = File(...), project_id: str | None = None) -> dict[str, Any]:
    staging = paths.cache_dir() / "uploads"
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / (file.filename or "upload.bin")
    target.write_bytes(await file.read())
    try:
        asset = await studio.import_file(target, project_id=project_id)
    finally:
        target.unlink(missing_ok=True)
    return asset.to_dict()


@router.get("/media/{asset_id}/file")
async def media_file(asset_id: str, thumb: bool = False) -> FileResponse:
    asset = await studio.get_asset(asset_id)
    if asset is None:
        raise NotFound(f"No asset {asset_id}")
    path = Path(asset.thumb_path) if thumb and asset.thumb_path else studio.asset_file(asset)
    if path is None or not path.exists():
        raise NotFound("The file for this asset is missing from disk")
    return FileResponse(
        path,
        media_type="image/jpeg" if thumb and asset.thumb_path else (asset.mime_type or "application/octet-stream"),
        filename=f"{asset_id}{path.suffix}",
    )


@router.get("/media/{asset_id}/data")
async def media_data(asset_id: str) -> dict[str, Any]:
    """Base64 payload, for embedding an asset back into a model request."""
    asset = await studio.get_asset(asset_id)
    if asset is None:
        raise NotFound(f"No asset {asset_id}")
    path = studio.asset_file(asset)
    if path is None:
        raise NotFound("The file for this asset is missing from disk")
    if path.stat().st_size > 12_000_000:
        raise ValidationFailed("This asset is too large to inline")
    return {
        "id": asset_id,
        "mime_type": asset.mime_type,
        "data": base64.b64encode(path.read_bytes()).decode(),
    }


@router.patch("/media/{asset_id}")
async def patch_media(asset_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from sqlalchemy import select

    from ..db.base import session_scope
    from ..db.models import Asset

    async with session_scope() as s:
        asset = (await s.execute(select(Asset).where(Asset.id == asset_id))).scalar_one_or_none()
        if asset is None:
            raise NotFound(f"No asset {asset_id}")
        for key in ("favorite", "tags", "prompt"):
            if key in patch:
                setattr(asset, key, patch[key])
        await s.flush()
        return asset.to_dict()


@router.delete("/media/{asset_id}")
async def delete_media(asset_id: str) -> dict[str, Any]:
    return {"ok": await studio.delete_asset(asset_id)}
