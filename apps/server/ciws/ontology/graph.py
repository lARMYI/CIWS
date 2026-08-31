"""The entity graph: upserts, traversal, and link analysis.

Node and edge dictionaries returned from here are already UI-shaped -- colours
resolved, degree computed, properties flattened. The graph canvas should be able
to render a response without reshaping it, because reshaping on the client is
where the two representations drift apart.

Everything is capped. ``neighbors`` at depth 3 on a well-connected workspace can
touch the entire graph, and a force-directed layout dies somewhere north of a
couple of thousand nodes, so the traversal stops at ``limit`` and says so rather
than returning something the browser cannot draw.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

import numpy as np
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, or_, select, text as sql_text

from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import now, truncate
from ..db.base import session_scope
from ..db.models import Edge, Entity
from ..db.vectors import index
from ..memory import embeddings
from . import schema

log = get_logger("ontology")

VECTOR_KIND = "entity"
MAX_NODES = 2000


def node_dict(entity: Entity, degree: int = 0) -> dict[str, Any]:
    meta = schema.entity_meta(entity.type)
    return {
        "id": entity.id,
        "type": entity.type,
        "type_label": meta["label"],
        "name": entity.name,
        "description": entity.description,
        "aliases": entity.aliases or [],
        "properties": entity.properties or {},
        "salience": round(entity.salience, 4),
        "mention_count": entity.mention_count,
        "confidence": entity.confidence,
        "degree": degree,
        "color": meta["color"],
        "icon": meta["icon"],
        "project_id": entity.project_id,
        "sources": entity.sources or [],
        "created_at": entity.created_at.isoformat() if entity.created_at else None,
    }


def edge_dict(edge: Edge) -> dict[str, Any]:
    meta = schema.edge_meta(edge.type)
    return {
        "id": edge.id,
        "source": edge.source_id,
        "target": edge.target_id,
        "type": edge.type,
        "label": edge.label or meta["label"],
        "weight": edge.weight,
        "confidence": edge.confidence,
        "directed": edge.directed,
        "observed_count": edge.observed_count,
        "color": meta["color"],
        "properties": edge.properties or {},
    }


def _embed_text(name: str, aliases: list[str], description: str) -> str:
    parts = [name, *aliases[:6]]
    if description:
        parts.append(truncate(description, 400))
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def upsert_entity(
    entity_type: str,
    name: str,
    *,
    project_id: str | None = None,
    description: str = "",
    properties: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
    confidence: float = 0.7,
    source: str | None = None,
) -> Entity:
    """Create the entity, or reinforce the one that is already there."""
    name = (name or "").strip()
    if not name:
        raise NotFound("An entity needs a name")

    etype = schema.normalize_type(entity_type)
    key = schema.canonical_key(etype, name)

    async with session_scope() as s:
        entity = (
            await s.execute(
                select(Entity).where(Entity.type == etype, Entity.canonical_key == key)
            )
        ).scalar_one_or_none()

        if entity is None:
            entity = Entity(
                type=etype,
                name=name,
                canonical_key=key,
                description=description,
                properties=properties or {},
                aliases=sorted(set(aliases or [])),
                confidence=confidence,
                project_id=project_id,
                sources=[source] if source else [],
                salience=1.0,
            )
            s.add(entity)
            created = True
        else:
            entity.mention_count += 1
            entity.salience += 1.0
            entity.confidence = max(entity.confidence, confidence)
            if description and len(description) > len(entity.description):
                entity.description = description
            if properties:
                # New values win, but never drop a key we already learned.
                entity.properties = {**(entity.properties or {}), **properties}
            if aliases:
                entity.aliases = sorted(set(entity.aliases or []) | set(aliases))
            if source and source not in (entity.sources or []):
                entity.sources = [*(entity.sources or []), source][-30:]
            created = False

        await s.flush()
        snapshot = (
            entity.id,
            entity.name,
            list(entity.aliases or []),
            entity.description,
            entity.project_id,
        )

    ent_id, ent_name, ent_aliases, ent_desc, ent_project = snapshot
    vector, backend = await embeddings.embed_one(_embed_text(ent_name, ent_aliases, ent_desc))
    await index.upsert(VECTOR_KIND, ent_id, vector, model=backend, project_id=ent_project)

    bus.publish(
        Topic.GRAPH_ENTITY, id=ent_id, type=etype, name=ent_name, created=created
    )
    return entity


async def get_entity(entity_id: str, *, follow_merges: bool = True) -> Entity | None:
    """Fetch an entity, following ``merged_into`` so stale ids still resolve."""
    seen: set[str] = set()
    async with session_scope() as s:
        current = entity_id
        while current and current not in seen:
            seen.add(current)
            entity = (
                await s.execute(select(Entity).where(Entity.id == current))
            ).scalar_one_or_none()
            if entity is None:
                return None
            if follow_merges and entity.merged_into:
                current = entity.merged_into
                continue
            return entity
    return None


async def update_entity(entity_id: str, **fields: Any) -> Entity | None:
    reembed = False
    async with session_scope() as s:
        entity = (await s.execute(select(Entity).where(Entity.id == entity_id))).scalar_one_or_none()
        if entity is None:
            return None
        for key, value in fields.items():
            if key in {"id", "created_at", "canonical_key"} or not hasattr(entity, key):
                continue
            if key in {"name", "description", "aliases"}:
                reembed = True
            setattr(entity, key, value)
        if "name" in fields or "type" in fields:
            entity.canonical_key = schema.canonical_key(entity.type, entity.name)
        await s.flush()
        snapshot = (entity.id, entity.name, list(entity.aliases or []), entity.description, entity.project_id)

    if reembed:
        eid, name, aliases, desc, project = snapshot
        vector, backend = await embeddings.embed_one(_embed_text(name, aliases, desc))
        await index.upsert(VECTOR_KIND, eid, vector, model=backend, project_id=project)
    return entity


async def delete_entity(entity_id: str) -> bool:
    async with session_scope() as s:
        await s.execute(
            sa_delete(Edge).where(or_(Edge.source_id == entity_id, Edge.target_id == entity_id))
        )
        result = await s.execute(sa_delete(Entity).where(Entity.id == entity_id))
        removed = bool(result.rowcount)
    if removed:
        await index.delete(VECTOR_KIND, entity_id)
    return removed


async def link(
    source_id: str,
    target_id: str,
    edge_type: str,
    *,
    label: str = "",
    weight: float = 1.0,
    confidence: float = 0.7,
    directed: bool = True,
    project_id: str | None = None,
    properties: dict[str, Any] | None = None,
    source: str | None = None,
) -> Edge:
    if source_id == target_id:
        raise NotFound("An entity cannot be linked to itself")

    etype = schema.normalize_edge_type(edge_type)
    async with session_scope() as s:
        existing = (
            await s.execute(
                select(Edge).where(
                    Edge.source_id == source_id,
                    Edge.target_id == target_id,
                    Edge.type == etype,
                )
            )
        ).scalar_one_or_none()

        if existing is not None:
            # Repeated observation is evidence, not a duplicate row.
            existing.observed_count += 1
            existing.weight = min(10.0, existing.weight + 0.25)
            existing.confidence = max(existing.confidence, confidence)
            if properties:
                existing.properties = {**(existing.properties or {}), **properties}
            if source and source not in (existing.sources or []):
                existing.sources = [*(existing.sources or []), source][-30:]
            await s.flush()
            edge = existing
        else:
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                type=etype,
                label=label or schema.edge_meta(etype)["label"],
                weight=weight,
                confidence=confidence,
                directed=directed and schema.edge_meta(etype).get("directed", True),
                project_id=project_id,
                properties=properties or {},
                sources=[source] if source else [],
            )
            s.add(edge)
            await s.flush()

        payload = edge_dict(edge)

    bus.publish(Topic.GRAPH_EDGE, **payload)
    return edge


async def unlink(edge_id: str) -> bool:
    async with session_scope() as s:
        result = await s.execute(sa_delete(Edge).where(Edge.id == edge_id))
        return bool(result.rowcount)


# ---------------------------------------------------------------------------
# Search and traversal
# ---------------------------------------------------------------------------


async def search_entities(
    query: str,
    *,
    limit: int = 25,
    types: list[str] | None = None,
    project_id: str | None = None,
) -> list[Entity]:
    """Hybrid name/alias/description search."""
    if not query.strip():
        async with session_scope() as s:
            stmt = select(Entity).where(Entity.merged_into.is_(None))
            if types:
                stmt = stmt.where(Entity.type.in_(types))
            if project_id:
                stmt = stmt.where(Entity.project_id == project_id)
            return list(
                (await s.execute(stmt.order_by(Entity.salience.desc()).limit(limit))).scalars().all()
            )

    from ..memory.store import _fts_query  # shared sanitiser

    vector, _ = await embeddings.embed_one(query)
    vector_hits = await index.search(VECTOR_KIND, vector, top_k=limit * 3, min_score=-1.0)
    scores: dict[str, float] = {h.ref_id: max(0.0, h.score) for h in vector_hits}

    match = _fts_query(query)
    async with session_scope() as s:
        if match:
            try:
                rows = (
                    await s.execute(
                        sql_text(
                            "SELECT id, bm25(entities_fts) AS rank FROM entities_fts "
                            "WHERE entities_fts MATCH :q ORDER BY rank LIMIT :n"
                        ),
                        {"q": match, "n": limit * 3},
                    )
                ).all()
                for rid, rank in rows:
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (1.0 + abs(float(rank)))
            except Exception as exc:  # noqa: BLE001
                log.debug("entity FTS unavailable: %s", exc)

        like = f"%{query.strip()[:60]}%"
        exact = (
            await s.execute(
                select(Entity.id).where(Entity.name.ilike(like)).limit(limit)
            )
        ).scalars().all()
        for rid in exact:
            scores[rid] = scores.get(rid, 0.0) + 0.8

        if not scores:
            return []

        stmt = select(Entity).where(Entity.id.in_(scores), Entity.merged_into.is_(None))
        if types:
            stmt = stmt.where(Entity.type.in_(types))
        if project_id:
            stmt = stmt.where(or_(Entity.project_id == project_id, Entity.project_id.is_(None)))
        found = (await s.execute(stmt)).scalars().all()

    ranked = sorted(found, key=lambda e: scores.get(e.id, 0.0) + e.salience * 0.01, reverse=True)
    # Vector search returns the whole index ordered by similarity, so without a
    # floor every entity in the workspace comes back as a "result".
    if ranked:
        best = scores.get(ranked[0].id, 0.0)
        cutoff = max(0.05, best * 0.30)
        ranked = [e for e in ranked if scores.get(e.id, 0.0) >= cutoff]
    return ranked[:limit]


async def _degrees(entity_ids: list[str]) -> dict[str, int]:
    if not entity_ids:
        return {}
    async with session_scope() as s:
        out: dict[str, int] = defaultdict(int)
        for column in (Edge.source_id, Edge.target_id):
            rows = (
                await s.execute(
                    select(column, func.count(Edge.id))
                    .where(column.in_(entity_ids))
                    .group_by(column)
                )
            ).all()
            for eid, count in rows:
                out[eid] += int(count)
        return dict(out)


async def neighbors(
    entity_id: str,
    *,
    depth: int = 1,
    limit: int = 250,
    types: list[str] | None = None,
    edge_types: list[str] | None = None,
) -> dict[str, Any]:
    """Breadth-first expansion from one entity."""
    root = await get_entity(entity_id)
    if root is None:
        raise NotFound(f"No entity {entity_id}")

    limit = min(limit, MAX_NODES)
    visited: set[str] = {root.id}
    frontier: deque[tuple[str, int]] = deque([(root.id, 0)])
    edge_ids: set[str] = set()
    truncated = False

    async with session_scope() as s:
        while frontier:
            current, level = frontier.popleft()
            if level >= depth:
                continue
            stmt = select(Edge).where(
                or_(Edge.source_id == current, Edge.target_id == current)
            )
            if edge_types:
                stmt = stmt.where(Edge.type.in_(edge_types))
            rows = (await s.execute(stmt.limit(limit))).scalars().all()
            for edge in rows:
                edge_ids.add(edge.id)
                other = edge.target_id if edge.source_id == current else edge.source_id
                if other in visited:
                    continue
                if len(visited) >= limit:
                    truncated = True
                    break
                visited.add(other)
                frontier.append((other, level + 1))
            if truncated:
                break

        node_stmt = select(Entity).where(Entity.id.in_(visited))
        if types:
            node_stmt = node_stmt.where(or_(Entity.type.in_(types), Entity.id == root.id))
        entities = (await s.execute(node_stmt)).scalars().all()
        kept = {e.id for e in entities}
        edges = [
            e
            for e in (
                await s.execute(select(Edge).where(Edge.id.in_(edge_ids)))
            ).scalars().all()
            if e.source_id in kept and e.target_id in kept
        ]

    degrees = await _degrees(list(kept))
    return {
        "root": root.id,
        "nodes": [node_dict(e, degrees.get(e.id, 0)) for e in entities],
        "edges": [edge_dict(e) for e in edges],
        "truncated": truncated,
        "depth": depth,
    }


async def subgraph(
    *,
    entity_ids: list[str] | None = None,
    project_id: str | None = None,
    types: list[str] | None = None,
    limit: int = 500,
    min_salience: float = 0.0,
) -> dict[str, Any]:
    """A slice of the graph, defaulting to the most salient entities."""
    limit = min(limit, MAX_NODES)
    async with session_scope() as s:
        stmt = select(Entity).where(Entity.merged_into.is_(None))
        if entity_ids:
            stmt = stmt.where(Entity.id.in_(entity_ids))
        if project_id:
            stmt = stmt.where(Entity.project_id == project_id)
        if types:
            stmt = stmt.where(Entity.type.in_(types))
        if min_salience:
            stmt = stmt.where(Entity.salience >= min_salience)
        entities = (
            await s.execute(stmt.order_by(Entity.salience.desc()).limit(limit))
        ).scalars().all()
        ids = [e.id for e in entities]

        edges = []
        if ids:
            edges = (
                await s.execute(
                    select(Edge).where(Edge.source_id.in_(ids), Edge.target_id.in_(ids))
                )
            ).scalars().all()

    degrees = await _degrees(ids)
    return {
        "nodes": [node_dict(e, degrees.get(e.id, 0)) for e in entities],
        "edges": [edge_dict(e) for e in edges],
        "truncated": len(entities) >= limit,
    }


async def shortest_path(source_id: str, target_id: str, *, max_depth: int = 6) -> dict[str, Any]:
    """Bidirectional BFS over the undirected projection.

    This is the question link analysis exists to answer -- "how is this person
    connected to that system?" -- so it ignores edge direction: a path through
    ``created_by`` backwards is still a path.
    """
    if source_id == target_id:
        entity = await get_entity(source_id)
        return {
            "found": bool(entity),
            "nodes": [node_dict(entity)] if entity else [],
            "edges": [],
            "hops": 0,
        }

    async with session_scope() as s:
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in (await s.execute(select(Edge))).scalars().all():
            adjacency[edge.source_id].append((edge.target_id, edge.id))
            adjacency[edge.target_id].append((edge.source_id, edge.id))

    if source_id not in adjacency and target_id not in adjacency:
        return {"found": False, "nodes": [], "edges": [], "hops": 0}

    forward: dict[str, tuple[str | None, str | None]] = {source_id: (None, None)}
    backward: dict[str, tuple[str | None, str | None]] = {target_id: (None, None)}
    fq, bq = deque([source_id]), deque([target_id])
    meet: str | None = None

    for _ in range(max_depth):
        for queue, seen, other in ((fq, forward, backward), (bq, backward, forward)):
            for _ in range(len(queue)):
                node = queue.popleft()
                for neighbor, edge_id in adjacency.get(node, []):
                    if neighbor in seen:
                        continue
                    seen[neighbor] = (node, edge_id)
                    if neighbor in other:
                        meet = neighbor
                        break
                    queue.append(neighbor)
                if meet:
                    break
            if meet:
                break
        if meet:
            break

    if meet is None:
        return {"found": False, "nodes": [], "edges": [], "hops": 0}

    def walk(table: dict[str, tuple[str | None, str | None]], start: str) -> tuple[list[str], list[str]]:
        nodes, edges = [start], []
        current = start
        while table[current][0] is not None:
            parent, edge_id = table[current]
            edges.append(edge_id)  # type: ignore[arg-type]
            nodes.append(parent)  # type: ignore[arg-type]
            current = parent  # type: ignore[assignment]
        return nodes, edges

    fnodes, fedges = walk(forward, meet)
    bnodes, bedges = walk(backward, meet)
    path_nodes = list(reversed(fnodes)) + bnodes[1:]
    path_edges = list(reversed(fedges)) + bedges

    async with session_scope() as s:
        entities = (await s.execute(select(Entity).where(Entity.id.in_(path_nodes)))).scalars().all()
        edges = (await s.execute(select(Edge).where(Edge.id.in_(path_edges)))).scalars().all()

    order = {eid: i for i, eid in enumerate(path_nodes)}
    ordered = sorted(entities, key=lambda e: order.get(e.id, 999))
    return {
        "found": True,
        "nodes": [node_dict(e) for e in ordered],
        "edges": [edge_dict(e) for e in edges],
        "hops": len(path_edges),
        "path": path_nodes,
    }


async def centrality(*, project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Degree centrality plus PageRank.

    Degree finds hubs; PageRank finds entities that are important *because
    important things point at them*, which is usually the more interesting
    answer when you are looking for what a workspace actually revolves around.
    """
    async with session_scope() as s:
        stmt = select(Entity).where(Entity.merged_into.is_(None))
        if project_id:
            stmt = stmt.where(Entity.project_id == project_id)
        entities = (await s.execute(stmt.limit(MAX_NODES))).scalars().all()
        if not entities:
            return []
        ids = [e.id for e in entities]
        pos = {eid: i for i, eid in enumerate(ids)}
        edges = (
            await s.execute(select(Edge).where(Edge.source_id.in_(ids), Edge.target_id.in_(ids)))
        ).scalars().all()

    n = len(ids)
    adjacency = np.zeros((n, n), dtype=np.float32)
    degree = np.zeros(n, dtype=np.int32)
    for edge in edges:
        i, j = pos[edge.source_id], pos[edge.target_id]
        adjacency[j, i] += edge.weight
        degree[i] += 1
        degree[j] += 1
        if not edge.directed:
            adjacency[i, j] += edge.weight

    col_sums = adjacency.sum(axis=0)
    # Dangling nodes (no outbound edges) would leak rank; spread them uniformly.
    dangling = col_sums == 0
    adjacency[:, dangling] = 1.0 / n
    col_sums = adjacency.sum(axis=0)
    adjacency /= np.where(col_sums == 0, 1.0, col_sums)

    damping = 0.85
    rank = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(24):
        updated = (1 - damping) / n + damping * (adjacency @ rank)
        if float(np.abs(updated - rank).sum()) < 1e-7:
            rank = updated
            break
        rank = updated

    max_degree = max(1, int(degree.max()))
    max_rank = float(rank.max()) or 1.0
    rows = [
        {
            "entity_id": e.id,
            "name": e.name,
            "type": e.type,
            "color": schema.entity_color(e.type),
            "degree": int(degree[pos[e.id]]),
            "pagerank": round(float(rank[pos[e.id]]), 6),
            "score": round(
                0.4 * (degree[pos[e.id]] / max_degree) + 0.6 * (float(rank[pos[e.id]]) / max_rank),
                4,
            ),
        }
        for e in entities
    ]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


async def timeline(*, project_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    async with session_scope() as s:
        stmt = select(Entity).where(Entity.merged_into.is_(None))
        if project_id:
            stmt = stmt.where(Entity.project_id == project_id)
        entities = (
            await s.execute(stmt.order_by(Entity.created_at.desc()).limit(limit))
        ).scalars().all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "color": schema.entity_color(e.type),
            "at": e.created_at.isoformat() if e.created_at else None,
            "mention_count": e.mention_count,
        }
        for e in entities
    ]


async def stats(project_id: str | None = None) -> dict[str, Any]:
    async with session_scope() as s:
        e_stmt = select(Entity.type, func.count(Entity.id)).group_by(Entity.type)
        if project_id:
            e_stmt = e_stmt.where(Entity.project_id == project_id)
        by_type = {t: int(c) for t, c in (await s.execute(e_stmt)).all()}

        edge_stmt = select(Edge.type, func.count(Edge.id)).group_by(Edge.type)
        if project_id:
            edge_stmt = edge_stmt.where(Edge.project_id == project_id)
        by_edge = {t: int(c) for t, c in (await s.execute(edge_stmt)).all()}

        merged = (
            await s.execute(select(func.count(Entity.id)).where(Entity.merged_into.isnot(None)))
        ).scalar() or 0

    entity_count = sum(by_type.values())
    edge_count = sum(by_edge.values())
    possible = entity_count * (entity_count - 1)
    return {
        "entities": entity_count,
        "edges": edge_count,
        "by_type": by_type,
        "by_edge_type": by_edge,
        "merged": int(merged),
        "density": round(edge_count / possible, 5) if possible else 0.0,
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract the entity graph from the text below.

Entity types to prefer: %s
Relationship types to prefer: %s
If nothing fits, invent a short snake_case type rather than forcing a bad fit.

Rules:
- Only entities actually present in the text. Never infer people or systems that are not named.
- Use the fullest form of each name that appears ("Sarah Chen", not "Sarah").
- relationships must reference entities by their exact "name" from your own entities list.
- Return {"entities": [], "relationships": []} if the text has no meaningful entities.

Return ONLY JSON:
{"entities":[{"name":"","type":"","description":"","aliases":[],"properties":{}}],
 "relationships":[{"source":"","target":"","type":"","label":"","confidence":0.0}]}

Text:
---
%s
---"""


async def extract_graph(
    text: str,
    *,
    project_id: str | None = None,
    model: str = "balanced",
    source_ref: str | None = None,
) -> dict[str, Any]:
    """Pull entities and relationships out of text and persist them."""
    from ..core.util import extract_json

    result: dict[str, Any] = {"entities": [], "edges": [], "skipped": 0}
    if not text.strip():
        return result

    try:
        from ..gateway.registry import gateway

        raw = await gateway.complete(
            EXTRACT_PROMPT
            % (
                ", ".join(list(schema.ENTITY_TYPES)[:15]),
                ", ".join(list(schema.EDGE_TYPES)[:15]),
                truncate(text, 14_000),
            ),
            model=model,
            max_tokens=4000,
            system="You are a precise information extraction system. You return only valid JSON.",
        )
    except Exception as exc:  # noqa: BLE001 - no model configured is normal
        log.debug("Graph extraction skipped: %s", exc)
        return result

    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return result

    by_name: dict[str, Entity] = {}
    for item in (parsed.get("entities") or [])[:60]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if len(name) < 2:
            result["skipped"] += 1
            continue
        try:
            entity = await upsert_entity(
                str(item.get("type") or "concept"),
                name,
                project_id=project_id,
                description=str(item.get("description") or "")[:1000],
                properties=item.get("properties") if isinstance(item.get("properties"), dict) else {},
                aliases=[str(a) for a in (item.get("aliases") or [])][:8],
                source=source_ref,
            )
            by_name[name.lower()] = entity
            for alias in entity.aliases or []:
                by_name.setdefault(str(alias).lower(), entity)
            result["entities"].append(node_dict(entity))
        except Exception as exc:  # noqa: BLE001
            log.debug("Skipped entity %r: %s", name, exc)
            result["skipped"] += 1

    for item in (parsed.get("relationships") or [])[:120]:
        if not isinstance(item, dict):
            continue
        src = by_name.get(str(item.get("source") or "").strip().lower())
        dst = by_name.get(str(item.get("target") or "").strip().lower())
        if src is None or dst is None or src.id == dst.id:
            result["skipped"] += 1
            continue
        try:
            edge = await link(
                src.id,
                dst.id,
                str(item.get("type") or "related_to"),
                label=str(item.get("label") or "")[:200],
                confidence=float(item.get("confidence") or 0.7),
                project_id=project_id,
                source=source_ref,
            )
            result["edges"].append(edge_dict(edge))
        except Exception as exc:  # noqa: BLE001
            log.debug("Skipped relationship: %s", exc)
            result["skipped"] += 1

    if result["entities"]:
        log.info(
            "Graph extraction: %d entities, %d edges, %d skipped",
            len(result["entities"]), len(result["edges"]), result["skipped"],
        )
    return result


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


async def find_duplicates(
    *, threshold: float = 0.90, project_id: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """Candidate merges, from vector similarity and from name collisions."""
    async with session_scope() as s:
        stmt = select(Entity).where(Entity.merged_into.is_(None))
        if project_id:
            stmt = stmt.where(Entity.project_id == project_id)
        entities = (await s.execute(stmt.limit(MAX_NODES))).scalars().all()
    if len(entities) < 2:
        return []

    by_id = {e.id: e for e in entities}
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    # Same type, same normalised name, different row -- an exact collision the
    # canonical key missed because of an alias or a rename.
    by_key: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        by_key[entity.canonical_key].append(entity)
        for alias in entity.aliases or []:
            by_key[schema.canonical_key(entity.type, str(alias))].append(entity)
    for group in by_key.values():
        unique = {e.id: e for e in group}
        if len(unique) < 2:
            continue
        ordered = sorted(unique.values(), key=lambda e: (-e.mention_count, e.created_at or now()))
        for other in ordered[1:]:
            key = tuple(sorted((ordered[0].id, other.id)))
            pairs[key] = {  # type: ignore[index]
                "a": node_dict(ordered[0]),
                "b": node_dict(other),
                "similarity": 1.0,
                "reason": "identical name or alias",
            }

    await index.load()
    kind_index = index._kinds.get(VECTOR_KIND)  # noqa: SLF001
    if kind_index is not None and kind_index.matrix is not None:
        position = {eid: i for i, eid in enumerate(kind_index.ids)}
        for entity in entities:
            pos = position.get(entity.id)
            if pos is None:
                continue
            sims = kind_index.matrix @ kind_index.matrix[pos]
            for i in np.argsort(-sims)[:6]:
                other_id = kind_index.ids[i]
                if other_id == entity.id or other_id not in by_id:
                    continue
                score = float(sims[i])
                if score < threshold:
                    break
                other = by_id[other_id]
                if other.type != entity.type:
                    continue  # a person and a project with the same name are not duplicates
                key = tuple(sorted((entity.id, other_id)))
                if key in pairs:  # type: ignore[comparison-overlap]
                    continue
                pairs[key] = {  # type: ignore[index]
                    "a": node_dict(entity),
                    "b": node_dict(other),
                    "similarity": round(score, 4),
                    "reason": "high semantic similarity",
                }

    # Drop anything a human has already looked at and rejected. Without this a
    # curation queue never empties: the same false positive is re-proposed every
    # time the panel opens, and the queue stops being a to-do list.
    dismissed = {
        tuple(sorted((entity.id, str(other))))
        for entity in entities
        for other in (entity.properties or {}).get(NOT_DUPLICATE_OF, [])
    }
    rows = [
        pair
        for key, pair in pairs.items()
        if tuple(sorted((pair["a"]["id"], pair["b"]["id"]))) not in dismissed
    ]
    rows.sort(key=lambda p: -p["similarity"])
    return rows[:limit]


#: Property key holding ids this entity has been judged *not* a duplicate of.
NOT_DUPLICATE_OF = "not_duplicate_of"


async def dismiss_duplicate(a_id: str, b_id: str) -> dict[str, Any]:
    """Record that two entities are genuinely different things.

    Written to both sides so the judgement survives whichever one is loaded
    first, and so it is visible on the entity itself rather than hidden in a
    side table nobody thinks to look in.
    """
    if a_id == b_id:
        raise ValidationFailed("An entity cannot be a duplicate of itself")

    async with session_scope() as s:
        rows = (
            await s.execute(select(Entity).where(Entity.id.in_([a_id, b_id])))
        ).scalars().all()
        found = {row.id: row for row in rows}
        for wanted in (a_id, b_id):
            if wanted not in found:
                raise NotFound(f"No entity {wanted}")

        for this_id, other_id in ((a_id, b_id), (b_id, a_id)):
            entity = found[this_id]
            properties = dict(entity.properties or {})
            existing = [str(v) for v in properties.get(NOT_DUPLICATE_OF, [])]
            if other_id not in existing:
                properties[NOT_DUPLICATE_OF] = [*existing, other_id]
                entity.properties = properties

    bus.publish(Topic.GRAPH_MERGE, action="dismissed", a=a_id, b=b_id)
    return {"ok": True, "a": a_id, "b": b_id}


async def merge_entities(keep_id: str, merge_ids: list[str]) -> Entity:
    """Fold entities together, repointing every edge and keeping provenance.

    The losers are soft-deleted via ``merged_into`` rather than removed, so an
    old id in a memory or a document citation still resolves to the survivor.
    """
    merge_ids = [m for m in merge_ids if m and m != keep_id]
    if not merge_ids:
        entity = await get_entity(keep_id)
        if entity is None:
            raise NotFound(f"No entity {keep_id}")
        return entity

    async with session_scope() as s:
        keep = (await s.execute(select(Entity).where(Entity.id == keep_id))).scalar_one_or_none()
        if keep is None:
            raise NotFound(f"No entity {keep_id}")
        losers = (await s.execute(select(Entity).where(Entity.id.in_(merge_ids)))).scalars().all()
        if not losers:
            return keep

        aliases = set(keep.aliases or [])
        sources = list(keep.sources or [])
        properties = dict(keep.properties or {})

        for loser in losers:
            aliases.add(loser.name)
            aliases |= set(loser.aliases or [])
            sources.extend(loser.sources or [])
            for k, v in (loser.properties or {}).items():
                properties.setdefault(k, v)
            keep.mention_count += loser.mention_count
            keep.salience += loser.salience
            if len(loser.description) > len(keep.description):
                keep.description = loser.description
            loser.merged_into = keep.id

        aliases.discard(keep.name)
        keep.aliases = sorted(aliases)[:60]
        keep.sources = sources[-40:]
        keep.properties = properties

        loser_ids = [loser.id for loser in losers]
        edges = (
            await s.execute(
                select(Edge).where(
                    or_(Edge.source_id.in_(loser_ids), Edge.target_id.in_(loser_ids))
                )
            )
        ).scalars().all()

        # Repoint, then drop anything that became a self-edge or a duplicate.
        seen: set[tuple[str, str, str]] = set()
        existing = (
            await s.execute(
                select(Edge).where(or_(Edge.source_id == keep.id, Edge.target_id == keep.id))
            )
        ).scalars().all()
        for edge in existing:
            seen.add((edge.source_id, edge.target_id, edge.type))

        for edge in edges:
            if edge.source_id in loser_ids:
                edge.source_id = keep.id
            if edge.target_id in loser_ids:
                edge.target_id = keep.id
            signature = (edge.source_id, edge.target_id, edge.type)
            if edge.source_id == edge.target_id or signature in seen:
                await s.delete(edge)
                continue
            seen.add(signature)

        await s.flush()
        snapshot = (keep.id, keep.name, list(keep.aliases or []), keep.description, keep.project_id)

    await index.delete_many(VECTOR_KIND, merge_ids)
    eid, name, alias_list, desc, project = snapshot
    vector, backend = await embeddings.embed_one(_embed_text(name, alias_list, desc))
    await index.upsert(VECTOR_KIND, eid, vector, model=backend, project_id=project)

    bus.publish(Topic.GRAPH_MERGE, keep=keep_id, merged=merge_ids)
    log.info("Merged %d entities into %s", len(merge_ids), keep_id)
    return keep


async def auto_resolve(
    *, threshold: float = 0.95, project_id: str | None = None, dry_run: bool = True
) -> dict[str, Any]:
    """Merge only the pairs confident enough not to need a human.

    The default is a dry run: a wrong automatic merge is annoying to undo, so
    the Ontology panel shows the proposals and you approve them.
    """
    candidates = await find_duplicates(threshold=threshold, project_id=project_id, limit=200)
    merged = 0
    for pair in candidates:
        if pair["similarity"] < threshold:
            continue
        keep, drop = pair["a"], pair["b"]
        if drop["mention_count"] > keep["mention_count"]:
            keep, drop = drop, keep
        if not dry_run:
            try:
                await merge_entities(keep["id"], [drop["id"]])
                merged += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("auto-merge skipped: %s", exc)
        else:
            merged += 1
    return {"candidates": len(candidates), "merged": merged, "dry_run": dry_run}
