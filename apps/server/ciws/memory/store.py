"""Memory: writing it, recalling it, and keeping it from rotting.

Recall is hybrid on purpose. Pure vector search misses exact tokens -- an error
code, a version number, a person's surname -- and pure keyword search misses
paraphrase. Both signals are computed, fused by reciprocal rank, then weighted
by importance and recency, and finally passed through MMR so the context window
gets twelve *different* facts rather than twelve wordings of one.

Consolidation is what stops a long-lived workspace from turning into sludge:
duplicates merge, clusters get summarised into higher-order insights, and stale
unpinned memories fade out. Pinned memories are never touched by any of it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, or_, select, text as sql_text

from ..core.config import get_settings
from ..core.errors import NotFound
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import extract_json, now, sha256, truncate
from ..db.base import session_scope
from ..db.models import Memory
from ..db.vectors import index, mmr, normalize
from . import embeddings

log = get_logger("memory")

KIND_ORDER = ("identity", "preference", "fact", "decision", "procedure", "insight", "event", "task")
VECTOR_KIND = "memory"


def _norm(content: str) -> str:
    return re.sub(r"\s+", " ", content.strip().lower())


def _fts_query(query: str) -> str:
    """Turn free text into something FTS5 will not choke on.

    User queries contain quotes, hyphens, ``AND``, and stray parentheses, all of
    which are FTS5 syntax. Quoting each token makes them literals.
    """
    from .embeddings import STOPWORDS

    tokens = re.findall(r"[\w']+", query.lower())
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS][:24]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def decay_factor(created: datetime | None, last_accessed: datetime | None, half_life_days: float) -> float:
    """Exponential decay on whichever is more recent: creation or last use."""
    if half_life_days <= 0:
        return 1.0
    reference = last_accessed or created
    if reference is None:
        return 1.0
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now() - reference).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / half_life_days))


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


async def remember(
    content: str,
    *,
    kind: str = "fact",
    importance: float = 0.5,
    confidence: float = 0.8,
    tags: list[str] | None = None,
    project_id: str | None = None,
    source: str = "agent",
    source_ref: str | None = None,
    summary: str = "",
    pinned: bool = False,
    meta: dict[str, Any] | None = None,
) -> Memory:
    """Store a memory, or strengthen the identical one that already exists."""
    content = content.strip()
    if not content:
        raise NotFound("Refusing to store an empty memory")

    digest = sha256(_norm(content))
    async with session_scope() as s:
        existing = (
            await s.execute(
                select(Memory).where(Memory.content_hash == digest, Memory.archived.is_(False))
            )
        ).scalars().first()
        if existing is not None:
            existing.access_count += 1
            existing.importance = max(existing.importance, importance)
            existing.last_accessed = now()
            merged = set(existing.tags or []) | set(tags or [])
            existing.tags = sorted(merged)
            if pinned:
                existing.pinned = True
            await s.flush()
            log.debug("Memory deduped onto %s", existing.id)
            return existing

        memory = Memory(
            content=content,
            summary=summary or truncate(content, 200, marker="..."),
            kind=kind,
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            tags=sorted(set(tags or [])),
            project_id=project_id,
            source=source,
            source_ref=source_ref,
            pinned=pinned,
            content_hash=digest,
            meta=meta or {},
            last_accessed=now(),
        )
        s.add(memory)
        await s.flush()
        mem_id, mem_project = memory.id, memory.project_id

    vector, backend = await embeddings.embed_one(f"{kind}: {content}")
    await index.upsert(VECTOR_KIND, mem_id, vector, model=backend, project_id=mem_project)

    bus.publish(
        Topic.MEMORY_WRITE, id=mem_id, kind=kind, content=truncate(content, 240), importance=importance
    )
    return memory


async def get(memory_id: str) -> Memory | None:
    async with session_scope() as s:
        return (await s.execute(select(Memory).where(Memory.id == memory_id))).scalar_one_or_none()


async def update_memory(memory_id: str, **fields: Any) -> Memory | None:
    reembed = False
    async with session_scope() as s:
        memory = (
            await s.execute(select(Memory).where(Memory.id == memory_id))
        ).scalar_one_or_none()
        if memory is None:
            return None
        for key, value in fields.items():
            if key in {"id", "created_at", "content_hash"} or not hasattr(memory, key):
                continue
            if key == "content" and value and value != memory.content:
                memory.content_hash = sha256(_norm(value))
                reembed = True
            setattr(memory, key, value)
        await s.flush()
        snapshot = (memory.id, memory.kind, memory.content, memory.project_id)

    if reembed:
        mem_id, kind, content, project_id = snapshot
        vector, backend = await embeddings.embed_one(f"{kind}: {content}")
        await index.upsert(VECTOR_KIND, mem_id, vector, model=backend, project_id=project_id)
    return memory


async def forget(memory_id: str) -> bool:
    async with session_scope() as s:
        result = await s.execute(sa_delete(Memory).where(Memory.id == memory_id))
        removed = bool(result.rowcount)
    if removed:
        await index.delete(VECTOR_KIND, memory_id)
    return removed


async def touch(memory_ids: list[str]) -> None:
    if not memory_ids:
        return
    async with session_scope() as s:
        rows = (await s.execute(select(Memory).where(Memory.id.in_(memory_ids)))).scalars().all()
        stamp = now()
        for row in rows:
            row.access_count += 1
            row.last_accessed = stamp


async def list_memories(
    *,
    project_id: str | None = None,
    kinds: list[str] | None = None,
    tags: list[str] | None = None,
    pinned: bool | None = None,
    archived: bool | None = False,
    query: str = "",
    limit: int = 100,
    offset: int = 0,
    order: str = "recent",
) -> list[Memory]:
    async with session_scope() as s:
        stmt = select(Memory)
        if project_id:
            stmt = stmt.where(Memory.project_id == project_id)
        if kinds:
            stmt = stmt.where(Memory.kind.in_(kinds))
        if pinned is not None:
            stmt = stmt.where(Memory.pinned.is_(pinned))
        if archived is not None:
            stmt = stmt.where(Memory.archived.is_(archived))
        if query:
            like = f"%{query}%"
            stmt = stmt.where(or_(Memory.content.ilike(like), Memory.summary.ilike(like)))
        if order == "importance":
            stmt = stmt.order_by(Memory.importance.desc(), Memory.created_at.desc())
        elif order == "accessed":
            stmt = stmt.order_by(Memory.last_accessed.desc().nullslast())
        else:
            stmt = stmt.order_by(Memory.created_at.desc())
        rows = (await s.execute(stmt.limit(limit).offset(offset))).scalars().all()

    if tags:
        wanted = set(tags)
        rows = [r for r in rows if wanted & set(r.tags or [])]
    return list(rows)


async def stats(project_id: str | None = None) -> dict[str, Any]:
    async with session_scope() as s:
        stmt = select(Memory.kind, func.count(Memory.id)).group_by(Memory.kind)
        if project_id:
            stmt = stmt.where(Memory.project_id == project_id)
        by_kind = {k: n for k, n in (await s.execute(stmt)).all()}

        total_stmt = select(
            func.count(Memory.id), func.avg(Memory.importance), func.sum(Memory.access_count)
        )
        if project_id:
            total_stmt = total_stmt.where(Memory.project_id == project_id)
        total, avg_importance, accesses = (await s.execute(total_stmt)).one()

        pinned_stmt = select(func.count(Memory.id)).where(Memory.pinned.is_(True))
        archived_stmt = select(func.count(Memory.id)).where(Memory.archived.is_(True))
        pinned = (await s.execute(pinned_stmt)).scalar() or 0
        archived = (await s.execute(archived_stmt)).scalar() or 0

    return {
        "total": int(total or 0),
        "by_kind": by_kind,
        "pinned": int(pinned),
        "archived": int(archived),
        "avg_importance": round(float(avg_importance or 0), 3),
        "total_accesses": int(accesses or 0),
    }


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RecallHit:
    memory: Memory
    score: float
    vector_score: float = 0.0
    text_score: float = 0.0
    recency: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "text_score": round(self.text_score, 4),
            "recency": round(self.recency, 4),
            "reasons": self.reasons,
        }


async def _lexical_hits(query: str, limit: int) -> dict[str, float]:
    """FTS5 with bm25 ranking, degrading to LIKE if the index is unavailable."""
    match = _fts_query(query)
    if not match:
        return {}
    async with session_scope() as s:
        try:
            rows = (
                await s.execute(
                    sql_text(
                        "SELECT id, bm25(memories_fts) AS rank FROM memories_fts "
                        "WHERE memories_fts MATCH :q ORDER BY rank LIMIT :n"
                    ),
                    {"q": match, "n": limit},
                )
            ).all()
            # bm25() returns lower-is-better; flip it into a 0..1 score.
            return {r[0]: 1.0 / (1.0 + abs(float(r[1]))) for r in rows}
        except Exception as exc:  # noqa: BLE001
            log.debug("FTS unavailable (%s); using LIKE scan", exc)
            like = f"%{query[:80]}%"
            rows = (
                await s.execute(
                    select(Memory.id).where(Memory.content.ilike(like)).limit(limit)
                )
            ).scalars().all()
            return {rid: 0.4 for rid in rows}


async def recall(
    query: str,
    *,
    limit: int = 12,
    project_id: str | None = None,
    kinds: list[str] | None = None,
    min_score: float = 0.0,
    diversity: float = 0.25,
    include_archived: bool = False,
) -> list[RecallHit]:
    cfg = get_settings().memory
    if not cfg.enabled or not query.strip():
        return []

    pool = max(limit * 4, 40)
    query_vec, _ = await embeddings.embed_one(query)
    vector_hits = await index.search(
        VECTOR_KIND, query_vec, top_k=pool, project_id=project_id, min_score=-1.0
    )
    lexical = await _lexical_hits(query, pool)

    candidate_ids = {h.ref_id for h in vector_hits} | set(lexical)
    if not candidate_ids:
        return []

    async with session_scope() as s:
        stmt = select(Memory).where(Memory.id.in_(candidate_ids))
        if not include_archived:
            stmt = stmt.where(Memory.archived.is_(False))
        if kinds:
            stmt = stmt.where(Memory.kind.in_(kinds))
        if project_id:
            stmt = stmt.where(or_(Memory.project_id == project_id, Memory.project_id.is_(None)))
        rows = {m.id: m for m in (await s.execute(stmt)).scalars().all()}

    if not rows:
        return []

    vector_scores = {h.ref_id: h.score for h in vector_hits}
    vector_rank = {h.ref_id: i for i, h in enumerate(vector_hits)}
    lexical_rank = {mid: i for i, mid in enumerate(sorted(lexical, key=lambda m: -lexical[m]))}

    # Similarity magnitudes are not comparable across embedding backends -- a
    # strong hash-embedding match scores ~0.2 where a transformer scores ~0.8 --
    # so both signals are rescaled against the best candidate in this result set.
    top_vector = max((abs(v) for v in vector_scores.values()), default=0.0) or 1.0
    top_lexical = max(lexical.values(), default=0.0) or 1.0

    #: Reciprocal-rank fusion constant from the original RRF paper. RRF alone is
    #: nearly flat on a short candidate list, so it contributes agreement between
    #: the two retrievers rather than carrying the ranking by itself.
    K = 60.0
    hits: list[RecallHit] = []

    for mid, memory in rows.items():
        reasons: list[str] = []
        vector_score = vector_scores.get(mid, 0.0)
        text_score = lexical.get(mid, 0.0)

        rrf = 0.0
        if mid in vector_rank:
            rrf += 1.0 / (K + vector_rank[mid])
            if vector_score >= cfg.min_similarity:
                reasons.append("semantic")
        if mid in lexical_rank:
            rrf += 1.0 / (K + lexical_rank[mid])
            reasons.append("keyword")

        relevance = (
            0.60 * max(0.0, vector_score) / top_vector
            + 0.25 * text_score / top_lexical
            + 0.15 * rrf / (2.0 / K)
        )

        recency = decay_factor(memory.created_at, memory.last_accessed, cfg.decay_half_life_days)
        score = relevance * (0.70 + 0.60 * memory.importance) * (0.65 + 0.35 * recency)

        # Pinning is a nudge, not an override: it should surface a memory when
        # relevance is close, never outrank a directly matching one.
        if memory.pinned:
            score += 0.04
            reasons.append("pinned")
        if memory.access_count > 4:
            score *= 1.0 + min(0.15, math.log1p(memory.access_count) / 30)
            reasons.append("frequently used")

        hits.append(
            RecallHit(
                memory=memory,
                score=score,
                vector_score=vector_score,
                text_score=text_score,
                recency=recency,
                reasons=reasons or ["weak match"],
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    # Drop candidates that only surfaced because a stopword matched.
    floor = 0.12 * (hits[0].score if hits else 0.0)
    hits = [h for h in hits if h.score >= max(floor, min_score) or h.memory.pinned]

    if diversity > 0 and len(hits) > limit:
        hits = await _diversify(hits, query_vec, limit, diversity)
    else:
        hits = hits[:limit]

    await touch([h.memory.id for h in hits])
    bus.publish(Topic.MEMORY_RECALL, query=truncate(query, 160), hits=len(hits))
    return hits


async def _diversify(
    hits: list[RecallHit], query_vec: list[float], limit: int, diversity: float
) -> list[RecallHit]:
    """Re-rank with MMR so near-identical memories do not crowd each other out."""
    await index.load()
    kind_index = index._kinds.get(VECTOR_KIND)  # noqa: SLF001 - same package
    if kind_index is None or kind_index.matrix is None:
        return hits[:limit]

    position = {mid: i for i, mid in enumerate(kind_index.ids)}
    candidates: list[tuple[str, np.ndarray, float]] = []
    for hit in hits[: limit * 3]:
        pos = position.get(hit.memory.id)
        if pos is None:
            continue
        candidates.append((hit.memory.id, kind_index.matrix[pos], hit.score))

    if not candidates:
        return hits[:limit]

    chosen = set(mmr(normalize(query_vec), candidates, limit, diversity))
    ordered = [h for h in hits if h.memory.id in chosen]
    # Backfill from the tail if MMR could not place everything (e.g. missing vectors).
    for hit in hits:
        if len(ordered) >= limit:
            break
        if hit.memory.id not in chosen:
            ordered.append(hit)
    return ordered[:limit]


async def build_context(
    query: str, *, limit: int = 12, project_id: str | None = None, max_chars: int = 4000
) -> str:
    """A compact markdown block for the system prompt. Empty when nothing fits."""
    hits = await recall(query, limit=limit, project_id=project_id)
    if not hits:
        return ""

    grouped: dict[str, list[RecallHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.memory.kind, []).append(hit)

    lines = ["## What you remember about this user and their work", ""]
    used = len(lines[0])
    for kind in sorted(grouped, key=lambda k: KIND_ORDER.index(k) if k in KIND_ORDER else 99):
        entries = sorted(grouped[kind], key=lambda h: h.memory.importance, reverse=True)
        header = f"**{kind.title()}**"
        lines.append(header)
        used += len(header)
        for hit in entries:
            body = hit.memory.content.strip().replace("\n", " ")
            line = f"- {body}  `[{hit.memory.id}]`"
            if used + len(line) > max_chars:
                lines.append("- ...(more memories available via the memory_search tool)")
                return "\n".join(lines)
            lines.append(line)
            used += len(line)
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You maintain the long-term memory of a personal intelligence workspace.

Read the exchange below and extract ONLY durable facts worth remembering weeks from now:
stable preferences, identity details, decisions and their reasons, commitments, recurring
people/projects/systems, and hard-won procedures.

Do NOT extract: pleasantries, restatements of the question, anything true only right now,
speculation, or the assistant's own reasoning.

Return a JSON array (possibly empty). Each element:
{"content": "a single self-contained sentence, written so it makes sense with no other context",
 "kind": "fact|preference|identity|decision|procedure|event|insight",
 "importance": 0.0-1.0,
 "tags": ["short","lowercase"]}

Return [] rather than inventing something. Exchange:
---
%s
---"""


async def extract_memories(
    text: str, *, project_id: str | None = None, source_ref: str | None = None, model: str = "fast"
) -> list[Memory]:
    """Ask a model what is worth keeping. Returns [] when no model is configured."""
    if not text.strip():
        return []
    try:
        from ..gateway.registry import gateway

        raw = await gateway.complete(
            EXTRACT_PROMPT % truncate(text, 12_000),
            model=model,
            max_tokens=2000,
            system="You return only valid JSON. No prose, no code fences.",
        )
    except Exception as exc:  # noqa: BLE001 - no model configured is the common case
        log.debug("Memory extraction skipped: %s", exc)
        return []

    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        return []

    created: list[Memory] = []
    for item in parsed[:20]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if len(content) < 8:
            continue
        try:
            created.append(
                await remember(
                    content,
                    kind=str(item.get("kind") or "fact"),
                    importance=float(item.get("importance") or 0.5),
                    tags=[str(t) for t in (item.get("tags") or [])][:8],
                    project_id=project_id,
                    source="extraction",
                    source_ref=source_ref,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Skipped a extracted memory: %s", exc)
    if created:
        log.info("Extracted %d memories", len(created))
    return created


async def extract_from_conversation(
    conversation_id: str, *, project_id: str | None = None, last_n: int = 12
) -> list[Memory]:
    from ..db.models import Message

    async with session_scope() as s:
        rows = (
            await s.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.seq.desc())
                .limit(last_n)
            )
        ).scalars().all()
    if not rows:
        return []
    rendered = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in reversed(rows) if m.content)
    return await extract_memories(rendered, project_id=project_id, source_ref=conversation_id)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

INSIGHT_PROMPT = """These related memories were recorded separately:

%s

If they together support ONE higher-order insight that none states alone, return it as
{"insight": "one sentence", "importance": 0.0-1.0}. If they are merely similar, or the
insight would just restate them, return {"insight": null}. Return JSON only."""


async def consolidate(
    *, project_id: str | None = None, model: str = "balanced", dry_run: bool = False
) -> dict[str, Any]:
    """Merge duplicates, synthesise insights, and fade what nobody uses.

    Safe to run repeatedly. Pinned memories are never merged away or archived.
    """
    cfg = get_settings().memory
    result = {"scanned": 0, "merged": 0, "insights": 0, "archived": 0, "dry_run": dry_run}

    async with session_scope() as s:
        stmt = select(Memory).where(Memory.archived.is_(False))
        if project_id:
            stmt = stmt.where(Memory.project_id == project_id)
        memories = list((await s.execute(stmt)).scalars().all())
    result["scanned"] = len(memories)
    if len(memories) < 2:
        return result

    await index.load()
    kind_index = index._kinds.get(VECTOR_KIND)  # noqa: SLF001
    if kind_index is None or kind_index.matrix is None:
        return result
    position = {mid: i for i, mid in enumerate(kind_index.ids)}
    by_id = {m.id: m for m in memories}

    merged_away: set[str] = set()
    clusters: list[list[str]] = []

    for memory in memories:
        if memory.id in merged_away:
            continue
        pos = position.get(memory.id)
        if pos is None:
            continue
        sims = kind_index.matrix @ kind_index.matrix[pos]
        near = [
            kind_index.ids[i]
            for i in np.argsort(-sims)[:8]
            if kind_index.ids[i] != memory.id
            and kind_index.ids[i] in by_id
            and kind_index.ids[i] not in merged_away
            and sims[i] >= 0.80
        ]
        duplicates = [
            mid for mid in near if float(sims[position[mid]]) >= 0.93 and not by_id[mid].pinned
        ]
        if duplicates:
            if not dry_run:
                await _merge_into(memory.id, duplicates)
            merged_away.update(duplicates)
            result["merged"] += len(duplicates)
        related = [m for m in near if m not in merged_away]
        if len(related) >= 2:
            clusters.append([memory.id, *related[:4]])

    for cluster in clusters[:6]:
        members = [by_id[m] for m in cluster if m in by_id and m not in merged_away]
        if len(members) < 3:
            continue
        insight = await _synthesize(members, model)
        if insight and not dry_run:
            await remember(
                insight["content"],
                kind="insight",
                importance=insight["importance"],
                project_id=project_id,
                source="consolidation",
                tags=["consolidated"],
                meta={"derived_from": [m.id for m in members]},
            )
            result["insights"] += 1

    cutoff = now() - timedelta(days=cfg.decay_half_life_days * 3)
    if cfg.importance_floor > 0 and not dry_run:
        async with session_scope() as s:
            stale = (
                await s.execute(
                    select(Memory).where(
                        Memory.archived.is_(False),
                        Memory.pinned.is_(False),
                        Memory.importance < cfg.importance_floor,
                        Memory.created_at < cutoff,
                    )
                )
            ).scalars().all()
            for memory in stale:
                if decay_factor(
                    memory.created_at, memory.last_accessed, cfg.decay_half_life_days
                ) < 0.1:
                    memory.archived = True
                    result["archived"] += 1

    bus.publish(Topic.MEMORY_CONSOLIDATE, **result)
    log.info("Consolidation: %s", result)
    return result


async def _merge_into(keep_id: str, drop_ids: list[str]) -> None:
    async with session_scope() as s:
        keep = (await s.execute(select(Memory).where(Memory.id == keep_id))).scalar_one_or_none()
        if keep is None:
            return
        drops = (await s.execute(select(Memory).where(Memory.id.in_(drop_ids)))).scalars().all()
        tags = set(keep.tags or [])
        for drop in drops:
            tags |= set(drop.tags or [])
            keep.access_count += drop.access_count
            keep.importance = max(keep.importance, drop.importance)
            drop.archived = True
            drop.superseded_by = keep.id
        keep.tags = sorted(tags)
    await index.delete_many(VECTOR_KIND, drop_ids)


async def _synthesize(members: list[Memory], model: str) -> dict[str, Any] | None:
    try:
        from ..gateway.registry import gateway

        rendered = "\n".join(f"- {m.content}" for m in members)
        raw = await gateway.complete(
            INSIGHT_PROMPT % rendered,
            model=model,
            max_tokens=500,
            system="You return only valid JSON.",
        )
    except Exception:  # noqa: BLE001
        return None
    parsed = extract_json(raw)
    if not isinstance(parsed, dict) or not parsed.get("insight"):
        return None
    return {
        "content": str(parsed["insight"]).strip(),
        "importance": float(parsed.get("importance") or 0.6),
    }


async def maybe_consolidate(project_id: str | None = None) -> dict[str, Any] | None:
    """Run a pass only once enough new memories have accumulated."""
    cfg = get_settings().memory
    async with session_scope() as s:
        pending = (
            await s.execute(
                select(func.count(Memory.id)).where(
                    Memory.archived.is_(False), Memory.source != "consolidation"
                )
            )
        ).scalar() or 0
    if pending < cfg.consolidate_after:
        return None
    return await consolidate(project_id=project_id)
