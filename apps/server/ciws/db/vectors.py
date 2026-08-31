"""Vector index over everything embeddable.

Deliberately simple: vectors live as float32 blobs in the ``embeddings`` table
and are mirrored into a NumPy matrix per kind. Search is an exact dense dot
product, which for a personal workspace (tens of thousands of vectors) runs in
single-digit milliseconds and never returns the wrong neighbour the way an
approximate index can.

Vectors are L2-normalised on write, so cosine similarity is just a dot product.
The matrix is a cache -- SQLite is the source of truth, and the cache is rebuilt
from it on boot.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from ..core.logging import get_logger
from .base import session_scope
from .models import Embedding

log = get_logger("db.vectors")


def to_blob(vec: Sequence[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


@dataclass(slots=True)
class Hit:
    ref_id: str
    score: float
    kind: str
    project_id: str | None = None


class _KindIndex:
    """One matrix per kind, so a memory search never scans document chunks."""

    __slots__ = ("ids", "projects", "matrix", "dim")

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.projects: list[str | None] = []
        self.matrix: np.ndarray | None = None
        self.dim: int = 0

    def append(self, ref_id: str, vec: np.ndarray, project_id: str | None) -> None:
        if self.matrix is None or self.dim == 0:
            self.dim = int(vec.shape[0])
            self.matrix = vec.reshape(1, -1).copy()
            self.ids = [ref_id]
            self.projects = [project_id]
            return
        if vec.shape[0] != self.dim:
            # Dimension changed (different embedding model). Drop the cache and
            # let it rebuild -- mixing dimensions silently is worse than a reload.
            log.warning("Embedding dim changed %s -> %s; index will rebuild", self.dim, vec.shape[0])
            self.ids.clear()
            self.projects.clear()
            self.matrix = None
            self.dim = 0
            return self.append(ref_id, vec, project_id)
        try:
            pos = self.ids.index(ref_id)
        except ValueError:
            self.ids.append(ref_id)
            self.projects.append(project_id)
            self.matrix = np.vstack([self.matrix, vec.reshape(1, -1)])
        else:
            self.matrix[pos] = vec
            self.projects[pos] = project_id

    def remove(self, ref_id: str) -> None:
        try:
            pos = self.ids.index(ref_id)
        except ValueError:
            return
        self.ids.pop(pos)
        self.projects.pop(pos)
        if self.matrix is not None:
            self.matrix = np.delete(self.matrix, pos, axis=0)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        project_id: str | None,
        allow: set[str] | None,
        min_score: float,
    ) -> list[tuple[str, float]]:
        if self.matrix is None or not self.ids or query.shape[0] != self.dim:
            return []
        scores = self.matrix @ query
        mask = np.ones(len(self.ids), dtype=bool)
        if project_id is not None:
            mask &= np.array([p == project_id or p is None for p in self.projects])
        if allow is not None:
            mask &= np.array([i in allow for i in self.ids])
        if not mask.any():
            return []
        scores = np.where(mask, scores, -np.inf)
        k = min(top_k, int(mask.sum()))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx if scores[i] >= min_score]


class VectorIndex:
    def __init__(self) -> None:
        self._kinds: dict[str, _KindIndex] = {}
        self._lock = threading.RLock()
        self._loaded = False

    async def load(self, force: bool = False) -> int:
        """Hydrate the in-memory matrices from SQLite."""
        if self._loaded and not force:
            return sum(len(k.ids) for k in self._kinds.values())
        async with session_scope() as s:
            rows = (await s.execute(select(Embedding))).scalars().all()
        with self._lock:
            self._kinds.clear()
            for row in rows:
                vec = from_blob(row.vector)
                if vec.size == 0:
                    continue
                self._kinds.setdefault(row.kind, _KindIndex()).append(row.ref_id, vec, row.project_id)
            self._loaded = True
        total = sum(len(k.ids) for k in self._kinds.values())
        log.info("Vector index loaded: %d vectors across %d kinds", total, len(self._kinds))
        return total

    async def upsert(
        self,
        kind: str,
        ref_id: str,
        vector: Sequence[float],
        *,
        model: str = "",
        project_id: str | None = None,
    ) -> None:
        blob = to_blob(vector)
        arr = from_blob(blob)
        async with session_scope() as s:
            existing = (
                await s.execute(
                    select(Embedding).where(Embedding.kind == kind, Embedding.ref_id == ref_id)
                )
            ).scalar_one_or_none()
            if existing:
                existing.vector = blob
                existing.dim = int(arr.shape[0])
                existing.model = model
                existing.project_id = project_id
            else:
                s.add(
                    Embedding(
                        kind=kind,
                        ref_id=ref_id,
                        vector=blob,
                        dim=int(arr.shape[0]),
                        model=model,
                        project_id=project_id,
                    )
                )
        with self._lock:
            self._kinds.setdefault(kind, _KindIndex()).append(ref_id, arr, project_id)

    async def upsert_many(
        self,
        kind: str,
        items: Iterable[tuple[str, Sequence[float]]],
        *,
        model: str = "",
        project_id: str | None = None,
    ) -> int:
        items = list(items)
        if not items:
            return 0
        ref_ids = [i[0] for i in items]
        async with session_scope() as s:
            await s.execute(
                sa_delete(Embedding).where(Embedding.kind == kind, Embedding.ref_id.in_(ref_ids))
            )
            for ref_id, vec in items:
                blob = to_blob(vec)
                s.add(
                    Embedding(
                        kind=kind,
                        ref_id=ref_id,
                        vector=blob,
                        dim=len(vec),
                        model=model,
                        project_id=project_id,
                    )
                )
        with self._lock:
            idx = self._kinds.setdefault(kind, _KindIndex())
            for ref_id, vec in items:
                idx.append(ref_id, normalize(vec), project_id)
        return len(items)

    async def delete(self, kind: str, ref_id: str) -> None:
        async with session_scope() as s:
            await s.execute(
                sa_delete(Embedding).where(Embedding.kind == kind, Embedding.ref_id == ref_id)
            )
        with self._lock:
            if kind in self._kinds:
                self._kinds[kind].remove(ref_id)

    async def delete_many(self, kind: str, ref_ids: Sequence[str]) -> None:
        if not ref_ids:
            return
        async with session_scope() as s:
            await s.execute(
                sa_delete(Embedding).where(Embedding.kind == kind, Embedding.ref_id.in_(list(ref_ids)))
            )
        with self._lock:
            idx = self._kinds.get(kind)
            if idx:
                for r in ref_ids:
                    idx.remove(r)

    async def search(
        self,
        kind: str,
        query: Sequence[float],
        *,
        top_k: int = 20,
        project_id: str | None = None,
        allow: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        await self.load()
        q = normalize(query)
        with self._lock:
            idx = self._kinds.get(kind)
            if idx is None:
                return []
            raw = idx.search(q, top_k, project_id, allow, min_score)
            proj_by_id = dict(zip(idx.ids, idx.projects))
        return [Hit(ref_id=r, score=s, kind=kind, project_id=proj_by_id.get(r)) for r, s in raw]

    async def search_multi(
        self,
        kinds: Sequence[str],
        query: Sequence[float],
        *,
        top_k: int = 20,
        project_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[Hit]:
        results: list[Hit] = []
        for kind in kinds:
            results.extend(
                await self.search(
                    kind, query, top_k=top_k, project_id=project_id, min_score=min_score
                )
            )
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                kind: {"count": len(k.ids), "dim": k.dim} for kind, k in sorted(self._kinds.items())
            }

    async def rebuild_from(self, kind: str, pairs: Iterable[tuple[str, Sequence[float]]], model: str = "") -> int:
        """Wipe and rewrite one kind. Used after an embedding model change."""
        async with session_scope() as s:
            await s.execute(sa_delete(Embedding).where(Embedding.kind == kind))
        with self._lock:
            self._kinds.pop(kind, None)
        return await self.upsert_many(kind, pairs, model=model)


#: Process-wide index.
index = VectorIndex()


async def warm() -> int:
    try:
        return await index.load()
    except Exception as exc:  # noqa: BLE001 - a cold index must not block boot
        log.warning("Vector index warm-up failed: %s", exc)
        return 0


def mmr(
    query: np.ndarray,
    candidates: list[tuple[str, np.ndarray, float]],
    k: int,
    diversity: float = 0.3,
) -> list[str]:
    """Maximal Marginal Relevance.

    Straight top-k similarity returns five paraphrases of the same fact. MMR
    trades a little relevance for coverage, which is what you actually want
    when stuffing a context window.
    """
    if not candidates:
        return []
    selected: list[str] = []
    selected_vecs: list[np.ndarray] = []
    pool = list(candidates)
    while pool and len(selected) < k:
        best_id, best_vec, best_score = None, None, -np.inf
        for cid, cvec, rel in pool:
            if selected_vecs:
                redundancy = max(float(cvec @ sv) for sv in selected_vecs)
            else:
                redundancy = 0.0
            score = (1 - diversity) * rel - diversity * redundancy
            if score > best_score:
                best_id, best_vec, best_score = cid, cvec, score
        if best_id is None or best_vec is None:
            break
        selected.append(best_id)
        selected_vecs.append(best_vec)
        pool = [c for c in pool if c[0] != best_id]
    return selected


async def gc_orphans() -> int:
    """Drop embeddings whose owning row is gone."""
    from .models import Asset, Chunk, Entity, Memory, Message

    table_for = {
        "memory": Memory,
        "chunk": Chunk,
        "entity": Entity,
        "message": Message,
        "asset": Asset,
    }
    removed = 0
    for kind, model in table_for.items():
        async with session_scope() as s:
            live = set((await s.execute(select(model.id))).scalars().all())
            embedded = (
                await s.execute(select(Embedding.ref_id).where(Embedding.kind == kind))
            ).scalars().all()
            dead = [r for r in embedded if r not in live]
            if dead:
                await s.execute(
                    sa_delete(Embedding).where(Embedding.kind == kind, Embedding.ref_id.in_(dead))
                )
                removed += len(dead)
        if dead:
            with index._lock:  # noqa: SLF001 - same module, intentional
                ki = index._kinds.get(kind)  # noqa: SLF001
                if ki:
                    for r in dead:
                        ki.remove(r)
    if removed:
        log.info("Vector GC removed %d orphaned embeddings", removed)
    return removed


async def periodic_gc(interval_s: int = 3600) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await gc_orphans()
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector GC failed: %s", exc)
