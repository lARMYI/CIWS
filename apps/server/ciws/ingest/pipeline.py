"""Document ingestion: file in, searchable corpus out.

The original bytes are always copied into ``CIWS_HOME/corpus`` before anything
is parsed. Extraction is lossy and parsers improve; keeping the source means a
document can be re-chunked or re-embedded later without asking you to find the
file again.

Status moves pending -> extracting -> chunking -> embedding -> ready, published
on the event bus at each step, because ingesting a big PDF takes long enough
that a progress bar is the difference between "working" and "broken".
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, or_, select, text as sql_text

from ..core import paths
from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import new_id, sha256, slugify, truncate
from ..db.base import session_scope
from ..db.models import Chunk, Document
from ..db.vectors import index
from ..memory import embeddings
from . import chunk as chunker
from . import extractors

log = get_logger("ingest")

VECTOR_KIND = "chunk"
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", ".cache", ".idea", ".vscode", "target", ".mypy_cache", ".pytest_cache",
    ".tox", "vendor", "site-packages",
}
MAX_FILE_BYTES = 25 * 1024 * 1024


async def _set_status(doc_id: str, status: str, *, error: str = "", **fields: Any) -> None:
    async with session_scope() as s:
        doc = (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
        if doc is None:
            return
        doc.status = status
        if error:
            doc.error = error[:2000]
        for key, value in fields.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
    bus.publish(
        Topic.INGEST_PROGRESS if status not in ("ready", "failed") else Topic.INGEST_DONE,
        document_id=doc_id,
        status=status,
        error=error,
        **{k: v for k, v in fields.items() if isinstance(v, (int, float, str))},
    )


async def _find_by_hash(digest: str) -> Document | None:
    async with session_scope() as s:
        return (
            await s.execute(
                select(Document).where(
                    Document.content_hash == digest, Document.status == "ready"
                )
            )
        ).scalars().first()


async def _process(
    doc_id: str,
    extracted: extractors.Extracted,
    *,
    project_id: str | None,
    is_code: bool = False,
) -> Document:
    """Chunk, embed and index an already-extracted document."""
    await _set_status(doc_id, "chunking", page_count=extracted.page_count)

    if is_code:
        pieces = chunker.chunk_code(extracted.text)
    else:
        pieces = chunker.chunk_text(extracted.text, page_offsets=extracted.page_offsets)

    if not pieces:
        await _set_status(
            doc_id, "ready", chunk_count=0, text=extracted.text[:200_000],
        )
        doc = await get_document(doc_id)
        assert doc is not None
        return doc

    async with session_scope() as s:
        await s.execute(sa_delete(Chunk).where(Chunk.document_id == doc_id))
        rows = [
            Chunk(
                document_id=doc_id,
                project_id=project_id,
                idx=piece.idx,
                text=piece.text,
                heading=piece.heading[:500],
                page=piece.page,
                token_count=piece.token_count,
                meta=piece.meta,
            )
            for piece in pieces
        ]
        s.add_all(rows)
        await s.flush()
        chunk_ids = [r.id for r in rows]

    await _set_status(doc_id, "embedding", chunk_count=len(pieces))

    # Embedding text carries the heading so a retrieved fragment is anchored to
    # its section, which measurably improves recall on structured documents.
    payloads = [
        f"{piece.heading}\n{piece.text}" if piece.heading else piece.text for piece in pieces
    ]
    vectors, backend = await embeddings.embed_texts(payloads)
    await index.upsert_many(
        VECTOR_KIND, zip(chunk_ids, vectors), model=backend, project_id=project_id
    )

    await _set_status(
        doc_id,
        "ready",
        chunk_count=len(pieces),
        text=extracted.text[:200_000],
        meta=extracted.meta,
    )
    doc = await get_document(doc_id)
    assert doc is not None
    log.info("Ingested %s: %d chunks", doc.title, len(pieces))
    return doc


async def ingest_file(
    path: Path | str,
    *,
    project_id: str | None = None,
    tags: list[str] | None = None,
    title: str = "",
    copy_to_corpus: bool = True,
) -> Document:
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValidationFailed(f"No such file: {path}")

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValidationFailed(
            f"{path.name} is {size / 1e6:.0f}MB, over the {MAX_FILE_BYTES / 1e6:.0f}MB limit."
        )

    digest = sha256(path.read_bytes())
    existing = await _find_by_hash(digest)
    if existing is not None:
        log.info("Already ingested (identical content): %s", existing.title)
        return existing

    stored_path = ""
    if copy_to_corpus:
        target = paths.corpus_dir() / f"{digest[:12]}_{slugify(path.stem, 60)}{path.suffix}"
        if not target.exists():
            shutil.copy2(path, target)
        stored_path = str(target)

    async with session_scope() as s:
        doc = Document(
            title=title or path.stem,
            source_type="file",
            source_uri=str(path),
            stored_path=stored_path,
            size_bytes=size,
            content_hash=digest,
            project_id=project_id,
            tags=tags or [],
            status="extracting",
        )
        s.add(doc)
        await s.flush()
        doc_id = doc.id

    bus.publish(Topic.INGEST_START, document_id=doc_id, title=doc.title, source=str(path))

    try:
        extracted = await extractors.extract(path)
    except Exception as exc:  # noqa: BLE001 - surface the reason on the document
        await _set_status(doc_id, "failed", error=str(exc))
        log.warning("Extraction failed for %s: %s", path.name, exc)
        raise

    async with session_scope() as s:
        row = (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one()
        row.mime_type = extracted.mime_type
        if not title and extracted.title:
            row.title = extracted.title

    if extracted.meta.get("needs_vision"):
        extracted.text = await _caption_image(path) or ""

    return await _process(
        doc_id,
        extracted,
        project_id=project_id,
        is_code=bool(extracted.meta.get("is_code")),
    )


async def _caption_image(path: Path) -> str:
    """Describe an image with a vision model so it becomes searchable."""
    import base64

    try:
        from ..gateway.registry import gateway
        from ..gateway.types import Capability, ChatMessage, ChatRequest, ContentPart

        model = await gateway.pick_by_capability(Capability.VISION)
        if not model:
            return ""
        data = base64.b64encode(path.read_bytes()).decode()
        mime = f"image/{path.suffix.lstrip('.').replace('jpg', 'jpeg')}"
        response = await gateway.chat(
            ChatRequest(
                model=model,
                max_tokens=800,
                messages=[
                    ChatMessage.user(
                        "Describe this image for a searchable archive. Transcribe any visible "
                        "text verbatim, then describe the content, layout and anything notable.",
                        parts=[ContentPart.image_b64(data, mime)],
                    )
                ],
            )
        )
        return response.content
    except Exception as exc:  # noqa: BLE001 - no vision model is fine
        log.debug("Image captioning skipped: %s", exc)
        return ""


async def ingest_text(
    text: str,
    *,
    title: str,
    project_id: str | None = None,
    source_uri: str = "",
    tags: list[str] | None = None,
) -> Document:
    if not text.strip():
        raise ValidationFailed("Nothing to ingest -- the text is empty")

    digest = sha256(text)
    existing = await _find_by_hash(digest)
    if existing is not None:
        return existing

    async with session_scope() as s:
        doc = Document(
            title=title or "Pasted text",
            source_type="paste" if not source_uri else "url",
            source_uri=source_uri,
            size_bytes=len(text.encode("utf-8")),
            content_hash=digest,
            mime_type="text/plain",
            project_id=project_id,
            tags=tags or [],
            status="chunking",
        )
        s.add(doc)
        await s.flush()
        doc_id = doc.id

    bus.publish(Topic.INGEST_START, document_id=doc_id, title=title)
    return await _process(
        doc_id, extractors.Extracted(text=text, title=title), project_id=project_id
    )


async def ingest_url(
    url: str, *, project_id: str | None = None, tags: list[str] | None = None
) -> Document:
    bus.publish(Topic.INGEST_START, url=url, title=url)
    extracted = await extractors.extract_url(url)
    if not extracted.text.strip():
        raise ValidationFailed(f"No readable text found at {url}")
    return await ingest_text(
        extracted.text,
        title=extracted.title or url,
        project_id=project_id,
        source_uri=url,
        tags=tags,
    )


async def ingest_directory(
    path: Path | str,
    *,
    project_id: str | None = None,
    recursive: bool = True,
    patterns: list[str] | None = None,
    max_files: int = 500,
    tags: list[str] | None = None,
) -> list[Document]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValidationFailed(f"{root} is not a directory")

    supported = extractors.supported_extensions()
    candidates: list[Path] = []
    walker = root.rglob("*") if recursive else root.glob("*")
    for candidate in walker:
        if len(candidates) >= max_files:
            break
        if not candidate.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in candidate.relative_to(root).parts[:-1]):
            continue
        if candidate.name.startswith("."):
            continue
        if candidate.suffix.lower() not in supported:
            continue
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        if patterns and not any(candidate.match(p) for p in patterns):
            continue
        candidates.append(candidate)

    log.info("Ingesting %d files from %s", len(candidates), root)
    results: list[Document] = []
    for i, candidate in enumerate(candidates, 1):
        try:
            results.append(
                await ingest_file(candidate, project_id=project_id, tags=tags)
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            log.warning("Skipped %s: %s", candidate.name, exc)
        bus.publish(
            Topic.INGEST_PROGRESS,
            directory=str(root),
            done=i,
            total=len(candidates),
            percent=round(100 * i / max(1, len(candidates))),
        )
        await asyncio.sleep(0)  # let the event loop breathe between files
    return results


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def search_corpus(
    query: str,
    *,
    limit: int = 12,
    project_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid chunk search across the corpus."""
    if not query.strip():
        return []

    from ..memory.store import _fts_query

    vector, _ = await embeddings.embed_one(query)
    hits = await index.search(
        VECTOR_KIND, vector, top_k=limit * 4, project_id=project_id, min_score=-1.0
    )
    scores: dict[str, float] = {h.ref_id: max(0.0, h.score) for h in hits}
    top_vector = max(scores.values(), default=0.0) or 1.0
    for key in scores:
        scores[key] = 0.7 * scores[key] / top_vector

    match = _fts_query(query)
    if match:
        async with session_scope() as s:
            try:
                rows = (
                    await s.execute(
                        sql_text(
                            "SELECT id, bm25(chunks_fts) AS rank FROM chunks_fts "
                            "WHERE chunks_fts MATCH :q ORDER BY rank LIMIT :n"
                        ),
                        {"q": match, "n": limit * 4},
                    )
                ).all()
                best = max((1.0 / (1.0 + abs(float(r[1]))) for r in rows), default=1.0) or 1.0
                for cid, rank in rows:
                    scores[cid] = scores.get(cid, 0.0) + 0.3 * (
                        1.0 / (1.0 + abs(float(rank))) / best
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug("chunk FTS unavailable: %s", exc)

    if not scores:
        return []

    async with session_scope() as s:
        stmt = select(Chunk).where(Chunk.id.in_(scores))
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        chunks = (await s.execute(stmt)).scalars().all()
        if not chunks:
            return []
        doc_ids = {c.document_id for c in chunks}
        docs = {
            d.id: d
            for d in (
                await s.execute(select(Document).where(Document.id.in_(doc_ids)))
            ).scalars().all()
        }

    ranked = sorted(chunks, key=lambda c: scores.get(c.id, 0.0), reverse=True)
    cutoff = max(0.05, scores.get(ranked[0].id, 0.0) * 0.25)
    results = []
    for c in ranked:
        score = scores.get(c.id, 0.0)
        if score < cutoff:
            break
        doc = docs.get(c.document_id)
        results.append(
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "document_title": doc.title if doc else "",
                "source_uri": doc.source_uri if doc else "",
                "text": c.text,
                "heading": c.heading,
                "page": c.page,
                "idx": c.idx,
                "score": round(score, 4),
            }
        )
        if len(results) >= limit:
            break
    return results


async def get_document(doc_id: str) -> Document | None:
    async with session_scope() as s:
        return (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()


async def get_chunks(doc_id: str, *, limit: int = 500, offset: int = 0) -> list[Chunk]:
    async with session_scope() as s:
        return list(
            (
                await s.execute(
                    select(Chunk)
                    .where(Chunk.document_id == doc_id)
                    .order_by(Chunk.idx)
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars().all()
        )


async def list_documents(
    *,
    project_id: str | None = None,
    status: str | None = None,
    query: str = "",
    tags: list[str] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Document]:
    async with session_scope() as s:
        stmt = select(Document)
        if project_id:
            stmt = stmt.where(Document.project_id == project_id)
        if status:
            stmt = stmt.where(Document.status == status)
        if query:
            like = f"%{query}%"
            stmt = stmt.where(or_(Document.title.ilike(like), Document.source_uri.ilike(like)))
        rows = list(
            (
                await s.execute(stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset))
            ).scalars().all()
        )
    if tags:
        wanted = set(tags)
        rows = [r for r in rows if wanted & set(r.tags or [])]
    return rows


async def delete_document(doc_id: str, *, remove_original: bool = True) -> bool:
    async with session_scope() as s:
        doc = (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
        if doc is None:
            return False
        chunk_ids = (
            await s.execute(select(Chunk.id).where(Chunk.document_id == doc_id))
        ).scalars().all()
        stored = doc.stored_path
        await s.execute(sa_delete(Chunk).where(Chunk.document_id == doc_id))
        await s.execute(sa_delete(Document).where(Document.id == doc_id))

    await index.delete_many(VECTOR_KIND, list(chunk_ids))
    if remove_original and stored:
        Path(stored).unlink(missing_ok=True)
    return True


async def reindex(doc_id: str) -> Document:
    """Re-chunk and re-embed from the preserved original."""
    doc = await get_document(doc_id)
    if doc is None:
        raise NotFound(f"No document {doc_id}")

    if doc.stored_path and Path(doc.stored_path).exists():
        extracted = await extractors.extract(Path(doc.stored_path))
    elif doc.text:
        extracted = extractors.Extracted(text=doc.text, title=doc.title)
    else:
        raise ValidationFailed("Nothing to reindex -- no stored original and no cached text")

    old_chunks = [c.id for c in await get_chunks(doc_id)]
    await index.delete_many(VECTOR_KIND, old_chunks)
    return await _process(doc_id, extracted, project_id=doc.project_id)


async def summarize_document(doc_id: str, *, model: str = "fast", force: bool = False) -> str:
    """Map-reduce summary. Falls back to a head excerpt with no model configured."""
    doc = await get_document(doc_id)
    if doc is None:
        raise NotFound(f"No document {doc_id}")
    if doc.summary and not force:
        return doc.summary

    chunks = await get_chunks(doc_id, limit=200)
    if not chunks:
        return doc.text[:800]

    try:
        from ..gateway.registry import gateway

        if len(chunks) <= 8:
            body = "\n\n".join(c.text for c in chunks)
            summary = await gateway.complete(
                f"Summarise this document in 4-6 sentences. Be concrete; name the "
                f"specifics it contains.\n\n---\n{truncate(body, 40_000)}\n---",
                model=model,
                max_tokens=700,
            )
        else:
            partials = []
            for group in range(0, len(chunks), 8):
                body = "\n\n".join(c.text for c in chunks[group : group + 8])
                partials.append(
                    await gateway.complete(
                        f"Summarise this section in 2 sentences:\n\n{truncate(body, 24_000)}",
                        model=model,
                        max_tokens=250,
                    )
                )
            summary = await gateway.complete(
                "These are section summaries of one document. Write a single coherent "
                "5-7 sentence summary of the whole thing.\n\n" + "\n".join(partials),
                model=model,
                max_tokens=800,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("Summarisation unavailable: %s", exc)
        summary = truncate(doc.text or chunks[0].text, 800, marker=" ...")

    async with session_scope() as s:
        row = (await s.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
        if row is not None:
            row.summary = summary
    return summary


async def stats(project_id: str | None = None) -> dict[str, Any]:
    async with session_scope() as s:
        stmt = select(
            func.count(Document.id), func.sum(Document.size_bytes), func.sum(Document.chunk_count)
        )
        if project_id:
            stmt = stmt.where(Document.project_id == project_id)
        docs, size, chunk_total = (await s.execute(stmt)).one()

        by_status = {
            k: int(v)
            for k, v in (
                await s.execute(
                    select(Document.status, func.count(Document.id)).group_by(Document.status)
                )
            ).all()
        }
        by_type = {
            k: int(v)
            for k, v in (
                await s.execute(
                    select(Document.source_type, func.count(Document.id)).group_by(
                        Document.source_type
                    )
                )
            ).all()
        }
    return {
        "documents": int(docs or 0),
        "chunks": int(chunk_total or 0),
        "bytes": int(size or 0),
        "by_status": by_status,
        "by_source": by_type,
    }
