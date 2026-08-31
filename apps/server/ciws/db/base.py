"""Async SQLite engine, session management, and schema bootstrap.

SQLite is the right call for a single-user hub: no daemon to babysit, the whole
workspace is one file you can copy to a USB stick, and with WAL mode it happily
handles a UI, several agents and a background ingest at once.

The pragmas below matter. Without WAL, a long ingest transaction blocks every
read and the UI appears to hang.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from ..core import paths
from ..core.logging import get_logger
from .models import Base

log = get_logger("db")

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_init_lock = asyncio.Lock()
_ready = False

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=10000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-32000",  # ~32MB page cache
    "PRAGMA mmap_size=268435456",
)

#: Full-text search over the things you actually search for.
FTS_SETUP = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        id UNINDEXED, content, summary, tags, tokenize='porter unicode61'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        id UNINDEXED, document_id UNINDEXED, text, heading, tokenize='porter unicode61'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        id UNINDEXED, conversation_id UNINDEXED, content, tokenize='porter unicode61'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
        id UNINDEXED, name, aliases, description, tokenize='porter unicode61'
    )
    """,
)

#: Triggers keep FTS in lockstep with the base tables without app-level bookkeeping.
FTS_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(id, content, summary, tags)
        VALUES (new.id, new.content, new.summary, new.tags);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        DELETE FROM memories_fts WHERE id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        DELETE FROM memories_fts WHERE id = old.id;
        INSERT INTO memories_fts(id, content, summary, tags)
        VALUES (new.id, new.content, new.summary, new.tags);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(id, document_id, text, heading)
        VALUES (new.id, new.document_id, new.text, new.heading);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(id, conversation_id, content)
        VALUES (new.id, new.conversation_id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
        INSERT INTO entities_fts(id, name, aliases, description)
        VALUES (new.id, new.name, new.aliases, new.description);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
        DELETE FROM entities_fts WHERE id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
        DELETE FROM entities_fts WHERE id = old.id;
        INSERT INTO entities_fts(id, name, aliases, description)
        VALUES (new.id, new.name, new.aliases, new.description);
    END
    """,
)


def get_engine():
    global _engine, _sessionmaker
    if _engine is None:
        url = f"sqlite+aiosqlite:///{paths.db_file()}"
        _engine = create_async_engine(
            url,
            echo=False,
            future=True,
            poolclass=StaticPool if ":memory:" in url else None,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(_engine.sync_engine, "connect")
        def _set_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            for pragma in PRAGMAS:
                try:
                    cur.execute(pragma)
                except Exception:  # noqa: BLE001 - older SQLite builds skip some pragmas
                    pass
            cur.close()

        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def init_db() -> None:
    """Create tables, FTS indexes and triggers. Safe to call repeatedly."""
    global _ready
    async with _init_lock:
        if _ready:
            return
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for stmt in FTS_SETUP + FTS_TRIGGERS:
                try:
                    await conn.execute(text(stmt))
                except Exception as exc:  # noqa: BLE001
                    log.warning("FTS setup skipped: %s", exc)
            await _backfill_fts(conn)
        _ready = True
        log.info("Database ready at %s", paths.db_file())


async def _backfill_fts(conn: Any) -> None:
    """Populate FTS for rows that predate the index (e.g. after an upgrade)."""
    pairs = (
        ("memories_fts", "SELECT id, content, summary, tags FROM memories", "(id, content, summary, tags)"),
        ("entities_fts", "SELECT id, name, aliases, description FROM entities", "(id, name, aliases, description)"),
    )
    for table, select, cols in pairs:
        try:
            count = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar() or 0
            if count:
                continue
            await conn.execute(text(f"INSERT INTO {table} {cols} {select}"))
        except Exception:  # noqa: BLE001
            pass


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session. Commits on success, rolls back on failure."""
    await init_db()
    maker = get_sessionmaker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session


async def close_db() -> None:
    global _engine, _sessionmaker, _ready
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
    _ready = False


async def db_stats() -> dict[str, Any]:
    """Row counts and file size -- shown on the Ops panel."""
    from .models import ALL_TABLES

    out: dict[str, Any] = {}
    async with session_scope() as s:
        for model in ALL_TABLES:
            try:
                res = await s.execute(text(f"SELECT count(*) FROM {model.__tablename__}"))
                out[model.__tablename__] = res.scalar() or 0
            except Exception:  # noqa: BLE001
                out[model.__tablename__] = -1
    f = paths.db_file()
    out["_size_bytes"] = f.stat().st_size if f.exists() else 0
    out["_path"] = str(f)
    return out


async def vacuum() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("VACUUM"))
        await conn.commit()
