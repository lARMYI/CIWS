"""Background jobs: due tasks, watched folders, and the reflection loop.

Two things in this build were configurable but inert. The ``tasks`` table had a
``due_at`` and an ``assignee`` and nothing ever came round to run them, and a
``folder`` hub could be created and started but never actually noticed a file.
Both are what turns CIWS from something you drive into something that also
works while you are not looking.

Design notes worth keeping:

* **One loop, not a thread per job.** Everything here is I/O bound and already
  async; a second thread would only add a way for the two to disagree about the
  database.
* **A failing job never stops the scheduler.** Each tick catches per job, so a
  bad watched path cannot take scheduled tasks down with it.
* **Ticks do not overlap.** A slow agent run holds the tick; the next one waits
  rather than starting a second copy of the same task.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .core.events import Topic, bus
from .core.logging import get_logger
from .core.util import now
from .db.base import session_scope
from .db.models import Hub, Task

log = get_logger("scheduler")

#: How often the loop wakes. Slow on purpose: these are minute-scale jobs, and
#: a tight poll on a laptop is a battery cost the user did not ask for.
TICK_SECONDS = 30.0

#: Files ingested per folder tick. A folder pointed at a large tree should fill
#: in over several ticks rather than block the loop on its first one.
MAX_FILES_PER_TICK = 20

_task: asyncio.Task[None] | None = None
_stop = asyncio.Event()


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------


async def due_tasks(limit: int = 5) -> list[Task]:
    """Open tasks that are due and assigned to an agent rather than a person."""
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(Task)
                .where(Task.status == "open")
                .where(Task.due_at.is_not(None))
                .where(Task.due_at <= now())
                .where(Task.assignee != "")
                .where(Task.assignee != "me")
                .order_by(Task.priority, Task.due_at)
                .limit(limit)
            )
        ).scalars().all()
        # Detached copies: the session closes before the caller runs anything.
        for row in rows:
            s.expunge(row)
        return list(rows)


async def run_due_tasks() -> int:
    """Hand every due task to its agent. Returns how many were started."""
    from .agents import presets, runtime

    started = 0
    for task in await due_tasks():
        agent = await presets.get_agent(task.assignee)
        if agent is None:
            log.warning(
                "Task %s is assigned to '%s', which is not an agent -- leaving it open",
                task.id,
                task.assignee,
            )
            continue

        # Claim it before running. A long run must not be picked up twice by the
        # next tick, and a crash mid-run should leave a visible "running", not a
        # task that quietly repeats forever.
        async with session_scope() as s:
            row = (await s.execute(select(Task).where(Task.id == task.id))).scalar_one_or_none()
            if row is None or row.status != "open":
                continue
            row.status = "running"

        prompt = task.title if not task.detail else f"{task.title}\n\n{task.detail}"
        bus.publish(Topic.SYSTEM_NOTICE, level="INFO", message=f"Running scheduled task: {task.title}")

        try:
            result = await runtime.run_agent(
                agent_slug=task.assignee, prompt=prompt, project_id=task.project_id
            )
            status, run_id, error = ("done" if result.ok else "failed"), result.run_id, result.error
        except Exception as exc:  # noqa: BLE001 - a bad task must not stop the loop
            log.exception("Scheduled task %s failed", task.id)
            status, run_id, error = "failed", None, f"{type(exc).__name__}: {exc}"

        async with session_scope() as s:
            row = (await s.execute(select(Task).where(Task.id == task.id))).scalar_one_or_none()
            if row is not None:
                row.status = status
                row.run_id = run_id
                meta = dict(row.meta or {})
                if error:
                    meta["error"] = error
                meta["last_run_at"] = now().isoformat()
                row.meta = meta

                # A repeating task schedules its next occurrence rather than
                # ending; otherwise "every morning" means "once".
                every_hours = float(meta.get("repeat_every_hours") or 0)
                if every_hours > 0 and status == "done":
                    row.status = "open"
                    row.due_at = now() + timedelta(hours=every_hours)

        started += 1

    return started


# ---------------------------------------------------------------------------
# Watched folders
# ---------------------------------------------------------------------------


async def _folder_hubs() -> list[Hub]:
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(Hub).where(Hub.kind == "folder").where(Hub.enabled.is_(True))
            )
        ).scalars().all()
        for row in rows:
            s.expunge(row)
        return list(rows)


async def scan_folder(hub: Hub) -> int:
    """Ingest anything in the watched folder that is new or has changed.

    Ingestion already deduplicates on content hash, so re-offering an unchanged
    file is cheap and idempotent -- which means this can stay a simple scan
    rather than a stateful diff that can get out of step with reality.
    """
    from .ingest import extractors, pipeline

    config = hub.config or {}
    root = Path(str(config.get("path") or "")).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Watched path is not a directory: {root}")

    recursive = bool(config.get("recursive", True))
    # supported_extensions() returns dotted forms (".md"); normalise both
    # sides so the comparison cannot silently match nothing.
    supported = {e.lower().lstrip(".") for e in extractors.supported_extensions()}

    seen = dict((hub.meta or {}).get("seen") or {})
    ingested = 0
    changed: dict[str, float] = {}

    walker = root.rglob("*") if recursive else root.glob("*")
    for path in walker:
        if ingested >= MAX_FILES_PER_TICK:
            break
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower().lstrip(".") not in supported:
            continue

        key = str(path)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        if seen.get(key) == stamp:
            continue

        try:
            await pipeline.ingest_file(path, project_id=config.get("project_id"))
            changed[key] = stamp
            ingested += 1
        except Exception as exc:  # noqa: BLE001 - one unreadable file is not a failure
            log.debug("Watched folder skipped %s: %s", path, exc)
            changed[key] = stamp  # do not retry it every tick

    if changed:
        async with session_scope() as s:
            row = (await s.execute(select(Hub).where(Hub.id == hub.id))).scalar_one_or_none()
            if row is not None:
                meta = dict(row.meta or {})
                meta["seen"] = {**seen, **changed}
                meta["last_scan_at"] = now().isoformat()
                row.meta = meta
                row.status = "ready"

    return ingested


async def scan_folders() -> int:
    total = 0
    for hub in await _folder_hubs():
        try:
            found = await scan_folder(hub)
            if found:
                log.info("Watched folder %s ingested %d file(s)", hub.slug, found)
            total += found
        except Exception as exc:  # noqa: BLE001
            log.debug("Folder hub %s scan failed: %s", hub.slug, exc)
            async with session_scope() as s:
                row = (await s.execute(select(Hub).where(Hub.id == hub.id))).scalar_one_or_none()
                if row is not None:
                    row.status = "error"
                    row.status_detail = str(exc)[:400]
    return total


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def reflect_if_due() -> Any:
    """The main agent's reflection pass, when enough new runs have accumulated.

    The due-ness checks live in :func:`ciws.agents.improve.maybe_reflect` and
    are a cheap count query, so putting this on the 30-second tick costs
    nothing until there is actually something to reflect on.
    """
    from .agents import improve

    report = await improve.maybe_reflect()
    if report is not None:
        bus.publish(
            Topic.SYSTEM_NOTICE,
            level="INFO",
            message=f"Reflection pass: {report.get('assessment') or 'completed'}",
        )
    return report


async def tick() -> dict[str, Any]:
    """One pass of every job. Exposed so a test does not have to wait 30s."""
    report: dict[str, Any] = {}
    for name, job in (
        ("tasks", run_due_tasks),
        ("folders", scan_folders),
        ("improve", reflect_if_due),
    ):
        try:
            report[name] = await job()
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the rest
            log.debug("Scheduler job %s failed: %s", name, exc)
            report[name] = f"failed: {exc}"
    return report


async def _loop() -> None:
    while not _stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_stop.wait(), timeout=TICK_SECONDS)
        if _stop.is_set():
            return
        await tick()


def start() -> bool:
    """Start the background loop. Idempotent."""
    global _task
    if _task is not None and not _task.done():
        return False
    _stop.clear()
    _task = asyncio.create_task(_loop())
    return True


async def stop() -> None:
    global _task
    _stop.set()
    if _task is not None:
        _task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _task
        _task = None


def running() -> bool:
    return _task is not None and not _task.done()
