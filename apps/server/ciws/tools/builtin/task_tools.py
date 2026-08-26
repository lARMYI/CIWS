"""Task board and clock.

The time tool exists because models are confidently wrong about the current
date, and a workspace full of "due next Tuesday" needs an anchor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from ...core.util import now
from ...db.base import session_scope
from ...db.models import Task
from ..registry import Risk, ToolOutput, registry

STATUSES = ["open", "in_progress", "blocked", "done", "cancelled"]


@registry.tool(
    "current_time",
    category="utility",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def current_time(ctx: Any = None) -> ToolOutput:
    """The current date and time. Check this before reasoning about anything dated."""
    stamp = now()
    local = datetime.now().astimezone()
    return ToolOutput.json(
        {
            "utc": stamp.isoformat(),
            "local": local.isoformat(),
            "timezone": str(local.tzinfo),
            "weekday": stamp.strftime("%A"),
            "date": stamp.strftime("%Y-%m-%d"),
        }
    )


@registry.tool(
    "task_add",
    category="tasks",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "detail": {"type": "string"},
            "priority": {"type": "integer", "description": "0 highest, 3 lowest.", "default": 2},
            "assignee": {"type": "string", "description": "'me' or an agent slug."},
            "due_in_days": {"type": "number", "description": "Relative due date."},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title"],
    },
)
async def task_add(
    title: str,
    detail: str = "",
    priority: int = 2,
    assignee: str = "",
    due_in_days: float | None = None,
    tags: list[str] | None = None,
    ctx: Any = None,
) -> ToolOutput:
    """Put something on the command board.

    Use when the user commits to work, or when you identify a follow-up that
    should outlive this conversation.
    """
    due = now() + timedelta(days=due_in_days) if due_in_days is not None else None
    async with session_scope() as s:
        task = Task(
            title=title,
            detail=detail,
            priority=max(0, min(int(priority), 3)),
            assignee=assignee,
            due_at=due,
            tags=tags or [],
            project_id=getattr(ctx, "project_id", None),
            run_id=getattr(ctx, "run_id", None),
        )
        s.add(task)
        await s.flush()
        task_id = task.id
    return ToolOutput.text(f"Task {task_id}: {title}", id=task_id)


@registry.tool(
    "task_list",
    category="tasks",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": STATUSES},
            "limit": {"type": "integer", "default": 40},
        },
    },
)
async def task_list(status: str = "", limit: int = 40, ctx: Any = None) -> ToolOutput:
    """What is on the board."""
    async with session_scope() as s:
        stmt = select(Task)
        if status:
            stmt = stmt.where(Task.status == status)
        else:
            stmt = stmt.where(Task.status.notin_(["done", "cancelled"]))
        if getattr(ctx, "project_id", None):
            stmt = stmt.where(Task.project_id == ctx.project_id)
        rows = (
            await s.execute(stmt.order_by(Task.priority, Task.created_at.desc()).limit(limit))
        ).scalars().all()

    if not rows:
        return ToolOutput.text("Nothing on the board.")
    lines = [f"{len(rows)} tasks:"]
    for t in rows:
        due = f" due {t.due_at.date()}" if t.due_at else ""
        who = f" @{t.assignee}" if t.assignee else ""
        lines.append(f"- [P{t.priority}] {t.title} ({t.status}){who}{due}  id={t.id}")
    return ToolOutput.text("\n".join(lines))


@registry.tool(
    "task_update",
    category="tasks",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status": {"type": "string", "enum": STATUSES},
            "title": {"type": "string"},
            "detail": {"type": "string"},
            "priority": {"type": "integer"},
        },
        "required": ["task_id"],
    },
)
async def task_update(task_id: str, ctx: Any = None, **fields: Any) -> ToolOutput:
    """Change a task's status or details."""
    async with session_scope() as s:
        task = (await s.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
        if task is None:
            return ToolOutput.error(f"No task {task_id}")
        for key, value in fields.items():
            if value is not None and hasattr(task, key) and key not in {"id", "created_at"}:
                setattr(task, key, value)
        await s.flush()
        return ToolOutput.text(f"Task {task_id} -> {task.status}: {task.title}")
