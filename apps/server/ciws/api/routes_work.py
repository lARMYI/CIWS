"""Conversations, agent runs, workflows and tasks.

Chat is served as SSE rather than over the WebSocket. The WebSocket carries
workspace-wide events -- what every panel needs to stay live -- while a chat
turn is a request with one consumer and a definite end. Keeping them apart
means a dropped socket does not lose a half-written answer.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select

from ..agents import presets, runtime
from ..core.errors import NotFound
from ..core.util import now, truncate
from ..db.base import session_scope
from ..db.models import Conversation, Message, Task
from ..gateway.types import ChatMessage, ContentPart, StreamEventType
from ..workflows import engine as workflows
from .deps import Auth

router = APIRouter(dependencies=[Auth])


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.get("/conversations")
async def list_conversations(
    project_id: str | None = None, limit: int = Query(60, le=300), archived: bool = False
) -> dict[str, Any]:
    async with session_scope() as s:
        stmt = select(Conversation).where(Conversation.archived.is_(archived))
        if project_id:
            stmt = stmt.where(Conversation.project_id == project_id)
        rows = (
            await s.execute(
                stmt.order_by(Conversation.pinned.desc(), Conversation.updated_at.desc()).limit(limit)
            )
        ).scalars().all()
        counts = {
            cid: n
            for cid, n in (
                await s.execute(
                    select(Message.conversation_id, func.count(Message.id)).group_by(
                        Message.conversation_id
                    )
                )
            ).all()
        }
    return {
        "conversations": [
            {**c.to_dict(), "message_count": counts.get(c.id, 0)} for c in rows
        ]
    }


@router.post("/conversations")
async def create_conversation(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    async with session_scope() as s:
        conversation = Conversation(
            title=str(body.get("title") or "New conversation"),
            project_id=body.get("project_id"),
            agent_id=body.get("agent_id"),
            model=str(body.get("model") or ""),
            system_prompt=str(body.get("system_prompt") or ""),
        )
        s.add(conversation)
        await s.flush()
        return conversation.to_dict()


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    async with session_scope() as s:
        conversation = (
            await s.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()
        if conversation is None:
            raise NotFound(f"No conversation {conversation_id}")
        messages = (
            await s.execute(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.seq)
            )
        ).scalars().all()
    return {**conversation.to_dict(), "messages": [m.to_dict() for m in messages]}


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str, patch: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    async with session_scope() as s:
        conversation = (
            await s.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()
        if conversation is None:
            raise NotFound(f"No conversation {conversation_id}")
        for key, value in patch.items():
            if key not in {"id", "created_at"} and hasattr(conversation, key):
                setattr(conversation, key, value)
        await s.flush()
        return conversation.to_dict()


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    async with session_scope() as s:
        result = await s.execute(
            sa_delete(Conversation).where(Conversation.id == conversation_id)
        )
        return {"ok": bool(result.rowcount)}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class Attachment(BaseModel):
    type: str = "image"
    data: str = ""
    url: str = ""
    mime_type: str = "image/png"
    name: str = ""


class ChatBody(BaseModel):
    message: str = ""
    conversation_id: str | None = None
    project_id: str | None = None
    agent: str = presets.MAIN_SLUG
    model: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    #: History is loaded from the DB; this only matters for stateless calls.
    include_history: bool = True
    auto_approve: bool = True


async def _load_history(conversation_id: str, limit: int = 40) -> list[ChatMessage]:
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.seq.desc())
                .limit(limit)
            )
        ).scalars().all()

    history: list[ChatMessage] = []
    for row in reversed(rows):
        if row.role == "user":
            parts = [
                ContentPart(**p) for p in (row.parts or []) if isinstance(p, dict)
            ]
            history.append(ChatMessage(role="user", content=row.content, parts=parts))
        elif row.role == "assistant" and row.content:
            history.append(ChatMessage(role="assistant", content=row.content))
    return history


async def _next_seq(conversation_id: str) -> int:
    async with session_scope() as s:
        top = (
            await s.execute(
                select(func.max(Message.seq)).where(Message.conversation_id == conversation_id)
            )
        ).scalar()
        return int(top or 0) + 1


async def _persist(conversation_id: str, **fields: Any) -> Message:
    async with session_scope() as s:
        message = Message(
            conversation_id=conversation_id, seq=await _next_seq(conversation_id), **fields
        )
        s.add(message)
        await s.flush()
        conversation = (
            await s.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()
        if conversation is not None:
            conversation.updated_at = now()
        return message


@router.post("/chat")
async def chat(body: ChatBody) -> StreamingResponse:
    """Stream an agent turn as Server-Sent Events."""
    conversation_id = body.conversation_id
    if not conversation_id:
        async with session_scope() as s:
            conversation = Conversation(
                title=truncate(body.message or "New conversation", 80, marker="..."),
                project_id=body.project_id,
                model=body.model,
            )
            s.add(conversation)
            await s.flush()
            conversation_id = conversation.id

    parts = [
        ContentPart(
            type="image" if a.type == "image" else "file",
            data=a.data,
            url=a.url,
            mime_type=a.mime_type,
        )
        for a in body.attachments
    ]
    await _persist(
        conversation_id,
        role="user",
        content=body.message,
        parts=[p.model_dump(mode="json") for p in parts],
    )
    history = await _load_history(conversation_id) if body.include_history else []
    # The turn just persisted is already in history; the runtime adds the prompt.
    history = history[:-1] if history else []

    async def event_stream() -> AsyncIterator[bytes]:
        collected: list[str] = []
        thinking: list[str] = []
        run_id = ""
        usage: dict[str, Any] = {}
        error = ""

        def frame(kind: str, **data: Any) -> bytes:
            return f"data: {json.dumps({'type': kind, **data})}\n\n".encode()

        yield frame("conversation", conversation_id=conversation_id)

        try:
            async for event in runtime.stream_agent(
                agent_slug=body.agent,
                prompt=body.message,
                history=history,
                conversation_id=conversation_id,
                project_id=body.project_id,
                model_override=body.model,
                auto_approve=body.auto_approve,
            ):
                if event.type is StreamEventType.START:
                    run_id = event.text
                    yield frame("start", run_id=run_id, model=event.model)
                elif event.type is StreamEventType.TEXT:
                    collected.append(event.text)
                    yield frame("delta", text=event.text)
                elif event.type is StreamEventType.THINKING:
                    thinking.append(event.text)
                    yield frame("thinking", text=event.text)
                elif event.type is StreamEventType.ERROR:
                    error = event.error
                    yield frame("error", error=event.error)
                elif event.type is StreamEventType.DONE:
                    usage = event.usage.model_dump() if event.usage else {}
                    yield frame(
                        "done",
                        run_id=run_id,
                        model=event.model,
                        finish_reason=event.finish_reason,
                        usage=usage,
                        duration_ms=event.index,
                    )
        except Exception as exc:  # noqa: BLE001 - the client must always get an end frame
            error = f"{type(exc).__name__}: {exc}"
            yield frame("error", error=error)

        answer = "".join(collected)
        if answer or error:
            await _persist(
                conversation_id,
                role="assistant",
                content=answer,
                thinking="".join(thinking),
                model=body.model,
                run_id=run_id or None,
                error=error,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                cost_usd=float(usage.get("cost_usd", 0.0)),
            )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@router.post("/chat/cancel/{run_id}")
async def cancel_run(run_id: str) -> dict[str, Any]:
    return {"ok": runtime.cancel(run_id)}


# ---------------------------------------------------------------------------
# Agents and runs
# ---------------------------------------------------------------------------


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    agents = await presets.list_agents(enabled_only=False)
    return {"agents": [a.to_dict() for a in agents]}


@router.post("/agents")
async def create_agent(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    agent = await presets.create_agent(**body)
    return agent.to_dict()


@router.patch("/agents/{slug}")
async def update_agent(slug: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    agent = await presets.update_agent(slug, **patch)
    if agent is None:
        raise NotFound(f"No agent {slug}")
    return agent.to_dict()


@router.delete("/agents/{slug}")
async def delete_agent(slug: str) -> dict[str, Any]:
    ok = await presets.delete_agent(slug)
    if not ok:
        raise HTTPException(400, "Built-in agents cannot be deleted -- disable it instead.")
    return {"ok": True}


@router.get("/runs")
async def list_runs(
    conversation_id: str | None = None, project_id: str | None = None, limit: int = Query(50, le=200)
) -> dict[str, Any]:
    runs = await runtime.list_runs(
        conversation_id=conversation_id, project_id=project_id, limit=limit
    )
    return {"runs": runs, "active": runtime.active_runs()}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = await runtime.get_run(run_id)
    if run is None:
        raise NotFound(f"No run {run_id}")
    return run


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


@router.get("/workflows")
async def list_workflows(project_id: str | None = None) -> dict[str, Any]:
    rows = await workflows.list_workflows(project_id)
    return {
        "workflows": [w.to_dict() for w in rows],
        "node_types": workflows.node_catalog(),
    }


@router.post("/workflows")
async def create_workflow(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    workflow = await workflows.create_workflow(**body)
    return workflow.to_dict()


@router.get("/workflows/examples")
async def workflow_examples() -> dict[str, Any]:
    return {"examples": workflows.example_workflows()}


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict[str, Any]:
    workflow = await workflows.get_workflow(workflow_id)
    if workflow is None:
        raise NotFound(f"No workflow {workflow_id}")
    runs = await workflows.list_runs(workflow_id, limit=20)
    return {
        **workflow.to_dict(),
        "problems": workflows.validate_graph(workflow.graph or {}),
        "runs": [r.to_dict() for r in runs],
    }


@router.patch("/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    workflow = await workflows.update_workflow(workflow_id, **patch)
    if workflow is None:
        raise NotFound(f"No workflow {workflow_id}")
    return {**workflow.to_dict(), "problems": workflows.validate_graph(workflow.graph or {})}


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict[str, Any]:
    return {"ok": await workflows.delete_workflow(workflow_id)}


@router.post("/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str, inputs: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    run = await workflows.run_workflow(workflow_id, inputs)
    return run.to_dict()


@router.post("/workflows/validate")
async def validate_workflow(graph: dict[str, Any] = Body(...)) -> dict[str, Any]:
    problems = workflows.validate_graph(graph)
    return {"ok": not problems, "problems": problems}


@router.get("/workflow-runs/{run_id}")
async def get_workflow_run(run_id: str) -> dict[str, Any]:
    run = await workflows.get_run(run_id)
    if run is None:
        raise NotFound(f"No workflow run {run_id}")
    return run.to_dict()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@router.get("/tasks")
async def list_tasks(
    status: str | None = None, project_id: str | None = None, limit: int = Query(100, le=500)
) -> dict[str, Any]:
    async with session_scope() as s:
        stmt = select(Task)
        if status:
            stmt = stmt.where(Task.status == status)
        if project_id:
            stmt = stmt.where(Task.project_id == project_id)
        rows = (
            await s.execute(stmt.order_by(Task.priority, Task.created_at.desc()).limit(limit))
        ).scalars().all()
    return {"tasks": [t.to_dict() for t in rows]}


@router.post("/tasks")
async def create_task(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    async with session_scope() as s:
        task = Task(**{k: v for k, v in body.items() if hasattr(Task, k)})
        s.add(task)
        await s.flush()
        return task.to_dict()


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    async with session_scope() as s:
        task = (await s.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
        if task is None:
            raise NotFound(f"No task {task_id}")
        for key, value in patch.items():
            if key not in {"id", "created_at"} and hasattr(task, key):
                setattr(task, key, value)
        await s.flush()
        return task.to_dict()


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> dict[str, Any]:
    async with session_scope() as s:
        result = await s.execute(sa_delete(Task).where(Task.id == task_id))
        return {"ok": bool(result.rowcount)}
