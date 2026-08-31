"""Tools for reading and writing long-term memory.

Descriptions here are written for the model, not for a developer. The
difference matters: a tool called ``memory_write`` with the description "writes
a memory" gets used constantly and badly. Telling the model *when not to* is
what keeps the memory store from filling with restated conversation.
"""

from __future__ import annotations

from typing import Any

from ...memory import store
from ..registry import Risk, ToolOutput, registry

MEMORY_KINDS = ["fact", "preference", "identity", "decision", "procedure", "event", "insight", "task"]


@registry.tool(
    "memory_search",
    category="memory",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you want to recall. Natural language works best.",
            },
            "limit": {"type": "integer", "description": "Max results (default 8).", "default": 8},
            "kinds": {
                "type": "array",
                "items": {"type": "string", "enum": MEMORY_KINDS},
                "description": "Optionally restrict to these kinds of memory.",
            },
        },
        "required": ["query"],
    },
)
async def memory_search(query: str, limit: int = 8, kinds: list[str] | None = None, ctx: Any = None) -> ToolOutput:
    """Search everything you have remembered about this user and their work.

    Use this before answering anything that depends on the user's context,
    history, preferences or past decisions -- and whenever they refer to
    something as though you should already know it.
    """
    hits = await store.recall(
        query,
        limit=max(1, min(limit, 40)),
        project_id=getattr(ctx, "project_id", None),
        kinds=kinds,
    )
    if not hits:
        return ToolOutput.text(f"No memories matched '{query}'.")

    lines = [f"{len(hits)} memories for '{query}':", ""]
    for hit in hits:
        memory = hit.memory
        flags = " ".join(filter(None, ["[pinned]" if memory.pinned else "", f"({memory.kind})"]))
        lines.append(f"- {memory.content} {flags}")
        lines.append(f"  id={memory.id} importance={memory.importance:.2f} match={hit.score:.3f}")
    return ToolOutput.text("\n".join(lines), count=len(hits))


@registry.tool(
    "memory_write",
    category="memory",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "One self-contained sentence. It must still make sense read months "
                    "later with no other context, so name people and things in full."
                ),
            },
            "kind": {"type": "string", "enum": MEMORY_KINDS, "default": "fact"},
            "importance": {
                "type": "number",
                "description": "0-1. Above 0.8 only for things that change how you work with them.",
                "default": 0.5,
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
)
async def memory_write(
    content: str,
    kind: str = "fact",
    importance: float = 0.5,
    tags: list[str] | None = None,
    ctx: Any = None,
) -> ToolOutput:
    """Remember something durable.

    Store only what will still matter later: stated preferences, decisions and
    their reasons, identity and relationship facts, commitments, hard-won
    procedures. Do NOT store the current question, your own reasoning, or
    anything true only for this conversation. Duplicates are merged
    automatically, so re-stating a known fact is harmless but pointless.
    """
    memory = await store.remember(
        content,
        kind=kind,
        importance=importance,
        tags=tags or [],
        project_id=getattr(ctx, "project_id", None),
        source="agent",
        source_ref=getattr(ctx, "conversation_id", None),
    )
    return ToolOutput.text(f"Remembered [{memory.kind}] {memory.id}: {memory.content}", id=memory.id)


@registry.tool(
    "memory_update",
    category="memory",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {"type": "string"},
            "content": {"type": "string", "description": "Replacement text."},
            "importance": {"type": "number"},
            "pinned": {"type": "boolean", "description": "Pin to keep it from ever fading."},
            "archived": {"type": "boolean", "description": "Archive instead of deleting."},
        },
        "required": ["memory_id"],
    },
)
async def memory_update(memory_id: str, ctx: Any = None, **fields: Any) -> ToolOutput:
    """Correct or re-weight an existing memory.

    Prefer this over writing a second memory when a fact has changed -- two
    contradictory memories are worse than one corrected one.
    """
    clean = {k: v for k, v in fields.items() if v is not None}
    memory = await store.update_memory(memory_id, **clean)
    if memory is None:
        return ToolOutput.error(f"No memory {memory_id}")
    return ToolOutput.text(f"Updated {memory_id}: {memory.content}")


@registry.tool(
    "memory_forget",
    category="memory",
    risk=Risk.DANGEROUS,
    parameters={
        "type": "object",
        "properties": {"memory_id": {"type": "string"}},
        "required": ["memory_id"],
    },
)
async def memory_forget(memory_id: str, ctx: Any = None) -> ToolOutput:
    """Permanently delete a memory. Requires the user's approval.

    Use only when the user asks you to forget something, or when a memory is
    plainly wrong. To de-emphasise something instead, archive it with
    memory_update.
    """
    memory = await store.get(memory_id)
    if memory is None:
        return ToolOutput.error(f"No memory {memory_id}")
    content = memory.content
    await store.forget(memory_id)
    return ToolOutput.text(f"Forgot {memory_id}: {content}")


@registry.tool(
    "memory_stats",
    category="memory",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def memory_stats(ctx: Any = None) -> ToolOutput:
    """How much you remember, broken down by kind."""
    return ToolOutput.json(await store.stats(getattr(ctx, "project_id", None)))
