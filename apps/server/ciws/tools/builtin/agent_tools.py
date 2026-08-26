"""Delegation: one agent handing work to another.

The runtime is imported lazily inside each function so this module stays
importable before `ciws.agents` exists in the import graph, and so a circular
import between the runtime and the tool registry is structurally impossible.
"""

from __future__ import annotations

from typing import Any

from ...core.errors import ToolError
from ..registry import Risk, ToolOutput, registry

MAX_DELEGATION_DEPTH = 3


@registry.tool(
    "agent_list",
    category="agents",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def agent_list(ctx: Any = None) -> ToolOutput:
    """Which specialist agents you can delegate to."""
    from ...agents import presets

    agents = await presets.list_agents()
    if not agents:
        return ToolOutput.text("No agents are configured.")
    lines = ["Available agents:"]
    lines += [f"- {a.slug}: {a.description or a.name}" for a in agents if a.enabled]
    return ToolOutput.text("\n".join(lines))


@registry.tool(
    "agent_delegate",
    category="agents",
    risk=Risk.WRITE,
    timeout_s=900,
    max_result_chars=40_000,
    parameters={
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Agent slug from agent_list."},
            "task": {
                "type": "string",
                "description": (
                    "A complete, self-contained brief. The sub-agent cannot see this "
                    "conversation, so restate every fact and constraint it needs."
                ),
            },
            "context": {"type": "string", "description": "Background material to pass along."},
        },
        "required": ["agent", "task"],
    },
)
async def agent_delegate(agent: str, task: str, context: str = "", ctx: Any = None) -> ToolOutput:
    """Hand a self-contained sub-task to a specialist agent and get its result.

    Worth it when the sub-task needs a different toolset, a different model, or
    enough reading that it would crowd out your own context. Not worth it for
    anything you can do in a step or two -- delegation costs a full model run.
    """
    from ...agents import runtime

    depth = int(getattr(ctx, "meta", {}).get("depth", 0)) if ctx else 0
    if depth >= MAX_DELEGATION_DEPTH:
        raise ToolError(
            "agent_delegate",
            f"Delegation depth limit ({MAX_DELEGATION_DEPTH}) reached. Do this one yourself.",
        )

    prompt = f"{task}\n\n## Context\n{context}" if context else task
    result = await runtime.run_agent(
        agent_slug=agent,
        prompt=prompt,
        project_id=getattr(ctx, "project_id", None),
        parent_run_id=getattr(ctx, "run_id", None),
        depth=depth + 1,
        persist_messages=False,
    )
    if result.error:
        return ToolOutput.error(f"{agent} failed: {result.error}")
    return ToolOutput.text(
        f"[{agent}] {result.content}",
        run_id=result.run_id,
        steps=result.steps,
        cost_usd=result.usage.cost_usd,
    )


@registry.tool(
    "think",
    category="agents",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {"thought": {"type": "string", "description": "Your reasoning."}},
        "required": ["thought"],
    },
)
async def think(thought: str, ctx: Any = None) -> ToolOutput:
    """A scratchpad for working something out mid-task.

    Nothing happens -- the value is the pause. Use it to plan before a batch of
    tool calls, or to reconsider when a result contradicts what you expected.
    """
    return ToolOutput.text("Noted.", thought=thought[:2000])
