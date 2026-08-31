"""The main agent's introspection and self-improvement tools.

These are how the agent participates in its own improvement loop rather than
merely being subjected to it: it can read its own performance record, trigger
a reflection pass, amend its standing directives, and forge new tools for
itself through the skill lifecycle in :mod:`ciws.tools.skills`.

Everything here is lazy-imported like the delegation tools, so this module
stays importable before the agents package exists in the import graph.
"""

from __future__ import annotations

from typing import Any

from ...core.util import dumps
from ..registry import Risk, ToolOutput, registry


def _agent_slug(ctx: Any) -> str:
    from ...agents.improve import MAIN_SLUG

    return getattr(ctx, "agent_slug", "") or MAIN_SLUG


@registry.tool(
    "self_status",
    category="self",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def self_status(ctx: Any = None) -> ToolOutput:
    """Your own performance record: run outcomes since your last reflection,
    user feedback, tool error rates, your learned directives, and your skills.

    Read this before reflecting or proposing a directive -- improvement that is
    not grounded in this evidence is guessing.
    """
    from ...agents import improve

    status = await improve.improvement_status(_agent_slug(ctx))
    return ToolOutput.json(status)


@registry.tool(
    "self_reflect",
    category="self",
    risk=Risk.WRITE,
    timeout_s=300,
    parameters={"type": "object", "properties": {}},
)
async def self_reflect(ctx: Any = None) -> ToolOutput:
    """Run a reflection pass on your own recent runs, now.

    It distils lessons into memory and may add or retire learned directives,
    within the configured consent policy. This also runs on a schedule; call it
    yourself when something just went wrong enough to be worth learning from
    immediately, or when the user asks you to reflect.
    """
    from ...agents import improve

    report = await improve.reflect(_agent_slug(ctx), force=True)
    return ToolOutput.json(report)


@registry.tool(
    "self_directive",
    category="self",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "retire"]},
            "text": {
                "type": "string",
                "description": "For add: the rule, phrased 'When X, do Y'.",
            },
            "directive_id": {"type": "string", "description": "For retire: the [dir_...] id."},
            "reason": {"type": "string", "description": "The evidence behind this change."},
        },
        "required": ["action"],
    },
)
async def self_directive(
    action: str, text: str = "", directive_id: str = "", reason: str = "", ctx: Any = None
) -> ToolOutput:
    """Amend your own standing directives -- the learned rules in your prompt.

    Adding follows the workspace consent policy: it may land as a proposal for
    the user to approve rather than applying immediately. Retiring is always
    yours to do; a directive you have to argue with is a directive that has
    stopped earning its place. Every change is audited and reversible.
    """
    from ...agents import improve

    slug = _agent_slug(ctx)
    if action == "add":
        if not text.strip():
            return ToolOutput.error("Give the directive text to add.")
        entry = await improve.add_directive(slug, text, reason=reason, origin="agent")
        if entry.get("deduped"):
            return ToolOutput.text("An equivalent directive already exists.")
        note = (
            "It is active from your next run."
            if entry["status"] == "active"
            else "It is PROPOSED and waits for the user's approval in the Improve panel."
        )
        return ToolOutput.text(f"Directive {entry['id']} recorded. {note}")
    if action == "retire":
        if not directive_id:
            return ToolOutput.error("Give the directive_id to retire.")
        hit = await improve.set_directive_status(slug, directive_id, "retired")
        if hit is None:
            return ToolOutput.error(f"No directive {directive_id} on {slug}.")
        return ToolOutput.text(f"Retired {directive_id}: {hit.get('text', '')}")
    return ToolOutput.error("action must be 'add' or 'retire'.")


@registry.tool(
    "skill_forge",
    category="self",
    risk=Risk.DANGEROUS,
    timeout_s=180,
    max_result_chars=30_000,
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short tool name; becomes skill_<slug>."},
            "description": {
                "type": "string",
                "description": "What the tool does and when to reach for it -- written for the model that will call it.",
            },
            "parameters": {
                "type": "object",
                "description": 'JSON schema for the arguments: {"type": "object", "properties": {...}}.',
            },
            "code": {
                "type": "string",
                "description": (
                    "Python defining run(...) at module level, taking the schema's "
                    "properties as keyword arguments and returning a string or a "
                    "JSON-serialisable value. May be async. Standard library only "
                    "unless you know the dependency is installed."
                ),
            },
            "test_code": {
                "type": "string",
                "description": (
                    "A script that exercises run() and raises (assert) on failure. "
                    "It runs in the same namespace as the code; use asyncio.run() "
                    "for an async run()."
                ),
            },
        },
        "required": ["name", "description", "code", "test_code"],
    },
)
async def skill_forge(
    name: str,
    description: str,
    code: str,
    test_code: str,
    parameters: dict[str, Any] | None = None,
    ctx: Any = None,
) -> ToolOutput:
    """Write a new tool for yourself: code plus a test that proves it works.

    Forge a skill when a capability gap keeps costing you steps -- a parser you
    keep re-writing inline, a calculation you keep getting wrong, an API shape
    you keep re-deriving. Do not forge what an existing tool already does.

    The tests run immediately in a subprocess; a skill whose tests fail is not
    stored. A stored skill still does not execute until it is activated --
    by the user in the Improve panel unless they have opted into
    auto-activation. Skills are not sandboxed, which is exactly why.
    """
    from .. import skills

    result = await skills.propose_skill(
        name,
        description,
        parameters or {"type": "object", "properties": {}},
        code,
        test_code,
        agent_slug=_agent_slug(ctx),
        origin_run_id=getattr(ctx, "run_id", None),
    )
    if not result.get("ok"):
        body = result.get("error", "Skill was not stored.")
        if result.get("test_report"):
            body += f"\n\nTest report:\n{dumps(result['test_report'])}"
        return ToolOutput.error(body)
    return ToolOutput.json(result)


@registry.tool(
    "skill_list",
    category="self",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def skill_list(ctx: Any = None) -> ToolOutput:
    """Your forged skills and their states. Only ACTIVE ones are callable."""
    from .. import skills

    rows = await skills.list_skills()
    if not rows:
        return ToolOutput.text("No skills forged yet.")
    lines = []
    for s in rows:
        report = s.last_test_report or {}
        lines.append(
            f"- skill_{s.slug} [{s.status}] v{s.version}: {s.description or s.name} "
            f"(used {s.use_count}x, {s.error_count} errors, "
            f"tests {'passing' if report.get('passed') else 'failing'})"
        )
    return ToolOutput.text("\n".join(lines))


@registry.tool(
    "skill_test",
    category="self",
    risk=Risk.DANGEROUS,
    timeout_s=120,
    parameters={
        "type": "object",
        "properties": {"slug": {"type": "string", "description": "The skill's slug."}},
        "required": ["slug"],
    },
)
async def skill_test(slug: str, ctx: Any = None) -> ToolOutput:
    """Re-run a stored skill's own tests and record the fresh report.

    Use it when a skill has been erroring in use -- a failing report is the
    evidence to revise it with skill_forge, or to disable it.
    """
    from .. import skills

    report = await skills.test_skill(slug)
    return ToolOutput.json(report)
