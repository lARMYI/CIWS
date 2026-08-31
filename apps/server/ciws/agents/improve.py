"""The main agent's self-improvement loop.

Three loops feed each other here:

* **Learning.** User feedback on runs and the evidence of the runs themselves
  (failures, tool errors, cost) are gathered into an evidence pack.
* **Reflection.** A model reads that evidence and distils it into *lessons*
  (memories of kind ``lesson``, recalled like any other memory) and
  *directives* -- standing behavioural rules appended to the agent's own
  system prompt on every future run.
* **Code enhancement.** When reflection or the agent itself decides a missing
  capability is the problem rather than a missing rule, the skill forge
  (:mod:`ciws.tools.skills`) lets the agent write, test and -- once approved --
  run a new tool. That loop lives in its own module; this one reports on it.

Directives are the part that makes improvement visible and reversible: each is
a short rule with an id, a reason, and a status, stored on the agent row
itself. Prompt-level changes apply automatically by default because they are
bounded and undoable in one click; self-written code does not, because it is
neither.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from ..core.config import get_settings
from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import extract_json, new_id, now, truncate
from ..db.base import session_scope
from ..db.models import AgentDef, AuditEvent, Run, ToolCall
from . import presets

log = get_logger("agents.improve")

#: The agent whose loops run unattended. Others can be reflected on demand.
MAIN_SLUG = presets.MAIN_SLUG

#: New directives per reflection pass. One pass changing half the playbook is
#: not improvement, it is a personality transplant.
MAX_NEW_DIRECTIVES_PER_PASS = 2
MAX_LESSONS_PER_PASS = 6


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


async def record_feedback(run_id: str, score: int, comment: str = "") -> dict[str, Any]:
    """Attach a human verdict to a run. Score is -1, 0 or 1.

    This is the ground truth the reflection pass weights everything else by:
    a run that looked clean in the trace but earned a thumbs-down is exactly
    the kind of failure the agent cannot see from the inside.
    """
    score = max(-1, min(1, int(score)))
    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
        if run is None:
            raise NotFound(f"No run {run_id}")
        meta = dict(run.meta or {})
        meta["feedback"] = {"score": score, "comment": truncate(comment, 1000), "at": now().isoformat()}
        run.meta = meta
        agent_slug = run.agent_slug
    log.info("Feedback %+d recorded on run %s", score, run_id)
    return {"run_id": run_id, "agent_slug": agent_slug, "score": score, "comment": comment}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


async def _runs_since(agent_slug: str, since: datetime | None) -> int:
    async with session_scope() as s:
        stmt = select(func.count(Run.id)).where(
            Run.agent_slug == agent_slug, Run.status != "running"
        )
        if since is not None:
            stmt = stmt.where(Run.started_at > since)
        return int((await s.execute(stmt)).scalar() or 0)


async def gather_evidence(
    agent_slug: str, since: datetime | None = None, *, max_samples: int = 6
) -> dict[str, Any]:
    """What actually happened in this agent's recent runs, as a model can read it."""
    async with session_scope() as s:
        stmt = select(Run).where(Run.agent_slug == agent_slug, Run.status != "running")
        if since is not None:
            stmt = stmt.where(Run.started_at > since)
        runs = (
            await s.execute(stmt.order_by(Run.started_at.desc()).limit(200))
        ).scalars().all()

        run_ids = select(Run.id).where(Run.agent_slug == agent_slug)
        if since is not None:
            run_ids = run_ids.where(Run.started_at > since)
        tool_rows = (
            await s.execute(
                select(ToolCall.tool, ToolCall.ok, func.count(ToolCall.id))
                .where(ToolCall.run_id.in_(run_ids))
                .group_by(ToolCall.tool, ToolCall.ok)
            )
        ).all()
        error_samples = (
            await s.execute(
                select(ToolCall.tool, ToolCall.error)
                .where(ToolCall.run_id.in_(run_ids), ToolCall.ok.is_(False))
                .order_by(ToolCall.created_at.desc())
                .limit(max_samples)
            )
        ).all()

    total = len(runs)
    failed = [r for r in runs if r.status == "failed"]
    cancelled = [r for r in runs if r.status == "cancelled"]
    feedback = [
        {
            "score": (r.meta or {}).get("feedback", {}).get("score"),
            "comment": (r.meta or {}).get("feedback", {}).get("comment", ""),
            "goal": truncate(r.goal, 160),
        }
        for r in runs
        if (r.meta or {}).get("feedback")
    ]

    tools: dict[str, dict[str, int]] = {}
    for name, ok, count in tool_rows:
        entry = tools.setdefault(name, {"calls": 0, "errors": 0})
        entry["calls"] += int(count)
        if not ok:
            entry["errors"] += int(count)

    return {
        "agent": agent_slug,
        "window_start": since.isoformat() if since else None,
        "runs": {
            "total": total,
            "completed": total - len(failed) - len(cancelled),
            "failed": len(failed),
            "cancelled": len(cancelled),
            "avg_steps": round(sum(r.steps for r in runs) / total, 1) if total else 0,
            "avg_duration_ms": int(sum(r.duration_ms for r in runs) / total) if total else 0,
            "total_cost_usd": round(sum(r.cost_usd for r in runs), 4),
        },
        "failure_samples": [
            {"goal": truncate(r.goal, 200), "error": truncate(r.error, 300)}
            for r in failed[:max_samples]
        ],
        "feedback": feedback[: max_samples * 2],
        "tools": tools,
        "tool_error_samples": [
            {"tool": name, "error": truncate(err or "", 240)} for name, err in error_samples
        ],
    }


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


def list_directives(agent: AgentDef) -> list[dict[str, Any]]:
    return [d for d in (agent.meta or {}).get("directives") or [] if isinstance(d, dict)]


def directive_block(agent: AgentDef) -> str:
    """The active directives as a system-prompt block, or "" when there are none."""
    active = [d for d in list_directives(agent) if d.get("status") == "active"]
    if not active:
        return ""
    lines = [
        "## Learned directives",
        "",
        "Rules you adopted by reflecting on your own past runs, oldest first. They",
        "carry the same weight as your persona. If one is wrong or has stopped",
        "earning its place, retire it with the self_directive tool rather than",
        "silently ignoring it.",
        "",
    ]
    lines += [f"- [{d.get('id')}] {d.get('text', '').strip()}" for d in active]
    return "\n".join(lines)


async def add_directive(
    agent_slug: str,
    text: str,
    *,
    reason: str = "",
    origin: str = "agent",
    status: str | None = None,
) -> dict[str, Any]:
    """Append a directive. Defaults to the configured consent policy.

    A directive that would push the active set past ``max_directives`` is
    recorded as ``proposed`` regardless of policy -- growth of the playbook
    past the cap is a decision for the human, not the loop.
    """
    cfg = get_settings().improve
    text = " ".join(text.split()).strip()
    if len(text) < 8:
        raise ValidationFailed("A directive needs to be an actual rule, not a fragment.")

    async with session_scope() as s:
        agent = (
            await s.execute(select(AgentDef).where(AgentDef.slug == agent_slug))
        ).scalar_one_or_none()
        if agent is None:
            raise NotFound(f"No agent '{agent_slug}'")

        meta = dict(agent.meta or {})
        directives = [d for d in meta.get("directives") or [] if isinstance(d, dict)]
        if any(d.get("text") == text and d.get("status") in ("active", "proposed") for d in directives):
            return {"deduped": True, "text": text}

        active_count = sum(1 for d in directives if d.get("status") == "active")
        resolved = status or ("active" if cfg.auto_apply_directives else "proposed")
        if resolved == "active" and active_count >= cfg.max_directives:
            resolved = "proposed"

        entry = {
            "id": new_id("dir"),
            "text": text,
            "reason": truncate(reason, 400),
            "status": resolved,
            "origin": origin,
            "created_at": now().isoformat(),
        }
        meta["directives"] = [*directives, entry]
        agent.meta = meta
        s.add(
            AuditEvent(
                actor=agent_slug,
                action="improve:directive",
                target=truncate(text, 200),
                outcome=resolved,
                detail={"origin": origin, "id": entry["id"]},
            )
        )

    bus.publish(Topic.IMPROVE_DIRECTIVE, agent=agent_slug, **entry)
    return entry


async def set_directive_status(
    agent_slug: str, directive_id: str, status: str
) -> dict[str, Any] | None:
    """Approve, retire or reject a directive. Returns the updated entry."""
    if status not in ("active", "proposed", "retired", "rejected"):
        raise ValidationFailed(f"'{status}' is not a directive status")

    async with session_scope() as s:
        agent = (
            await s.execute(select(AgentDef).where(AgentDef.slug == agent_slug))
        ).scalar_one_or_none()
        if agent is None:
            raise NotFound(f"No agent '{agent_slug}'")

        meta = dict(agent.meta or {})
        directives = [dict(d) for d in meta.get("directives") or [] if isinstance(d, dict)]
        hit: dict[str, Any] | None = None
        for d in directives:
            if d.get("id") == directive_id:
                d["status"] = status
                d["decided_at"] = now().isoformat()
                hit = d
                break
        if hit is None:
            return None
        meta["directives"] = directives
        agent.meta = meta
        s.add(
            AuditEvent(
                actor=agent_slug,
                action="improve:directive",
                target=truncate(hit.get("text", ""), 200),
                outcome=status,
                detail={"id": directive_id},
            )
        )

    bus.publish(Topic.IMPROVE_DIRECTIVE, agent=agent_slug, **hit)
    return hit


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

REFLECT_PROMPT = """You are the self-improvement pass of an agent called "%(name)s".
You are reviewing its recent performance to make it measurably better.

Its current learned directives (rules already in force):
%(directives)s

Evidence from its recent runs -- statuses, failures, tool errors, and direct
human feedback (feedback outweighs everything else):
%(evidence)s

Return a JSON object:
{
  "assessment": "2-3 sentences on how it is actually doing, grounded in the evidence",
  "lessons": [
    {"content": "one durable, self-contained lesson worth recalling in future work",
     "importance": 0.0-1.0, "tags": ["short","lowercase"]}
  ],
  "directives": [
    {"text": "a standing rule, phrased 'When X, do Y' -- concrete enough to follow",
     "reason": "the evidence that motivates it"}
  ],
  "retire": ["ids of existing directives the evidence shows are wrong or stale"]
}

Rules:
- Propose at most %(max_new)d new directives, and only when the evidence clearly
  supports them. An empty list is the right answer for a clean record.
- Never restate the persona or an existing directive.
- Lessons record what happened; directives change what happens next. Do not
  duplicate one as the other.
- Do not invent evidence. Return JSON only."""


async def reflect(
    agent_slug: str = MAIN_SLUG, *, force: bool = False, model: str = "balanced"
) -> dict[str, Any]:
    """One reflection pass: evidence in, lessons and directive changes out.

    Skips itself -- cheaply, before any model call -- unless enough new runs
    have accumulated and the minimum interval has passed. ``force`` is for the
    human (or the agent's own self_reflect tool) asking for a pass now.
    """
    cfg = get_settings().improve
    if not cfg.enabled and not force:
        return {"skipped": "self-improvement is disabled in settings"}

    agent = await presets.get_agent(agent_slug)
    if agent is None:
        return {"skipped": f"no agent '{agent_slug}'"}

    state = dict((agent.meta or {}).get("improve") or {})
    since: datetime | None = None
    if state.get("last_reflect_at"):
        try:
            since = datetime.fromisoformat(state["last_reflect_at"])
        except ValueError:
            since = None

    if not force:
        if since is not None and now() - since < timedelta(minutes=cfg.reflect_min_interval_minutes):
            return {"skipped": "reflected recently"}
        fresh_runs = await _runs_since(agent.slug, since)
        if fresh_runs < cfg.reflect_after_runs:
            return {"skipped": f"only {fresh_runs} new runs", "runs": fresh_runs}

    evidence = await gather_evidence(agent.slug, since)
    if not evidence["runs"]["total"]:
        return {"skipped": "no finished runs to reflect on"}

    directives = list_directives(agent)
    active = [d for d in directives if d.get("status") == "active"]
    rendered_directives = (
        "\n".join(f"- [{d['id']}] {d['text']}" for d in active) or "(none yet)"
    )

    from ..core.util import dumps

    try:
        from ..gateway.registry import gateway

        raw = await gateway.complete(
            REFLECT_PROMPT
            % {
                "name": agent.name,
                "directives": rendered_directives,
                "evidence": truncate(dumps(evidence), 14_000),
                "max_new": MAX_NEW_DIRECTIVES_PER_PASS,
            },
            model=model,
            max_tokens=2500,
            system="You return only valid JSON. No prose, no code fences.",
        )
    except Exception as exc:  # noqa: BLE001 - no model configured is the common case
        log.debug("Reflection skipped: %s", exc)
        return {"skipped": f"no model available ({exc})"}

    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {"skipped": "the model did not return a usable reflection"}

    report: dict[str, Any] = {
        "agent": agent.slug,
        "assessment": truncate(str(parsed.get("assessment") or ""), 800),
        "evidence": evidence["runs"],
        "lessons": 0,
        "directives_added": [],
        "directives_retired": [],
    }

    # Lessons land in ordinary memory, so hybrid recall surfaces them exactly
    # when a similar situation comes round again.
    from ..memory import store

    for item in (parsed.get("lessons") or [])[:MAX_LESSONS_PER_PASS]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if len(content) < 8:
            continue
        try:
            await store.remember(
                content,
                kind="lesson",
                importance=float(item.get("importance") or 0.6),
                tags=sorted({*(str(t) for t in item.get("tags") or []), "self-improvement"}),
                source="reflection",
                source_ref=f"reflect:{agent.slug}",
            )
            report["lessons"] += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("Skipped a lesson: %s", exc)

    for item in (parsed.get("directives") or [])[:MAX_NEW_DIRECTIVES_PER_PASS]:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        entry = await add_directive(
            agent.slug,
            str(item["text"]),
            reason=str(item.get("reason") or ""),
            origin="reflection",
        )
        if not entry.get("deduped"):
            report["directives_added"].append(entry)

    known_ids = {d.get("id") for d in directives}
    for directive_id in parsed.get("retire") or []:
        if directive_id in known_ids:
            hit = await set_directive_status(agent.slug, str(directive_id), "retired")
            if hit:
                report["directives_retired"].append(directive_id)

    async with session_scope() as s:
        row = (
            await s.execute(select(AgentDef).where(AgentDef.slug == agent.slug))
        ).scalar_one_or_none()
        if row is not None:
            meta = dict(row.meta or {})
            meta["improve"] = {
                "last_reflect_at": now().isoformat(),
                "reflections": int(state.get("reflections") or 0) + 1,
                "last_report": {
                    "assessment": report["assessment"],
                    "lessons": report["lessons"],
                    "directives_added": len(report["directives_added"]),
                    "directives_retired": len(report["directives_retired"]),
                    "runs": evidence["runs"],
                },
            }
            row.meta = meta
        s.add(
            AuditEvent(
                actor=agent.slug,
                action="improve:reflect",
                target=f"{evidence['runs']['total']} runs",
                outcome="ok",
                detail={
                    "lessons": report["lessons"],
                    "directives_added": len(report["directives_added"]),
                    "directives_retired": len(report["directives_retired"]),
                },
            )
        )

    bus.publish(
        Topic.IMPROVE_REFLECT,
        agent=agent.slug,
        assessment=report["assessment"],
        lessons=report["lessons"],
        directives_added=len(report["directives_added"]),
        directives_retired=len(report["directives_retired"]),
    )
    log.info(
        "Reflection on %s: %d lessons, %d directives added, %d retired",
        agent.slug,
        report["lessons"],
        len(report["directives_added"]),
        len(report["directives_retired"]),
    )
    return report


async def maybe_reflect() -> dict[str, Any] | None:
    """The scheduler's entry point: reflect on the main agent if it is due."""
    if not get_settings().improve.enabled:
        return None
    report = await reflect(MAIN_SLUG)
    return None if report.get("skipped") else report


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


async def improvement_status(agent_slug: str = MAIN_SLUG) -> dict[str, Any]:
    """Everything the improvement loop knows about itself, for the UI and the agent."""
    agent = await presets.get_agent(agent_slug)
    if agent is None:
        raise NotFound(f"No agent '{agent_slug}'")

    state = dict((agent.meta or {}).get("improve") or {})
    since: datetime | None = None
    if state.get("last_reflect_at"):
        try:
            since = datetime.fromisoformat(state["last_reflect_at"])
        except ValueError:
            since = None

    evidence = await gather_evidence(agent.slug, since)

    skills_summary: list[dict[str, Any]] = []
    try:
        from ..tools import skills

        skills_summary = [
            {
                "slug": s.slug,
                "name": s.name,
                "status": s.status,
                "version": s.version,
                "use_count": s.use_count,
                "error_count": s.error_count,
            }
            for s in await skills.list_skills()
        ]
    except Exception as exc:  # noqa: BLE001 - the forge is optional at this layer
        log.debug("Skill summary unavailable: %s", exc)

    cfg = get_settings().improve
    return {
        "agent": {"slug": agent.slug, "name": agent.name},
        "state": state,
        "directives": list_directives(agent),
        "since_last_reflection": evidence,
        "skills": skills_summary,
        "config": {
            "enabled": cfg.enabled,
            "reflect_after_runs": cfg.reflect_after_runs,
            "reflect_min_interval_minutes": cfg.reflect_min_interval_minutes,
            "max_directives": cfg.max_directives,
            "auto_apply_directives": cfg.auto_apply_directives,
            "auto_activate_skills": cfg.auto_activate_skills,
            "max_skills": cfg.max_skills,
        },
    }
