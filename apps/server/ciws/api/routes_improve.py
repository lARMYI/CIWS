"""The self-improvement surface.

Status for the Improve view, feedback from the conversation, a manual
reflection trigger, and the two approval queues -- directives (prompt rules)
and skills (self-authored code). The queues exist because the consent policy
routes anything the loop is not trusted to apply on its own here, where a
human clicks.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from ..agents import improve
from ..core.errors import NotFound
from ..tools import skills
from .deps import Auth

router = APIRouter(dependencies=[Auth])


@router.get("/improve/status")
async def improvement_status(agent: str = improve.MAIN_SLUG) -> dict[str, Any]:
    return await improve.improvement_status(agent)


@router.post("/improve/reflect")
async def trigger_reflection(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """A human asking for a pass now -- so it bypasses the due-ness checks."""
    return await improve.reflect(str(body.get("agent") or improve.MAIN_SLUG), force=True)


class FeedbackBody(BaseModel):
    run_id: str
    score: int  # -1, 0 or 1
    comment: str = ""


@router.post("/improve/feedback")
async def submit_feedback(body: FeedbackBody) -> dict[str, Any]:
    return await improve.record_feedback(body.run_id, body.score, body.comment)


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------

#: What a human may do to a directive, and the status each verb lands on.
_DIRECTIVE_ACTIONS = {"approve": "active", "retire": "retired", "reject": "rejected"}


@router.post("/improve/directives/{directive_id}")
async def decide_directive(
    directive_id: str, body: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    action = str(body.get("action") or "")
    if action not in _DIRECTIVE_ACTIONS:
        raise HTTPException(400, f"action must be one of {sorted(_DIRECTIVE_ACTIONS)}")
    agent = str(body.get("agent") or improve.MAIN_SLUG)
    hit = await improve.set_directive_status(agent, directive_id, _DIRECTIVE_ACTIONS[action])
    if hit is None:
        raise NotFound(f"No directive {directive_id} on {agent}")
    return hit


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


@router.get("/improve/skills")
async def list_skills(status: str | None = None) -> dict[str, Any]:
    rows = await skills.list_skills(status=status)
    return {"skills": [s.to_dict() for s in rows]}


@router.get("/improve/skills/{slug}")
async def get_skill(slug: str) -> dict[str, Any]:
    skill = await skills.get_skill(slug)
    if skill is None:
        raise NotFound(f"No skill '{slug}'")
    return skill.to_dict()


_SKILL_ACTIONS = {"approve": "active", "disable": "disabled", "reject": "rejected"}


@router.post("/improve/skills/{slug}")
async def decide_skill(slug: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = str(body.get("action") or "")
    if action == "test":
        return {"ok": True, "test_report": await skills.test_skill(slug)}
    if action not in _SKILL_ACTIONS:
        raise HTTPException(400, f"action must be 'test' or one of {sorted(_SKILL_ACTIONS)}")
    return await skills.set_skill_status(slug, _SKILL_ACTIONS[action], actor="user")
