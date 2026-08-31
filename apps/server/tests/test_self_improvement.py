"""The main agent and its three improvement loops.

No API key, no network. The reflection model is faked at the gateway seam, and
skills genuinely execute -- in a subprocess, exactly as they do in production --
so what is asserted here is the part that matters: evidence in, lessons and
directives out, and self-written code that cannot run without consent.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from ciws.agents import improve, presets, runtime
from ciws.core.config import get_settings
from ciws.db.base import session_scope
from ciws.db.models import Memory, Run, Skill
from ciws.gateway.registry import gateway
from ciws.tools import skills
from ciws.tools.registry import ToolContext, registry


@pytest.fixture(autouse=True)
def tools_loaded():
    from ciws.tools.registry import load_builtin_tools

    load_builtin_tools()


async def _seed_runs(agent_slug: str, completed: int = 2, failed: int = 1) -> list[str]:
    rows = [
        Run(agent_slug=agent_slug, goal=f"task {i}", status="completed", steps=3)
        for i in range(completed)
    ] + [
        Run(
            agent_slug=agent_slug,
            goal=f"doomed task {i}",
            status="failed",
            error="web_fetch timed out",
            steps=8,
        )
        for i in range(failed)
    ]
    async with session_scope() as s:
        s.add_all(rows)
        await s.flush()
        return [r.id for r in rows]


def _fake_reflection(monkeypatch, payload: dict) -> list[str]:
    """Replace the gateway's one-shot completion with a canned reflection."""
    prompts: list[str] = []

    async def complete(prompt: str, **kw) -> str:
        prompts.append(prompt)
        return json.dumps(payload)

    monkeypatch.setattr(gateway, "complete", complete)
    return prompts


# ---------------------------------------------------------------------------
# The main agent
# ---------------------------------------------------------------------------


async def test_main_agent_is_seeded_and_is_the_fallback():
    await presets.seed_presets()
    main = await presets.get_agent(presets.MAIN_SLUG)
    assert main is not None
    assert main.builtin
    assert main.tools == ["*"]

    # An unknown slug lands on the main agent, not on a specialist.
    agent = await runtime._load_agent("no-such-agent")  # noqa: SLF001
    assert agent.slug == presets.MAIN_SLUG


async def test_main_agent_sees_the_self_improvement_tools():
    await presets.seed_presets()
    main = await presets.get_agent(presets.MAIN_SLUG)
    granted = {spec.name for spec in registry.specs(main.tools)}
    for name in ("self_status", "self_reflect", "self_directive", "skill_forge", "skill_list"):
        assert name in granted, f"main agent is missing {name}"

    # Specialists with narrow grants must not have inherited them.
    scout = await presets.get_agent("scout")
    scout_tools = {spec.name for spec in registry.specs(scout.tools)}
    assert "skill_forge" not in scout_tools
    assert "self_directive" not in scout_tools


# ---------------------------------------------------------------------------
# Learning: feedback
# ---------------------------------------------------------------------------


async def test_feedback_lands_on_the_run_and_in_the_evidence(api):
    await presets.seed_presets()
    run_ids = await _seed_runs(presets.MAIN_SLUG)

    r = await api.post(
        "/api/improve/feedback",
        json={"run_id": run_ids[0], "score": -1, "comment": "wrong file entirely"},
    )
    assert r.status_code == 200, r.text

    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_ids[0]))).scalar_one()
        assert run.meta["feedback"]["score"] == -1

    evidence = await improve.gather_evidence(presets.MAIN_SLUG)
    assert evidence["runs"]["total"] == 3
    assert evidence["runs"]["failed"] == 1
    assert any("wrong file" in f["comment"] for f in evidence["feedback"])


async def test_feedback_on_a_missing_run_is_a_404(api):
    r = await api.post("/api/improve/feedback", json={"run_id": "run_nope", "score": 1})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Reflection: lessons and directives
# ---------------------------------------------------------------------------


async def test_reflection_writes_lessons_and_applies_directives(monkeypatch):
    await presets.seed_presets()
    await _seed_runs(presets.MAIN_SLUG)
    prompts = _fake_reflection(
        monkeypatch,
        {
            "assessment": "Solid on research, keeps timing out on fetches.",
            "lessons": [
                {"content": "Fetching example.com times out; search snippets suffice there.",
                 "importance": 0.7, "tags": ["web"]}
            ],
            "directives": [
                {"text": "When a web fetch times out, fall back to search snippets instead of retrying.",
                 "reason": "one in three runs failed on fetch timeouts"}
            ],
            "retire": [],
        },
    )

    report = await improve.reflect(presets.MAIN_SLUG, force=True)
    assert report.get("skipped") is None, report
    assert report["lessons"] == 1
    assert len(report["directives_added"]) == 1
    # The evidence actually reached the model.
    assert "doomed task" in prompts[0]

    async with session_scope() as s:
        lesson = (
            await s.execute(select(Memory).where(Memory.kind == "lesson"))
        ).scalars().first()
        assert lesson is not None
        assert "self-improvement" in lesson.tags

    # auto_apply_directives defaults on, so the rule binds from the next run.
    main = await presets.get_agent(presets.MAIN_SLUG)
    directives = improve.list_directives(main)
    assert directives[0]["status"] == "active"
    system, _ = await runtime._build_system(main, "hello", None, "")  # noqa: SLF001
    assert "fall back to search snippets" in system
    assert "## Learned directives" in system


async def test_reflection_skips_when_nothing_is_due(monkeypatch):
    await presets.seed_presets()
    _fake_reflection(monkeypatch, {"assessment": "should never be called"})
    # No runs at all: both the scheduled path and a non-forced call stand down.
    assert await improve.maybe_reflect() is None
    report = await improve.reflect(presets.MAIN_SLUG)
    assert "skipped" in report


async def test_proposed_directives_wait_for_approval(api, monkeypatch):
    await presets.seed_presets()
    monkeypatch.setattr(get_settings().improve, "auto_apply_directives", False)

    entry = await improve.add_directive(
        presets.MAIN_SLUG, "When asked for code, always run it before answering.",
        reason="two thumbs-down on untested code", origin="agent",
    )
    assert entry["status"] == "proposed"

    main = await presets.get_agent(presets.MAIN_SLUG)
    assert improve.directive_block(main) == ""

    r = await api.post(f"/api/improve/directives/{entry['id']}", json={"action": "approve"})
    assert r.status_code == 200
    assert r.json()["status"] == "active"

    main = await presets.get_agent(presets.MAIN_SLUG)
    assert "always run it before answering" in improve.directive_block(main)


async def test_directive_cap_routes_growth_to_the_human(monkeypatch):
    await presets.seed_presets()
    monkeypatch.setattr(get_settings().improve, "max_directives", 2)

    first = await improve.add_directive(presets.MAIN_SLUG, "When X happens, always do Y.")
    second = await improve.add_directive(presets.MAIN_SLUG, "When Y happens, always do Z.")
    third = await improve.add_directive(presets.MAIN_SLUG, "When Z happens, always do W.")
    assert first["status"] == "active"
    assert second["status"] == "active"
    assert third["status"] == "proposed", "the cap must not be self-expandable"


async def test_retiring_a_directive_removes_it_from_the_prompt():
    await presets.seed_presets()
    entry = await improve.add_directive(presets.MAIN_SLUG, "When testing, always assert twice.")
    main = await presets.get_agent(presets.MAIN_SLUG)
    assert entry["id"] in improve.directive_block(main)

    hit = await improve.set_directive_status(presets.MAIN_SLUG, entry["id"], "retired")
    assert hit is not None
    main = await presets.get_agent(presets.MAIN_SLUG)
    assert improve.directive_block(main) == ""


async def test_scheduler_reflects_once_enough_runs_accumulate(monkeypatch):
    from ciws import scheduler

    await presets.seed_presets()
    monkeypatch.setattr(get_settings().improve, "reflect_after_runs", 2)
    await _seed_runs(presets.MAIN_SLUG, completed=2, failed=1)
    _fake_reflection(
        monkeypatch,
        {"assessment": "fine", "lessons": [], "directives": [], "retire": []},
    )

    report = await scheduler.reflect_if_due()
    assert report is not None and report.get("skipped") is None

    # The pass just ran, so the next tick has nothing new to reflect on.
    assert await scheduler.reflect_if_due() is None


# ---------------------------------------------------------------------------
# Code enhancement: the skill forge
# ---------------------------------------------------------------------------

GOOD_CODE = """
def run(a=0, b=0):
    return {"sum": a + b}
"""

GOOD_TEST = """
assert run(a=2, b=3)["sum"] == 5
assert run()["sum"] == 0
"""


async def test_skill_lifecycle_proposed_then_approved_then_callable(api):
    await presets.seed_presets()
    result = await skills.propose_skill(
        "adder", "Adds two numbers.", {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}}},
        GOOD_CODE, GOOD_TEST, agent_slug=presets.MAIN_SLUG,
    )
    assert result["ok"], result
    assert result["test_report"]["passed"]
    # auto_activate_skills defaults off: stored, visible, not executable.
    assert result["skill"]["status"] == "proposed"
    assert registry.get("skill_adder") is None

    r = await api.post("/api/improve/skills/adder", json={"action": "approve"})
    assert r.status_code == 200, r.text
    assert r.json()["skill"]["status"] == "active"
    assert registry.get("skill_adder") is not None

    output = await registry.call(
        "skill_adder", {"a": 20, "b": 22}, ToolContext(auto_approve=True)
    )
    assert not output.is_error, output.content
    assert json.loads(output.content) == {"sum": 42}

    stored = await skills.get_skill("adder")
    assert stored.use_count == 1
    registry.unregister("skill_adder")


async def test_failing_tests_mean_no_skill_is_stored():
    result = await skills.propose_skill(
        "broken", "Never works.", {"type": "object", "properties": {}},
        "def run():\n    return 1\n", "assert run() == 2\n",
    )
    assert not result["ok"]
    assert not result["test_report"]["passed"]
    assert await skills.get_skill("broken") is None


async def test_skill_without_run_or_without_tests_is_refused():
    no_run = await skills.propose_skill(
        "norun", "x", {"type": "object", "properties": {}}, "x = 1\n", "assert True\n"
    )
    assert not no_run["ok"] and "run()" in no_run["error"]

    untested = await skills.propose_skill(
        "untested", "x", {"type": "object", "properties": {}}, GOOD_CODE, "   "
    )
    assert not untested["ok"] and "test" in untested["error"].lower()


async def test_revision_of_an_active_skill_faces_consent_again(api):
    await skills.propose_skill(
        "adder", "Adds.", {"type": "object", "properties": {}}, GOOD_CODE, GOOD_TEST
    )
    await api.post("/api/improve/skills/adder", json={"action": "approve"})

    revised = await skills.propose_skill(
        "adder", "Adds, remembering nothing.", {"type": "object", "properties": {}},
        "def run(a=0, b=0):\n    return {\"sum\": a + b, \"v\": 2}\n",
        "assert run(a=1, b=1)[\"v\"] == 2\n",
    )
    assert revised["ok"]
    assert revised["skill"]["version"] == 2
    assert revised["skill"]["status"] == "proposed", "old approval must not cover new code"
    assert registry.get("skill_adder") is None


async def test_skills_are_gated_on_the_python_capability(monkeypatch, api):
    await skills.propose_skill(
        "gated", "x", {"type": "object", "properties": {}}, GOOD_CODE, GOOD_TEST
    )
    await api.post("/api/improve/skills/gated", json={"action": "approve"})

    monkeypatch.setattr(get_settings().security, "allow_python_tool", False)
    output = await registry.call("skill_gated", {"a": 1}, ToolContext(auto_approve=True))
    assert output.is_error
    assert "disabled" in output.content

    proposal = await skills.propose_skill(
        "another", "x", {"type": "object", "properties": {}}, GOOD_CODE, GOOD_TEST
    )
    assert not proposal["ok"]
    registry.unregister("skill_gated")


async def test_active_skills_register_at_boot():
    await skills.propose_skill(
        "booted", "x", {"type": "object", "properties": {}}, GOOD_CODE, GOOD_TEST
    )
    await skills.set_skill_status("booted", "active", actor="user")
    registry.unregister("skill_booted")  # simulate a fresh process

    assert await skills.load_active_skills() == 1
    assert registry.get("skill_booted") is not None
    registry.unregister("skill_booted")


# ---------------------------------------------------------------------------
# The agent-facing tools and the API surface
# ---------------------------------------------------------------------------


async def test_self_tools_round_trip():
    await presets.seed_presets()
    ctx = ToolContext(agent_slug=presets.MAIN_SLUG, auto_approve=True)

    status = await registry.call("self_status", {}, ctx)
    assert not status.is_error
    assert json.loads(status.content)["agent"]["slug"] == presets.MAIN_SLUG

    added = await registry.call(
        "self_directive",
        {"action": "add", "text": "When unsure of a path, list the directory first.",
         "reason": "invented paths twice"},
        ctx,
    )
    assert not added.is_error, added.content

    main = await presets.get_agent(presets.MAIN_SLUG)
    directive_id = improve.list_directives(main)[0]["id"]
    retired = await registry.call(
        "self_directive", {"action": "retire", "directive_id": directive_id}, ctx
    )
    assert not retired.is_error

    forged = await registry.call(
        "skill_forge",
        {"name": "doubler", "description": "Doubles a number.",
         "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
         "code": "def run(n=0):\n    return {\"doubled\": n * 2}\n",
         "test_code": "assert run(n=4)[\"doubled\"] == 8\n"},
        ctx,
    )
    assert not forged.is_error, forged.content
    listed = await registry.call("skill_list", {}, ctx)
    assert "doubler" in listed.content and "proposed" in listed.content


async def test_improve_status_endpoint_and_auth(api, anon):
    await presets.seed_presets()
    r = await api.get("/api/improve/status")
    assert r.status_code == 200
    body = r.json()
    assert body["agent"]["slug"] == presets.MAIN_SLUG
    assert "directives" in body and "config" in body and "skills" in body

    denied = await anon.get("/api/improve/status")
    assert denied.status_code in (401, 403)


async def test_skill_decision_endpoint_rejects_nonsense(api):
    await skills.propose_skill(
        "victim", "x", {"type": "object", "properties": {}}, GOOD_CODE, GOOD_TEST
    )
    r = await api.post("/api/improve/skills/victim", json={"action": "explode"})
    assert r.status_code == 400

    r = await api.post("/api/improve/skills/victim", json={"action": "reject"})
    assert r.status_code == 200
    async with session_scope() as s:
        row = (await s.execute(select(Skill).where(Skill.slug == "victim"))).scalar_one()
        assert row.status == "rejected"
