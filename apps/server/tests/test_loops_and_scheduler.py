"""M3: loops inside the product, and jobs that run unattended.

Two gaps this closes. The workflow engine had ``branch`` and ``map`` and
``validate_graph`` explicitly rejected cycles, which is correct for a DAG and
meant CIWS could not express an iterate-until-good-enough workflow at all. And
the ``tasks`` table had a ``due_at`` and an ``assignee`` that nothing ever came
round to act on, while a ``folder`` hub could be created and started without
ever noticing a file.

The loop primitives keep the graph acyclic by iterating *inside* a node, the
way ``map`` already fans out inside one. Every loop reports why it stopped:
a loop that ends without saying why is a loop nobody can debug.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ciws import scheduler
from ciws.core.errors import ValidationFailed
from ciws.core.util import now
from ciws.workflows import engine


def _ctx() -> engine.RunContext:
    return engine.RunContext(run_id="wfr_test", workflow_id="wf_test")


@pytest.fixture
def counting_node():
    """Register a deterministic inner node so loops are testable without a model."""
    calls: list[str] = []

    @engine.node("t_count", "Counter", "logic", "Appends a tick for each pass.")
    async def _counter(config, inputs, ctx):
        value = engine._text(engine._first(inputs))
        calls.append(value)
        return {"out": value + "x"}

    yield calls
    engine.NODE_TYPES.pop("t_count", None)


# ---------------------------------------------------------------------------
# until
# ---------------------------------------------------------------------------


async def test_until_stops_when_the_condition_holds(counting_node):
    result = await engine.NODE_TYPES["until"].run(
        {"node_type": "t_count", "max_iterations": 10, "until_contains": "xxx"},
        {"in": ""},
        _ctx(),
    )
    assert result["exit_reason"] == "condition"
    assert result["iterations"] == 3
    assert result["out"] == "xxx"


async def test_until_stops_at_the_ceiling_and_says_so(counting_node):
    result = await engine.NODE_TYPES["until"].run(
        {"node_type": "t_count", "max_iterations": 4, "until_contains": "never-appears"},
        {"in": ""},
        _ctx(),
    )
    assert result["exit_reason"] == "ceiling"
    assert result["iterations"] == 4


async def test_until_stops_when_the_output_stops_changing():
    """A loop whose output has converged will not start moving again."""

    @engine.node("t_fixed", "Fixed", "logic", "Always returns the same value.")
    async def _fixed(config, inputs, ctx):
        return {"out": "settled"}

    try:
        result = await engine.NODE_TYPES["until"].run(
            {"node_type": "t_fixed", "max_iterations": 10, "until_contains": "never"},
            {"in": "start"},
            _ctx(),
        )
        assert result["exit_reason"] == "converged"
        assert result["iterations"] < 10, "a converged loop kept paying for model calls"
    finally:
        engine.NODE_TYPES.pop("t_fixed", None)


async def test_convergence_detection_can_be_turned_off():
    @engine.node("t_fixed2", "Fixed", "logic", "Always returns the same value.")
    async def _fixed(config, inputs, ctx):
        return {"out": "settled"}

    try:
        result = await engine.NODE_TYPES["until"].run(
            {
                "node_type": "t_fixed2",
                "max_iterations": 3,
                "until_contains": "never",
                "stop_on_repeat": False,
            },
            {"in": "start"},
            _ctx(),
        )
        assert result["exit_reason"] == "ceiling"
        assert result["iterations"] == 3
    finally:
        engine.NODE_TYPES.pop("t_fixed2", None)


async def test_until_respects_the_hard_ceiling(counting_node):
    """Config cannot raise the limit past the module's own maximum."""
    result = await engine.NODE_TYPES["until"].run(
        {"node_type": "t_count", "max_iterations": 10_000, "until_contains": "never"},
        {"in": ""},
        _ctx(),
    )
    assert result["iterations"] <= engine.MAX_ITERATIONS


async def test_until_stops_on_cancellation(counting_node):
    ctx = _ctx()
    ctx.cancelled.set()
    result = await engine.NODE_TYPES["until"].run(
        {"node_type": "t_count", "max_iterations": 5, "until_contains": "never"}, {"in": ""}, ctx
    )
    assert result["exit_reason"] == "cancelled"
    assert result["iterations"] == 0


async def test_until_rejects_an_unknown_inner_node():
    with pytest.raises(ValidationFailed):
        await engine.NODE_TYPES["until"].run({"node_type": "not_a_node"}, {"in": "x"}, _ctx())


async def test_a_loop_cannot_nest_itself():
    """Guards against a config that would recurse until the stack gives out."""
    with pytest.raises(ValidationFailed):
        await engine.NODE_TYPES["until"].run({"node_type": "until"}, {"in": "x"}, _ctx())


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_and_critic():
    """A worker that improves each pass and a critic that approves on the third."""
    seen: list[str] = []

    @engine.node("t_worker", "Worker", "logic", "Adds a word each pass.")
    async def _worker(config, inputs, ctx):
        brief = engine._text(engine._first(inputs))
        seen.append(brief)
        # Improves with each attempt, so the critic eventually approves.
        return {"out": "draft " * len(seen)}

    @engine.node("t_critic", "Critic", "logic", "Approves once the draft is long enough.")
    async def _critic(config, inputs, ctx):
        text = engine._text(engine._first(inputs))
        if text.count("draft") >= 3:
            return {"out": "APPROVED -- this is good enough."}
        return {"out": "REJECTED -- still too thin, add more."}

    yield seen
    engine.NODE_TYPES.pop("t_worker", None)
    engine.NODE_TYPES.pop("t_critic", None)


async def test_supervisor_loops_until_the_critic_approves(worker_and_critic):
    result = await engine.NODE_TYPES["supervisor"].run(
        {
            "worker_type": "t_worker",
            "critic_type": "t_critic",
            "max_iterations": 5,
        },
        {"in": "write something"},
        _ctx(),
    )
    assert result["exit_reason"] == "approved"
    assert result["iterations"] >= 2
    assert "APPROVED" in result["verdict"]


async def test_the_worker_is_shown_why_it_was_rejected(worker_and_critic):
    """Retrying without the objection is a re-roll, not a revision."""
    await engine.NODE_TYPES["supervisor"].run(
        {"worker_type": "t_worker", "critic_type": "t_critic", "max_iterations": 5},
        {"in": "write something"},
        _ctx(),
    )
    briefs = worker_and_critic
    assert len(briefs) >= 2
    assert "REVIEWER SAID" in briefs[1]
    assert "REJECTED" in briefs[1], "the critic's reason never reached the next attempt"


async def test_supervisor_gives_up_at_the_ceiling_with_its_last_attempt():
    @engine.node("t_w2", "Worker", "logic", "Never improves.")
    async def _worker(config, inputs, ctx):
        return {"out": "same as always"}

    @engine.node("t_c2", "Critic", "logic", "Never satisfied.")
    async def _critic(config, inputs, ctx):
        return {"out": "REJECTED -- no."}

    try:
        result = await engine.NODE_TYPES["supervisor"].run(
            {"worker_type": "t_w2", "critic_type": "t_c2", "max_iterations": 3},
            {"in": "task"},
            _ctx(),
        )
        assert result["exit_reason"] == "ceiling"
        assert result["iterations"] == 3
        assert result["out"] == "same as always", "the best attempt was thrown away"
    finally:
        engine.NODE_TYPES.pop("t_w2", None)
        engine.NODE_TYPES.pop("t_c2", None)


async def test_supervisor_rejects_unknown_worker_or_critic():
    with pytest.raises(ValidationFailed):
        await engine.NODE_TYPES["supervisor"].run(
            {"worker_type": "nope", "critic_type": "model"}, {"in": "x"}, _ctx()
        )


# ---------------------------------------------------------------------------
# The loop nodes inside a real workflow
# ---------------------------------------------------------------------------


async def test_a_workflow_containing_a_loop_validates_and_runs(counting_node):
    """The graph stays acyclic; the iteration happens inside the node."""
    graph = {
        "nodes": [
            {"id": "i", "type": "input", "config": {"key": "seed"}},
            {
                "id": "loop",
                "type": "until",
                "config": {
                    "node_type": "t_count",
                    "max_iterations": 6,
                    "until_contains": "xxx",
                },
            },
            {"id": "o", "type": "output", "config": {"key": "result"}},
            {"id": "why", "type": "output", "config": {"key": "exit_reason"}},
        ],
        "edges": [
            {"source": "i", "target": "loop"},
            {"source": "loop", "target": "o", "source_output": "out"},
            {"source": "loop", "target": "why", "source_output": "exit_reason"},
        ],
    }
    assert engine.validate_graph(graph) == []

    workflow = await engine.create_workflow(name="Refine", graph=graph)
    run = await engine.run_workflow(workflow.id, {"seed": ""})
    assert run.status == "completed"
    assert run.outputs["result"] == "xxx"


def test_the_node_catalogue_advertises_the_loop_primitives():
    catalogue = {spec["type"] for spec in engine.node_catalog()}
    assert {"until", "supervisor"} <= catalogue

    until = next(s for s in engine.node_catalog() if s["type"] == "until")
    assert "exit_reason" in until["outputs"], "a loop must report why it stopped"


def test_cycles_are_still_rejected():
    """The loop nodes exist so the graph never needs to be cyclic."""
    problems = engine.validate_graph(
        {
            "nodes": [{"id": "a", "type": "prompt"}, {"id": "b", "type": "prompt"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
        }
    )
    assert any("Cycle" in p for p in problems)


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_agents():
    from ciws.agents import presets

    await presets.seed_presets()


async def test_a_due_task_assigned_to_an_agent_runs(seeded_agents, monkeypatch):
    from ciws.agents import runtime
    from ciws.db.base import session_scope
    from ciws.db.models import Task

    ran: dict = {}

    async def fake_run(**kwargs):
        ran.update(kwargs)
        return runtime.AgentResult(run_id="run_fake", content="done")

    monkeypatch.setattr(runtime, "run_agent", fake_run)

    async with session_scope() as s:
        s.add(
            Task(
                title="Summarise yesterday's incidents",
                detail="Keep it to five lines.",
                assignee="analyst",
                due_at=now() - timedelta(minutes=1),
            )
        )

    started = await scheduler.run_due_tasks()
    assert started == 1
    assert ran["agent_slug"] == "analyst"
    assert "Summarise yesterday's incidents" in ran["prompt"]
    assert "five lines" in ran["prompt"], "the task detail never reached the agent"


async def test_a_task_that_is_not_due_yet_is_left_alone(seeded_agents):
    from ciws.db.base import session_scope
    from ciws.db.models import Task

    async with session_scope() as s:
        s.add(Task(title="Later", assignee="analyst", due_at=now() + timedelta(hours=2)))

    assert await scheduler.run_due_tasks() == 0


async def test_a_task_assigned_to_a_person_is_not_run_by_an_agent(seeded_agents):
    from ciws.db.base import session_scope
    from ciws.db.models import Task

    async with session_scope() as s:
        s.add(Task(title="Call the vendor", assignee="me", due_at=now() - timedelta(minutes=1)))

    assert await scheduler.run_due_tasks() == 0


async def test_a_task_is_claimed_so_it_cannot_run_twice(seeded_agents, monkeypatch):
    """A slow run must not be picked up again by the next tick."""
    from ciws.agents import runtime
    from ciws.db.base import session_scope
    from ciws.db.models import Task

    entered = asyncio.Event()
    release = asyncio.Event()
    runs = 0

    async def slow_run(**kwargs):
        nonlocal runs
        runs += 1
        entered.set()
        await release.wait()
        return runtime.AgentResult(run_id="run_slow", content="ok")

    monkeypatch.setattr(runtime, "run_agent", slow_run)

    async with session_scope() as s:
        s.add(Task(title="Slow one", assignee="analyst", due_at=now() - timedelta(minutes=1)))

    first = asyncio.create_task(scheduler.run_due_tasks())
    await asyncio.wait_for(entered.wait(), timeout=5)

    assert await scheduler.run_due_tasks() == 0, "the task was picked up while already running"

    release.set()
    await first
    assert runs == 1


async def test_a_failed_run_marks_the_task_failed_not_done(seeded_agents, monkeypatch):
    from ciws.agents import runtime
    from ciws.db.base import session_scope
    from ciws.db.models import Task
    from sqlalchemy import select

    async def failing_run(**kwargs):
        return runtime.AgentResult(run_id="run_bad", error="the model refused")

    monkeypatch.setattr(runtime, "run_agent", failing_run)

    async with session_scope() as s:
        s.add(Task(title="Doomed", assignee="analyst", due_at=now() - timedelta(minutes=1)))

    await scheduler.run_due_tasks()

    async with session_scope() as s:
        task = (await s.execute(select(Task))).scalars().first()
    assert task.status == "failed"
    assert "refused" in (task.meta or {}).get("error", "")


async def test_a_repeating_task_reschedules_itself(seeded_agents, monkeypatch):
    """'Every morning' has to mean more than once."""
    from ciws.agents import runtime
    from ciws.db.base import session_scope
    from ciws.db.models import Task
    from sqlalchemy import select

    async def ok_run(**kwargs):
        return runtime.AgentResult(run_id="run_ok", content="fine")

    monkeypatch.setattr(runtime, "run_agent", ok_run)

    async with session_scope() as s:
        s.add(
            Task(
                title="Daily digest",
                assignee="analyst",
                due_at=now() - timedelta(minutes=1),
                meta={"repeat_every_hours": 24},
            )
        )

    await scheduler.run_due_tasks()

    async with session_scope() as s:
        task = (await s.execute(select(Task))).scalars().first()
    assert task.status == "open", "a repeating task closed itself"
    # SQLite hands back a naive datetime; compare on the same footing.
    due = task.due_at.replace(tzinfo=None)
    assert due > now().replace(tzinfo=None), "the next occurrence was not scheduled"


async def test_a_task_assigned_to_a_missing_agent_stays_open(monkeypatch):
    from ciws.db.base import session_scope
    from ciws.db.models import Task
    from sqlalchemy import select

    async with session_scope() as s:
        s.add(
            Task(title="Orphan", assignee="no-such-agent", due_at=now() - timedelta(minutes=1))
        )

    assert await scheduler.run_due_tasks() == 0

    async with session_scope() as s:
        task = (await s.execute(select(Task))).scalars().first()
    assert task.status == "open", "an unrunnable task was silently consumed"


# ---------------------------------------------------------------------------
# Watched folders
# ---------------------------------------------------------------------------


async def test_a_watched_folder_ingests_what_it_finds(tmp_path: Path):
    from ciws.hubs import registry as hubs
    from ciws.ingest import pipeline

    watched = tmp_path / "inbox"
    watched.mkdir()
    (watched / "handbook.md").write_text("## Retention\nLogs are kept 90 days.\n", "utf-8")

    hub = await hubs.create_hub(
        name="Inbox", kind="folder", config={"path": str(watched), "recursive": True}
    )
    found = await scheduler.scan_folder(hub)
    assert found == 1

    results = await pipeline.search_corpus("how long are logs kept", limit=3)
    assert results and "90 days" in results[0]["text"]


async def test_an_unchanged_file_is_not_re_ingested(tmp_path: Path):
    from ciws.hubs import registry as hubs

    watched = tmp_path / "inbox"
    watched.mkdir()
    (watched / "note.md").write_text("Something.", "utf-8")

    hub = await hubs.create_hub(name="Inbox2", kind="folder", config={"path": str(watched)})
    assert await scheduler.scan_folder(hub) == 1

    refreshed = await hubs.get_hub(hub.slug)
    assert await scheduler.scan_folder(refreshed) == 0, "an unchanged file was scanned again"


async def test_a_new_file_is_picked_up_on_the_next_scan(tmp_path: Path):
    from ciws.hubs import registry as hubs

    watched = tmp_path / "inbox"
    watched.mkdir()
    (watched / "first.md").write_text("One.", "utf-8")

    hub = await hubs.create_hub(name="Inbox3", kind="folder", config={"path": str(watched)})
    await scheduler.scan_folder(hub)

    (watched / "second.md").write_text("Two.", "utf-8")
    refreshed = await hubs.get_hub(hub.slug)
    assert await scheduler.scan_folder(refreshed) == 1


async def test_unsupported_files_are_ignored(tmp_path: Path):
    from ciws.hubs import registry as hubs

    watched = tmp_path / "inbox"
    watched.mkdir()
    (watched / "photo.heic").write_bytes(b"\x00\x01\x02")
    (watched / ".hidden.md").write_text("secret", "utf-8")

    hub = await hubs.create_hub(name="Inbox4", kind="folder", config={"path": str(watched)})
    assert await scheduler.scan_folder(hub) == 0


async def test_a_missing_watched_path_is_an_error_not_a_crash(tmp_path: Path):
    from ciws.hubs import registry as hubs

    hub = await hubs.create_hub(
        name="Ghost", kind="folder", config={"path": str(tmp_path / "not-there")}
    )
    with pytest.raises(FileNotFoundError):
        await scheduler.scan_folder(hub)

    # scan_folders swallows it and records the failure on the hub.
    await scheduler.scan_folders()
    after = await hubs.get_hub(hub.slug)
    assert after.status == "error"


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


async def test_a_tick_runs_every_job_and_reports():
    report = await scheduler.tick()
    assert set(report) == {"tasks", "folders"}


async def test_one_failing_job_does_not_stop_the_others(monkeypatch):
    async def boom():
        raise RuntimeError("folder scan exploded")

    monkeypatch.setattr(scheduler, "scan_folders", boom)
    report = await scheduler.tick()
    assert report["tasks"] == 0
    assert "failed" in str(report["folders"])


async def test_start_is_idempotent_and_stop_is_clean():
    assert scheduler.start() is True
    assert scheduler.running() is True
    assert scheduler.start() is False, "a second scheduler loop was started"

    await scheduler.stop()
    assert scheduler.running() is False
    await scheduler.stop()
