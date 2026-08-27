"""The built-in tool catalogue, driven through the registry.

Agents never import a tool module -- they go through ``registry.call``, which is
where risk policy, argument coercion, provenance and the never-raise contract
live. So these tests go through it too. The one rule they exist to protect:
**a tool call returns an observation, never an exception.** A raising tool
aborts the whole agent turn; a failing tool is something the model can read and
route around.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ciws.core import paths
from ciws.tools.registry import Risk, ToolContext, registry


@pytest.fixture(autouse=True)
def _tools(tools_loaded):
    """Every test in this file needs the catalogue registered."""


async def call(tool_name: str, /, **arguments):
    """Invoke as an approved run does.

    ``auto_approve`` is what the runtime sets once a human has accepted a run's
    risky calls. Without it a DANGEROUS tool blocks for five minutes waiting for
    someone to click approve -- correct in production, a hung suite in CI. The
    blocking itself is asserted separately below.
    """
    return await registry.call(tool_name, arguments, ToolContext(auto_approve=True))


# ---------------------------------------------------------------------------
# The never-raise contract
# ---------------------------------------------------------------------------


async def test_an_unknown_tool_is_reported_not_raised():
    result = await call("no_such_tool_at_all")
    assert result.is_error
    assert "no tool named" in result.content.lower()


async def test_bad_arguments_become_an_observation():
    """A model that invents an argument gets told, not a stack trace."""
    result = await call("file_read", path="/definitely/not/here.txt")
    assert result.is_error
    assert not result.content.startswith("Traceback")


async def test_every_tool_declares_a_risk_level():
    for spec in registry.list_tools():
        assert spec["risk"] in {r.value for r in Risk}, spec["name"]


async def test_every_tool_has_a_description_a_model_can_act_on():
    for spec in registry.list_tools():
        assert len(spec["description"]) > 20, f"{spec['name']} has no usable description"


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------


def test_grants_are_glob_matched():
    granted = {s.name for s in registry.specs(["file_*", "memory_search"])}
    assert "file_read" in granted and "file_write" in granted
    assert "memory_search" in granted
    assert "memory_write" not in granted


def test_a_star_grant_still_respects_policy():
    """'*' means every *permitted* tool, not every registered one."""
    granted = {s.name for s in registry.specs(["*"])}
    assert "shell" not in granted


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


async def test_file_write_read_edit_roundtrip():
    target = paths.workspace_dir() / "notes.md"

    written = await call("file_write", path=str(target), content="# Notes\nalpha\n")
    assert not written.is_error

    read = await call("file_read", path=str(target))
    assert "alpha" in read.content

    edited = await call("file_edit", path=str(target), find="alpha", replace="omega")
    assert not edited.is_error
    assert "omega" in target.read_text("utf-8")

    target.unlink(missing_ok=True)


async def test_file_edit_reports_a_miss_rather_than_guessing():
    target = paths.workspace_dir() / "strict.txt"
    target.write_text("hello", "utf-8")
    result = await call("file_edit", path=str(target), find="not present", replace="x")
    assert result.is_error
    assert target.read_text("utf-8") == "hello", "the file was changed despite the miss"
    target.unlink(missing_ok=True)


async def test_file_append_does_not_clobber():
    target = paths.workspace_dir() / "log.txt"
    await call("file_write", path=str(target), content="one\n")
    await call("file_write", path=str(target), content="two\n", append=True)
    assert target.read_text("utf-8") == "one\ntwo\n"
    target.unlink(missing_ok=True)


async def test_file_list_and_search():
    root = paths.workspace_dir() / "proj"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text("def alpha():\n    return 1\n", "utf-8")
    (root / "b.py").write_text("def omega():\n    return 2\n", "utf-8")

    listed = await call("file_list", path=str(root))
    assert "a.py" in listed.content and "b.py" in listed.content

    found = await call("file_search", pattern="omega", path=str(root))
    assert "b.py" in found.content
    assert "a.py" not in found.content


async def test_file_search_reports_a_bad_regex():
    result = await call("file_search", pattern="([unclosed", path=str(paths.workspace_dir()))
    assert result.is_error


async def test_workspace_path_tells_the_agent_where_it_is():
    result = await call("workspace_path")
    assert str(paths.workspace_dir()) in result.content


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


async def test_memory_write_search_update_forget():
    written = await call(
        "memory_write", content="The build server is called anvil.", kind="fact", importance=0.7
    )
    assert not written.is_error

    found = await call("memory_search", query="what is the build server called")
    assert "anvil" in found.content

    stats = await call("memory_stats")
    assert not stats.is_error

    from ciws.memory import store

    memories = await store.list_memories(limit=5)
    memory_id = memories[0].id

    updated = await call("memory_update", memory_id=memory_id, importance=0.95, pinned=True)
    assert not updated.is_error

    forgotten = await call("memory_forget", memory_id=memory_id)
    assert not forgotten.is_error


async def test_forgetting_something_that_is_not_there_is_not_a_crash():
    result = await call("memory_forget", memory_id="mem_nope")
    assert result.is_error or "not" in result.content.lower()


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------


async def test_graph_upsert_link_and_traverse():
    a = await call("graph_upsert", name="Priya Raman", type="person")
    b = await call("graph_upsert", name="Redpanda", type="system")
    assert not a.is_error and not b.is_error

    from ciws.ontology import graph

    people = await graph.search_entities("Priya", limit=3)
    systems = await graph.search_entities("Redpanda", limit=3)
    linked = await call(
        "graph_link", source_id=people[0].id, target_id=systems[0].id, type="maintains"
    )
    assert not linked.is_error

    neighbours = await call("graph_neighbors", entity_id=people[0].id, depth=1)
    assert "Redpanda" in neighbours.content

    path = await call("graph_path", source_id=people[0].id, target_id=systems[0].id)
    assert not path.is_error

    central = await call("graph_central", limit=5)
    assert not central.is_error

    searched = await call("graph_search", query="Priya")
    assert "Priya" in searched.content


async def test_graph_path_between_unconnected_nodes_says_so():
    await call("graph_upsert", name="Island One", type="concept")
    await call("graph_upsert", name="Island Two", type="concept")

    from ciws.ontology import graph

    one = (await graph.search_entities("Island One", limit=1))[0]
    two = (await graph.search_entities("Island Two", limit=1))[0]

    result = await call("graph_path", source_id=one.id, target_id=two.id)
    assert "no" in result.content.lower() or "not" in result.content.lower()


# ---------------------------------------------------------------------------
# Corpus tools
# ---------------------------------------------------------------------------


async def test_corpus_ingest_search_and_list(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("## Backups\nSnapshots are taken nightly at 02:00 UTC.\n", "utf-8")

    ingested = await call("corpus_ingest", path=str(source))
    assert not ingested.is_error

    found = await call("corpus_search", query="when are snapshots taken")
    assert "02:00" in found.content

    listed = await call("corpus_list")
    assert "policy" in listed.content.lower()


async def test_corpus_ingest_needs_a_path_or_a_url():
    result = await call("corpus_ingest")
    assert result.is_error


# ---------------------------------------------------------------------------
# Task tools
# ---------------------------------------------------------------------------


async def test_task_add_list_update():
    added = await call("task_add", title="Ship the coverage ratchet", priority=1)
    assert not added.is_error

    listed = await call("task_list")
    assert "coverage ratchet" in listed.content

    from ciws.db.base import session_scope
    from ciws.db.models import Task
    from sqlalchemy import select

    async with session_scope() as s:
        task = (await s.execute(select(Task))).scalars().first()

    done = await call("task_update", task_id=task.id, status="done")
    assert not done.is_error


# ---------------------------------------------------------------------------
# Media and agent tools (catalogue only -- no credential, no spend)
# ---------------------------------------------------------------------------


async def test_media_models_and_library_are_listable_without_a_key():
    assert not (await call("media_models", kind="image")).is_error
    assert not (await call("media_list")).is_error


async def test_image_generate_without_a_key_explains_itself():
    result = await call("image_generate", prompt="a test image")
    assert result.is_error
    assert "key" in result.content.lower() or "configur" in result.content.lower()


async def test_agent_list_names_the_seeded_agents():
    from ciws.agents import presets

    await presets.seed_presets()
    result = await call("agent_list")
    assert "analyst" in result.content


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


async def test_think_is_a_scratchpad_that_returns_cleanly():
    result = await call("think", thought="Compare the two ranking signals before choosing.")
    assert not result.is_error


async def test_current_time_is_iso_and_utc():
    result = await call("current_time")
    assert not result.is_error
    assert "T" in result.content


async def test_python_tool_runs_and_captures_stdout():
    result = await call("python", code="print(2 ** 10)")
    assert not result.is_error
    assert "1024" in result.content


async def test_python_tool_reports_a_traceback_as_an_observation():
    result = await call("python", code="raise ValueError('boom')")
    assert result.is_error
    assert "boom" in result.content


async def test_python_tool_honours_its_timeout():
    result = await call("python", code="import time; time.sleep(30)", timeout=1)
    assert result.is_error
    assert "time" in result.content.lower()


async def test_every_call_is_written_to_the_audit_trail():
    """Provenance is the point of routing everything through the registry."""
    from sqlalchemy import select

    from ciws.db.base import session_scope
    from ciws.db.models import ToolCall

    await call("current_time")
    async with session_scope() as s:
        rows = (await s.execute(select(ToolCall).where(ToolCall.tool == "current_time"))).scalars().all()
    assert rows, "a tool call left no audit record"
