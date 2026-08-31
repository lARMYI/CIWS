"""M2: the workspace accumulating on its own.

The product's whole claim is that the workspace gets more useful the longer you
use it. Before this, that only happened if the user wrote every memory by hand
-- ``extract_memories`` and ``extract_graph`` existed and nothing called them,
and ``memory.extract_entities`` / ``memory.consolidate_after`` were settings
that controlled nothing.

These tests cover the wiring rather than the extraction quality: extraction
needs a model, and the suite has no credential, so a scripted provider stands in
for one. What is asserted is that capture is *attempted* under the right policy
and skipped under the wrong one, that the capture task cannot be collected
mid-flight, and that citations resolve only to sources that were really offered.
"""

from __future__ import annotations

import asyncio

import pytest

from ciws.agents import presets, runtime
from ciws.gateway.registry import gateway
from ciws.gateway.types import ModelInfo
from ciws.memory import store

from tests.test_agent_loop import ScriptedProvider


@pytest.fixture
def scripted_gateway(monkeypatch):
    """Install a scripted provider for one test only.

    monkeypatch.setitem rather than a bare assignment: a provider left in the
    registry leaks into every later test, and routes that walk every provider
    (/api/system reads p.website) then fail a long way from here.
    """

    def install(script):
        provider = ScriptedProvider(script)
        monkeypatch.setitem(gateway.providers(), "test", provider)
        monkeypatch.setattr(
            gateway,
            "_models",
            {
                "test/scripted": ModelInfo(
                    id="test/scripted", provider="test", name="scripted", context_window=128_000
                )
            },
            raising=False,
        )
        monkeypatch.setattr(gateway, "_models_loaded_at", 1e12, raising=False)
        return provider

    return install


async def _drain_capture_tasks() -> None:
    """Capture is deliberately off the response path; wait for it here."""
    for _ in range(50):
        if not runtime._capture_tasks:
            return
        await asyncio.gather(*list(runtime._capture_tasks), return_exceptions=True)
    return


# ---------------------------------------------------------------------------
# Capture policy
# ---------------------------------------------------------------------------


async def test_capture_is_attempted_after_a_successful_run(scripted_gateway, monkeypatch):
    await presets.seed_presets()
    scripted_gateway([("The deploy window is Friday afternoon.", [])])

    captured: dict = {}

    async def fake_extract(text, **kwargs):
        captured["text"] = text
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(store, "extract_memories", fake_extract)

    result = await runtime.run_agent(
        agent_slug="analyst", prompt="when do we deploy?", model_override="test/scripted"
    )
    assert result.ok
    await _drain_capture_tasks()

    assert "text" in captured, "a finished run captured nothing"
    assert "when do we deploy?" in captured["text"]
    assert "Friday afternoon" in captured["text"], "the answer was not offered for extraction"
    assert captured["kwargs"].get("source_ref") == result.run_id


async def test_an_amnesiac_agent_never_writes_to_the_shared_store(scripted_gateway, monkeypatch):
    """memory_scope 'none' is a promise, not a preference."""
    await presets.seed_presets()
    await presets.update_agent("scout", memory_scope="none")
    scripted_gateway([("Some answer.", [])])

    called = False

    async def fake_extract(text, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(store, "extract_memories", fake_extract)

    result = await runtime.run_agent(
        agent_slug="scout", prompt="look something up", model_override="test/scripted"
    )
    assert result.ok
    await _drain_capture_tasks()
    assert called is False, "an agent scoped to no memory still wrote to the store"


async def test_a_failed_run_does_not_capture(scripted_gateway, monkeypatch):
    """A wrong answer should not become a remembered fact."""
    await presets.seed_presets()
    scripted_gateway([("", [])])

    called = False

    async def fake_extract(text, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(store, "extract_memories", fake_extract)

    await runtime.run_agent(
        agent_slug="analyst", prompt="produce nothing", model_override="test/scripted"
    )
    await _drain_capture_tasks()
    assert called is False


async def test_capture_also_extracts_entities_when_enabled(scripted_gateway, monkeypatch):
    from ciws.core.config import get_settings
    from ciws.ontology import graph

    await presets.seed_presets()
    assert get_settings().memory.extract_entities is True
    scripted_gateway([("Sarah Chen owns the Helios migration.", [])])

    seen: dict = {}

    async def fake_graph_extract(text, **kwargs):
        seen["text"] = text
        return {"entities": [], "edges": []}

    monkeypatch.setattr(store, "extract_memories", lambda *a, **k: _empty())
    monkeypatch.setattr(graph, "extract_graph", fake_graph_extract)

    await runtime.run_agent(
        agent_slug="analyst", prompt="who owns Helios?", model_override="test/scripted"
    )
    await _drain_capture_tasks()
    assert "Sarah Chen" in seen.get("text", ""), "entities were never extracted from the run"


async def _empty():
    return []


async def test_a_capture_task_is_held_so_it_cannot_be_collected(scripted_gateway, monkeypatch):
    """Regression: the task set existed but nothing was added to it.

    asyncio keeps only a weak reference to a task, so an unreferenced capture
    could be garbage collected mid-flight -- indistinguishable from extraction
    having found nothing.
    """
    await presets.seed_presets()
    scripted_gateway([("An answer worth remembering.", [])])

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_extract(text, **kwargs):
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr(store, "extract_memories", slow_extract)

    await runtime.run_agent(
        agent_slug="analyst", prompt="remember this", model_override="test/scripted"
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    assert runtime._capture_tasks, "the in-flight capture task was not retained"

    release.set()
    await _drain_capture_tasks()


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


async def test_consolidation_waits_for_the_threshold(monkeypatch):
    from ciws.core.config import get_settings

    calls = []

    async def fake_consolidate(**kwargs):
        calls.append(kwargs)
        return {"merged": 0}

    monkeypatch.setattr(store, "consolidate", fake_consolidate)
    monkeypatch.setattr(get_settings().memory, "consolidate_after", 5)
    runtime._since_consolidation.clear()

    await runtime._maybe_consolidate(2, None)
    await runtime._maybe_consolidate(2, None)
    assert not calls, "consolidation ran before the threshold"

    await runtime._maybe_consolidate(2, None)
    assert calls, "consolidation never ran despite passing the threshold"


async def test_consolidation_counts_per_project(monkeypatch):
    from ciws.core.config import get_settings

    calls = []

    async def fake_consolidate(**kwargs):
        calls.append(kwargs.get("project_id"))
        return {}

    monkeypatch.setattr(store, "consolidate", fake_consolidate)
    monkeypatch.setattr(get_settings().memory, "consolidate_after", 3)
    runtime._since_consolidation.clear()

    await runtime._maybe_consolidate(2, "prj_a")
    await runtime._maybe_consolidate(2, "prj_b")
    assert not calls, "one project's writes triggered another project's pass"

    await runtime._maybe_consolidate(2, "prj_a")
    assert calls == ["prj_a"]


async def test_a_threshold_of_zero_disables_consolidation(monkeypatch):
    from ciws.core.config import get_settings

    called = False

    async def fake_consolidate(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(store, "consolidate", fake_consolidate)
    monkeypatch.setattr(get_settings().memory, "consolidate_after", 0)
    runtime._since_consolidation.clear()

    await runtime._maybe_consolidate(1000, None)
    assert called is False


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_a_marker_resolves_to_the_source_it_names():
    sources = [
        {"ref": "mem_abc123", "type": "memory", "text": "Deploys run on Friday."},
        {"ref": "chk_def456", "type": "chunk", "text": "Logs are kept 90 days."},
    ]
    answer = "Deploys run on Friday [mem_abc123]. Logs are kept for 90 days [chk_def456]."
    cited = runtime._resolve_citations(answer, sources)
    assert [c["ref"] for c in cited] == ["mem_abc123", "chk_def456"]


def test_an_invented_marker_resolves_to_nothing():
    """A citation that leads nowhere is worse than no citation."""
    sources = [{"ref": "mem_abc123", "type": "memory", "text": "real"}]
    answer = "This is definitely true [mem_fabricated99]."
    assert runtime._resolve_citations(answer, sources) == []


def test_a_repeated_marker_is_listed_once():
    sources = [{"ref": "mem_abc123", "type": "memory", "text": "real"}]
    answer = "One [mem_abc123]. Two [mem_abc123]. Three [mem_abc123]."
    assert len(runtime._resolve_citations(answer, sources)) == 1


def test_an_answer_with_no_markers_cites_nothing():
    sources = [{"ref": "mem_abc123", "type": "memory", "text": "real"}]
    assert runtime._resolve_citations("Just a plain answer.", sources) == []


def test_citations_are_empty_when_nothing_was_offered():
    assert runtime._resolve_citations("Something [mem_abc123].", []) == []


# ---------------------------------------------------------------------------
# Sourced context
# ---------------------------------------------------------------------------


async def test_context_returns_a_descriptor_for_every_memory_it_includes():
    await store.remember("Deploys run Fridays at 16:00 UTC.", kind="procedure", importance=0.8)
    await store.remember("Staging lives in eu-west-2.", kind="fact")

    text, sources = await store.build_context_with_sources("when do deploys run")
    assert text
    assert sources, "context was built with no resolvable sources"

    for source in sources:
        assert source["ref"] in text, "a source was recorded but never shown to the model"
        assert source["type"] in {"memory", "chunk"}


async def test_context_includes_the_users_own_documents(tmp_path):
    """The corpus was searchable by tool but never reached the system prompt."""
    from ciws.ingest import pipeline

    source = tmp_path / "ops.md"
    source.write_text("## Retention\nRaw event logs are retained for 90 days.\n", "utf-8")
    await pipeline.ingest_file(source)

    text, sources = await store.build_context_with_sources("how long are logs retained")
    assert "90 days" in text, "an ingested document never reached the agent's context"
    assert any(s["type"] == "chunk" for s in sources)


async def test_an_empty_workspace_produces_no_context():
    text, sources = await store.build_context_with_sources("anything at all")
    assert text == ""
    assert sources == []


async def test_the_citation_rule_is_only_added_when_there_is_something_to_cite():
    await presets.seed_presets()
    agent = await presets.get_agent("analyst")

    empty, _ = await runtime._build_system(agent, "nothing here", None, "")
    assert "Citing what you were given" not in empty

    await store.remember("The build server is called anvil.", kind="fact", importance=0.9)
    filled, sources = await runtime._build_system(agent, "what is the build server", None, "")
    assert "Citing what you were given" in filled
    assert sources


async def test_a_run_records_what_it_was_allowed_to_cite(scripted_gateway, api):
    await presets.seed_presets()
    await store.remember("The staging cluster is in eu-west-2.", kind="fact", importance=0.9)
    scripted_gateway([("Staging is in eu-west-2.", [])])

    result = await runtime.run_agent(
        agent_slug="analyst", prompt="where is staging?", model_override="test/scripted"
    )
    assert result.ok

    trace = await api.get(f"/api/runs/{result.run_id}")
    assert trace.status_code == 200
    body = trace.json()
    assert "citations" in body and "sources_offered" in body
    assert body["sources_offered"], "the run did not record the context it was given"
