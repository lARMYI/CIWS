"""Contract tests for the HTTP surface.

These exist because the six panels in the UI are the only other consumer of
these routes, and a changed response shape breaks them silently -- TypeScript
cannot check across a network boundary. Each test pins the fields the frontend
actually reads, not the whole payload, so adding a field stays cheap and
removing or renaming one is loud.

Written against the real ASGI app through an in-process transport: real routing,
real dependency injection, real error handlers, no socket.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Health and bootstrap
# ---------------------------------------------------------------------------


async def test_health_is_public(anon):
    """The health probe must not require a token -- start.sh polls it."""
    response = await anon.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["app"] == "CIWS"
    assert "version" in body


async def test_system_reports_every_subsystem(api):
    response = await api.get("/api/system")
    assert response.status_code == 200
    body = response.json()
    for key in ("version", "paths", "settings", "providers", "embeddings", "tools"):
        assert key in body, f"/system lost '{key}', which the Systems panel renders"
    assert isinstance(body["providers"], list) and body["providers"], "provider list went empty"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def test_memory_write_read_roundtrip(api):
    created = await api.post(
        "/api/memory",
        json={"content": "The deploy window is Friday.", "kind": "procedure", "importance": 0.8},
    )
    assert created.status_code == 200
    memory = created.json()
    assert memory["id"].startswith("mem_")
    assert memory["content"] == "The deploy window is Friday."
    assert memory["kind"] == "procedure"
    assert memory["importance"] == pytest.approx(0.8)

    listed = await api.get("/api/memory")
    assert listed.status_code == 200
    body = listed.json()
    assert "memories" in body
    assert any(m["id"] == memory["id"] for m in body["memories"])


async def test_memory_search_returns_scored_hits(api):
    await api.post("/api/memory", json={"content": "Redpanda replaced Kafka in the ingest tier."})
    await api.post("/api/memory", json={"content": "Lunch is at one."})

    response = await api.post("/api/memory/search", json={"query": "what replaced kafka"})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits, "search returned nothing for a direct lexical match"
    # to_dict() flattens the memory onto the hit; the Knowledge panel reads it flat.
    assert "Redpanda" in hits[0]["content"]
    assert isinstance(hits[0]["score"], (int, float))


async def test_memory_patch_and_delete(api):
    memory = (await api.post("/api/memory", json={"content": "Temporary note."})).json()

    patched = await api.patch(f"/api/memory/{memory['id']}", json={"pinned": True})
    assert patched.status_code == 200
    assert patched.json()["pinned"] is True

    deleted = await api.delete(f"/api/memory/{memory['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True


async def test_unknown_memory_is_a_typed_404(api):
    response = await api.patch("/api/memory/mem_does_not_exist", json={"pinned": True})
    assert response.status_code == 404
    body = response.json()
    assert "error" in body, "typed errors must carry a machine-readable code"
    assert "message" in body, "typed errors must carry a human-readable message"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


async def test_graph_entity_and_link_shapes(api):
    schema = await api.get("/api/graph/schema")
    assert schema.status_code == 200
    assert "entity_types" in schema.json()

    a = (await api.post("/api/graph/entity", json={"type": "person", "name": "Sarah Chen"})).json()
    b = (await api.post("/api/graph/entity", json={"type": "project", "name": "Helios"})).json()
    assert a["id"].startswith("ent_")
    # The Ontology canvas colours nodes by these three fields.
    for key in ("id", "type", "name"):
        assert key in a

    edge = await api.post(
        "/api/graph/link", json={"source_id": a["id"], "target_id": b["id"], "type": "owns"}
    )
    assert edge.status_code == 200
    link = edge.json()
    assert link["source"] == a["id"] and link["target"] == b["id"]

    graph = await api.get("/api/graph")
    assert graph.status_code == 200
    body = graph.json()
    assert "nodes" in body and "edges" in body


async def test_graph_path_and_centrality(api):
    a = (await api.post("/api/graph/entity", json={"type": "person", "name": "A"})).json()
    b = (await api.post("/api/graph/entity", json={"type": "project", "name": "B"})).json()
    await api.post("/api/graph/link", json={"source_id": a["id"], "target_id": b["id"], "type": "owns"})

    path = await api.get("/api/graph/path", params={"source": a["id"], "target": b["id"]})
    assert path.status_code == 200
    assert path.json()["found"] is True

    centrality = await api.get("/api/graph/centrality")
    assert centrality.status_code == 200


async def test_graph_search_is_not_a_full_table_scan(api):
    for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
        await api.post("/api/graph/entity", json={"type": "concept", "name": name})
    await api.post("/api/graph/entity", json={"type": "person", "name": "Marguerite Fontaine"})

    response = await api.get("/api/graph/search", params={"q": "Marguerite", "limit": 10})
    assert response.status_code == 200
    found = response.json()["entities"]
    assert found[0]["name"] == "Marguerite Fontaine"
    assert len(found) < 6, "the relevance floor stopped applying"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


async def test_corpus_ingest_search_and_delete(api, tmp_path: Path):
    source = tmp_path / "handbook.md"
    source.write_text("## Retention\nRaw event logs are retained for 90 days.\n", "utf-8")

    ingested = await api.post("/api/corpus/ingest", json={"path": str(source)})
    assert ingested.status_code == 200
    doc = ingested.json()
    assert doc["id"].startswith("doc_")
    assert doc["status"] == "ready"

    listed = await api.get("/api/corpus")
    assert listed.status_code == 200
    assert "documents" in listed.json()

    found = await api.post("/api/corpus/search", json={"query": "how long are logs kept"})
    assert found.status_code == 200
    results = found.json()["results"]
    assert results and "90 days" in results[0]["text"]

    chunks = await api.get(f"/api/corpus/{doc['id']}")
    assert chunks.status_code == 200

    removed = await api.delete(f"/api/corpus/{doc['id']}")
    assert removed.status_code == 200


# ---------------------------------------------------------------------------
# Models, tools, agents, workflows
# ---------------------------------------------------------------------------


async def test_models_route_survives_having_no_keys(api):
    """Every provider is unconfigured in tests. That is a valid state, not a 500."""
    response = await api.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body and "count" in body

    health = await api.get("/api/models/health", params={"refresh": "false"})
    assert health.status_code == 200
    assert "providers" in health.json()


async def test_tools_route_lists_specs(api, tools_loaded):
    response = await api.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) >= 30
    sample = tools[0]
    for key in ("name", "description", "risk"):
        assert key in sample, f"the Tools catalogue renders '{key}'"


async def test_agents_are_seeded_and_patchable(api):
    from ciws.agents import presets

    await presets.seed_presets()

    listed = await api.get("/api/agents")
    assert listed.status_code == 200
    agents = listed.json()["agents"]
    slugs = {a["slug"] for a in agents}
    assert {"analyst", "researcher", "engineer"} <= slugs

    patched = await api.patch("/api/agents/analyst", json={"max_steps": 12})
    assert patched.status_code == 200
    assert patched.json()["max_steps"] == 12


async def test_workflow_validation_rejects_a_cycle(api):
    response = await api.post(
        "/api/workflows/validate",
        json={
            "nodes": [{"id": "a", "type": "prompt"}, {"id": "b", "type": "prompt"}],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},
            ],
        },
    )
    assert response.status_code == 200
    problems = response.json()["problems"]
    assert any("Cycle" in p for p in problems)


async def test_workflow_create_and_run(api):
    created = await api.post(
        "/api/workflows",
        json={
            "name": "Echo",
            "graph": {
                "nodes": [
                    {"id": "i", "type": "input", "config": {"key": "text"}},
                    {"id": "u", "type": "code", "config": {"expression": "value.upper()"}},
                    {"id": "o", "type": "output", "config": {"key": "result"}},
                ],
                "edges": [
                    {"source": "i", "target": "u"},
                    {"source": "u", "target": "o"},
                ],
            },
        },
    )
    assert created.status_code == 200
    workflow = created.json()

    run = await api.post(f"/api/workflows/{workflow['id']}/run", json={"text": "ciws"})
    assert run.status_code == 200
    body = run.json()
    assert body["status"] == "completed"
    assert body["outputs"]["result"] == "CIWS"


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


async def test_media_library_is_empty_not_broken(api):
    response = await api.get("/api/media")
    assert response.status_code == 200
    assert response.json()["assets"] == []


async def test_media_models_route_lists_backends(api):
    response = await api.get("/api/media/models", params={"kind": "image"})
    assert response.status_code == 200
    models = response.json()["models"]
    assert models, "no image backends listed"
    assert all("id" in m for m in models)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


async def test_settings_round_trip(api):
    read = await api.get("/api/settings")
    assert read.status_code == 200
    assert "security" in read.json()

    written = await api.patch("/api/settings", json={"agent": {"max_steps": 25}})
    assert written.status_code == 200
