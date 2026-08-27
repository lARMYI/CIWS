"""Memory, ontology, ingestion and workflow behaviour.

These lock in the parts with subtle logic that a refactor could silently break:
recall ranking, entity resolution, heading-aware chunking, and DAG validation.
Everything runs on the built-in local embeddings, so the suite needs no key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ciws.ingest import chunk as chunker
from ciws.ingest import pipeline
from ciws.memory import embeddings, store
from ciws.ontology import graph, schema
from ciws.workflows import engine as workflows


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def test_recall_ranks_the_relevant_memory_first():
    await store.remember("The user prefers dark UI themes and dense layouts.", kind="preference", importance=0.8)
    await store.remember("CIWS stores everything in a single SQLite file.", kind="fact", importance=0.7)
    await store.remember("Deploys happen on Fridays after the smoke suite passes.", kind="procedure")
    await store.remember("Sarah Chen leads the platform team.", kind="fact")

    hits = await store.recall("dark theme preference", limit=3)
    assert hits, "recall returned nothing"
    assert "dark UI" in hits[0].memory.content

    hits = await store.recall("when do deploys happen", limit=3)
    assert "Deploys" in hits[0].memory.content


async def test_recall_does_not_return_the_whole_store_for_a_narrow_query():
    for i in range(8):
        await store.remember(f"Unrelated fact number {i} about widgets.", kind="fact")
    await store.remember("The database password rotation runs quarterly.", kind="procedure")

    hits = await store.recall("password rotation schedule", limit=10)
    assert hits
    assert "password rotation" in hits[0].memory.content
    # A relevance floor must apply -- otherwise every memory is a "result".
    assert len(hits) < 9


async def test_identical_memories_are_deduplicated():
    first = await store.remember("The deploy window is Friday afternoon.", kind="procedure")
    second = await store.remember("The deploy window is Friday afternoon.", kind="procedure")
    assert first.id == second.id
    assert second.access_count >= 1

    stats = await store.stats()
    assert stats["total"] == 1


async def test_pinned_memory_does_not_outrank_a_direct_match():
    await store.remember("The user's name is Jordan.", kind="identity", importance=0.95, pinned=True)
    await store.remember("The staging cluster lives in eu-west-2.", kind="fact", importance=0.4)

    hits = await store.recall("which region is staging in", limit=3)
    assert hits
    assert "eu-west-2" in hits[0].memory.content


async def test_local_embeddings_are_deterministic():
    a = embeddings.hash_embed("the quick brown fox")
    b = embeddings.hash_embed("the quick brown fox")
    assert a == b
    assert len(a) == embeddings.HASH_DIM
    assert abs(sum(x * x for x in a) - 1.0) < 1e-4  # L2 normalised


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


async def test_entities_dedupe_on_canonical_key():
    first = await graph.upsert_entity("person", "Sarah Chen")
    again = await graph.upsert_entity("Person", "  sarah   chen  ")
    assert first.id == again.id
    assert again.mention_count == 2


async def test_type_aliases_normalise():
    assert schema.normalize_type("Company") == "organization"
    assert schema.normalize_type("repos") == "system"
    # An unknown type is allowed through, slugified -- the ontology is open.
    assert schema.normalize_type("Spacecraft Fleet") == "spacecraft_fleet"


async def test_same_name_different_type_stays_distinct():
    person = await graph.upsert_entity("person", "Atlas")
    project = await graph.upsert_entity("project", "Atlas")
    assert person.id != project.id


async def test_repeat_links_strengthen_rather_than_duplicate():
    a = await graph.upsert_entity("person", "Jo")
    b = await graph.upsert_entity("organization", "Acme")
    first = await graph.link(a.id, b.id, "works_for")
    second = await graph.link(a.id, b.id, "employed_by")  # alias of works_for
    assert first.id == second.id
    assert second.observed_count == 2


async def test_shortest_path_ignores_edge_direction():
    a = await graph.upsert_entity("person", "A")
    b = await graph.upsert_entity("project", "B")
    c = await graph.upsert_entity("system", "C")
    await graph.link(a.id, b.id, "owns")
    await graph.link(c.id, b.id, "part_of")  # points the other way

    path = await graph.shortest_path(a.id, c.id)
    assert path["found"]
    assert path["hops"] == 2


async def test_merge_repoints_edges_and_keeps_provenance():
    keep = await graph.upsert_entity("person", "Sarah Chen", description="Platform lead")
    drop = await graph.upsert_entity("person", "Sarah Chen (Platform)", description="Platform lead")
    org = await graph.upsert_entity("organization", "Acme")
    await graph.link(drop.id, org.id, "works_for")

    merged = await graph.merge_entities(keep.id, [drop.id])
    assert "Sarah Chen (Platform)" in merged.aliases

    neighbourhood = await graph.neighbors(keep.id, depth=1)
    assert any(n["id"] == org.id for n in neighbourhood["nodes"])

    # The old id must still resolve, so stale citations do not dangle.
    resolved = await graph.get_entity(drop.id)
    assert resolved is not None and resolved.id == keep.id


async def test_merge_does_not_create_self_edges():
    a = await graph.upsert_entity("person", "One")
    b = await graph.upsert_entity("person", "Two")
    await graph.link(a.id, b.id, "knows")
    merged = await graph.merge_entities(a.id, [b.id])
    result = await graph.neighbors(merged.id, depth=1)
    assert all(e["source"] != e["target"] for e in result["edges"])


async def test_entity_search_has_a_relevance_floor():
    for name in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
        await graph.upsert_entity("concept", name)
    await graph.upsert_entity("person", "Marguerite Fontaine")

    found = await graph.search_entities("Marguerite", limit=10)
    assert found
    assert found[0].name == "Marguerite Fontaine"
    assert len(found) < 6


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_chunking_is_heading_aware():
    text = (
        "# Handbook\n\n"
        "## Deployment\nDeploys run every Friday at 16:00 UTC.\n\n"
        "## Retention\nRaw logs are kept for 90 days.\n"
    )
    chunks = chunker.chunk_text(text, target_tokens=200)
    headings = [c.heading for c in chunks]
    assert "Deployment" in headings
    assert "Retention" in headings
    assert all(c.text.strip() for c in chunks)


def test_oversized_paragraph_is_split_not_dropped():
    text = "Intro.\n\n" + ("word " * 4000)
    chunks = chunker.chunk_text(text, target_tokens=120)
    assert len(chunks) > 3
    assert sum(len(c.text) for c in chunks) > 10_000


async def test_ingest_chunks_embeds_and_searches(tmp_path: Path):
    source = tmp_path / "handbook.md"
    source.write_text(
        "# Ops Handbook\n\n"
        "## Incident Response\n"
        "Sev1 incidents page the on-call rotation immediately.\n\n"
        "## Data Retention\n"
        "Raw event logs are retained for 90 days.\n",
        "utf-8",
    )

    doc = await pipeline.ingest_file(source)
    assert doc.status == "ready"
    assert doc.chunk_count >= 2

    results = await pipeline.search_corpus("how long are event logs retained", limit=3)
    assert results
    assert "90 days" in results[0]["text"]
    assert results[0]["heading"] == "Data Retention"


async def test_identical_files_are_ingested_once(tmp_path: Path):
    source = tmp_path / "note.txt"
    source.write_text("CIWS keeps everything local.", "utf-8")
    first = await pipeline.ingest_file(source)
    second = await pipeline.ingest_file(source)
    assert first.id == second.id


async def test_deleting_a_document_removes_its_chunks(tmp_path: Path):
    source = tmp_path / "temp.md"
    source.write_text("## Section\nSomething searchable about turbines.\n", "utf-8")
    doc = await pipeline.ingest_file(source)
    assert await pipeline.search_corpus("turbines", limit=2)

    assert await pipeline.delete_document(doc.id)
    assert await pipeline.get_chunks(doc.id) == []
    assert await pipeline.search_corpus("turbines", limit=2) == []


async def test_unreadable_format_raises_a_typed_error(tmp_path: Path):
    from ciws.core.errors import ValidationFailed

    binary = tmp_path / "thing.bin"
    binary.write_bytes(bytes(range(256)) * 40)
    with pytest.raises(ValidationFailed):
        await pipeline.ingest_file(binary)


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def test_validate_names_the_cycle():
    problems = workflows.validate_graph(
        {
            "nodes": [{"id": "a", "type": "prompt"}, {"id": "b", "type": "prompt"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
        }
    )
    assert any("Cycle" in p for p in problems)


def test_validate_catches_unknown_types_and_dangling_edges():
    problems = workflows.validate_graph(
        {
            "nodes": [{"id": "a", "type": "input"}, {"id": "b", "type": "not_a_node"}],
            "edges": [{"source": "a", "target": "ghost"}],
        }
    )
    assert any("unknown type" in p for p in problems)
    assert any("ghost" in p for p in problems)


async def test_workflow_runs_and_captures_outputs():
    workflow = await workflows.create_workflow(
        name="Echo",
        graph={
            "nodes": [
                {"id": "i", "type": "input", "config": {"key": "text"}},
                {"id": "p", "type": "prompt", "config": {"template": ">> {in} <<"}},
                {"id": "u", "type": "code", "config": {"expression": "value.upper()"}},
                {"id": "o", "type": "output", "config": {"key": "result"}},
            ],
            "edges": [
                {"source": "i", "target": "p"},
                {"source": "p", "target": "u"},
                {"source": "u", "target": "o"},
            ],
        },
    )
    run = await workflows.run_workflow(workflow.id, {"text": "ciws online"})
    assert run.status == "completed"
    assert run.outputs["result"] == ">> CIWS ONLINE <<"


async def test_a_failed_node_skips_descendants_but_not_siblings():
    workflow = await workflows.create_workflow(
        name="Isolation",
        graph={
            "nodes": [
                {"id": "i", "type": "input", "config": {"key": "text"}},
                {"id": "bad", "type": "code", "config": {"expression": "1/0"}},
                {"id": "after", "type": "prompt", "config": {"template": "never {in}"}},
                {"id": "ok", "type": "prompt", "config": {"template": "sibling saw {in}"}},
                {"id": "o", "type": "output", "config": {"key": "ok"}},
            ],
            "edges": [
                {"source": "i", "target": "bad"},
                {"source": "bad", "target": "after"},
                {"source": "i", "target": "ok"},
                {"source": "ok", "target": "o"},
            ],
        },
    )
    run = await workflows.run_workflow(workflow.id, {"text": "x"})
    states = {k: v["status"] for k, v in run.node_states.items()}
    assert states["bad"] == "failed"
    assert states["after"] == "skipped"
    assert states["ok"] == "done"
    assert run.outputs["ok"] == "sibling saw x"


# ---------------------------------------------------------------------------
# Curation (M4)
# ---------------------------------------------------------------------------


async def test_duplicates_are_proposed_for_a_near_identical_pair():
    await graph.upsert_entity("person", "Sarah Chen", description="Platform lead")
    await graph.upsert_entity("person", "Sarah Chen (Platform)", description="Platform lead")

    pairs = await graph.find_duplicates(threshold=0.7)
    assert pairs, "no candidate was proposed for an obvious duplicate"
    names = {p["a"]["name"] for p in pairs} | {p["b"]["name"] for p in pairs}
    assert "Sarah Chen" in names


async def test_a_dismissed_pair_stops_being_proposed():
    """Otherwise a curation queue never empties and stops being a to-do list."""
    # An alias collision is a guaranteed candidate, so the test exercises the
    # dismissal rather than the similarity threshold.
    a = await graph.upsert_entity("person", "Alex Rivera", description="Engineer")
    b = await graph.upsert_entity(
        "person", "Alexander Rivera", description="Engineer", aliases=["Alex Rivera"]
    )

    before = await graph.find_duplicates(threshold=0.7)
    assert any({p["a"]["id"], p["b"]["id"]} == {a.id, b.id} for p in before)

    await graph.dismiss_duplicate(a.id, b.id)

    after = await graph.find_duplicates(threshold=0.7)
    assert not any({p["a"]["id"], p["b"]["id"]} == {a.id, b.id} for p in after)


async def test_dismissal_is_recorded_on_both_entities():
    a = await graph.upsert_entity("concept", "Alpha One")
    b = await graph.upsert_entity("concept", "Alpha Two")
    await graph.dismiss_duplicate(a.id, b.id)

    left = await graph.get_entity(a.id)
    right = await graph.get_entity(b.id)
    assert b.id in (left.properties or {}).get(graph.NOT_DUPLICATE_OF, [])
    assert a.id in (right.properties or {}).get(graph.NOT_DUPLICATE_OF, [])


async def test_dismissing_twice_does_not_duplicate_the_record():
    a = await graph.upsert_entity("concept", "Beta One")
    b = await graph.upsert_entity("concept", "Beta Two")
    await graph.dismiss_duplicate(a.id, b.id)
    await graph.dismiss_duplicate(a.id, b.id)

    left = await graph.get_entity(a.id)
    assert (left.properties or {}).get(graph.NOT_DUPLICATE_OF, []).count(b.id) == 1


async def test_dismissing_an_entity_against_itself_is_refused():
    from ciws.core.errors import ValidationFailed

    a = await graph.upsert_entity("concept", "Solo")
    with pytest.raises(ValidationFailed):
        await graph.dismiss_duplicate(a.id, a.id)


async def test_dismissing_an_unknown_entity_is_a_typed_error():
    from ciws.core.errors import NotFound

    a = await graph.upsert_entity("concept", "Real")
    with pytest.raises(NotFound):
        await graph.dismiss_duplicate(a.id, "ent_does_not_exist")


async def test_merging_still_works_after_a_dismissal_elsewhere():
    """A dismissal must not block unrelated merges."""
    a = await graph.upsert_entity("person", "Casey Doe")
    b = await graph.upsert_entity("person", "Casey Doe (Ops)")
    other = await graph.upsert_entity("concept", "Unrelated")
    await graph.dismiss_duplicate(a.id, other.id)

    merged = await graph.merge_entities(a.id, [b.id])
    assert "Casey Doe (Ops)" in merged.aliases
