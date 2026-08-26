"""Tools for querying and extending the entity graph."""

from __future__ import annotations

from typing import Any

from ...ontology import graph, schema
from ..registry import Risk, ToolOutput, registry


def _fmt(node: dict[str, Any]) -> str:
    bits = [f"{node['name']} ({node['type']})", f"id={node['id']}"]
    if node.get("degree"):
        bits.append(f"links={node['degree']}")
    if node.get("description"):
        bits.append(f"-- {node['description'][:140]}")
    return "  ".join(bits)


@registry.tool(
    "graph_search",
    category="graph",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A name, alias, or description to look for."},
            "types": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Optional filter, e.g. {list(schema.ENTITY_TYPES)[:6]}",
            },
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
)
async def graph_search(query: str, types: list[str] | None = None, limit: int = 10, ctx: Any = None) -> ToolOutput:
    """Find entities -- people, organizations, systems, projects, concepts.

    Use this to resolve a name to an entity id before exploring its connections.
    """
    found = await graph.search_entities(
        query, limit=limit, types=types, project_id=getattr(ctx, "project_id", None)
    )
    if not found:
        return ToolOutput.text(f"No entities matched '{query}'.")
    lines = [f"{len(found)} entities matching '{query}':"]
    lines += [f"- {_fmt(graph.node_dict(e))}" for e in found]
    return ToolOutput.text("\n".join(lines), count=len(found))


@registry.tool(
    "graph_neighbors",
    category="graph",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "From graph_search."},
            "depth": {"type": "integer", "description": "1 or 2. Above 2 gets noisy.", "default": 1},
            "limit": {"type": "integer", "default": 40},
        },
        "required": ["entity_id"],
    },
)
async def graph_neighbors(entity_id: str, depth: int = 1, limit: int = 40, ctx: Any = None) -> ToolOutput:
    """What an entity is connected to, and how.

    This is how you answer "who works on X", "what depends on Y", "what does
    this person own".
    """
    result = await graph.neighbors(entity_id, depth=max(1, min(depth, 3)), limit=limit)
    nodes = {n["id"]: n for n in result["nodes"]}
    root = nodes.get(result["root"], {}).get("name", entity_id)

    lines = [f"{root} -- {len(result['edges'])} connections across {len(nodes)} entities:"]
    for edge in result["edges"]:
        src = nodes.get(edge["source"], {}).get("name", edge["source"])
        dst = nodes.get(edge["target"], {}).get("name", edge["target"])
        arrow = "->" if edge["directed"] else "<->"
        lines.append(f"- {src} {arrow}[{edge['label']}]{arrow} {dst}")
    if result.get("truncated"):
        lines.append(f"(truncated at {limit} entities)")
    return ToolOutput.text("\n".join(lines), nodes=len(nodes), edges=len(result["edges"]))


@registry.tool(
    "graph_path",
    category="graph",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "target_id": {"type": "string"},
            "max_depth": {"type": "integer", "default": 6},
        },
        "required": ["source_id", "target_id"],
    },
)
async def graph_path(source_id: str, target_id: str, max_depth: int = 6, ctx: Any = None) -> ToolOutput:
    """Find how two entities are connected, however indirectly.

    Link analysis: use this when asked whether two things are related, or how.
    Edge direction is ignored -- a chain through "created_by" backwards is still
    a real connection.
    """
    result = await graph.shortest_path(source_id, target_id, max_depth=max_depth)
    if not result["found"]:
        return ToolOutput.text(f"No path within {max_depth} hops.")
    names = [n["name"] for n in result["nodes"]]
    labels = [e["label"] for e in result["edges"]]
    chain = names[0]
    for label, name in zip(labels, names[1:]):
        chain += f" --[{label}]--> {name}"
    return ToolOutput.text(f"{result['hops']} hops:\n{chain}", hops=result["hops"])


@registry.tool(
    "graph_upsert",
    category="graph",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Fullest form of the name."},
            "type": {"type": "string", "description": f"e.g. {list(schema.ENTITY_TYPES)[:8]}"},
            "description": {"type": "string"},
            "properties": {"type": "object", "description": "Typed attributes."},
            "aliases": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "type"],
    },
)
async def graph_upsert(
    name: str,
    type: str,  # noqa: A002 - the model expects this parameter name
    description: str = "",
    properties: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
    ctx: Any = None,
) -> ToolOutput:
    """Record an entity, or enrich one that already exists.

    Matching on name is automatic, so calling this for a known entity adds
    detail rather than creating a twin. Never record a secret value here.
    """
    entity = await graph.upsert_entity(
        type,
        name,
        description=description,
        properties=properties or {},
        aliases=aliases or [],
        project_id=getattr(ctx, "project_id", None),
        source=getattr(ctx, "run_id", None),
    )
    return ToolOutput.text(
        f"Entity {entity.id}: {entity.name} ({entity.type}), seen {entity.mention_count}x",
        id=entity.id,
    )


@registry.tool(
    "graph_link",
    category="graph",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "target_id": {"type": "string"},
            "type": {"type": "string", "description": f"e.g. {list(schema.EDGE_TYPES)[:8]}"},
            "label": {"type": "string", "description": "Optional human phrasing."},
            "confidence": {"type": "number", "default": 0.7},
        },
        "required": ["source_id", "target_id", "type"],
    },
)
async def graph_link(
    source_id: str,
    target_id: str,
    type: str,  # noqa: A002
    label: str = "",
    confidence: float = 0.7,
    ctx: Any = None,
) -> ToolOutput:
    """Connect two entities. Repeat observations strengthen an existing link."""
    edge = await graph.link(
        source_id,
        target_id,
        type,
        label=label,
        confidence=confidence,
        project_id=getattr(ctx, "project_id", None),
        source=getattr(ctx, "run_id", None),
    )
    return ToolOutput.text(f"Linked: {edge.id} ({edge.type}, seen {edge.observed_count}x)")


@registry.tool(
    "graph_extract",
    category="graph",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to mine for entities and relationships."}
        },
        "required": ["text"],
    },
)
async def graph_extract(text: str, ctx: Any = None) -> ToolOutput:
    """Pull every entity and relationship out of a passage in one pass.

    Far cheaper than calling graph_upsert repeatedly when you have a document,
    a meeting note, or a page of research to absorb.
    """
    result = await graph.extract_graph(
        text,
        project_id=getattr(ctx, "project_id", None),
        source_ref=getattr(ctx, "run_id", None),
    )
    if not result["entities"]:
        return ToolOutput.text("No entities found (or no model is configured for extraction).")
    names = ", ".join(f"{e['name']} ({e['type']})" for e in result["entities"][:25])
    return ToolOutput.text(
        f"Added {len(result['entities'])} entities and {len(result['edges'])} links.\n{names}",
        entities=len(result["entities"]),
        edges=len(result["edges"]),
    )


@registry.tool(
    "graph_central",
    category="graph",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "default": 15}},
    },
)
async def graph_central(limit: int = 15, ctx: Any = None) -> ToolOutput:
    """The most important entities in the workspace, by connectivity.

    Use this to orient yourself when the user asks what a project or body of
    work actually revolves around.
    """
    rows = await graph.centrality(project_id=getattr(ctx, "project_id", None), limit=limit)
    if not rows:
        return ToolOutput.text("The graph is empty.")
    lines = ["Most central entities:"]
    lines += [
        f"- {r['name']} ({r['type']})  links={r['degree']} rank={r['pagerank']:.4f} id={r['entity_id']}"
        for r in rows
    ]
    return ToolOutput.text("\n".join(lines))
