"""The entity ontology: typed nodes, evidence-bearing links, and link analysis."""

from . import schema
from .graph import (
    auto_resolve,
    centrality,
    delete_entity,
    edge_dict,
    extract_graph,
    find_duplicates,
    get_entity,
    link,
    merge_entities,
    neighbors,
    node_dict,
    search_entities,
    shortest_path,
    stats,
    subgraph,
    timeline,
    unlink,
    update_entity,
    upsert_entity,
)

__all__ = [
    "schema", "auto_resolve", "centrality", "delete_entity", "edge_dict", "extract_graph",
    "find_duplicates", "get_entity", "link", "merge_entities", "neighbors", "node_dict",
    "search_entities", "shortest_path", "stats", "subgraph", "timeline", "unlink",
    "update_entity", "upsert_entity",
]
