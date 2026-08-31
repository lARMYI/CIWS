"""The ontology's vocabulary -- suggested, not enforced.

Palantir-style tools usually ship a closed ontology that a data engineer curates
up front. That is the right call for an enterprise with a governance team, and
the wrong one for a personal hub: the moment a model wants to record a
``spacecraft`` or a ``lease_agreement`` you would rather it did than dropped
the fact.

So these types are the *known* vocabulary -- they get colours, icons and
suggested properties in the UI, and models are steered toward them -- but
:func:`normalize_type` lets an unknown type through as a slug rather than
rejecting it. Curation happens afterwards, in the Ontology panel, where you can
see what drifted in and merge it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

ENTITY_TYPES: dict[str, dict[str, Any]] = {
    "person": {
        "label": "Person", "icon": "user", "color": "#22d3ee",
        "description": "A human being.",
        "common_properties": ["role", "email", "organization", "location", "phone"],
    },
    "organization": {
        "label": "Organization", "icon": "building", "color": "#a78bfa",
        "description": "A company, team, agency, or group.",
        "common_properties": ["industry", "size", "location", "website", "founded"],
    },
    "project": {
        "label": "Project", "icon": "folder-git", "color": "#34d399",
        "description": "A named body of work with a goal.",
        "common_properties": ["status", "owner", "started", "due", "repository"],
    },
    "system": {
        "label": "System", "icon": "server", "color": "#60a5fa",
        "description": "A service, application, or piece of infrastructure.",
        "common_properties": ["environment", "language", "url", "owner", "version"],
    },
    "location": {
        "label": "Location", "icon": "map-pin", "color": "#fbbf24",
        "description": "A place, physical or virtual.",
        "common_properties": ["country", "region", "coordinates", "timezone"],
    },
    "event": {
        "label": "Event", "icon": "calendar", "color": "#f472b6",
        "description": "Something that happened or will happen at a time.",
        "common_properties": ["date", "duration", "participants", "outcome"],
    },
    "concept": {
        "label": "Concept", "icon": "lightbulb", "color": "#818cf8",
        "description": "An idea, method, standard, or abstraction.",
        "common_properties": ["domain", "definition", "source"],
    },
    "document": {
        "label": "Document", "icon": "file-text", "color": "#94a3b8",
        "description": "A written artifact.",
        "common_properties": ["author", "created", "url", "format"],
    },
    "artifact": {
        "label": "Artifact", "icon": "package", "color": "#fb923c",
        "description": "A produced thing: a build, an image, a dataset, a model file.",
        "common_properties": ["kind", "size", "path", "created_by"],
    },
    "dataset": {
        "label": "Dataset", "icon": "database", "color": "#2dd4bf",
        "description": "A structured collection of records.",
        "common_properties": ["rows", "schema", "source", "updated"],
    },
    "credential": {
        "label": "Credential", "icon": "key", "color": "#f87171",
        "description": "An account or access grant. Never store the secret itself here.",
        "common_properties": ["service", "scope", "owner", "expires"],
    },
    "model": {
        "label": "Model", "icon": "cpu", "color": "#c084fc",
        "description": "An AI model.",
        "common_properties": ["provider", "context_window", "modalities", "cost"],
    },
    "tool": {
        "label": "Tool", "icon": "wrench", "color": "#4ade80",
        "description": "A capability an agent can invoke.",
        "common_properties": ["category", "risk", "source"],
    },
    "task": {
        "label": "Task", "icon": "check-square", "color": "#facc15",
        "description": "A unit of work to be done.",
        "common_properties": ["status", "assignee", "due", "priority"],
    },
    "topic": {
        "label": "Topic", "icon": "hash", "color": "#38bdf8",
        "description": "A subject area that recurs across the workspace.",
        "common_properties": ["parent", "keywords"],
    },
}

EDGE_TYPES: dict[str, dict[str, Any]] = {
    "works_for":   {"label": "works for",   "inverse": "employs",       "directed": True,  "color": "#22d3ee"},
    "member_of":   {"label": "member of",   "inverse": "has member",    "directed": True,  "color": "#22d3ee"},
    "knows":       {"label": "knows",       "inverse": "knows",         "directed": False, "color": "#7dd3fc"},
    "located_in":  {"label": "located in",  "inverse": "contains",      "directed": True,  "color": "#fbbf24"},
    "part_of":     {"label": "part of",     "inverse": "has part",      "directed": True,  "color": "#a78bfa"},
    "depends_on":  {"label": "depends on",  "inverse": "required by",   "directed": True,  "color": "#f87171"},
    "owns":        {"label": "owns",        "inverse": "owned by",      "directed": True,  "color": "#34d399"},
    "created_by":  {"label": "created by",  "inverse": "created",       "directed": True,  "color": "#94a3b8"},
    "mentions":    {"label": "mentions",    "inverse": "mentioned in",  "directed": True,  "color": "#64748b"},
    "related_to":  {"label": "related to",  "inverse": "related to",    "directed": False, "color": "#818cf8"},
    "derived_from": {"label": "derived from", "inverse": "source of",   "directed": True,  "color": "#2dd4bf"},
    "precedes":    {"label": "precedes",    "inverse": "follows",       "directed": True,  "color": "#f472b6"},
    "causes":      {"label": "causes",      "inverse": "caused by",     "directed": True,  "color": "#fb923c"},
    "contradicts": {"label": "contradicts", "inverse": "contradicted by", "directed": False, "color": "#ef4444"},
    "supports":    {"label": "supports",    "inverse": "supported by",  "directed": True,  "color": "#4ade80"},
    "assigned_to": {"label": "assigned to", "inverse": "assigned",      "directed": True,  "color": "#facc15"},
    "uses":        {"label": "uses",        "inverse": "used by",       "directed": True,  "color": "#60a5fa"},
}

#: Free-text a model might emit, mapped onto the known vocabulary.
_TYPE_ALIASES = {
    "human": "person", "people": "person", "individual": "person", "user": "person",
    "employee": "person", "contact": "person", "author": "person",
    "company": "organization", "org": "organization", "team": "organization",
    "business": "organization", "agency": "organization", "institution": "organization",
    "group": "organization", "vendor": "organization",
    "initiative": "project", "programme": "project", "program": "project", "product": "project",
    "service": "system", "application": "system", "app": "system", "software": "system",
    "platform": "system", "repo": "system", "repository": "system", "infrastructure": "system",
    "place": "location", "city": "location", "country": "location", "region": "location",
    "office": "location", "site": "location",
    "meeting": "event", "incident": "event", "milestone": "event", "release": "event",
    "idea": "concept", "theory": "concept", "method": "concept", "technique": "concept",
    "standard": "concept", "framework": "concept", "principle": "concept",
    "file": "document", "paper": "document", "report": "document", "article": "document",
    "note": "document", "spec": "document",
    "asset": "artifact", "image": "artifact", "video": "artifact", "build": "artifact",
    "data": "dataset", "table": "dataset", "database": "dataset",
    "account": "credential", "key": "credential", "token": "credential", "login": "credential",
    "llm": "model", "ai_model": "model", "algorithm": "model",
    "capability": "tool", "function": "tool", "command": "tool",
    "todo": "task", "ticket": "task", "issue": "task", "action_item": "task",
    "subject": "topic", "theme": "topic", "category": "topic", "tag": "topic",
}

_EDGE_ALIASES = {
    "employed_by": "works_for", "employee_of": "works_for", "works_at": "works_for",
    "belongs_to": "member_of", "in_team": "member_of",
    "friend_of": "knows", "colleague_of": "knows", "collaborates_with": "knows",
    "based_in": "located_in", "lives_in": "located_in", "situated_in": "located_in",
    "component_of": "part_of", "subsystem_of": "part_of", "contains": "part_of",
    "requires": "depends_on", "needs": "depends_on", "built_on": "depends_on",
    "owner_of": "owns", "manages": "owns", "maintains": "owns", "responsible_for": "owns",
    "authored_by": "created_by", "written_by": "created_by", "made_by": "created_by",
    "references": "mentions", "cites": "mentions", "discusses": "mentions",
    "associated_with": "related_to", "linked_to": "related_to", "similar_to": "related_to",
    "based_on": "derived_from", "extends": "derived_from", "forked_from": "derived_from",
    "before": "precedes", "leads_to": "precedes", "followed_by": "precedes",
    "results_in": "causes", "triggers": "causes", "produces": "causes",
    "conflicts_with": "contradicts", "disagrees_with": "contradicts",
    "confirms": "supports", "evidences": "supports", "backs": "supports",
    "owned_by_task": "assigned_to", "assignee": "assigned_to",
    "utilizes": "uses", "consumes": "uses", "calls": "uses",
}

_slug_re = re.compile(r"[^a-z0-9]+")
_ws_re = re.compile(r"\s+")


def _slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _slug_re.sub("_", value.strip().lower()).strip("_")


def normalize_type(raw: str) -> str:
    """Map free text onto a known entity type, or slugify it and let it through."""
    if not raw:
        return "concept"
    slug = _slug(raw)
    if slug in ENTITY_TYPES:
        return slug
    if slug in _TYPE_ALIASES:
        return _TYPE_ALIASES[slug]
    singular = slug[:-1] if slug.endswith("s") and len(slug) > 3 else slug
    if singular in ENTITY_TYPES:
        return singular
    if singular in _TYPE_ALIASES:
        return _TYPE_ALIASES[singular]
    return slug or "concept"


def normalize_edge_type(raw: str) -> str:
    if not raw:
        return "related_to"
    slug = _slug(raw)
    if slug in EDGE_TYPES:
        return slug
    if slug in _EDGE_ALIASES:
        return _EDGE_ALIASES[slug]
    return slug or "related_to"


def canonical_key(entity_type: str, name: str) -> str:
    """Identity key for deduplication.

    Case, punctuation and spacing collapse; the type stays in the key so a
    person named "Atlas" and a project named "Atlas" remain distinct entities.
    """
    cleaned = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip().lower()
    cleaned = _ws_re.sub(" ", cleaned)
    # Strip leading articles so "The Acme Corporation" == "Acme Corporation".
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    return f"{normalize_type(entity_type)}:{cleaned}"[:320]


def entity_meta(entity_type: str) -> dict[str, Any]:
    return ENTITY_TYPES.get(
        entity_type,
        {
            "label": entity_type.replace("_", " ").title(),
            "icon": "circle",
            "color": "#64748b",
            "description": "Custom type discovered from your data.",
            "common_properties": [],
        },
    )


def edge_meta(edge_type: str) -> dict[str, Any]:
    return EDGE_TYPES.get(
        edge_type,
        {
            "label": edge_type.replace("_", " "),
            "inverse": edge_type.replace("_", " "),
            "directed": True,
            "color": "#475569",
        },
    )


def entity_color(entity_type: str) -> str:
    return str(entity_meta(entity_type)["color"])


def edge_color(edge_type: str) -> str:
    return str(edge_meta(edge_type)["color"])


def describe() -> dict[str, Any]:
    """The vocabulary, shaped for the UI's type pickers and legend."""
    return {
        "entity_types": [{"type": k, **v} for k, v in ENTITY_TYPES.items()],
        "edge_types": [{"type": k, **v} for k, v in EDGE_TYPES.items()],
    }
