"""The CIWS object model.

Four layers, all in one SQLite file so the workspace is a single portable thing:

* **Discourse** -- Project, Conversation, Message. What was said.
* **Knowledge**  -- Memory, Entity, Edge, Document, Chunk. What is known.
* **Action**     -- AgentDef, Run, RunStep, ToolCall, Workflow. What was done.
* **Substrate**  -- Asset, Hub, Embedding, AuditEvent. What backs it all.

Every row that can be recalled semantically has a matching row in ``embeddings``
keyed by ``(kind, ref_id)``, which keeps one vector index over memories, chunks,
entities and messages alike.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..core.util import new_id


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        exclude = exclude or set()
        out: dict[str, Any] = {}
        for col in self.__table__.columns:
            if col.name in exclude:
                continue
            val = getattr(self, col.name)
            out[col.name] = val.isoformat() if isinstance(val, datetime) else val
        return out


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ---------------------------------------------------------------------------
# Discourse
# ---------------------------------------------------------------------------


class Project(Base, TimestampMixin):
    """A workspace scope. Memory, documents and runs can be filtered to one."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("prj"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(16), default="#22d3ee")
    icon: Mapped[str] = mapped_column(String(40), default="layers")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="project")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("cnv"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(300), default="Untitled")
    summary: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(120), default="")
    agent_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    project: Mapped["Project | None"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.seq"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conv_seq", "conversation_id", "seq"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("msg"))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(20), index=True)  # user|assistant|system|tool
    content: Mapped[str] = mapped_column(Text, default="")
    thinking: Mapped[str] = mapped_column(Text, default="")
    #: Structured content parts (images, files, tool results) as provider-neutral blocks.
    parts: Mapped[list[Any]] = mapped_column(JSON, default=list)
    tool_calls: Mapped[list[Any]] = mapped_column(JSON, default=list)
    tool_call_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    provider: Mapped[str] = mapped_column(String(60), default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    error: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


class Memory(Base, TimestampMixin):
    """A durable, recallable fact, preference, event or insight."""

    __tablename__ = "memories"
    __table_args__ = (Index("ix_memories_kind_importance", "kind", "importance"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("mem"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="fact", index=True)
    # fact | preference | event | insight | procedure | identity | relationship | task | lesson
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, default="")
    importance: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    #: Exponential decay is applied at recall time, not written here.
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(60), default="agent")  # agent|user|ingest|consolidation
    source_ref: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    superseded_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Entity(Base, TimestampMixin):
    """A node in the ontology: person, org, place, system, concept, event, artifact."""

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("type", "canonical_key", name="uq_entity_type_key"),
        Index("ix_entities_type_name", "type", "name"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ent"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(48), index=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    canonical_key: Mapped[str] = mapped_column(String(320), index=True)
    aliases: Mapped[list[Any]] = mapped_column(JSON, default=list)
    description: Mapped[str] = mapped_column(Text, default="")
    #: Free-form typed attributes -- the ontology stays open by design.
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    salience: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    merged_into: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    sources: Mapped[list[Any]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Edge(Base, TimestampMixin):
    """A typed, directed, weighted link between two entities."""

    __tablename__ = "edges"
    __table_args__ = (
        Index("ix_edges_src_type", "source_id", "type"),
        Index("ix_edges_dst_type", "target_id", "type"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("edg"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    directed: Mapped[bool] = mapped_column(Boolean, default=True)
    observed_count: Mapped[int] = mapped_column(Integer, default=1)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sources: Mapped[list[Any]] = mapped_column(JSON, default=list)
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Document(Base, TimestampMixin):
    """An ingested source file, web page or pasted body of text."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("doc"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    source_type: Mapped[str] = mapped_column(String(32), default="file", index=True)  # file|url|paste|hub
    source_uri: Mapped[str] = mapped_column(Text, default="")
    stored_path: Mapped[str] = mapped_column(Text, default="")
    mime_type: Mapped[str] = mapped_column(String(120), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    # pending|extracting|chunking|embedding|ready|failed
    error: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_doc_idx", "document_id", "idx"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("chk"))
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    heading: Mapped[str] = mapped_column(String(500), default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Embedding(Base):
    """One vector index over every embeddable object in the workspace."""

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("kind", "ref_id", name="uq_embedding_ref"),
        Index("ix_embeddings_kind_model", "kind", "model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(24), index=True)  # memory|chunk|entity|message|asset
    ref_id: Mapped[str] = mapped_column(String(40), index=True)
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    #: float32 little-endian, L2-normalised at write time.
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class AgentDef(Base, TimestampMixin):
    """A saved agent: persona, model policy, tool grants, memory scope."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("agt"))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(120), default="balanced")
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_steps: Mapped[int] = mapped_column(Integer, default=24)
    tools: Mapped[list[Any]] = mapped_column(JSON, default=list)  # tool names or ["*"]
    tool_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    memory_scope: Mapped[str] = mapped_column(String(24), default="shared")  # shared|private|none
    color: Mapped[str] = mapped_column(String(16), default="#22d3ee")
    icon: Mapped[str] = mapped_column(String(40), default="cpu")
    builtin: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Run(Base):
    """One execution of an agent, from prompt to final answer."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("run"))
    conversation_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    agent_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    agent_slug: Mapped[str] = mapped_column(String(80), default="")
    parent_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    goal: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    # running|completed|failed|cancelled|awaiting_approval
    model: Mapped[str] = mapped_column(String(120), default="")
    steps: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class RunStep(Base):
    """A single reason/act cycle inside a run -- the audit trail of thought."""

    __tablename__ = "run_steps"
    __table_args__ = (Index("ix_run_steps_run_idx", "run_id", "idx"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("stp"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(24), default="model")  # model|tool|reflect|handoff
    thinking: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list[Any]] = mapped_column(JSON, default=list)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ToolCall(Base):
    """Every tool invocation, with arguments and result, kept for provenance."""

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("tc"))
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    step_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    call_id: Mapped[str] = mapped_column(String(80), default="")
    tool: Mapped[str] = mapped_column(String(120), index=True)
    source: Mapped[str] = mapped_column(String(60), default="builtin")  # builtin|mcp:<hub>
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[str] = mapped_column(Text, default="")
    ok: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str] = mapped_column(Text, default="")
    approved_by: Mapped[str] = mapped_column(String(40), default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class Skill(Base, TimestampMixin):
    """A tool an agent wrote for itself: code, tests, and an approval state.

    Self-authored code never runs from ``proposed`` -- it must pass its own
    tests and then be activated, by a human unless the user has opted into
    auto-activation. The code stays in the database rather than on disk so a
    backup carries the agent's acquired skills with everything else.
    """

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("skl"))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    #: JSON-schema for the tool's arguments, exactly as the model will see it.
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    code: Mapped[str] = mapped_column(Text, default="")
    test_code: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="proposed", index=True)
    # proposed | active | disabled | rejected
    version: Mapped[int] = mapped_column(Integer, default=1)
    agent_slug: Mapped[str] = mapped_column(String(80), default="")
    origin_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    last_test_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Workflow(Base, TimestampMixin):
    """A saved DAG of model, tool and agent nodes."""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfl"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    graph: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # {nodes: [], edges: []}
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    schedule: Mapped[str] = mapped_column(String(80), default="")  # cron-ish, empty = manual
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wfr"))
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    outputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    node_states: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------


class Asset(Base, TimestampMixin):
    """A generated or imported media file: image, video, audio, 3D."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ast"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(24), default="image", index=True)  # image|video|audio|model3d
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    # queued|running|ready|failed
    prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(60), default="", index=True)
    model: Mapped[str] = mapped_column(String(160), default="")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    file_path: Mapped[str] = mapped_column(Text, default="")
    thumb_path: Mapped[str] = mapped_column(Text, default="")
    mime_type: Mapped[str] = mapped_column(String(80), default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    remote_url: Mapped[str] = mapped_column(Text, default="")
    job_id: Mapped[str] = mapped_column(String(160), default="", index=True)
    parent_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    error: Mapped[str] = mapped_column(Text, default="")
    favorite: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Hub(Base, TimestampMixin):
    """A connector: an MCP server, a watched folder, a web search backend, an API."""

    __tablename__ = "hubs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("hub"))
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # mcp_stdio|mcp_http|folder|websearch|http_api|database
    description: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    autostart: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(24), default="stopped", index=True)
    # stopped|starting|ready|error|disabled
    status_detail: Mapped[str] = mapped_column(Text, default="")
    tool_count: Mapped[int] = mapped_column(Integer, default=0)
    tools: Mapped[list[Any]] = mapped_column(JSON, default=list)
    last_ok: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelRecord(Base, TimestampMixin):
    """A model the hub knows about, with live health and rolling usage."""

    __tablename__ = "model_records"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)  # "provider/model"
    provider: Mapped[str] = mapped_column(String(60), index=True)
    name: Mapped[str] = mapped_column(String(160))
    display_name: Mapped[str] = mapped_column(String(200), default="")
    family: Mapped[str] = mapped_column(String(60), default="")
    context_window: Mapped[int] = mapped_column(Integer, default=0)
    max_output: Mapped[int] = mapped_column(Integer, default=0)
    modalities: Mapped[list[Any]] = mapped_column(JSON, default=list)  # text|image|audio|video
    capabilities: Mapped[list[Any]] = mapped_column(JSON, default=list)  # tools|vision|json|thinking
    input_cost_per_mtok: Mapped[float] = mapped_column(Float, default=0.0)
    output_cost_per_mtok: Mapped[float] = mapped_column(Float, default=0.0)
    local: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    favorite: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    status: Mapped[str] = mapped_column(String(24), default="unknown", index=True)
    status_detail: Mapped[str] = mapped_column(Text, default="")
    last_checked: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    call_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AuditEvent(Base):
    """Append-only record of consequential actions."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(80), default="system", index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    target: Mapped[str] = mapped_column(String(200), default="")
    outcome: Mapped[str] = mapped_column(String(24), default="ok", index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Task(Base, TimestampMixin):
    """A unit of work the hub is tracking -- surfaced on the command board."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("tsk"))
    project_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(400))
    detail: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    # open|in_progress|blocked|done|cancelled
    priority: Mapped[int] = mapped_column(Integer, default=2, index=True)  # 0 highest
    assignee: Mapped[str] = mapped_column(String(80), default="")  # agent slug or "me"
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


ALL_TABLES = (
    Project, Conversation, Message, Memory, Entity, Edge, Document, Chunk,
    Embedding, AgentDef, Run, RunStep, ToolCall, Skill, Workflow, WorkflowRun,
    Asset, Hub, ModelRecord, AuditEvent, Task,
)
