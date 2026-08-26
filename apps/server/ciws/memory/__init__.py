"""Persistent memory: embeddings, hybrid recall, extraction, consolidation."""

from .embeddings import backend_info, embed_one, embed_texts, dimension
from .store import (
    RecallHit,
    build_context,
    consolidate,
    extract_from_conversation,
    extract_memories,
    forget,
    get,
    list_memories,
    maybe_consolidate,
    recall,
    remember,
    stats,
    update_memory,
)

__all__ = [
    "RecallHit", "backend_info", "build_context", "consolidate", "dimension",
    "embed_one", "embed_texts", "extract_from_conversation", "extract_memories",
    "forget", "get", "list_memories", "maybe_consolidate", "recall", "remember",
    "stats", "update_memory",
]
