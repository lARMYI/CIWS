"""Hub connectors: MCP servers, folders, and web search."""

from . import websearch
from .registry import (
    BUILTIN_HUBS,
    HUB_KINDS,
    create_hub,
    delete_hub,
    get_hub,
    list_hubs,
    manager,
    seed_builtin_hubs,
    test_hub,
    update_hub,
)

__all__ = [
    "websearch", "BUILTIN_HUBS", "HUB_KINDS", "create_hub", "delete_hub", "get_hub",
    "list_hubs", "manager", "seed_builtin_hubs", "test_hub", "update_hub",
]
