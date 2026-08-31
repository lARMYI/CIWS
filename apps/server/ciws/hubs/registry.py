"""Hubs: the connectors that give agents reach beyond the workspace.

A hub is a stored connection -- an MCP server, a watched folder, a web-search
backend. The manager owns their lifecycle and bridges MCP tools into the tool
registry under a ``<hub>__<tool>`` namespace, so two servers can both expose
``search`` without colliding.

Nothing auto-launches. The seeded MCP servers ship disabled, because silently
spawning ``npx`` subprocesses on first boot is not a thing software should do
to you.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import or_, select

from ..core import paths
from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import now, slugify
from ..db.base import session_scope
from ..db.models import Hub
from ..mcp.client import MCPClient, open_client

log = get_logger("hubs")

MCP_KINDS = {"mcp_stdio", "mcp_http"}
HUB_KINDS = MCP_KINDS | {"folder", "websearch", "http_api"}

#: Connectors offered out of the box. Data, not rows -- seeded on first boot.
BUILTIN_HUBS: list[dict[str, Any]] = [
    {
        "slug": "workspace",
        "name": "Workspace Folder",
        "kind": "folder",
        "description": "The scratch directory agents can read and write freely.",
        "config": {"path": ""},  # filled in at seed time
        "enabled": True,
        "builtin": True,
    },
    {
        "slug": "web",
        "name": "Web Search",
        "kind": "websearch",
        "description": "Search and fetch pages. Works keyless via DuckDuckGo; add a "
                       "Tavily, Brave, Serper or Exa key for better results.",
        "config": {"backend": "auto"},
        "enabled": True,
        "builtin": True,
    },
    {
        "slug": "mcp-filesystem",
        "name": "MCP: Filesystem",
        "kind": "mcp_stdio",
        "description": "Read and write files under directories you nominate.",
        "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "<PATH>"],
        },
        "enabled": False,
        "builtin": True,
    },
    {
        "slug": "mcp-git",
        "name": "MCP: Git",
        "kind": "mcp_stdio",
        "description": "Inspect and operate on a local git repository.",
        "config": {"command": "uvx", "args": ["mcp-server-git", "--repository", "<PATH>"]},
        "enabled": False,
        "builtin": True,
    },
    {
        "slug": "mcp-sqlite",
        "name": "MCP: SQLite",
        "kind": "mcp_stdio",
        "description": "Query a SQLite database.",
        "config": {"command": "uvx", "args": ["mcp-server-sqlite", "--db-path", "<PATH>"]},
        "enabled": False,
        "builtin": True,
    },
    {
        "slug": "mcp-fetch",
        "name": "MCP: Fetch",
        "kind": "mcp_stdio",
        "description": "Fetch a URL and convert it to markdown.",
        "config": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "enabled": False,
        "builtin": True,
    },
    {
        "slug": "mcp-github",
        "name": "MCP: GitHub",
        "kind": "mcp_stdio",
        "description": "Issues, pull requests and code search. Needs GITHUB_TOKEN.",
        "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${github}"},
        },
        "enabled": False,
        "builtin": True,
    },
]


def _resolve_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ``${name}`` in env values from the credential vault.

    Keeps real tokens out of the hub row while still letting an MCP subprocess
    receive them.
    """
    from ..core import secrets

    resolved = dict(config)
    env = dict(resolved.get("env") or {})
    for key, value in list(env.items()):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env[key] = secrets.get(value[2:-1]) or ""
    if env:
        resolved["env"] = env
    return resolved


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def seed_builtin_hubs() -> int:
    """Insert any missing built-in hub. Idempotent; never overwrites your edits."""
    added = 0
    async with session_scope() as s:
        existing = set((await s.execute(select(Hub.slug))).scalars().all())
        for spec in BUILTIN_HUBS:
            if spec["slug"] in existing:
                continue
            config = dict(spec["config"])
            if spec["slug"] == "workspace":
                config["path"] = str(paths.workspace_dir())
            s.add(
                Hub(
                    slug=spec["slug"],
                    name=spec["name"],
                    kind=spec["kind"],
                    description=spec["description"],
                    config=config,
                    enabled=spec["enabled"],
                    autostart=spec["enabled"],
                    builtin=True,
                    status="ready" if spec["kind"] not in MCP_KINDS else "stopped",
                )
            )
            added += 1
    if added:
        log.info("Seeded %d built-in hubs", added)
    return added


async def list_hubs(*, kind: str | None = None, enabled: bool | None = None) -> list[Hub]:
    async with session_scope() as s:
        stmt = select(Hub)
        if kind:
            stmt = stmt.where(Hub.kind == kind)
        if enabled is not None:
            stmt = stmt.where(Hub.enabled.is_(enabled))
        return list((await s.execute(stmt.order_by(Hub.name))).scalars().all())


async def get_hub(id_or_slug: str) -> Hub | None:
    async with session_scope() as s:
        return (
            await s.execute(select(Hub).where(or_(Hub.id == id_or_slug, Hub.slug == id_or_slug)))
        ).scalar_one_or_none()


async def create_hub(
    *,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    description: str = "",
    slug: str = "",
    enabled: bool = True,
    autostart: bool = False,
) -> Hub:
    if kind not in HUB_KINDS:
        raise ValidationFailed(f"Unknown hub kind '{kind}'. One of: {', '.join(sorted(HUB_KINDS))}")

    config = config or {}
    required = {
        "mcp_stdio": ["command"],
        "mcp_http": ["url"],
        "folder": ["path"],
    }.get(kind, [])
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValidationFailed(f"A '{kind}' hub needs: {', '.join(missing)}")

    slug = slug or slugify(name)
    async with session_scope() as s:
        if (await s.execute(select(Hub).where(Hub.slug == slug))).scalar_one_or_none():
            slug = f"{slug}-{int(time.time()) % 10000}"
        hub = Hub(
            slug=slug,
            name=name,
            kind=kind,
            description=description,
            config=config,
            enabled=enabled,
            autostart=autostart,
            status="stopped" if kind in MCP_KINDS else "ready",
        )
        s.add(hub)
        await s.flush()
        return hub


async def update_hub(id_or_slug: str, **fields: Any) -> Hub:
    async with session_scope() as s:
        hub = (
            await s.execute(select(Hub).where(or_(Hub.id == id_or_slug, Hub.slug == id_or_slug)))
        ).scalar_one_or_none()
        if hub is None:
            raise NotFound(f"No hub '{id_or_slug}'")
        for key, value in fields.items():
            if key in {"id", "slug", "builtin", "created_at"} or not hasattr(hub, key):
                continue
            setattr(hub, key, value)
        await s.flush()
        return hub


async def delete_hub(id_or_slug: str) -> bool:
    hub = await get_hub(id_or_slug)
    if hub is None:
        return False
    if hub.kind in MCP_KINDS:
        await manager.stop(hub.slug)
    async with session_scope() as s:
        await s.execute(sa_delete(Hub).where(Hub.id == hub.id))
    return True


async def test_hub(id_or_slug: str) -> dict[str, Any]:
    """Try the connection and report, without changing stored state."""
    hub = await get_hub(id_or_slug)
    if hub is None:
        raise NotFound(f"No hub '{id_or_slug}'")

    started = time.perf_counter()
    try:
        if hub.kind in MCP_KINDS:
            client = await open_client(hub.kind, _resolve_secrets(hub.config), hub.slug)
            try:
                tools = await client.list_tools()
            finally:
                await client.close()
            return {
                "ok": True,
                "detail": f"Connected to {client.server_info.get('name', hub.slug)}",
                "tool_count": len(tools),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        if hub.kind == "folder":
            from pathlib import Path

            path = Path(str(hub.config.get("path", ""))).expanduser()
            ok = path.is_dir()
            return {
                "ok": ok,
                "detail": str(path) if ok else f"{path} is not a directory",
                "tool_count": 0,
                "latency_ms": 0,
            }
        if hub.kind == "websearch":
            from . import websearch

            backends = [b for b in websearch.available_backends() if b["configured"]]
            return {
                "ok": bool(backends),
                "detail": ", ".join(b["label"] for b in backends) or "no backend available",
                "tool_count": 2,
                "latency_ms": 0,
            }
        return {"ok": True, "detail": "No test defined for this kind", "tool_count": 0, "latency_ms": 0}
    except Exception as exc:  # noqa: BLE001 - a test that raises is a failed test, not a crash
        return {
            "ok": False,
            "detail": str(exc)[:400],
            "tool_count": 0,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------


class MCPManager:
    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def _mark(self, slug: str, status: str, detail: str = "", **fields: Any) -> None:
        try:
            await update_hub(slug, status=status, status_detail=detail[:1000], **fields)
        except NotFound:
            pass
        bus.publish(Topic.HUB_STATUS, hub=slug, status=status, detail=detail[:300], **fields)

    async def start(self, id_or_slug: str) -> Hub:
        hub = await get_hub(id_or_slug)
        if hub is None:
            raise NotFound(f"No hub '{id_or_slug}'")
        if hub.kind not in MCP_KINDS:
            return hub
        if not hub.enabled:
            await self._mark(hub.slug, "disabled", "Hub is disabled")
            return hub

        async with self._lock:
            if hub.slug in self._clients:
                return hub
            await self._mark(hub.slug, "starting")
            try:
                client = await open_client(hub.kind, _resolve_secrets(hub.config), hub.slug)
                tools = await client.list_tools()
            except Exception as exc:  # noqa: BLE001 - a broken hub is a status, not a crash
                log.warning("Hub %s failed to start: %s", hub.slug, exc)
                await self._mark(hub.slug, "error", str(exc))
                return await get_hub(hub.slug) or hub

            self._clients[hub.slug] = client
            self._tools[hub.slug] = tools

        await self._mark(
            hub.slug,
            "ready",
            f"{client.server_info.get('name', hub.slug)} -- {len(tools)} tools",
            tool_count=len(tools),
            tools=[t["name"] for t in tools],
            last_ok=now(),
        )
        log.info("Hub %s ready with %d tools", hub.slug, len(tools))
        return await get_hub(hub.slug) or hub

    async def stop(self, id_or_slug: str) -> Hub | None:
        hub = await get_hub(id_or_slug)
        slug = hub.slug if hub else id_or_slug
        async with self._lock:
            client = self._clients.pop(slug, None)
            self._tools.pop(slug, None)
        if client is not None:
            await client.close()
            await self._mark(slug, "stopped")
        return hub

    async def restart(self, id_or_slug: str) -> Hub:
        await self.stop(id_or_slug)
        return await self.start(id_or_slug)

    async def start_all(self) -> int:
        hubs = await list_hubs(enabled=True)
        started = 0
        for hub in hubs:
            if hub.kind in MCP_KINDS and hub.autostart:
                try:
                    await self.start(hub.slug)
                    started += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("Autostart failed for %s: %s", hub.slug, exc)
        return started

    def tool_specs(self) -> list[dict[str, Any]]:
        """MCP tools, namespaced so two hubs can expose the same tool name."""
        out: list[dict[str, Any]] = []
        for slug, tools in self._tools.items():
            for tool in tools:
                out.append(
                    {
                        "name": f"{slug}__{tool['name']}",
                        "description": f"[{slug}] {tool['description']}"[:1000],
                        "parameters": tool["parameters"],
                        "hub": slug,
                    }
                )
        return out

    async def call(self, namespaced: str, arguments: dict[str, Any] | None = None) -> str:
        slug, _, tool = namespaced.partition("__")
        client = self._clients.get(slug)
        if client is None:
            available = ", ".join(sorted(self._clients)) or "none running"
            raise NotFound(f"Hub '{slug}' is not running (running: {available})")
        return await client.call_tool(tool, arguments or {})

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            slug: {
                "connected": client.connected,
                "server": client.server_info,
                "tools": len(self._tools.get(slug, [])),
            }
            for slug, client in self._clients.items()
        }

    def running(self) -> list[str]:
        return sorted(self._clients)

    async def shutdown(self) -> None:
        for slug in list(self._clients):
            await self.stop(slug)


manager = MCPManager()
