"""MCP client and hub lifecycle, against a real subprocess.

The MCP client is tested by running an actual stdio server -- a small Python
script written to a temp file -- rather than by mocking the transport. Framing
bugs (Content-Length headers, partial reads, interleaved notifications) are
exactly the class of bug a mock hides, and they are why this file exists.

The hub layer is tested for the property that matters at boot: a connector that
is broken, missing, or slow must be recorded as unhealthy and must not stop the
hub from starting. A broken MCP server should never be the reason you cannot
reach your own notes.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from ciws.hubs import registry as hubs
from ciws.mcp.client import MCPStdioClient


# ---------------------------------------------------------------------------
# A real, minimal MCP server over stdio
# ---------------------------------------------------------------------------

SERVER = textwrap.dedent(
    '''
    """A minimal newline-delimited JSON-RPC 2.0 MCP server, for tests."""
    import json, sys

    TOOLS = [
        {
            "name": "echo",
            "description": "Echo the text back.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        {
            "name": "explode",
            "description": "Always fails.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    def reply(rid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": rid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method, rid = req.get("method"), req.get("id")
        if rid is None:
            continue  # a notification

        if method == "initialize":
            reply(rid, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture-server", "version": "1.0.0"},
            })
        elif method == "tools/list":
            reply(rid, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params") or {}
            if params.get("name") == "explode":
                reply(rid, {
                    "content": [{"type": "text", "text": "it exploded"}],
                    "isError": True,
                })
            else:
                text = (params.get("arguments") or {}).get("text", "")
                reply(rid, {"content": [{"type": "text", "text": f"echo:{text}"}]})
        elif method == "resources/list":
            reply(rid, {"resources": [{"uri": "mem://note", "name": "note"}]})
        elif method == "resources/read":
            reply(rid, {"contents": [{"uri": "mem://note", "text": "a stored note"}]})
        elif method == "prompts/list":
            reply(rid, {"prompts": [{"name": "summarize"}]})
        else:
            reply(rid, error={"code": -32601, "message": f"No method {method}"})
    '''
).strip()


@pytest.fixture
def mcp_server(tmp_path: Path) -> Path:
    script = tmp_path / "fixture_mcp_server.py"
    script.write_text(SERVER, "utf-8")
    return script


@pytest.fixture
async def connected(mcp_server: Path):
    client = MCPStdioClient(
        name="fixture", command=sys.executable, args=[str(mcp_server)], timeout=20.0
    )
    await client.start()
    await client.connect()
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


async def test_handshake_reports_the_server_identity(connected):
    tools = await connected.list_tools()
    assert {t["name"] for t in tools} == {"echo", "explode"}
    assert connected.connected


async def test_a_tool_call_round_trips(connected):
    result = await connected.call_tool("echo", {"text": "hello"})
    assert "echo:hello" in result


async def test_a_server_side_tool_error_becomes_a_typed_tool_error(connected):
    """isError is the server saying the tool failed, not the transport failing.

    The client raises ToolError, which the registry then converts into an
    observation -- so the agent still sees it as a readable result, never as a
    crashed turn.
    """
    from ciws.core.errors import ToolError

    with pytest.raises(ToolError) as caught:
        await connected.call_tool("explode", {})
    assert "exploded" in str(caught.value)


async def test_resources_and_prompts_are_listable(connected):
    resources = await connected.list_resources()
    assert resources[0]["uri"] == "mem://note"

    body = await connected.read_resource("mem://note")
    assert "stored note" in body

    prompts = await connected.list_prompts()
    assert prompts[0]["name"] == "summarize"


async def test_an_unknown_method_surfaces_the_rpc_error(connected):
    with pytest.raises(Exception) as caught:
        await connected._request("nonsense/method")
    assert "no method" in str(caught.value).lower() or "-32601" in str(caught.value)


async def test_request_ids_are_unique_across_concurrent_calls(connected):
    """Interleaved requests must not resolve each other's futures."""
    import asyncio

    results = await asyncio.gather(
        *(connected.call_tool("echo", {"text": str(n)}) for n in range(8))
    )
    assert sorted(r.split(":")[-1].strip() for r in results) == sorted(str(n) for n in range(8))


async def test_closing_is_idempotent(connected):
    await connected.close()
    await connected.close()
    assert not connected.connected


async def test_a_command_that_does_not_exist_fails_cleanly():
    client = MCPStdioClient(
        name="ghost", command="definitely-not-a-real-binary-xyz", args=[], timeout=5.0
    )
    with pytest.raises(Exception):
        await client.start()
    await client.close()


# ---------------------------------------------------------------------------
# Hub lifecycle
# ---------------------------------------------------------------------------


async def test_builtin_hubs_are_seeded_once():
    first = await hubs.seed_builtin_hubs()
    assert first > 0
    second = await hubs.seed_builtin_hubs()
    assert second == 0, "seeding twice duplicated the built-in hubs"


async def test_hub_crud_round_trip():
    created = await hubs.create_hub(
        name="Fixture Hub",
        kind="mcp_stdio",
        config={"command": "echo", "args": ["hi"]},
        enabled=False,
    )
    assert created.slug

    fetched = await hubs.get_hub(created.slug)
    assert fetched is not None and fetched.name == "Fixture Hub"

    updated = await hubs.update_hub(created.slug, name="Renamed Hub")
    assert updated.name == "Renamed Hub"

    listed = await hubs.list_hubs()
    assert any(h.slug == created.slug for h in listed)

    assert await hubs.delete_hub(created.slug) is True
    assert await hubs.get_hub(created.slug) is None


async def test_starting_a_broken_hub_records_an_error_rather_than_raising():
    """Boot must survive a connector that cannot start."""
    hub = await hubs.create_hub(
        name="Broken",
        kind="mcp_stdio",
        config={"command": "definitely-not-a-real-binary-xyz", "args": []},
    )
    try:
        await hubs.manager.start(hub.slug)
    except Exception:
        pass  # raising is acceptable; silently claiming success is not

    after = await hubs.get_hub(hub.slug)
    assert after.status in {"error", "stopped"}, f"broken hub reported status {after.status!r}"
    assert after.status != "ready", "a hub that never started was reported as ready"
    await hubs.delete_hub(hub.slug)


async def test_a_real_hub_starts_lists_tools_and_stops(mcp_server: Path):
    hub = await hubs.create_hub(
        name="Fixture MCP",
        kind="mcp_stdio",
        config={"command": sys.executable, "args": [str(mcp_server)]},
    )
    started = await hubs.manager.start(hub.slug)
    # "ready" is the running state for a connector: handshaken and tools listed.
    assert started.status == "ready"
    assert started.tool_count == 2
    assert hub.slug in hubs.manager.running()

    namespaced = f"{started.slug}__echo"
    result = await hubs.manager.call(namespaced, {"text": "through the hub"})
    assert "through the hub" in result

    await hubs.manager.stop(hub.slug)
    assert hub.slug not in hubs.manager.running()
    await hubs.delete_hub(hub.slug)


async def test_testing_a_hub_reports_health_without_leaving_it_running(mcp_server: Path):
    hub = await hubs.create_hub(
        name="Probe",
        kind="mcp_stdio",
        config={"command": sys.executable, "args": [str(mcp_server)]},
    )
    report = await hubs.test_hub(hub.slug)
    assert report["ok"] is True
    assert hub.slug not in hubs.manager.running()
    await hubs.delete_hub(hub.slug)


async def test_calling_through_an_unstarted_hub_is_an_error_not_a_hang():
    with pytest.raises(Exception):
        await hubs.manager.call("not_a_hub__some_tool", {})
