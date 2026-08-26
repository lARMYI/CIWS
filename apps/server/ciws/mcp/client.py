"""A dependency-free MCP client.

MCP is JSON-RPC 2.0 over one of two transports, and both are small enough to
implement directly. Doing so keeps the hub installable with no extra packages
and, more usefully, means a protocol quirk is something to fix here rather than
wait on upstream for.

Two transports:

* **stdio** -- launch the server as a subprocess, newline-delimited JSON on its
  stdin/stdout. A reader task resolves pending futures by id; stderr is drained
  into the log so a server that dies explains itself.
* **HTTP** -- POST JSON-RPC. Servers may answer with either plain JSON or an SSE
  stream, so the response content type decides how the reply is read.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from typing import Any

from ..core.errors import ToolError
from ..core.logging import get_logger

log = get_logger("mcp.client")

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "ciws", "version": "0.1.0"}
DEFAULT_TIMEOUT = 60.0


class MCPClient:
    """Shared JSON-RPC bookkeeping for both transports."""

    def __init__(self, name: str = "mcp", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.name = name
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self._next_id = 0
        self._connected = False

    def _rpc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    async def connect(self) -> dict[str, Any]:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                "clientInfo": CLIENT_INFO,
            },
        )
        self.server_info = result.get("serverInfo", {}) if isinstance(result, dict) else {}
        self.capabilities = result.get("capabilities", {}) if isinstance(result, dict) else {}
        with contextlib.suppress(Exception):
            await self._notify("notifications/initialized")
        self._connected = True
        return self.server_info

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            result = await self._request("tools/list")
        except Exception as exc:  # noqa: BLE001 - a server may expose no tools at all
            log.debug("%s: tools/list failed: %s", self.name, exc)
            return []
        tools = (result or {}).get("tools") or []
        return [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for t in tools
            if t.get("name")
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if not isinstance(result, dict):
            return str(result)

        parts: list[str] = []
        for block in result.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image":
                parts.append(f"[image: {block.get('mimeType', 'image')}]")
            elif btype == "resource":
                resource = block.get("resource") or {}
                parts.append(resource.get("text") or f"[resource: {resource.get('uri', '')}]")
            else:
                parts.append(json.dumps(block, ensure_ascii=False))

        text = "\n".join(p for p in parts if p) or json.dumps(
            result.get("structuredContent", result), ensure_ascii=False
        )
        if result.get("isError"):
            raise ToolError(name, text[:2000])
        return text

    async def list_resources(self) -> list[dict[str, Any]]:
        try:
            result = await self._request("resources/list")
            return (result or {}).get("resources") or []
        except Exception:  # noqa: BLE001
            return []

    async def read_resource(self, uri: str) -> str:
        result = await self._request("resources/read", {"uri": uri})
        contents = (result or {}).get("contents") or []
        return "\n".join(c.get("text", "") for c in contents if c.get("text"))

    async def list_prompts(self) -> list[dict[str, Any]]:
        try:
            result = await self._request("prompts/list")
            return (result or {}).get("prompts") or []
        except Exception:  # noqa: BLE001
            return []

    async def close(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected


class MCPStdioClient(MCPClient):
    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        name: str = "mcp",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(name, timeout)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None

    async def start(self) -> None:
        executable = shutil.which(self.command)
        if executable is None:
            raise ToolError(
                self.name,
                f"Command '{self.command}' is not on PATH. "
                f"Install it, or give the full path in the hub's configuration.",
            )

        self._process = await asyncio.create_subprocess_exec(
            executable,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
            cwd=self.cwd,
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr = asyncio.create_task(self._drain_stderr())

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue  # servers sometimes emit banner text on stdout
                msg_id = message.get("id")
                if msg_id is None:
                    continue  # a notification from the server; nothing waiting on it
                future = self._pending.pop(int(msg_id), None)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message["error"] or {}
                    future.set_exception(
                        ToolError(self.name, f"{error.get('code', '')}: {error.get('message', '')}")
                    )
                else:
                    future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: reader stopped: %s", self.name, exc)
        finally:
            # A dead server must not leave callers waiting forever.
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ToolError(self.name, "MCP server closed the connection"))
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        assert self._process and self._process.stderr
        with contextlib.suppress(asyncio.CancelledError, Exception):
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    log.debug("[%s] %s", self.name, text[:400])

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise ToolError(self.name, "MCP server is not running")
        if self._process.returncode is not None:
            raise ToolError(self.name, f"MCP server exited with code {self._process.returncode}")
        self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._process is None:
            await self.start()
        rpc_id = self._rpc_id()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rpc_id] = future
        await self._send({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(rpc_id, None)
            raise ToolError(self.name, f"'{method}' timed out after {self.timeout:.0f}s") from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def close(self) -> None:
        await super().close()
        for task in (self._reader, self._stderr):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        self._process = None


class MCPHttpClient(MCPClient):
    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        name: str = "mcp",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(name, timeout)
        self.url = url
        self.headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self._session_id = ""

    def _all_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    async def _post(self, payload: dict[str, Any]) -> Any:
        from ..gateway.base import http_client

        client = http_client(f"mcp-{self.name}", timeout=self.timeout)
        response = await client.post(self.url, json=payload, headers=self._all_headers())
        if response.status_code >= 400:
            raise ToolError(self.name, f"HTTP {response.status_code}: {response.text[:300]}")

        session = response.headers.get("mcp-session-id")
        if session:
            self._session_id = session

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Streamable HTTP: the reply is the last data: frame carrying our id.
            for raw in response.text.splitlines():
                if not raw.startswith("data:"):
                    continue
                try:
                    message = json.loads(raw[5:].strip())
                except json.JSONDecodeError:
                    continue
                if message.get("id") == payload.get("id"):
                    return message
            raise ToolError(self.name, "No matching reply in the SSE stream")
        if not response.content:
            return {}
        return response.json()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        message = await self._post(
            {"jsonrpc": "2.0", "id": self._rpc_id(), "method": method, "params": params or {}}
        )
        if isinstance(message, dict) and "error" in message:
            error = message["error"] or {}
            raise ToolError(self.name, f"{error.get('code', '')}: {error.get('message', '')}")
        return (message or {}).get("result") if isinstance(message, dict) else message

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        with contextlib.suppress(Exception):
            await self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})


async def open_client(kind: str, config: dict[str, Any], name: str) -> MCPClient:
    """Build and connect a client from a hub's stored configuration."""
    timeout = float(config.get("timeout", DEFAULT_TIMEOUT))
    if kind == "mcp_stdio":
        client = MCPStdioClient(
            command=str(config.get("command", "")),
            args=[str(a) for a in (config.get("args") or [])],
            env={str(k): str(v) for k, v in (config.get("env") or {}).items()},
            cwd=config.get("cwd") or None,
            name=name,
            timeout=timeout,
        )
        await client.start()
    elif kind == "mcp_http":
        client = MCPHttpClient(
            url=str(config.get("url", "")),
            headers={str(k): str(v) for k, v in (config.get("headers") or {}).items()},
            name=name,
            timeout=timeout,
        )
    else:
        raise ToolError(name, f"'{kind}' is not an MCP transport")

    await client.connect()
    return client
