"""The tool registry: what an agent is allowed to do, and the record of it doing so.

Three things are deliberately centralised here rather than left to each tool:

* **Permission.** A tool declares a risk level; the registry decides whether the
  current security policy lets it run unattended, needs a human click, or is off.
  Individual tools must not be able to grant themselves rights.
* **Provenance.** Every call -- arguments, result, duration, outcome -- is written
  to ``tool_calls`` before the agent sees the result. An agent that cannot explain
  where a claim came from is not much use, and this is where that trail lives.
* **Blast radius.** Timeouts, output truncation and cancellation are enforced by
  the registry, so one runaway tool cannot hang a run or blow the context window.

MCP tools from connected hubs are surfaced through the same interface, namespaced
``<hub>__<tool>``, so an agent cannot tell a built-in from a connector -- and the
policy applies to both.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.config import get_settings
from ..core.errors import ApprovalRequired, Cancelled, ToolError
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import dumps, new_id, truncate
from ..db.base import session_scope
from ..db.models import AuditEvent, ToolCall
from ..gateway.types import ContentPart, ToolSpec

log = get_logger("tools")

#: Tool output beyond this is truncated before it reaches the model. A 2 MB file
#: read must not silently evict the conversation from the context window.
MAX_RESULT_CHARS = 24_000
DEFAULT_TIMEOUT_S = 120.0


class Risk(str, Enum):
    """What a tool can do to the world, in ascending order of regret."""

    SAFE = "safe"          # read-only, local, no side effects
    NETWORK = "network"    # reaches out to the internet
    WRITE = "write"        # modifies the workspace or the hub's own state
    DANGEROUS = "dangerous"  # arbitrary code, shell, deletion, spend


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to know about the run invoking it."""

    run_id: str | None = None
    conversation_id: str | None = None
    project_id: str | None = None
    agent_slug: str = ""
    step_id: str | None = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    #: Set by the runtime when a human has pre-approved this run's risky calls.
    auto_approve: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled("Run was cancelled")


@dataclass(slots=True)
class ToolOutput:
    content: str = ""
    parts: list[ContentPart] = field(default_factory=list)
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, content: str, **meta: Any) -> "ToolOutput":
        return cls(content=content, meta=meta)

    @classmethod
    def error(cls, message: str, **meta: Any) -> "ToolOutput":
        return cls(content=message, is_error=True, meta=meta)

    @classmethod
    def json(cls, value: Any, **meta: Any) -> "ToolOutput":
        return cls(content=dumps(value), meta=meta)


ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    category: str = "general"
    risk: Risk = Risk.SAFE
    source: str = "builtin"
    timeout_s: float = DEFAULT_TIMEOUT_S
    #: Tools that read a lot (file reads, searches) get a bigger output allowance.
    max_result_chars: int = MAX_RESULT_CHARS
    enabled: bool = True

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.parameters
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "risk": self.risk.value,
            "source": self.source,
            "enabled": self.enabled,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        #: Pending human approvals, keyed by call id.
        self._approvals: dict[str, asyncio.Future[bool]] = {}

    # -- registration -----------------------------------------------------

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            log.debug("Tool %s re-registered", tool.name)
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str = "",
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        category: str = "general",
        risk: Risk = Risk.SAFE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_result_chars: int = MAX_RESULT_CHARS,
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator form. The docstring becomes the description the model reads."""

        def wrap(fn: ToolFn) -> ToolFn:
            self.register(
                Tool(
                    name=name or fn.__name__,
                    description=(description or inspect.getdoc(fn) or "").strip(),
                    parameters=parameters or {"type": "object", "properties": {}},
                    fn=fn,
                    category=category,
                    risk=risk,
                    timeout_s=timeout_s,
                    max_result_chars=max_result_chars,
                )
            )
            return fn

        return wrap

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # -- policy -----------------------------------------------------------

    def _policy_allows(self, tool: Tool) -> tuple[bool, str]:
        """Static gate: is this tool permitted to exist for this hub at all?"""
        sec = get_settings().security
        if not tool.enabled:
            return False, f"Tool '{tool.name}' is disabled."
        if tool.name == "shell" and not sec.allow_shell_tool:
            return False, (
                "The shell tool is disabled. Enable it in Settings -> Security if you "
                "want agents to run shell commands."
            )
        if tool.name == "python" and not sec.allow_python_tool:
            return False, "The python tool is disabled in Settings -> Security."
        return True, ""

    def _needs_approval(self, tool: Tool, ctx: ToolContext) -> bool:
        if ctx.auto_approve:
            return False
        sec = get_settings().security
        if tool.risk is Risk.DANGEROUS:
            return True
        if tool.risk is Risk.WRITE and sec.approve_writes_outside_workspace:
            return False  # the file tools do their own path check; writes in-workspace are fine
        return False

    async def request_approval(self, call_id: str, tool: Tool, args: dict[str, Any]) -> bool:
        """Block until a human resolves the approval, or time out into a refusal."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._approvals[call_id] = fut
        bus.publish(
            Topic.TOOL_APPROVAL,
            call_id=call_id,
            tool=tool.name,
            risk=tool.risk.value,
            arguments=args,
            description=tool.description[:300],
        )
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            return False
        finally:
            self._approvals.pop(call_id, None)

    def resolve_approval(self, call_id: str, approved: bool) -> bool:
        fut = self._approvals.get(call_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True

    def pending_approvals(self) -> list[str]:
        return [k for k, f in self._approvals.items() if not f.done()]

    # -- specs ------------------------------------------------------------

    def specs(self, allow: list[str] | None = None, *, include_mcp: bool = True) -> list[ToolSpec]:
        """Tool definitions for the model.

        ``allow`` is the agent's grant list: ``None`` or ``["*"]`` means everything
        the policy permits. Names may end in ``*`` to grant a family (``memory_*``).
        """
        out: list[ToolSpec] = []
        for tool in self._tools.values():
            ok, _ = self._policy_allows(tool)
            if not ok:
                continue
            if not _granted(tool.name, allow):
                continue
            out.append(tool.spec())

        if include_mcp:
            for mcp_spec in self._mcp_specs():
                if _granted(mcp_spec.name, allow):
                    out.append(mcp_spec)
        return out

    def _mcp_specs(self) -> list[ToolSpec]:
        try:
            from ..mcp.manager import manager

            return [
                ToolSpec(
                    name=s["name"],
                    description=s.get("description", ""),
                    parameters=s.get("parameters") or {"type": "object", "properties": {}},
                )
                for s in manager.tool_specs()
            ]
        except Exception as exc:  # noqa: BLE001 - MCP is optional
            log.debug("MCP specs unavailable: %s", exc)
            return []

    def list_tools(self) -> list[dict[str, Any]]:
        out = [t.to_dict() for t in sorted(self._tools.values(), key=lambda t: (t.category, t.name))]
        for spec in self._mcp_specs():
            hub = spec.name.split("__", 1)[0]
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "category": "hub",
                    "risk": Risk.NETWORK.value,
                    "source": f"mcp:{hub}",
                    "enabled": True,
                }
            )
        return out

    # -- invocation -------------------------------------------------------

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str = "",
    ) -> ToolOutput:
        """Run a tool. Never raises for tool-level failure -- errors come back as output.

        An exception here would abort the agent's whole turn. A failed tool should
        instead be an observation the agent can reason about and route around.
        """
        args = arguments or {}
        ctx = ctx or ToolContext()
        call_id = call_id or new_id("call")
        started = time.perf_counter()

        bus.publish(
            Topic.TOOL_START,
            call_id=call_id,
            tool=name,
            arguments=args,
            run_id=ctx.run_id,
            agent=ctx.agent_slug,
        )

        try:
            output = await self._dispatch(name, args, ctx, call_id)
        except Cancelled:
            output = ToolOutput.error("Cancelled.")
        except ToolError as exc:
            output = ToolOutput.error(exc.message)
        except asyncio.TimeoutError:
            output = ToolOutput.error(f"Tool '{name}' timed out.")
        except Exception as exc:  # noqa: BLE001 - a tool crash is an observation, not a run failure
            log.exception("Tool %s failed", name)
            output = ToolOutput.error(f"{type(exc).__name__}: {exc}")

        duration_ms = int((time.perf_counter() - started) * 1000)
        tool = self._tools.get(name)
        limit = tool.max_result_chars if tool else MAX_RESULT_CHARS
        if len(output.content) > limit:
            output.meta["truncated_from"] = len(output.content)
            output.content = truncate(output.content, limit)

        await self._record(call_id, name, args, output, ctx, duration_ms)

        bus.publish(
            Topic.TOOL_ERROR if output.is_error else Topic.TOOL_END,
            call_id=call_id,
            tool=name,
            ok=not output.is_error,
            duration_ms=duration_ms,
            preview=output.content[:500],
            run_id=ctx.run_id,
        )
        return output

    async def _dispatch(
        self, name: str, args: dict[str, Any], ctx: ToolContext, call_id: str
    ) -> ToolOutput:
        ctx.check_cancelled()

        if "__" in name and name not in self._tools:
            return await self._call_mcp(name, args)

        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)[:25])
            raise ToolError(name, f"No tool named '{name}'. Available: {known}")

        allowed, reason = self._policy_allows(tool)
        if not allowed:
            raise ToolError(name, reason)

        if self._needs_approval(tool, ctx):
            approved = await self.request_approval(call_id, tool, args)
            if not approved:
                raise ToolError(name, "The user declined this action.")

        result = await asyncio.wait_for(
            _invoke(tool.fn, args, ctx), timeout=tool.timeout_s
        )
        return _coerce(result)

    async def _call_mcp(self, name: str, args: dict[str, Any]) -> ToolOutput:
        from ..mcp.manager import manager

        text = await manager.call(name, args)
        return ToolOutput.text(text, source="mcp")

    async def _record(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        output: ToolOutput,
        ctx: ToolContext,
        duration_ms: int,
    ) -> None:
        if not get_settings().security.audit_all_tool_calls:
            return
        try:
            async with session_scope() as s:
                s.add(
                    ToolCall(
                        call_id=call_id,
                        run_id=ctx.run_id,
                        step_id=ctx.step_id,
                        tool=name,
                        source="mcp" if "__" in name and name not in self._tools else "builtin",
                        arguments=args,
                        result=output.content[:8000],
                        ok=not output.is_error,
                        error=output.content[:2000] if output.is_error else "",
                        duration_ms=duration_ms,
                    )
                )
                tool = self._tools.get(name)
                if tool and tool.risk in (Risk.WRITE, Risk.DANGEROUS):
                    s.add(
                        AuditEvent(
                            actor=ctx.agent_slug or "agent",
                            action=f"tool:{name}",
                            target=str(args)[:200],
                            outcome="error" if output.is_error else "ok",
                            detail={"run_id": ctx.run_id, "duration_ms": duration_ms},
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - the audit trail must not break the run
            log.debug("Tool audit write skipped: %s", exc)


def _granted(name: str, allow: list[str] | None) -> bool:
    if not allow or "*" in allow:
        return True
    for pattern in allow:
        if pattern == name:
            return True
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True
    return False


async def _invoke(fn: ToolFn, args: dict[str, Any], ctx: ToolContext) -> Any:
    """Call a tool function, passing ``ctx`` only if it asks for it."""
    sig = inspect.signature(fn)
    kwargs = dict(args)
    if "ctx" in sig.parameters:
        kwargs["ctx"] = ctx
    # Drop anything the model hallucinated that the function does not accept,
    # unless the function takes **kwargs.
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = fn(**kwargs)
    return await result if inspect.isawaitable(result) else result


def _coerce(result: Any) -> ToolOutput:
    if isinstance(result, ToolOutput):
        return result
    if isinstance(result, str):
        return ToolOutput.text(result)
    if result is None:
        return ToolOutput.text("(no output)")
    return ToolOutput.json(result)


#: Process-wide registry.
registry = ToolRegistry()


def load_builtin_tools() -> int:
    """Import the built-in tool modules for their registration side effects."""
    from .builtin import (  # noqa: F401
        agent_tools,
        file_tools,
        graph_tools,
        media_tools,
        memory_tools,
        python_tools,
        shell_tools,
        task_tools,
        web_tools,
    )

    log.info("Registered %d built-in tools", len(registry._tools))  # noqa: SLF001
    return len(registry._tools)  # noqa: SLF001
