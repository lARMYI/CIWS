"""End-to-end agent loop tests, driven by a scripted provider.

No API key, no network. A fake provider replays a fixed script of turns, which
lets the tests assert the things that actually break in an agent runtime and are
otherwise only observable against a live model:

* tool calls are dispatched, and their results come back in ONE tool turn
* the loop terminates when the model stops asking for tools
* a tool that raises becomes an observation, not a crashed run
* runs, steps and tool calls are all persisted for the trace
* cancellation stops the loop
* the step ceiling holds
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ciws.agents import presets, runtime
from ciws.gateway.registry import gateway
from ciws.gateway.types import (
    ChatRequest,
    ModelInfo,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from ciws.tools.registry import Risk, ToolOutput, registry


class ScriptedProvider:
    """A provider that replays turns. Each turn is (text, [(tool, args), ...])."""

    id = "test"
    label = "Scripted"
    requires_key = False
    local = True
    env_hint = ""
    supports_embeddings = False

    def __init__(self, script: list[tuple[str, list[tuple[str, dict]]]]) -> None:
        self.script = script
        self.turn = 0
        self.requests: list[ChatRequest] = []

    def configured(self) -> bool:
        return True

    def api_key(self) -> str | None:
        return "test"

    async def models(self, refresh: bool = False) -> list[ModelInfo]:
        return await self.list_models()

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="test/scripted", provider="test", name="scripted",
                display_name="Scripted", context_window=128_000, max_output=8000,
            )
        ]

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        index = min(self.turn, len(self.script) - 1)
        text, calls = self.script[index]
        self.turn += 1

        yield StreamEvent(type=StreamEventType.START, model=request.model)
        for piece in text.split(" "):
            if piece:
                yield StreamEvent(type=StreamEventType.TEXT, text=piece + " ")
        for name, args in calls:
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL, tool_call=ToolCall(name=name, arguments=args)
            )
        usage = Usage(input_tokens=100, output_tokens=20, cost_usd=0.001)
        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE, finish_reason="stop", usage=usage, model=request.model
        )

    async def health(self):
        from ciws.gateway.types import HealthStatus

        return HealthStatus(provider=self.id, ok=True, detail="scripted", model_count=1)


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted provider as the whole gateway for one test."""

    def install(script):
        provider = ScriptedProvider(script)
        monkeypatch.setitem(gateway.providers(), "test", provider)
        gateway._models = {m.id: m for m in [
            ModelInfo(id="test/scripted", provider="test", name="scripted", context_window=128_000)
        ]}
        gateway._models_loaded_at = 1e12  # keep the cache from refreshing mid-test
        return provider

    return install


@pytest.fixture(autouse=True)
def tools_loaded():
    from ciws.tools.registry import load_builtin_tools

    load_builtin_tools()


async def _agent(slug: str = "analyst"):
    await presets.seed_presets()
    return await presets.get_agent(slug)


async def test_single_turn_answer(scripted):
    scripted([("The answer is 42.", [])])
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="What is the answer?", model_override="test/scripted"
    )
    assert result.ok, result.error
    assert "42" in result.content
    assert result.steps == 1


async def test_tool_call_round_trip(scripted):
    calls: list[dict] = []

    @registry.tool(
        "probe_tool",
        description="A tool that records its arguments.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        risk=Risk.SAFE,
    )
    async def probe_tool(value: str = "") -> ToolOutput:
        calls.append({"value": value})
        return ToolOutput.text(f"probed:{value}")

    provider = scripted(
        [
            ("Let me check.", [("probe_tool", {"value": "alpha"})]),
            ("The probe returned alpha.", []),
        ]
    )
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="Use the probe", model_override="test/scripted"
    )

    assert result.ok, result.error
    assert calls == [{"value": "alpha"}]
    assert result.steps == 2

    # The tool result must come back as exactly one tool turn -- splitting them
    # teaches models to stop batching parallel calls.
    second = provider.requests[1]
    tool_turns = [m for m in second.messages if m.role == "tool"]
    assert len(tool_turns) == 1
    assert tool_turns[0].tool_results[0].content == "probed:alpha"
    registry.unregister("probe_tool")


async def test_parallel_tool_calls_return_in_one_turn(scripted):
    @registry.tool(
        "echo_tool",
        description="Echoes.",
        parameters={"type": "object", "properties": {"n": {"type": "string"}}},
        risk=Risk.SAFE,
    )
    async def echo_tool(n: str = "") -> ToolOutput:
        return ToolOutput.text(f"echo-{n}")

    provider = scripted(
        [
            (
                "Checking three things.",
                [("echo_tool", {"n": "1"}), ("echo_tool", {"n": "2"}), ("echo_tool", {"n": "3"})],
            ),
            ("All three came back.", []),
        ]
    )
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="Check three", model_override="test/scripted"
    )
    assert result.ok
    tool_turns = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert len(tool_turns) == 1
    assert len(tool_turns[0].tool_results) == 3
    assert {r.content for r in tool_turns[0].tool_results} == {"echo-1", "echo-2", "echo-3"}
    registry.unregister("echo_tool")


async def test_failing_tool_becomes_an_observation(scripted):
    @registry.tool(
        "broken_tool",
        description="Always fails.",
        parameters={"type": "object", "properties": {}},
        risk=Risk.SAFE,
    )
    async def broken_tool() -> ToolOutput:
        raise RuntimeError("the thing exploded")

    provider = scripted(
        [
            ("Trying it.", [("broken_tool", {})]),
            ("That tool failed, so here is what I can say instead.", []),
        ]
    )
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="Try the broken tool", model_override="test/scripted"
    )

    # The run completes; the failure is visible to the model, not fatal.
    assert result.ok, result.error
    tool_turns = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert tool_turns[0].tool_results[0].is_error
    assert "exploded" in tool_turns[0].tool_results[0].content
    registry.unregister("broken_tool")


async def test_unknown_tool_is_reported_not_raised(scripted):
    provider = scripted(
        [("Calling something imaginary.", [("no_such_tool", {})]), ("Understood.", [])]
    )
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="go", model_override="test/scripted"
    )
    assert result.ok
    results = [m for m in provider.requests[1].messages if m.role == "tool"][0].tool_results
    assert results[0].is_error
    assert "No tool named" in results[0].content


async def test_run_trace_is_persisted(scripted):
    @registry.tool(
        "traced_tool",
        description="Traced.",
        parameters={"type": "object", "properties": {}},
        risk=Risk.SAFE,
    )
    async def traced_tool() -> ToolOutput:
        return ToolOutput.text("ok")

    scripted([("Working.", [("traced_tool", {})]), ("Done.", [])])
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="trace me", model_override="test/scripted"
    )

    trace = await runtime.get_run(result.run_id)
    assert trace is not None
    assert trace["status"] == "completed"
    assert len(trace["steps"]) == 2
    assert any(c["tool"] == "traced_tool" and c["ok"] for c in trace["tool_calls"])
    assert trace["input_tokens"] > 0
    registry.unregister("traced_tool")


async def test_step_ceiling_stops_a_runaway(scripted):
    @registry.tool(
        "loop_tool",
        description="Loops.",
        parameters={"type": "object", "properties": {}},
        risk=Risk.SAFE,
    )
    async def loop_tool() -> ToolOutput:
        return ToolOutput.text("again")

    # A model that only ever asks for another tool call.
    scripted([("Again.", [("loop_tool", {})])])
    await presets.seed_presets()  # the fixture drops tables, so seed before editing
    await presets.update_agent("analyst", max_steps=4)
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="loop forever", model_override="test/scripted"
    )
    assert result.steps == 4
    assert "Stopped after 4 steps" in result.error
    await presets.update_agent("analyst", max_steps=30)
    registry.unregister("loop_tool")


async def test_cancellation_stops_the_run(scripted):
    import asyncio

    @registry.tool(
        "slow_tool",
        description="Slow.",
        parameters={"type": "object", "properties": {}},
        risk=Risk.SAFE,
    )
    async def slow_tool() -> ToolOutput:
        await asyncio.sleep(0.4)
        return ToolOutput.text("finally")

    scripted([("Working.", [("slow_tool", {})])])

    async def cancel_soon(run_id_holder: dict) -> None:
        for _ in range(60):
            await asyncio.sleep(0.05)
            active = runtime.active_runs()
            if active:
                runtime.cancel(active[0])
                run_id_holder["id"] = active[0]
                return

    holder: dict = {}
    canceller = asyncio.create_task(cancel_soon(holder))
    result = await runtime.run_agent(
        agent_slug="analyst", prompt="start something long", model_override="test/scripted"
    )
    await canceller
    assert "Cancelled" in result.error or result.steps <= 2
    registry.unregister("slow_tool")


async def test_agent_tool_grants_are_enforced(scripted):
    """The scout agent must not see tools it was not granted."""
    await presets.seed_presets()
    scout = await presets.get_agent("scout")
    granted = {spec.name for spec in registry.specs(scout.tools)}
    assert "web_search" in granted
    assert "python" not in granted
    assert "file_write" not in granted
    assert "memory_write" not in granted


async def test_memory_context_is_injected(scripted):
    from ciws.memory import store

    await store.remember(
        "The user's deploy window is Friday afternoon.", kind="procedure", importance=0.9
    )
    provider = scripted([("Noted.", [])])
    await runtime.run_agent(
        agent_slug="analyst",
        prompt="When is the deploy window?",
        model_override="test/scripted",
    )
    system = provider.requests[0].system
    assert "deploy window is Friday" in system
