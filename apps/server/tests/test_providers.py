"""Provider adapter tests, driven by recorded-shape fixtures.

**What these do and do not prove.** The fixtures are hand-authored to the
documented wire formats, not captured from live APIs -- this build has never had
a provider credential. So they prove that the adapters *build the right request*
and *parse a correctly-shaped response*; they do not prove the shapes are
current. ``scripts/record_fixture.py`` turns a real call into a fixture for
anyone who has a key, and these tests then replay it forever without spend.

The request-building half is the part that matters most and is fully verifiable
offline: the current Anthropic generation *rejects* ``temperature`` and
``budget_tokens`` with a 400 where older models require them, so a single
mistake in the quirk table is a hard failure on every call to that model. That
table is what the first half of this file pins.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ciws.gateway import catalog
from ciws.gateway.providers.anthropic import AnthropicProvider
from ciws.gateway.providers.openai_compat import OpenAIProvider
from ciws.gateway.types import ChatMessage, ChatRequest, StreamEventType, ToolSpec


# ---------------------------------------------------------------------------
# Anthropic request construction -- the quirk matrix
# ---------------------------------------------------------------------------

#: Models where sampling and budget_tokens were removed. Sending either is a 400.
CURRENT_GENERATION = [
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-fable-5",
]

#: Models that still accept sampling parameters.
PREVIOUS_GENERATION = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
]


def _request(model: str, **kwargs: Any) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
        **kwargs,
    )


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider()


@pytest.mark.parametrize("model", CURRENT_GENERATION)
def test_temperature_is_never_sent_to_the_current_generation(anthropic_provider, model):
    """Sending temperature to these models is a 400, not a warning."""
    kwargs = anthropic_provider._build_kwargs(_request(model, temperature=0.7))
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.parametrize("model", CURRENT_GENERATION)
def test_budget_tokens_is_never_sent_to_the_current_generation(anthropic_provider, model):
    kwargs = anthropic_provider._build_kwargs(_request(model, thinking_budget=8000))
    thinking = kwargs.get("thinking")
    assert thinking is not None, "thinking was requested but not configured"
    assert "budget_tokens" not in thinking
    assert thinking["type"] == "adaptive"


@pytest.mark.parametrize("model", CURRENT_GENERATION)
def test_thinking_display_is_asked_for_explicitly(anthropic_provider, model):
    """display defaults to 'omitted' on these models -- a silent empty stream."""
    kwargs = anthropic_provider._build_kwargs(_request(model, thinking_budget=8000))
    assert kwargs["thinking"]["display"] == "summarized"


@pytest.mark.parametrize("model", PREVIOUS_GENERATION)
def test_sampling_still_reaches_the_previous_generation(anthropic_provider, model):
    kwargs = anthropic_provider._build_kwargs(_request(model, temperature=0.5))
    assert kwargs["temperature"] == pytest.approx(0.5)


def test_temperature_is_clamped_into_range(anthropic_provider):
    hot = anthropic_provider._build_kwargs(_request(PREVIOUS_GENERATION[0], temperature=9.0))
    cold = anthropic_provider._build_kwargs(_request(PREVIOUS_GENERATION[0], temperature=-3.0))
    assert hot["temperature"] == 1.0
    assert cold["temperature"] == 0.0


def test_fable_thinks_even_when_nothing_asked_for_it(anthropic_provider):
    """Thinking is always on for Fable; the parameter cannot be omitted-and-off."""
    kwargs = anthropic_provider._build_kwargs(_request("anthropic/claude-fable-5"))
    assert kwargs["thinking"]["type"] == "adaptive"


def test_max_tokens_is_capped_at_the_model_ceiling(anthropic_provider):
    kwargs = anthropic_provider._build_kwargs(
        _request("anthropic/claude-haiku-4-5-20251001", max_tokens=999_999)
    )
    info = catalog.lookup("anthropic/claude-haiku-4-5-20251001")
    assert kwargs["max_tokens"] <= (info.max_output if info else 32_000)


def test_the_model_id_is_sent_without_its_provider_prefix(anthropic_provider):
    kwargs = anthropic_provider._build_kwargs(_request("anthropic/claude-opus-5"))
    assert kwargs["model"] == "claude-opus-5"


def test_a_long_system_prompt_gets_a_cache_breakpoint(anthropic_provider):
    request = _request("anthropic/claude-opus-5")
    request.system = "x" * 8000
    kwargs = anthropic_provider._build_kwargs(request)
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_short_system_prompt_is_sent_plain(anthropic_provider):
    """Below the minimum cacheable prefix a breakpoint just wastes one of four."""
    request = _request("anthropic/claude-opus-5")
    request.system = "Be brief."
    kwargs = anthropic_provider._build_kwargs(request)
    assert isinstance(kwargs["system"], str)


def test_the_last_tool_carries_the_cache_breakpoint(anthropic_provider):
    request = _request("anthropic/claude-opus-5")
    request.tools = [
        ToolSpec(name="alpha", description="a", parameters={"type": "object", "properties": {}}),
        ToolSpec(name="omega", description="o", parameters={"type": "object", "properties": {}}),
    ]
    kwargs = anthropic_provider._build_kwargs(request)
    assert "cache_control" not in kwargs["tools"][0]
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_stop_sequences_are_capped_at_four(anthropic_provider):
    request = _request("anthropic/claude-opus-5")
    request.stop = ["a", "b", "c", "d", "e", "f"]
    kwargs = anthropic_provider._build_kwargs(request)
    assert len(kwargs["stop_sequences"]) == 4


# ---------------------------------------------------------------------------
# The catalogue that drives all of the above
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", CURRENT_GENERATION)
def test_catalogue_marks_the_current_generation_correctly(model: str):
    q = catalog.quirks(model)
    assert q.get("sampling_allowed") is False, f"{model} would be sent temperature"
    assert q.get("adaptive_thinking") is True
    assert q.get("prefill_allowed") is False


@pytest.mark.parametrize("model", PREVIOUS_GENERATION)
def test_catalogue_marks_the_previous_generation_correctly(model: str):
    assert catalog.quirks(model).get("sampling_allowed") is True


def test_split_id_round_trips():
    assert catalog.split_id("anthropic/claude-opus-5") == ("anthropic", "claude-opus-5")
    # A slash inside the model name must survive -- OpenRouter and fal both use them.
    assert catalog.split_id("fal/fal-ai/flux/dev") == ("fal", "fal-ai/flux/dev")


def test_no_seeded_anthropic_model_carries_a_date_suffix_it_should_not():
    """Current model ids are complete as-is; a stale date suffix is a 404."""
    dated = [
        m.id for m in catalog.seed_for("anthropic")
        if m.name.startswith(("claude-opus-5", "claude-sonnet-5"))
        and any(part.isdigit() and len(part) == 8 for part in m.name.split("-"))
    ]
    assert not dated, f"date-suffixed ids will 404: {dated}"


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming, replayed from a fixture
# ---------------------------------------------------------------------------


def _sse(*chunks: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


@pytest.fixture
def mock_openai(monkeypatch):
    """Serve a canned SSE body through the adapter's own HTTP path."""

    def install(payload: bytes, status: int = 200):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status,
                content=payload,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "ciws.gateway.providers.openai_compat.http_client", lambda *a, **k: client
        )
        return client

    return install


async def _collect(provider, request):
    return [event async for event in provider.stream_chat(request)]


async def test_text_deltas_are_streamed_and_finished(mock_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(
        _sse(
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " world"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        )
    )
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))

    text = "".join(e.text or "" for e in events if e.type is StreamEventType.TEXT)
    assert text == "Hello world"
    assert events[0].type is StreamEventType.START
    assert events[-1].type is StreamEventType.DONE
    assert events[-1].finish_reason == "stop"


async def test_streamed_tool_calls_are_reassembled_by_index(mock_openai, monkeypatch):
    """id and name arrive on the first fragment only; arguments arrive in pieces.

    Reassembling by id rather than index loses every fragment after the first,
    which shows up as a tool call with empty arguments.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(
        _sse(
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "function": {"name": "search", "arguments": ""}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"quer'}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'y": "redpanda"}'}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        )
    )
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))

    calls = [e.tool_call for e in events if e.type is StreamEventType.TOOL_CALL]
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "redpanda"}


async def test_two_parallel_tool_calls_do_not_bleed_into_each_other(mock_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(
        _sse(
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "alpha", "arguments": '{"n":'}},
                {"index": 1, "id": "c1", "function": {"name": "omega", "arguments": '{"n":'}},
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": " 1}"}},
                {"index": 1, "function": {"arguments": " 2}"}},
            ]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        )
    )
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))

    calls = {c.name: c.arguments for c in
             (e.tool_call for e in events if e.type is StreamEventType.TOOL_CALL)}
    assert calls == {"alpha": {"n": 1}, "omega": {"n": 2}}


async def test_reasoning_content_is_surfaced_as_thinking(mock_openai, monkeypatch):
    """Reasoning arrives under three different key names across vendors."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(
        _sse(
            {"choices": [{"index": 0, "delta": {"reasoning_content": "weighing options"}}]},
            {"choices": [{"index": 0, "delta": {"content": "done"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
    )
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))
    assert any(e.type is StreamEventType.THINKING for e in events)


async def test_a_mid_stream_error_becomes_an_error_event_not_an_exception(
    mock_openai, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(_sse({"error": {"message": "context length exceeded"}}))
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))

    errors = [e for e in events if e.type is StreamEventType.ERROR]
    assert errors and "context length" in errors[0].error


async def test_malformed_tool_arguments_do_not_kill_the_stream(mock_openai, monkeypatch):
    """A truncated JSON argument blob should degrade, not raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_openai(
        _sse(
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "search", "arguments": '{"query": '}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        )
    )
    events = await _collect(OpenAIProvider(), _request("openai/gpt-4.1"))
    assert events[-1].type is StreamEventType.DONE
