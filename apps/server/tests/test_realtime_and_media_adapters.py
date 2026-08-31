"""The WebSocket event bus, the web tools, and the remaining media adapters.

The WebSocket is the spine of every live panel, and it has one property worth
protecting above correctness of any single event: **a laggy subscriber must
never stall a producer.** Per-subscriber queues drop their own oldest events on
overflow, so a browser tab left open on a slow machine cannot apply
backpressure to a running agent.

The media adapters here are the ones that poll -- Replicate and fal both return
a job handle and expect you to come back for the result -- which is the part of
those integrations most likely to be wrong.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import pytest

from ciws.core.events import EventBus, Topic, bus
from ciws.media import providers as media_providers
from ciws.media.types import ImageRequest

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# The event bus
# ---------------------------------------------------------------------------


async def test_a_subscriber_receives_published_events():
    local = EventBus()
    async with local.subscribe("*") as sub:
        local.publish(Topic.MEMORY_WRITE, id="mem_1")
        event = await asyncio.wait_for(sub.queue.get(), timeout=2)
    assert event.topic == Topic.MEMORY_WRITE
    assert event.data["id"] == "mem_1"


async def test_topic_filters_are_honoured():
    local = EventBus()
    async with local.subscribe("memory.*") as sub:
        local.publish(Topic.GRAPH_ENTITY, id="ent_1")
        local.publish(Topic.MEMORY_WRITE, id="mem_1")
        event = await asyncio.wait_for(sub.queue.get(), timeout=2)
    assert event.topic == Topic.MEMORY_WRITE, "a filtered-out topic was delivered"


async def test_a_slow_subscriber_drops_its_own_events_rather_than_blocking():
    """The property the whole design turns on.

    If the queue blocked instead of dropping, one stalled browser tab would
    stall the agent producing the events.
    """
    local = EventBus()
    async with local.subscribe("*") as sub:
        # Publish far more than any sane queue bound without ever reading.
        for n in range(5000):
            local.publish(Topic.SYSTEM_NOTICE, n=n)

        # Publishing never blocked; the queue stayed bounded and counted the loss.
        assert sub.queue.qsize() <= 2048
        assert sub.dropped > 0, "an over-full queue kept everything instead of dropping"
        event = await asyncio.wait_for(sub.queue.get(), timeout=2)
        assert event.topic == Topic.SYSTEM_NOTICE


async def test_publishing_with_no_subscribers_is_harmless():
    local = EventBus()
    local.publish(Topic.SYSTEM_NOTICE, message="nobody is listening")


async def test_two_subscribers_both_receive_the_same_event():
    local = EventBus()
    async with local.subscribe("*") as first, local.subscribe("*") as second:
        local.publish(Topic.RUN_START, run_id="run_1")
        a = await asyncio.wait_for(first.queue.get(), timeout=2)
        b = await asyncio.wait_for(second.queue.get(), timeout=2)
    assert a.data["run_id"] == b.data["run_id"] == "run_1"


async def test_an_event_carrying_a_datetime_does_not_kill_the_socket():
    """Regression: one unserialisable value used to take down every connection.

    The bus replays recent history to each new subscriber, so a single event
    carrying a datetime poisoned the replay buffer and every subsequent connect
    died inside send_json -- the whole UI went dead until a restart.
    """
    import json
    from datetime import datetime, timezone
    from decimal import Decimal
    from pathlib import Path as _Path

    local = EventBus()
    local.publish(
        Topic.INGEST_DONE,
        finished_at=datetime.now(timezone.utc),
        where=_Path("/tmp/doc.md"),
        cost=Decimal("0.25"),
        tags={"a", "b"},
    )
    replayed = local.replay("*", limit=5)
    assert replayed

    encoded = json.dumps(replayed[0].to_dict())
    assert "finished_at" in encoded

    round_tripped = json.loads(encoded)["data"]
    assert isinstance(round_tripped["finished_at"], str)
    assert round_tripped["where"] == "/tmp/doc.md"
    assert round_tripped["cost"] == 0.25
    assert sorted(round_tripped["tags"]) == ["a", "b"]


def test_the_shared_bus_exposes_the_canonical_topics():
    for name in ("RUN_START", "RUN_END", "TOOL_START", "TOOL_END", "MEMORY_WRITE",
                 "GRAPH_ENTITY", "MEDIA_DONE", "INGEST_DONE", "SYSTEM_NOTICE"):
        assert hasattr(Topic, name), f"Topic.{name} disappeared; a panel subscribes to it"
    assert bus is not None


# ---------------------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------------------


async def test_the_socket_refuses_a_bad_token(app):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect("/api/ws?token=wrong") as socket:
                socket.receive_json()
    assert caught.value.code == 1008


async def test_the_socket_delivers_a_hello_then_live_events(app, monkeypatch):
    """Auth is covered by the rejection test above and by test_security.

    This one is about the payload: a client must learn what it is connected to
    before events start arriving, or a panel cannot tell "connected and idle"
    from "never connected".
    """
    from starlette.testclient import TestClient

    from ciws.core.config import get_settings

    monkeypatch.setattr(get_settings().security, "require_token", False)

    # Subscribe to a topic nothing else publishes. The bus replays up to 40
    # historical events to every new subscriber, so a "*" subscription in a
    # full-suite run arrives pre-loaded with other tests' traffic.
    marker = "test.marker"

    with TestClient(app) as client:
        with client.websocket_connect(f"/api/ws?topics={marker}") as socket:
            hello = socket.receive_json()
            assert hello["topic"] == "connected"
            assert marker in hello["data"]["patterns"]

            bus.publish(marker, message="live event")
            event = socket.receive_json()

    assert event["topic"] == marker
    assert event["data"]["message"] == "live event"


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_web(monkeypatch):
    def install(module: str, responder):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return responder(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(f"{module}.http_client", lambda *a, **k: client, raising=False)
        return seen

    return install


async def test_web_search_tool_formats_results_for_a_model(tools_loaded, mock_web):
    from ciws.tools.registry import ToolContext, registry

    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Turbine Docs</a>
      <a class="result__snippet">All about turbines.</a>
    </div>
    """
    mock_web("ciws.hubs.websearch", lambda r: httpx.Response(200, text=html))

    result = await registry.call(
        "web_search", {"query": "turbines"}, ToolContext(auto_approve=True)
    )
    assert not result.is_error
    assert "Turbine Docs" in result.content
    assert "example.com" in result.content


async def test_web_search_failing_is_an_observation_not_a_crash(tools_loaded, mock_web):
    from ciws.tools.registry import ToolContext, registry

    mock_web("ciws.hubs.websearch", lambda r: httpx.Response(500, text="boom"))
    result = await registry.call(
        "web_search", {"query": "anything"}, ToolContext(auto_approve=True)
    )
    assert result.is_error
    assert not result.content.startswith("Traceback")


async def test_web_fetch_tool_returns_readable_text(tools_loaded, monkeypatch):
    from ciws.tools.registry import ToolContext, registry

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><head><title>Doc</title></head><body><p>Snapshots run nightly.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(responder), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)

    result = await registry.call(
        "web_fetch", {"url": "https://example.com/doc"}, ToolContext(auto_approve=True)
    )
    assert not result.is_error
    assert "nightly" in result.content


# ---------------------------------------------------------------------------
# Media adapters that poll for a result
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_media(monkeypatch):
    def install(responder):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return responder(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "ciws.media.providers.http_client", lambda *a, **k: client, raising=False
        )
        return seen

    return install


async def test_replicate_creates_a_prediction_then_polls_it(mock_media, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8-test")
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={"id": "pred_1", "status": "starting",
                      "urls": {"get": "https://api.replicate.com/v1/predictions/pred_1"}},
            )
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(200, json={"id": "pred_1", "status": "processing"})
        return httpx.Response(
            200,
            json={"id": "pred_1", "status": "succeeded",
                  "output": ["https://replicate.delivery/out.png"]},
        )

    seen = mock_media(responder)
    results = await media_providers.Replicate().generate_image(
        ImageRequest(prompt="a turbine", model="replicate/black-forest-labs/flux-schnell")
    )

    methods = [r.method for r in seen]
    assert methods[0] == "POST", "no prediction was created"
    assert "GET" in methods, "the adapter never polled for the result"
    assert results


async def test_replicate_surfaces_a_failed_prediction(mock_media, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8-test")

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={"id": "p", "status": "starting",
                      "urls": {"get": "https://api.replicate.com/v1/predictions/p"}},
            )
        return httpx.Response(
            200, json={"id": "p", "status": "failed", "error": "NSFW content detected"}
        )

    mock_media(responder)
    from ciws.core.errors import CIWSError

    with pytest.raises(CIWSError) as caught:
        await media_providers.Replicate().generate_image(
            ImageRequest(prompt="x", model="replicate/black-forest-labs/flux-schnell")
        )
    assert "nsfw" in str(caught.value).lower()


async def test_fal_sends_its_key_as_a_key_header(mock_media, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-test")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"images": [{"url": "https://fal.media/out.png"}]},
        )

    seen = mock_media(responder)
    await media_providers.Fal().generate_image(
        ImageRequest(prompt="a turbine", model="fal/fal-ai/flux/dev")
    )
    auth = seen[0].headers.get("authorization", "")
    assert auth.lower().startswith("key "), f"fal expects 'Key <token>', got {auth!r}"


async def test_stability_posts_multipart_not_json(mock_media, monkeypatch):
    monkeypatch.setenv("STABILITY_API_KEY", "sk-stab")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"image": base64.b64encode(PNG_1PX).decode()})

    seen = mock_media(responder)
    await media_providers.Stability().generate_image(
        ImageRequest(prompt="a turbine", model="stability/core")
    )
    content_type = seen[0].headers.get("content-type", "")
    assert "multipart/form-data" in content_type, (
        f"Stability's v2beta endpoints take multipart, got {content_type!r}"
    )


async def test_comfyui_being_offline_is_a_clean_error(mock_media):
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    mock_media(responder)

    # A local backend that is not running must fail loudly, not hang or return
    # an empty result that looks like a successful render of nothing.
    with pytest.raises((httpx.ConnectError, Exception)) as caught:
        await media_providers.ComfyUI().generate_image(
            ImageRequest(prompt="x", model="comfyui/workflow")
        )
    assert caught.value is not None
