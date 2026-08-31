"""The remaining gateway adapters, web search, and the shell tool.

Google and Ollama speak neither the Anthropic nor the OpenAI wire format, so
each has its own translation layer -- the place a provider-neutral gateway most
often gets quietly wrong. Both are exercised here through their real HTTP path
with a canned response.

Web search matters for a different reason: the keyless DuckDuckGo fallback is
what makes a fresh install able to search at all, and it is the backend most
likely to rot silently.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ciws.gateway.providers.google import GoogleProvider
from ciws.gateway.providers.ollama import OllamaProvider
from ciws.gateway.types import ChatMessage, ChatRequest, StreamEventType
from ciws.hubs import websearch


def _request(model: str, **kwargs: Any) -> ChatRequest:
    return ChatRequest(
        model=model, messages=[ChatMessage(role="user", content="hello")], **kwargs
    )


@pytest.fixture
def mock_http(monkeypatch):
    """Install a mock transport into a module's http_client and capture requests."""

    def install(module: str, responder):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return responder(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(f"{module}.http_client", lambda *a, **k: client, raising=False)
        return seen

    return install


def _sse(*chunks: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode()


async def _collect(provider, request):
    return [event async for event in provider.stream_chat(request)]


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


async def test_google_translates_to_contents_and_parts(mock_http, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {"candidates": [{"content": {"parts": [{"text": "Hi "}], "role": "model"}}]},
                {"candidates": [{"content": {"parts": [{"text": "there"}], "role": "model"}},],
                 "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}},
            ),
            headers={"content-type": "text/event-stream"},
        )

    seen = mock_http("ciws.gateway.providers.google", responder)
    events = await _collect(GoogleProvider(), _request("google/gemini-2.0-flash"))

    assert seen, "google adapter issued no request"
    body = json.loads(seen[0].read().decode())
    # Google's schema is contents[].parts[], not messages[].
    assert "contents" in body
    assert body["contents"][0]["parts"][0]["text"] == "hello"

    text = "".join(e.text or "" for e in events if e.type is StreamEventType.TEXT)
    assert text == "Hi there"


async def test_google_sends_the_key_without_putting_it_in_the_path(mock_http, monkeypatch):
    """A key in the URL ends up in logs and proxy history."""
    monkeypatch.setenv("GOOGLE_API_KEY", "leaky-key")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"candidates": []}),
                              headers={"content-type": "text/event-stream"})

    seen = mock_http("ciws.gateway.providers.google", responder)
    await _collect(GoogleProvider(), _request("google/gemini-2.0-flash"))
    assert "leaky-key" not in str(seen[0].url.path)


async def test_google_system_prompt_uses_system_instruction(mock_http, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"candidates": []}),
                              headers={"content-type": "text/event-stream"})

    seen = mock_http("ciws.gateway.providers.google", responder)
    request = _request("google/gemini-2.0-flash")
    request.system = "Be concise."
    await _collect(GoogleProvider(), request)

    body = json.loads(seen[0].read().decode())
    assert "systemInstruction" in body or "system_instruction" in body


async def test_google_without_a_key_is_not_configured():
    assert GoogleProvider().configured() is False


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_ollama_defaults_to_the_local_daemon():
    """Local inference must never leave the machine."""
    url = OllamaProvider().base_url
    assert "127.0.0.1" in url or "localhost" in url


def test_ollama_needs_no_key():
    provider = OllamaProvider()
    assert provider.requires_key is False
    assert provider.local is True


async def test_ollama_streams_ndjson_not_sse(mock_http):
    """Ollama emits newline-delimited JSON, not Server-Sent Events."""

    def responder(request: httpx.Request) -> httpx.Response:
        body = (
            json.dumps({"message": {"role": "assistant", "content": "Local "}, "done": False})
            + "\n"
            + json.dumps({"message": {"role": "assistant", "content": "reply"}, "done": False})
            + "\n"
            + json.dumps(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 7,
                    "eval_count": 2,
                }
            )
            + "\n"
        )
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})

    mock_http("ciws.gateway.providers.ollama", responder)
    events = await _collect(OllamaProvider(), _request("ollama/llama3.2"))

    text = "".join(e.text or "" for e in events if e.type is StreamEventType.TEXT)
    assert text == "Local reply"
    assert events[-1].type is StreamEventType.DONE


async def test_ollama_reports_zero_cost(mock_http):
    """Local inference genuinely is free -- this zero is not a 'not seeded'."""

    def responder(request: httpx.Request) -> httpx.Response:
        body = json.dumps(
            {"message": {"role": "assistant", "content": "x"}, "done": True,
             "prompt_eval_count": 100, "eval_count": 50}
        ) + "\n"
        return httpx.Response(200, content=body.encode())

    mock_http("ciws.gateway.providers.ollama", responder)
    events = await _collect(OllamaProvider(), _request("ollama/llama3.2"))
    usage = next((e.usage for e in reversed(events) if e.usage), None)
    assert usage is not None
    assert usage.cost_usd == 0.0


async def test_ollama_lists_the_models_it_has_pulled(mock_http):
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [
                {"name": "llama3.2:latest", "size": 2_000_000_000},
                {"name": "qwen2.5:7b", "size": 4_000_000_000},
            ]},
        )

    mock_http("ciws.gateway.providers.ollama", responder)
    models = await OllamaProvider().list_models()
    assert {m.name for m in models} == {"llama3.2:latest", "qwen2.5:7b"}
    assert all(m.provider == "ollama" for m in models)


async def test_ollama_being_offline_is_an_empty_list_not_a_crash(mock_http):
    """A daemon that is not running must not break the model catalogue."""

    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    mock_http("ciws.gateway.providers.ollama", responder)
    provider = OllamaProvider()
    models = await provider.models()
    assert models == []

    health = await provider.health()
    assert health.ok is False


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


def test_the_keyless_fallback_is_always_available():
    backends = {b["id"] for b in websearch.available_backends()}
    assert "duckduckgo" in backends


def test_backends_report_whether_they_are_configured():
    for backend in websearch.available_backends():
        assert "configured" in backend
    keyed = [b for b in websearch.available_backends() if b["id"] != "duckduckgo"]
    assert all(b["configured"] is False for b in keyed), "a backend claimed a key it has not got"


def test_pick_falls_back_to_duckduckgo_with_no_keys():
    assert websearch._pick() == "duckduckgo"


async def test_duckduckgo_parses_results_out_of_html(mock_http):
    html = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">Example Docs</a>
      <a class="result__snippet">Everything about turbines.</a>
    </div>
    """

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    mock_http("ciws.hubs.websearch", responder)
    results = await websearch.search("turbines", limit=3, backend="duckduckgo")
    assert results
    assert "example.com" in results[0]["url"]
    assert results[0]["source"] == "duckduckgo"


async def test_a_failing_backend_says_which_one_and_why(mock_http):
    """Silently returning zero results would look identical to "nothing matched"."""
    from ciws.core.errors import ProviderError

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    mock_http("ciws.hubs.websearch", responder)
    with pytest.raises(ProviderError) as caught:
        await websearch.search("anything", limit=3, backend="duckduckgo")
    assert "duckduckgo" in str(caught.value)


async def test_fetch_page_returns_readable_text(monkeypatch):
    """fetch_page delegates to extract_url, which builds its own client inline."""

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "<html><head><title>Ops</title></head>"
                "<body><p>Retention is 90 days.</p></body></html>"
            ),
            headers={"content-type": "text/html"},
        )

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(responder), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)

    page = await websearch.fetch_page("https://example.com/ops")
    assert "90 days" in page["text"]
    assert page["title"] == "Ops"


def test_untag_strips_markup():
    assert "<" not in websearch._untag("<b>bold</b> and <i>italic</i>")


def test_ddg_redirects_are_unwrapped():
    unwrapped = websearch._ddg_url("/l/?uddg=https%3A%2F%2Fexample.com%2Fa")
    assert unwrapped == "https://example.com/a"


# ---------------------------------------------------------------------------
# The shell tool, once it is switched on
# ---------------------------------------------------------------------------


async def test_shell_runs_only_when_explicitly_enabled(tools_loaded, monkeypatch):
    from ciws.core.config import get_settings
    from ciws.tools.registry import ToolContext, registry

    settings = get_settings()
    monkeypatch.setattr(settings.security, "allow_shell_tool", True)

    result = await registry.call(
        "shell", {"command": "echo hello-from-shell"}, ToolContext(auto_approve=True)
    )
    assert not result.is_error
    assert "hello-from-shell" in result.content


async def test_shell_reports_a_non_zero_exit_as_an_observation(tools_loaded, monkeypatch):
    from ciws.core.config import get_settings
    from ciws.tools.registry import ToolContext, registry

    settings = get_settings()
    monkeypatch.setattr(settings.security, "allow_shell_tool", True)

    result = await registry.call(
        "shell", {"command": "exit 3"}, ToolContext(auto_approve=True)
    )
    assert result.is_error
    assert "3" in result.content
