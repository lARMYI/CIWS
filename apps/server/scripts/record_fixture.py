#!/usr/bin/env python3
"""Record one real provider round-trip as a replayable test fixture.

The test suite runs with no credentials, so the provider fixtures in
``tests/fixtures/`` are hand-authored to the documented wire formats. That
proves the adapters parse a correctly-shaped response; it does not prove the
shape is still current. This script closes that gap for anyone who has a key:
one real call, captured to disk, replayed forever afterwards at no cost.

    python scripts/record_fixture.py anthropic/claude-opus-5
    python scripts/record_fixture.py openai/gpt-4.1 --prompt "say hi" --tools

Secrets are scrubbed on the way out -- the request headers are dropped entirely
and the recorded body is passed through the vault's redaction pass -- but read
the file before committing it. A response can echo content you did not intend to
publish, and no scrubber knows that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ciws.core import secrets  # noqa: E402
from ciws.gateway.registry import gateway  # noqa: E402
from ciws.gateway.types import ChatMessage, ChatRequest, ToolSpec  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def record(model: str, prompt: str, with_tools: bool) -> Path:
    provider_id, _ = model.split("/", 1)
    provider = gateway.providers().get(provider_id)
    if provider is None:
        raise SystemExit(f"No provider '{provider_id}'. Known: {', '.join(gateway.providers())}")
    if not provider.configured():
        raise SystemExit(
            f"{provider_id} has no credential. Export {provider.env_hint or 'its API key'} "
            f"or add it in Systems -> Credentials, then run this again."
        )

    request = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=1024,
        tools=[WEATHER_TOOL] if with_tools else [],
    )

    events: list[dict] = []
    async for event in provider.stream_chat(request):
        events.append(json.loads(event.model_dump_json()))

    fixture = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "tools": with_tools,
        "events": events,
    }

    # The vault knows every key it holds; scrub all of them from the body.
    body = secrets.redact(json.dumps(fixture, indent=2, ensure_ascii=False))

    FIXTURES.mkdir(parents=True, exist_ok=True)
    slug = model.replace("/", "_")
    if with_tools:
        slug += "_tools"
    out = FIXTURES / f"{slug}.json"
    out.write_text(body + "\n", "utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qualified model id, e.g. anthropic/claude-opus-5")
    parser.add_argument("--prompt", default="Reply with exactly: ok")
    parser.add_argument(
        "--tools", action="store_true", help="Offer a tool, to capture tool_use blocks"
    )
    args = parser.parse_args()

    out = asyncio.run(record(args.model, args.prompt, args.tools))
    print(f"Wrote {out}")
    print("Read it before committing -- a response can echo more than you meant to publish.")


if __name__ == "__main__":
    main()
