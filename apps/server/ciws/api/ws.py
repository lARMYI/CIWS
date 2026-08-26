"""The live event socket.

One socket per browser tab carries every workspace event: streaming tokens,
tool starts and stops, memories being written, entities appearing, renders
finishing, hubs changing state. Panels subscribe to topic patterns rather than
polling, which is what lets the graph animate while a chat is streaming.

Backpressure is handled by dropping, not blocking: a tab that stops reading
loses its oldest events rather than stalling the agent producing them.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.events import bus
from ..core.logging import get_logger
from .deps import require_ws_auth

log = get_logger("api.ws")
router = APIRouter()

HEARTBEAT_SECONDS = 25.0


@router.websocket("/ws")
async def event_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    if not await require_ws_auth(websocket):
        return

    patterns = websocket.query_params.get("topics", "*").split(",")
    subscription = bus.subscribe(*[p.strip() for p in patterns if p.strip()])

    await websocket.send_json(
        {
            "topic": "connected",
            "data": {"patterns": subscription.patterns, "subscribers": bus.subscriber_count},
        }
    )
    for event in bus.replay(*subscription.patterns, limit=40):
        await websocket.send_json(event.to_dict())

    async def pump() -> None:
        async for event in subscription.stream():
            await websocket.send_json(event.to_dict())

    async def heartbeat() -> None:
        # Without this an idle socket dies to an intermediary's timeout, and the
        # UI silently stops updating with no error to show.
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await websocket.send_json({"topic": "ping", "data": {"dropped": subscription.dropped}})

    async def reader() -> None:
        # Read to notice disconnects promptly; the client sends nothing meaningful.
        while True:
            await websocket.receive_text()

    tasks = [asyncio.create_task(c()) for c in (pump, heartbeat, reader)]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError, Exception):
                task.result()
    finally:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
        subscription.close()
