"""In-process event bus.

Every interesting thing that happens inside CIWS -- a token streaming out of a
model, a tool starting, a memory being written, an entity appearing in the
ontology, a video render finishing -- is published here. The WebSocket layer
subscribes and fans events out to the UI, so the whole workspace stays live
without polling.

Subscribers get a bounded queue. A slow consumer drops its oldest events rather
than applying backpressure to the agent that is producing them: a laggy browser
tab must never stall a running agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator


def jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` will accept.

    Publishers are ordinary application code and pass whatever they have to
    hand -- a ``datetime`` from a database row, a ``Path``, an ``Enum``. The
    WebSocket serialises events with ``send_json``, so one unserialisable value
    used to raise inside the send and tear down the socket.

    That failure was worse than it looks: the bus replays recent history to
    every new subscriber, so a single event carrying a ``datetime`` poisoned the
    replay buffer and killed *every subsequent connection* -- the whole UI went
    dead until a restart. Coercing here keeps a publisher's convenience from
    becoming a transport failure.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return str(value)


@dataclass(slots=True)
class Event:
    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """The wire form. Always JSON-serialisable -- see ``jsonable``."""
        return {
            "id": self.id,
            "topic": self.topic,
            "ts": self.ts,
            "data": jsonable(self.data),
        }


class Subscription:
    def __init__(self, bus: "EventBus", patterns: tuple[str, ...], maxsize: int) -> None:
        self._bus = bus
        self.patterns = patterns
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def matches(self, topic: str) -> bool:
        return any(fnmatch.fnmatchcase(topic, p) for p in self.patterns)

    def offer(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
                self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(event)

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._bus.unsubscribe(self)

    async def stream(self) -> AsyncIterator[Event]:
        while True:
            yield await self.queue.get()


class EventBus:
    def __init__(self, history: int = 500) -> None:
        self._subs: set[Subscription] = set()
        self._history: deque[Event] = deque(maxlen=history)

    def publish(self, topic: str, **data: Any) -> Event:
        ev = Event(topic=topic, data=data)
        self._history.append(ev)
        for sub in list(self._subs):
            if sub.matches(topic):
                sub.offer(ev)
        return ev

    def subscribe(self, *patterns: str, maxsize: int = 2048) -> Subscription:
        sub = Subscription(self, patterns or ("*",), maxsize)
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    def replay(self, *patterns: str, limit: int = 100) -> list[Event]:
        pats = patterns or ("*",)
        hits = [e for e in self._history if any(fnmatch.fnmatchcase(e.topic, p) for p in pats)]
        return hits[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


#: Process-wide bus. One hub, one bus.
bus = EventBus()


# ---------------------------------------------------------------------------
# Canonical topics. Keep these in sync with apps/web/src/lib/events.ts
# ---------------------------------------------------------------------------

class Topic:
    RUN_START = "run.start"
    RUN_STEP = "run.step"
    RUN_DELTA = "run.delta"
    RUN_THINKING = "run.thinking"
    RUN_MESSAGE = "run.message"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"
    RUN_CANCEL = "run.cancel"

    TOOL_START = "tool.start"
    TOOL_END = "tool.end"
    TOOL_ERROR = "tool.error"
    TOOL_APPROVAL = "tool.approval"

    MEMORY_WRITE = "memory.write"
    MEMORY_RECALL = "memory.recall"
    MEMORY_CONSOLIDATE = "memory.consolidate"

    GRAPH_ENTITY = "graph.entity"
    GRAPH_EDGE = "graph.edge"
    GRAPH_MERGE = "graph.merge"

    MEDIA_START = "media.start"
    MEDIA_PROGRESS = "media.progress"
    MEDIA_DONE = "media.done"
    MEDIA_ERROR = "media.error"

    INGEST_START = "ingest.start"
    INGEST_PROGRESS = "ingest.progress"
    INGEST_DONE = "ingest.done"

    HUB_STATUS = "hub.status"
    MODEL_STATUS = "model.status"
    WORKFLOW_NODE = "workflow.node"
    SYSTEM_NOTICE = "system.notice"
