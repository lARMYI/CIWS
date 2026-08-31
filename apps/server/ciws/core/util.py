"""Small shared helpers."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:24]
    return f"{prefix}_{raw}" if prefix else raw


def now() -> datetime:
    return datetime.now(timezone.utc)


def now_ts() -> float:
    return time.time()


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def slugify(text: str, max_len: int = 64) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].strip("-") or "untitled"


def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate.

    Good enough for budgeting and context-window guards; never used for billing.
    Roughly 4 characters per token with a nudge for whitespace-heavy text.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4 + text.count(" ") * 0.08))


def truncate(text: str, limit: int, marker: str = "\n...[truncated]...\n") -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head - len(marker)
    return text[:head] + marker + (text[-tail:] if tail > 0 else "")


def chunks(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def safe_json(value: Any, default: Any = None) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return default


def dumps(value: Any) -> str:
    """JSON encode with a fallback for anything exotic an LLM handed us."""
    return json.dumps(value, ensure_ascii=False, default=str)


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model's prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def human_bytes(n: int | float) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= step
    return f"{n:.1f}PB"
