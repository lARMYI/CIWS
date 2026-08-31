"""Splitting documents into retrievable pieces.

The naive approach -- fixed windows of N characters -- produces chunks that
start mid-sentence and carry no indication of what section they came from. When
one of those lands in a model's context it reads as noise.

So splitting descends a hierarchy: headings, then paragraphs, then sentences,
and only cuts mid-sentence when a single sentence is itself oversized. Each
chunk carries the nearest heading above it, which is what makes a retrieved
fragment legible on its own. Consecutive chunks overlap by a sentence or two so
a fact spanning a boundary is not lost by both sides.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from ..core.util import estimate_tokens

#: Markdown ATX headings, Setext underlines, and the "## Slide 3" / "## Sheet"
#: markers the extractors emit.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.{1,200})$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^(.{1,120})\n(={3,}|-{3,})\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=[.!?][\"')\]])\s+")
_CODE_BLOCK_RE = re.compile(r"^(?:def |class |function |const |export |public |private |async def )", re.MULTILINE)


@dataclass(slots=True)
class TextChunk:
    text: str
    idx: int = 0
    heading: str = ""
    page: int = 0
    token_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _page_for(offset: int, page_offsets: list[int]) -> int:
    if not page_offsets:
        return 0
    return max(1, bisect_right(page_offsets, offset))


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _sections(text: str) -> list[tuple[str, str, int]]:
    """``(heading, body, offset)`` triples, in document order."""
    marks: list[tuple[int, int, str]] = []
    for match in _HEADING_RE.finditer(text):
        marks.append((match.start(), match.end(), match.group(2).strip()))
    for match in _SETEXT_RE.finditer(text):
        marks.append((match.start(), match.end(), match.group(1).strip()))
    marks.sort()

    if not marks:
        return [("", text, 0)]

    sections: list[tuple[str, str, int]] = []
    if marks[0][0] > 0:
        preamble = text[: marks[0][0]].strip()
        if preamble:
            sections.append(("", preamble, 0))
    for i, (_, end, heading) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if body:
            sections.append((heading, body, end))
    return sections


def _pack(
    units: list[str], target_chars: int, overlap_chars: int
) -> list[tuple[str, int]]:
    """Greedily fill chunks from units, carrying an overlap tail forward."""
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_len = 0
    consumed = 0
    start_offset = 0

    for unit in units:
        unit_len = len(unit) + 1
        if current and current_len + unit_len > target_chars:
            chunks.append(("\n".join(current), start_offset))
            # Carry the tail of this chunk into the next one.
            tail: list[str] = []
            tail_len = 0
            for piece in reversed(current):
                if tail_len + len(piece) > overlap_chars:
                    break
                tail.insert(0, piece)
                tail_len += len(piece) + 1
            start_offset = consumed - tail_len
            current = list(tail)
            current_len = tail_len
        current.append(unit)
        current_len += unit_len
        consumed += unit_len

    if current:
        chunks.append(("\n".join(current), max(0, start_offset)))
    return chunks


def _hard_split(text: str, target_chars: int) -> list[str]:
    """Last resort for a single oversized unit -- split on whitespace runs."""
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + target_chars)
        if end < len(text):
            window = text.rfind(" ", cursor + int(target_chars * 0.6), end)
            if window > cursor:
                end = window
        out.append(text[cursor:end].strip())
        cursor = end
    return [o for o in out if o]


def chunk_text(
    text: str,
    *,
    target_tokens: int = 550,
    overlap_tokens: int = 80,
    heading_aware: bool = True,
    page_offsets: list[int] | None = None,
) -> list[TextChunk]:
    text = (text or "").strip()
    if not text:
        return []

    # Token estimation runs ~4 chars/token; the chunker works in characters
    # because that is what splitting actually operates on.
    target_chars = max(400, target_tokens * 4)
    overlap_chars = max(0, min(overlap_tokens * 4, target_chars // 3))
    page_offsets = page_offsets or []

    sections = _sections(text) if heading_aware else [("", text, 0)]
    chunks: list[TextChunk] = []

    for heading, body, section_offset in sections:
        if len(body) <= target_chars:
            units_source = [body]
            packed = [(body, 0)]
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            units: list[str] = []
            for paragraph in paragraphs:
                if len(paragraph) <= target_chars:
                    units.append(paragraph)
                    continue
                sentences = _split_sentences(paragraph)
                for sentence in sentences or [paragraph]:
                    if len(sentence) <= target_chars:
                        units.append(sentence)
                    else:
                        units.extend(_hard_split(sentence, target_chars))
            units_source = units
            packed = _pack(units_source, target_chars, overlap_chars)

        for body_text, local_offset in packed:
            body_text = body_text.strip()
            if not body_text:
                continue
            absolute = section_offset + local_offset
            chunks.append(
                TextChunk(
                    text=body_text,
                    idx=len(chunks),
                    heading=heading,
                    page=_page_for(absolute, page_offsets),
                    token_count=estimate_tokens(body_text),
                    meta={"offset": absolute},
                )
            )

    for i, chunk in enumerate(chunks):
        chunk.idx = i
    return chunks


def chunk_code(text: str, *, target_tokens: int = 600) -> list[TextChunk]:
    """Split source on top-level definitions so a function stays whole."""
    text = (text or "").strip()
    if not text:
        return []
    target_chars = max(500, target_tokens * 4)

    lines = text.splitlines()
    boundaries = [0]
    for i, line in enumerate(lines):
        if i and _CODE_BLOCK_RE.match(line):
            boundaries.append(i)
    boundaries.append(len(lines))

    blocks: list[str] = []
    for start, stop in zip(boundaries, boundaries[1:]):
        block = "\n".join(lines[start:stop]).strip()
        if block:
            blocks.append(block)
    if not blocks:
        blocks = [text]

    packed = _pack(blocks, target_chars, target_chars // 8)
    chunks: list[TextChunk] = []
    for i, (body, offset) in enumerate(packed):
        first = body.strip().splitlines()[0] if body.strip() else ""
        chunks.append(
            TextChunk(
                text=body,
                idx=i,
                heading=first[:120],
                token_count=estimate_tokens(body),
                meta={"offset": offset, "is_code": True},
            )
        )
    return chunks
