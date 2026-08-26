"""Filesystem and corpus tools.

Path handling is the security surface here. Every path is resolved to an
absolute real path before any check, because ``workspace/../../.ssh/id_rsa``
looks harmless until it is resolved. Reads are permitted anywhere the OS allows
-- the user's own files are the point of a personal hub -- but writes outside
the workspace are gated behind an approval, and a short deny-list of credential
paths is refused outright regardless of setting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...core import paths
from ...core.config import get_settings
from ...core.errors import ToolError
from ...core.util import human_bytes
from ...ingest import pipeline
from ..registry import Risk, ToolOutput, registry

MAX_READ_CHARS = 60_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".next", ".cache"}

#: Refused for read and write regardless of configuration. Not a complete
#: defence -- it is a guard against an agent wandering into credentials while
#: doing something otherwise reasonable.
DENY_FRAGMENTS = (
    ".ssh", ".aws/credentials", ".gnupg", ".config/gh/hosts.yml",
    "id_rsa", "id_ed25519", ".netrc", "shadow", "vault.key", "vault.enc",
)


def _resolve(raw: str) -> Path:
    if not raw or not raw.strip():
        raise ToolError("file", "A path is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = paths.workspace_dir() / path
    resolved = path.resolve()

    lowered = str(resolved).lower().replace("\\", "/")
    for fragment in DENY_FRAGMENTS:
        if fragment in lowered:
            raise ToolError("file", f"Refusing to touch '{raw}' -- it looks like a credential file.")
    return resolved


def _in_workspace(path: Path) -> bool:
    try:
        path.relative_to(paths.workspace_dir().resolve())
        return True
    except ValueError:
        return False


def _check_write(path: Path) -> None:
    if _in_workspace(path):
        return
    if get_settings().security.approve_writes_outside_workspace:
        raise ToolError(
            "file",
            f"Writing outside the workspace is restricted. '{path}' is not under "
            f"{paths.workspace_dir()}. Ask the user to move the file into the workspace, "
            f"or to turn off 'approve writes outside workspace' in Settings -> Security.",
        )


@registry.tool(
    "file_read",
    category="files",
    risk=Risk.SAFE,
    max_result_chars=MAX_READ_CHARS + 2000,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute, or relative to the workspace."},
            "start_line": {"type": "integer", "description": "1-based. Omit to read from the top."},
            "max_lines": {"type": "integer", "description": "Default 800."},
        },
        "required": ["path"],
    },
)
async def file_read(path: str, start_line: int = 1, max_lines: int = 800, ctx: Any = None) -> ToolOutput:
    """Read a text file from disk, with line numbers."""
    target = _resolve(path)
    if not target.exists():
        raise ToolError("file_read", f"No such file: {target}")
    if target.is_dir():
        raise ToolError("file_read", f"{target} is a directory -- use file_list")

    size = target.stat().st_size
    if size > 20 * 1024 * 1024:
        raise ToolError("file_read", f"{target.name} is {human_bytes(size)}, too large to read")

    try:
        text = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        raise ToolError("file_read", f"Could not read {target}: {exc}") from exc

    lines = text.splitlines()
    start = max(1, start_line)
    window = lines[start - 1 : start - 1 + max(1, max_lines)]
    numbered = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(window))
    if len(numbered) > MAX_READ_CHARS:
        numbered = numbered[:MAX_READ_CHARS] + "\n... (truncated)"

    header = f"{target}  ({len(lines)} lines, {human_bytes(size)})"
    if start > 1 or len(window) < len(lines):
        header += f"  showing {start}-{start + len(window) - 1}"
    return ToolOutput.text(f"{header}\n\n{numbered}", path=str(target), lines=len(lines))


@registry.tool(
    "file_write",
    category="files",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean", "default": False},
        },
        "required": ["path", "content"],
    },
)
async def file_write(path: str, content: str, append: bool = False, ctx: Any = None) -> ToolOutput:
    """Write a text file. Creates parent directories as needed.

    Writes go to the workspace by default. Writing elsewhere is refused unless
    the user has relaxed it in Settings.
    """
    target = _resolve(path)
    _check_write(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError("file_write", f"Could not write {target}: {exc}") from exc
    return ToolOutput.text(
        f"{'Appended to' if append else 'Wrote'} {target} ({human_bytes(len(content.encode()))})",
        path=str(target),
    )


@registry.tool(
    "file_edit",
    category="files",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "find": {"type": "string", "description": "Exact text to replace. Must appear once."},
            "replace": {"type": "string"},
        },
        "required": ["path", "find", "replace"],
    },
)
async def file_edit(path: str, find: str, replace: str, ctx: Any = None) -> ToolOutput:
    """Replace an exact string in a file.

    The target must appear exactly once -- include surrounding lines to make it
    unique rather than guessing. Refuses ambiguous edits rather than picking one.
    """
    target = _resolve(path)
    _check_write(target)
    if not target.exists():
        raise ToolError("file_edit", f"No such file: {target}")

    text = target.read_text("utf-8", errors="replace")
    count = text.count(find)
    if count == 0:
        raise ToolError("file_edit", "That exact text is not in the file.")
    if count > 1:
        raise ToolError(
            "file_edit", f"That text appears {count} times. Include more context to disambiguate."
        )
    target.write_text(text.replace(find, replace), "utf-8")
    return ToolOutput.text(f"Edited {target}", path=str(target))


@registry.tool(
    "file_list",
    category="files",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory. Defaults to the workspace."},
            "pattern": {"type": "string", "description": "Glob, e.g. '**/*.py'."},
            "limit": {"type": "integer", "default": 200},
        },
    },
)
async def file_list(path: str = "", pattern: str = "*", limit: int = 200, ctx: Any = None) -> ToolOutput:
    """List directory contents, optionally by glob."""
    target = _resolve(path) if path else paths.workspace_dir()
    if not target.is_dir():
        raise ToolError("file_list", f"{target} is not a directory")

    entries: list[str] = []
    try:
        for item in sorted(target.glob(pattern)):
            if any(part in SKIP_DIRS for part in item.parts):
                continue
            if item.is_dir():
                entries.append(f"  {item.name}/")
            else:
                entries.append(f"  {item.name}  ({human_bytes(item.stat().st_size)})")
            if len(entries) >= limit:
                entries.append(f"  ... (stopped at {limit})")
                break
    except OSError as exc:
        raise ToolError("file_list", f"Could not list {target}: {exc}") from exc

    return ToolOutput.text(f"{target}\n" + ("\n".join(entries) or "  (empty)"), path=str(target))


@registry.tool(
    "file_search",
    category="files",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex to find."},
            "path": {"type": "string", "description": "Root directory. Defaults to the workspace."},
            "glob": {"type": "string", "description": "File filter, e.g. '*.py'.", "default": "*"},
            "limit": {"type": "integer", "default": 60},
        },
        "required": ["pattern"],
    },
)
async def file_search(
    pattern: str, path: str = "", glob: str = "*", limit: int = 60, ctx: Any = None
) -> ToolOutput:
    """Search file contents for a pattern, returning file:line matches."""
    import re

    root = _resolve(path) if path else paths.workspace_dir()
    if not root.is_dir():
        raise ToolError("file_search", f"{root} is not a directory")
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ToolError("file_search", f"Invalid pattern: {exc}") from exc

    hits: list[str] = []
    for item in root.rglob(glob):
        if len(hits) >= limit:
            break
        if not item.is_file() or any(part in SKIP_DIRS for part in item.parts):
            continue
        if item.stat().st_size > 4 * 1024 * 1024:
            continue
        try:
            for number, line in enumerate(item.read_text("utf-8", errors="ignore").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{item}:{number}: {line.strip()[:180]}")
                    if len(hits) >= limit:
                        break
        except OSError:
            continue

    if not hits:
        return ToolOutput.text(f"No matches for '{pattern}' under {root}")
    return ToolOutput.text(f"{len(hits)} matches:\n" + "\n".join(hits), count=len(hits))


@registry.tool(
    "corpus_search",
    category="knowledge",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 8},
        },
        "required": ["query"],
    },
)
async def corpus_search(query: str, limit: int = 8, ctx: Any = None) -> ToolOutput:
    """Search ingested documents by meaning, not just keywords.

    This searches the user's own document corpus -- PDFs, notes, spreadsheets,
    web pages they have saved. Prefer it over guessing when a question is about
    their material. Cite the document title in your answer.
    """
    results = await pipeline.search_corpus(
        query, limit=limit, project_id=getattr(ctx, "project_id", None)
    )
    if not results:
        return ToolOutput.text(f"Nothing in the corpus matched '{query}'.")

    blocks = [f"{len(results)} passages for '{query}':"]
    for hit in results:
        location = f"{hit['document_title']}"
        if hit["heading"]:
            location += f" > {hit['heading']}"
        if hit["page"]:
            location += f" (p.{hit['page']})"
        blocks.append(f"\n--- {location}  [score {hit['score']:.3f}]\n{hit['text']}")
    return ToolOutput.text("\n".join(blocks), count=len(results))


@registry.tool(
    "corpus_ingest",
    category="knowledge",
    risk=Risk.WRITE,
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory to ingest."},
            "url": {"type": "string", "description": "Or a URL to fetch and ingest."},
        },
    },
)
async def corpus_ingest(path: str = "", url: str = "", ctx: Any = None) -> ToolOutput:
    """Add a file, folder or web page to the searchable corpus."""
    project_id = getattr(ctx, "project_id", None)
    if url:
        doc = await pipeline.ingest_url(url, project_id=project_id)
        return ToolOutput.text(f"Ingested '{doc.title}' ({doc.chunk_count} chunks) from {url}")
    if not path:
        raise ToolError("corpus_ingest", "Give a path or a url")

    target = _resolve(path)
    if target.is_dir():
        docs = await pipeline.ingest_directory(target, project_id=project_id)
        return ToolOutput.text(
            f"Ingested {len(docs)} files from {target}:\n"
            + "\n".join(f"- {d.title} ({d.chunk_count} chunks)" for d in docs[:30])
        )
    doc = await pipeline.ingest_file(target, project_id=project_id)
    return ToolOutput.text(f"Ingested '{doc.title}' ({doc.chunk_count} chunks)", id=doc.id)


@registry.tool(
    "corpus_list",
    category="knowledge",
    risk=Risk.SAFE,
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 40}},
    },
)
async def corpus_list(query: str = "", limit: int = 40, ctx: Any = None) -> ToolOutput:
    """List documents in the corpus, so you know what is available to search."""
    docs = await pipeline.list_documents(
        query=query, limit=limit, project_id=getattr(ctx, "project_id", None)
    )
    if not docs:
        return ToolOutput.text("The corpus is empty. Use corpus_ingest to add documents.")
    lines = [f"{len(docs)} documents:"]
    lines += [
        f"- {d.title}  [{d.status}, {d.chunk_count} chunks]  id={d.id}"
        for d in docs
    ]
    return ToolOutput.text("\n".join(lines))


@registry.tool(
    "workspace_path",
    category="files",
    risk=Risk.SAFE,
    parameters={"type": "object", "properties": {}},
)
async def workspace_path(ctx: Any = None) -> ToolOutput:
    """Where the workspace lives on disk -- the directory you can write to freely."""
    return ToolOutput.json(
        {
            "workspace": str(paths.workspace_dir()),
            "corpus": str(paths.corpus_dir()),
            "assets": str(paths.assets_dir()),
            "cwd": os.getcwd(),
        }
    )
