"""The FastAPI application.

Boot order matters and is explicit here: database, then vector index, then the
tool registry, then seeded content, then hubs. Each stage is wrapped so a
failure degrades the hub rather than preventing it from starting -- a broken MCP
server should not be the reason you cannot reach your own notes.

The built UI, when present, is served from this same origin. That is what makes
the token workable: the page is handed its token at load, so nothing has to be
copied or pasted, and a page from any other origin has no way to obtain it.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import routes_core, routes_knowledge, routes_work, ws
from .api.deps import get_token
from .core import logging as ciws_logging
from .core.config import get_settings
from .core.errors import CIWSError
from .core.events import Topic, bus
from .core.logging import get_logger

log = get_logger("app")

BANNER = r"""
   ___ ___ _   _ ___
  / __|_ _| | | / __|   Cognitive Intelligence Workspace System
 | (__ | || |/\| \__ \  local-first  ::  your models, your memory, your machine
  \___|___|\_/\_/|___/  v{version}
"""


def ui_dir() -> Path | None:
    """Where the built frontend lives, if it has been built."""
    candidates = [
        Path(__file__).parent / "static",                       # packaged
        Path(__file__).parent.parent.parent / "web" / "dist",   # monorepo dev
    ]
    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return None


async def _boot() -> None:
    """Bring the workspace up, stage by stage, tolerating partial failure."""
    from .db.base import init_db
    from .db.vectors import warm
    from .tools.registry import load_builtin_tools

    settings = get_settings()
    print(BANNER.format(version=__version__))

    await init_db()

    stages: list[tuple[str, Any]] = []

    try:
        count = await warm()
        stages.append(("vector index", f"{count} vectors"))
    except Exception as exc:  # noqa: BLE001
        stages.append(("vector index", f"unavailable ({exc})"))

    try:
        stages.append(("tools", f"{load_builtin_tools()} registered"))
    except Exception as exc:  # noqa: BLE001
        log.exception("Tool registration failed")
        stages.append(("tools", f"failed ({exc})"))

    try:
        from .agents import presets

        added = await presets.seed_presets()
        stages.append(("agents", f"{added} seeded" if added else "already present"))
    except Exception as exc:  # noqa: BLE001
        stages.append(("agents", f"failed ({exc})"))

    try:
        from .hubs import registry as hub_registry

        added = await hub_registry.seed_builtin_hubs()
        started = await hub_registry.manager.start_all()
        stages.append(("hubs", f"{added} seeded, {started} started"))
    except Exception as exc:  # noqa: BLE001
        stages.append(("hubs", f"failed ({exc})"))

    try:
        from .workflows import engine

        added = await engine.seed_examples()
        stages.append(("workflows", f"{added} examples" if added else "already present"))
    except Exception as exc:  # noqa: BLE001
        stages.append(("workflows", f"failed ({exc})"))

    try:
        from .memory import embeddings

        info = await embeddings.backend_info()
        stages.append(("embeddings", f"{info['backend']} ({info['quality']})"))
    except Exception as exc:  # noqa: BLE001
        stages.append(("embeddings", f"failed ({exc})"))

    for name, detail in stages:
        log.info("  %-14s %s", name, detail)

    url = f"http://{settings.host}:{settings.port}"
    if settings.security.require_token:
        log.info("  %-14s %s/?token=%s", "open", url, get_token())
    else:
        log.info("  %-14s %s  (token disabled)", "open", url)

    bus.publish(Topic.SYSTEM_NOTICE, level="INFO", message="CIWS is ready")


async def _shutdown() -> None:
    from .db.base import close_db
    from .gateway.base import close_clients
    from .hubs.registry import manager

    try:
        await manager.shutdown()
    except Exception as exc:  # noqa: BLE001
        log.debug("Hub shutdown: %s", exc)
    await close_clients()
    await close_db()
    log.info("CIWS stopped")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await _boot()
    yield
    await _shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    ciws_logging.setup(settings.log_level, redact=settings.security.redact_secrets_in_logs)

    app = FastAPI(
        title="CIWS",
        description="Cognitive Intelligence Workspace System -- a local-first agentic hub.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # The UI is same-origin in normal use; these origins cover a Vite dev server
    # and an Electron shell. Credentials are never cookie-based, so a permissive
    # list here does not weaken the token.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            f"http://{settings.host}:{settings.port}", "app://ciws",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(CIWSError)
    async def ciws_error_handler(request: Request, exc: CIWSError) -> JSONResponse:
        # Typed errors carry a status and a message meant for a human; anything
        # else would show the user a stack trace they cannot act on.
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    app.include_router(routes_core.public, prefix="/api")
    app.include_router(routes_core.router, prefix="/api")
    app.include_router(routes_work.router, prefix="/api")
    app.include_router(routes_knowledge.router, prefix="/api")
    app.include_router(ws.router, prefix="/api")

    static = ui_dir()
    if static is not None:
        _mount_ui(app, static)
    else:
        @app.get("/", response_class=HTMLResponse)
        async def no_ui() -> str:
            return _PLACEHOLDER.format(version=__version__, token=get_token())

    return app


def _mount_ui(app: FastAPI, static: Path) -> None:
    """Serve the built SPA, injecting the session token into the page."""
    assets = static / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index_html = (static / "index.html").read_text("utf-8")

    def page() -> str:
        settings = get_settings()
        bootstrap = json.dumps(
            {
                "token": get_token() if settings.security.require_token else "",
                "version": __version__,
                "requiresToken": settings.security.require_token,
            }
        )
        return index_html.replace(
            "</head>", f"<script>window.__CIWS__={bootstrap};</script></head>", 1
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page()

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def spa(full_path: str) -> Any:
        """Client-side routing: unknown paths fall through to the app shell."""
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": "not_found", "path": full_path})
        candidate = static / full_path
        if candidate.is_file():
            from fastapi.responses import FileResponse

            return FileResponse(candidate)
        return HTMLResponse(page())


_PLACEHOLDER = """<!doctype html>
<html><head><meta charset="utf-8"><title>CIWS</title>
<style>
 body {{ background:#07090c; color:#c8d3e0; font:14px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;
        display:grid; place-items:center; min-height:100vh; margin:0; }}
 .card {{ max-width:640px; padding:36px; border:1px solid #1b2530; border-radius:6px; background:#0b0f14; }}
 h1 {{ color:#22d3ee; font-size:20px; letter-spacing:.14em; margin:0 0 6px; }}
 code {{ background:#111820; padding:2px 7px; border-radius:3px; color:#7dd3fc; }}
 .dim {{ color:#5b6b7d; }}
 a {{ color:#22d3ee; }}
</style></head>
<body><div class="card">
<h1>CIWS v{version}</h1>
<p class="dim">The backend is running. The interface has not been built yet.</p>
<p>From the repository root:</p>
<p><code>cd apps/web && npm install && npm run build</code></p>
<p>Then restart the server. For live reloading during development, run
<code>npm run dev</code> and open <a href="http://localhost:5173">localhost:5173</a>.</p>
<p class="dim">The API is live at <a href="/api/docs">/api/docs</a>.<br>
Session token: <code>{token}</code></p>
</div></body></html>"""


app = create_app()
