"""Test fixtures.

Every test runs against a throwaway ``CIWS_HOME``, so the suite never touches a
real workspace and each test starts from an empty database.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session", autouse=True)
def isolated_home() -> Path:
    """Point CIWS_HOME at a temp directory before anything imports paths."""
    root = Path(tempfile.mkdtemp(prefix="ciws-test-"))
    os.environ["CIWS_HOME"] = str(root)
    # Never let a developer's real key leak into a test run.
    for name in list(os.environ):
        if name.endswith("_API_KEY") or name.endswith("_API_TOKEN"):
            os.environ.pop(name, None)
    return root


@pytest.fixture(autouse=True)
async def clean_db(isolated_home: Path):
    """Reset the schema between tests so ordering never matters."""
    from ciws.db.base import close_db, get_engine, init_db
    from ciws.db.models import Base
    from ciws.db.vectors import index

    await init_db()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await close_db()
    await init_db()
    index._kinds.clear()  # noqa: SLF001
    index._loaded = False  # noqa: SLF001
    yield
    await close_db()


@pytest.fixture
async def app():
    """The real ASGI app, without the boot lifespan.

    ``clean_db`` has already prepared the schema, and running ``_boot`` here
    would seed against a database the next test is about to drop. Subsystems a
    route needs (tools, agents, hubs) are seeded by the test that needs them.
    """
    from ciws.app import create_app

    return create_app()


@pytest.fixture
async def api(app):
    """An authenticated client. Sends the token the way the UI does."""
    import httpx

    from ciws.api.deps import get_token

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://ciws.test",
        headers={"x-ciws-token": get_token()},
    ) as client:
        yield client


@pytest.fixture
async def anon(app):
    """A client with no credentials, for asserting that auth actually bites."""
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ciws.test") as client:
        yield client


@pytest.fixture
def tools_loaded():
    from ciws.tools.registry import load_builtin_tools

    load_builtin_tools()
