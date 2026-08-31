"""Auth and shared request plumbing.

The server binds to loopback by default, which is most of the protection. The
token exists for the case that is not: any web page you visit can issue requests
to ``127.0.0.1``, and without a shared secret that page could drive your agents
and read your memory.

The token is generated on first run and written to ``CIWS_HOME/token``. The UI
reads it from the page bootstrap when served by this same server, so you never
type it. Set ``security.require_token`` to false only if you understand that a
random website can then talk to your hub.
"""

from __future__ import annotations

import secrets as pysecrets
from typing import Annotated

from fastapi import Depends, Header, Query, WebSocket

from ..core import paths
from ..core.config import get_settings
from ..core.errors import Unauthorized

_token: str | None = None


def get_token() -> str:
    """The session token, created on first use and persisted."""
    global _token
    if _token is not None:
        return _token
    token_file = paths.home() / "token"
    if token_file.exists():
        stored = token_file.read_text("utf-8").strip()
        if stored:
            _token = stored
            return _token
    _token = pysecrets.token_urlsafe(32)
    token_file.write_text(_token, "utf-8")
    try:
        token_file.chmod(0o600)
    except OSError:
        pass
    return _token


def rotate_token() -> str:
    global _token
    _token = None
    (paths.home() / "token").unlink(missing_ok=True)
    return get_token()


def _valid(candidate: str | None) -> bool:
    if not get_settings().security.require_token:
        return True
    if not candidate:
        return False
    return pysecrets.compare_digest(candidate.strip(), get_token())


async def require_auth(
    authorization: Annotated[str | None, Header()] = None,
    x_ciws_token: Annotated[str | None, Header()] = None,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Accept the token as a bearer header, a custom header, or a query param.

    The query param exists for EventSource and for opening an asset URL in a new
    tab, neither of which can set headers.
    """
    candidate = x_ciws_token or token
    if not candidate and authorization:
        scheme, _, value = authorization.partition(" ")
        candidate = value if scheme.lower() == "bearer" else authorization
    if not _valid(candidate):
        raise Unauthorized(
            "Missing or invalid token. The UI passes it automatically; for direct API "
            "calls send it as 'Authorization: Bearer <token>'. The token is in "
            f"{paths.home() / 'token'}."
        )


async def require_ws_auth(websocket: WebSocket) -> bool:
    """WebSocket auth, closing with a policy-violation code when it fails."""
    candidate = websocket.query_params.get("token") or websocket.headers.get("x-ciws-token")
    if _valid(candidate):
        return True
    await websocket.close(code=1008, reason="Invalid token")
    return False


#: Applied to every router that touches user data.
Auth = Depends(require_auth)
