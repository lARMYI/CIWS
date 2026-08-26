"""Command line entry point: ``ciws`` or ``python -m ciws``."""

from __future__ import annotations

import argparse
import sys
import webbrowser


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ciws",
        description="CIWS - Cognitive Intelligence Workspace System.",
    )
    parser.add_argument("--host", default=None, help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Port (default 8787).")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    parser.add_argument("--open", action="store_true", help="Open the UI in your browser.")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-token", action="store_true",
                        help="Disable the auth token for this run. Loopback only.")
    parser.add_argument("--token", action="store_true", help="Print the session token and exit.")
    parser.add_argument("--home", default=None, help="Override CIWS_HOME for this run.")
    args = parser.parse_args()

    if args.home:
        import os

        os.environ["CIWS_HOME"] = args.home

    if args.token:
        from .api.deps import get_token

        print(get_token())
        return 0

    from .core.config import get_settings, save_settings

    patch = {}
    if args.host:
        patch["host"] = args.host
    if args.port:
        patch["port"] = args.port
    if args.log_level:
        patch["log_level"] = args.log_level
    if args.no_token:
        patch["security"] = {"require_token": False}
    if patch:
        save_settings(patch)

    settings = get_settings()
    url = f"http://{settings.host}:{settings.port}"

    if args.open:
        import threading

        # Give uvicorn a moment to bind before the browser races it.
        threading.Timer(1.8, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "ciws.app:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        log_config=None,  # our own logging is already configured
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
