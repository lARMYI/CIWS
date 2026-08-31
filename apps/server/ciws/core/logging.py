"""Logging that never leaks a key.

Every record passes through a filter that scrubs known secret values before the
line reaches a file or a terminal. Logs are written to ``CIWS_HOME/logs`` and
mirrored onto the event bus so the Ops panel can tail them live.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Any

from rich.logging import RichHandler

from . import paths, secrets
from .events import Topic, bus

_configured = False


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - logging must never raise
            return True
        clean = secrets.redact(msg)
        if clean != msg:
            record.msg = clean
            record.args = ()
        return True


class BusHandler(logging.Handler):
    """Mirror WARNING+ onto the event bus for the Ops panel."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            bus.publish(
                Topic.SYSTEM_NOTICE,
                level=record.levelname,
                logger=record.name,
                message=secrets.redact(record.getMessage()),
            )
        except Exception:  # noqa: BLE001
            pass


def setup(level: str = "INFO", redact: bool = True) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setLevel(level)

    logfile = paths.logs_dir() / "ciws.log"
    rotating = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    rotating.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s")
    )
    rotating.setLevel(logging.DEBUG)

    mirror = BusHandler()
    mirror.setLevel(logging.WARNING)

    for h in (console, rotating, mirror):
        if redact:
            h.addFilter(RedactFilter())
        root.addHandler(h)

    for noisy in ("httpx", "httpcore", "urllib3", "watchfiles", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sys.excepthook = _excepthook
    _configured = True


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
    logging.getLogger("ciws").critical("Unhandled exception", exc_info=(exc_type, exc, tb))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ciws.{name}" if not name.startswith("ciws") else name)


def tail(lines: int = 200) -> list[str]:
    f = paths.logs_dir() / "ciws.log"
    if not f.exists():
        return []
    try:
        content = f.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]
