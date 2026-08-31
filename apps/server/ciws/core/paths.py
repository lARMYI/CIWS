"""Filesystem layout for CIWS.

Everything CIWS knows lives under a single root directory so the whole
workspace can be backed up, moved between machines, or wiped in one action.
Default root is ``~/.ciws``; override with the ``CIWS_HOME`` env var.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def home() -> Path:
    root = os.environ.get("CIWS_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".ciws"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    """SQLite database and vector index files."""
    return _sub("data")


def assets_dir() -> Path:
    """Generated and imported media (images, video, audio)."""
    return _sub("assets")


def corpus_dir() -> Path:
    """Original ingested documents, preserved byte-for-byte."""
    return _sub("corpus")


def logs_dir() -> Path:
    return _sub("logs")


def cache_dir() -> Path:
    return _sub("cache")


def workspace_dir() -> Path:
    """Scratch space agents are allowed to read and write."""
    return _sub("workspace")


def config_file() -> Path:
    return home() / "config.json"


def vault_file() -> Path:
    return home() / "vault.enc"


def key_file() -> Path:
    return home() / "vault.key"


def db_file() -> Path:
    return data_dir() / "ciws.db"


def describe() -> dict[str, str]:
    return {
        "home": str(home()),
        "data": str(data_dir()),
        "assets": str(assets_dir()),
        "corpus": str(corpus_dir()),
        "logs": str(logs_dir()),
        "cache": str(cache_dir()),
        "workspace": str(workspace_dir()),
        "database": str(db_file()),
    }
