"""Encrypted secret vault for provider credentials.

API keys are the crown jewels of a hub like this -- one file holds the keys to
every model, every media generator, every connector. They are stored in
``vault.enc`` encrypted with Fernet (AES-128-CBC + HMAC). The key lives beside
it in ``vault.key`` with ``0600`` permissions.

That protects against the realistic local threat: a synced folder, a backup
tarball, a screen-share of a config file, another user account on the same
machine. It does not protect against malware already running as you -- nothing
stored on the same disk can. Environment variables always win over the vault,
so an OS keychain or a secrets manager can front it if you want that.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from . import paths

_lock = threading.RLock()
_cache: dict[str, str] | None = None

#: Environment variables consulted before the vault, per provider id.
ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "perplexity": ("PERPLEXITY_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
    "replicate": ("REPLICATE_API_TOKEN",),
    "fal": ("FAL_KEY", "FAL_API_KEY"),
    "stability": ("STABILITY_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
    "brave": ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"),
    "serper": ("SERPER_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "github": ("GITHUB_TOKEN", "GH_TOKEN"),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
}


def _fernet() -> Fernet:
    kf = paths.key_file()
    if not kf.exists():
        kf.write_bytes(Fernet.generate_key())
        try:
            kf.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Windows filesystems without POSIX modes
    return Fernet(kf.read_bytes())


def _load() -> dict[str, str]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        vf = paths.vault_file()
        if not vf.exists():
            _cache = {}
            return _cache
        try:
            raw = _fernet().decrypt(vf.read_bytes())
            _cache = json.loads(raw.decode("utf-8"))
        except (InvalidToken, ValueError, OSError):
            # A corrupt or key-mismatched vault should degrade to "no secrets",
            # never crash the hub on boot.
            _cache = {}
        return _cache


def _flush(data: dict[str, str]) -> None:
    global _cache
    with _lock:
        blob = _fernet().encrypt(json.dumps(data).encode("utf-8"))
        vf = paths.vault_file()
        vf.write_bytes(blob)
        try:
            vf.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        _cache = data


def get(name: str) -> str | None:
    """Resolve a credential: environment first, then the vault."""
    for env in ENV_ALIASES.get(name, ()):
        val = os.environ.get(env)
        if val:
            return val.strip()
    direct = os.environ.get(f"CIWS_KEY_{name.upper()}")
    if direct:
        return direct.strip()
    val = _load().get(name)
    return val.strip() if val else None


def put(name: str, value: str) -> None:
    data = dict(_load())
    if value:
        data[name] = value.strip()
    else:
        data.pop(name, None)
    _flush(data)


def delete(name: str) -> None:
    data = dict(_load())
    if data.pop(name, None) is not None:
        _flush(data)


def has(name: str) -> bool:
    return bool(get(name))


def source_of(name: str) -> str:
    """Where a credential came from -- shown in the UI so you can trust it."""
    for env in ENV_ALIASES.get(name, ()):
        if os.environ.get(env):
            return f"env:{env}"
    if os.environ.get(f"CIWS_KEY_{name.upper()}"):
        return f"env:CIWS_KEY_{name.upper()}"
    if _load().get(name):
        return "vault"
    return "missing"


def mask(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def status() -> list[dict[str, Any]]:
    """Non-secret summary of every known credential slot."""
    names = sorted(set(ENV_ALIASES) | set(_load()))
    out = []
    for n in names:
        val = get(n)
        out.append(
            {
                "name": n,
                "configured": bool(val),
                "masked": mask(val),
                "source": source_of(n),
                "env_names": list(ENV_ALIASES.get(n, ())),
            }
        )
    return out


def all_values() -> list[str]:
    """Every live secret value -- used only to redact them out of logs."""
    vals = [v for v in _load().values() if v]
    for aliases in ENV_ALIASES.values():
        for env in aliases:
            v = os.environ.get(env)
            if v:
                vals.append(v)
    return [v for v in vals if len(v) >= 8]


def redact(text: str) -> str:
    """Replace any known secret substring with a marker."""
    if not text:
        return text
    for v in all_values():
        if v in text:
            text = text.replace(v, "[REDACTED]")
    return text
