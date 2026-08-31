"""Runtime configuration.

Layered, lowest precedence first:

1. Built-in defaults (this file)
2. ``config.json`` in ``CIWS_HOME``   -- edited from the Settings panel
3. Environment variables / ``.env``   -- prefixed ``CIWS_``
4. Secret vault                       -- API keys only, encrypted at rest

Provider API keys are resolved separately by :mod:`ciws.core.secrets` so they
never land in a plaintext config file.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths


class RoutingPolicy(BaseModel):
    """Which model to reach for when a request asks for a capability, not a name."""

    fast: str = "anthropic/claude-haiku-4-5-20251001"
    balanced: str = "anthropic/claude-sonnet-5"
    deep: str = "anthropic/claude-opus-5"
    vision: str = "anthropic/claude-sonnet-5"
    embed: str = "local/hash-embed-768"
    local: str = "ollama/llama3.1"
    cheap: str = "anthropic/claude-haiku-4-5-20251001"
    fallbacks: list[str] = Field(default_factory=list)


class AgentDefaults(BaseModel):
    max_steps: int = 24
    max_tool_calls_per_step: int = 8
    temperature: float = 0.7
    timeout_seconds: int = 900
    parallel_tools: bool = True
    reflect_every: int = 0  # 0 disables periodic self-critique


class MemoryConfig(BaseModel):
    enabled: bool = True
    auto_capture: bool = True
    recall_limit: int = 12
    min_similarity: float = 0.18
    decay_half_life_days: float = 90.0
    importance_floor: float = 0.0
    extract_entities: bool = True
    consolidate_after: int = 200  # new memories before a consolidation pass


class ImproveConfig(BaseModel):
    """The main agent's self-improvement loops.

    Two different consent defaults on purpose. Directives are prompt text --
    bounded, visible, reversible in one click -- so they apply themselves.
    Skills are code the agent wrote; code waits for a human yes unless the
    user explicitly opts into auto-activation.
    """

    enabled: bool = True
    reflect_after_runs: int = 8          # completed runs before a reflection pass
    reflect_min_interval_minutes: int = 30
    max_directives: int = 10
    auto_apply_directives: bool = True
    auto_activate_skills: bool = False
    max_skills: int = 24


class SecurityConfig(BaseModel):
    """Local-first does not mean unguarded."""

    require_token: bool = True
    bind_host: str = "127.0.0.1"
    allow_shell_tool: bool = False
    allow_python_tool: bool = True
    approve_writes_outside_workspace: bool = True
    audit_all_tool_calls: bool = True
    redact_secrets_in_logs: bool = True


class MediaConfig(BaseModel):
    default_image_model: str = "openai/gpt-image-1"
    default_video_model: str = "google/veo-3.0-generate-001"
    default_image_size: str = "1024x1024"
    keep_originals: bool = True
    comfyui_url: str = "http://127.0.0.1:8188"
    a1111_url: str = "http://127.0.0.1:7860"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIWS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "CIWS"
    host: str = "127.0.0.1"
    port: int = 8787
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dev_mode: bool = False

    ollama_url: str = "http://127.0.0.1:11434"
    lmstudio_url: str = "http://127.0.0.1:1234/v1"
    vllm_url: str = ""

    routing: RoutingPolicy = Field(default_factory=RoutingPolicy)
    agents: AgentDefaults = Field(default_factory=AgentDefaults)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    improve: ImproveConfig = Field(default_factory=ImproveConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)

    def to_public_dict(self) -> dict[str, Any]:
        """Config safe to hand to the UI -- there are no secrets in here by design."""
        return json.loads(self.model_dump_json())


_settings: Settings | None = None


def _load_overrides() -> dict[str, Any]:
    f = paths.config_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(**_load_overrides())
    return _settings


def save_settings(patch: dict[str, Any]) -> Settings:
    """Merge ``patch`` into ``config.json`` and reload the live settings object."""
    global _settings
    merged = deep_merge(_load_overrides(), patch)
    paths.config_file().write_text(json.dumps(merged, indent=2), "utf-8")
    _settings = Settings(**merged)
    return _settings


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
