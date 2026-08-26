"""Seed model catalog.

Two sources of truth, deliberately kept apart:

* **Live discovery** -- every provider is asked what it can serve right now
  (``GET /v1/models``, ``ollama list``, OpenRouter's catalog). This is
  authoritative for *which* models exist.
* **This file** -- metadata the model-list endpoints do not return: pricing,
  request quirks, capability flags.

Where a provider publishes pricing itself (OpenRouter does; Ollama is free),
live data wins. Where a number here is left at ``0.0`` it means "not seeded" --
the Models panel lets you type the real figure, and cost tracking simply reads
zero until you do rather than showing an invented number.

Anthropic entries carry request quirks in ``meta`` because the newer models
*reject* parameters the older ones require. Sending ``temperature`` to Opus 5
is a 400, not a soft ignore.
"""

from __future__ import annotations

from .types import Capability, Modality, ModelInfo

TEXT = [Modality.TEXT]
TEXT_IMAGE = [Modality.TEXT, Modality.IMAGE]

CORE = [Capability.TOOLS, Capability.STREAMING, Capability.JSON]
CORE_VISION = [*CORE, Capability.VISION]
CORE_VISION_THINK = [*CORE_VISION, Capability.THINKING, Capability.CACHING]


def _anthropic(
    name: str,
    display: str,
    ctx: int,
    max_out: int,
    cin: float,
    cout: float,
    *,
    adaptive_thinking: bool = True,
    sampling: bool = False,
    effort: bool = True,
    thinking_always_on: bool = False,
) -> ModelInfo:
    """Anthropic model entry.

    ``sampling`` records whether ``temperature`` / ``top_p`` are accepted --
    they were removed on Opus 5, Opus 4.7/4.8, Sonnet 5 and Fable 5, where
    sending them returns a 400. ``adaptive_thinking`` selects
    ``{"type": "adaptive"}`` over the older ``budget_tokens`` form.
    """
    return ModelInfo(
        id=f"anthropic/{name}",
        provider="anthropic",
        name=name,
        display_name=display,
        family="claude",
        context_window=ctx,
        max_output=max_out,
        modalities=TEXT_IMAGE,
        capabilities=CORE_VISION_THINK,
        input_cost_per_mtok=cin,
        output_cost_per_mtok=cout,
        meta={
            "adaptive_thinking": adaptive_thinking,
            "sampling_allowed": sampling,
            "effort_supported": effort,
            "thinking_always_on": thinking_always_on,
            "prefill_allowed": False,
        },
    )


#: Anthropic catalogue. Figures from the Claude API reference (cached 2026-06-24).
ANTHROPIC_MODELS: list[ModelInfo] = [
    _anthropic("claude-opus-5", "Claude Opus 5", 1_000_000, 128_000, 5.00, 25.00),
    _anthropic("claude-fable-5", "Claude Fable 5", 1_000_000, 128_000, 10.00, 50.00, thinking_always_on=True),
    _anthropic("claude-sonnet-5", "Claude Sonnet 5", 1_000_000, 128_000, 2.00, 10.00),
    _anthropic("claude-opus-4-8", "Claude Opus 4.8", 1_000_000, 128_000, 5.00, 25.00),
    _anthropic("claude-opus-4-7", "Claude Opus 4.7", 1_000_000, 128_000, 5.00, 25.00),
    _anthropic("claude-opus-4-6", "Claude Opus 4.6", 1_000_000, 128_000, 5.00, 25.00, sampling=True),
    _anthropic("claude-sonnet-4-6", "Claude Sonnet 4.6", 1_000_000, 128_000, 3.00, 15.00, sampling=True),
    # Haiku 4.5 predates adaptive thinking: it still wants {"type":"enabled","budget_tokens":N}.
    _anthropic(
        "claude-haiku-4-5", "Claude Haiku 4.5", 200_000, 64_000, 1.00, 5.00,
        adaptive_thinking=False, sampling=True, effort=False,
    ),
]

#: Aliases people actually type, mapped onto catalogue ids.
ALIASES: dict[str, str] = {
    "opus": "anthropic/claude-opus-5",
    "opus-5": "anthropic/claude-opus-5",
    "sonnet": "anthropic/claude-sonnet-5",
    "haiku": "anthropic/claude-haiku-4-5",
    "fable": "anthropic/claude-fable-5",
    "claude": "anthropic/claude-opus-5",
    "anthropic/claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5",
    "gpt": "openai/gpt-4o",
    "gemini": "google/gemini-2.0-flash",
}


def _openai(
    name: str, display: str, ctx: int, max_out: int, cin: float, cout: float,
    caps: list[Capability] | None = None, mods: list[Modality] | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=f"openai/{name}", provider="openai", name=name, display_name=display,
        family="gpt", context_window=ctx, max_output=max_out,
        modalities=mods or TEXT_IMAGE, capabilities=caps or CORE_VISION,
        input_cost_per_mtok=cin, output_cost_per_mtok=cout,
    )


#: Widely-published OpenAI figures. Live ``/v1/models`` adds anything newer.
OPENAI_MODELS: list[ModelInfo] = [
    _openai("gpt-4o", "GPT-4o", 128_000, 16_384, 2.50, 10.00),
    _openai("gpt-4o-mini", "GPT-4o mini", 128_000, 16_384, 0.15, 0.60),
    _openai("gpt-4.1", "GPT-4.1", 1_047_576, 32_768, 2.00, 8.00),
    _openai("gpt-4.1-mini", "GPT-4.1 mini", 1_047_576, 32_768, 0.40, 1.60),
    _openai("o3-mini", "o3-mini", 200_000, 100_000, 1.10, 4.40, caps=[*CORE, Capability.THINKING], mods=TEXT),
    ModelInfo(
        id="openai/text-embedding-3-small", provider="openai", name="text-embedding-3-small",
        display_name="Embedding 3 Small", family="embedding", context_window=8191,
        modalities=TEXT, capabilities=[Capability.EMBEDDING],
        input_cost_per_mtok=0.02, meta={"dim": 1536},
    ),
    ModelInfo(
        id="openai/text-embedding-3-large", provider="openai", name="text-embedding-3-large",
        display_name="Embedding 3 Large", family="embedding", context_window=8191,
        modalities=TEXT, capabilities=[Capability.EMBEDDING],
        input_cost_per_mtok=0.13, meta={"dim": 3072},
    ),
]

GOOGLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="google/gemini-2.0-flash", provider="google", name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash", family="gemini", context_window=1_048_576,
        max_output=8192, modalities=[Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO],
        capabilities=CORE_VISION, input_cost_per_mtok=0.10, output_cost_per_mtok=0.40,
    ),
    ModelInfo(
        id="google/gemini-2.0-flash-lite", provider="google", name="gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash Lite", family="gemini", context_window=1_048_576,
        max_output=8192, modalities=TEXT_IMAGE, capabilities=CORE_VISION,
        input_cost_per_mtok=0.075, output_cost_per_mtok=0.30,
    ),
    ModelInfo(
        id="google/text-embedding-004", provider="google", name="text-embedding-004",
        display_name="Google Text Embedding 004", family="embedding", context_window=2048,
        modalities=TEXT, capabilities=[Capability.EMBEDDING], meta={"dim": 768},
    ),
]

#: Providers whose catalogue is entirely live. Seeding them would go stale fast
#: and their ``/v1/models`` endpoints are reliable.
LIVE_ONLY_PROVIDERS = {
    "ollama", "lmstudio", "openrouter", "groq", "mistral",
    "deepseek", "xai", "together", "vllm",
}

SEED_MODELS: list[ModelInfo] = [*ANTHROPIC_MODELS, *OPENAI_MODELS, *GOOGLE_MODELS]

_BY_ID: dict[str, ModelInfo] = {m.id: m for m in SEED_MODELS}


def seed_for(provider: str) -> list[ModelInfo]:
    return [m for m in SEED_MODELS if m.provider == provider]


def lookup(model_id: str) -> ModelInfo | None:
    """Resolve a model id or alias against the seed catalogue."""
    if model_id in _BY_ID:
        return _BY_ID[model_id]
    resolved = ALIASES.get(model_id)
    if resolved and resolved in _BY_ID:
        return _BY_ID[resolved]
    # Bare name with no provider prefix -- accept the first unambiguous match.
    if "/" not in model_id:
        hits = [m for m in SEED_MODELS if m.name == model_id]
        if len(hits) == 1:
            return hits[0]
    return None


def quirks(model_id: str) -> dict[str, object]:
    """Request-shaping flags for a model, with permissive defaults.

    Unknown models get the conservative reading: sampling allowed (nearly every
    provider accepts ``temperature``), no adaptive thinking.
    """
    info = lookup(model_id)
    if info is None:
        return {"sampling_allowed": True, "adaptive_thinking": False, "effort_supported": False}
    base = {"sampling_allowed": True, "adaptive_thinking": False, "effort_supported": False}
    base.update(info.meta)
    return base


def split_id(model_id: str) -> tuple[str, str]:
    """``"anthropic/claude-opus-5"`` -> ``("anthropic", "claude-opus-5")``."""
    resolved = ALIASES.get(model_id, model_id)
    if "/" in resolved:
        provider, _, name = resolved.partition("/")
        return provider, name
    return "", resolved


def merge(live: list[ModelInfo], provider: str) -> list[ModelInfo]:
    """Overlay seed metadata (pricing, quirks) onto live-discovered models."""
    seeded = {m.id: m for m in seed_for(provider)}
    out: list[ModelInfo] = []
    seen: set[str] = set()
    for model in live:
        seen.add(model.id)
        base = seeded.get(model.id)
        if base is None:
            out.append(model)
            continue
        merged = model.model_copy(deep=True)
        if not merged.input_cost_per_mtok:
            merged.input_cost_per_mtok = base.input_cost_per_mtok
        if not merged.output_cost_per_mtok:
            merged.output_cost_per_mtok = base.output_cost_per_mtok
        if not merged.context_window:
            merged.context_window = base.context_window
        if not merged.max_output:
            merged.max_output = base.max_output
        if not merged.display_name:
            merged.display_name = base.display_name
        if not merged.capabilities:
            merged.capabilities = base.capabilities
        merged.meta = {**base.meta, **merged.meta}
        out.append(merged)
    # Keep seeded models the live endpoint did not list -- some providers only
    # return models you have already used.
    for mid, model in seeded.items():
        if mid not in seen:
            out.append(model)
    return out
