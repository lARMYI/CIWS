"""Embeddings, with a floor that never falls out.

Three backends, tried in order of quality:

1. A provider embedding model (OpenAI, Google, Ollama) when one is configured.
2. ``sentence-transformers`` if it happens to be installed.
3. A deterministic hashed-feature projection computed right here.

The third one matters more than it looks. It has no dependencies, no network,
and no key, which means a brand-new CIWS install recalls memories and searches
documents on first launch. Semantic quality is lower than a trained encoder --
it captures lexical overlap and character-level similarity rather than meaning
-- but it is stable, instant, and free, and it degrades gracefully instead of
throwing.

Vectors from different backends are not comparable, so the index is keyed by
backend id and rebuilt when you switch.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

import numpy as np

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.util import chunks

log = get_logger("memory.embeddings")

HASH_DIM = 768
HASH_MODEL_ID = "local/hash-embed-768"
BATCH_SIZE = 96

_backend: str | None = None
_st_model: Any = None
_lock = asyncio.Lock()
_word_re = re.compile(r"[a-z0-9']+")

#: Words that carry almost no retrieval signal. Excluded from word and bigram
#: features so a query like "where is the data" is not dominated by "is" and
#: "the" -- with a small corpus, IDF weighting alone does not save you.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing done for from had has have he
her hers him his how i if in into is it its me my of on or our ours out she should so some
such than that the their them then there these they this those to too us was we were what
when where which while who whom why will with would you your yours am been being not no nor
""".split())


# ---------------------------------------------------------------------------
# Local hashed-feature backend
# ---------------------------------------------------------------------------


def _features(text: str) -> list[tuple[str, float]]:
    """Word unigrams, bigrams and character trigrams with sublinear weighting."""
    lowered = text.lower()
    words = _word_re.findall(lowered)
    if not words:
        words = [lowered[:32]] if lowered.strip() else ["\x00empty"]

    content = [w for w in words if w not in STOPWORDS and len(w) > 1] or words
    counts: dict[str, float] = {}
    for w in content:
        counts[f"w:{w}"] = counts.get(f"w:{w}", 0.0) + 1.0
    for a, b in zip(content, content[1:]):
        key = f"b:{a}_{b}"
        counts[key] = counts.get(key, 0.0) + 1.0

    # Character trigrams give partial credit for morphology and typos, which is
    # what keeps "authenticate" close to "authentication".
    padded = "  " + " ".join(content) + " "
    for i in range(len(padded) - 2):
        tri = padded[i : i + 3]
        if tri.strip():
            counts[f"c:{tri}"] = counts.get(f"c:{tri}", 0.0) + 0.35

    return [(k, 1.0 + math.log(v)) for k, v in counts.items()]


def hash_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """Signed feature hashing into a fixed-width vector.

    Two hashes per feature: one picks the bucket, one picks the sign. The sign
    trick makes collisions cancel out on average instead of accumulating.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for feature, weight in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        bucket = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[bucket] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _try_sentence_transformers() -> Any:
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embeddings: using sentence-transformers all-MiniLM-L6-v2")
        return _st_model
    except Exception as exc:  # noqa: BLE001 - model download can fail offline
        log.debug("sentence-transformers unavailable: %s", exc)
        return None


async def _resolve_backend() -> str:
    """Decide once which backend to use, and remember it."""
    global _backend
    async with _lock:
        if _backend is not None:
            return _backend

        configured = get_settings().routing.embed
        if configured and configured != HASH_MODEL_ID:
            try:
                from ..gateway.registry import gateway

                result = await gateway.embed(["probe"], configured)
                if result.vectors and result.vectors[0]:
                    _backend = configured
                    log.info("Embeddings: using provider model %s (dim %d)", configured, result.dim)
                    return _backend
            except Exception as exc:  # noqa: BLE001
                log.info("Embedding model %s unavailable (%s); falling back", configured, exc)

        if _try_sentence_transformers() is not None:
            _backend = "local/all-MiniLM-L6-v2"
            return _backend

        _backend = HASH_MODEL_ID
        log.info("Embeddings: using built-in hashed features (no provider configured)")
        return _backend


def reset_backend() -> None:
    """Forget the cached choice -- call after changing the embedding setting."""
    global _backend, _st_model
    _backend = None
    _st_model = None


async def embed_texts(texts: list[str], model: str = "") -> tuple[list[list[float]], str]:
    """Embed a batch. Returns ``(vectors, backend_id)``.

    Never raises: a provider failure mid-batch falls back to local embeddings
    for the whole batch, so the caller always gets vectors of one consistent
    kind rather than a mix it cannot compare.
    """
    if not texts:
        return [], await _resolve_backend()

    backend = model or await _resolve_backend()

    if backend == HASH_MODEL_ID:
        return [hash_embed(t) for t in texts], backend

    if backend.startswith("local/all-MiniLM"):
        st = _try_sentence_transformers()
        if st is not None:
            loop = asyncio.get_running_loop()
            arr = await loop.run_in_executor(
                None, lambda: st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            )
            return [list(map(float, row)) for row in arr], backend
        return [hash_embed(t) for t in texts], HASH_MODEL_ID

    try:
        from ..gateway.registry import gateway

        vectors: list[list[float]] = []
        for batch in chunks(texts, BATCH_SIZE):
            result = await gateway.embed(batch, backend)
            vectors.extend(result.vectors)
        if len(vectors) != len(texts):
            raise ValueError(f"provider returned {len(vectors)} vectors for {len(texts)} inputs")
        return vectors, backend
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding via %s failed (%s); using local fallback", backend, exc)
        return [hash_embed(t) for t in texts], HASH_MODEL_ID


async def embed_one(text: str, model: str = "") -> tuple[list[float], str]:
    vectors, backend = await embed_texts([text], model)
    return (vectors[0] if vectors else hash_embed(text)), backend


def dimension() -> int:
    if _backend == HASH_MODEL_ID or _backend is None:
        return HASH_DIM
    if _backend.startswith("local/all-MiniLM"):
        return 384
    from ..gateway import catalog

    info = catalog.lookup(_backend)
    return int(info.meta.get("dim", 1536)) if info else 1536


async def backend_info() -> dict[str, Any]:
    backend = await _resolve_backend()
    return {
        "backend": backend,
        "dim": dimension(),
        "local": backend.startswith("local/"),
        "quality": (
            "lexical"
            if backend == HASH_MODEL_ID
            else "semantic (local)"
            if backend.startswith("local/")
            else "semantic (provider)"
        ),
        "note": (
            "No embedding provider is configured, so recall uses built-in hashed "
            "features. Set an embedding model in Settings for stronger semantic search."
            if backend == HASH_MODEL_ID
            else ""
        ),
    }
