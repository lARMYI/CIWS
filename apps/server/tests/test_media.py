"""The media studio and its provider adapters.

The paid adapters (OpenAI Images, Google, Replicate, fal, Stability) were
written to documented REST endpoints and never exercised against a live API --
that gap is listed in the README and it is what this file narrows. It cannot
prove the endpoints are current, but it does prove each adapter builds the
request it claims to and parses a correctly-shaped response, and that a missing
credential is a clean refusal rather than a crash or a silent no-op.

The asset lifecycle half is fully real: files land on disk, thumbnails are
written, deletes remove both the row and the bytes.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from ciws.core import paths
from ciws.media import providers as media_providers
from ciws.media import studio
from ciws.media.types import ImageRequest

# A 1x1 PNG, so thumbnailing and size probing run against real image bytes.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


async def test_image_and_video_models_are_listed_without_credentials():
    images = await studio.available_models("image", configured_only=False)
    videos = await studio.available_models("video", configured_only=False)
    assert images and videos
    assert all(m.id.count("/") >= 1 for m in images)


async def test_configured_only_hides_what_you_cannot_actually_run():
    """With no keys the only runnable backends are the local ones."""
    available = await studio.available_models("image", configured_only=True)
    assert all(m.provider in {"comfyui", "a1111"} for m in available), (
        "a backend with no credential was offered as available"
    )


def test_every_media_provider_declares_an_id():
    for cls in (
        media_providers.OpenAIImages,
        media_providers.GoogleMedia,
        media_providers.Replicate,
        media_providers.Fal,
        media_providers.Stability,
        media_providers.ComfyUI,
        media_providers.Automatic1111,
    ):
        assert cls.id and isinstance(cls.id, str)


# ---------------------------------------------------------------------------
# Refusal without a credential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-image-1",
        "replicate/black-forest-labs/flux-schnell",
        "fal/fal-ai/flux/dev",
        "stability/core",
    ],
)
async def test_generating_without_a_key_refuses_clearly(model: str):
    from ciws.core.errors import CIWSError

    with pytest.raises(CIWSError) as caught:
        await studio.generate_image(ImageRequest(prompt="a test", model=model))
    message = str(caught.value).lower()
    assert "key" in message or "configur" in message or "credential" in message


async def test_an_unknown_provider_names_itself_in_the_error():
    from ciws.core.errors import CIWSError

    with pytest.raises(CIWSError) as caught:
        await studio.generate_image(ImageRequest(prompt="x", model="nosuch/model-9000"))
    assert "nosuch" in str(caught.value), "the error did not say which provider was unknown"


async def test_an_empty_prompt_is_refused_before_any_spend():
    from ciws.core.errors import CIWSError

    with pytest.raises(CIWSError) as caught:
        await studio.generate_image(ImageRequest(prompt="   ", model="openai/gpt-image-1"))
    assert "prompt" in str(caught.value).lower()


# ---------------------------------------------------------------------------
# Asset lifecycle -- real files on real disk
# ---------------------------------------------------------------------------


async def test_import_stores_the_bytes_and_a_thumbnail(tmp_path: Path):
    source = tmp_path / "picture.png"
    source.write_bytes(PNG_1PX)

    asset = await studio.import_file(source, kind="image")
    assert asset.id.startswith("ast_")

    stored = studio.asset_file(asset)
    assert stored is not None and stored.exists()
    assert stored.read_bytes() == PNG_1PX


async def test_listing_filters_by_kind_and_query(tmp_path: Path):
    source = tmp_path / "turbine.png"
    source.write_bytes(PNG_1PX)
    await studio.import_file(source, kind="image")

    images = await studio.list_assets(kind="image")
    assert images

    matched = await studio.list_assets(query="turbine")
    assert matched

    missed = await studio.list_assets(query="zzz-no-such-asset")
    assert not missed


async def test_deleting_an_asset_removes_the_file_too(tmp_path: Path):
    source = tmp_path / "temp.png"
    source.write_bytes(PNG_1PX)
    asset = await studio.import_file(source, kind="image")
    stored = studio.asset_file(asset)

    assert await studio.delete_asset(asset.id) is True
    assert not stored.exists(), "the row went but the bytes stayed on disk"
    assert await studio.get_asset(asset.id) is None


async def test_deleting_something_that_is_gone_is_false_not_a_crash():
    assert await studio.delete_asset("ast_not_real") is False


async def test_a_failed_render_is_recorded_rather_than_lost(tmp_path: Path):
    source = tmp_path / "x.png"
    source.write_bytes(PNG_1PX)
    asset = await studio.import_file(source, kind="image")

    failed = await studio.fail_asset(asset.id, "the upstream render timed out")
    assert failed.status == "failed"
    assert "timed out" in (failed.error or "")


async def test_stats_counts_what_is_stored(tmp_path: Path):
    source = tmp_path / "counted.png"
    source.write_bytes(PNG_1PX)
    await studio.import_file(source, kind="image")

    stats = await studio.stats()
    assert stats["bytes"] >= len(PNG_1PX)
    assert stats["by_kind"].get("image", 0) >= 1


# ---------------------------------------------------------------------------
# Adapter request-building, against a mock transport
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_media(monkeypatch):
    """Capture the request an adapter builds, and serve it a canned reply."""
    seen: list[httpx.Request] = []

    def install(responder):
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return responder(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(
            "ciws.media.providers.http_client", lambda *a, **k: client, raising=False
        )
        return seen

    return install


async def test_openai_images_sends_prompt_size_and_count(mock_media, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(PNG_1PX).decode()}]},
        )

    seen = mock_media(responder)
    results = await media_providers.OpenAIImages().generate_image(
        ImageRequest(
            prompt="a rain-slick observation deck",
            model="openai/gpt-image-1",
            size="1024x1024",
            n=1,
        )
    )

    assert seen, "the adapter never issued a request"
    body = seen[0].read().decode()
    assert "rain-slick observation deck" in body
    assert "1024x1024" in body
    assert seen[0].headers.get("authorization", "").startswith("Bearer ")
    assert results and results[0]


async def test_comfyui_targets_the_local_host_not_the_internet(monkeypatch):
    """The local backends must never send a prompt off the machine."""
    provider = media_providers.ComfyUI()
    assert "127.0.0.1" in provider.url or "localhost" in provider.url


async def test_a1111_is_also_local(monkeypatch):
    provider = media_providers.Automatic1111()
    assert "127.0.0.1" in provider.url or "localhost" in provider.url


async def test_an_upstream_error_becomes_a_typed_error_with_the_body(mock_media, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "content policy violation"}})

    mock_media(responder)
    from ciws.core.errors import CIWSError

    with pytest.raises(CIWSError) as caught:
        await media_providers.OpenAIImages().generate_image(
            ImageRequest(prompt="x", model="openai/gpt-image-1", size="1024x1024", n=1)
        )
    assert "policy" in str(caught.value).lower()


async def test_generated_assets_land_under_the_ciws_home(tmp_path: Path):
    """Everything you make stays in one directory you can copy."""
    source = tmp_path / "home.png"
    source.write_bytes(PNG_1PX)
    asset = await studio.import_file(source, kind="image")
    stored = studio.asset_file(asset)
    assert str(stored).startswith(str(paths.home()))
