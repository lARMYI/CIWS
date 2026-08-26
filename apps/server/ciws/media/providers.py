"""Image and video generation backends.

Two shapes of API here. The synchronous ones (OpenAI images, Stability,
Automatic1111) return pixels on the same request. The asynchronous ones
(Replicate, fal, Veo, ComfyUI) hand back a job id and expect polling -- which is
where most of the fiddly code lives, since each one spells "still working"
differently and none of them are consistent about where the output URL appears.

Every poll loop publishes progress and honours a wall-clock ceiling, so a stuck
render surfaces as a failed asset instead of a coroutine that never returns.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

from ..core import secrets
from ..core.config import get_settings
from ..core.errors import MissingCredential, ProviderError
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..gateway.base import http_client, raise_for_status
from .types import GenerationResult, ImageRequest, MediaModel, VideoRequest

log = get_logger("media.providers")

IMAGE_POLL_TIMEOUT = 300.0
VIDEO_POLL_TIMEOUT = 900.0


class MediaProvider:
    id = "provider"
    label = "Provider"
    kinds: tuple[str, ...] = ("image",)
    requires_key = True
    local = False
    env_hint = ""

    def configured(self) -> bool:
        return (not self.requires_key) or bool(secrets.get(self.id))

    def key(self) -> str:
        value = secrets.get(self.id)
        if not value:
            raise MissingCredential(self.id, self.env_hint)
        return value

    async def models(self) -> list[MediaModel]:
        return []

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        raise ProviderError(self.id, f"{self.label} does not generate images", retryable=False)

    async def generate_video(self, req: VideoRequest) -> GenerationResult:
        raise ProviderError(self.id, f"{self.label} does not generate video", retryable=False)

    def _progress(self, **data: Any) -> None:
        bus.publish(Topic.MEDIA_PROGRESS, provider=self.id, **data)


def _model_name(model_id: str) -> str:
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


async def _poll(
    describe: str,
    check: Any,
    *,
    timeout: float,
    interval: float = 2.0,
    provider: str = "",
) -> Any:
    """Call ``check()`` until it returns non-None or the clock runs out."""
    started = time.monotonic()
    attempt = 0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            raise ProviderError(
                provider or "media", f"{describe} timed out after {int(elapsed)}s", retryable=True
            )
        result = await check()
        if result is not None:
            return result
        attempt += 1
        bus.publish(
            Topic.MEDIA_PROGRESS,
            provider=provider,
            stage=describe,
            elapsed_s=round(elapsed, 1),
            attempt=attempt,
        )
        # Back off gently: fast early polls feel responsive, slow later ones
        # stop hammering a render that clearly needs minutes.
        await asyncio.sleep(min(10.0, interval * (1.0 + attempt * 0.15)))


# ---------------------------------------------------------------------------
# OpenAI images
# ---------------------------------------------------------------------------


class OpenAIImages(MediaProvider):
    id = "openai"
    label = "OpenAI Images"
    kinds = ("image",)
    env_hint = "OPENAI_API_KEY"

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="openai/gpt-image-1", provider=self.id, name="gpt-image-1",
                display_name="GPT Image 1", kind="image",
                sizes=["1024x1024", "1536x1024", "1024x1536"],
                supports_image_input=True,
                description="Strong prompt adherence and legible text rendering.",
                params_schema={"quality": ["auto", "low", "medium", "high"]},
            ),
            MediaModel(
                id="openai/dall-e-3", provider=self.id, name="dall-e-3",
                display_name="DALL-E 3", kind="image",
                sizes=["1024x1024", "1792x1024", "1024x1792"],
                description="Illustrative and stylised output.",
                params_schema={"quality": ["standard", "hd"], "style": ["vivid", "natural"]},
            ),
        ]

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        name = _model_name(req.model) or "gpt-image-1"
        client = http_client("openai-images", timeout=IMAGE_POLL_TIMEOUT)
        headers = {"authorization": f"Bearer {self.key()}", "content-type": "application/json"}

        payload: dict[str, Any] = {
            "model": name,
            "prompt": req.prompt,
            "n": min(req.n, 4) if name != "dall-e-3" else 1,  # dall-e-3 only ever returns one
            "size": req.size,
        }
        if req.quality:
            payload["quality"] = req.quality
        if req.style and name == "dall-e-3":
            payload["style"] = req.style
        if name == "dall-e-3":
            payload["response_format"] = "b64_json"

        url = "https://api.openai.com/v1/images/generations"
        if req.init_image_b64:
            # Edits take multipart, not JSON.
            files = {
                "image": ("init.png", base64.b64decode(req.init_image_b64), "image/png"),
            }
            data = {"model": name, "prompt": req.prompt, "n": str(payload["n"]), "size": req.size}
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={"authorization": f"Bearer {self.key()}"},
                files=files,
                data=data,
            )
        else:
            response = await client.post(url, headers=headers, json=payload)

        raise_for_status(self.id, response)
        body = response.json()
        width, height = req.dimensions()
        out: list[GenerationResult] = []
        for item in body.get("data") or []:
            b64 = item.get("b64_json")
            out.append(
                GenerationResult(
                    data=base64.b64decode(b64) if b64 else None,
                    remote_url=item.get("url", ""),
                    mime_type="image/png",
                    width=width,
                    height=height,
                    meta={"revised_prompt": item.get("revised_prompt", "")},
                )
            )
        if not out:
            raise ProviderError(self.id, "OpenAI returned no images")
        return out


# ---------------------------------------------------------------------------
# Google (Imagen + Veo)
# ---------------------------------------------------------------------------


class GoogleMedia(MediaProvider):
    id = "google"
    label = "Google Imagen / Veo"
    kinds = ("image", "video")
    env_hint = "GOOGLE_API_KEY or GEMINI_API_KEY"
    base = "https://generativelanguage.googleapis.com/v1beta"

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="google/imagen-3.0-generate-002", provider=self.id,
                name="imagen-3.0-generate-002", display_name="Imagen 3", kind="image",
                sizes=["1024x1024"], supports_negative_prompt=True,
                description="Photorealistic image generation.",
                params_schema={"aspectRatio": ["1:1", "3:4", "4:3", "9:16", "16:9"]},
            ),
            MediaModel(
                id="google/veo-3.0-generate-001", provider=self.id,
                name="veo-3.0-generate-001", display_name="Veo 3", kind="video",
                max_duration_s=8.0, supports_image_input=True, supports_negative_prompt=True,
                description="Text-to-video with audio.",
                params_schema={"aspectRatio": ["16:9", "9:16"]},
            ),
        ]

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.key(), "content-type": "application/json"}

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        name = _model_name(req.model) or "imagen-3.0-generate-002"
        client = http_client("google-media", timeout=IMAGE_POLL_TIMEOUT)
        params: dict[str, Any] = {
            "sampleCount": min(req.n, 4),
            "aspectRatio": req.params.get("aspectRatio", "1:1"),
        }
        if req.negative_prompt:
            params["negativePrompt"] = req.negative_prompt
        if req.seed is not None:
            params["seed"] = req.seed

        response = await client.post(
            f"{self.base}/models/{name}:predict",
            headers=self._headers(),
            json={"instances": [{"prompt": req.prompt}], "parameters": params},
        )
        raise_for_status(self.id, response)
        width, height = req.dimensions()
        out = []
        for pred in response.json().get("predictions") or []:
            b64 = pred.get("bytesBase64Encoded") or pred.get("image", {}).get("bytesBase64Encoded")
            if not b64:
                continue
            out.append(
                GenerationResult(
                    data=base64.b64decode(b64),
                    mime_type=pred.get("mimeType", "image/png"),
                    width=width,
                    height=height,
                )
            )
        if not out:
            raise ProviderError(self.id, "Imagen returned no images")
        return out

    async def generate_video(self, req: VideoRequest) -> GenerationResult:
        name = _model_name(req.model) or "veo-3.0-generate-001"
        client = http_client("google-media", timeout=VIDEO_POLL_TIMEOUT)

        instance: dict[str, Any] = {"prompt": req.prompt}
        if req.init_image_b64:
            instance["image"] = {"bytesBase64Encoded": req.init_image_b64, "mimeType": "image/png"}
        params: dict[str, Any] = {"aspectRatio": req.aspect_ratio}
        if req.negative_prompt:
            params["negativePrompt"] = req.negative_prompt

        start = await client.post(
            f"{self.base}/models/{name}:predictLongRunning",
            headers=self._headers(),
            json={"instances": [instance], "parameters": params},
        )
        raise_for_status(self.id, start)
        operation = start.json().get("name", "")
        if not operation:
            raise ProviderError(self.id, "Veo did not return an operation name")

        async def check() -> Any:
            poll = await client.get(f"{self.base}/{operation}", headers=self._headers())
            raise_for_status(self.id, poll)
            body = poll.json()
            if not body.get("done"):
                return None
            if body.get("error"):
                raise ProviderError(self.id, str(body["error"])[:300], retryable=False)
            return body.get("response") or {}

        response = await _poll(
            "Veo render", check, timeout=VIDEO_POLL_TIMEOUT, interval=6.0, provider=self.id
        )

        samples = (
            response.get("generatedSamples")
            or response.get("generateVideoResponse", {}).get("generatedSamples")
            or response.get("predictions")
            or []
        )
        for sample in samples:
            video = sample.get("video") or sample
            uri = video.get("uri") or video.get("url") or ""
            b64 = video.get("bytesBase64Encoded")
            if b64:
                return GenerationResult(
                    data=base64.b64decode(b64), mime_type="video/mp4",
                    duration_s=req.duration_s, job_id=operation,
                )
            if uri:
                # The download URL still needs the API key.
                fetch = await client.get(uri, headers={"x-goog-api-key": self.key()})
                raise_for_status(self.id, fetch)
                return GenerationResult(
                    data=fetch.content, mime_type="video/mp4",
                    duration_s=req.duration_s, job_id=operation,
                )
        raise ProviderError(self.id, "Veo finished but returned no video")


# ---------------------------------------------------------------------------
# Replicate
# ---------------------------------------------------------------------------


class Replicate(MediaProvider):
    id = "replicate"
    label = "Replicate"
    kinds = ("image", "video")
    env_hint = "REPLICATE_API_TOKEN"
    base = "https://api.replicate.com/v1"

    async def models(self) -> list[MediaModel]:
        # Replicate hosts thousands of models; these are the ones worth defaulting to.
        # Any `owner/name` slug works via the custom-model field in the Studio.
        return [
            MediaModel(
                id="replicate/black-forest-labs/flux-1.1-pro", provider=self.id,
                name="black-forest-labs/flux-1.1-pro", display_name="FLUX 1.1 Pro",
                kind="image", sizes=["1024x1024", "1440x1024", "1024x1440"],
                description="High-fidelity image generation.",
            ),
            MediaModel(
                id="replicate/black-forest-labs/flux-schnell", provider=self.id,
                name="black-forest-labs/flux-schnell", display_name="FLUX Schnell",
                kind="image", sizes=["1024x1024"], description="Fast and inexpensive.",
            ),
            MediaModel(
                id="replicate/stability-ai/stable-video-diffusion", provider=self.id,
                name="stability-ai/stable-video-diffusion", display_name="Stable Video Diffusion",
                kind="video", max_duration_s=4.0, supports_image_input=True,
                description="Image-to-video.",
            ),
        ]

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.key()}", "content-type": "application/json"}

    async def _run(self, model: str, payload: dict[str, Any], timeout: float) -> Any:
        client = http_client("replicate", timeout=timeout)
        if ":" in model:  # pinned version hash
            body = {"version": model.split(":", 1)[1], "input": payload}
            url = f"{self.base}/predictions"
        else:
            body = {"input": payload}
            url = f"{self.base}/models/{model}/predictions"

        start = await client.post(url, headers=self._headers(), json=body)
        raise_for_status(self.id, start)
        prediction = start.json()
        poll_url = (prediction.get("urls") or {}).get("get") or f"{self.base}/predictions/{prediction.get('id')}"

        async def check() -> Any:
            poll = await client.get(poll_url, headers=self._headers())
            raise_for_status(self.id, poll)
            data = poll.json()
            status = data.get("status")
            if status in ("succeeded",):
                return data
            if status in ("failed", "canceled"):
                raise ProviderError(
                    self.id, f"Prediction {status}: {data.get('error') or 'no detail'}",
                    retryable=False,
                )
            return None

        return await _poll("Replicate prediction", check, timeout=timeout, provider=self.id)

    async def _download(self, url: str, timeout: float) -> bytes:
        client = http_client("replicate", timeout=timeout)
        response = await client.get(url)
        raise_for_status(self.id, response)
        return response.content

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        model = _model_name(req.model) or "black-forest-labs/flux-schnell"
        width, height = req.dimensions()
        payload: dict[str, Any] = {
            "prompt": req.prompt, "width": width, "height": height,
            "num_outputs": min(req.n, 4), **req.params,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.init_image_b64:
            payload["image"] = f"data:image/png;base64,{req.init_image_b64}"
            payload["prompt_strength"] = req.strength

        data = await self._run(model, payload, IMAGE_POLL_TIMEOUT)
        output = data.get("output")
        urls = output if isinstance(output, list) else [output]
        results = []
        for url in [u for u in urls if isinstance(u, str)]:
            results.append(
                GenerationResult(
                    data=await self._download(url, IMAGE_POLL_TIMEOUT),
                    remote_url=url, mime_type="image/png",
                    width=width, height=height, job_id=str(data.get("id", "")),
                )
            )
        if not results:
            raise ProviderError(self.id, "Replicate returned no output")
        return results

    async def generate_video(self, req: VideoRequest) -> GenerationResult:
        model = _model_name(req.model) or "stability-ai/stable-video-diffusion"
        payload: dict[str, Any] = {"prompt": req.prompt, **req.params}
        if req.init_image_b64:
            payload["input_image"] = f"data:image/png;base64,{req.init_image_b64}"
        if req.seed is not None:
            payload["seed"] = req.seed

        data = await self._run(model, payload, VIDEO_POLL_TIMEOUT)
        output = data.get("output")
        url = output[0] if isinstance(output, list) and output else output
        if not isinstance(url, str):
            raise ProviderError(self.id, "Replicate returned no video URL")
        return GenerationResult(
            data=await self._download(url, VIDEO_POLL_TIMEOUT),
            remote_url=url, mime_type="video/mp4",
            duration_s=req.duration_s, job_id=str(data.get("id", "")),
        )


# ---------------------------------------------------------------------------
# fal.ai
# ---------------------------------------------------------------------------


class Fal(MediaProvider):
    id = "fal"
    label = "fal.ai"
    kinds = ("image", "video")
    env_hint = "FAL_KEY"

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="fal/fal-ai/flux/dev", provider=self.id, name="fal-ai/flux/dev",
                display_name="FLUX dev", kind="image", sizes=["1024x1024"],
                description="Fast, high-quality image generation.",
            ),
            MediaModel(
                id="fal/fal-ai/kling-video/v1/standard/text-to-video", provider=self.id,
                name="fal-ai/kling-video/v1/standard/text-to-video",
                display_name="Kling Video", kind="video", max_duration_s=10.0,
                description="Text-to-video.",
            ),
        ]

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Key {self.key()}", "content-type": "application/json"}

    async def _run(self, model: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        client = http_client("fal", timeout=timeout)
        start = await client.post(
            f"https://queue.fal.run/{model}", headers=self._headers(), json=payload
        )
        raise_for_status(self.id, start)
        queued = start.json()
        status_url = queued.get("status_url") or ""
        response_url = queued.get("response_url") or ""
        if not status_url:
            return queued  # some endpoints answer inline

        async def check() -> Any:
            poll = await client.get(status_url, headers=self._headers())
            raise_for_status(self.id, poll)
            body = poll.json()
            state = body.get("status")
            if state == "COMPLETED":
                return True
            if state in ("FAILED", "CANCELLED"):
                raise ProviderError(self.id, f"fal job {state}", retryable=False)
            return None

        await _poll("fal job", check, timeout=timeout, provider=self.id)
        final = await client.get(response_url or status_url, headers=self._headers())
        raise_for_status(self.id, final)
        return final.json()

    async def _download(self, url: str) -> bytes:
        response = await http_client("fal", timeout=300).get(url)
        raise_for_status(self.id, response)
        return response.content

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        model = _model_name(req.model) or "fal-ai/flux/dev"
        width, height = req.dimensions()
        payload: dict[str, Any] = {
            "prompt": req.prompt, "num_images": min(req.n, 4),
            "image_size": {"width": width, "height": height}, **req.params,
        }
        if req.negative_prompt:
            payload["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            payload["seed"] = req.seed

        data = await self._run(model, payload, IMAGE_POLL_TIMEOUT)
        results = []
        for item in data.get("images") or []:
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                GenerationResult(
                    data=await self._download(url), remote_url=url,
                    mime_type=item.get("content_type", "image/png"),
                    width=item.get("width", width), height=item.get("height", height),
                    seed=int(data.get("seed") or 0),
                )
            )
        if not results:
            raise ProviderError(self.id, "fal returned no images")
        return results

    async def generate_video(self, req: VideoRequest) -> GenerationResult:
        model = _model_name(req.model) or "fal-ai/kling-video/v1/standard/text-to-video"
        payload: dict[str, Any] = {
            "prompt": req.prompt, "duration": str(int(req.duration_s)),
            "aspect_ratio": req.aspect_ratio, **req.params,
        }
        if req.init_image_b64:
            payload["image_url"] = f"data:image/png;base64,{req.init_image_b64}"

        data = await self._run(model, payload, VIDEO_POLL_TIMEOUT)
        video = data.get("video") or {}
        url = video.get("url") if isinstance(video, dict) else ""
        if not url:
            raise ProviderError(self.id, "fal returned no video")
        return GenerationResult(
            data=await self._download(url), remote_url=url,
            mime_type="video/mp4", duration_s=req.duration_s,
        )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class Stability(MediaProvider):
    id = "stability"
    label = "Stability AI"
    kinds = ("image", "video")
    env_hint = "STABILITY_API_KEY"
    base = "https://api.stability.ai/v2beta"

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="stability/core", provider=self.id, name="core",
                display_name="Stable Image Core", kind="image",
                sizes=["1024x1024"], supports_negative_prompt=True,
                params_schema={"aspect_ratio": ["1:1", "16:9", "9:16", "3:2", "2:3"]},
            ),
            MediaModel(
                id="stability/ultra", provider=self.id, name="ultra",
                display_name="Stable Image Ultra", kind="image",
                sizes=["1024x1024"], supports_negative_prompt=True,
            ),
            MediaModel(
                id="stability/image-to-video", provider=self.id, name="image-to-video",
                display_name="Stable Video Diffusion", kind="video",
                max_duration_s=4.0, supports_image_input=True,
            ),
        ]

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        name = _model_name(req.model) or "core"
        client = http_client("stability", timeout=IMAGE_POLL_TIMEOUT)
        form: dict[str, Any] = {
            "prompt": (None, req.prompt),
            "output_format": (None, "png"),
            "aspect_ratio": (None, req.params.get("aspect_ratio", "1:1")),
        }
        if req.negative_prompt:
            form["negative_prompt"] = (None, req.negative_prompt)
        if req.seed is not None:
            form["seed"] = (None, str(req.seed))

        response = await client.post(
            f"{self.base}/stable-image/generate/{name}",
            headers={"authorization": f"Bearer {self.key()}", "accept": "image/*"},
            files=form,
        )
        raise_for_status(self.id, response)
        width, height = req.dimensions()
        return [
            GenerationResult(
                data=response.content, mime_type="image/png", width=width, height=height,
                seed=int(response.headers.get("seed") or 0),
            )
        ]

    async def generate_video(self, req: VideoRequest) -> GenerationResult:
        if not req.init_image_b64:
            raise ProviderError(
                self.id, "Stable Video Diffusion needs a source image", retryable=False
            )
        client = http_client("stability", timeout=VIDEO_POLL_TIMEOUT)
        start = await client.post(
            f"{self.base}/image-to-video",
            headers={"authorization": f"Bearer {self.key()}"},
            files={"image": ("init.png", base64.b64decode(req.init_image_b64), "image/png")},
            data={"seed": str(req.seed or 0), "cfg_scale": "1.8", "motion_bucket_id": "127"},
        )
        raise_for_status(self.id, start)
        job_id = start.json().get("id", "")
        if not job_id:
            raise ProviderError(self.id, "Stability did not return a job id")

        async def check() -> Any:
            poll = await client.get(
                f"{self.base}/image-to-video/result/{job_id}",
                headers={"authorization": f"Bearer {self.key()}", "accept": "video/*"},
            )
            if poll.status_code == 202:
                return None
            raise_for_status(self.id, poll)
            return poll.content

        content = await _poll(
            "Stability video", check, timeout=VIDEO_POLL_TIMEOUT, interval=8.0, provider=self.id
        )
        return GenerationResult(
            data=content, mime_type="video/mp4", duration_s=req.duration_s, job_id=job_id
        )


# ---------------------------------------------------------------------------
# Local: ComfyUI and Automatic1111
# ---------------------------------------------------------------------------

#: Minimal txt2img graph. Override wholesale via params["workflow"] to run your own.
COMFY_DEFAULT_WORKFLOW: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0, "steps": 24, "cfg": 7.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "CIWS", "images": ["8", 0]}},
}


class ComfyUI(MediaProvider):
    id = "comfyui"
    label = "ComfyUI (local)"
    kinds = ("image",)
    requires_key = False
    local = True

    @property
    def url(self) -> str:
        return get_settings().media.comfyui_url.rstrip("/")

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="comfyui/txt2img", provider=self.id, name="txt2img",
                display_name="ComfyUI Workflow", kind="image", local=True,
                sizes=["512x512", "768x768", "1024x1024", "1024x1536"],
                supports_negative_prompt=True,
                description="Runs against your local ComfyUI. Supply your own workflow JSON "
                            "in params.workflow to use any graph you have built.",
                params_schema={"steps": 24, "cfg": 7.0, "ckpt_name": "string", "workflow": "object"},
            )
        ]

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        import copy

        client = http_client("comfyui", timeout=IMAGE_POLL_TIMEOUT)
        workflow = copy.deepcopy(req.params.get("workflow") or COMFY_DEFAULT_WORKFLOW)
        width, height = req.dimensions()

        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = req.prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = req.negative_prompt
        if "5" in workflow:
            workflow["5"]["inputs"].update(
                {"width": width, "height": height, "batch_size": min(req.n, 4)}
            )
        if "3" in workflow:
            workflow["3"]["inputs"]["seed"] = req.seed or int(time.time())
            for key in ("steps", "cfg", "sampler_name", "scheduler"):
                if key in req.params:
                    workflow["3"]["inputs"][key] = req.params[key]
        if "4" in workflow and req.params.get("ckpt_name"):
            workflow["4"]["inputs"]["ckpt_name"] = req.params["ckpt_name"]

        client_id = f"ciws-{int(time.time())}"
        start = await client.post(
            f"{self.url}/prompt", json={"prompt": workflow, "client_id": client_id}
        )
        raise_for_status(self.id, start)
        prompt_id = start.json().get("prompt_id", "")
        if not prompt_id:
            raise ProviderError(self.id, "ComfyUI did not return a prompt id")

        async def check() -> Any:
            poll = await client.get(f"{self.url}/history/{prompt_id}")
            if poll.status_code != 200:
                return None
            history = poll.json().get(prompt_id)
            if not history:
                return None
            status = (history.get("status") or {}).get("status_str", "")
            if status == "error":
                raise ProviderError(self.id, "ComfyUI workflow errored", retryable=False)
            return history if history.get("outputs") else None

        history = await _poll(
            "ComfyUI render", check, timeout=IMAGE_POLL_TIMEOUT, interval=1.5, provider=self.id
        )

        results: list[GenerationResult] = []
        for node in (history.get("outputs") or {}).values():
            for image in node.get("images") or []:
                fetch = await client.get(
                    f"{self.url}/view",
                    params={
                        "filename": image.get("filename", ""),
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    },
                )
                raise_for_status(self.id, fetch)
                results.append(
                    GenerationResult(
                        data=fetch.content, mime_type="image/png",
                        width=width, height=height, job_id=prompt_id, cost_usd=0.0,
                    )
                )
        if not results:
            raise ProviderError(self.id, "ComfyUI produced no images")
        return results


class Automatic1111(MediaProvider):
    id = "a1111"
    label = "Automatic1111 (local)"
    kinds = ("image",)
    requires_key = False
    local = True

    @property
    def url(self) -> str:
        return get_settings().media.a1111_url.rstrip("/")

    async def models(self) -> list[MediaModel]:
        return [
            MediaModel(
                id="a1111/txt2img", provider=self.id, name="txt2img",
                display_name="A1111 txt2img", kind="image", local=True,
                sizes=["512x512", "768x768", "1024x1024"],
                supports_negative_prompt=True, supports_image_input=True,
                params_schema={"steps": 25, "cfg_scale": 7.0, "sampler_name": "Euler a"},
            )
        ]

    async def generate_image(self, req: ImageRequest) -> list[GenerationResult]:
        client = http_client("a1111", timeout=IMAGE_POLL_TIMEOUT)
        width, height = req.dimensions()
        payload: dict[str, Any] = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": width,
            "height": height,
            "batch_size": min(req.n, 4),
            "steps": req.params.get("steps", 25),
            "cfg_scale": req.params.get("cfg_scale", 7.0),
            "sampler_name": req.params.get("sampler_name", "Euler a"),
            "seed": req.seed if req.seed is not None else -1,
        }
        endpoint = "txt2img"
        if req.init_image_b64:
            endpoint = "img2img"
            payload["init_images"] = [req.init_image_b64]
            payload["denoising_strength"] = req.strength

        response = await client.post(f"{self.url}/sdapi/v1/{endpoint}", json=payload)
        raise_for_status(self.id, response)
        body = response.json()
        info = json.loads(body.get("info", "{}")) if isinstance(body.get("info"), str) else {}
        return [
            GenerationResult(
                data=base64.b64decode(img), mime_type="image/png",
                width=width, height=height, cost_usd=0.0,
                seed=int((info.get("all_seeds") or [0])[0]) if info else 0,
            )
            for img in body.get("images") or []
        ] or [GenerationResult()]


PROVIDERS: dict[str, MediaProvider] = {
    p.id: p
    for p in (OpenAIImages(), GoogleMedia(), Replicate(), Fal(), Stability(), ComfyUI(), Automatic1111())
}


def for_kind(kind: str) -> list[MediaProvider]:
    return [p for p in PROVIDERS.values() if kind in p.kinds]


def get_provider(provider_id: str) -> MediaProvider:
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise ProviderError(provider_id, f"Unknown media provider '{provider_id}'", retryable=False)
    return provider
