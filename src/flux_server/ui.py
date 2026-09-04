import base64
import json
import os
import shlex
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field, replace
from io import BytesIO
from typing import Any

import gradio as gr
import openai
import PIL.Image
import PIL.ImageChops
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.components.gallery import GalleryData, GalleryImage
from gradio.data_classes import ImageData

from flux_server.config import MAX_STEPS, Settings

CUSTOM_SIZE = "custom"
SIZE_CHOICES = [
    "1024x1024",
    "1024x768",
    "768x1024",
    "1280x720",
    "720x1280",
    "768x768",
    "512x512",
    "auto",
    CUSTOM_SIZE,
]
QUALITY_CHOICES = ["auto", "low", "medium", "high", "standard", "hd"]
FORMAT_CHOICES = ["png", "jpeg", "webp"]
MAX_SEED = 2**32 - 1
EDIT_PLACEHOLDER = "ref.png"
PROMPT_HINT = "Shift+Enter to run"
EXAMPLE_PROMPTS = [
    "A cat holding a sign that says hello world",
    "A cozy cabin in a snowy forest at dusk, warm light in the windows",
    "Studio photo of a red sports car, dramatic lighting, 85mm lens",
    "Watercolor illustration of a lighthouse on a cliff, stormy sea",
]
MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
EXTENSIONS = {mime: ext for ext, mime in MIME_TYPES.items()}


def data_url(b64: str, output_format: str) -> str:
    return f"data:{MIME_TYPES.get(output_format, 'image/png')};base64,{b64}"


def decode_data_url(url: str) -> PIL.Image.Image:
    if not url.startswith("data:"):
        raise ValueError("not a data URL")
    img = PIL.Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
    img.load()
    return img


def _data_url_mime(url: str) -> str:
    return url[len("data:") :].split(";", 1)[0].split(",", 1)[0] or "image/png"


def gallery_value(urls: list[str], captions: list[str]) -> list[dict[str, Any]]:
    """Pre-serialized Gallery value holding inline data URLs (nothing cached on the server).

    Emitted from an event with `postprocess=False`; Gradio's file cache skips `path=None` entries.
    """
    root = []
    for i, (url, caption) in enumerate(zip(urls, captions, strict=True)):
        mime = _data_url_mime(url)
        ext = EXTENSIONS.get(mime, "png")
        seed = caption.removeprefix("seed=") if caption.startswith("seed=") else ""
        name = f"seed-{seed}.{ext}" if seed.isdigit() else f"image-{i + 1}.{ext}"
        image = ImageData(url=url, orig_name=name, mime_type=mime)
        root.append(GalleryImage(image=image, caption=caption))
    return GalleryData(root=root).model_dump()


@dataclass(frozen=True)
class GenParams:
    prompt: str
    size: str | None
    quality: str | None
    steps: int | None
    seed: int | None
    n: int
    output_format: str
    output_compression: int = 100
    model: str | None = None
    strength: float | None = None  # edits only


@dataclass(frozen=True)
class UIResult:
    images: list[tuple[PIL.Image.Image, str]]
    meta: dict[str, Any]
    curl: str
    urls: list[str] = field(default_factory=list)  # data URLs parallel to images (API bytes)


def make_client(api_base: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=api_base, api_key="local", max_retries=0, timeout=3600)


def normalize_api_base(value: str | None, default: str) -> str:
    return (value or "").strip().rstrip("/") or default


def resolve_size(size: str | None, width: float | None, height: float | None) -> str | None:
    size = (size or "").strip()
    if size == CUSTOM_SIZE:
        return f"{int(width)}x{int(height)}" if width and height else None
    return size or None


def build_generate_kwargs(model_id: str, p: GenParams) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": p.model or model_id,
        "prompt": p.prompt,
        "n": p.n,
        "output_format": p.output_format,
    }
    if p.output_format != "png":
        kwargs["output_compression"] = p.output_compression
    if p.size and p.size != "auto":
        kwargs["size"] = p.size
    if p.quality and p.quality != "auto":
        kwargs["quality"] = p.quality
    extra: dict[str, Any] = {}
    if p.seed is not None and p.seed >= 0:
        extra["seed"] = p.seed
    if p.steps and p.steps > 0:
        extra["steps"] = p.steps
    if p.strength is not None and 0 < p.strength <= 1:
        extra["strength"] = p.strength
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def _flatten(kwargs: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in kwargs.items() if k != "extra_body"}
    flat.update(kwargs.get("extra_body") or {})
    return flat


def curl_for_generate(api_base: str, kwargs: dict[str, Any]) -> str:
    body = json.dumps(_flatten(kwargs), ensure_ascii=False)
    return (
        f"curl -s -X POST {api_base}/images/generations "
        f"-H 'content-type: application/json' -d {shlex.quote(body)}"
    )


def curl_for_edit(
    api_base: str, kwargs: dict[str, Any], filenames: list[str], mask: str | None = None
) -> str:
    parts = [f"curl -s -X POST {api_base}/images/edits"]
    parts += [f"-F {shlex.quote(f'image[]=@{name}')}" for name in filenames]
    if mask:
        parts.append(f"-F {shlex.quote(f'mask=@{mask}')}")
    parts += [f"-F {shlex.quote(f'{k}={v}')}" for k, v in _flatten(kwargs).items()]
    return " ".join(parts)


def request_preview(
    api_base: str,
    model_id: str,
    p: GenParams,
    filenames: list[str] | None = None,
    mask: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Request metadata + curl for the current form state; needs no server round-trip."""
    kwargs = build_generate_kwargs(model_id, p)
    if filenames is None:
        meta = {"endpoint": f"{api_base}/images/generations", "request": _flatten(kwargs)}
        return meta, curl_for_generate(api_base, kwargs)
    names = filenames or [EDIT_PLACEHOLDER]
    request = {"image[]": names, **({"mask": mask} if mask else {}), **_flatten(kwargs)}
    meta = {"endpoint": f"{api_base}/images/edits", "request": request}
    return meta, curl_for_edit(api_base, kwargs, names, mask)


def _seed_of(item: Any) -> int | None:
    seed = getattr(item, "seed", None)
    if seed is None:
        seed = (getattr(item, "model_extra", None) or {}).get("seed")
    return seed


def decode_images(data: list[Any]) -> list[tuple[PIL.Image.Image, str]]:
    out: list[tuple[PIL.Image.Image, str]] = []
    for item in data:
        seed = _seed_of(item)
        img = PIL.Image.open(BytesIO(base64.b64decode(item.b64_json)))
        img.load()
        out.append((img, f"seed={seed if seed is not None else '?'}"))
    return out


def _header_seconds(headers: Any) -> float | None:
    value = headers.get("x-generation-seconds")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _collect(raw: Any, p: GenParams, elapsed: float, curl: str, refs: int) -> UIResult:
    resp = raw.parse()
    images = decode_images(resp.data)
    output_format = resp.output_format or p.output_format
    urls = [data_url(item.b64_json, output_format) for item in resp.data]
    meta = {
        "seeds": [_seed_of(item) for item in resp.data],
        "size": resp.size,
        "steps_requested": p.steps,
        "quality": resp.quality,
        "elapsed_server_seconds": _header_seconds(raw.headers),
        "elapsed_client_seconds": round(elapsed, 3),
        "created": resp.created,
        "output_format": resp.output_format,
        "n": p.n,
        "done": len(images),
        "refs": refs,
        "request_id": raw.headers.get("x-request-id"),
    }
    return UIResult(images=images, meta=meta, curl=curl, urls=urls)


def _headers(request_id: str | None) -> dict[str, str] | None:
    return {"X-Request-Id": request_id} if request_id else None


def run_generate(
    client: openai.OpenAI,
    model_id: str,
    api_base: str,
    p: GenParams,
    request_id: str | None = None,
) -> UIResult:
    kwargs = build_generate_kwargs(model_id, p)
    t0 = time.perf_counter()
    raw = client.images.with_raw_response.generate(**kwargs, extra_headers=_headers(request_id))
    elapsed = time.perf_counter() - t0
    return _collect(raw, p, elapsed, curl_for_generate(api_base, kwargs), refs=0)


def run_edit(
    client: openai.OpenAI,
    model_id: str,
    api_base: str,
    file_paths: list[str],
    p: GenParams,
    request_id: str | None = None,
    mask_path: str | None = None,
) -> UIResult:
    kwargs = build_generate_kwargs(model_id, p)
    with ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in file_paths]
        mask = stack.enter_context(open(mask_path, "rb")) if mask_path else openai.NOT_GIVEN
        t0 = time.perf_counter()
        raw = client.images.with_raw_response.edit(
            image=files, mask=mask, **kwargs, extra_headers=_headers(request_id)
        )
        elapsed = time.perf_counter() - t0
    curl = curl_for_edit(api_base, kwargs, [str(path) for path in file_paths], mask_path)
    return _collect(raw, p, elapsed, curl, refs=len(file_paths))


def _never_stop() -> bool:
    return False


def _merge_meta(parts: list[dict[str, Any]], p: GenParams, started: float) -> dict[str, Any]:
    seeds = [seed for m in parts for seed in m["seeds"]]
    server = [m["elapsed_server_seconds"] for m in parts]
    return {
        **parts[-1],
        "seeds": seeds,
        "elapsed_server_seconds": round(sum(server), 3) if None not in server else None,
        "elapsed_client_seconds": round(time.perf_counter() - started, 3),
        "n": p.n,
        "done": len(seeds),
    }


def _iter_split(
    p: GenParams,
    run_one: Callable[[GenParams], UIResult],
    curl: str,
    should_stop: Callable[[], bool],
) -> Iterator[UIResult]:
    """One n=1 request per image so each shows up as soon as it is done.

    Seeds chain the way the server does internally (s, s+1, ...), so the images are
    identical to those of a single request with the full n.
    """
    if p.n <= 1:
        yield run_one(p)
        return
    images: list[tuple[PIL.Image.Image, str]] = []
    urls: list[str] = []
    parts: list[dict[str, Any]] = []
    seed = p.seed
    started = time.perf_counter()
    for i in range(p.n):
        if i and should_stop():
            cancelled = {**_merge_meta(parts, p, started), "cancelled": True}
            yield UIResult(images, cancelled, curl, urls=urls)
            return
        result = run_one(replace(p, n=1, seed=seed))
        images = [*images, *result.images]
        urls = [*urls, *result.urls]
        parts.append(result.meta)
        used = next(iter(result.meta["seeds"]), None)  # server-chosen when seed was None
        seed = (used + 1) % 2**32 if used is not None else None
        meta = _merge_meta(parts, p, started)
        if i + 1 < p.n and should_stop():  # cancelled while this image was rendering
            yield UIResult(images, {**meta, "cancelled": True}, curl, urls=urls)
            return
        yield UIResult(images, meta, curl, urls=urls)


def iter_generate(
    client: openai.OpenAI,
    model_id: str,
    api_base: str,
    p: GenParams,
    request_id: str | None = None,
    should_stop: Callable[[], bool] = _never_stop,
) -> Iterator[UIResult]:
    def run_one(q: GenParams) -> UIResult:
        return run_generate(client, model_id, api_base, q, request_id)

    curl = curl_for_generate(api_base, build_generate_kwargs(model_id, p))
    yield from _iter_split(p, run_one, curl, should_stop)


def iter_edit(
    client: openai.OpenAI,
    model_id: str,
    api_base: str,
    file_paths: list[str],
    p: GenParams,
    request_id: str | None = None,
    mask_path: str | None = None,
    should_stop: Callable[[], bool] = _never_stop,
) -> Iterator[UIResult]:
    def run_one(q: GenParams) -> UIResult:
        return run_edit(client, model_id, api_base, file_paths, q, request_id, mask_path)

    kwargs = build_generate_kwargs(model_id, p)
    curl = curl_for_edit(api_base, kwargs, [str(path) for path in file_paths], mask_path)
    yield from _iter_split(p, run_one, curl, should_stop)


def cancel_request(client: openai.OpenAI, request_id: str) -> bool:
    """POST /v1/images/{id}/cancel; False when nothing with that id is in flight."""
    try:
        client.post(f"/images/{request_id}/cancel", cast_to=dict[str, Any])
    except openai.NotFoundError:
        return False
    return True


def error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        source = inner if isinstance(inner, dict) else body
        return source.get("code")
    return None


def format_error(exc: Exception) -> str:
    if isinstance(exc, openai.APIStatusError):
        message = str(exc)
        body = exc.body
        if isinstance(body, dict):
            inner = body.get("error")
            source = inner if isinstance(inner, dict) else body
            message = source.get("message") or message
        return f"HTTP {exc.status_code}: {message}"
    if isinstance(exc, openai.APIConnectionError):
        return f"Cannot reach API: {exc}"
    return f"{type(exc).__name__}: {exc}"


def progress_text(meta: dict[str, Any]) -> str:
    """One-line run status shown under the buttons (persistent, unlike toasts)."""
    n, done = meta.get("n") or 0, meta.get("done") or 0
    if meta.get("cancelled"):
        return f"Cancelled · {done}/{n} done"
    if done < n:
        return f"Generating · {done}/{n} done"
    server, client = meta.get("elapsed_server_seconds"), meta.get("elapsed_client_seconds")
    timing = ""
    if server is not None and client is not None:
        timing = f" · server {server:.1f}s / total {client:.1f}s"
    return f"Done · {done} {'image' if done == 1 else 'images'}{timing}"


def _item_path(item: Any) -> str | None:
    if isinstance(item, (tuple, list)):
        item = item[0] if item else None
    if isinstance(item, dict):
        item = item.get("image", item.get("path"))
        if isinstance(item, dict):
            item = item.get("path")
    return str(item) if item else None


def gallery_paths(value: Any) -> list[str]:
    """File paths of a Gallery value (list of (path, caption) tuples; None -> [])."""
    return [path for path in (_item_path(item) for item in value or []) if path]


def _item_url(item: Any) -> str | None:
    if isinstance(item, (tuple, list)):
        item = item[0] if item else None
    if isinstance(item, dict):
        item = item.get("image", item)
        if isinstance(item, dict):
            item = item.get("url")
    return item if isinstance(item, str) and item.startswith("data:") else None


def select_gallery_image(items: Any, index: Any) -> list[PIL.Image.Image]:
    """[image] decoded from the selected result-gallery item (index clamped); [] when empty.

    `items` is the raw frontend Gallery value (`preprocess=False`): a list of
    `{"image": {"url": ...}, "caption": ...}` dicts; `(url, caption)` tuples and plain data URL
    strings are accepted for direct calls.
    """
    items = list(items or [])
    if not items:
        return []
    try:
        i = int(index or 0)
    except (TypeError, ValueError):
        i = 0
    url = _item_url(items[min(max(i, 0), len(items) - 1)])
    if url is None:
        raise gr.Error("Result unavailable, please generate again")
    return [decode_data_url(url)]


def mask_from_editor(value: dict[str, Any] | None) -> PIL.Image.Image | None:
    """'L' mask (255 = painted on any layer) sized like the background; None if unpainted."""
    layers = [layer for layer in (value or {}).get("layers") or [] if layer is not None]
    if not layers or value is None:
        return None
    background = value.get("background")
    size = background.size if background is not None else layers[0].size
    mask = PIL.Image.new("L", size, 0)
    for layer in layers:
        alpha = layer.convert("RGBA").getchannel("A").point(lambda a: 255 if a > 0 else 0)
        if alpha.size != size:
            alpha = alpha.resize(size, PIL.Image.Resampling.NEAREST)
        mask = PIL.ImageChops.lighter(mask, alpha)
    return mask if mask.getbbox() is not None else None


def write_temp_mask(mask: PIL.Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        mask.convert("L").save(f, "PNG")
        return f.name


def status_line(health: dict[str, Any]) -> str:
    model = health.get("model_id") or health.get("model") or "?"
    parts: list[str | None] = [
        f"**{model}**",
        health.get("device"),
        health.get("dtype"),
        f"offload {health['offload']}" if health.get("offload") is not None else None,
        f"default {health['default_steps']} steps"
        if health.get("default_steps") is not None
        else None,
        f"in flight {health.get('in_flight', 0)}",
    ]
    return " · ".join(str(part) for part in parts if part)


def _params(
    prompt: str | None,
    size: str | None,
    width: float | None,
    height: float | None,
    quality: str | None,
    steps: float | None,
    seed: float | None,
    n: float | None,
    output_format: str | None,
    output_compression: float | None,
    model: str | None = None,
    strength: float | None = None,
) -> GenParams:
    return GenParams(
        prompt=prompt or "",
        size=resolve_size(size, width, height),
        quality=(quality or "").strip() or None,
        steps=int(steps) if steps else None,
        seed=int(seed) if seed is not None and seed >= 0 else None,
        n=int(n or 1),
        output_format=output_format or "png",
        output_compression=int(output_compression) if output_compression is not None else 100,
        model=(model or "").strip() or None,
        strength=round(float(strength), 4) if strength else None,
    )


def _controls(settings: Settings, size_default: str) -> list[Any]:
    quality_map = " ".join(f"{k}={v}" for k, v in settings.quality_steps.items())
    with gr.Row():
        size = gr.Dropdown(
            SIZE_CHOICES,
            value=size_default,
            label="size",
            allow_custom_value=True,
            info="Type WxH directly (multiples of 16); auto = default / reference size; "
            "custom = width/height below",
        )
        quality = gr.Dropdown(
            QUALITY_CHOICES,
            value="auto",
            label="quality",
            allow_custom_value=True,
            info=f"Determines the step count when steps=0: {quality_map}",
        )
        output_format = gr.Radio(FORMAT_CHOICES, value="png", label="output_format")
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            steps = gr.Slider(
                0,
                MAX_STEPS,
                value=settings.default_steps,
                step=1,
                label="steps",
                info="0 = determined by quality",
            )
            seed = gr.Number(
                value=-1,
                precision=0,
                minimum=-1,
                maximum=MAX_SEED,
                label="seed",
                info="-1 = random; with n>1 the seed increments by 1 per image",
            )
            n = gr.Slider(
                1,
                settings.max_n,
                value=1,
                step=1,
                label="n",
                info="Images are returned one by one and shown as they finish",
            )
        with gr.Accordion("Custom size (used when size=custom)", open=False), gr.Row():
            width = gr.Slider(256, 2048, value=1024, step=16, label="width")
            height = gr.Slider(256, 2048, value=1024, step=16, label="height")
        output_compression = gr.Slider(
            0, 100, value=100, step=1, label="output_compression", info="jpeg / webp only"
        )
    return [size, width, height, quality, steps, seed, n, output_format, output_compression]


def _result_components(
    action_label: str,
) -> tuple[gr.JSON, gr.JSON, gr.Code, gr.Gallery, gr.Button]:
    gallery = gr.Gallery(
        label="Results",
        columns=2,
        height=640,
        object_fit="contain",
        preview=True,
        interactive=False,  # holds inline data URLs; also read as an input (send to edit)
    )
    with gr.Row():
        action = gr.Button(action_label, variant="secondary")
    with gr.Accordion("Request / response details", open=False):
        with gr.Row():
            req_meta = gr.JSON(label="Request (live from form)")
            resp_meta = gr.JSON(label="Response metadata (updates with progress)")
        curl = gr.Code(language="shell", label="Equivalent curl", interactive=False)
    return req_meta, resp_meta, curl, gallery, action


def _bind(demo, buttons, prompt, inputs, preview_fn, run_fn, cancel_fn, outputs, api_name) -> None:
    button, cancel_button, request_id, status = buttons
    req_meta, resp_meta, curl, gallery = outputs
    live: dict[str, Any] = {
        "inputs": inputs,
        "outputs": [req_meta, curl],
        "show_progress": "hidden",
        "api_name": False,
    }
    gr.on([c.change for c in inputs], preview_fn, trigger_mode="always_last", **live)
    demo.load(preview_fn, **live)
    # A fresh id is stored before the run starts so the cancel button can target it.
    gr.on([button.click, prompt.submit], preview_fn, **live).then(
        lambda: uuid.uuid4().hex, outputs=request_id, show_progress="hidden", api_name=False
    ).then(
        run_fn,
        inputs=[request_id, *inputs],
        outputs=[gallery, resp_meta, status],
        api_name=api_name,
        show_progress="minimal",  # keep finished images visible while the rest stream in
        postprocess=False,  # gallery value is pre-serialized data URLs; Gradio must not cache files
    )
    cancel_button.click(
        cancel_fn,
        inputs=[inputs[0], request_id],
        outputs=status,
        show_progress="hidden",
        api_name=False,
    )


def _on_select(evt: gr.SelectData) -> int:
    return evt.index


def _run_buttons(label: str) -> tuple[gr.Button, gr.Button, gr.State, gr.Markdown]:
    with gr.Row():
        run = gr.Button(label, variant="primary", scale=3)
        cancel = gr.Button("Cancel", variant="stop", scale=1)
    status = gr.Markdown("")
    return run, cancel, gr.State(""), status


def build_ui(settings: Settings) -> gr.Blocks:
    default_base = settings.api_base
    default_model = settings.model_id
    runs: dict[str, bool] = {}  # in-progress runs by request id -> stop before the next image?

    def on_preview(api_base, model, prompt, *controls):
        p = _params(prompt, *controls, model=model)
        return request_preview(normalize_api_base(api_base, default_base), default_model, p)

    def on_edit_preview(api_base, model, refs, editor, mask, strength, prompt, *controls):
        p = _params(prompt, *controls, model=model, strength=strength)
        names = gallery_paths(refs)
        base = normalize_api_base(api_base, default_base)
        painted = mask_from_editor(editor) is not None
        mask_name = "<painted>" if painted else (str(mask) if mask else None)
        return request_preview(base, default_model, p, names, mask_name)

    def on_refs_change(refs, previous):
        first = next(iter(gallery_paths(refs)), None)
        if first == previous:
            return gr.skip(), previous  # adding more refs keeps the painting
        return first, first

    def stream(results: Iterator[UIResult], request_id: str, n: int):
        """Push every partial UIResult to the gallery; a cancel ends the run gracefully."""
        runs[request_id] = False
        items: list[dict[str, Any]] = []
        meta: dict[str, Any] = {"request_id": request_id, "n": n, "done": 0}
        yield gr.skip(), gr.skip(), progress_text(meta)
        try:
            for result in results:
                items = gallery_value(result.urls, [cap for _, cap in result.images])
                meta = result.meta
                yield items, meta, progress_text(meta)
        except openai.APIStatusError as exc:
            if error_code(exc) != "cancelled":
                yield gr.skip(), gr.skip(), f"Error: {format_error(exc)}"
                raise gr.Error(format_error(exc)) from exc
            meta = {**meta, "cancelled": True}
            yield items, meta, progress_text(meta)
        except Exception as exc:
            yield gr.skip(), gr.skip(), f"Error: {format_error(exc)}"
            raise gr.Error(format_error(exc)) from exc
        finally:
            runs.pop(request_id, None)

    def on_generate(request_id, api_base, model, prompt, *controls):
        base = normalize_api_base(api_base, default_base)
        p = _params(prompt, *controls, model=model)
        request_id = request_id or uuid.uuid4().hex
        results = iter_generate(
            make_client(base),
            default_model,
            base,
            p,
            request_id,
            should_stop=lambda: runs.get(request_id, False),
        )
        yield from stream(results, request_id, p.n)

    def on_edit(request_id, api_base, model, refs, editor, mask, strength, prompt, *controls):
        paths = gallery_paths(refs)
        if not paths:
            raise gr.Error("Upload at least one reference image")
        base = normalize_api_base(api_base, default_base)
        p = _params(prompt, *controls, model=model, strength=strength)
        request_id = request_id or uuid.uuid4().hex
        client = make_client(base)
        painted = mask_from_editor(editor)
        temp_mask = write_temp_mask(painted) if painted is not None else None
        try:
            results = iter_edit(
                client,
                default_model,
                base,
                paths,
                p,
                request_id,
                mask_path=temp_mask or (str(mask) if mask else None),
                should_stop=lambda: runs.get(request_id, False),
            )
            yield from stream(results, request_id, p.n)
        finally:
            if temp_mask:
                with suppress(FileNotFoundError):
                    os.unlink(temp_mask)

    def on_cancel(api_base, request_id):
        if not request_id:
            gr.Warning("No generation has been started")
            return gr.skip()
        if request_id in runs:
            runs[request_id] = True  # covers the gap between two n=1 requests
        base = normalize_api_base(api_base, default_base)
        try:
            cancelled = cancel_request(make_client(base), request_id)
        except Exception as exc:
            raise gr.Error(format_error(exc)) from exc
        if cancelled:
            return "Cancel requested; stops after the current denoising step…"
        if request_id in runs:
            return "Cancel requested; stops after the current image…"
        gr.Warning("No generation in progress (it may have finished)")
        return gr.skip()

    def on_refresh(api_base):
        base = normalize_api_base(api_base, default_base)
        client = make_client(base)
        try:
            root = client.with_options(base_url=base.removesuffix("/v1"))
            return {
                "health": root.get("/health", cast_to=dict[str, Any]),
                "models": client.models.list().model_dump(),
            }
        except Exception as exc:
            return {"error": format_error(exc)}

    def on_status(api_base):
        base = normalize_api_base(api_base, default_base)
        root = make_client(base).with_options(base_url=base.removesuffix("/v1"), timeout=10)
        try:
            return status_line(root.get("/health", cast_to=dict[str, Any]))
        except openai.APIStatusError as exc:
            if exc.status_code == 503:
                return "Loading model…"
            return f"Warning: {format_error(exc)}"
        except Exception as exc:
            return f"Warning: {format_error(exc)}"

    def on_send(items, index):
        return select_gallery_image(items, index), gr.Tabs(selected="edit")

    with gr.Blocks(
        title="FLUX.2 [klein] 4B Playground", analytics_enabled=False, fill_width=True
    ) as demo:
        with gr.Row():
            gr.Markdown("# FLUX.2 [klein] 4B Playground", scale=1)
            status_md = gr.Markdown("", scale=1)
        with gr.Accordion("Connection (API base / model)", open=False), gr.Row():
            api_base = gr.Textbox(
                value=default_base,
                label="API base",
                info="Used by every request and curl example; "
                "point it at a remote server if needed",
                scale=2,
            )
            model = gr.Textbox(
                value=default_model,
                label="model",
                info="model field sent in requests; not validated by the server "
                "(try gpt-image-1 to test compatibility)",
                scale=1,
            )
        demo.load(on_status, inputs=api_base, outputs=status_md, api_name=False)
        api_base.change(on_status, inputs=api_base, outputs=status_md, api_name=False)

        with gr.Tabs() as tabs:
            with gr.Tab("Text to Image", id="generate"), gr.Row():
                with gr.Column(scale=1, min_width=380):
                    prompt = gr.Textbox(
                        value=EXAMPLE_PROMPTS[0], lines=3, label="prompt", info=PROMPT_HINT
                    )
                    buttons = _run_buttons("Generate")
                    controls = _controls(settings, "1024x1024")
                    gr.Examples(
                        [[p] for p in EXAMPLE_PROMPTS], inputs=[prompt], label="Example prompts"
                    )
                with gr.Column(scale=1, min_width=480):
                    *outputs, gallery, send = _result_components("Send to Image to Image")
                    sel = gr.State(0)
                inputs = [api_base, model, prompt, *controls]
                _bind(
                    demo,
                    buttons,
                    prompt,
                    inputs,
                    on_preview,
                    on_generate,
                    on_cancel,
                    [*outputs, gallery],
                    "generate",
                )
                gallery.select(_on_select, None, sel, api_name=False)

            with gr.Tab("Image to Image", id="edit"), gr.Row():
                with gr.Column(scale=1, min_width=380):
                    refs = gr.Gallery(
                        label="Reference images (multiple allowed; order = image[] order)",
                        interactive=True,
                        type="filepath",
                        file_types=["image"],
                        columns=4,
                        height=200,
                        object_fit="contain",
                        format="png",  # results are fed back as PIL images; keep them lossless
                    )
                    first_ref_state = gr.State(None)
                    with gr.Accordion(
                        "Inpainting: paint the area to repaint on the first image (optional)",
                        open=False,
                    ):
                        editor = gr.ImageEditor(
                            label="Mask canvas",
                            type="pil",
                            image_mode="RGBA",
                            sources=(),  # background is set from the refs gallery only
                            transforms=(),
                            layers=gr.LayerOptions(allow_additional_layers=False),
                            brush=gr.Brush(
                                colors=["rgba(255, 60, 60, 0.6)"],
                                default_size=40,
                                color_mode="fixed",
                            ),
                            eraser=gr.Eraser(default_size=40),
                            height=420,
                            placeholder="Upload a reference image first; "
                            "the canvas loads the first one automatically",
                        )
                        mask = gr.File(
                            file_count="single",
                            file_types=["image"],
                            label="Or upload a mask file (transparent / white = repaint)",
                        )
                    with gr.Row():
                        strength = gr.Slider(
                            0,
                            1,
                            value=0,
                            step=0.05,
                            label="strength",
                            info="0 = off (native multi-image editing); >0 = how far to move "
                            "away from the first image, granularity about 1/steps; "
                            "size must be auto in mask/strength mode",
                        )
                    gr.Markdown(
                        "Inpainting: the painted area takes precedence over an uploaded mask "
                        "file (file: transparent area = repaint; without an alpha channel, "
                        "white = repaint); the second image is the reference for the repainted "
                        "area; size must be auto."
                    )
                    edit_prompt = gr.Textbox(
                        value="make it a snowy winter scene",
                        lines=3,
                        label="prompt",
                        info=PROMPT_HINT,
                    )
                    edit_buttons = _run_buttons("Edit")
                    edit_controls = _controls(settings, "auto")
                with gr.Column(scale=1, min_width=480):
                    *edit_outputs, edit_gallery, reuse = _result_components(
                        "Continue editing with selected result"
                    )
                    edit_sel = gr.State(0)
                edit_inputs = [
                    api_base,
                    model,
                    refs,
                    editor,
                    mask,
                    strength,
                    edit_prompt,
                    *edit_controls,
                ]
                refs.change(
                    on_refs_change,
                    inputs=[refs, first_ref_state],
                    outputs=[editor, first_ref_state],
                    show_progress="hidden",
                    api_name=False,
                )
                _bind(
                    demo,
                    edit_buttons,
                    edit_prompt,
                    edit_inputs,
                    on_edit_preview,
                    on_edit,
                    on_cancel,
                    [*edit_outputs, edit_gallery],
                    "edit",
                )
                edit_gallery.select(_on_select, None, edit_sel, api_name=False)
                reuse.click(
                    select_gallery_image,
                    [edit_gallery, edit_sel],
                    refs,
                    api_name=False,
                    preprocess=False,  # raw gallery dicts; data URLs must not be fetched/cached
                )

            with gr.Tab("Status", id="status"):
                refresh = gr.Button("Refresh")
                status = gr.JSON(label="Status")
                refresh.click(on_refresh, inputs=api_base, outputs=status, api_name="status")

        send.click(on_send, [gallery, sel], [refs, tabs], api_name=False, preprocess=False)

    return demo


def mount_ui(app: FastAPI, settings: Settings) -> None:
    # Results are inline data URLs; they must not be mirrored into the browser's localStorage.
    gr.mount_gradio_app(app, build_ui(settings), path="/ui", run_history=False)

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse("/ui", status_code=302)
